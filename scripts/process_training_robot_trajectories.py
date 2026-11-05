"""Pre-process raw state-mode robot trajectories into per-sample pkls for fast
Pi0.5 training.

The robot trajectories collected with ``obs_mode="state"`` store cameras as
sibling mp4 files and a flat ``traj_0/obs`` tensor with no ``agent/qpos`` or
``sensor_data`` subgroup. This script reads each demo from
``raw_training_trajectories/`` and emits one pickle per (task, demo,
timestep) into ``postprocessed_robot_trajectories/data/<idx>.pkl`` plus a ``postprocessed_robot_trajectories/meta/``
directory holding ``stats.json`` and ``index.jsonl``. ``LfOVLADataset``
reads this format with one pickle load per sample — no h5 decompression at
training time.

There is no video-demo manifest: ``LfOVLADataset`` discovers the eligible
human-demo feature dirs by scanning ``video_demonstration_features/`` directly
at load time (the feature tree mirrors the raw human-demo tree and shares
task names), so this script only handles the robot side.

The per-sample pkls are **shared across all methods**; what differs is the
index file that decides which timesteps a method samples:

===========================  ===============================================
Method                       Index it reads
===========================  ===============================================
Vid. Cond. pi0.5             ``meta/index.jsonl``
VLM + pi0.5                  ``meta/index.jsonl``
UniSkill (robot skills)      ``meta/index.jsonl`` (or ``full_index.jsonl``
                             to align with the robomimic HDF5s)
Vid2Robot                    ``meta/vid2robot_index.jsonl``
===========================  ===============================================

``--model`` selects which of those to produce. ``vid2robot`` needs a second
pass over the trimmed tail (what ``--extend-only`` does), so selecting it runs
that pass automatically rather than leaving training to fail later on a
missing ``vid2robot_index.jsonl``.

CLI::

    python scripts/process_training_robot_trajectories.py \\
        --data-root ~/data/lfo_benchmark \\
        [--model all | vid_cond_pi05 | vlm_pi05 | uniskill | vid2robot] \\
        [--task pick_cube] [--max-demos N] [--overwrite] \\
        [--pi05-robot-cameras base_camera hand_camera] \\
        [--pi05-image-size 224] [--pi05-action-horizon 20]

The legacy ``h5_training_trajectories/`` output mode (rewriting raw state-mode
h5s into a sensor_data-mode h5 via ``_backport_one``) was removed in
favour of the per-sample pkl path. If you still need that artifact, check
out an earlier revision of this script.
"""

from __future__ import annotations

import argparse
import json
import logging
import os
import pickle
import re
import sys
from pathlib import Path

import cv2
import h5py
import imageio.v2 as imageio
import numpy as np

logger = logging.getLogger("process_training_robot_trajectories")

#: ManiSkill articulation state layout: ``[root_pose(7), root_lin_vel(3), root_ang_vel(3), qpos(N), qvel(N)]``.
#: For the 9-DoF panda (7 arm + 2 gripper) this is 7 + 3 + 3 + 9 + 9 = 31 columns, so ``qpos``
#: lives at offset 13. (Earlier this was set to 7, which sliced root velocity + the first 3 arm
#: joints into ``obs/agent/qpos`` — the zeroed-arm placeholder bug.)
_ARTICULATION_QPOS_OFFSET = 13
_ARTICULATION_QPOS_LEN = 9

#: Filenames that look like ``<name>.mp4`` and aren't the generic playback alias.
_CAMERA_MP4_RE = re.compile(r"^(?!0\.mp4$)(?P<cam>[A-Za-z0-9_]+)\.mp4$")


#: Iteration order over robot-trajectory buckets. Order matters: pkl indices
#: are assigned sequentially during ``_iter_traj_dirs`` so any new bucket must
#: be appended *after* the existing ones to keep existing
#: ``postprocessed_robot_trajectories/meta/*.jsonl`` indices valid.
_BUCKET_ORDER = ("without_distraction", "with_distraction")


def _iter_traj_dirs(training_root: Path, only_task: str | None) -> list[tuple[str, str, Path]]:
    """Walk ``raw_training_trajectories/<bucket>/.../<demo>/`` in deterministic
    order and return ``(bucket, task, demo_dir)`` tuples.

    The layout is ``<bucket>/<task>/<demo>/`` and ``task`` is the task-dir
    name.

    Buckets missing from disk are silently skipped, so adding a new bucket
    is purely append-only at the iteration tail.
    """
    out: list[tuple[str, str, Path]] = []
    for bucket in _BUCKET_ORDER:
        bucket_dir = training_root / bucket
        if not bucket_dir.is_dir():
            continue
        for task_dir in sorted(p for p in bucket_dir.iterdir() if p.is_dir()):
            if only_task is not None and task_dir.name != only_task:
                continue
            for demo_dir in sorted(p for p in task_dir.iterdir() if p.is_dir()):
                out.append((bucket, task_dir.name, demo_dir))
    return out


def _camera_mp4_files(traj_dir: Path) -> list[tuple[str, Path]]:
    out: list[tuple[str, Path]] = []
    for p in sorted(traj_dir.glob("*.mp4")):
        m = _CAMERA_MP4_RE.match(p.name)
        if m:
            out.append((m.group("cam"), p))
    return out


def _read_mp4(mp4_path: Path) -> np.ndarray:
    reader = imageio.get_reader(str(mp4_path), format="ffmpeg")
    frames = np.stack([f for f in reader])
    reader.close()
    if frames.dtype != np.uint8:
        frames = frames.astype(np.uint8)
    return frames


## NOTE: the legacy ``_backport_one`` helper that wrote out
## ``h5_training_trajectories/<task>/<demo>/<ts>.h5`` (sensor_data-mode h5
## with all RGB frames packed inside) was removed when the per-sample pkl
## path became the canonical input format for Pi0.5 / UniSkill
## training. Callers that depended on the legacy h5 layout should either
## (a) re-extract from the existing ``raw_training_trajectories/`` mp4 +
## state-mode h5 files using an earlier revision of this script, or
## (b) be migrated to read from ``postprocessed_robot_trajectories/`` instead.


def _pack_robot_state_8d(arm_qpos: np.ndarray, gripper_qpos: np.ndarray) -> np.ndarray:
    """Pack the 9-D panda qpos into the 8-D state expected by LfOVLADataset:
    7 arm joints + 1 mean gripper value."""
    state = np.empty((8,), dtype=np.float32)
    state[:7] = arm_qpos.astype(np.float32, copy=False)
    state[7] = float(np.mean(gripper_qpos))
    return state


def _resize_frame_uint8(rgb_hwc: np.ndarray, side: int) -> np.ndarray:
    """Resize one (H, W, 3) uint8 frame to (side, side, 3) uint8 with INTER_AREA."""
    return cv2.resize(rgb_hwc, (side, side), interpolation=cv2.INTER_AREA)


def _emit_postprocessed_robot_trajectories_for_demo(
    src_h5_path: Path,
    *,
    task: str,
    demo_id: str,
    bucket: str,
    pi05_data_root: Path,
    robot_cameras: list[str],
    image_size: int,
    action_horizon: int,
    prompt: str,
    sample_idx: int,
    overwrite: bool,
) -> int:
    """Read one source h5 + sibling camera mp4s and write one ``{idx}.pkl`` per
    valid timestep into ``pi05_data_root``. Returns the number of pkls written.
    """
    src_traj_dir = src_h5_path.parent
    cams = _camera_mp4_files(src_traj_dir)
    cam_dict = {name: path for name, path in cams}
    missing = [c for c in robot_cameras if c not in cam_dict]
    if missing:
        logger.warning("  pi05: skip %s — missing mp4s for %s", src_h5_path, missing)
        return 0

    cam_frames_resized: dict[str, np.ndarray] = {}
    for cam in robot_cameras:
        raw_frames = _read_mp4(cam_dict[cam])  # (T, H, W, 3) uint8
        T_src = raw_frames.shape[0]
        small = np.empty((T_src, image_size, image_size, 3), dtype=np.uint8)
        for ti in range(T_src):
            small[ti] = _resize_frame_uint8(raw_frames[ti], image_size)
        cam_frames_resized[cam] = small

    with h5py.File(src_h5_path, "r") as f:
        actions = f["traj_0/actions"][()]  # (T-1, A)
        art_group_keys = list(f["traj_0/env_states/articulations"].keys())
        # The qpos slice live at offset 13:22 in the articulation row (root_pose +
        # root_lin_vel + root_ang_vel = 13 cols before qpos).
        robot_key = "panda_wristcam_hd" if "panda_wristcam_hd" in art_group_keys else art_group_keys[0]
        art = f[f"traj_0/env_states/articulations/{robot_key}"][()]

    qpos_full = art[:, _ARTICULATION_QPOS_OFFSET : _ARTICULATION_QPOS_OFFSET + _ARTICULATION_QPOS_LEN].astype(np.float32)
    T_act = int(actions.shape[0])
    # Number of valid sample start timesteps: t in [0, T_act - action_horizon).
    n_samples = max(0, T_act - action_horizon)
    if n_samples == 0:
        return 0

    pi05_data_root.mkdir(parents=True, exist_ok=True)
    written = 0
    for t in range(n_samples):
        out_path = pi05_data_root / f"{sample_idx + t}.pkl"
        if out_path.exists() and not overwrite:
            written += 1
            continue
        per_cam_uint8: dict[str, np.ndarray] = {cam: cam_frames_resized[cam][t] for cam in robot_cameras}
        sample = {
            "task": task,
            "bucket": bucket,
            "robot_h5": str(src_h5_path),
            "demo_id": demo_id,
            "robot_time_index": int(t),
            "actions": actions[t : t + action_horizon].astype(np.float32, copy=False),
            "qpos": qpos_full[t].astype(np.float32, copy=False),
            "state": _pack_robot_state_8d(qpos_full[t, :7], qpos_full[t, 7:9]),
            "prompt": prompt,
            "robot_cameras": list(robot_cameras),
            **{f"observation/{cam}": per_cam_uint8[cam] for cam in robot_cameras},
        }
        tmp = pi05_data_root / f".{sample_idx + t}.tmp.pkl"
        with open(tmp, "wb") as fh:
            pickle.dump(sample, fh, protocol=pickle.HIGHEST_PROTOCOL)
        tmp.replace(out_path)
        written += 1
    return written


def _emit_trailing_pkls_for_demo(
    src_h5_path: Path,
    *,
    task: str,
    demo_id: str,
    bucket: str,
    pi05_data_root: Path,
    robot_cameras: list[str],
    image_size: int,
    action_horizon: int,
    prompt: str,
    sample_idx_start: int,
    existing_n_samples: int,
    overwrite: bool,
) -> tuple[int, list[dict]]:
    """Write the trailing pkls for one demo: ``t in [existing_n_samples, T_act)``.

    These cover the timesteps the pi05 horizon trim originally dropped. Their
    action chunks are *padded with the last real action* up to ``action_horizon``
    so the on-disk pkl shape is identical to the existing ones — only callers
    whose horizon fits without padding (filtered via the per-baseline index
    files) should sample these. Returns ``(num_written, new_index_entries)``.
    """
    from decord import VideoReader, cpu
    src_traj_dir = src_h5_path.parent
    cams = _camera_mp4_files(src_traj_dir)
    cam_dict = {name: path for name, path in cams}
    missing = [c for c in robot_cameras if c not in cam_dict]
    if missing:
        logger.warning("  extend: skip %s — missing mp4s for %s", src_h5_path, missing)
        return 0, []

    with h5py.File(src_h5_path, "r") as f:
        actions = f["traj_0/actions"][()]  # (T_act, A)
        art_keys = list(f["traj_0/env_states/articulations"].keys())
        robot_key = "panda_wristcam_hd" if "panda_wristcam_hd" in art_keys else art_keys[0]
        art = f[f"traj_0/env_states/articulations/{robot_key}"][()]
    T_act = int(actions.shape[0])
    qpos_full = art[:, _ARTICULATION_QPOS_OFFSET : _ARTICULATION_QPOS_OFFSET + _ARTICULATION_QPOS_LEN].astype(np.float32)

    new_ts = list(range(existing_n_samples, T_act))
    if not new_ts:
        return 0, []

    # Decode only the trailing camera frames (much cheaper than reading the
    # full mp4). Per-camera frame index t corresponds to action timestep t.
    cam_trailing: dict[str, np.ndarray] = {}
    for cam in robot_cameras:
        vr = VideoReader(str(cam_dict[cam]), ctx=cpu(0))
        idxs = [min(t, len(vr) - 1) for t in new_ts]
        raw = vr.get_batch(idxs).asnumpy()  # (n, H, W, 3) uint8
        resized = np.empty((raw.shape[0], image_size, image_size, 3), dtype=np.uint8)
        for i, frame in enumerate(raw):
            resized[i] = _resize_frame_uint8(frame, image_size)
        cam_trailing[cam] = resized

    pi05_data_root.mkdir(parents=True, exist_ok=True)
    written = 0
    new_entries: list[dict] = []
    for i, t in enumerate(new_ts):
        global_idx = sample_idx_start + i
        out_path = pi05_data_root / f"{global_idx}.pkl"
        new_entries.append({"idx": global_idx, "task": task,
                            "demo_id": demo_id, "bucket": bucket, "t": int(t)})
        if out_path.exists() and not overwrite:
            written += 1
            continue
        # actions[t : t+horizon] padded by repeating the last real action.
        chunk = actions[t : t + action_horizon].astype(np.float32, copy=False)
        if chunk.shape[0] < action_horizon:
            last = chunk[-1:]
            pad = np.repeat(last, action_horizon - chunk.shape[0], axis=0)
            chunk = np.concatenate([chunk, pad], axis=0)
        per_cam_uint8 = {cam: cam_trailing[cam][i] for cam in robot_cameras}
        sample = {
            "task": task,
            "bucket": bucket,
            "robot_h5": str(src_h5_path),
            "demo_id": demo_id,
            "robot_time_index": int(t),
            "actions": chunk,
            "qpos": qpos_full[t].astype(np.float32, copy=False),
            "state": _pack_robot_state_8d(qpos_full[t, :7], qpos_full[t, 7:9]),
            "prompt": prompt,
            "robot_cameras": list(robot_cameras),
            **{f"observation/{cam}": per_cam_uint8[cam] for cam in robot_cameras},
        }
        tmp = pi05_data_root / f".{global_idx}.tmp.pkl"
        with open(tmp, "wb") as fh:
            pickle.dump(sample, fh, protocol=pickle.HIGHEST_PROTOCOL)
        tmp.replace(out_path)
        written += 1
    return written, new_entries


def _extend_main(args: argparse.Namespace) -> None:
    """``--extend-only`` mode: append the trailing pkls each demo's pi05 trim
    dropped, then emit per-baseline index files (pi05 unchanged,
    Vid2Robot chunk=4). The on-disk pkls are SHARED across all baselines; the
    index file is what decides which timesteps a baseline samples.
    """
    data_root = Path(os.path.expanduser(args.data_root))
    pi05_root = data_root / args.postprocessed_robot_trajectories_subdir
    pi05_data_root = pi05_root / "data"
    pi05_meta_root = pi05_root / "meta"
    src_root = data_root / args.input_subdir

    # Read existing meta/index.jsonl into per-demo timestep + global-idx info.
    existing_path = pi05_meta_root / "index.jsonl"
    if not existing_path.is_file():
        raise FileNotFoundError(
            f"{existing_path} not found; run process_training_robot_trajectories.py "
            "in its default (non-extend) mode first.")
    existing: list[dict] = []
    with open(existing_path) as f:
        for line in f:
            line = line.strip()
            if not line:
                continue
            existing.append(json.loads(line))
    if not existing:
        raise SystemExit(f"{existing_path} is empty")
    n_existing_per_demo: dict[tuple[str, str], int] = {}
    for e in existing:
        # Legacy entries lack a "bucket" field — treat them as without_distraction
        # (the only bucket that existed pre-migration). New entries include it.
        e_bucket = e.get("bucket", "without_distraction")
        key = (e_bucket, e["task"], e["demo_id"])
        n_existing_per_demo[key] = max(n_existing_per_demo.get(key, 0), int(e["t"]) + 1)
    max_existing_idx = max(int(e["idx"]) for e in existing)
    logger.info("extend: %d existing entries across %d (bucket,task,demo) keys; max idx=%d",
                len(existing), len(n_existing_per_demo), max_existing_idx)

    # Per-demo T_act + pkls to write.
    traj_dirs = _iter_traj_dirs(src_root, args.task)
    if args.max_demos is not None:
        traj_dirs = traj_dirs[: args.max_demos]
    next_idx = max_existing_idx + 1
    all_new_entries: list[dict] = []
    n_tasks_seen = 0
    for bucket, task_name, traj_dir in traj_dirs:
        h5_files = [p for p in traj_dir.glob("*.h5") if ".state." not in p.name]
        if not h5_files:
            continue
        src_h5 = h5_files[0]
        existing_n = n_existing_per_demo.get((bucket, task_name, traj_dir.name), 0)
        try:
            with h5py.File(src_h5, "r") as f:
                T_act = int(f["traj_0/actions"].shape[0])
        except Exception as e:
            logger.error("  extend skip %s (h5 read error): %s", traj_dir, e)
            continue
        if T_act <= existing_n:
            continue  # nothing to add
        wrote, new_entries = _emit_trailing_pkls_for_demo(
            src_h5,
            task=task_name,
            demo_id=traj_dir.name,
            bucket=bucket,
            pi05_data_root=pi05_data_root,
            robot_cameras=list(args.pi05_robot_cameras),
            image_size=args.pi05_image_size,
            action_horizon=args.pi05_action_horizon,
            prompt=args.pi05_prompt,
            sample_idx_start=next_idx,
            existing_n_samples=existing_n,
            overwrite=args.overwrite,
        )
        next_idx += len(new_entries)
        all_new_entries.extend(new_entries)
        n_tasks_seen += 1
        if n_tasks_seen % 50 == 0:
            logger.info("extend: %d demos processed, %d new pkls so far",
                        n_tasks_seen, len(all_new_entries))

    logger.info("extend: wrote %d new pkls across %d demos", len(all_new_entries),
                n_tasks_seen)

    # Per-demo T_act cache for filtering (read once, used for all index splits).
    t_act_per_demo: dict[tuple[str, str, str], int] = {}
    for bucket, task_name, traj_dir in traj_dirs:
        h5_files = [p for p in traj_dir.glob("*.h5") if ".state." not in p.name]
        if h5_files:
            with h5py.File(h5_files[0], "r") as f:
                t_act_per_demo[(bucket, task_name, traj_dir.name)] = int(f["traj_0/actions"].shape[0])

    full_entries = sorted(existing + all_new_entries, key=lambda e: e["idx"])

    def _filter_for_chunk(chunk: int) -> list[dict]:
        out = []
        for e in full_entries:
            e_bucket = e.get("bucket", "without_distraction")
            T_act = t_act_per_demo.get((e_bucket, e["task"], e["demo_id"]))
            if T_act is None:
                continue
            if int(e["t"]) + chunk <= T_act:
                out.append(e)
        return out

    v2r_entries = _filter_for_chunk(args.vid2robot_chunk)

    (pi05_meta_root / "full_index.jsonl").write_text(
        "\n".join(json.dumps(e) for e in full_entries))
    (pi05_meta_root / "vid2robot_index.jsonl").write_text(
        "\n".join(json.dumps(e) for e in v2r_entries))
    logger.info("extend: index files written — full=%d, "
                "vid2robot(chunk=%d)=%d (existing pi05 index.jsonl untouched)",
                len(full_entries),
                args.vid2robot_chunk, len(v2r_entries))


def main() -> None:
    parser = argparse.ArgumentParser()
    parser.add_argument("--data-root", default=os.environ.get("ROBOREEL_DATA"),
                        help="Dataset root (default: $ROBOREEL_DATA).")
    parser.add_argument(
        "--input-subdir",
        default="raw_training_trajectories",
        help="Subdir under data-root holding the raw state-mode trajectories.",
    )
    parser.add_argument(
        "--model", action="append",
        choices=["all", "vid_cond_pi05", "vlm_pi05", "uniskill", "vid2robot"],
        help="Which method(s) to produce index files for; repeatable "
             "(default: all). The per-sample pkls are shared; only the index "
             "differs. Selecting vid2robot (or all) also runs the tail pass "
             "that writes full_index.jsonl + vid2robot_index.jsonl.",
    )
    parser.add_argument("--task", default=None, help="Only this task subdir.")
    parser.add_argument(
        "--overwrite", action="store_true",
        help="Re-emit pkls even if the per-sample files already exist.",
    )
    parser.add_argument(
        "--max-demos", type=int, default=None,
        help="Process only the first N demos per task (after sort). Useful for smoke tests.",
    )
    parser.add_argument(
        "--postprocessed-robot-trajectories-subdir", default="postprocessed_robot_trajectories",
        help="Subdir under data-root to write the per-sample robot-trajectory pkls into.",
    )
    parser.add_argument(
        "--pi05-robot-cameras", nargs="+",
        default=["base_camera", "hand_camera"],
        help="Robot cameras to include in each pi05 sample pkl (default: base + hand).",
    )
    parser.add_argument(
        "--pi05-image-size", type=int, default=224,
        help="Side length the robot frames are downsampled to in each pi05 pkl (default: 224).",
    )
    parser.add_argument(
        "--pi05-action-horizon", type=int, default=20,
        help="Number of action steps stored per pi05 sample (default: 20).",
    )
    parser.add_argument(
        "--pi05-prompt", default="complete the task",
        help="Language instruction stored in every pi05 sample.",
    )
    parser.add_argument(
        "--pi05-features-subdir", default="video_demonstration_features",
        help="Subdir under data-root holding the pi05 human-demo feature files (recorded in stats.json).",
    )
    parser.add_argument(
        "--pi05-distract", default="without_distraction",
        help="Distraction sub-bucket under video_demonstration_features/ that pi05 samples are paired with.",
    )
    parser.add_argument(
        "--extend-only", action="store_true",
        help="Append the trailing pkls each demo's pi05 horizon trim dropped "
             "(t in [T_act - action_horizon, T_act)); update only the per-baseline "
             "index files (full_index, vid2robot_index). The existing "
             "meta/index.jsonl (pi05) is left untouched.",
    )
    parser.add_argument(
        "--vid2robot-chunk", type=int, default=4,
        help="Vid2Robot chunk_size (action chunk length used to filter vid2robot_index.jsonl).",
    )
    args = parser.parse_args()

    logging.basicConfig(level=logging.INFO, format="%(asctime)s %(levelname)s %(message)s")

    project_root = Path(__file__).resolve().parents[1]
    for pth in (project_root, project_root / "src"):
        if str(pth) not in sys.path:
            sys.path.insert(0, str(pth))

    models = args.model or ["all"]
    if "all" in models:
        models = ["vid_cond_pi05", "vlm_pi05", "uniskill", "vid2robot"]
    # Every method reads the shared pkls + meta/index.jsonl that the base pass
    # writes; only vid2robot needs the extra tail pass on top of it.
    need_extend = "vid2robot" in models

    if args.extend_only:
        _extend_main(args)
        return

    logger.info("processing for model(s): %s%s", ", ".join(models),
                " (vid2robot index pass included)" if need_extend else "")

    data_root = Path(os.path.expanduser(args.data_root))
    src_root = data_root / args.input_subdir
    if not src_root.is_dir():
        raise FileNotFoundError(f"{src_root} not found")

    pi05_root = data_root / args.postprocessed_robot_trajectories_subdir
    pi05_data_root = pi05_root / "data"
    pi05_meta_root = pi05_root / "meta"

    traj_dirs = _iter_traj_dirs(src_root, args.task)
    if not traj_dirs:
        raise SystemExit(f"No trajectory dirs under {src_root}")
    if args.max_demos is not None:
        traj_dirs = traj_dirs[: args.max_demos]
    logger.info("Found %d trajectory dir(s) under %s", len(traj_dirs), src_root)
    logger.info("Writing pi05 per-sample pkls under %s", pi05_data_root)
    pi05_data_root.mkdir(parents=True, exist_ok=True)
    pi05_meta_root.mkdir(parents=True, exist_ok=True)

    sample_idx = 0
    pi05_index: list[dict] = []
    action_dim_seen = 0
    for bucket, task_name, traj_dir in traj_dirs:
        h5_files = [p for p in traj_dir.glob("*.h5") if ".state." not in p.name]
        if not h5_files:
            logger.warning("  skip %s: no h5 file", traj_dir)
            continue
        src_h5 = h5_files[0]
        # --- per-sample pi05 pkls ---
        if True:
            # Skip the demo if its action chunk would be missing required cameras.
            try:
                with h5py.File(src_h5, "r") as f:
                    n_actions = int(f["traj_0/actions"].shape[0])
                    adim = int(f["traj_0/actions"].shape[1])
            except Exception as e:
                logger.error("  pi05 skip %s (h5 read error): %s", traj_dir, e)
                continue
            action_dim_seen = max(action_dim_seen, adim)
            n_samples = max(0, n_actions - args.pi05_action_horizon)
            for t in range(n_samples):
                pi05_index.append({"idx": sample_idx + t, "task": task_name,
                                   "demo_id": traj_dir.name, "bucket": bucket,
                                   "t": int(t)})
            try:
                wrote = _emit_postprocessed_robot_trajectories_for_demo(
                    src_h5,
                    task=task_name,
                    bucket=bucket,
                    demo_id=traj_dir.name,
                    pi05_data_root=pi05_data_root,
                    robot_cameras=list(args.pi05_robot_cameras),
                    image_size=args.pi05_image_size,
                    action_horizon=args.pi05_action_horizon,
                    prompt=args.pi05_prompt,
                    sample_idx=sample_idx,
                    overwrite=args.overwrite,
                )
            except Exception as e:
                logger.error("  pi05 error %s: %s", traj_dir, e)
                wrote = 0
            logger.info("  pi05 %s: wrote %d / expected %d sample pkls",
                        traj_dir.relative_to(src_root), wrote, n_samples)
            sample_idx += n_samples

    # --- pi05 meta files (robot side only; LfOVLADataset discovers human-demo
    # feature dirs by scanning video_demonstration_features/ directly) ---
    if True:
        stats = {
            "total_samples": sample_idx,
            "action_dim": action_dim_seen,
            "action_horizon": args.pi05_action_horizon,
            "image_size": args.pi05_image_size,
            "robot_cameras": list(args.pi05_robot_cameras),
            "prompt": args.pi05_prompt,
            "distract": args.pi05_distract,
            "features_subdir": args.pi05_features_subdir,
            "tasks": sorted({entry["task"] for entry in pi05_index}),
        }
        (pi05_meta_root / "stats.json").write_text(json.dumps(stats, indent=2))
        (pi05_meta_root / "index.jsonl").write_text("\n".join(json.dumps(e) for e in pi05_index))
        logger.info("pi05 meta written: %d total samples, %d tasks", stats["total_samples"], len(stats["tasks"]))

    if need_extend:
        # Vid2Robot samples shorter action chunks, so it can use timesteps the
        # pi05 horizon trimmed off the tail of each demo. Append those pkls and
        # write the per-method index files; meta/index.jsonl is left untouched.
        logger.info("running the vid2robot index pass")
        _extend_main(args)


if __name__ == "__main__":
    main()
