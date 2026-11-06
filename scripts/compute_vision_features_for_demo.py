"""Precompute SigLIP token embeddings for human demonstration videos and
write one bundled file per (demo, camera).

Reads from::

    <data_root>/human_demos/{train,eval}/{no_distraction,with_distraction}/<task>/<demo_id>/<camera>.mp4

Each demo dir contains one mp4 per camera (e.g. ``front.mp4``,
``egocentric.mp4``). The frames are decoded with ``imageio``, run through
SigLIP, pooled at three patch grids (8x8 / 4x4 / 2x2), and the per-(demo,
camera) embeddings are written as one bundled file at::

    <data_root>/video_demonstration_features/<bucket>/<task>/<demo_id>/<camera>.npy

Each bundled ``.npy`` holds a dict with stacked arrays::

    {
      "image_emb_8x8": (T, 64, 2048) bfloat16,
      "image_emb_4x4": (T, 16, 2048) bfloat16,
      "image_emb_2x2": (T,  4, 2048) bfloat16,
      "pos_emb_8x8":   (T, 64, 768)  float32,
      "pos_emb_4x4":   (T, 16, 768)  float32,
      "pos_emb_2x2":   (T,  4, 768)  float32,
      "state_emb":     (T,  8)       float32,
      "timesteps":     (T,)          int32,
    }

The dataset's ``__getitem__`` reads one ``np.load`` per (sample × demo_camera)
instead of one per timestep — collapsing ~T disk syscalls per sample down to
one.

The SigLIP tokenizer + 3D positional embeddings are built **once** per
process and reused across all (demo, camera) pairs. The previous
implementation rebuilt the tokenizer per call which (a) was slow because
of repeated 1.6GB pickle loads and JAX JIT compiles and (b) leaked device
memory per call.

If existing per-timestep ``token_emb_<t>_<camera>.npy`` files are present in
a demo dir (from older runs of this script), they are stacked into the
bundled ``<camera>.npy`` without re-running SigLIP — no GPU work needed for
the conversion.

For multi-GPU parallel runs, launch one process per GPU with
``CUDA_VISIBLE_DEVICES`` set and ``--shard-id``/``--num-shards`` to
partition demos by index modulo num_shards.

CLI::

    python scripts/compute_vision_features_for_demo.py \\
        --data-root ~/data/lfo_benchmark \\
        [--task pick_cube] [--cameras egocentric front left right] [--overwrite] \\
        [--shard-id 0 --num-shards 2]
"""

from __future__ import annotations

import argparse
import os
import sys
import time
from pathlib import Path

import imageio.v2 as imageio
import numpy as np

#: Cameras emitted by the human-demo capture rig. Override with ``--cameras``.
#: Each camera ``<cam>`` is read from ``<demo_dir>/<cam>.mp4``.
DEFAULT_CAMERAS = ("egocentric", "front", "left", "right")


def _read_camera_mp4(demo_dir: Path, camera: str) -> np.ndarray | None:
    """Return ``(T, H, W, 3)`` uint8 RGB from ``<demo_dir>/<camera>.mp4``, or ``None``."""
    mp4_path = demo_dir / f"{camera}.mp4"
    if not mp4_path.is_file():
        return None
    reader = imageio.get_reader(str(mp4_path), format="ffmpeg")
    frames = np.stack([f for f in reader])
    reader.close()
    if frames.ndim != 4 or frames.shape[-1] != 3:
        return None
    if frames.dtype != np.uint8:
        frames = frames.astype(np.uint8)
    return frames


class _PersistentEncoder:
    """SigLIP tokenizer + 3D positional embeddings, built once and reused.

    Mirrors the pooled outputs that ``DemonstrationBuffer.add_buffer`` produces
    (``image_emb_{8x8,4x4,2x2}``, ``pos_emb_{8x8,4x4,2x2}``, ``state_emb``)
    so the ``token_emb_*.npy`` schema matches the previous implementation
    byte-for-byte.
    """

    def __init__(self, max_T: int = 1024):
        # Lazy imports so ``--help`` and the CLI parse don't pay JAX startup cost.
        import jax
        import jax.numpy as jnp
        import einops  # noqa: F401  (used inside encode below)

        from openpi.shared import image_tools
        from demonstration_vla_suite.shared.siglip_tokenizer import SigLipTokenizer
        from demonstration_vla_suite.shared.posemb_3d import PosEmb3D
        from demonstration_vla_suite.shared.data_utils import pool_tokens_to_size

        self._jax = jax
        self._jnp = jnp
        self._image_tools = image_tools
        self._pool = pool_tokens_to_size
        # Capture einops as an attribute too — keeps the encode() body explicit.
        import einops as _einops
        self._einops = _einops

        siglip = SigLipTokenizer()
        self._siglip_call = jax.jit(siglip.__call__)

        # Match RoboMME default: precompute positional embeddings on CPU when
        # available so the GPU is free for SigLIP forward passes.
        cpu = next((d for d in jax.local_devices() if d.platform == "cpu"), None)
        ctx = jax.default_device(cpu) if cpu is not None else jax.default_device(jax.devices()[0])
        with ctx:
            pos_embedder = PosEmb3D(dim=768)
            ranges = jnp.arange(max_T)
            self._pe = {
                "8x8": np.asarray(pos_embedder(ranges, 8), dtype=np.float32),
                "4x4": np.asarray(pos_embedder(ranges, 4), dtype=np.float32),
                "2x2": np.asarray(pos_embedder(ranges, 2), dtype=np.float32),
            }
        self._max_T = max_T

    def encode_demo_camera(
        self, out_dir: Path, rgb: np.ndarray, camera: str, overwrite: bool
    ) -> str:
        """Encode one camera's frames and write a single bundled ``<camera>.npy``."""
        out_dir.mkdir(parents=True, exist_ok=True)
        T = int(rgb.shape[0])
        if T > self._max_T:
            raise ValueError(f"T={T} exceeds max_T={self._max_T}")

        bundle_path = out_dir / f"{camera}.npy"
        if bundle_path.is_file() and not overwrite:
            return f"skip-existing bundle T={T}"

        # Fast path: if old per-timestep files already exist, just stack them
        # instead of re-running SigLIP. Saves a lot of GPU time when migrating
        # an existing extraction to the bundled format.
        per_step = [out_dir / f"token_emb_{t}_{camera}.npy" for t in range(T)]
        if all(p.is_file() for p in per_step):
            return self._bundle_from_existing(per_step, bundle_path, camera)

        x = (rgb.astype(np.float32) / 255.0) * 2.0 - 1.0
        x = self._jnp.array(x[:, None, ...])  # (T, 1, H, W, 3) — num_views=1
        x_flat = self._einops.rearrange(x, "k v h w c -> (k v) h w c")
        x_flat = self._image_tools.resize_with_pad(x_flat, 224, 224)
        x = self._einops.rearrange(x_flat, "(k v) h w c -> k v h w c", k=T, v=1)
        out = self._siglip_call(x)  # (T, 1, 256, 2048) bfloat16

        # Pooled per-grid image embeddings. Drop the singleton view axis so the
        # bundled shape is (T, P, D) rather than (T, 1, P, D).
        p8 = np.asarray(self._jax.device_get(self._pool(out, 64)))[:, 0]   # (T, 64, 2048)
        p4 = np.asarray(self._jax.device_get(self._pool(out, 16)))[:, 0]   # (T, 16, 2048)
        p2 = np.asarray(self._jax.device_get(self._pool(out, 4)))[:, 0]    # (T,  4, 2048)

        bundle = {
            "image_emb_8x8": p8,
            "image_emb_4x4": p4,
            "image_emb_2x2": p2,
            "pos_emb_8x8":   self._pe["8x8"][:T],
            "pos_emb_4x4":   self._pe["4x4"][:T],
            "pos_emb_2x2":   self._pe["2x2"][:T],
            "state_emb":     np.zeros((T, 8), dtype=np.float32),
            "timesteps":     np.arange(T, dtype=np.int32),
        }
        self._atomic_save(bundle_path, bundle)
        return f"wrote bundle T={T}"

    @staticmethod
    def _atomic_save(path: Path, obj: dict) -> None:
        """``np.save`` to ``<path>.tmp`` then rename, so an interrupted run
        never leaves a half-written ``.npy``. ``np.save`` auto-appends ``.npy``
        if the suffix isn't present, so we use a tmp name that already ends
        in ``.npy``."""
        tmp = path.with_name(f".{path.stem}.tmp.npy")
        np.save(tmp, obj, allow_pickle=True)
        tmp.replace(path)

    def _bundle_from_existing(self, per_step_paths: list[Path], bundle_path: Path, camera: str) -> str:
        """Stack per-timestep ``token_emb_<t>_<cam>.npy`` files into one
        bundled ``<cam>.npy`` without running SigLIP. Used when migrating an
        existing extraction layout to the new bundled output.
        """
        per_step_dicts: list[dict] = []
        for p in per_step_paths:
            raw = np.load(p, allow_pickle=True)
            d = raw.item() if raw.shape == () else raw
            per_step_dicts.append(d)
        keys_with_t_axis = [
            "image_emb_8x8", "image_emb_4x4", "image_emb_2x2",
            "pos_emb_8x8", "pos_emb_4x4", "pos_emb_2x2",
        ]
        bundle: dict = {}
        for key in keys_with_t_axis:
            stacked = np.stack([np.asarray(d[key]) for d in per_step_dicts], axis=0)
            # The old per-timestep schema kept a leading view axis of 1 for
            # image/pos. Squeeze it so the bundled shape is (T, P, D).
            if stacked.ndim >= 3 and stacked.shape[1] == 1:
                stacked = stacked[:, 0]
            bundle[key] = stacked
        # state_emb was a per-step (8,) vector; stack to (T, 8).
        bundle["state_emb"] = np.stack(
            [np.asarray(d["state_emb"]).astype(np.float32) for d in per_step_dicts], axis=0
        )
        bundle["timesteps"] = np.arange(len(per_step_dicts), dtype=np.int32)
        self._atomic_save(bundle_path, bundle)
        return f"bundled-from-existing T={len(per_step_dicts)}"


def main() -> None:
    parser = argparse.ArgumentParser()
    parser.add_argument(
        "--data-root",
        default=os.environ.get("ROBOREEL_DATA"),
        help="Dataset root (default: $ROBOREEL_DATA).",
    )
    parser.add_argument(
        "--video-types",
        nargs="+",
        default=["without_distraction", "with_distraction"],
        help="Distraction buckets to process (default: both).",
    )
    parser.add_argument(
        "--output-subdir",
        default="video_demonstration_features",
        help="Subdir under data-root to write per-timestep token embeddings into "
             "(default: video_demonstration_features).",
    )
    parser.add_argument(
        "--task",
        default=None,
        help="If set, only process this task.",
    )
    parser.add_argument(
        "--max-T", type=int, default=1024,
        help="Max frame count per demo the SigLIP encoder can handle in one "
             "shot. Longer demos error out. Bump this for buckets whose demos "
             "exceed 1024 frames.",
    )
    parser.add_argument(
        "--cameras",
        nargs="+",
        default=list(DEFAULT_CAMERAS),
        help="Cameras to extract. Each <cam> is read from <demo_dir>/<cam>.mp4.",
    )
    parser.add_argument(
        "--overwrite",
        action="store_true",
        help="Recompute embeddings even if all per-timestep .npy files already exist.",
    )
    parser.add_argument(
        "--max-demos",
        type=int,
        default=None,
        help="Process only the first N demos per task (after sort). Useful for smoke tests.",
    )
    parser.add_argument(
        "--shard-id",
        type=int,
        default=0,
        help="Shard index for multi-GPU runs. With ``--num-shards N``, this process "
             "handles demos whose sorted index ``i`` satisfies ``i %% N == shard_id``.",
    )
    parser.add_argument(
        "--num-shards",
        type=int,
        default=1,
        help="Total number of parallel shards (default: 1, i.e. no sharding).",
    )
    args = parser.parse_args()

    project_root = Path(__file__).resolve().parents[1]
    for pth in (project_root, project_root / "src", project_root / "src" / "external_models"):
        sp = str(pth)
        if sp not in sys.path:
            sys.path.insert(0, sp)

    data_root = Path(os.path.expanduser(args.data_root))
    feat_root = data_root / args.output_subdir
    feat_root.mkdir(parents=True, exist_ok=True)

    from dataset_layout import iter_human_demos
    tasks = [args.task] if args.task else None
    demo_dirs = [
        (f"{bucket}/{task}", demo_dir)
        for bucket in args.video_types
        for task, _demo_id, demo_dir in iter_human_demos(data_root, bucket, tasks)
    ]
    if not demo_dirs:
        raise SystemExit(f"No human demos for {args.video_types} under {data_root}/human_demos")
    if args.max_demos is not None:
        demo_dirs = demo_dirs[: args.max_demos]
    if args.num_shards > 1:
        demo_dirs = [
            (t, d) for i, (t, d) in enumerate(demo_dirs)
            if i % args.num_shards == args.shard_id
        ]
        print(f"[shard {args.shard_id}/{args.num_shards}] {len(demo_dirs)} demos", flush=True)
    else:
        print(f"Found {len(demo_dirs)} demo dir(s) under {data_root}/human_demos")
    print(f"Writing token embeddings under {feat_root}", flush=True)

    encoder: "_PersistentEncoder | None" = None  # lazy: only built if SigLIP needed

    def _list_existing_per_step(out_dir: Path, camera: str) -> list[Path]:
        return sorted(out_dir.glob(f"token_emb_*_{camera}.npy"))

    for task_name, demo_dir in demo_dirs:
        out_dir = feat_root / task_name / demo_dir.name
        out_dir.mkdir(parents=True, exist_ok=True)
        statuses: list[str] = []
        t0 = time.time()
        try:
            for cam in args.cameras:
                bundle_path = out_dir / f"{cam}.npy"
                # Path 1: bundle already exists.
                if bundle_path.is_file() and not args.overwrite:
                    statuses.append(f"{cam}=skip-bundle")
                    continue
                # Path 2: per-timestep files exist — stack into a bundle without
                # decoding mp4 or running SigLIP (very fast, no GPU needed).
                per_step = _list_existing_per_step(out_dir, cam)
                if per_step:
                    if encoder is None:
                        # Use the dedicated cheap path; _bundle_from_existing
                        # doesn't need JAX/SigLIP, so we don't pay the heavy
                        # _PersistentEncoder init here.
                        _bundle_from_existing_static(per_step, bundle_path)
                    else:
                        encoder._bundle_from_existing(per_step, bundle_path, cam)
                    statuses.append(f"{cam}=bundled-from-existing T={len(per_step)}")
                    continue
                # Path 3: need to actually extract — pay mp4 decode + lazy SigLIP init.
                rgb = _read_camera_mp4(demo_dir, cam)
                if rgb is None:
                    statuses.append(f"{cam}=missing")
                    continue
                if encoder is None:
                    print(f"Initializing SigLIP encoder (max_T={args.max_T})...", flush=True)
                    encoder = _PersistentEncoder(max_T=args.max_T)
                status = encoder.encode_demo_camera(out_dir, rgb, cam, args.overwrite)
                statuses.append(f"{cam}={status}")
            if not statuses:
                print(f"  skip {demo_dir}: no requested cameras found", flush=True)
                continue
            elapsed = time.time() - t0
            print(
                f"  {demo_dir.relative_to(data_root)} -> {out_dir.relative_to(feat_root)}: "
                f"{', '.join(statuses)} [{elapsed:.1f}s]",
                flush=True,
            )
        except Exception as e:
            print(f"  error {demo_dir}: {e}", flush=True)


def _bundle_from_existing_static(per_step_paths: list[Path], bundle_path: Path) -> None:
    """Module-level twin of :meth:`_PersistentEncoder._bundle_from_existing` —
    usable without spinning up JAX/SigLIP."""
    per_step_dicts: list[dict] = []
    for p in per_step_paths:
        raw = np.load(p, allow_pickle=True)
        d = raw.item() if raw.shape == () else raw
        per_step_dicts.append(d)
    keys_with_t_axis = [
        "image_emb_8x8", "image_emb_4x4", "image_emb_2x2",
        "pos_emb_8x8", "pos_emb_4x4", "pos_emb_2x2",
    ]
    bundle: dict = {}
    for key in keys_with_t_axis:
        stacked = np.stack([np.asarray(d[key]) for d in per_step_dicts], axis=0)
        if stacked.ndim >= 3 and stacked.shape[1] == 1:
            stacked = stacked[:, 0]
        bundle[key] = stacked
    bundle["state_emb"] = np.stack(
        [np.asarray(d["state_emb"]).astype(np.float32) for d in per_step_dicts], axis=0
    )
    bundle["timesteps"] = np.arange(len(per_step_dicts), dtype=np.int32)
    tmp = bundle_path.with_name(f".{bundle_path.stem}.tmp.npy")
    np.save(tmp, bundle, allow_pickle=True)
    tmp.replace(bundle_path)


if __name__ == "__main__":
    os.environ.setdefault("XLA_PYTHON_CLIENT_PREALLOCATE", "false")
    main()
