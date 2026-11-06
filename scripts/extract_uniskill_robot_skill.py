#!/usr/bin/env python3
"""Extract robot-trajectory skill embeddings using the upstream UniSkill IDM.

Thin wrapper around ``UniSkill/extract_skill.py`` — the robot-side skill
extraction shares one core implementation with the human-side
``scripts/extract_uniskill_human_skill.py`` rather than duplicating the
IDM + Depth-Anything forward loop.

This script only adds the LfO-specific glue:

  * **Discovery** — reads ``postprocessed_robot_trajectories/meta/index.jsonl``
    and groups sample idxs by ``(task, demo_id)``.
  * **Frame loading** — stacks ``observation/<camera>`` from each demo's
    per-sample pkls into a ``(T, H, W, 3)`` array.
  * **Output formatting** — pads the upstream ``(T-interval, skill_dim)``
    result to ``(T, 1, skill_dim)`` and writes the robomimic layout that
    ``SequenceDataset(goal_mode="skill")`` consumes:

        <skill-dir>/<task>/demo_<i>/base.npy        (T, 1, skill_dim)
        <skill-dir>/<task>/demo_<i>/aug_<k>.npy

The core frames→IDM→skills computation is entirely
``extract_skill.extract_latents_for_demo`` — unmodified upstream code.

Usage:
    python scripts/extract_uniskill_robot_skill.py \
        --samples-root $ROBOREEL_DATA/postprocessed_robot_trajectories \
        --idm-checkpoint /path/to/idm.pth \
        --skill-dir /path/to/skills \
        --aug-num 5 --device cuda:0
"""

from __future__ import annotations

import argparse
import json
import os
import pickle
import sys
from collections import defaultdict

import numpy as np
import torch
from transformers import AutoImageProcessor

# Resolve ``from extract_skill import ...`` to the upstream UniSkill module.
_REPO_ROOT = os.path.dirname(os.path.dirname(os.path.abspath(__file__)))
_UNISKILL_ROOT = os.path.join(_REPO_ROOT, "src", "external_models", "UniSkill")
if os.path.isdir(_UNISKILL_ROOT) and _UNISKILL_ROOT not in sys.path:
    sys.path.insert(0, _UNISKILL_ROOT)

from extract_skill import extract_latents_for_demo, load_depth_estimator, load_idm


def parse_args() -> argparse.Namespace:
    p = argparse.ArgumentParser(description="Extract robot skills via the upstream UniSkill IDM")
    p.add_argument("--samples-root", required=True,
                   help="postprocessed_robot_trajectories root holding data/ and meta/.")
    p.add_argument("--task", default=None, help="Optional: only this task.")
    p.add_argument("--camera-key", default="base_camera",
                   help="Which observation/<camera> key to read from each pkl.")
    p.add_argument("--idm-checkpoint", required=True, help="Pretrained IDM weights (.pth).")
    p.add_argument("--skill-dir", required=True, help="Output directory for skill .npy files.")
    p.add_argument("--depth-model", default="depth-anything/Depth-Anything-V2-Small-hf")
    p.add_argument("--aug-num", type=int, default=5,
                   help="Number of augmented skill variants per demo. Each "
                        "aug_<k>.npy is the ISD output on a freshly "
                        "color-jittered (I_t, I_{t+k}) pair (paper recipe).")
    p.add_argument("--image-jitter", action=argparse.BooleanOptionalAction,
                   default=True,
                   help="Apply real torchvision ColorJitter to the raw frames "
                        "before each aug forward, matching the paper's "
                        "policy-time augmentation. Disable with "
                        "--no-image-jitter to fall back to the legacy "
                        "Gaussian-noise-in-skill-space behaviour "
                        "(aug = base + aug_noise * N(0,I)).")
    p.add_argument("--jitter-brightness", type=float, default=0.4,
                   help="Max brightness jitter (XSkill default, paper "
                        "doesn't specify a value).")
    p.add_argument("--jitter-contrast",   type=float, default=0.4)
    p.add_argument("--jitter-saturation", type=float, default=0.4)
    p.add_argument("--jitter-hue",        type=float, default=0.1)
    p.add_argument("--aug-noise", type=float, default=0.1,
                   help="Legacy: stdev of Gaussian noise added in skill space "
                        "when --no-image-jitter is set. Unused with the "
                        "default image-jitter recipe.")
    p.add_argument("--batch-size", type=int, default=64)
    p.add_argument("--skill-interval", type=int, default=20,
                   help="Temporal offset between IDM frame pairs (per the "
                        "UniSkill paper's stage-2 policy recipe, k=20). Must "
                        "match the value used by the human-side "
                        "extract_uniskill_human_skill.py so robot+human skill "
                        "banks are operating at the same temporal scope.")
    p.add_argument("--prefetch-workers", type=int, default=2,
                   help="Background threads for batch prep (passed to extract_latents_for_demo).")
    p.add_argument("--amp", action="store_true", help="Enable autocast in the model forwards.")
    p.add_argument("--overwrite", action="store_true",
                   help="Re-extract demos even if their skill files already exist.")
    # IDM architecture params — consumed by the upstream load_idm.
    p.add_argument("--num-layers", type=int, default=8)
    p.add_argument("--num-heads", type=int, default=4)
    p.add_argument("--hidden-dim", type=int, default=256)
    p.add_argument("--skill-dim", type=int, default=64)
    p.add_argument("--out-dim", type=int, default=768)
    p.add_argument("--idm-resolution", type=int, default=224)
    p.add_argument("--device", default="cuda")
    p.add_argument("--index-filename", default="index.jsonl",
                   help="meta/<filename> to read sample index from. Pass "
                        "full_index.jsonl to align with robomimic HDF5s built "
                        "via convert_lfo_to_robomimic --index-filename full_index.jsonl.")
    return p.parse_args()


def _entry_bucket(entry: dict) -> str:
    """Resolve the bucket for an index entry. New entries carry an explicit
    ``"bucket"`` field; legacy entries (written before the bucket migration)
    predate the split and are all ``without_distraction``.
    """
    return entry.get("bucket") or "without_distraction"


def discover_demos(samples_root: str, task_filter: str | None,
                   index_filename: str = "index.jsonl",
                   ) -> list[tuple[str, str, list[tuple[str, list[int]]]]]:
    """Group sample idxs by ``(bucket, task, demo_id)`` from
    ``meta/<index_filename>``.

    Returns ``[(bucket, task, [(demo_id, [sorted-by-t sample_idx, ...]), ...]),
    ...]`` with deterministic ordering.

    ``index_filename`` defaults to ``index.jsonl`` (pi05's pi05-horizon-trimmed
    range). Pass ``full_index.jsonl`` for the post-extend full T_act range.
    """
    index_path = os.path.join(samples_root, "meta", index_filename)
    if not os.path.isfile(index_path):
        raise FileNotFoundError(
            f"Expected {index_path}. Run scripts/process_training_robot_trajectories.py first."
        )
    # Key: (bucket, task). For flat buckets we use ``task = bucket`` so the
    # downstream output path can be ``<output>/<bucket>/demo_<i>/``.
    groups: dict[tuple[str, str], dict[str, list[tuple[int, int]]]] = defaultdict(
        lambda: defaultdict(list))
    with open(index_path) as f:
        for line in f:
            line = line.strip()
            if not line:
                continue
            e = json.loads(line)
            bucket = _entry_bucket(e)
            task = e["task"]
            groups[(bucket, task)][e["demo_id"]].append((int(e["t"]), int(e["idx"])))

    out: list[tuple[str, str, list[tuple[str, list[int]]]]] = []
    for (bucket, task) in sorted(groups):
        if task_filter is not None and task != task_filter:
            continue
        demos = [(demo_id, [idx for _t, idx in sorted(groups[(bucket, task)][demo_id])])
                 for demo_id in sorted(groups[(bucket, task)])]
        if demos:
            out.append((bucket, task, demos))
    return out


def load_demo_frames(samples_root: str, sample_idxs: list[int], camera_key: str) -> np.ndarray:
    """Stack ``observation/<camera_key>`` across a demo's pkls → (T,H,W,3) uint8."""
    data_dir = os.path.join(samples_root, "data")
    obs_key = f"observation/{camera_key}"
    frames = []
    for idx in sample_idxs:
        with open(os.path.join(data_dir, f"{idx}.pkl"), "rb") as fh:
            frames.append(pickle.load(fh)[obs_key])
    return np.stack(frames, axis=0)


def _demo_already_done(out_dir: str, aug_num: int) -> bool:
    if not os.path.isfile(os.path.join(out_dir, "base.npy")):
        return False
    return all(os.path.isfile(os.path.join(out_dir, f"aug_{k}.npy"))
               for k in range(aug_num))


def main() -> None:
    args = parse_args()
    device = torch.device(args.device)

    groups = discover_demos(args.samples_root, args.task,
                            index_filename=args.index_filename)
    if not groups:
        print(f"No demos found under {args.samples_root}")
        return
    total = sum(len(demos) for _b, _t, demos in groups)
    print(f"Discovered {total} demos across {len(groups)} (bucket, task) group(s) in {args.samples_root}")

    print(f"Loading IDM from {args.idm_checkpoint}...")
    idm, skill_dim = load_idm(args, device)
    print(f"Loading depth estimator: {args.depth_model}...")
    depth_processor = AutoImageProcessor.from_pretrained(args.depth_model)
    depth_estimator = load_depth_estimator(args.depth_model, device)

    done = skipped = 0
    for bucket, task_name, demos in groups:
        print(f"\nBucket {bucket!r} / Task {task_name!r}: {len(demos)} demos")
        for demo_idx, (demo_id, sample_idxs) in enumerate(demos):
            out_dir = os.path.join(args.skill_dir, bucket, task_name, f"demo_{demo_idx}")
            if not args.overwrite and _demo_already_done(out_dir, args.aug_num):
                skipped += 1
                continue

            base_path = os.path.join(out_dir, "base.npy")
            reuse_base = (not args.overwrite) and os.path.isfile(base_path)
            if reuse_base:
                skills = np.load(base_path).astype(np.float32)
                T = skills.shape[0]
            else:
                frames = load_demo_frames(args.samples_root, sample_idxs, args.camera_key)
                T = frames.shape[0]

                skills = extract_latents_for_demo(
                    frames,
                    idm=idm,
                    idm_resolution=args.idm_resolution,
                    depth_processor=depth_processor,
                    depth_estimator=depth_estimator,
                    device=device,
                    batch_size=args.batch_size,
                    interval=args.skill_interval,
                    skill_dim=skill_dim,
                    prefetch_workers=args.prefetch_workers,
                    use_amp=args.amp,
                )

                if skills.ndim == 2:
                    skills = skills.reshape(0, 1, skills.shape[-1])
                if skills.shape[0] == 0:
                    skills = np.zeros((T, 1, skill_dim), dtype=np.float32)
                elif skills.shape[0] < T:
                    pad = np.tile(skills[-1:], (T - skills.shape[0], 1, 1))
                    skills = np.concatenate([skills, pad], axis=0)
                skills = skills.astype(np.float32)

                os.makedirs(out_dir, exist_ok=True)
                np.save(base_path, skills)

            # Augmentation strategy:
            #   --image-jitter (default, paper recipe): for each aug_k, apply
            #     a fresh random torchvision ColorJitter to the raw frames and
            #     re-run the ISD forward. Each aug_<k>.npy is therefore the
            #     ISD output on a *different* color-domain perturbation of
            #     the same (I_t, I_{t+k}) pair — narrows the train/test
            #     embodiment gap by exposing the policy to the kind of
            #     appearance shift it will see when prompted by a human
            #     video at eval time.
            #   --no-image-jitter (legacy): add Gaussian noise in skill
            #     space (skills + sigma * N(0,I)). Cheaper (no extra ISD
            #     forwards) but theoretically much weaker — the noise is
            #     independent across timesteps and unrelated to the
            #     structured color shift the paper's recipe targets.
            if args.image_jitter and args.aug_num > 0:
                import torch as _torch
                from torchvision.transforms import v2 as _TF
                _jitter = _TF.ColorJitter(
                    brightness=args.jitter_brightness,
                    contrast=args.jitter_contrast,
                    saturation=args.jitter_saturation,
                    hue=args.jitter_hue,
                )
                for k in range(args.aug_num):
                    # Per-(demo, aug) seed so the same demo's aug_k is
                    # reproducible across overwrite/redo runs.
                    seed = (hash((demo_idx, k)) ^ 0x9E3779B1) & 0x7FFFFFFF
                    g = _torch.Generator().manual_seed(seed)
                    _torch.manual_seed(seed)
                    # frames: (T, H, W, 3) uint8 → (T, 3, H, W) uint8 for
                    # torchvision v2 ColorJitter, then back.
                    t = _torch.from_numpy(frames).permute(0, 3, 1, 2).contiguous()
                    jittered = _jitter(t)
                    jittered_frames = (
                        jittered.permute(0, 2, 3, 1).contiguous().numpy()
                    ).astype(np.uint8)
                    aug_skills = extract_latents_for_demo(
                        jittered_frames,
                        idm=idm,
                        idm_resolution=args.idm_resolution,
                        depth_processor=depth_processor,
                        depth_estimator=depth_estimator,
                        device=device,
                        batch_size=args.batch_size,
                        interval=args.skill_interval,
                        skill_dim=skill_dim,
                        prefetch_workers=args.prefetch_workers,
                        use_amp=args.amp,
                    )
                    if aug_skills.ndim == 2:
                        aug_skills = aug_skills.reshape(0, 1, aug_skills.shape[-1])
                    if aug_skills.shape[0] == 0:
                        aug_skills = np.zeros((T, 1, skill_dim), dtype=np.float32)
                    elif aug_skills.shape[0] < T:
                        pad = np.tile(aug_skills[-1:], (T - aug_skills.shape[0], 1, 1))
                        aug_skills = np.concatenate([aug_skills, pad], axis=0)
                    aug_skills = aug_skills.astype(np.float32)
                    np.save(os.path.join(out_dir, f"aug_{k}.npy"), aug_skills)
            else:
                for k in range(args.aug_num):
                    aug = skills + args.aug_noise * np.random.randn(*skills.shape).astype(np.float32)
                    np.save(os.path.join(out_dir, f"aug_{k}.npy"), aug)
            done += 1
            print(f"  demo_{demo_idx} (id {demo_id}): {T} frames -> {skills.shape}  "
                  f"(aug: {'image-jitter' if args.image_jitter else 'gauss-noise'})")

    print(f"\nDone. Extracted {done} demos, skipped {skipped} already-present. "
          f"Skills under {args.skill_dir}/")


if __name__ == "__main__":
    main()
