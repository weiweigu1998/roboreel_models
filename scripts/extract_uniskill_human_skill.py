#!/usr/bin/env python3
"""Extract UniSkill skill latents from RoboReel human demonstration videos.

Refers to ``UniSkill/extract_skill.py`` (which works on LIBERO ``.pkl`` files)
and adapts the same IDM/Depth-Anything pipeline to walk our human demo mp4s:

    <data-root>/raw_video_demonstrations/<distract>/<task>/<demo>/<camera>.mp4

For each video, runs the IDM on consecutive ``(t, t + skill_interval)`` frame
pairs and writes the per-pair skill sequence to:

    <output-root>/<distract>/<task>/<demo>/<camera>.npy
        shape (num_pairs, 1, skill_dim) float32
        num_pairs = T - skill_interval

A single GPU is used; the device id is set via ``--device cuda:N``. The default
``--skill-interval`` is 20 (per the UniSkill paper).

Usage:
    python scripts/extract_uniskill_human_skill.py \\
        --idm-checkpoint /path/to/idm.pth \\
        --output-root $ROBOREEL_DATA/uniskill_skills \\
        --device cuda:0 \\
        [--data-root $ROBOREEL_DATA] \\
        [--tasks pick_cube empty_basket] \\
        [--cameras front egocentric] \\
        [--human-distract] \\
        [--skill-interval 20] [--batch-size 64] [--amp] [--overwrite] [--progress]
"""

from __future__ import annotations

import argparse
import os
import sys
from pathlib import Path
from typing import Iterable, List

import imageio.v2 as imageio
import numpy as np
import torch
from transformers import AutoImageProcessor

# Make ``from extract_skill import ...`` resolve to the upstream module.
_PROJECT_ROOT = Path(__file__).resolve().parents[1]
_UNISKILL_ROOT = _PROJECT_ROOT / "src" / "external_models" / "UniSkill"
if str(_UNISKILL_ROOT) not in sys.path:
    sys.path.insert(0, str(_UNISKILL_ROOT))

from extract_skill import (  # noqa: E402  -- import after sys.path mutation
    extract_latents_for_demo,
    load_depth_estimator,
    load_idm,
)

#: Sub-directory layout under ``--data-root``.
RAW_VIDEO_SUBDIR = "raw_video_demonstrations"
DISTRACT_DIRS = {True: "with_distraction", False: "without_distraction"}

#: Human demo camera. The LfO benchmark conditions on the front view only.
DEFAULT_CAMERAS = ("front",)


def parse_args() -> argparse.Namespace:
    p = argparse.ArgumentParser(description=__doc__.splitlines()[0])

    # --- data + output ---
    p.add_argument(
        "--data-root",
        type=Path,
        default=(Path(os.environ["ROBOREEL_DATA"]) if os.environ.get("ROBOREEL_DATA") else None),
        help="LfO benchmark root containing raw_video_demonstrations/.",
    )
    p.add_argument(
        "--output-root",
        type=Path,
        required=True,
        help="Directory to write per-(task, demo, camera) skill .npy files into.",
    )
    p.add_argument(
        "--human-distract",
        action="store_true",
        help="Read from with_distraction/ (default: without_distraction/).",
    )
    p.add_argument(
        "--video-type",
        default=None,
        help="Custom bucket name under raw_video_demonstrations/. "
             "Overrides --human-distract when set.",
    )
    p.add_argument(
        "--no-task-subdir",
        action="store_true",
        help="Bucket has no per-task layer; demos sit directly under the bucket. "
             "Output drops the task segment too "
             "(output-root/<video-type>/<demo>/<camera>.npy).",
    )
    p.add_argument(
        "--tasks",
        nargs="+",
        default=None,
        help="Optional task name filter (e.g. --tasks pick_cube empty_basket). "
             "Default: all tasks present under raw_video_demonstrations/<distract>/.",
    )
    p.add_argument(
        "--cameras",
        nargs="+",
        default=list(DEFAULT_CAMERAS),
        choices=list(DEFAULT_CAMERAS),
        help=f"Cameras to extract skills for. Default: {list(DEFAULT_CAMERAS)}.",
    )

    # --- model checkpoint + arch ---
    p.add_argument(
        "--idm-checkpoint",
        type=Path,
        required=True,
        help="Path to the pretrained UniSkill IDM checkpoint (.pth).",
    )
    p.add_argument(
        "--depth-model",
        type=str,
        default="depth-anything/Depth-Anything-V2-Small-hf",
        help="Hugging Face id or local path for the depth estimator.",
    )
    p.add_argument(
        "--num-layers", type=int, default=8,
        help="IDM transformer layers (must match the checkpoint).",
    )
    p.add_argument(
        "--num-heads", type=int, default=4,
        help="IDM attention heads (must match the checkpoint).",
    )
    p.add_argument(
        "--hidden-dim", type=int, default=256,
        help="IDM transformer hidden dim (must match the checkpoint).",
    )
    p.add_argument(
        "--skill-dim", type=int, default=64,
        help="Latent skill dim emitted by the IDM (must match the checkpoint).",
    )
    p.add_argument(
        "--out-dim", type=int, default=768,
        help="IDM out-dim (unused under return_skill=True; kept for arch parity).",
    )
    p.add_argument(
        "--idm-resolution", type=int, default=224,
        help="Resolution fed to the IDM and depth estimator (default: 224).",
    )

    # --- inference knobs ---
    p.add_argument(
        "--device",
        type=str,
        default="cuda:0",
        help="Designated CUDA device id, e.g. cuda:0 / cuda:3 (default: cuda:0).",
    )
    p.add_argument(
        "--skill-interval",
        type=int,
        default=20,
        help="Temporal offset between frame pairs (default: 20 per the UniSkill paper).",
    )
    p.add_argument(
        "--batch-size", type=int, default=64,
        help="Frame pairs per IDM forward pass (default: 64).",
    )
    p.add_argument(
        "--prefetch-workers", type=int, default=2,
        help="Background threads for batch prep; 0 disables prefetch (default: 2).",
    )
    p.add_argument(
        "--amp", action="store_true",
        help="Enable torch.cuda.amp autocast around model forwards.",
    )
    p.add_argument(
        "--overwrite", action="store_true",
        help="Recompute skills even if the output .npy already exists.",
    )
    p.add_argument(
        "--progress", action="store_true",
        help="Show a tqdm progress bar over (demo, camera) units.",
    )
    return p.parse_args()


def _resolve_video_type(human_distract: bool, video_type: str | None) -> str:
    """``--video-type`` overrides ``--human-distract`` when both supplied."""
    if video_type:
        return video_type
    return DISTRACT_DIRS[bool(human_distract)]


def _iter_demo_videos(
    data_root: Path,
    video_type: str,
    task_filter: List[str] | None,
    cameras: List[str],
    no_task_subdir: bool = False,
) -> List[tuple[str, str, str, Path]]:
    """Return ``[(task, demo, camera, mp4_path), ...]`` in sorted order.

    With ``no_task_subdir`` the bucket has no per-task layer; demos sit
    directly under it and the returned ``task`` segment is ``""`` so output
    paths composed as ``output_root / video_type / task / demo`` collapse
    cleanly to ``output_root / video_type / demo``.
    """
    root = data_root / RAW_VIDEO_SUBDIR / video_type
    if not root.is_dir():
        raise FileNotFoundError(f"{root} not found")
    out: List[tuple[str, str, str, Path]] = []
    if no_task_subdir:
        for demo_dir in sorted(p for p in root.iterdir() if p.is_dir()):
            for cam in cameras:
                mp4 = demo_dir / f"{cam}.mp4"
                if mp4.is_file():
                    out.append(("", demo_dir.name, cam, mp4))
        return out
    for task_dir in sorted(p for p in root.iterdir() if p.is_dir()):
        if task_filter is not None and task_dir.name not in task_filter:
            continue
        for demo_dir in sorted(p for p in task_dir.iterdir() if p.is_dir()):
            for cam in cameras:
                mp4 = demo_dir / f"{cam}.mp4"
                if mp4.is_file():
                    out.append((task_dir.name, demo_dir.name, cam, mp4))
    return out


def _read_mp4_frames(mp4: Path) -> np.ndarray:
    """Return ``(T, H, W, 3)`` uint8 RGB from an mp4."""
    reader = imageio.get_reader(str(mp4), format="ffmpeg")
    frames = np.stack([f for f in reader])
    reader.close()
    if frames.dtype != np.uint8:
        frames = frames.astype(np.uint8)
    if frames.ndim != 4 or frames.shape[-1] != 3:
        raise ValueError(f"Unexpected frame shape {frames.shape} for {mp4}")
    return frames


def main() -> None:
    args = parse_args()

    device = torch.device(args.device)
    if device.type == "cuda":
        if not torch.cuda.is_available():
            raise RuntimeError("CUDA requested but no GPU is available.")
        torch.cuda.set_device(device)
        torch.backends.cudnn.benchmark = True

    args.output_root.mkdir(parents=True, exist_ok=True)
    distract_dir = _resolve_video_type(args.human_distract, args.video_type)

    units = _iter_demo_videos(args.data_root, distract_dir, args.tasks, args.cameras,
                              no_task_subdir=args.no_task_subdir)
    if not units:
        raise SystemExit(
            f"No mp4s found under {args.data_root / RAW_VIDEO_SUBDIR / distract_dir} "
            f"(tasks={args.tasks}, cameras={args.cameras})."
        )

    # --- load IDM + depth model once ---
    idm, skill_dim = load_idm(args, device)
    depth_processor = AutoImageProcessor.from_pretrained(args.depth_model)
    depth_estimator = load_depth_estimator(args.depth_model, device)

    print(
        f"Extracting skills from {len(units)} (demo, camera) units "
        f"on {device}, skill_interval={args.skill_interval}, "
        f"checkpoint={args.idm_checkpoint}, out -> {args.output_root}",
        flush=True,
    )

    iterator: Iterable[tuple[str, str, str, Path]]
    if args.progress:
        from tqdm import tqdm
        iterator = tqdm(units)
    else:
        iterator = units

    skipped = wrote = errored = 0
    for task, demo, cam, mp4 in iterator:
        out_dir = args.output_root / distract_dir / task / demo
        out_npy = out_dir / f"{cam}.npy"
        if out_npy.exists() and not args.overwrite:
            skipped += 1
            continue
        try:
            frames = _read_mp4_frames(mp4)
        except Exception as e:
            print(f"  skip (decode error) {mp4}: {e}", flush=True)
            errored += 1
            continue
        if frames.shape[0] <= args.skill_interval:
            print(
                f"  skip (T={frames.shape[0]} <= interval={args.skill_interval}) {mp4}",
                flush=True,
            )
            errored += 1
            continue
        latents = extract_latents_for_demo(
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
        out_dir.mkdir(parents=True, exist_ok=True)
        # Persist atomically: write to a tmp sibling then rename, so an
        # interrupted run never leaves a half-written .npy. Tmp filename
        # must end in ``.npy`` because ``np.save`` auto-appends the suffix
        # otherwise.
        tmp_path = out_dir / f".{cam}.tmp.npy"
        np.save(tmp_path, latents.astype(np.float32, copy=False))
        tmp_path.replace(out_npy)
        wrote += 1

    print(
        f"done: wrote={wrote} skipped(existing)={skipped} errored={errored} "
        f"total_units={len(units)}",
        flush=True,
    )


if __name__ == "__main__":
    main()
