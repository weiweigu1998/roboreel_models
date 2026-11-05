#!/usr/bin/env python3
"""Pre-extract human prompt videos to per-demo npy frame bundles.

Mirrors the ``LfODataset`` npy-bundle pattern (see ``src/demonstration_vla_suite/
dataset/lfo_vla_dataset.py``): one ``.npy`` per (demo, camera) so the training
dataset only loads the frames it actually needs (``np.load(..., mmap_mode='r')``
+ index) instead of decoding the full mp4 each ``__getitem__``.

Spatial-only downsample: every frame of the mp4 is resized to ``S x S`` at
extraction time (default S=224, matching Vid2RobotConfig) and stacked as a
single ``(T, S, S, 3) uint8`` array. Temporal sampling (the paper's 16-frame
first/last/sorted-middle sampler) stays at the dataset layer so frame indices
re-randomize each ``__getitem__``.

Output layout — kept separate from the read-only ``raw_video_demonstrations``
tree so extracted bundles can be deleted/rebuilt independently::

    <data_root>/downsampled_demonstration_videos/<video_type>/<task>/<trajectory_idx>/<camera>.npy

The image size is implicit in the bundle's stored shape; pass ``--out-subdir``
to write to a parallel tree if you need multiple resolutions co-resident.
"""

from __future__ import annotations

import argparse
import os
import sys
from pathlib import Path

import numpy as np
from PIL import Image

REPO = Path(__file__).resolve().parents[1]
for p in (REPO, REPO / "src"):
    sp = str(p)
    if sp not in sys.path:
        sys.path.insert(0, sp)

from lfo_data import HUMAN_VIDEO_SUBDIR  # noqa: E402


def parse_args() -> argparse.Namespace:
    p = argparse.ArgumentParser(description="Pre-extract mp4 frames to npy bundles.")
    p.add_argument("--data-root", required=True,
                   help="LfO data root (contains raw_video_demonstrations/).")
    p.add_argument("--out-subdir", default="downsampled_demonstration_videos",
                   help="Output root, relative to --data-root. "
                        "Default mirrors the raw_video_demonstrations/ layout under "
                        "downsampled_demonstration_videos/<type>/<task>/<demo>/<cam>.npy.")
    p.add_argument("--image-size", type=int, default=224,
                   help="Spatial resize target (S). Output shape is "
                        "(T, S, S, 3) uint8. Default 224 matches Vid2RobotConfig.")
    p.add_argument("--camera", default="front",
                   help="Camera basename for the mp4 (e.g. 'front').")
    p.add_argument("--tasks", nargs="*", default=None,
                   help="Restrict to a subset of task dirs (default: all).")
    p.add_argument("--video-types", nargs="*",
                   default=["without_distraction", "with_distraction"],
                   help="Distraction buckets to walk.")
    p.add_argument("--overwrite", action="store_true",
                   help="Re-extract even if the .npy already exists.")
    p.add_argument("--no-task-subdir", action="store_true",
                   help="Bucket has no per-task layer; demos sit directly under "
                        "the bucket. Output drops the task segment.")
    return p.parse_args()


def extract_one(mp4: Path, out: Path, image_size: int) -> tuple[int, int]:
    from decord import VideoReader, cpu
    vr = VideoReader(str(mp4), ctx=cpu(0))
    total = len(vr)
    if total <= 0:
        raise ValueError(f"Empty video: {mp4}")
    raw = vr.get_batch(list(range(total))).asnumpy()  # (T, H, W, 3) uint8
    arr = np.empty((total, image_size, image_size, 3), dtype=np.uint8)
    for i, frame in enumerate(raw):
        arr[i] = np.asarray(
            Image.fromarray(frame).resize((image_size, image_size), Image.BILINEAR),
            dtype=np.uint8)
    # ``np.save`` appends ``.npy`` to bare paths; opening the tmp file ourselves
    # avoids that surprise so the atomic rename targets the right name.
    tmp = out.with_suffix(out.suffix + ".tmp")
    with open(tmp, "wb") as f:
        np.save(f, arr)
    os.replace(tmp, out)
    return total, arr.nbytes


def main() -> None:
    args = parse_args()
    data_root = Path(args.data_root)
    in_root = data_root / HUMAN_VIDEO_SUBDIR
    if not in_root.is_dir():
        raise SystemExit(f"missing {in_root}")
    out_root = data_root / args.out_subdir

    npy_name = f"{args.camera}.npy"
    mp4_name = f"{args.camera}.mp4"
    total_mp4s = total_kept = total_written = total_bytes = 0
    for video_type in args.video_types:
        type_in = in_root / video_type
        if not type_in.is_dir():
            continue
        # Flat-bucket mode: demos sit directly under the bucket dir, with no
        # per-task layer. Synthesise an empty "task" name so the output path
        # collapses to <bucket>/<demo>/<cam>.npy.
        if args.no_task_subdir:
            task_iter = [("", type_in)]
        else:
            task_iter = [(p.name, p) for p in sorted(type_in.iterdir()) if p.is_dir()]
        for task_name, task_dir in task_iter:
            if args.tasks is not None and task_name not in args.tasks:
                continue
            for demo_dir in sorted(task_dir.iterdir()):
                if not demo_dir.is_dir():
                    continue
                mp4 = demo_dir / mp4_name
                if not mp4.is_file():
                    continue
                total_mp4s += 1
                out_dir = out_root / video_type / task_name / demo_dir.name
                out = out_dir / npy_name
                if out.is_file() and not args.overwrite:
                    total_kept += 1
                    continue
                out_dir.mkdir(parents=True, exist_ok=True)
                t, nbytes = extract_one(mp4, out, args.image_size)
                total_written += 1
                total_bytes += nbytes
                if total_written % 50 == 0:
                    print(f"[extract] wrote {total_written} npys "
                          f"(latest {out.relative_to(out_root)} | {t} frames, "
                          f"{nbytes/1e6:.1f} MB)", flush=True)
    print(f"[extract] done. mp4s={total_mp4s}  kept(existing)={total_kept}  "
          f"written={total_written}  size={total_bytes/1e9:.2f} GB  "
          f"out_root={out_root}", flush=True)


if __name__ == "__main__":
    main()
