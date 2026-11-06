"""
Extract keyframes from a robot manipulation demo's front-camera video using a
VLM, and persist the result as a per-demo JSON annotation at:

    <data_root>/bpp_keyframe_annotations/<bucket>/<task>/<demo_id>.json

Front camera only — multi-camera support was dropped, since the front view
already shows the full task progress. The script writes no JPEGs; only the
JSON. The keyframe list returned by the VLM is always augmented to include
the first and last *sampled* frame indices so downstream consumers have
guaranteed start/end anchors.

Usage:
    python scripts/extract_keyframes.py \\
        --video $ROBOREEL_DATA/human_demos/train/no_distraction/pick_cube/0042/front.mp4 \\
        --task pick_cube
"""

import argparse
import json
import os
import sys
from pathlib import Path

sys.path.insert(0, str(Path(__file__).parent.parent / "src"))

from loguru import logger
from keyframes.pipeline import extract_keyframes
from dataset_layout import identify_human_video
from lfo_task_descriptions import GT_ANNOTATION_PATH, load_gt_annotations

_DEFAULT_DATA_ROOT = os.environ.get("ROBOREEL_DATA")
_ANNOTATION_SUBDIR = "bpp_keyframe_annotations"


def _resolve_task_description(task_arg: str, gt_path: str) -> str:
    """Resolve ``--task`` into the actual prompt sent to the VLM.

    Two accepted forms (checked in order):
      * a single task key (e.g. ``"pick_cube"``) — return the per-task GT
        description.
      * anything else — treat as a literal natural-language description.
    """
    if not task_arg:
        return task_arg
    gt = load_gt_annotations(gt_path)
    if task_arg in gt:
        return gt[task_arg]
    return task_arg


def _annotation_path(data_root: Path, bucket: str, task: str | None,
                     demo_id: str) -> Path:
    """Canonical output JSON path for one demo."""
    base = data_root / _ANNOTATION_SUBDIR
    if task is None:
        raise ValueError(f"task name required for bucket {bucket!r}.")
    return base / bucket / task / f"{demo_id}.json"


def parse_args():
    p = argparse.ArgumentParser(
        description=("Extract per-demo keyframe indices (front camera only) "
                     "and persist them as JSON under "
                     "<data_root>/bpp_keyframe_annotations/."))
    p.add_argument("--video", required=True,
                   help="Path to the input front-camera MP4 file.")
    p.add_argument("--task", required=True,
                   help="Either a per-task key like 'pick_cube' (resolved "
                        "via the GT annotations file) or a literal "
                        "natural-language description.")
    p.add_argument("--data-root", default=_DEFAULT_DATA_ROOT,
                   help=f"LfO dataset root (default: {_DEFAULT_DATA_ROOT}).")
    p.add_argument("--gt-annotations", default=GT_ANNOTATION_PATH,
                   help=f"Path to the task-sentence JSON "
                        f"(default: {GT_ANNOTATION_PATH}).")
    p.add_argument("--bucket", default=None,
                   help="Override the bucket name (otherwise inferred from --video).")
    p.add_argument("--task-name", default=None,
                   help="Override the snake-case task name (otherwise inferred "
                        "from --video).")
    p.add_argument("--demo-id", default=None,
                   help="Override the demo_id (otherwise inferred from --video).")
    p.add_argument("--output-path", default=None,
                   help="If set, write the JSON to this exact path instead of "
                        "the canonical bpp_keyframe_annotations/ layout.")
    p.add_argument("--target-fps", type=float, default=2.0,
                   help="Frame sampling rate (default: 2.0).")
    p.add_argument("--max-frames", type=int, default=32,
                   help="Max frames sent to VLM (default: 32).")
    p.add_argument("--model", default="gpt-5.4-mini",
                   help="OpenAI model name (default: gpt-5.4-mini).")
    p.add_argument("--overwrite", action="store_true",
                   help="Re-run even if the annotation JSON already exists.")
    return p.parse_args()


def main():
    args = parse_args()
    data_root = Path(args.data_root)
    video_path = Path(args.video)

    # Resolve demo identity (bucket, task, demo_id). CLI overrides win.
    inferred: tuple[str, str | None, str] | None = None
    try:
        inferred = identify_human_video(video_path)
    except ValueError:
        if not (args.bucket and args.demo_id):
            raise
    bucket = args.bucket or (inferred and inferred[0])
    task = args.task_name or (inferred and inferred[1])
    demo_id = args.demo_id or (inferred and inferred[2])
    if not (bucket and demo_id):
        raise SystemExit(
            "Could not determine (bucket, demo_id) — pass --bucket / --demo-id "
            "(and --task-name for per-task buckets) or place the video under "
            "<data_root>/human_demos/.")
    if not task:
        raise SystemExit(
            f"Bucket {bucket!r} requires a task name "
            f"(pass --task-name or place the video at "
            f"<data_root>/human_demos/<split>/<bucket>/<task>/<demo>/<cam>.mp4).")

    task_description = _resolve_task_description(args.task, args.gt_annotations)

    if args.output_path:
        out_path = Path(args.output_path)
    else:
        out_path = _annotation_path(data_root, bucket=bucket, task=task,
                                     demo_id=demo_id)
    out_path.parent.mkdir(parents=True, exist_ok=True)

    if out_path.is_file() and not args.overwrite:
        logger.info(f"{out_path} already exists; pass --overwrite to redo. skipping.")
        return

    logger.info(f"Video:  {video_path}")
    logger.info(f"Demo:   bucket={bucket} task={task} demo_id={demo_id}")
    if task_description != args.task:
        logger.info(f"Task:   {args.task} -> {task_description!r}")
    else:
        logger.info(f"Task:   {task_description}")
    logger.info(f"Output: {out_path}")

    # Pure function: returns the manifest dict, no disk I/O. The pipeline
    # always pads keyframe_indices with the first and last sampled frame
    # indices so consumers have guaranteed start/end anchors.
    manifest = extract_keyframes(
        video_path=video_path,
        task_description=task_description,
        target_fps=args.target_fps,
        max_frames=args.max_frames,
        model=args.model,
    )

    record = {
        "bucket": bucket,
        "task": task,
        "demo_id": demo_id,
        "task_description": task_description,
        "model": args.model,
        "video": manifest["video"],
        "sampled_frame_indices": manifest["sampled_frame_indices"],
        "keyframe_indices": manifest["keyframe_indices"],
        "created_at": manifest["created_at"],
    }
    out_path.write_text(json.dumps(record, indent=2))

    n_sampled = len(manifest["sampled_frame_indices"])
    n_keyframes = len(manifest["keyframe_indices"])
    logger.info(f"Sampled {n_sampled} frames → {n_keyframes} keyframes (incl. first+last)")
    logger.info(f"Keyframe indices: {manifest['keyframe_indices']}")
    logger.info(f"Wrote {out_path}")


if __name__ == "__main__":
    main()
