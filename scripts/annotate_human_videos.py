#!/usr/bin/env python3
"""Language-annotate every human demonstration trajectory with a VLM.

For each human demonstration video (front camera by default) under
``raw_video_demonstrations/<video_type>/<task>/<demo>/``, run a vision-language
model and store a short natural-language task *instruction* (an imperative
command) for the demonstrated task. Robot trajectories are *not* annotated.

Output: ``<data_root>/human_video_annotation_<model_name>.json`` — a JSON object

    {
      "model": "<hf model id>", "prompt": "...", "camera": "front",
      "num_frames": 16,
      "annotations": { "<video_type>/<task>/<demo>": "<annotation>", ... }
    }

Re-running with ``--skip-existing`` (the default) annotates only the demos
missing from ``annotations``, so late-arriving data is picked up without
re-annotating anything already done. The JSON is flushed periodically, so an
interrupted run resumes cleanly.

Default model: Qwen3-VL-4B-Instruct. Run in the ``uniskill`` conda env
(needs transformers >= 5).
"""

from __future__ import annotations

import argparse
import json
import os
import time
from pathlib import Path

import numpy as np
import torch
from decord import VideoReader, cpu
from PIL import Image
from transformers import AutoModelForImageTextToText, AutoProcessor

HUMAN_SUBDIR = "raw_video_demonstrations"

DEFAULT_PROMPT = (
    "These images are frames sampled in chronological order from a video of a "
    "person performing a manipulation task. Write the complete and clean task "
    "instruction for this video so a different human operator can reproduce the "
    "task without looking at the video."
)


def parse_args() -> argparse.Namespace:
    p = argparse.ArgumentParser(description=__doc__.splitlines()[0])
    p.add_argument("--data-root", default=os.environ.get("ROBOREEL_DATA"),
                   help="Dataset root holding raw_video_demonstrations/.")
    p.add_argument("--model", default="Qwen/Qwen3-VL-4B-Instruct",
                   help="HuggingFace model id of the VLM.")
    p.add_argument("--model-name", default="qwen3-vl-4b",
                   help="Short slug used in the output filename.")
    p.add_argument("--camera", default="front",
                   help="Which per-demo camera mp4 to annotate.")
    p.add_argument("--video-types", nargs="+",
                   default=["without_distraction", "with_distraction"],
                   help="Distraction buckets to scan.")
    p.add_argument("--tasks", nargs="+", default=None,
                   help="Limit to these task names (default: all present).")
    p.add_argument("--num-frames", type=int, default=16,
                   help="Number of frames uniformly sampled per video.")
    p.add_argument("--frame-size", type=int, default=640,
                   help="Max longer-side pixels per frame; larger frames are "
                        "downscaled to bound the VLM token count.")
    p.add_argument("--max-new-tokens", type=int, default=96)
    p.add_argument("--prompt", default=DEFAULT_PROMPT)
    p.add_argument("--device", default="cuda:0")
    p.add_argument("--skip-existing", action=argparse.BooleanOptionalAction,
                   default=True,
                   help="Skip demos already present in the output JSON "
                        "(default: on — so re-runs only annotate new data).")
    p.add_argument("--limit", type=int, default=None,
                   help="Annotate at most this many demos (for testing).")
    p.add_argument("--save-every", type=int, default=20,
                   help="Flush the JSON every N annotations.")
    p.add_argument("--no-task-subdir", action="store_true",
                   help="Bucket has no per-task layer; demos sit directly under "
                        "the bucket dir (key becomes '<video_type>/<demo>').")
    return p.parse_args()


def discover_demos(data_root: str, video_types, camera: str, tasks,
                   no_task_subdir: bool = False):
    """Return a sorted list of ``(key, mp4_path)`` for every human demo video.

    With per-task layout (default), ``key`` is ``"<video_type>/<task>/<demo>"``.
    With ``no_task_subdir`` set, demos sit directly under the bucket and
    ``key`` is ``"<video_type>/<demo>"``.
    """
    root = Path(data_root) / HUMAN_SUBDIR
    out: list[tuple[str, str]] = []
    for video_type in video_types:
        vt_dir = root / video_type
        if not vt_dir.is_dir():
            continue
        if no_task_subdir:
            for demo_dir in sorted(p for p in vt_dir.iterdir() if p.is_dir()):
                mp4 = demo_dir / f"{camera}.mp4"
                if mp4.is_file():
                    out.append((f"{video_type}/{demo_dir.name}", str(mp4)))
            continue
        for task_dir in sorted(p for p in vt_dir.iterdir() if p.is_dir()):
            if tasks and task_dir.name not in tasks:
                continue
            for demo_dir in sorted(p for p in task_dir.iterdir() if p.is_dir()):
                mp4 = demo_dir / f"{camera}.mp4"
                if mp4.is_file():
                    out.append((f"{video_type}/{task_dir.name}/{demo_dir.name}",
                                str(mp4)))
    return out


def main() -> None:
    args = parse_args()
    out_path = Path(args.data_root) / f"human_video_annotation_{args.model_name}.json"

    annotations: dict[str, str] = {}
    if out_path.is_file():
        prev = json.loads(out_path.read_text())
        annotations = dict(prev.get("annotations", {}))
        print(f"[annotate] loaded {len(annotations)} existing annotations "
              f"from {out_path}", flush=True)

    demos = discover_demos(args.data_root, args.video_types, args.camera, args.tasks,
                           no_task_subdir=args.no_task_subdir)
    todo = [(k, v) for k, v in demos
            if not (args.skip_existing and k in annotations)]
    if args.limit is not None:
        todo = todo[: args.limit]
    print(f"[annotate] {len(demos)} human demo videos found; {len(todo)} to "
          f"annotate (skip_existing={args.skip_existing})", flush=True)
    if not todo:
        print("[annotate] nothing to do.", flush=True)
        return

    def save():
        out_path.write_text(json.dumps({
            "model": args.model,
            "prompt": args.prompt,
            "camera": args.camera,
            "num_frames": args.num_frames,
            "annotations": dict(sorted(annotations.items())),
        }, indent=2))

    print(f"[annotate] loading {args.model} on {args.device} ...", flush=True)
    model = AutoModelForImageTextToText.from_pretrained(
        args.model, dtype="auto").to(args.device).eval()
    processor = AutoProcessor.from_pretrained(args.model)

    def annotate(mp4_path: str) -> str:
        # Sample frames with decord and feed them as images. Qwen3-VL's native
        # "video" path trips a get_rope_index StopIteration in transformers
        # 5.3.0; frames-as-images sidesteps it and is fine for captioning.
        vr = VideoReader(mp4_path, ctx=cpu(0))
        total = len(vr)
        idxs = [int(round(i)) for i in np.linspace(0, total - 1, args.num_frames)]
        frames = []
        for j in idxs:
            img = Image.fromarray(vr[j].asnumpy())
            longest = max(img.size)
            if longest > args.frame_size:
                scale = args.frame_size / longest
                img = img.resize((round(img.width * scale), round(img.height * scale)),
                                 Image.BILINEAR)
            frames.append(img)
        content = [{"type": "image", "image": f} for f in frames]
        content.append({"type": "text", "text": args.prompt})
        messages = [{"role": "user", "content": content}]
        inputs = processor.apply_chat_template(
            messages, add_generation_prompt=True, tokenize=True,
            return_dict=True, return_tensors="pt").to(args.device)
        with torch.no_grad():
            generated = model.generate(**inputs, max_new_tokens=args.max_new_tokens,
                                       do_sample=False)
        new_tokens = generated[:, inputs["input_ids"].shape[1]:]
        text = processor.batch_decode(new_tokens, skip_special_tokens=True)[0]
        return " ".join(text.split())

    t0, done = time.time(), 0
    for i, (key, mp4) in enumerate(todo, 1):
        try:
            annotations[key] = annotate(mp4)
            done += 1
        except Exception as exc:  # one bad video must not sink the whole run
            print(f"[annotate] ERROR {key}: {type(exc).__name__}: {exc}", flush=True)
            continue
        if i % args.save_every == 0 or i == len(todo):
            save()
            print(f"[annotate] {i}/{len(todo)}  ({(time.time() - t0) / i:.1f}s/demo)"
                  f"  {key} -> {annotations[key][:80]}", flush=True)
    save()
    print(f"[annotate] done — {done} newly annotated, {len(annotations)} total "
          f"-> {out_path}", flush=True)


if __name__ == "__main__":
    main()
