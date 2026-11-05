"""Shared LfO data helpers — ``postprocessed_robot_trajectories`` → model inputs.

Demo discovery, per-timestep sample loading, and frame decoding, defined once
so the training scripts and the evaluation runners agree on layout. Depends
only on numpy / Pillow at import time (no torch / jax).

Layout assumed:

* robot trajectories — ``<samples_root>/data/<idx>.pkl`` with a
  ``<samples_root>/meta/<index>.jsonl`` listing valid ``(task, demo_id, t, idx)``
  start timesteps. Each model family gets its own index file because the set of
  valid start timesteps depends on the action horizon; the pkls are shared.
* human demo videos — ``<data_root>/raw_video_demonstrations/<bucket>/<task>/<demo>/front.mp4``
"""

from __future__ import annotations

import json
import os
import pickle
from collections import defaultdict

import numpy as np
from PIL import Image

# Stage-0a pkls are written under numpy 2.x; interpreters running numpy 1.x
# need the unpickle compat shim installed before any pickle.load.
# See src/np2_pickle_compat.py.
import sys as _sys
_sys.path.insert(0, os.path.dirname(os.path.abspath(__file__)))
import np2_pickle_compat  # noqa: F401,E402

#: Human demonstration videos: ``raw_video_demonstrations/<bucket>/<task>/<demo>/<cam>.mp4``.
HUMAN_VIDEO_SUBDIR = "raw_video_demonstrations"
#: Distraction buckets a config may draw human demo videos from.
VIDEO_TYPE_DIRS = ("without_distraction", "with_distraction")
#: Default frames sampled per demonstration video. Callers that need a
#: different stack length pass ``num_frames`` explicitly.
NUM_VIDEO_FRAMES = 20

def _infer_entry_bucket(entry: dict) -> str:
    """Resolve the bucket for a ``meta/*.jsonl`` entry. New entries carry an
    explicit ``"bucket"`` field; legacy entries (pre-bucket migration) predate
    the split and are all ``without_distraction``.
    """
    return entry.get("bucket") or "without_distraction"


def discover_demos(samples_root: str, tasks: list[str] | None = None,
                   index_filename: str = "index.jsonl",
                   include_video_types: list[str] | tuple[str, ...] | None = None,
                   ) -> dict[str, dict[str, dict[str, list[int]]]]:
    """Group ``postprocessed_robot_trajectories`` sample idxs by
    ``(bucket, task, demo_id)``.

    Returns ``{bucket: {task: {demo_id: [sorted-by-t sample_idx, ...]}}}``.
    Every robot trajectory whose bucket is in ``include_video_types`` is
    included (``None`` ⇒ no bucket filter).

    ``index_filename`` selects which ``meta/<name>.jsonl`` to read. Default is
    ``index.jsonl``; Vid2Robot passes ``vid2robot_index.jsonl``. The pkls
    referenced are identical — only the set of valid sample-start timesteps
    differs.
    """
    index_path = os.path.join(samples_root, "meta", index_filename)
    if not os.path.isfile(index_path):
        raise FileNotFoundError(
            f"Expected {index_path}. Run scripts/process_training_robot_trajectories.py first."
        )
    allowed_buckets = set(include_video_types) if include_video_types else None
    groups: dict[str, dict[str, dict[str, list[tuple[int, int]]]]] = defaultdict(
        lambda: defaultdict(lambda: defaultdict(list)))
    with open(index_path) as f:
        for line in f:
            line = line.strip()
            if not line:
                continue
            e = json.loads(line)
            if tasks is not None and e["task"] not in tasks:
                continue
            bucket = _infer_entry_bucket(e)
            if allowed_buckets is not None and bucket not in allowed_buckets:
                continue
            groups[bucket][e["task"]][e["demo_id"]].append((int(e["t"]), int(e["idx"])))
    out: dict[str, dict[str, dict[str, list[int]]]] = {}
    for bucket, by_task in groups.items():
        out[bucket] = {}
        for task, demos in by_task.items():
            out[bucket][task] = {
                d: [idx for _t, idx in sorted(pairs)] for d, pairs in demos.items()
            }
    return out


def discover_human_demos(data_root: str, tasks: list[str] | None = None,
                         include_video_types=("without_distraction",),
                         max_demo_index: int | None = None
                         ) -> dict[str, dict[str, dict[str, str]]]:
    """Return ``{bucket: {task: {demo_id: front-cam mp4 path}}}`` of human demos.

    ``demo_id`` is the demo directory name (e.g. ``"0042"``).

    ``max_demo_index`` caps the numeric demo id (e.g. 79 → demos 0000-0079).
    The video encoders condition on the ``front`` camera only, per the LfO
    shared protocol.
    """
    out: dict[str, dict[str, dict[str, str]]] = defaultdict(
        lambda: defaultdict(dict))
    for video_type in include_video_types:
        type_root = os.path.join(data_root, HUMAN_VIDEO_SUBDIR, video_type)
        if not os.path.isdir(type_root):
            continue
        for task in sorted(os.listdir(type_root)):
            if tasks is not None and task not in tasks:
                continue
            task_dir = os.path.join(type_root, task)
            if not os.path.isdir(task_dir):
                continue
            for demo in sorted(os.listdir(task_dir)):
                try:
                    demo_id = int(demo, 10)
                except ValueError:
                    continue
                if max_demo_index is not None and demo_id > max_demo_index:
                    continue
                mp4 = os.path.join(task_dir, demo, "front.mp4")
                if os.path.isfile(mp4):
                    out[video_type][task][demo] = mp4
    # Materialize the nested defaultdicts so downstream isinstance checks
    # don't trip over them.
    return {b: {t: dict(d) for t, d in by_task.items()} for b, by_task in out.items()}


def preprocess_image(rgb_uint8: np.ndarray, image_size: int = 100) -> np.ndarray:
    """Resize an (H,W,3) uint8 frame to (image_size, image_size, 3) float32 in [0,1]."""
    img = Image.fromarray(rgb_uint8).resize((image_size, image_size), Image.BILINEAR)
    return (np.asarray(img, dtype=np.float32) / 255.0)


def load_video_frames(mp4_path: str, num_frames: int = NUM_VIDEO_FRAMES,
                      image_size: int = 100, rng: np.random.Generator | None = None
                      ) -> np.ndarray:
    """Decode an mp4 to ``(num_frames, image_size, image_size, 3)`` float32 in [0,1].

    With ``rng`` the frame indices are a random subsample (re-randomised each
    call, for training); otherwise they are uniformly spaced (deterministic
    eval). If the clip is shorter than ``num_frames`` the sample is taken with
    replacement so the encoder always sees a fixed-length stack.
    """
    from decord import VideoReader, cpu
    vr = VideoReader(mp4_path, ctx=cpu(0))
    total = len(vr)
    if total <= 0:
        raise ValueError(f"Empty video: {mp4_path}")
    if rng is not None:
        replace = total < num_frames
        idxs = sorted(int(i) for i in rng.choice(total, size=num_frames, replace=replace))
    else:
        idxs = [int(round(i)) for i in np.linspace(0, total - 1, num_frames)]
    frames = vr.get_batch(idxs).asnumpy()  # (num_frames, H, W, 3) uint8
    out = np.empty((num_frames, image_size, image_size, 3), dtype=np.float32)
    for i, frame in enumerate(frames):
        out[i] = preprocess_image(frame, image_size)
    return out


def load_sample(samples_root: str, idx: int) -> dict:
    """Load one postprocessed per-timestep pkl."""
    with open(os.path.join(samples_root, "data", f"{idx}.pkl"), "rb") as f:
        return pickle.load(f)
