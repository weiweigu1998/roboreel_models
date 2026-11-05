from __future__ import annotations

from datetime import datetime, timezone
from pathlib import Path

from dotenv import load_dotenv

load_dotenv()

from keyframes.extractor import sample_frames
from keyframes.vlm_client import select_keyframes


def extract_keyframes(
    video_path: str | Path,
    task_description: str,
    target_fps: float = 4.0,
    max_frames: int = 32,
    model: str = "gpt-5.4-mini",
) -> dict:
    """Sample frames from ``video_path`` and run the VLM keyframe selector.

    Returns the manifest dict in memory; this function performs no disk I/O
    (the caller decides where to persist the result). The dict has::

        {
          "video":                 str,            # the input MP4 path
          "task":                  str,            # natural-language prompt
          "model":                 str,            # OpenAI model name used
          "sampled_frame_indices": list[int],      # original-video frame indices
          "keyframe_indices":      list[int],      # VLM-selected subset
          "created_at":            iso8601 UTC,
        }
    """
    video_path = Path(video_path)
    frames, frame_indices = sample_frames(video_path, target_fps=target_fps,
                                          max_frames=max_frames)
    keyframe_indices = list(select_keyframes(frames, frame_indices,
                                             task_description, model=model))

    # Always include the first and last *sampled* frames as keyframes,
    # regardless of what the VLM returns — they bracket the trajectory and
    # are useful anchors for downstream consumers. Sorted-unique union with
    # the VLM's picks.
    forced = [frame_indices[0], frame_indices[-1]] if frame_indices else []
    keyframe_indices = sorted(set(keyframe_indices) | set(forced))

    return {
        "video": str(video_path),
        "task": task_description,
        "model": model,
        "sampled_frame_indices": list(frame_indices),
        "keyframe_indices": keyframe_indices,
        "created_at": datetime.now(timezone.utc).isoformat(),
    }
