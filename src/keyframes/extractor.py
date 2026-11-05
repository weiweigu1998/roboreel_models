from __future__ import annotations

from pathlib import Path

import cv2
import numpy as np


def sample_frames(
    video_path: str | Path,
    target_fps: float = 2.0,
    max_frames: int = 32,
) -> tuple[list[np.ndarray], list[int]]:
    """Sample frames from an MP4 at approximately target_fps.

    Returns:
        frames: list of HxWx3 uint8 RGB arrays
        frame_indices: corresponding original video frame numbers
    """
    cap = cv2.VideoCapture(str(video_path))
    if not cap.isOpened():
        raise ValueError(f"Cannot open video: {video_path}")

    try:
        total_frames = int(cap.get(cv2.CAP_PROP_FRAME_COUNT))
        native_fps = cap.get(cv2.CAP_PROP_FPS) or 30.0

        n_target = min(max_frames, max(1, round(total_frames * target_fps / native_fps)))
        stride = max(1, total_frames // n_target)

        frame_indices = list(range(0, total_frames, stride))[:max_frames]

        frames: list[np.ndarray] = []
        for idx in frame_indices:
            cap.set(cv2.CAP_PROP_POS_FRAMES, idx)
            ok, frame = cap.read()
            if not ok:
                continue
            frames.append(cv2.cvtColor(frame, cv2.COLOR_BGR2RGB))
    finally:
        cap.release()

    return frames, frame_indices[: len(frames)]
