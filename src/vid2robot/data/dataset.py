"""LfO -> Vid2Robot dataset adapter.

Builds (prompt_video, state_video, robot_full_video, action_bins, text_embed)
tuples from:
  * ``human_demos/<split>/<type>/<task>/<demo>/front.mp4`` — prompt video
  * ``postprocessed_robot_trajectories/data/<idx>.pkl``        — robot history
    (uses ``observation/base_camera``) + 4-step future action chunk.

Reuses the shared helpers (``src/lfo_data.py``) for demo discovery so every
model family pairs robot trajectories with human demos the same way.
"""

from __future__ import annotations

import os
from collections import defaultdict
from dataclasses import dataclass

import numpy as np
from PIL import Image
from torch.utils.data import Dataset

from dataset_layout import identify_human_video
from lfo_data import discover_demos, discover_human_demos, load_sample

from .action_codec import ActionCodec, fit_codec_from_samples
from .sampling import sample_prompt_indices, sample_state_indices
from .text_embeddings import build_text_embeddings


def _resize(rgb_uint8: np.ndarray, image_size: int) -> np.ndarray:
    """(H, W, 3) uint8 -> (image_size, image_size, 3) float32 in [0,1]."""
    img = Image.fromarray(rgb_uint8).resize((image_size, image_size), Image.BILINEAR)
    return np.asarray(img, dtype=np.float32) / 255.0


@dataclass
class _DemoIndex:
    """All sample idxs for one ``(bucket, task, demo_id)`` robot trajectory."""
    bucket: str
    task: str
    demo_id: str
    sample_idxs: list[int]   # sorted by t (i.e. sample_idxs[t] -> pkl idx)


class Vid2RobotDataset(Dataset):
    """One element = one ``(robot_demo, current_t, human_prompt_video)`` triple.

    Iteration: for each robot trajectory we step through every timestep ``t``.
    Each ``__getitem__`` randomly samples a *human* demo video from the same
    task and randomly samples 16 frames for the prompt video, 16 frames for
    the robot full-demo video, plus the 8-frame history ending at ``t``.

    The dataset can be used in two modes:
      * ``with_robot_video=True`` (default) — returns the auxiliary robot
        full-clip needed by VVCL / VTCL / TCC.
      * ``with_robot_video=False`` — main action loss only, smaller batches.
    """

    def __init__(self,
                 data_root: str,
                 tasks: list[str],
                 *,
                 samples_subdir: str = "postprocessed_robot_trajectories",
                 robot_camera: str = "base_camera",
                 image_size: int = 224,
                 num_prompt_frames: int = 16,
                 num_state_frames: int = 8,
                 chunk_size: int = 4,
                 include_video_types: tuple[str, ...] = ("without_distraction",),
                 max_demo_index: int | None = None,
                 codec: ActionCodec | None = None,
                 seed: int = 0,
                 with_robot_video: bool = True,
                 prompt_camera: str = "front",
                 prompt_frames_subdir: str = "downsampled_demonstration_videos",
                 prompt_frame_cache_size: int = 4096,
                 text_encoder_model_id: str = "google/siglip-base-patch16-224",
                 text_encoder_device: str = "cpu") -> None:
        self.data_root = data_root
        self.samples_root = os.path.join(data_root, samples_subdir)
        self.robot_camera = robot_camera
        self.image_size = image_size
        self.num_prompt_frames = num_prompt_frames
        self.num_state_frames = num_state_frames
        self.chunk_size = chunk_size
        self.with_robot_video = with_robot_video
        self.prompt_camera = prompt_camera
        # Pre-extracted frame bundles live at
        #   {data_root}/{prompt_frames_subdir}/{video_type}/{task}/{demo}/<cam>.npy
        # Written by scripts/downsample_demo_resolution.py.
        self._prompt_frames_root = os.path.join(data_root, prompt_frames_subdir)
        # Per-process LRU cache of np.load(..., mmap_mode='r') views, keyed by
        # npy path. Mirrors LfOVLADataset._load_bundle (lfo_vla_dataset.py:265).
        self._prompt_frame_cache: dict[str, np.ndarray] = {}
        self._prompt_frame_cache_size = int(prompt_frame_cache_size)
        self._warned_missing_npy = False
        self.rng = np.random.default_rng(seed)

        # Robot trajectories — filtered by include_video_types (same buckets as
        # the human side, per the new same-bucket-same-task sampling rule).
        # Uses the vid2robot-specific index (filtered by chunk_size=4) so the
        # trailing pkls whose padded action chunk would leak into the chunk
        # are excluded.
        self.include_video_types = tuple(include_video_types)
        groups = discover_demos(self.samples_root, tasks=list(tasks),
                                index_filename="vid2robot_index.jsonl",
                                include_video_types=list(include_video_types))
        self.demos: list[_DemoIndex] = []
        for bucket, by_task in groups.items():
            for task, by_demo in by_task.items():
                for demo_id, sample_idxs in by_demo.items():
                    self.demos.append(_DemoIndex(
                        bucket=bucket, task=task, demo_id=demo_id,
                        sample_idxs=sample_idxs))
        if not self.demos:
            raise RuntimeError(f"No robot demos discovered under {self.samples_root}")

        # Each entry: (demo_idx, t)
        self.items: list[tuple[int, int]] = []
        for i, d in enumerate(self.demos):
            for t in range(len(d.sample_idxs)):
                self.items.append((i, t))

        # Human prompt videos — same buckets as robot side; nested as
        # {bucket: {task: {demo_id: mp4_path}}}.
        self.human_demos = discover_human_demos(
            data_root, tasks=list(tasks),
            include_video_types=list(include_video_types),
            max_demo_index=max_demo_index,
        )
        for bucket in self.include_video_types:
            missing = [t for t in tasks if not self.human_demos.get(bucket, {}).get(t)]
            if missing:
                raise RuntimeError(f"No human prompt videos for bucket {bucket!r} tasks {missing}")

        self.codec = codec or self._fit_default_codec()

        # Frozen SigLIP text embeddings for VTCL (Paper §II-E (4)). Computed
        # once and cached as numpy arrays so the dataset state stays
        # serialisable across DataLoader workers.
        #
        # One key per task name, with the curated GT sentence as the
        # encoded text.
        self._text_embeddings: dict[str, np.ndarray] = build_text_embeddings(
            list(tasks), device=text_encoder_device,
            model_id=text_encoder_model_id)

    # ------------------------------------------------------------------
    def _fit_default_codec(self, n_samples: int = 200) -> ActionCodec:
        """Build an ActionCodec from a small sweep of the trajectories."""
        idxs = self.rng.choice(len(self.items), size=min(n_samples, len(self.items)),
                               replace=False)
        actions = []
        for i in idxs:
            demo_idx, t = self.items[int(i)]
            pkl_idx = self.demos[demo_idx].sample_idxs[t]
            pkl = load_sample(self.samples_root, pkl_idx)
            actions.append(np.asarray(pkl["actions"], dtype=np.float32))
        stacked = np.concatenate(actions, axis=0)  # (N*horizon, 8)
        return fit_codec_from_samples(stacked, num_bins=256)

    # ------------------------------------------------------------------
    def __len__(self) -> int:
        return len(self.items)

    # ------------------------------------------------------------------
    def _load_pkl(self, demo: _DemoIndex, t: int) -> dict:
        return load_sample(self.samples_root, demo.sample_idxs[t])

    def _state_video(self, demo: _DemoIndex, t: int) -> np.ndarray:
        idxs = sample_state_indices(len(demo.sample_idxs), t, self.num_state_frames)
        out = np.empty((self.num_state_frames, self.image_size, self.image_size, 3),
                       dtype=np.float32)
        for i, ti in enumerate(idxs):
            pkl = self._load_pkl(demo, ti)
            out[i] = _resize(pkl[f"observation/{self.robot_camera}"], self.image_size)
        return out

    def _robot_full_video(self, demo: _DemoIndex, rng: np.random.Generator
                          ) -> np.ndarray:
        L = len(demo.sample_idxs)
        idxs = sample_prompt_indices(L, self.num_prompt_frames, rng=rng)
        out = np.empty((self.num_prompt_frames, self.image_size, self.image_size, 3),
                       dtype=np.float32)
        for i, ti in enumerate(idxs):
            pkl = self._load_pkl(demo, ti)
            out[i] = _resize(pkl[f"observation/{self.robot_camera}"], self.image_size)
        return out

    # ------------------------------------------------------------------
    def _npy_path_for_mp4(self, mp4: str) -> str:
        """Map ``human_demos/<split>/<type>/<task>/<demo>/<cam>.mp4`` ->
        ``<data_root>/<prompt_frames_subdir>/<type>/<task>/<demo>/<cam>.npy``."""
        video_type, task, demo = identify_human_video(mp4)
        return os.path.join(self._prompt_frames_root, video_type, task, demo,
                            f"{self.prompt_camera}.npy")

    def _load_prompt_frame_bundle(self, mp4: str) -> np.ndarray | None:
        """Return ``(T, S, S, 3) uint8`` memmap for ``mp4`` if its npy bundle
        exists; otherwise ``None`` (caller falls back to decord)."""
        npy_path = self._npy_path_for_mp4(mp4)
        cached = self._prompt_frame_cache.get(npy_path)
        if cached is not None:
            return cached
        if not os.path.isfile(npy_path):
            return None
        frames = np.load(npy_path, mmap_mode="r")  # (T, S, S, 3) uint8
        # Simple FIFO bound — same shape as LfOVLADataset._load_bundle.
        if len(self._prompt_frame_cache) >= self._prompt_frame_cache_size:
            self._prompt_frame_cache.pop(next(iter(self._prompt_frame_cache)))
        self._prompt_frame_cache[npy_path] = frames
        return frames

    def _human_prompt_video(self, bucket: str, task: str,
                            rng: np.random.Generator
                            ) -> tuple[np.ndarray, str]:
        # Random pick within the same bucket + same task.
        pool = self.human_demos[bucket][task]
        demo_ids = list(pool.keys())
        mp4 = pool[demo_ids[int(rng.integers(0, len(demo_ids)))]]
        bundle = self._load_prompt_frame_bundle(mp4)
        if bundle is not None:
            total = int(bundle.shape[0])
            idxs = sample_prompt_indices(total, self.num_prompt_frames, rng=rng)
            # mmap indexing copies only the requested frames into a new uint8
            # buffer; cast/normalize to float32 in [0, 1].
            sub = np.ascontiguousarray(bundle[idxs])  # (num_frames, S, S, 3) uint8
            out = sub.astype(np.float32) / 255.0
            return out, mp4

        # No npy bundle yet — fall back to decord seeking only the needed frames
        # (still lazy: VideoReader.get_batch decodes just those indices, not the
        # whole clip). Log once so the user knows to run downsample_demo_resolution.
        if not self._warned_missing_npy:
            print(f"[v2r] note: no pre-extracted .npy at "
                  f"{self._npy_path_for_mp4(mp4)} -- falling back to decord. "
                  f"Run scripts/downsample_demo_resolution.py to speed this up.",
                  flush=True)
            self._warned_missing_npy = True
        from decord import VideoReader, cpu
        vr = VideoReader(mp4, ctx=cpu(0))
        total = len(vr)
        if total <= 0:
            raise ValueError(f"Empty prompt video {mp4}")
        idxs = sample_prompt_indices(total, self.num_prompt_frames, rng=rng)
        frames = vr.get_batch(idxs).asnumpy()
        out = np.empty((self.num_prompt_frames, self.image_size, self.image_size, 3),
                       dtype=np.float32)
        for i, frame in enumerate(frames):
            out[i] = _resize(frame, self.image_size)
        return out, mp4

    # ------------------------------------------------------------------
    def __getitem__(self, index: int) -> dict:
        # Local RNG per __getitem__ so DataLoader worker shuffling is reproducible.
        rng = np.random.default_rng(int(self.rng.integers(0, 2**31 - 1)))
        demo_idx, t = self.items[index]
        demo = self.demos[demo_idx]
        pkl = self._load_pkl(demo, t)

        # Action targets: first ``chunk_size`` steps of the stored future-action
        # chunk, binned per ``codec``.
        future = np.asarray(pkl["actions"], dtype=np.float32)[:self.chunk_size]
        if future.shape[0] < self.chunk_size:
            # Pad by repeating the last action (covers end-of-trajectory).
            pad = np.repeat(future[-1:], self.chunk_size - future.shape[0], axis=0)
            future = np.concatenate([future, pad], axis=0)
        action_bins = self.codec.encode(future)  # (chunk, action_dim)

        state_video = self._state_video(demo, t)
        prompt_video, mp4_path = self._human_prompt_video(
            demo.bucket, demo.task, rng)
        item = {
            "task": demo.task,
            "prompt_video": prompt_video.astype(np.float32),
            "state_video": state_video.astype(np.float32),
            "action_bins": action_bins.astype(np.int64),
            "prompt_video_path": mp4_path,
            "text_embedding": self._text_embeddings[demo.task].astype(np.float32),
        }
        if self.with_robot_video:
            item["robot_full_video"] = self._robot_full_video(demo, rng).astype(np.float32)
        return item
