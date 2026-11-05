"""Unified LfO VLA dataset: per-sample pkls (pre-resized robot RGB + state +
action chunk) joined with bundled per-(demo, camera) demo feature files at
__getitem__ time.

Layout under ``sample_data_root``::

    postprocessed_robot_trajectories/
        data/<idx>.pkl              # one (task, demo, timestep) sample per file,
                                    # written by scripts/process_training_robot_trajectories.py
                                    # with --pi05-samples. Each pkl is:
                                    #   {
                                    #     "task": str,
                                    #     "robot_h5": str,
                                    #     "demo_id": str,
                                    #     "robot_time_index": int,
                                    #     "actions": (action_horizon, A) float32,
                                    #     "qpos":    (9,) float32,
                                    #     "state":   (8,) float32,
                                    #     "prompt":  str,
                                    #     "robot_cameras": [str, ...],
                                    #     "observation/<cam>": (H, W, 3) uint8,
                                    #   }
        meta/
            stats.json              # total_samples, action_dim, action_horizon, distract, etc.
            index.jsonl             # debug: idx -> (task, demo_id, t)

    video_demonstration_features/<distract>/<task>/<demo_id>/<camera>.npy
        ↑ written by scripts/compute_vision_features_for_demo.py. Per
        (demo, camera) the bundled file holds a dict of stacked arrays:
            {
              "image_emb_8x8": (T, 64, 2048),  "pos_emb_8x8": (T, 64, 768),
              "image_emb_4x4": (T, 16, 2048),  "pos_emb_4x4": (T, 16, 768),
              "image_emb_2x2": (T,  4, 2048),  "pos_emb_2x2": (T,  4, 768),
              "state_emb":     (T, 8),         "timesteps":   (T,) int32
            }

Returned samples include fields for both training pipelines:

- **π₀.₅** / OpenPI-style: ``observation/image`` (HWC uint8), ``observation.state``,
  ``actions``, ``prompt``.
- **LeRobot-style**: ``observation.images.main`` (CHW float in [0,1]),
  ``task`` (same string as ``prompt``), ``action``.
- **Shared demo tokens**: ``demo_image_emb_<grid>``, ``demo_pos_emb_<grid>``,
  ``demo_state_emb``, ``demo_timesteps``.

``__getitem__`` cost: one pickle load (~0.3 ms) + one ``np.load`` per demo
camera. Both steps run in O(1) per sample regardless of how big the source
trajectory was — the old code path opened the source h5 and re-read every
RGB frame on every cache miss (~10 s per miss at bs=32).
"""

from __future__ import annotations

import json
import logging
import pickle
from pathlib import Path

import numpy as np

# Stage-0a pkls are written under numpy 2.x (uniskill env); this process may
# run numpy 1.x (pi05 env) — install the unpickle compat shim before any
# pickle.load. See src/np2_pickle_compat.py.
import os as _os
import sys as _sys
_sys.path.insert(
    0, _os.path.dirname(_os.path.dirname(_os.path.dirname(_os.path.abspath(__file__)))))
import np2_pickle_compat  # noqa: F401,E402

logger = logging.getLogger(__name__)


def _to_numpy_leaf(x: object) -> np.ndarray:
    """Convert array-like (including JAX) to ``numpy.ndarray`` without importing JAX here."""
    return np.asarray(x)


def _load_pickle(path: Path) -> dict:
    with open(path, "rb") as f:
        return pickle.load(f)


def _infer_entry_bucket(entry: dict) -> str:
    """Fallback bucket label for legacy index entries lacking a "bucket" field."""
    return entry.get("bucket") or "without_distraction"


def _load_dict_npy(path: Path) -> dict:
    raw = np.load(path, allow_pickle=True)
    return raw.item() if raw.shape == () else raw


class LfOVLADataset:
    """Fast dataset over pre-baked per-sample pkls and bundled demo features.

    Args:
        sample_data_root: Directory containing ``postprocessed_robot_trajectories/`` and
            ``video_demonstration_features/``.
        action_horizon: Length of each action chunk. Must be ``<= `` the
            ``action_horizon`` that ``process_training_robot_trajectories.py`` was
            invoked with when writing the pkls.
        prompt: Language instruction. If not None, overrides the prompt in
            each loaded pkl (kept for compatibility with the train scripts).
        demo_cameras: Camera names whose bundled files to read for each
            sample's demonstration view, e.g. ``["front"]``.
        robot_cameras: Robot-side cameras that must be present in each pkl
            under the ``observation/<cam>`` key.
        max_demo_index: Upper bound (inclusive) on numeric demo folder id.
            Demos whose folder name parses to ``> max_demo_index`` are excluded.
            Default ``99`` for ``0000``–``0099``. Use e.g. ``79`` for Experiment 3
            (train on demos 0000-0079, eval on the held-out 0080-0099 subset).
        include_video_types: Distraction buckets under
            ``video_demonstration_features/`` to draw candidate human demos
            from — ``["without_distraction"]`` for Exp 1/3, both buckets for
            Exp 2/4. Candidates for a task are pooled across the listed types.
        seed: RNG seed for choosing a random eligible video demo dir per
            sample (so the same robot sample can pair with a different demo
            across epochs).
        max_action_dim: If set, pad action chunks to this width.
        demo_emb_spatial: Patch-grid tag for the cached embeddings, e.g.
            ``\"8x8\"`` (LeRobot-style) or ``\"4x4\"`` (Pi0.5 token-per-image=16).
        samples_subdir: Subdir under ``sample_data_root`` holding the pkls
            (default ``postprocessed_robot_trajectories``).
    """

    def __init__(
        self,
        sample_data_root: str | Path,
        action_horizon: int,
        prompt: str,
        *,
        demo_cameras: list[str] | tuple[str, ...] | str = ("front",),
        robot_cameras: list[str] | str = ("base_camera", "hand_camera"),
        max_demo_index: int = 99,
        include_video_types: list[str] | tuple[str, ...] | str = ("without_distraction",),
        seed: int = 0,
        max_action_dim: int | None = None,
        demo_emb_spatial: str = "8x8",
        samples_subdir: str = "postprocessed_robot_trajectories",
        keyframe_annotation_dir: str | Path | None = None,
    ) -> None:
        if isinstance(demo_cameras, str):
            demo_cameras = [demo_cameras]
        self.demo_cameras = list(demo_cameras)
        if not self.demo_cameras:
            raise ValueError("demo_cameras must contain at least one camera name.")

        if isinstance(robot_cameras, str):
            robot_cameras = [robot_cameras]
        self.robot_cameras = list(robot_cameras)
        if not self.robot_cameras:
            raise ValueError("robot_cameras must contain at least one camera name.")

        if isinstance(include_video_types, str):
            include_video_types = [include_video_types]
        self.include_video_types = list(include_video_types)
        if not self.include_video_types:
            raise ValueError("include_video_types must contain at least one entry.")

        self.sample_data_root = Path(sample_data_root).expanduser().resolve()
        self.action_horizon = int(action_horizon)
        self.prompt = prompt
        self.demo_emb_spatial = demo_emb_spatial
        self._demo_image_key = f"image_emb_{self.demo_emb_spatial}"
        self._demo_pos_key = f"pos_emb_{self.demo_emb_spatial}"
        self._seed = int(seed)

        self.samples_root = self.sample_data_root / samples_subdir
        self.data_root = self.samples_root / "data"
        self.meta_root = self.samples_root / "meta"
        stats_path = self.meta_root / "stats.json"
        if not stats_path.is_file():
            raise FileNotFoundError(
                f"Expected postprocessed_robot_trajectories at {self.samples_root} "
                f"with meta/stats.json. Run "
                f"`scripts/process_training_robot_trajectories.py` first."
            )

        stats = json.loads(stats_path.read_text())
        # Materialize `samples` as a list of (task, demo_id, t) triplets so
        # callers that scan ``dataset.samples`` for task / demo coverage keep
        # working without re-deriving from per-pkl reads. ``_pkl_indices`` maps
        # dataset position → on-disk pkl idx after the bucket filter. The
        # parallel ``_buckets`` array tracks each sample's distraction bucket
        # so ``__getitem__`` can route same-bucket-same-task sampling without
        # re-reading the pkl.
        index_path = self.meta_root / "index.jsonl"
        if not index_path.is_file():
            raise FileNotFoundError(f"index.jsonl missing at {index_path}")
        allowed_buckets = set(self.include_video_types)
        self.samples: list[tuple[str, str, int]] = []
        self._pkl_indices: list[int] = []
        self._buckets: list[str] = []
        with open(index_path, "r") as f:
            for line in f:
                line = line.strip()
                if not line:
                    continue
                entry = json.loads(line)
                bucket = _infer_entry_bucket(entry)
                if bucket not in allowed_buckets:
                    continue
                self.samples.append((entry["task"], entry["demo_id"], int(entry["t"])))
                self._pkl_indices.append(int(entry["idx"]))
                self._buckets.append(bucket)
        self._total_samples = len(self.samples)
        if self._total_samples == 0:
            raise RuntimeError(
                f"No index entries for buckets {self.include_video_types} at {index_path}"
            )
        baked_horizon = int(stats.get("action_horizon", self.action_horizon))
        if self.action_horizon > baked_horizon:
            raise ValueError(
                f"action_horizon={self.action_horizon} requested but pkls were "
                f"baked with action_horizon={baked_horizon}. Re-run preprocessing "
                f"with --pi05-action-horizon {self.action_horizon}+."
            )
        self.action_dim = int(stats.get("action_dim", 0))
        if max_action_dim is not None:
            self.action_dim = max(self.action_dim, int(max_action_dim))

        # Per-demo keyframe annotations (front camera only) — used only when
        # the demonstration YAML asks for ``demonstration_sampling_type ==
        # "keyframe_based_sampling"``. We cache them once into a small in-RAM
        # dict so the hot ``__getitem__`` path doesn't pay per-sample JSON
        # load cost. Layout on disk:
        #
        #   <keyframe_annotation_dir>/<bucket>/<task>/<demo_id>.json
        #
        # Missing demos resolve to an empty array and the sampler falls back
        # to uniform sampling at train time, so partial keyframe coverage
        # remains usable.
        self._keyframes_by_demo: dict[tuple[str, str, str], np.ndarray] = {}
        self.keyframe_annotation_dir: Path | None = (
            Path(keyframe_annotation_dir).expanduser().resolve()
            if keyframe_annotation_dir is not None else None
        )
        if self.keyframe_annotation_dir is not None and self.keyframe_annotation_dir.is_dir():
            for json_path in self.keyframe_annotation_dir.rglob("*.json"):
                try:
                    rec = json.loads(json_path.read_text())
                except json.JSONDecodeError:
                    logger.warning("skipping malformed keyframe JSON: %s", json_path)
                    continue
                b = rec.get("bucket")
                t = rec.get("task")
                d = rec.get("demo_id")
                kf = rec.get("keyframe_indices") or []
                if not (b and d):
                    continue
                self._keyframes_by_demo[(b, t, str(d))] = np.asarray(kf, dtype=np.int64)
            logger.info(
                "LfOVLADataset: loaded %d keyframe annotation(s) from %s",
                len(self._keyframes_by_demo), self.keyframe_annotation_dir,
            )

        # Candidate human demos per (bucket, task) — scanned directly from
        # the feature tree (no manifest). Layout::
        #
        #   video_demonstration_features/
        #       without_distraction/<task>/<demo>/<cam>.npy
        #       with_distraction/<task>/<demo>/<cam>.npy
        #
        # ``include_video_types`` is one bucket per training condition.
        # Sampling stays inside the same bucket.
        features_root = self.sample_data_root / "video_demonstration_features"
        self._video_dirs_by_bucket_task: dict[str, dict[str, list[Path]]] = {}
        for video_type in self.include_video_types:
            type_root = features_root / video_type
            if not type_root.is_dir():
                continue
            bucket_map = self._video_dirs_by_bucket_task.setdefault(video_type, {})
            for task_dir in sorted(p for p in type_root.iterdir() if p.is_dir()):
                for demo_dir in sorted(p for p in task_dir.iterdir() if p.is_dir()):
                    try:
                        demo_id_int = int(demo_dir.name, 10)
                    except ValueError:
                        continue
                    if demo_id_int > int(max_demo_index):
                        continue
                    if all((demo_dir / f"{cam}.npy").is_file() for cam in self.demo_cameras):
                        bucket_map.setdefault(task_dir.name, []).append(demo_dir)

        if not self._video_dirs_by_bucket_task:
            raise RuntimeError(
                f"No human-demo feature dirs with bundled <cam>.npy for cameras "
                f"{self.demo_cameras} under {features_root} for video types "
                f"{self.include_video_types}. Run "
                f"`scripts/compute_vision_features_for_demo.py "
                f"--cameras {' '.join(self.demo_cameras)}` first."
            )

        # Per-(demo_dir, cam) cache for already-decoded bundles. ``np.load``
        # of a pickled dict still pays an unpickle cost on first access, so
        # cache once you've decoded.
        self._bundle_cache: dict[Path, dict] = {}

        if not self.data_root.is_dir():
            raise FileNotFoundError(f"Sample pkl dir missing: {self.data_root}")

        n_pairs = sum(len(by_task) for by_task in self._video_dirs_by_bucket_task.values())
        logger.info(
            "LfOVLADataset: %d samples (buckets=%s), action_dim=%d, "
            "action_horizon=%d, bucket_task_pairs_with_demos=%d, demo_emb_spatial=%s",
            self._total_samples,
            self.include_video_types,
            self.action_dim,
            self.action_horizon,
            n_pairs,
            self.demo_emb_spatial,
        )

    def __len__(self) -> int:
        return self._total_samples

    def _pick_video_demo_dir(self, bucket: str, task: str,
                             sample_index: int) -> Path:
        choices = self._video_dirs_by_bucket_task.get(bucket, {}).get(task)
        if not choices:
            raise KeyError(
                f"No eligible video demo dirs for bucket={bucket!r}, task={task!r} "
                f"with demo_cameras={self.demo_cameras}. Bundles may be missing."
            )
        seq = np.random.SeedSequence([self._seed & 0xFFFFFFFF, sample_index & 0xFFFFFFFF])
        rng = np.random.default_rng(seq)
        return choices[int(rng.integers(0, len(choices)))]

    def _load_bundle(self, demo_dir: Path, cam: str) -> dict:
        key = demo_dir / f"{cam}.npy"
        cached = self._bundle_cache.get(key)
        if cached is not None:
            return cached
        data = _load_dict_npy(key)
        if len(self._bundle_cache) > 4096:
            self._bundle_cache.pop(next(iter(self._bundle_cache)))
        self._bundle_cache[key] = data
        return data

    def _pad_action_chunk(self, act: np.ndarray) -> np.ndarray:
        if act.shape[-1] == self.action_dim:
            return act.astype(np.float32, copy=False)
        padded = np.zeros((act.shape[0], self.action_dim), dtype=np.float32)
        padded[:, : act.shape[-1]] = act.astype(np.float32, copy=False)
        return padded

    def __getitem__(self, index: int) -> dict:
        pkl_idx = self._pkl_indices[index]
        sample = _load_pickle(self.data_root / f"{pkl_idx}.pkl")
        task = sample["task"]
        bucket = self._buckets[index]
        demo_id = self.samples[index][1]

        actions = sample["actions"][: self.action_horizon]
        act_chunk = self._pad_action_chunk(actions)

        state = sample["state"].astype(np.float32, copy=False)
        prompt = self.prompt if self.prompt is not None else sample["prompt"]

        demo_dir = self._pick_video_demo_dir(bucket, task, index)
        # Read one bundle per requested demo camera and stack views.
        per_view_img: list[np.ndarray] = []
        per_view_pos: list[np.ndarray] = []
        state_emb_arr: np.ndarray | None = None
        timesteps: np.ndarray | None = None
        for cam in self.demo_cameras:
            bundle = self._load_bundle(demo_dir, cam)
            per_view_img.append(_to_numpy_leaf(bundle[self._demo_image_key]))  # (T, P, D)
            per_view_pos.append(_to_numpy_leaf(bundle[self._demo_pos_key]))    # (T, P, Dpos)
            if state_emb_arr is None:  # state_emb and timesteps are camera-independent
                state_emb_arr = _to_numpy_leaf(bundle["state_emb"])
                timesteps = _to_numpy_leaf(bundle["timesteps"]).astype(np.int64, copy=False)
        # Stack views into (T, num_views, P, D).
        demo_image = np.stack(per_view_img, axis=1).astype(np.float32, copy=False)
        demo_pos = np.stack(per_view_pos, axis=1).astype(np.float32, copy=False)
        demo_state = state_emb_arr.astype(np.float32, copy=False)
        demo_ts = timesteps if timesteps is not None else np.arange(demo_image.shape[0], dtype=np.int64)

        out: dict = {
            "observation/state": state,
            "observation.state": state,
            "actions": act_chunk,
            "action": act_chunk,
            "prompt": prompt,
            "task": prompt,
            "demo_video_dir": str(demo_dir),
            "demo_video_id": demo_dir.name,
            "demo_task": task,
            f"demo_image_emb_{self.demo_emb_spatial}": demo_image,
            f"demo_pos_emb_{self.demo_emb_spatial}": demo_pos,
            "demo_state_emb": demo_state,
            "demo_timesteps": demo_ts,
            "robot_h5": sample.get("robot_h5", ""),
            "robot_time_index": np.int64(sample.get("robot_time_index", 0)),
        }
        # Inject this demo's keyframe annotation when the cache holds one. The
        # task key in the cache mirrors the JSON's snake-case task name.
        # Absent entry → ``demo_key_frame_indices`` is omitted, and
        # ``get_demonstration_tokens`` will raise loudly if the YAML asked for
        # ``keyframe_based_sampling``. No silent fallback to uniform.
        kf = self._keyframes_by_demo.get((bucket, task, demo_id))
        if kf is not None:
            out["demo_key_frame_indices"] = kf

        # Per-camera robot image keys. Camera index → role mapping:
        #   0 → main  (legacy ``observation/image`` + ``observation.images.main``)
        #   1 → wrist (legacy ``observation/wrist_image`` + ``observation.images.wrist``)
        # All cameras also get ``observation.images.<cam_name>`` so callers can
        # pick by name.
        for cam_idx, cam in enumerate(self.robot_cameras):
            rgb_t = sample[f"observation/{cam}"]
            chw_t = (np.transpose(rgb_t, (2, 0, 1)).astype(np.float32) / 255.0)
            out[f"observation.images.{cam}"] = chw_t
            out[f"observation/{cam}"] = rgb_t
            if cam_idx == 0:
                out["observation/image"] = rgb_t
                out["observation.images.main"] = chw_t
            elif cam_idx == 1:
                out["observation/wrist_image"] = rgb_t
                out["observation.images.wrist"] = chw_t

        return out

    def compute_stats(self) -> dict:
        """LeRobot-style mean/std for ``observation.state`` and ``action`` (robot trajectories only)."""
        state_sum = state_sumsq = None
        state_count = 0
        action_sum = action_sumsq = None
        action_count = 0
        for i in range(self._total_samples):
            with open(self.data_root / f"{self._pkl_indices[i]}.pkl", "rb") as f:
                d = pickle.load(f)
            s = d["state"].astype(np.float64)
            act = self._pad_action_chunk(d["actions"][: self.action_horizon]).astype(np.float64)
            if state_sum is None:
                state_sum = np.zeros_like(s)
                state_sumsq = np.zeros_like(s)
                action_sum = np.zeros((self.action_dim,), dtype=np.float64)
                action_sumsq = np.zeros((self.action_dim,), dtype=np.float64)
            state_sum += s
            state_sumsq += s * s
            state_count += 1
            action_sum += act.sum(axis=0)
            action_sumsq += np.square(act).sum(axis=0)
            action_count += act.shape[0]
        state_mean = state_sum / state_count
        state_std = np.sqrt(np.maximum(state_sumsq / state_count - state_mean * state_mean, 0.0))
        action_mean = action_sum / action_count
        action_std = np.sqrt(np.maximum(action_sumsq / action_count - action_mean * action_mean, 0.0))
        return {
            "state_mean": state_mean.astype(np.float32),
            "state_std": state_std.astype(np.float32),
            "action_mean": action_mean.astype(np.float32),
            "action_std": action_std.astype(np.float32),
        }


# Back-compat alias — older train scripts import ``LfODataset``.
LfODataset = LfOVLADataset
