"""Demonstration Pi0.5 (LfOPi0) eval runner — loads the checkpoint directly and rolls it out.

This is the *demonstration-conditioned* Pi0.5 (conditioned on demo-video SigLIP
tokens). The plain Pi0.5 (no video conditioning) has its own runner.

Bypasses ``DemoVLAPolicy`` for the model *call* (builds ``LfOPi0`` via ``cfg.load(restore_params)``
and JITs ``model.sample_actions``), but reuses the **same OpenPI transform stack** that
``scripts/train_vid_cond_pi05.build_train_config`` builds, so the live observation is converted to
model input exactly the way training converts a dataset sample:

    DemoVLAInputs(model_type=PI05, robot_cameras=...)   # one image slot per robot camera
      → DeltaActions(make_bool_mask(7, -1))             # no-op here (no 'actions' in a live obs)
      → ResizeImages(224, 224)                          # no-op: images are pre-resized to 224²
      → TokenizePrompt(PaligemmaTokenizer)
      → PadStatesAndActions(action_dim)

The action chunk the model predicts is run back through training's output transform —
``AbsoluteActions(make_bool_mask(7, -1))`` then ``DemoVLAOutputs()`` — which inverts the
delta-arm / absolute-gripper repacking and slices to the 8-D action space, before stepping the env.

Demo conditioning is the precomputed SigLIP tokens (``static_*`` from ``get_demonstration_tokens``,
same call as training) — not re-encoded mp4 frames.
"""

from __future__ import annotations

import json
import logging
import math
from collections import deque
from pathlib import Path

import numpy as np
import torch

from demonstration_vla_suite.shared.maniskill_io import extract_obs_for_pi05, _extract_rgb_uint8_hwc
from evaluation.utils import (
    DEMO_DISTRACTION_DIRS, _success_from_info, demo_feature_dir, load_train_config,
)

#: Per-demo BPP/VLM keyframe annotations live under this subdir of ``data_dir``;
#: see scripts/extract_keyframes.py. Layout::
#:
#:   <data_dir>/bpp_keyframe_annotations/<bucket>/<task>/<demo_id>.json
_KEYFRAME_ANNOTATION_SUBDIR = "bpp_keyframe_annotations"

logger = logging.getLogger("evaluation.demo_pi05")


class DemoPi05Runner:
    name = "demo_pi05"

    def __init__(self, checkpoint_dir: str | Path, train_cfg: dict):
        import jax
        import openpi.models.model as _model
        import openpi.transforms as transforms
        import openpi.shared.nnx_utils as nnx_utils
        from openpi.models.tokenizer import PaligemmaTokenizer
        from demonstration_vla_suite.models.lfo_pi0 import LfOPi0Config
        from demonstration_vla_suite.models.config.utils import get_demonstration_config
        from demonstration_vla_suite.models.lfo_observation import HistAugObservation
        from demonstration_vla_suite.policies.transforms import DemoVLAInputs, DemoVLAOutputs

        self._jax = jax
        self._HistAugObservation = HistAugObservation

        self.checkpoint_dir = Path(checkpoint_dir)
        self.default_output_dir = self.checkpoint_dir / "eval"
        self.data_dir = train_cfg["data_dir"]
        self.tasks = [t for t in (train_cfg.get("tasks") or [train_cfg.get("task")]) if t]
        self.demonstration_config = train_cfg.get("demonstration_config") or "perceptual-framesamp-context.yaml"
        self.demo_cameras = train_cfg.get("demo_cameras") or (
            [train_cfg["demo_camera"]] if train_cfg.get("demo_camera") else ["front"]
        )
        self.robot_cameras = list(train_cfg.get("robot_cameras") or ["base_camera"])
        self.robot_camera = self.robot_cameras[0]  # camera used for the rendered video
        self.prompt = train_cfg.get("prompt") or "complete the task"
        use_lfo = self.demonstration_config not in (None, "none", "None")

        cfg = LfOPi0Config(
            pi05=True,
            action_horizon=20,
            action_dim=32,
            max_token_len=64,
            discrete_state_input=False,
            use_demonstration=use_lfo,
            demonstration_config=self.demonstration_config if use_lfo else None,
        )
        self.action_horizon = int(cfg.action_horizon)
        self.action_dim = int(cfg.action_dim)

        logger.info("Loading LfOPi0 from %s", self.checkpoint_dir)
        params = _model.restore_params(self.checkpoint_dir / "params", dtype=jax.numpy.bfloat16)
        model = cfg.load(params)
        self._sample_fn = nnx_utils.module_jit(model.sample_actions)

        # Same transform stack as scripts/train_vid_cond_pi05.build_train_config — keep in sync.
        _delta_mask = transforms.make_bool_mask(7, -1)
        self._input_transform = transforms.compose([
            DemoVLAInputs(model_type=_model.ModelType.PI05, robot_cameras=tuple(self.robot_cameras)),
            transforms.DeltaActions(_delta_mask),               # no-op: a live obs has no 'actions'
            transforms.ResizeImages(224, 224),                  # no-op: extract_obs_for_pi05 pre-resizes
            transforms.TokenizePrompt(PaligemmaTokenizer(max_len=cfg.max_token_len)),
            transforms.PadStatesAndActions(cfg.action_dim),
        ])
        self._output_transform = transforms.compose([
            transforms.AbsoluteActions(_delta_mask),            # invert training's DeltaActions
            DemoVLAOutputs(),                                   # slice to the 8-D action space
        ])
        logger.info("Pi0.5 ready: robot_cameras=%s action_horizon=%d action_dim=%d",
                    self.robot_cameras, self.action_horizon, self.action_dim)

        self._demo_cfg = get_demonstration_config(self.demonstration_config) if use_lfo else None
        if self._demo_cfg is not None:
            g = int(round(math.sqrt(int(self._demo_cfg.token_per_image))))
            self._demo_spatial = f"{g}x{g}"
        else:
            self._demo_spatial = "8x8"
        self._rng = jax.random.key(int(train_cfg.get("seed", 42)))

        # When the demonstration YAML asks for ``keyframe_based_sampling``, eagerly
        # load all per-demo keyframe-index JSONs so ``load_demo_static`` can look
        # them up in O(1). Keyed by ``(bucket, task, demo_id_str)`` to match
        # the training-side cache in ``LfOVLADataset`` (see its docstring).
        self._demonstration_sampling_type = (
            str(self._demo_cfg.demonstration_sampling_type) if self._demo_cfg is not None else "uniform_frame_sampling"
        )
        self._keyframes_by_demo: dict[tuple[str, str, str], np.ndarray] = {}
        if self._demonstration_sampling_type == "keyframe_based_sampling":
            kf_root = Path(self.data_dir) / _KEYFRAME_ANNOTATION_SUBDIR
            if not kf_root.is_dir():
                logger.warning(
                    "demonstration_sampling_type=keyframe_based_sampling but no annotations "
                    "found at %s; load_demo_static will not inject demo_key_frame_indices and "
                    "sample_demonstration_timesteps will raise. Generate annotations with "
                    "scripts/extract_keyframes.py first.", kf_root,
                )
            else:
                for json_path in kf_root.rglob("*.json"):
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
                    "DemoPi05Runner: loaded %d keyframe annotation(s) from %s",
                    len(self._keyframes_by_demo), kf_root,
                )

    # -- demo conditioning -------------------------------------------------

    def load_demo_static(self, demo_distract: bool, task: str, demo_id: int) -> dict:
        from demonstration_vla_suite.shared.data_utils import (
            get_demonstration_tokens, load_cached_demo_sample,
        )
        if self._demo_cfg is None:
            return {}
        feat_dir = demo_feature_dir(self.data_dir, demo_distract, task, demo_id)
        sample = load_cached_demo_sample(feat_dir, self.demo_cameras, self._demo_spatial)
        # If the demonstration YAML asks for keyframe_based_sampling, inject this
        # demo's keyframe-index array into ``sample`` so that
        # ``get_demonstration_tokens`` → ``sample_demonstration_timesteps`` picks
        # those frames instead of evenly-spaced ones. Cache layout matches the
        # training side (LfOVLADataset): keyed by (bucket, task_or_None, demo_id).
        # ``demo_id`` is stored as the zero-padded 4-digit string used in the
        # annotation JSONs and the on-disk demo directories.
        if self._demonstration_sampling_type == "keyframe_based_sampling":
            bucket = DEMO_DISTRACTION_DIRS[bool(demo_distract)]
            demo_id_str = f"{int(demo_id):04d}"
            kf = self._keyframes_by_demo.get((bucket, task, demo_id_str))
            if kf is None:
                logger.warning(
                    "no keyframe annotation for (bucket=%s, task=%s, demo_id=%s); "
                    "sample_demonstration_timesteps will raise. Generate via "
                    "scripts/extract_keyframes.py.", bucket, task, demo_id_str,
                )
            else:
                sample["demo_key_frame_indices"] = kf
        fields = get_demonstration_tokens(sample, self._demo_cfg)
        return {
            "static_image_emb": np.asarray(fields["static_image_emb"], dtype=np.float32),
            "static_pos_emb": np.asarray(fields["static_pos_emb"], dtype=np.float32),
            "static_state_emb": np.asarray(fields["static_state_emb"], dtype=np.float32),
            "static_mask": np.asarray(fields["static_mask"], dtype=bool),
        }

    # -- observation packing -----------------------------------------------

    def _build_observation(self, env_obs, demo_static: dict):
        """live ManiSkill obs (+ demo tokens) → batched ``HistAugObservation``.

        Builds the raw per-step dict the way ``LfOVLADataset`` would (``observation/<cam>``
        HWC uint8 224², ``observation/state`` 8-D, ``prompt``, ``static_*``), runs the *same*
        ``DemoVLAInputs → DeltaActions → ResizeImages → TokenizePrompt → PadStatesAndActions``
        stack as training, then adds a leading batch dim. Returns ``(observation, state_padded)``;
        ``state_padded`` is needed by the output transform to invert the delta-action repacking.
        """
        data: dict = dict(extract_obs_for_pi05(env_obs, cameras=self.robot_cameras))
        data.pop("observation/image", None)  # the per-camera keys carry the alias's content
        data["prompt"] = self.prompt
        data.update(demo_static)
        transformed = self._input_transform(data)
        jnp = self._jax.numpy

        def _batch1(v):
            if v is None:
                return None
            if isinstance(v, dict):
                return {k: _batch1(x) for k, x in v.items()}
            if isinstance(v, np.ndarray):
                return jnp.asarray(v[None, ...])
            if isinstance(v, (np.bool_, bool)):
                return jnp.asarray(np.array([v], dtype=np.bool_))
            if isinstance(v, (int, float, np.integer, np.floating)):
                return jnp.asarray(np.array([v]))
            return v

        batched = {k: _batch1(v) for k, v in transformed.items() if v is not None}
        observation = self._HistAugObservation.from_dict(batched)
        return observation, np.asarray(transformed["state"], dtype=np.float32)

    # -- rollout -----------------------------------------------------------

    def rollout(self, env, obs, demo_static: dict, max_rollout_steps: int, chunk_size: int, video_frames) -> bool:
        if video_frames is not None:
            video_frames.append(_extract_rgb_uint8_hwc(obs, self.robot_camera).copy())
        action_queue: deque[np.ndarray] = deque()
        info: dict = {}
        for _step in range(max_rollout_steps):
            if not action_queue:
                observation, state_padded = self._build_observation(obs, demo_static)
                self._rng, step_rng = self._jax.random.split(self._rng)
                raw = np.asarray(self._sample_fn(step_rng, observation)[0]).astype(np.float32)  # (H, action_dim)
                chunk = np.asarray(
                    self._output_transform({"actions": raw, "state": state_padded})["actions"],
                    dtype=np.float32,
                )  # (H, 8): arm = delta + state, gripper = absolute
                for a in chunk[:chunk_size]:
                    action_queue.append(np.asarray(a, dtype=np.float32))
            action = action_queue.popleft()
            obs, _, terminated, truncated, info = env.step(torch.from_numpy(action).float().unsqueeze(0))
            if video_frames is not None:
                video_frames.append(_extract_rgb_uint8_hwc(obs, self.robot_camera).copy())
            if _success_from_info(info):
                return True
            if terminated or truncated:
                break
        return _success_from_info(info)


def build_runner(eval_cfg) -> DemoPi05Runner:
    """Factory used by ``scripts/evaluation.py``. Reads ``checkpoint_dir`` from the eval config
    and the rest (data_dir, demonstration_config, demo_cameras, robot_cameras, prompt, tasks)
    from the ``train_config.json`` persisted beside the checkpoint."""
    ckpt = eval_cfg.raw.get("checkpoint_dir")
    if not ckpt:
        raise ValueError("demo_pi05 eval config must set 'checkpoint_dir'.")
    train_cfg = load_train_config(ckpt)
    logger.info("Loaded train config: tasks=%s, demonstration_config=%s, data_dir=%s",
                train_cfg.get("tasks"), train_cfg.get("demonstration_config"), train_cfg.get("data_dir"))
    return DemoPi05Runner(ckpt, train_cfg)
