"""Vid2Robot eval runner — video-conditioned policy, 256-bin action head.

Vid2Robot (Jain et al. 2024) conditions the policy on a 16-frame prompt video
of the same task; the robot side sees an 8-frame history of ``base_camera``
ending at the current step. The prompt encoder is large (ViT-B/16 + Perceiver
Resampler) but the prompt clip is fixed for a rollout, so ``load_demo_static``
runs the prompt branch **once** per demo and caches ``z_prompt``; per-step
inference only runs the state encoder + fusion + decoder.

The policy decodes a ``chunk_size``-step action chunk (paper: 4) per inference;
the runner respects ``eval_cfg.chunk_size`` and executes that many actions
before the next forward pass (open-loop chunking like Pi0.5; set
``chunk_size=1`` in the eval config to roll fully closed-loop). The 8-D action
matches the LfO benchmark's ``pd_joint_pos`` controller (7 arm joints +
gripper close in [-1, 1]); the per-dim 256-bin head is decoded to the env's
native range via the ``ActionCodec`` saved in the checkpoint.

Runs in the ``vid2robot`` conda env (PyTorch).
"""

from __future__ import annotations

import logging
import sys
from collections import deque
from pathlib import Path

import numpy as np

logger = logging.getLogger("evaluation.vid2robot")

_REPO = Path(__file__).resolve().parents[2]
for _p in (str(_REPO / "src"),):
    if _p not in sys.path:
        sys.path.insert(0, _p)

# numpy 2.x -> 1.x unpickle shim (the Stage-0a pkls and codec dict may have
# been written under numpy 2.x). See src/np2_pickle_compat.py.
import np2_pickle_compat  # noqa: F401,E402

from lfo_data import HUMAN_VIDEO_SUBDIR, load_video_frames  # noqa: E402
from evaluation.utils import _success_from_info, load_train_config  # noqa: E402

# ``_extract_rgb_uint8_hwc`` lives in ``demonstration_vla_suite.shared.maniskill_io``,
# which pulls in ``h5py``/``mani_skill`` at module load. Those aren't part of
# the slim ``vid2robot`` env, so we defer the import to rollout time — the
# runner can be built and the prompt video encoded without them.
_extract_rgb_uint8_hwc = None  # populated lazily by ``_load_maniskill_helpers``


def _load_maniskill_helpers():
    global _extract_rgb_uint8_hwc
    if _extract_rgb_uint8_hwc is None:
        from demonstration_vla_suite.shared.maniskill_io import (
            _extract_rgb_uint8_hwc as _impl,
        )
        _extract_rgb_uint8_hwc = _impl
    return _extract_rgb_uint8_hwc

#: Demo-video distraction bucket selected by the eval cell.
_DISTRACT_DIRS = {True: "with_distraction", False: "without_distraction"}


class Vid2RobotRunner:
    name = "vid2robot"

    def __init__(self, checkpoint_path: str | Path, train_cfg: dict):
        import torch

        from vid2robot.config import Vid2RobotConfig
        from vid2robot.data.action_codec import ActionCodec
        from vid2robot.model import Vid2Robot

        self._torch = torch
        self.checkpoint_path = Path(checkpoint_path)
        # ``default_output_dir`` follows the runner-contract convention: the
        # output directory sits beside the checkpoint file (or under its
        # parent if a single .pt was passed).
        out_parent = self.checkpoint_path if self.checkpoint_path.is_dir() else self.checkpoint_path.parent
        self.default_output_dir = out_parent / "eval"
        self.data_dir = train_cfg["data_root"]
        self.tasks = list(train_cfg["tasks"])
        self.prompt_camera = train_cfg.get("prompt_camera", "front")
        self.robot_cameras = [train_cfg.get("robot_camera", "base_camera")]
        self.robot_camera = self.robot_cameras[0]

        # Build the model from the saved config snapshot so the wrapper agrees
        # with how the checkpoint was trained.
        self._cfg = Vid2RobotConfig(
            action_dim=int(train_cfg.get("action_dim", 8)),
            chunk_size=int(train_cfg.get("chunk_size", 4)),
            num_bins=int(train_cfg.get("num_bins", 256)),
            num_prompt_frames=int(train_cfg.get("num_prompt_frames", 16)),
            num_state_frames=int(train_cfg.get("num_state_frames", 8)),
            image_size=int(train_cfg.get("image_size", 224)),
        )
        self.image_size = self._cfg.image_size
        self.num_prompt_frames = self._cfg.num_prompt_frames
        self.num_state_frames = self._cfg.num_state_frames
        # The runner's ``action_horizon`` is the default chunk_size if the eval
        # config doesn't override (utils.run_eval_protocol contract).
        self.action_horizon = self._cfg.chunk_size

        self._device = torch.device("cuda" if torch.cuda.is_available() else "cpu")
        self._dtype = torch.float16 if self._device.type == "cuda" else torch.float32

        ckpt = torch.load(str(self.checkpoint_path), map_location=self._device,
                          weights_only=False)
        model = Vid2Robot(self._cfg).to(self._device).eval()
        model.load_state_dict(ckpt["model"])
        # Save the model on the runner; we'll keep it on CPU/GPU at fp16 to
        # match training's autocast precision (numerically safe at inference).
        self._model = model
        if "codec" in ckpt:
            self._codec = ActionCodec.from_dict(ckpt["codec"])
        else:
            # Older checkpoints may have written a sidecar action_codec.json.
            codec_json = out_parent / "action_codec.json"
            if codec_json.is_file():
                import json
                self._codec = ActionCodec.from_dict(json.loads(codec_json.read_text()))
            else:
                raise FileNotFoundError(
                    f"Vid2Robot checkpoint at {self.checkpoint_path} does not "
                    f"contain an action codec; neither does {codec_json}. "
                    "Re-train so the codec is persisted, or supply one manually.")
        if self._codec.action_dim != self._cfg.action_dim:
            raise ValueError(
                f"Checkpoint codec action_dim={self._codec.action_dim} disagrees "
                f"with model action_dim={self._cfg.action_dim}.")
        logger.info("Vid2Robot restored from %s (step=%s)",
                    self.checkpoint_path, ckpt.get("step", "?"))

    # -- demo conditioning (encode the demonstration video once) ------------

    def load_demo_static(self, demo_distract: bool, task: str, demo_id: int):
        """Load the human prompt video and run it through the prompt encoder.

        Returns a dict with the cached ``z_prompt`` on the model's device — the
        rollout reuses it on every step. The video is sampled with the paper's
        16-frame "first + last + sorted-middle" sampler but with ``rng=None``
        (uniformly spaced) so eval is deterministic per (task, demo_id).
        """
        torch = self._torch
        mp4 = (Path(self.data_dir) / HUMAN_VIDEO_SUBDIR
               / _DISTRACT_DIRS[bool(demo_distract)] / task
               / f"{int(demo_id):04d}" / f"{self.prompt_camera}.mp4")
        if not mp4.is_file():
            raise FileNotFoundError(f"Vid2Robot prompt video not found: {mp4}")
        # ``load_video_frames`` with rng=None returns 16 uniformly-spaced frames,
        # already resized to image_size and normalized to [0, 1] float32.
        frames = load_video_frames(str(mp4), self.num_prompt_frames,
                                   self.image_size, rng=None)  # (16, S, S, 3) float32
        prompt_video = torch.from_numpy(frames).unsqueeze(0).to(self._device)
        z_prompt = self._model.encode_prompt(prompt_video.to(self._dtype))
        return {"z_prompt": z_prompt, "mp4_path": str(mp4)}

    # -- per-step inference -------------------------------------------------

    @staticmethod
    def _resize_frame(rgb_uint8: np.ndarray, image_size: int) -> np.ndarray:
        """(H, W, 3) uint8 -> (S, S, 3) float32 in [0, 1].

        Uses PIL bilinear to match what the dataset uses at training time
        (``src/vid2robot/data/dataset.py:_resize``).
        """
        from PIL import Image
        img = Image.fromarray(rgb_uint8).resize((image_size, image_size), Image.BILINEAR)
        return np.asarray(img, dtype=np.float32) / 255.0

    def _build_state_video(self, history: deque) -> "torch.Tensor":
        """Stack the 8-frame ``base_camera`` history into a model-ready tensor."""
        torch = self._torch
        # ``history`` already contains up to ``num_state_frames`` float32 frames.
        # Left-pad by repeating the oldest frame so the model sees a fixed
        # length (matches Vid2RobotDataset._state_video / sample_state_indices).
        if len(history) < self.num_state_frames:
            pad = [history[0]] * (self.num_state_frames - len(history))
            frames = pad + list(history)
        else:
            frames = list(history)
        arr = np.stack(frames, axis=0)  # (T_s, S, S, 3) float32 in [0, 1]
        return torch.from_numpy(arr).unsqueeze(0).to(self._device).to(self._dtype)

    def _decode_chunk(self, action_logits) -> np.ndarray:
        """``(1, chunk, action_dim, num_bins)`` -> ``(chunk, action_dim)`` env actions."""
        bins = action_logits.argmax(dim=-1)  # (1, chunk, action_dim)
        bins = bins[0].detach().cpu().numpy()  # (chunk, action_dim)
        return self._codec.decode(bins).astype(np.float32)

    # -- rollout ------------------------------------------------------------

    def rollout(self, env, obs, demo_static: dict, max_rollout_steps: int,
                chunk_size: int, video_frames) -> bool:
        torch = self._torch
        z_prompt = demo_static["z_prompt"]

        # Seed the history buffer from the first observation.
        extract_rgb = _load_maniskill_helpers()
        rgb0 = extract_rgb(obs, self.robot_camera)
        history: deque[np.ndarray] = deque(maxlen=self.num_state_frames)
        history.append(self._resize_frame(rgb0, self.image_size))
        if video_frames is not None:
            video_frames.append(rgb0.copy())

        action_queue: deque[np.ndarray] = deque()
        info: dict = {}
        for _step in range(max_rollout_steps):
            if not action_queue:
                state_video = self._build_state_video(history)
                logits = self._model.predict_action_logits(z_prompt, state_video)
                chunk = self._decode_chunk(logits)  # (chunk_full, action_dim)
                for a in chunk[:chunk_size]:
                    action_queue.append(np.asarray(a, dtype=np.float32))
            action = action_queue.popleft()
            obs, _r, terminated, truncated, info = env.step(
                torch.from_numpy(action).float().unsqueeze(0))
            rgb = extract_rgb(obs, self.robot_camera)
            history.append(self._resize_frame(rgb, self.image_size))
            if video_frames is not None:
                video_frames.append(rgb.copy())
            if _success_from_info(info):
                return True
            if terminated or truncated:
                break
        return _success_from_info(info)


def build_runner(eval_cfg) -> Vid2RobotRunner:
    """Factory used by ``scripts/evaluation.py``.

    Reads either ``checkpoint_path`` (a single ``.pt``) or ``checkpoint_dir``
    (a directory holding ``ckpt.pt``, or the ``vid2robot_step*.pt`` of a live
    training run) from the eval config, and the training context from the
    sibling ``train_config.json``.
    """
    raw = eval_cfg.raw
    ckpt_path = raw.get("checkpoint_path")
    ckpt_dir = raw.get("checkpoint_dir")
    if ckpt_path is None and ckpt_dir is None:
        raise ValueError("vid2robot eval config must set 'checkpoint_path' or 'checkpoint_dir'.")
    if ckpt_path is None:
        released = Path(ckpt_dir) / "ckpt.pt"
        if released.is_file():
            ckpt_path = str(released)
        else:
            cands = sorted(Path(ckpt_dir).glob("vid2robot_step*.pt"),
                           key=lambda p: int(p.stem.split("step")[-1]))
            if not cands:
                raise FileNotFoundError(
                    f"No ckpt.pt or vid2robot_step*.pt under {ckpt_dir} — set "
                    "'checkpoint_path' to a specific .pt, or point "
                    "'checkpoint_dir' at the train output dir.")
            ckpt_path = str(cands[-1])
    train_cfg = load_train_config(Path(ckpt_path).parent)
    return Vid2RobotRunner(ckpt_path, train_cfg)
