"""lerobot Pi0.5 FFT eval wrapper — plain language-conditioned Pi0.5.

LfO eval runner for the two language-conditioned Pi0.5 baselines
(``lerobot_pi05_fft_gt`` and ``lerobot_pi05_fft_vlm``) — full fine-tunes of
``lerobot/pi05_base`` trained by ``scripts/train_vlm_pi05.py``. The
two variants differ only in the annotation source used at training time
(ground-truth vs Qwen3-VL-4B autoannotation), recorded in the checkpoint's
``train_config.json``. This wrapper handles both — variant detection is
automatic.

Conditions on a **language instruction** (the annotation of the eval demo)
and rolls the policy out in the ManiSkill env, mirroring the
``demo_pi05_runner`` runner interface so ``scripts/evaluation.py``
can dispatch to it via the ``_RUNNER_MODULES`` registry (model key
``lerobot_pi05_fft``).

Runs in the ``lerobot_pi05`` conda env (lerobot 0.5.x + ``[pi]``).
"""

from __future__ import annotations

import json
import logging
import sys
from collections import deque
from pathlib import Path

import numpy as np
import torch

logger = logging.getLogger("evaluation.lerobot_pi05_fft")

_REPO = Path(__file__).resolve().parents[2]
for _p in (str(_REPO / "src"), str(_REPO / "src" / "external_models")):
    if _p not in sys.path:
        sys.path.insert(0, _p)

# Drop the lerobot submodule from sys.path so the pip-installed lerobot
# (the lerobot_pi05 env's [pi] install) is imported instead. The submodule
# is pinned at an older commit and trips
# dataclass-ordering bugs on python 3.12 plus a cascading-dep chain
# (datasets, diffusers, …) when its `lerobot.policies` package eagerly
# imports every model. The pip-installed lerobot ships all its deps via
# the [pi] extras, so use that here.
_LEROBOT_SUBMODULE = str(_REPO / "src" / "external_models" / "lerobot" / "src")
sys.path[:] = [p for p in sys.path if p != _LEROBOT_SUBMODULE]

from demonstration_vla_suite.shared.maniskill_io import (  # noqa: E402
    _extract_rgb_uint8_hwc, extract_obs_for_pi05)
from evaluation.utils import _success_from_info, load_train_config  # noqa: E402
from lerobot.configs.policies import PreTrainedConfig  # noqa: E402
from lerobot.configs.types import FeatureType, NormalizationMode, PolicyFeature  # noqa: E402
from lerobot.policies.pi05 import PI05Policy, make_pi05_pre_post_processors  # noqa: E402
from lerobot.utils.constants import ACTION, OBS_IMAGES, OBS_STATE  # noqa: E402

# Demonstration-video distraction bucket selected by the eval cell.
_DISTRACT_DIRS = {True: "with_distraction", False: "without_distraction"}


class LeRobotPi05FFTRunner:
    name = "lerobot_pi05_fft"

    def __init__(self, checkpoint_dir: str | Path, train_cfg: dict):
        self.checkpoint_dir = Path(checkpoint_dir)
        self.default_output_dir = self.checkpoint_dir / "eval"
        self.train_cfg = train_cfg
        # ``train_cfg["device"]`` is the training-time device string (e.g.
        # ``cuda:3`` if training ran on that host's GPU 3). At eval time
        # ``scripts/evaluation.py`` has pinned CUDA_VISIBLE_DEVICES via --gpu
        # so only one device is visible to this process and it's always
        # ``cuda:0`` from torch's perspective. Ignore the recorded train device.
        self.device = "cuda:0" if torch.cuda.is_available() else "cpu"
        self.robot_cameras = list(train_cfg["robot_cameras"])
        self.action_horizon = int(train_cfg.get("chunk", 20))
        # The lerobot_pi05_fft baselines train across all LfO tasks at once
        # (no task subset in train_config). Default the eval to the full
        # 10-task set; allow an explicit ``tasks`` field in train_config or
        # the eval config to override.
        self.tasks = list(train_cfg.get("tasks") or [
            "bowl_in_plate", "close_drawer", "empty_basket", "open_lid",
            "pick_cube", "press_toaster", "push_cube", "stack_cups",
            "toys_in_basket", "toys_in_drawer",
        ])

        # Locate the weights inside checkpoint_dir. Unlike the LoRA-era
        # wrapper, which loaded the base + a PEFT adapter separately, FFT
        # saves the full model via policy.save_pretrained(<subdir>/), so we
        # load that directly. A released checkpoint is a single ``ckpt/``;
        # a live training run has one ``<step>/`` per save, newest last.
        step_dir = self.checkpoint_dir / "ckpt"
        if not step_dir.is_dir():
            steps = sorted((d for d in self.checkpoint_dir.iterdir()
                            if d.is_dir() and d.name.isdigit()), key=lambda d: int(d.name))
            if not steps:
                raise FileNotFoundError(
                    f"No ckpt/ or <step>/ subdir in {self.checkpoint_dir}")
            step_dir = steps[-1]

        stats_raw = json.loads((self.checkpoint_dir / "norm_stats.json").read_text())
        stats = {k: {s: torch.tensor(v, dtype=torch.float32) for s, v in d.items()}
                 for k, d in stats_raw.items()}

        # Rebuild the exact config train_vlm_pi05.py used. We pull
        # the architecture/discriminator from the saved step_dir (so we
        # don't re-download pi05_base over the network at eval time) and
        # then patch the data-dependent feature shapes from train_cfg.
        state_dim = int(train_cfg["state_dim"])
        action_dim = int(train_cfg["action_dim"])
        config = PreTrainedConfig.from_pretrained(step_dir)
        config.chunk_size = int(train_cfg["chunk"])
        config.n_action_steps = int(train_cfg["chunk"])
        config.device = self.device
        config.input_features = {
            f"{OBS_IMAGES}.{cam}": PolicyFeature(type=FeatureType.VISUAL, shape=(3, 224, 224))
            for cam in self.robot_cameras
        }
        config.input_features[OBS_STATE] = PolicyFeature(type=FeatureType.STATE, shape=(state_dim,))
        config.output_features = {ACTION: PolicyFeature(type=FeatureType.ACTION, shape=(action_dim,))}
        config.normalization_mapping = {
            "VISUAL": NormalizationMode.IDENTITY,
            "STATE": NormalizationMode.QUANTILES,
            "ACTION": NormalizationMode.QUANTILES,
        }
        # Load the FFT checkpoint directly — full weights, no adapter merge.
        self.policy = PI05Policy.from_pretrained(step_dir, config=config).to(self.device).eval()
        self.preprocessor, self.postprocessor = make_pi05_pre_post_processors(
            config, dataset_stats=stats)
        variant = train_cfg.get("variant", "unknown")
        logger.info("lerobot_pi05_fft restored: step_dir=%s  variant=%s  base=%s",
                    step_dir, variant, train_cfg.get("pretrained"))

        # Language instructions — the annotations the model was trained on.
        # load_demo_static returns the eval demo's annotation. We honour the
        # variant: training used either ground_truth_natural_language_annotation
        # or human_video_annotation_qwen3-vl-4b; we read back the same file.
        ann_path = train_cfg.get("annotations") or str(
            Path(train_cfg["data_root"]) / "human_video_annotation_qwen3-vl-4b.json")
        self._annotations = json.loads(Path(ann_path).read_text()).get("annotations", {})
        logger.info("lerobot_pi05_fft annotations <- %s (%d entries)",
                    ann_path, len(self._annotations))

    # -- demo conditioning (the demo's language instruction) ---------------

    def load_demo_static(self, demo_distract: bool, task: str, demo_id: int) -> str:
        key = f"{_DISTRACT_DIRS[bool(demo_distract)]}/{task}/{int(demo_id):04d}"
        instruction = self._annotations.get(key)
        if instruction is None:  # fall back to a generic task instruction
            instruction = f"complete the {task.replace('_', ' ')} task"
        return instruction

    # -- rollout -----------------------------------------------------------

    def _build_batch(self, obs, instruction: str) -> dict:
        """Raw lerobot batch (B=1) from a live ManiSkill observation."""
        od = dict(extract_obs_for_pi05(obs, cameras=self.robot_cameras))
        batch: dict = {}
        for cam in self.robot_cameras:
            rgb = np.asarray(od[f"observation/{cam}"], dtype=np.float32) / 255.0
            batch[f"{OBS_IMAGES}.{cam}"] = (
                torch.from_numpy(rgb).permute(2, 0, 1).unsqueeze(0))  # (1,3,H,W)
        batch[OBS_STATE] = torch.from_numpy(
            np.asarray(od["observation/state"], dtype=np.float32)).unsqueeze(0)    # (1, state_dim)
        batch["task"] = [instruction]
        return batch

    def rollout(self, env, obs, demo_static, max_rollout_steps: int,
                chunk_size: int, video_frames) -> bool:
        instruction = str(demo_static)
        if video_frames is not None:
            video_frames.append(_extract_rgb_uint8_hwc(obs, self.robot_cameras[0]).copy())
        action_queue: deque[np.ndarray] = deque()
        info: dict = {}

        for _step in range(max_rollout_steps):
            if not action_queue:
                batch = self.preprocessor(self._build_batch(obs, instruction))
                chunk = self.policy.predict_action_chunk(batch)          # (1, chunk, action_dim)
                chunk = self.postprocessor(chunk)                        # unnormalize
                chunk = np.asarray(chunk[0].detach().cpu(), dtype=np.float32)
                for a in chunk[:chunk_size]:
                    action_queue.append(a)
            action = action_queue.popleft()
            obs, _r, terminated, truncated, info = env.step(
                torch.from_numpy(action).float().unsqueeze(0))
            if video_frames is not None:
                video_frames.append(_extract_rgb_uint8_hwc(obs, self.robot_cameras[0]).copy())
            if _success_from_info(info):
                return True
            if terminated or truncated:
                break
        return _success_from_info(info)


def build_runner(eval_cfg) -> LeRobotPi05FFTRunner:
    """Factory used by ``scripts/evaluation.py``. ``checkpoint_dir`` is the
    ``train_vlm_pi05.py`` output dir (holds the per-step full-weight
    subdirs + ``norm_stats.json`` + ``train_config.json``)."""
    ckpt = eval_cfg.raw.get("checkpoint_dir")
    if not ckpt:
        raise ValueError("lerobot_pi05_fft eval config must set 'checkpoint_dir'.")
    train_cfg = load_train_config(ckpt)
    return LeRobotPi05FFTRunner(ckpt, train_cfg)
