"""UniSkill eval runner — wraps ``src.wrappers.uniskill.UniSkillPipeline``.

UniSkill is not trained with the LfO training scripts, so it has no sibling
``train_config.json``: the eval config carries its model path
(``policy_checkpoint``), the pre-extracted human-skill directory (``skill_dir``),
and the data context (``data_dir``, optionally ``tasks`` / ``demo_camera`` /
``robot_cameras``).

The skill is **not** re-extracted from the demo video at eval time. Instead the
per-demo skill sequence is loaded from the pre-extracted UniSkill skill
representations (``skill_dir/<distract>/<task>/<demo_id>/<camera>.npy``) and set
directly on the pipeline via ``set_skill``. The IDM is therefore not needed for
evaluation — only the skill-conditioned policy.
"""

from __future__ import annotations

import logging
from pathlib import Path

import numpy as np
import torch

from demonstration_vla_suite.shared.maniskill_io import _extract_rgb_uint8_hwc
from evaluation.utils import _success_from_info, DEMO_DISTRACTION_DIRS

logger = logging.getLogger("evaluation.uniskill")

_DEFAULT_UNISKILL_CONFIG = str(Path(__file__).resolve().parents[2] / "configs" / "uniskill.json")


class UniSkillRunner:
    name = "uniskill"

    def __init__(
        self,
        policy_checkpoint: str,
        skill_dir: str,
        uniskill_config: str | None,
        data_dir: str,
        tasks: list[str] | None = None,
        demo_camera: str = "front",
        robot_cameras: list[str] | None = None,
        idm_checkpoint: str | None = None,
    ):
        from src.wrappers.uniskill import UniSkillPipeline

        self.data_dir = data_dir
        self.skill_dir = Path(skill_dir)
        self.tasks = list(tasks) if tasks else ["pick_cube"]
        self.demo_camera = demo_camera
        self.robot_camera = (robot_cameras or ["base_camera"])[0]
        # UniSkill predicts one action per env step (no action chunking).
        self.action_horizon = 1
        self.default_output_dir = Path(policy_checkpoint).resolve().parent / "eval"

        logger.info("Building UniSkillPipeline (policy=%s, skill_dir=%s)",
                    policy_checkpoint, skill_dir)
        # ``idm_checkpoint`` is optional: evaluation conditions on the
        # pre-extracted skills (``set_skill``), so the IDM is never run — the
        # pipeline builds it unloaded and only the policy is used.
        self._pipeline = UniSkillPipeline(
            idm_checkpoint=idm_checkpoint,
            policy_checkpoint=policy_checkpoint,
            config_path=uniskill_config or _DEFAULT_UNISKILL_CONFIG,
        )

    # -- demo conditioning (pre-extracted skill sequence) ------------------

    def load_demo_static(self, demo_distract: bool, task: str, demo_id: int) -> np.ndarray:
        """Load the pre-extracted UniSkill skill sequence for this demo.

        Layout: ``skill_dir/<distract>/<task>/<demo_id>/<camera>.npy`` — a
        ``[num_pairs, 1, skill_dim]`` float32 array (the IDM per-frame-pair
        skills, produced offline by the skill-extraction step).
        """
        skill_path = (
            self.skill_dir
            / DEMO_DISTRACTION_DIRS[demo_distract]
            / task
            / f"{demo_id:04d}"
            / f"{self.demo_camera}.npy"
        )
        if not skill_path.is_file():
            raise FileNotFoundError(f"Pre-extracted UniSkill skill not found: {skill_path}")
        return np.load(skill_path)

    # -- rollout -----------------------------------------------------------

    def rollout(self, env, obs, demo_static, max_rollout_steps: int, chunk_size: int, video_frames) -> bool:
        # demo_static is the pre-extracted skill sequence (num_pairs, 1, skill_dim).
        self._pipeline.set_skill(torch.from_numpy(np.asarray(demo_static)))
        if torch.cuda.is_available():
            torch.cuda.empty_cache()
        self._pipeline.reset_episode()

        if video_frames is not None:
            video_frames.append(_extract_rgb_uint8_hwc(obs, self.robot_camera).copy())
        info: dict = {}
        for _step in range(max_rollout_steps):
            action = np.asarray(self._pipeline.predict_action(obs), dtype=np.float32)
            obs, _, terminated, truncated, info = env.step(torch.from_numpy(action).float().unsqueeze(0))
            if video_frames is not None:
                video_frames.append(_extract_rgb_uint8_hwc(obs, self.robot_camera).copy())
            if _success_from_info(info):
                return True
            if terminated or truncated:
                break
        return _success_from_info(info)


def build_runner(eval_cfg) -> UniSkillRunner:
    raw = eval_cfg.raw
    for k in ("policy_checkpoint", "skill_dir", "data_dir"):
        if not raw.get(k):
            raise ValueError(f"uniskill eval config must set '{k}'.")
    return UniSkillRunner(
        policy_checkpoint=raw["policy_checkpoint"],
        skill_dir=raw["skill_dir"],
        uniskill_config=raw.get("uniskill_config"),
        data_dir=raw["data_dir"],
        tasks=raw.get("tasks"),
        demo_camera=raw.get("demo_camera", "front"),
        robot_cameras=raw.get("robot_cameras"),
        idm_checkpoint=raw.get("idm_checkpoint"),
    )
