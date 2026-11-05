"""
UniSkill end-to-end pipeline wrapper.

This module wraps the two UniSkill components — the Inverse Dynamics Model (IDM)
and the skill-conditioned Diffusion Policy (DiffusionPolicyUNet) — into a single
pipeline class that can interact directly with our ManiSkill benchmark environments.

Pipeline overview:
  1. **Skill extraction (offline, once per episode):**
     Given a demonstration video (sequence of RGB frames), the IDM extracts a
     per-frame-pair skill vector. These are averaged into a single skill embedding
     that summarizes the demonstrated behaviour.

  2. **Action prediction (online, every step):**
     At each environment step, the policy receives the current observation dict
     (proprioceptive state + camera images) together with the pre-computed skill
     embedding, and produces an action via DDIM denoising.

Usage:
    pipeline = UniSkillPipeline(
        idm_checkpoint="path/to/idm.pth",
        policy_checkpoint="path/to/model_epoch_N.pth",
    )
    pipeline.set_skill_from_demo(demo_frames)   # np.ndarray [T, H, W, 3]
    obs, _ = env.reset()
    action = pipeline.predict_action(obs)
"""

from __future__ import annotations

import json
import sys
from collections import OrderedDict
from dataclasses import dataclass, field
from pathlib import Path
from typing import Dict, Optional

import numpy as np
import torch
import torch.nn.functional as F
import torchvision

# ---------------------------------------------------------------------------
# Path setup — add external model source trees so their imports resolve.
# ---------------------------------------------------------------------------
_WORKSPACE_ROOT = Path(__file__).resolve().parents[2]
_UNISKILL_ROOT = _WORKSPACE_ROOT / "src" / "external_models" / "UniSkill"
_POLICY_ROOT = _WORKSPACE_ROOT / "src" / "external_models" / "UniSkill-Policy"

for _p in (_UNISKILL_ROOT, _POLICY_ROOT):
    _s = str(_p)
    if _s not in sys.path:
        sys.path.insert(0, _s)


# ---------------------------------------------------------------------------
# Default config path
# ---------------------------------------------------------------------------
_DEFAULT_CONFIG_PATH = _WORKSPACE_ROOT / "configs" / "uniskill.json"


# ---------------------------------------------------------------------------
# Configuration dataclass
# ---------------------------------------------------------------------------
@dataclass
class UniSkillConfig:
    """All hyperparameters for both IDM and policy, with production defaults.

    Typically loaded from configs/uniskill.json via UniSkillConfig.from_json().
    """

    # --- IDM ---
    idm_num_layers: int = 8
    idm_num_heads: int = 4
    idm_hidden_dim: int = 256
    idm_skill_dim: int = 64
    idm_out_dim: int = 768
    idm_resolution: int = 224
    idm_window_size: int = 2
    idm_patch_size: int = 16

    # --- Depth estimator (used by IDM for skill extraction) ---
    depth_model: str = "depth-anything/Depth-Anything-V2-Small-hf"

    # --- Policy ---
    policy_config_path: Optional[str] = None  # defaults to bundled config
    ac_dim: int = 7
    img_size: int = 128  # raw RGB input size (must be > crop size in config)

    # --- Skill extraction ---
    skill_interval: int = 1  # temporal offset between IDM frame pairs
    skill_batch_size: int = 64

    # --- Observation key mapping ---
    # Maps our environment's observation keys to the policy's expected keys.
    # This is the bridge between ManiSkill obs format and UniSkill-Policy format.
    # Keys: policy expected name -> value: (source, env_key) where source is
    # "agent", "extra", or "sensor" (for camera images).
    #
    # For low_dim keys the value is [source_group, key, [start, end]] where the
    # third element is a two-element list that becomes a slice.
    # For rgb keys the value is ["sensor", camera_name, modality].
    #
    # Defaults correspond to the OpenMicrowave-v1 environment with a Panda robot.
    # Override for other environments as needed.
    obs_key_mapping: Dict[str, tuple] = field(default_factory=lambda: {
        # low_dim observations
        "ee_pos": ("extra", "tcp_pose", slice(0, 3)),          # first 3 of tcp_pose
        "ee_ori": ("extra", "tcp_pose", "quat2axisangle"),     # quat [3:7] -> axis-angle (3D)
        "gripper_states": ("agent", "qpos", slice(7, 9)),      # last 2 joints = gripper
        "joint_states": ("agent", "qpos", slice(0, 7)),        # first 7 joints
        # rgb observations — camera name in the environment
        "agentview_rgb": ("sensor", "base_camera1", "rgb"),
        "eye_in_hand_rgb": ("sensor", "base_camera4", "rgb"),
    })

    @classmethod
    def from_json(cls, path: Optional[str] = None) -> "UniSkillConfig":
        """Load config from a JSON file.

        The JSON should have a top-level "policy" key containing the config
        fields. Two-element integer lists in obs_key_mapping values are
        automatically converted to slice objects.

        Args:
            path: Path to JSON config file. Defaults to configs/uniskill.json.
        """
        if path is None:
            path = str(_DEFAULT_CONFIG_PATH)

        with open(path, "r") as f:
            raw = json.load(f)

        policy_cfg = raw["policy"]

        # Convert obs_key_mapping list values to tuples, and [start, end]
        # lists into slice objects.
        if "obs_key_mapping" in policy_cfg:
            converted = {}
            for key, spec in policy_cfg["obs_key_mapping"].items():
                parsed = []
                for element in spec:
                    if isinstance(element, list) and len(element) == 2 and all(isinstance(x, int) for x in element):
                        parsed.append(slice(element[0], element[1]))
                    else:
                        parsed.append(element)
                converted[key] = tuple(parsed)
            policy_cfg["obs_key_mapping"] = converted

        return cls(**policy_cfg)


class UniSkillPipeline:
    """End-to-end pipeline wrapping UniSkill IDM + Diffusion Policy."""

    def __init__(
        self,
        idm_checkpoint: Optional[str] = None,
        policy_checkpoint: Optional[str] = None,
        config: Optional[UniSkillConfig] = None,
        config_path: Optional[str] = None,
        device: Optional[torch.device] = None,
    ):
        """
        Args:
            idm_checkpoint: Path to pretrained IDM weights (.pth). If None,
                the IDM is initialised with random weights (for testing).
            policy_checkpoint: Path to pretrained policy checkpoint (.pth).
                If None, the policy is initialised with random weights.
            config: Hyperparameter configuration. If None, loaded from
                config_path (or the default configs/uniskill.json).
            config_path: Path to JSON config file. Only used when config
                is None. Defaults to configs/uniskill.json.
            device: Torch device. Auto-selects CUDA if available.
        """
        if config is not None:
            self.cfg = config
        else:
            self.cfg = UniSkillConfig.from_json(config_path)
        self.device = device or torch.device("cuda" if torch.cuda.is_available() else "cpu")

        # Skill sequence: [num_pairs, 1, skill_dim] for per-timestep skills,
        # or [1, 1, skill_dim] if set directly. See _get_current_skill().
        self._skill_sequence: Optional[torch.Tensor] = None
        self._step_counter: int = 0

        self._build_idm(idm_checkpoint)
        self._build_depth_estimator()
        self._build_policy(policy_checkpoint)

    # ------------------------------------------------------------------
    # Model construction
    # ------------------------------------------------------------------

    def _build_idm(self, checkpoint: Optional[str]) -> None:
        """Instantiate the IDM and optionally load pretrained weights."""
        # Prevent pretrained ResNet weight download inside IDM
        orig_resnet18 = torchvision.models.resnet18

        def _resnet18_no_pretrained(*a, **kw):
            kw.pop("pretrained", None)
            kw["weights"] = None
            return orig_resnet18(*a, **kw)

        torchvision.models.resnet18 = _resnet18_no_pretrained

        from dynamics.idm import IDM

        self.idm = IDM(
            num_layers=self.cfg.idm_num_layers,
            num_heads=self.cfg.idm_num_heads,
            hidden_dim=self.cfg.idm_hidden_dim,
            skill_dim=self.cfg.idm_skill_dim,
            out_dim=self.cfg.idm_out_dim,
            idm_resolution=self.cfg.idm_resolution,
            window_size=self.cfg.idm_window_size,
            patch_size=self.cfg.idm_patch_size,
        )

        if checkpoint is not None:
            state_dict = torch.load(checkpoint, map_location="cpu")
            self.idm.load_state_dict(state_dict)

        self.idm.requires_grad_(False)
        self.idm.eval()
        self.idm.to(self.device)

        # Restore original resnet18
        torchvision.models.resnet18 = orig_resnet18

    def _build_policy(self, checkpoint: Optional[str]) -> None:
        """Instantiate the DiffusionPolicyUNet and optionally load weights."""
        import robomimic.utils.obs_utils as ObsUtils
        from robomimic.config import config_factory
        from robomimic.algo.diffusion_policy import DiffusionPolicyUNet

        # Load config
        config_path = self.cfg.policy_config_path
        if config_path is None:
            config_path = str(_POLICY_ROOT / "configs" / "uniskill_policy.json")

        with open(config_path, "r") as f:
            ext_cfg = json.load(f)

        config = config_factory(ext_cfg["algo_name"])
        with config.values_unlocked():
            config.update(ext_cfg)
        config.lock()

        ObsUtils.initialize_obs_utils_with_config(config)

        # Load checkpoint once so we can read canonical obs_key_shapes and
        # ac_dim.  The key order in obs_key_shapes determines the
        # feature-concatenation order inside the obs encoder and must exactly
        # match what was used at training time.
        ckpt_dict = None
        if checkpoint is not None:
            from robomimic.utils.file_utils import maybe_dict_from_checkpoint
            ckpt_dict = maybe_dict_from_checkpoint(ckpt_path=checkpoint)
            raw_shapes = ckpt_dict["shape_metadata"]["all_shapes"]
            obs_key_shapes = OrderedDict(
                (k, tuple(v)) for k, v in raw_shapes.items()
            )
            ac_dim = ckpt_dict["shape_metadata"]["ac_dim"]
        else:
            # Fall back to the known alphabetical order used by the released
            # libero.pth checkpoint when no checkpoint is provided.
            obs_key_shapes = OrderedDict([
                ("agentview_rgb", (3, self.cfg.img_size, self.cfg.img_size)),
                ("ee_ori", (3,)),
                ("ee_pos", (3,)),
                ("eye_in_hand_rgb", (3, self.cfg.img_size, self.cfg.img_size)),
                ("gripper_states", (2,)),
                ("joint_states", (7,)),
            ])
            ac_dim = self.cfg.ac_dim

        # Subclass to skip optimizer creation (not needed for inference)
        class _InferencePolicy(DiffusionPolicyUNet):
            def _create_optimizers(self):
                self.optimizers = {}
                self.lr_schedulers = {}

        self.policy = _InferencePolicy(
            algo_config=config.algo,
            obs_config=config.observation,
            global_config=config,
            obs_key_shapes=obs_key_shapes,
            ac_dim=ac_dim,
            device=self.device,
        )

        if ckpt_dict is not None:
            self.policy.deserialize(ckpt_dict["model"])

        self.policy.set_eval()
        self.policy.reset()

        # Action denormalization stats. Saved by robomimic alongside the
        # policy weights; their `RolloutPolicy` wrapper applies these at
        # inference time. We bypass `RolloutPolicy` (the UniSkill policy
        # needs a ``skill`` kwarg that `RolloutPolicy.__call__` doesn't
        # forward) and apply them ourselves in `predict_action` via the
        # same `ObsUtils.unnormalize_dict` utility. Without this step the
        # policy emits zero-mean unit-std normalized actions and the env
        # interprets them as nonsensical raw joint targets.
        ans = ckpt_dict.get("action_normalization_stats") if ckpt_dict is not None else None
        if ans is not None:
            # Move stats onto the same device as the policy so the inline
            # `actions * scale + offset` doesn't trip the cuda↔cpu guard.
            self._action_norm_stats = {
                k: {sk: torch.as_tensor(sv, dtype=torch.float32).to(self.device)
                    for sk, sv in v.items()}
                for k, v in ans.items()
            }
        else:
            self._action_norm_stats = None

        self._obs_key_shapes = obs_key_shapes
        self._policy_config = config

    # ------------------------------------------------------------------
    # Depth estimation
    # ------------------------------------------------------------------

    def _build_depth_estimator(self) -> None:
        """Load the depth estimator and processor once at construction time."""
        from transformers import AutoImageProcessor, AutoModelForDepthEstimation

        # use_fast=False matches the slow processor used when pre-computed
        # skills (and therefore the policy) were trained; the fast processor
        # produces slightly different depth maps that shift skill vectors.
        self.depth_processor = AutoImageProcessor.from_pretrained(
            self.cfg.depth_model, use_fast=False
        )
        self.depth_estimator = AutoModelForDepthEstimation.from_pretrained(self.cfg.depth_model)
        self.depth_estimator.requires_grad_(False)
        self.depth_estimator.eval()
        self.depth_estimator.to(self.device)

    # ------------------------------------------------------------------
    # Skill extraction from demonstration
    # ------------------------------------------------------------------

    def set_skill_from_demo(
        self,
        demo_frames: np.ndarray,
        interval: Optional[int] = None,
    ) -> torch.Tensor:
        """
        Extract a per-timestep skill sequence from a demonstration video.

        The IDM processes consecutive frame pairs (curr, next) to produce
        per-pair skill vectors.  These are stored as a sequence and indexed
        at each rollout step (matching evaluate.py behaviour).  When the
        rollout exceeds the number of extracted skills, the last skill is
        reused.

        Args:
            demo_frames: RGB frames as uint8 array of shape [T, H, W, 3].
            interval: Temporal offset between paired frames. Defaults to
                config.skill_interval.

        Returns:
            Skill tensor of shape [num_pairs, 1, skill_dim], also stored
            internally.
        """
        if interval is None:
            interval = self.cfg.skill_interval

        num_frames = demo_frames.shape[0]
        num_pairs = num_frames - interval
        if num_pairs <= 0:
            raise ValueError(
                f"Demonstration has {num_frames} frames but needs at least "
                f"{interval + 1} for skill extraction with interval={interval}."
            )

        depth_processor = self.depth_processor
        depth_estimator = self.depth_estimator
        idm_res = self.cfg.idm_resolution
        batch_size = self.cfg.skill_batch_size

        all_skills: list[torch.Tensor] = []

        for start in range(0, num_pairs, batch_size):
            end = min(start + batch_size, num_pairs)
            curr_idx = np.arange(start, end)
            next_idx = curr_idx + interval

            # Prepare RGB tensors [B, 3, H, W] normalised to [0, 1]
            curr_rgb = (
                torch.from_numpy(demo_frames[curr_idx])
                .permute(0, 3, 1, 2)
                .float()
                .div_(255.0)
            )
            next_rgb = (
                torch.from_numpy(demo_frames[next_idx])
                .permute(0, 3, 1, 2)
                .float()
                .div_(255.0)
            )

            # Resize to IDM resolution and stack as visual pair [B, 2, 3, H, W]
            curr_visual = F.interpolate(
                curr_rgb.to(self.device), size=(idm_res, idm_res),
                mode="bilinear", align_corners=False,
            )
            next_visual = F.interpolate(
                next_rgb.to(self.device), size=(idm_res, idm_res),
                mode="bilinear", align_corners=False,
            )
            visual_pair = torch.stack([curr_visual, next_visual], dim=1)

            # Compute depth maps
            depth_inputs_list = (
                [demo_frames[i] for i in curr_idx]
                + [demo_frames[i] for i in next_idx]
            )
            depth_batch = depth_processor(
                images=depth_inputs_list, do_rescale=False, return_tensors="pt",
            )
            depth_batch = {k: v.to(self.device) for k, v in depth_batch.items()}

            with torch.no_grad():
                depth_out = depth_estimator(**depth_batch)
                depth_maps = (
                    depth_out.predicted_depth
                    if hasattr(depth_out, "predicted_depth")
                    else depth_out[0]
                )
                if depth_maps.ndim == 4 and depth_maps.size(1) == 1:
                    depth_maps = depth_maps.squeeze(1)

                curr_depth, next_depth = torch.chunk(depth_maps, 2, dim=0)
                depth_pair = torch.stack([curr_depth, next_depth], dim=1)
                depth_pair = F.interpolate(
                    depth_pair, size=(idm_res, idm_res),
                    mode="bilinear", align_corners=False,
                )

                skills = self.idm(depth_pair, visual_pair, return_skill=True)
                # skills shape: [B, 1, skill_dim]
                all_skills.append(skills.float())

        # Store the full per-timestep skill sequence
        self._skill_sequence = torch.cat(all_skills, dim=0)  # [num_pairs, 1, skill_dim]
        self._step_counter = 0
        return self._skill_sequence

    def set_skill(self, skill: torch.Tensor) -> None:
        """
        Directly set the skill sequence (e.g. from a pre-computed .npy file).

        Args:
            skill: Tensor of shape:
                - [skill_dim]          — single skill, reused every step
                - [T, skill_dim]       — per-timestep sequence
                - [T, 1, skill_dim]    — per-timestep sequence (as from IDM)
        """
        if skill.ndim == 1:
            skill = skill.unsqueeze(0).unsqueeze(0)  # [1, 1, D]
        elif skill.ndim == 2:
            skill = skill.unsqueeze(1)               # [T, 1, D]
        self._skill_sequence = skill.to(self.device).float()
        self._step_counter = 0

    @property
    def skill(self) -> Optional[torch.Tensor]:
        """Currently stored skill sequence, or None if not yet set."""
        return self._skill_sequence

    def _get_current_skill(self) -> torch.Tensor:
        """
        Get the skill for the current timestep.

        Indexes into the skill sequence using the step counter (matching
        evaluate.py lines 201–204). When steps exceed the sequence length,
        the last skill is reused.

        Returns:
            Skill tensor of shape [1, 1, skill_dim].
        """
        idx = min(self._step_counter, self._skill_sequence.shape[0] - 1)
        return self._skill_sequence[idx:idx + 1]  # [1, 1, skill_dim]

    # ------------------------------------------------------------------
    # Observation translation
    # ------------------------------------------------------------------

    @staticmethod
    def _quat_to_axis_angle(quat: torch.Tensor, convention: str = "wxyz") -> torch.Tensor:
        """Convert quaternion to axis-angle (3D).

        Uses the same formula as robosuite.utils.transform_utils.quat2axisangle
        (acos + sqrt(1 - w^2)) for numerical parity with LIBERO evaluation.

        Args:
            quat: Quaternion tensor of shape [..., 4].
            convention: Quaternion element order.
                "wxyz" — ManiSkill format [w, x, y, z] (default).
                "xyzw" — robosuite / LIBERO format [x, y, z, w].
        """
        if convention == "xyzw":
            xyz = quat[..., :3]
            w = quat[..., 3]
        else:  # wxyz
            w = quat[..., 0]
            xyz = quat[..., 1:]
        w = w.clamp(-1.0, 1.0)
        den = torch.sqrt(1.0 - w * w).clamp(min=1e-8)
        angle = 2.0 * torch.acos(w)
        return xyz * (angle / den).unsqueeze(-1)

    def _translate_obs(self, env_obs: dict) -> dict:
        """
        Translate an environment observation dict into the format expected by
        the UniSkill policy.

        Supports two observation formats controlled by obs_key_mapping:

        **ManiSkill (nested):** source_spec starts with "agent", "extra", or
        "sensor".  The obs dict is nested:
            {"agent": {"qpos": ...}, "extra": {"tcp_pose": ...},
             "sensor_data": {"cam": {"rgb": ...}}}

        **Flat (e.g. LIBERO / robosuite):** source_spec starts with "flat" or
        "flat_rgb".  The obs dict is a flat dictionary:
            {"robot0_eef_pos": ..., "agentview_image": ...}

        Args:
            env_obs: Raw observation dict from env.step() or env.reset().

        Returns:
            Dict mapping policy obs keys to tensors of shape [1, *obs_shape].
        """
        policy_obs = {}

        for policy_key, source_spec in self.cfg.obs_key_mapping.items():
            spec_type = source_spec[0]

            if spec_type in ("sensor", "flat_rgb"):
                # --- RGB image ---
                if spec_type == "sensor":
                    _, camera_name, modality = source_spec
                    img = env_obs["sensor_data"][camera_name][modality]
                else:  # flat_rgb
                    img = env_obs[source_spec[1]]

                if isinstance(img, np.ndarray):
                    img = torch.from_numpy(img)
                img = img.float()
                if img.ndim == 4:
                    img = img[0].permute(2, 0, 1)
                elif img.ndim == 3:
                    img = img.permute(2, 0, 1)
                # Handle RGBA → RGB (ManiSkill "Color" has 4 channels)
                if img.shape[0] == 4:
                    img = img[:3]
                if img.max() > 1.0:
                    img = img / 255.0
                img = F.interpolate(
                    img.unsqueeze(0),
                    size=(self.cfg.img_size, self.cfg.img_size),
                    mode="bilinear",
                    align_corners=False,
                ).squeeze(0)
                policy_obs[policy_key] = img.unsqueeze(0).to(self.device)

            elif spec_type == "flat":
                # --- Flat low-dim (e.g. LIBERO) ---
                val = env_obs[source_spec[1]]
                if isinstance(val, np.ndarray):
                    val = torch.from_numpy(val)
                # Keep native precision (float64) for quat conversion,
                # matching evaluate.py which computes in numpy float64.
                if val.ndim >= 2:
                    val = val[0]
                if len(source_spec) > 2:
                    transform = source_spec[2]
                    if transform == "quat2axisangle":
                        val = self._quat_to_axis_angle(val.double(), convention="wxyz").float()
                    elif transform == "quat2axisangle_xyzw":
                        val = self._quat_to_axis_angle(val.double(), convention="xyzw").float()
                    else:
                        val = val[transform].float()
                else:
                    val = val.float()
                policy_obs[policy_key] = val.unsqueeze(0).to(self.device)

            else:
                # --- Nested low-dim (ManiSkill) ---
                source_group, source_key = source_spec[0], source_spec[1]
                val = env_obs[source_group][source_key]
                if isinstance(val, np.ndarray):
                    val = torch.from_numpy(val)
                val = val.float()
                if val.ndim >= 2:
                    val = val[0]
                if len(source_spec) > 2:
                    transform = source_spec[2]
                    if transform == "quat2axisangle":
                        quat = val[3:7]
                        val = self._quat_to_axis_angle(quat, convention="wxyz")
                    elif transform == "quat2axisangle_xyzw":
                        quat = val[3:7]
                        val = self._quat_to_axis_angle(quat, convention="xyzw")
                    else:
                        val = val[transform]
                policy_obs[policy_key] = val.unsqueeze(0).to(self.device)

        return policy_obs

    # ------------------------------------------------------------------
    # Action prediction
    # ------------------------------------------------------------------

    def predict_action(self, env_obs: dict) -> np.ndarray:
        """
        Predict a robot action given the current environment observation.

        This translates the ManiSkill observation into the policy's format,
        feeds it together with the stored skill embedding into the diffusion
        policy, and returns the predicted action as a numpy array.

        Args:
            env_obs: Raw observation dict from env.step() or env.reset().

        Returns:
            Action as numpy array of shape [ac_dim].

        Raises:
            RuntimeError: If no skill has been set (call set_skill_from_demo
                or set_skill first).
        """
        if self._skill_sequence is None:
            raise RuntimeError(
                "No skill embedding set. Call set_skill_from_demo() or "
                "set_skill() before predicting actions."
            )

        obs_dict = self._translate_obs(env_obs)
        current_skill = self._get_current_skill()

        with torch.no_grad():
            action = self.policy.get_action(
                obs_dict=obs_dict,
                skill=current_skill,
            )
        self._step_counter += 1

        # Denormalize via robomimic's saved `action_normalization_stats`
        # (scale + offset). Mirrors what `RolloutPolicy.__call__` does at
        # inference time; we have to do it inline since UniSkill's policy
        # takes a `skill` kwarg `RolloutPolicy.__call__` doesn't forward.
        if self._action_norm_stats is not None:
            from robomimic.utils import obs_utils as ObsUtils
            ac_in = action if action.ndim == 2 else action.unsqueeze(0)
            ac_dict = ObsUtils.unnormalize_dict(
                {"actions": ac_in}, normalization_stats=self._action_norm_stats)
            action = ac_dict["actions"]

        # action shape: [1, ac_dim] -> [ac_dim] numpy
        if isinstance(action, torch.Tensor):
            action = action.cpu().numpy()
        if action.ndim == 2:
            action = action[0]

        return action

    def predict_action_batched(self, obs_list: list) -> np.ndarray:
        """
        Predict actions for a batch of environment observations.

        Translates each observation, stacks them into a batch, and runs
        a single batched forward pass through the policy.  The step counter
        is incremented once (not per environment).

        Args:
            obs_list: List of raw observation dicts, one per environment.

        Returns:
            Actions as numpy array of shape [num_envs, ac_dim].
        """
        if self._skill_sequence is None:
            raise RuntimeError(
                "No skill embedding set. Call set_skill_from_demo() or "
                "set_skill() before predicting actions."
            )

        num_envs = len(obs_list)
        # Translate each obs and stack into batched tensors
        translated = [self._translate_obs(obs) for obs in obs_list]
        batched_obs = {}
        for key in translated[0]:
            batched_obs[key] = torch.cat([t[key] for t in translated], dim=0)

        current_skill = self._get_current_skill()
        # Expand skill to batch: [1, 1, D] -> [num_envs, 1, D]
        batched_skill = current_skill.expand(num_envs, -1, -1)

        with torch.no_grad():
            actions = self.policy.get_action(
                obs_dict=batched_obs,
                skill=batched_skill,
            )
        self._step_counter += 1

        # Same denorm as in predict_action; see comment there.
        if self._action_norm_stats is not None:
            from robomimic.utils import obs_utils as ObsUtils
            ac_dict = ObsUtils.unnormalize_dict(
                {"actions": actions}, normalization_stats=self._action_norm_stats)
            actions = ac_dict["actions"]

        if isinstance(actions, torch.Tensor):
            actions = actions.cpu().numpy()

        return actions

    def reset_episode(self) -> None:
        """
        Reset the policy's internal state for a new episode.

        Resets the step counter (so skill indexing restarts) and the policy's
        internal observation/action queues.  The skill sequence itself is
        preserved — call set_skill_from_demo() or set_skill() to change it.
        """
        self._step_counter = 0
        self.policy.set_eval()
        self.policy.reset()

    # ------------------------------------------------------------------
    # Convenience
    # ------------------------------------------------------------------

    def __repr__(self) -> str:
        if self._skill_sequence is not None:
            seq_len = self._skill_sequence.shape[0]
            skill_info = f"skill=set ({seq_len} steps, dim={self.cfg.idm_skill_dim})"
        else:
            skill_info = f"skill=unset (dim={self.cfg.idm_skill_dim})"
        return (
            f"UniSkillPipeline(\n"
            f"  device={self.device},\n"
            f"  {skill_info},\n"
            f"  ac_dim={self.cfg.ac_dim},\n"
            f"  idm_resolution={self.cfg.idm_resolution},\n"
            f"  img_size={self.cfg.img_size},\n"
            f")"
        )
