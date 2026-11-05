"""Single evaluation entry point.

Picks a runner from the eval config's ``model`` field
(``demo_pi05`` | ``uniskill`` | ``lerobot_pi05_fft`` | ``vid2robot``) and hands it to the shared
2×2 protocol driver (``evaluation.utils.run_eval_protocol``).

Eval config schema (``configs/eval_maniskill_*/<name>.json``)::

    # demo_pi05 — checkpoint-path driven; everything model/data related
    # comes from the train_config.json persisted beside the checkpoint:
    { "model": "demo_pi05", "checkpoint_dir": ".../29999", "scope": "full", "save_video": false }

    # uniskill — not trained with the LfO scripts, so it carries its paths + data context:
    { "model": "uniskill", "idm_checkpoint": "...", "policy_checkpoint": "...",
      "uniskill_config": "configs/uniskill.json", "data_dir": "...",
      "tasks": ["pick_cube"], "demo_camera": "front", "robot_cameras": ["base_camera"],
      "scope": "full", "save_video": false }

Usage::

    python scripts/evaluation.py --config configs/eval_maniskill_vla/full.json [--gpu N]

``--gpu N`` (alias ``--cuda-device``) pins this run to one GPU — sets ``CUDA_VISIBLE_DEVICES``
and the EGL render device — before mani_skill / jax / torch get imported.
"""

from __future__ import annotations

import argparse
import logging
import os
import sys
from pathlib import Path


def _pin_cuda_device_from_argv() -> None:
    """``--gpu``/``--cuda-device <id>`` → ``CUDA_VISIBLE_DEVICES`` + ``MUJOCO_EGL_DEVICE_ID``.

    Applied before mani_skill / jax / torch are imported (they grab the GPU on import).
    """
    p = argparse.ArgumentParser(add_help=False)
    p.add_argument("--gpu", "--cuda-device", dest="gpu", default=None)
    gpu = p.parse_known_args()[0].gpu
    if gpu is not None:
        os.environ["CUDA_VISIBLE_DEVICES"] = str(gpu)
        os.environ["MUJOCO_EGL_DEVICE_ID"] = str(gpu)


_pin_cuda_device_from_argv()

os.environ.setdefault("MUJOCO_GL", "egl")
os.environ.setdefault("MUJOCO_EGL_DEVICE_ID", "0")

_PROJECT_ROOT = Path(__file__).resolve().parents[1]
_THIS_DIR = Path(__file__).resolve().parent
# Drop the script's own directory from sys.path: this file is ``scripts/evaluation.py``
# and would otherwise shadow the ``src/evaluation`` package on ``import evaluation.*``.
sys.path[:] = [p for p in sys.path if Path(p or ".").resolve() != _THIS_DIR]
for _p in (
    _PROJECT_ROOT,
    _PROJECT_ROOT / "src",
    _PROJECT_ROOT / "src" / "external_models",
    _PROJECT_ROOT / "src" / "external_models" / "lerobot" / "src",
    _PROJECT_ROOT / "src" / "external_models" / "UniSkill",
    _PROJECT_ROOT / "src" / "external_models" / "UniSkill-Policy",
    _PROJECT_ROOT / "roboreel_env" / "src",
):
    _s = str(_p)
    if _s not in sys.path:
        sys.path.insert(0, _s)

import mani_skill.envs  # noqa: F401  — registers built-in ManiSkill envs
import envs  # noqa: F401  — registers the RoboReel envs

from evaluation.utils import load_eval_config, run_eval_protocol

logging.basicConfig(level=logging.INFO, format="%(asctime)s %(levelname)s %(message)s")
logger = logging.getLogger("evaluation")


_RUNNER_MODULES = {
    "demo_pi05": "evaluation.demo_pi05_runner",
    "lerobot_pi05_fft": "evaluation.lerobot_pi05_fft_wrapper",
    "uniskill": "evaluation.uniskill_runner",
    "vid2robot": "evaluation.vid2robot_runner",
}


def main() -> None:
    p = argparse.ArgumentParser(description="Evaluate an LfO policy via the shared 2x2 protocol.")
    p.add_argument("--config", required=True, help="Path to the eval config JSON.")
    p.add_argument("--gpu", "--cuda-device", dest="gpu", default=None,
                   help="CUDA device id to run on, e.g. '3' (sets CUDA_VISIBLE_DEVICES and the EGL "
                        "render device; applied at import time, before mani_skill/jax/torch load).")
    args = p.parse_args()

    eval_cfg = load_eval_config(args.config)
    if eval_cfg.model not in _RUNNER_MODULES:
        raise ValueError(f"Unknown model {eval_cfg.model!r}. Expected one of {sorted(_RUNNER_MODULES)}.")

    import importlib
    runner_mod = importlib.import_module(_RUNNER_MODULES[eval_cfg.model])
    runner = runner_mod.build_runner(eval_cfg)
    run_eval_protocol(eval_cfg, runner)


if __name__ == "__main__":
    main()
