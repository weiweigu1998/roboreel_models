"""Train the UniSkill diffusion policy (Robomimic) from a RoboReel config.

Robomimic's ``train.py`` calls ``os.path.expanduser`` on dataset paths but not
``os.path.expandvars``, so the ``${ROBOREEL_DATA}`` / ``${ROBOREEL_CKPT}``
placeholders our configs use would reach it unexpanded. This launcher expands
them, writes the resolved config to a temporary file, and hands that to
Robomimic — so UniSkill is driven the same way as the other three methods.

The inverse-skill-dynamics model is **not** trained here: the paper uses the
UniSkill authors' released checkpoint unmodified. Only the policy over its
skill latents is trained, from the latents produced by
``scripts/extract_uniskill_robot_skill.py``.

Usage:
    python scripts/train_uniskill_policy.py --config configs/train_uniskill_policy/nd_pretrained_isd.json
"""

from __future__ import annotations

import argparse
import json
import os
import subprocess
import sys
import tempfile
from pathlib import Path

_PROJECT_ROOT = Path(__file__).resolve().parents[1]
sys.path.insert(0, str(_PROJECT_ROOT / "src"))

from config_paths import expand_paths  # noqa: E402

_POLICY_REPO = _PROJECT_ROOT / "src" / "external_models" / "UniSkill-Policy"


def main() -> int:
    ap = argparse.ArgumentParser(description=__doc__.split("\n")[0])
    ap.add_argument("--config", required=True,
                    help="Robomimic config JSON, e.g. "
                         "configs/train_uniskill_policy/nd_pretrained_isd.json")
    ap.add_argument("--gpu", default=None, help="CUDA device id.")
    ap.add_argument("--dry-run", action="store_true",
                    help="Resolve the config and print the command, then exit.")
    args = ap.parse_args()

    train_py = _POLICY_REPO / "robomimic" / "scripts" / "train.py"
    if not train_py.is_file():
        print(f"error: {train_py} not found — run "
              "`git submodule update --init src/external_models/UniSkill-Policy`.",
              file=sys.stderr)
        return 1

    cfg = expand_paths(json.loads(Path(args.config).read_text()))

    env = dict(os.environ)
    env["PYTHONPATH"] = os.pathsep.join(
        [str(_POLICY_REPO), str(_PROJECT_ROOT / "src"), env.get("PYTHONPATH", "")]
    ).rstrip(os.pathsep)
    if args.gpu is not None:
        env["CUDA_VISIBLE_DEVICES"] = str(args.gpu)

    # Robomimic reads the config by path, so the resolved copy has to be a file.
    with tempfile.NamedTemporaryFile("w", suffix=".json", prefix="roboreel_uniskill_",
                                     delete=False) as fh:
        json.dump(cfg, fh, indent=2)
        resolved = fh.name

    cmd = [sys.executable, str(train_py), "--config", resolved]
    print(f"config   : {args.config} (resolved -> {resolved})")
    print(f"command  : {' '.join(cmd)}")
    if args.dry_run:
        return 0
    try:
        return subprocess.call(cmd, env=env)
    finally:
        os.unlink(resolved)


if __name__ == "__main__":
    raise SystemExit(main())
