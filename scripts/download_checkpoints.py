"""
Download the RoboReel policy checkpoints from HuggingFace.

    https://huggingface.co/LogosRoboticsLab/roboreel_models

The repository is laid out as ``<condition>/<method>/``, where the condition is
the training suite (``nd`` = no distraction, ``wd`` = with distraction) and
each method directory holds the evaluated checkpoint as ``ckpt/``,
``ckpt.pt`` or ``ckpt.pth`` alongside the run's own ``train_config.json``.

The paper's ED suite (clean training, distracted evaluation) reuses the ``nd``
checkpoints, so it needs no separate download.

Usage:
    # one method, one condition (~6 GB)
    python scripts/download_checkpoints.py --method vid_cond_pi05_keyframes --condition nd

    # everything (~50 GB)
    python scripts/download_checkpoints.py --dest /path/to/checkpoints

    python scripts/download_checkpoints.py --dry-run

Resumes safely: re-running skips files already present and complete.
"""

import argparse
import os
import sys

from huggingface_hub import HfApi, snapshot_download
from huggingface_hub.utils import filter_repo_objects

REPO_ID = "LogosRoboticsLab/roboreel_models"

METHODS = [
    "vlm_pi05",                 # VLM + pi0.5
    "vid_cond_pi05_uniform",    # Vid. Cond. pi0.5 (Uniform)
    "vid_cond_pi05_keyframes",  # Vid. Cond. pi0.5 (Keyframes)
    "vid2robot",
    "uniskill_policy",
]
CONDITIONS = ["nd", "wd"]


def _human(n: int) -> str:
    return f"{n / 1e9:.2f} GB" if n >= 1e9 else f"{n / 1e6:.1f} MB"


def build_patterns(methods: list[str], conditions: list[str]) -> list[str]:
    # ``*`` spans ``/`` in hub patterns, so one entry covers a whole subtree.
    return ["README.md"] + [f"{c}/{m}/*" for c in conditions for m in methods]


def resolve_selection(patterns: list[str], revision: str | None):
    """The files ``patterns`` select, as ``(paths, total_bytes)``.

    Uses the same ``filter_repo_objects`` that ``snapshot_download`` applies
    internally, so this is what would actually be fetched.
    """
    info = HfApi().model_info(REPO_ID, revision=revision, files_metadata=True)
    size_of = {s.rfilename: (s.size or 0) for s in info.siblings}
    selected = list(filter_repo_objects(list(size_of), allow_patterns=patterns))
    return selected, sum(size_of[p] for p in selected)


def main() -> int:
    ap = argparse.ArgumentParser(
        description="Download the RoboReel policy checkpoints from HuggingFace.")
    ap.add_argument("--method", action="append", choices=METHODS + ["all"],
                    help="method to fetch; repeatable (default: all)")
    ap.add_argument("--condition", action="append", choices=CONDITIONS + ["all"],
                    help="training suite to fetch; repeatable (default: all)")
    ap.add_argument("--dest", default=os.environ.get("ROBOREEL_CKPT"),
                    help="target directory; defaults to $ROBOREEL_CKPT")
    ap.add_argument("--revision", default=None,
                    help="model revision (branch, tag or commit); default: main")
    ap.add_argument("--workers", type=int, default=8,
                    help="parallel download workers (default: 8)")
    ap.add_argument("--dry-run", action="store_true",
                    help="report what would be downloaded, then exit")
    args = ap.parse_args()

    if not args.dest:
        ap.error("set --dest or the ROBOREEL_CKPT environment variable")

    methods = METHODS if not args.method or "all" in args.method else args.method
    conditions = (CONDITIONS if not args.condition or "all" in args.condition
                  else args.condition)

    patterns = build_patterns(methods, conditions)
    paths, total = resolve_selection(patterns, args.revision)
    if not paths:
        print(f"error: nothing matched in {REPO_ID} — the layout may have changed.",
              file=sys.stderr)
        return 1

    print(f"methods    : {', '.join(methods)}")
    print(f"conditions : {', '.join(conditions)}")
    print(f"source     : {REPO_ID}" + (f" @ {args.revision}" if args.revision else ""))
    print(f"dest       : {args.dest}")
    print(f"files      : {len(paths)}  ({_human(total)})")

    if args.dry_run:
        for p in paths:
            print(f"   {p}")
        return 0

    snapshot_download(repo_id=REPO_ID, repo_type="model", revision=args.revision,
                      local_dir=args.dest, max_workers=args.workers,
                      allow_patterns=patterns)
    print(f"\ndone -> {args.dest}")
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
