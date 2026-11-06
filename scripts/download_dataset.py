"""
Download the RoboReel dataset from HuggingFace.

    https://huggingface.co/datasets/LogosRoboticsLab/roboreel

Three subsets, because the full dataset is large and most users do not need
all of it:

  exclude-depth  (default)  everything except the human-demo ``depths.h5``
                            files. Depth is only needed if you work with the
                            raw depth video or camera calibration; no method
                            in the paper uses it.
  eval                      only the human-demo RGB videos under
                            ``human_demos/eval/``, the held-out demonstrations
                            the evaluation protocol conditions on. Evaluating a
                            released checkpoint also needs the per-method
                            preprocessing in the README, run on these videos.
  full                      everything, depth included.

Usage:
    python scripts/download_dataset.py --dest /path/to/roboreel
    python scripts/download_dataset.py --subset eval --dest /path/to/roboreel
    python scripts/download_dataset.py --subset full --dry-run

Resumes safely: re-running skips files already present and complete, so an
interrupted download can simply be restarted with the same arguments.
"""

import argparse
import os
import sys

from huggingface_hub import HfApi, snapshot_download
from huggingface_hub.utils import filter_repo_objects

REPO_ID = "LogosRoboticsLab/roboreel"

# ``*`` matches ``/`` in these patterns, so a single ``*`` spans path segments.
SUBSETS = {
    "exclude-depth": {"ignore_patterns": ["*/depths.h5"]},
    "eval": {"allow_patterns": ["human_demos/eval/*.mp4", "README.md"]},
    "full": {},
}
DEFAULT_SUBSET = "exclude-depth"


def _human(n: int) -> str:
    return f"{n / 1e9:.2f} GB" if n >= 1e9 else f"{n / 1e6:.1f} MB"


def resolve_selection(subset: str, revision: str | None):
    """The files ``subset`` selects, as ``(paths, total_bytes)``.

    Applies the same ``filter_repo_objects`` that ``snapshot_download`` uses
    internally, so this is what would actually be fetched rather than an
    estimate.
    """
    info = HfApi().dataset_info(REPO_ID, revision=revision, files_metadata=True)
    size_of = {s.rfilename: (s.size or 0) for s in info.siblings}
    selected = list(filter_repo_objects(list(size_of), **SUBSETS[subset]))
    return selected, sum(size_of[p] for p in selected)


def main() -> int:
    ap = argparse.ArgumentParser(
        description="Download the RoboReel dataset from HuggingFace.",
        formatter_class=argparse.RawDescriptionHelpFormatter)
    ap.add_argument("--subset", choices=sorted(SUBSETS), default=DEFAULT_SUBSET,
                    help=f"which files to fetch (default: {DEFAULT_SUBSET})")
    ap.add_argument("--dest", default=os.environ.get("ROBOREEL_DATA"),
                    help="target directory; defaults to $ROBOREEL_DATA")
    ap.add_argument("--revision", default=None,
                    help="dataset revision (branch, tag or commit); default: main")
    ap.add_argument("--workers", type=int, default=8,
                    help="parallel download workers (default: 8)")
    ap.add_argument("--dry-run", action="store_true",
                    help="report what would be downloaded, then exit")
    args = ap.parse_args()

    if not args.dest:
        ap.error("set --dest or the ROBOREEL_DATA environment variable")

    paths, total = resolve_selection(args.subset, args.revision)
    if not paths:
        print(f"error: subset '{args.subset}' matched no files in {REPO_ID} — "
              "the dataset layout may have changed.", file=sys.stderr)
        return 1

    print(f"subset : {args.subset}"
          f"{' (default)' if args.subset == DEFAULT_SUBSET else ''}")
    print(f"source : {REPO_ID}" + (f" @ {args.revision}" if args.revision else ""))
    print(f"dest   : {args.dest}")
    print(f"files  : {len(paths)}  ({_human(total)})")

    if args.dry_run:
        for p in paths[:10]:
            print(f"   {p}")
        if len(paths) > 10:
            print(f"   ... and {len(paths) - 10} more")
        return 0

    snapshot_download(repo_id=REPO_ID, repo_type="dataset",
                      revision=args.revision, local_dir=args.dest,
                      max_workers=args.workers, **SUBSETS[args.subset])
    print(f"\ndone -> {args.dest}")
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
