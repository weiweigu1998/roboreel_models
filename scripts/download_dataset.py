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
                            ``human_demos/eval/``. This is what the
                            evaluation protocol conditions the policies on,
                            so it is enough to reproduce the reported numbers
                            from released checkpoints — no training data.
  full                      everything, depth included.

After downloading, the HuggingFace layout is linked into the layout the
scripts and runners actually walk (see ``link_layout``):

    human_demos/{train,eval}/no_distraction/<task>/<ep>/<cam>.mp4
        -> raw_video_demonstrations/without_distraction/<task>/<ep>/<cam>.mp4
    robot_trajectories/no_distraction/<task>/<ep>/*
        -> raw_training_trajectories/without_distraction/<task>/<ep>/*

Relative symlinks, so nothing is copied and the tree stays movable.

Usage:
    python scripts/download_dataset.py --dest /path/to/roboreel
    python scripts/download_dataset.py --subset eval --dest /path/to/roboreel
    python scripts/download_dataset.py --subset full --dry-run
    python scripts/download_dataset.py --layout-only --dest /path/to/roboreel

Resumes safely: re-running skips files already present and complete, so an
interrupted download can simply be restarted with the same arguments.
"""

import argparse
import os
import pathlib
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


#: Dataset bucket name -> the name the code walks (``lfo_data.VIDEO_TYPE_DIRS``).
BUCKET_ALIASES = {
    "no_distraction": "without_distraction",
    "with_distraction": "with_distraction",
}
#: Human-demo splits to merge into one tree. Episode numbers do not overlap
#: (``train`` is 0000-0079, ``eval`` is 0080-0099), so merging is lossless and
#: the eval protocol selects episodes by number, not by directory.
HUMAN_SPLITS = ("train", "eval")


def link_layout(dest: pathlib.Path, verbose: bool = True) -> tuple[int, int]:
    """Link the published layout into the one the code reads.

    The dataset is published as
    ``human_demos/{train,eval}/<bucket>/<task>/<episode>/`` and
    ``robot_trajectories/<bucket>/<task>/<episode>/``, while the scripts and
    runners walk ``raw_video_demonstrations/<bucket>/...`` and
    ``raw_training_trajectories/<bucket>/...`` with ``<bucket>`` spelled
    ``without_distraction``. Rather than duplicating ~100 GB, every file is
    linked across with a relative symlink.

    ``human_demos/train/pose_defined`` and
    ``human_demos/excluded_human_demos_with_distraction`` are skipped: neither
    is a bucket the code walks, and no reported result uses them.

    Returns ``(human_links, robot_links)``. Idempotent — existing correct links
    are left alone, so it is safe to re-run after a resumed download.
    """
    made = [0, 0]

    def link_dir(src_dir: pathlib.Path, dst_dir: pathlib.Path, which: int,
                 skip_names: frozenset[str] = frozenset()) -> None:
        for src in sorted(src_dir.iterdir()):
            if not src.is_file() or src.name in skip_names:
                continue
            dst = dst_dir / src.name
            rel = os.path.relpath(src, dst_dir)
            if dst.is_symlink():
                if os.readlink(dst) == rel:
                    continue
                dst.unlink()
            elif dst.exists():
                continue  # a real file already there — leave it alone
            dst_dir.mkdir(parents=True, exist_ok=True)
            dst.symlink_to(rel)
            made[which] += 1

    for split in HUMAN_SPLITS:
        for ds_bucket, code_bucket in BUCKET_ALIASES.items():
            root = dest / "human_demos" / split / ds_bucket
            if not root.is_dir():
                continue
            for task in sorted(p for p in root.iterdir() if p.is_dir()):
                for ep in sorted(p for p in task.iterdir() if p.is_dir()):
                    link_dir(ep,
                             dest / "raw_video_demonstrations" / code_bucket / task.name / ep.name,
                             0, skip_names=frozenset({"depths.h5"}))

    for ds_bucket, code_bucket in BUCKET_ALIASES.items():
        root = dest / "robot_trajectories" / ds_bucket
        if not root.is_dir():
            continue
        for task in sorted(p for p in root.iterdir() if p.is_dir()):
            for ep in sorted(p for p in task.iterdir() if p.is_dir()):
                link_dir(ep,
                         dest / "raw_training_trajectories" / code_bucket / task.name / ep.name, 1)

    if verbose:
        print(f"layout : {made[0]} human-demo + {made[1]} robot-trajectory link(s) "
              f"under raw_video_demonstrations/ and raw_training_trajectories/")
    return tuple(made)


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
    ap.add_argument("--layout-only", action="store_true",
                    help="skip downloading; just (re)build the layout links")
    ap.add_argument("--no-link-layout", action="store_true",
                    help="download only, without linking the layout the code reads")
    args = ap.parse_args()

    if not args.dest:
        ap.error("set --dest or the ROBOREEL_DATA environment variable")

    if args.layout_only:
        dest = pathlib.Path(args.dest)
        if not dest.is_dir():
            print(f"error: {dest} does not exist — download first.", file=sys.stderr)
            return 1
        link_layout(dest)
        return 0

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
    if not args.no_link_layout:
        link_layout(pathlib.Path(args.dest))
    print(f"\ndone -> {args.dest}")
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
