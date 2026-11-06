"""Where the published RoboReel dataset keeps its raw data.

    human_demos/{train,eval}/{no_distraction,with_distraction}/<task>/<demo>/<camera>.mp4
    robot_trajectories/{no_distraction,with_distraction}/<task>/<demo>/

Every script and runner reads raw data through these helpers. The code names the
clean bucket ``without_distraction``: the environment's ``init_states``, the
released checkpoints' ``train_config.json`` and the derived artifact trees all
use that name, so it is translated only here, at the raw-data boundary.

Dependency-free so every conda environment can import it.
"""

from __future__ import annotations

from pathlib import Path

HUMAN_DEMOS_SUBDIR = "human_demos"
ROBOT_TRAJECTORIES_SUBDIR = "robot_trajectories"
HUMAN_SPLITS = ("train", "eval")
BUCKETS = ("without_distraction", "with_distraction")

_TO_PUBLISHED = {"without_distraction": "no_distraction"}
_FROM_PUBLISHED = {v: k for k, v in _TO_PUBLISHED.items()}


def published_bucket(bucket: str) -> str:
    """The dataset's directory name for a code-side bucket name."""
    return _TO_PUBLISHED.get(bucket, bucket)


def iter_human_demos(data_root, bucket: str, tasks=None) -> list[tuple[str, str, Path]]:
    """``[(task, demo_id, demo_dir), ...]`` for one bucket, across both splits.

    Splits that were not downloaded are skipped, so this also works on the
    ``eval`` subset of the dataset.
    """
    out: list[tuple[str, str, Path]] = []
    for split in HUMAN_SPLITS:
        root = Path(data_root) / HUMAN_DEMOS_SUBDIR / split / published_bucket(bucket)
        if not root.is_dir():
            continue
        for task_dir in root.iterdir():
            if not task_dir.is_dir() or (tasks is not None and task_dir.name not in tasks):
                continue
            out.extend((task_dir.name, d.name, d) for d in task_dir.iterdir() if d.is_dir())
    return sorted(out, key=lambda e: (e[0], e[1]))


def human_demo_dir(data_root, bucket: str, task: str, demo_id) -> Path:
    """Directory of one human demo, whichever split holds it."""
    demo = demo_id if isinstance(demo_id, str) else f"{int(demo_id):04d}"
    candidates = [
        Path(data_root) / HUMAN_DEMOS_SUBDIR / split / published_bucket(bucket) / task / demo
        for split in HUMAN_SPLITS
    ]
    for cand in candidates:
        if cand.is_dir():
            return cand
    raise FileNotFoundError(f"Human demo not found; looked in {[str(c) for c in candidates]}")


def identify_human_video(path) -> tuple[str, str, str]:
    """``(bucket, task, demo_id)`` of ``.../human_demos/<split>/<bucket>/<task>/<demo>/<cam>.mp4``."""
    parts = Path(path).parts
    if len(parts) < 6 or parts[-6] != HUMAN_DEMOS_SUBDIR or parts[-5] not in HUMAN_SPLITS:
        raise ValueError(
            f"{path} is not under {HUMAN_DEMOS_SUBDIR}/<split>/<bucket>/<task>/<demo>/<camera>.mp4")
    published, task, demo_id = parts[-4], parts[-3], parts[-2]
    return _FROM_PUBLISHED.get(published, published), task, demo_id


def robot_trajectories_root(data_root, bucket: str) -> Path:
    """``robot_trajectories/<bucket>/``, holding ``<task>/<demo>/``."""
    return Path(data_root) / ROBOT_TRAJECTORIES_SUBDIR / published_bucket(bucket)
