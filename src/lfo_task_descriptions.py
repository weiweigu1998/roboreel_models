"""Centralized natural-language task description resolution for LfO baselines.

The pkl/index ``task`` field is the snake-case task key that
``process_training_robot_trajectories.py`` wrote. Consumers should NOT rely on that
string as their training prompt — instead they resolve the natural-language
sentence at runtime from the curated source-of-truth file:

  <data_root>/ground_truth_natural_language_annotation.json
      {task_key: "Curated natural-language command."}     (10 entries)

This module is intentionally tiny and dependency-free (stdlib only) so it can
be imported under any per-baseline conda env (vid2robot, keyframe, …).
"""

from __future__ import annotations

import json
import os
from functools import lru_cache

#: Default source-of-truth path; callers override it via the ``path`` argument.
GT_ANNOTATION_PATH = os.path.join(
    os.environ.get("ROBOREEL_DATA", ""),
    "ground_truth_natural_language_annotation.json")


@lru_cache(maxsize=4)
def load_gt_annotations(path: str = GT_ANNOTATION_PATH) -> dict[str, str]:
    """``{task_key: gt_sentence}`` from the curated ground-truth file.

    Returns an empty dict if the file is missing, so callers running without
    the dataset mounted still work.
    """
    if not os.path.isfile(path):
        return {}
    with open(path) as f:
        return dict(json.load(f))
