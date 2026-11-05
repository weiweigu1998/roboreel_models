"""Video frame sampling helpers (Paper §II-D).

* Prompt video: ``N=16`` frames, always include first + last, sort time-asc.
* Robot history: current frame + ``k-1=7`` preceding frames; left-pad with
  the first frame in the demo if there are fewer than 7 prior frames.
"""

from __future__ import annotations

import numpy as np


def sample_prompt_indices(total_frames: int, num_frames: int,
                          rng: np.random.Generator | None = None) -> list[int]:
    """Pick ``num_frames`` indices from ``[0, total_frames-1]``.

    First + last always included; remaining ``num_frames-2`` indices uniformly
    sampled (with replacement if ``total < num_frames-2``), then sorted.
    """
    if total_frames <= 0:
        raise ValueError(f"total_frames must be > 0, got {total_frames}")
    if num_frames < 2:
        raise ValueError(f"num_frames must be >= 2, got {num_frames}")
    if total_frames == 1:
        return [0] * num_frames
    if rng is None:
        rng = np.random.default_rng()
    inner_pool = total_frames - 2  # frames strictly between first and last
    need = num_frames - 2
    if inner_pool <= 0:
        middle = [rng.integers(0, total_frames) for _ in range(need)]
    else:
        replace = need > inner_pool
        choice = rng.choice(inner_pool, size=need, replace=replace)
        middle = [1 + int(c) for c in choice]
    idxs = [0] + sorted(middle) + [total_frames - 1]
    return idxs


def sample_state_indices(demo_length: int, current_t: int, num_frames: int
                         ) -> list[int]:
    """Return ``num_frames`` indices ending at ``current_t``; left-pad with
    index 0 when there aren't enough prior frames."""
    if demo_length <= 0:
        raise ValueError(f"demo_length must be > 0, got {demo_length}")
    if current_t < 0 or current_t >= demo_length:
        raise ValueError(f"current_t={current_t} out of range [0,{demo_length})")
    start = current_t - (num_frames - 1)
    out: list[int] = []
    for i in range(start, current_t + 1):
        out.append(max(i, 0))
    return out
