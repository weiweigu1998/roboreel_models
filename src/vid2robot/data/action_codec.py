"""Per-dim action scaling and 256-bin discretization (Paper §II-D).

The paper scales each action dim to [0,1] using its observed range, then bins
into 256. LfO benchmark stores 8-D ``pd_joint_pos`` actions (7 arm joints
in ~[-3, 3] + 1 gripper close in [-1, 1]).

``ActionCodec`` holds per-dim ``low``/``high`` and provides ``encode`` /
``decode`` that map between raw actions and integer bin indices.
"""

from __future__ import annotations

from dataclasses import dataclass

import numpy as np


@dataclass
class ActionCodec:
    low: np.ndarray   # (action_dim,) float32
    high: np.ndarray  # (action_dim,) float32
    num_bins: int = 256

    def __post_init__(self) -> None:
        if self.low.shape != self.high.shape:
            raise ValueError("low/high shapes must match")
        if (self.high <= self.low).any():
            raise ValueError(f"need high > low per-dim; got low={self.low} high={self.high}")

    @property
    def action_dim(self) -> int:
        return int(self.low.shape[0])

    def encode(self, actions: np.ndarray) -> np.ndarray:
        """(..., action_dim) raw -> (..., action_dim) int64 bin indices."""
        a = np.asarray(actions, dtype=np.float32)
        scaled = (a - self.low) / (self.high - self.low)
        scaled = np.clip(scaled, 0.0, 1.0)
        bins = np.floor(scaled * self.num_bins).astype(np.int64)
        return np.clip(bins, 0, self.num_bins - 1)

    def decode(self, bins: np.ndarray) -> np.ndarray:
        """Inverse: bin center -> raw action value."""
        b = np.asarray(bins, dtype=np.float32)
        scaled = (b + 0.5) / self.num_bins
        return self.low + scaled * (self.high - self.low)

    def to_dict(self) -> dict:
        return {
            "low": self.low.tolist(),
            "high": self.high.tolist(),
            "num_bins": int(self.num_bins),
        }

    @classmethod
    def from_dict(cls, d: dict) -> "ActionCodec":
        return cls(
            low=np.asarray(d["low"], dtype=np.float32),
            high=np.asarray(d["high"], dtype=np.float32),
            num_bins=int(d.get("num_bins", 256)),
        )


def fit_codec_from_samples(actions: np.ndarray, num_bins: int = 256,
                           margin: float = 0.05) -> ActionCodec:
    """Build a codec from a stack of observed actions ``(N, action_dim)``.

    Adds a small relative ``margin`` outside the observed [min, max] so a
    handful of unseen extremes still bin cleanly. For dims that look like
    fixed-range tokens (e.g. gripper at [-1, 1]) the margin is harmless.
    """
    a = np.asarray(actions, dtype=np.float32)
    lo = a.min(axis=0)
    hi = a.max(axis=0)
    span = hi - lo
    pad = np.maximum(span * margin, 1e-3)
    return ActionCodec(low=(lo - pad).astype(np.float32),
                       high=(hi + pad).astype(np.float32),
                       num_bins=num_bins)
