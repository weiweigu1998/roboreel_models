"""Single-query attention pooling (CoCa-style; Paper §II-E, VVCL / VTCL).

Used to reduce the 64 prompt-resampler tokens to a single d-dim embedding per
video for the SigLIP losses (§II-E (3) and (4)).
"""

from __future__ import annotations

import torch
import torch.nn as nn

from .perceiver import CrossAttention, FeedForward


class AttentionPool(nn.Module):
    def __init__(self, dim: int = 768, num_heads: int = 8) -> None:
        super().__init__()
        self.query = nn.Parameter(torch.randn(1, dim) * 0.02)
        self.cross_attn = CrossAttention(dim, num_heads, dim // num_heads)
        self.ff = FeedForward(dim)
        self.norm = nn.LayerNorm(dim)

    def forward(self, tokens: torch.Tensor) -> torch.Tensor:
        """``tokens``: (B, N, D) -> (B, D)."""
        B = tokens.shape[0]
        q = self.query.unsqueeze(0).expand(B, -1, -1)
        q = q + self.cross_attn(q, tokens)
        q = q + self.ff(q)
        return self.norm(q.squeeze(1))
