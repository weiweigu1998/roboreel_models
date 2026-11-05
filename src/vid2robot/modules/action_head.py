"""Action position embeddings + per-dim 256-bin classification head.

Paper §II-C: "The size of the action position embedding is N × d ... Each
output token of the action decoder corresponds to an action dimension ...
We project each token embedding to 256 dimensions, and a softmax layer is
applied on the top to obtain the bin."

For chunked prediction (chunk_size=4, ACT-style) we use ``chunk_size *
action_dim`` learnable queries; the head then folds them back to
``(chunk, action_dim, num_bins)``.
"""

from __future__ import annotations

import torch
import torch.nn as nn


class ActionHead(nn.Module):
    def __init__(self, dim: int = 768, action_dim: int = 8,
                 chunk_size: int = 4, num_bins: int = 256) -> None:
        super().__init__()
        self.action_dim = action_dim
        self.chunk_size = chunk_size
        self.num_bins = num_bins
        n_queries = chunk_size * action_dim
        self.position_embeddings = nn.Parameter(
            torch.randn(n_queries, dim) * 0.02)
        self.head = nn.Linear(dim, num_bins)

    def queries(self, batch_size: int) -> torch.Tensor:
        return self.position_embeddings.unsqueeze(0).expand(batch_size, -1, -1)

    def decode(self, decoded: torch.Tensor) -> torch.Tensor:
        """``decoded``: (B, chunk*action_dim, D) -> logits (B, chunk, action_dim, num_bins)."""
        B = decoded.shape[0]
        logits = self.head(decoded)  # (B, chunk*action_dim, num_bins)
        return logits.reshape(B, self.chunk_size, self.action_dim, self.num_bins)
