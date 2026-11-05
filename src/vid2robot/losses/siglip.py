"""SigLIP loss (Zhai et al. 2023) used for VVCL (§II-E (3)) and VTCL (§II-E (4)).

Implements the form in the paper:
    Y = 2 I_B - 1
    logits = (Z1 @ Z2.T) * t + b
    L = -mean( log(sigmoid(Y * logits)) )

with learnable scalar ``t`` (paper uses ``temperature`` symbol; we keep its
``log_t`` exponentiated like upstream SigLIP) and bias ``b``.
"""

from __future__ import annotations

import math

import torch
import torch.nn as nn
import torch.nn.functional as F


class SigLIPLoss(nn.Module):
    def __init__(self, init_temperature: float = 10.0,
                 init_bias: float = -10.0) -> None:
        super().__init__()
        # ``log_t`` is the learnable parameter; t = exp(log_t).
        self.log_t = nn.Parameter(torch.tensor(math.log(init_temperature)))
        self.bias = nn.Parameter(torch.tensor(float(init_bias)))

    def forward(self, Z1: torch.Tensor, Z2: torch.Tensor) -> torch.Tensor:
        """``Z1``, ``Z2``: (B, D). Returns scalar loss.

        Diagonal pairs are positives (label +1), all other off-diagonal pairs
        are negatives (label -1). L2-normalize before the dot product to mirror
        the upstream SigLIP recipe.
        """
        z1 = F.normalize(Z1, dim=-1)
        z2 = F.normalize(Z2, dim=-1)
        logits = (z1 @ z2.t()) * self.log_t.exp() + self.bias
        B = z1.shape[0]
        Y = 2.0 * torch.eye(B, device=z1.device, dtype=z1.dtype) - 1.0
        # SigLIP loss: -log sigmoid(Y * logits) averaged over all B*B pairs.
        return -F.logsigmoid(Y * logits).mean()
