"""Temporal Cycle Consistency loss (Dwibedi et al. 2019).

Paper §II-E (2). For each frame t of the prompt video, compute a soft-neighbor
~E^t_{p->r} in the robot embeddings via softmax over -||E_p^t - E_r^k||^2;
then cycle back to find ^E^t_{p->r->p} and minimize (^E^t - t)^2. Applied
symmetrically (prompt->robot->prompt and robot->prompt->robot).

Inputs are per-frame video embeddings ``E_p (B, T_p, D)`` and ``E_r (B, T_r,
D)`` produced by spatial-mean-pooling the ViT outputs and applying the 2-layer
MLP projector ``Phi`` (paper).
"""

from __future__ import annotations

import torch
import torch.nn as nn


class AlignmentProjector(nn.Module):
    """2-layer MLP "alignment pooling layer" Phi (paper §II-E (2))."""

    def __init__(self, in_dim: int = 768, hidden_dim: int = 256,
                 out_dim: int = 128) -> None:
        super().__init__()
        self.net = nn.Sequential(
            nn.Linear(in_dim, hidden_dim),
            nn.ReLU(inplace=True),
            nn.Linear(hidden_dim, out_dim),
        )

    def forward(self, frame_embeds: torch.Tensor) -> torch.Tensor:
        """``frame_embeds``: (B, T, D) -> (B, T, out_dim)."""
        return self.net(frame_embeds)


def _cycle_once(E_a: torch.Tensor, E_b: torch.Tensor) -> torch.Tensor:
    """Per-sample TCC half-cycle: a -> b -> a, return mean (cycled_idx - t)^2.

    ``E_a``, ``E_b``: (T_a, D) / (T_b, D) for a single video pair.
    """
    Ta = E_a.shape[0]
    Tb = E_b.shape[0]
    # Negative squared distance from each E_a^t to every E_b^k.
    d_ab = -torch.cdist(E_a, E_b).pow(2)  # (T_a, T_b)
    alpha = torch.softmax(d_ab, dim=-1)
    tilde = alpha @ E_b  # (T_a, D) soft-neighbour in E_b

    # Cycle back: distance from tilde to every E_a^j.
    d_ba = -torch.cdist(tilde, E_a).pow(2)  # (T_a, T_a)
    beta = torch.softmax(d_ba, dim=-1)
    idxs = torch.arange(Ta, device=E_a.device, dtype=E_a.dtype)
    cycled = (beta * idxs.unsqueeze(0)).sum(dim=-1)  # (T_a,) recovered index
    return ((cycled - idxs) / max(Ta - 1, 1)).pow(2).mean()


def tcc_loss(E_p: torch.Tensor, E_r: torch.Tensor) -> torch.Tensor:
    """Symmetric TCC over a batch of paired prompt and robot per-frame embeddings.

    ``E_p``: (B, T_p, D), ``E_r``: (B, T_r, D). Returns scalar mean loss.
    """
    B = E_p.shape[0]
    losses = []
    for i in range(B):
        l1 = _cycle_once(E_p[i], E_r[i])
        l2 = _cycle_once(E_r[i], E_p[i])
        losses.append((l1 + l2) * 0.5)
    return torch.stack(losses).mean()
