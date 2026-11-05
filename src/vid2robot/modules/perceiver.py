"""Perceiver Resampler (Paper §II-C, Table IV).

Compresses a long flat sequence of patch tokens (16x196 prompt or 8x196 state)
into a fixed set of 64 latent tokens. Two layers, 12 heads, d=768, with the
``Flamingo``-style design: cross-attn(latents <- concat([video, latents])) +
self-attn(latents) + FFN.
"""

from __future__ import annotations

import torch
import torch.nn as nn


class CrossAttention(nn.Module):
    def __init__(self, dim: int, num_heads: int, dim_head: int) -> None:
        super().__init__()
        inner = dim_head * num_heads
        self.num_heads = num_heads
        self.scale = dim_head ** -0.5
        self.norm_q = nn.LayerNorm(dim)
        self.norm_kv = nn.LayerNorm(dim)
        self.to_q = nn.Linear(dim, inner, bias=False)
        self.to_kv = nn.Linear(dim, inner * 2, bias=False)
        self.to_out = nn.Linear(inner, dim, bias=False)

    def forward(self, q: torch.Tensor, kv: torch.Tensor) -> torch.Tensor:
        h = self.num_heads
        q_in = self.norm_q(q)
        kv_in = self.norm_kv(kv)
        Q = self.to_q(q_in)
        K, V = self.to_kv(kv_in).chunk(2, dim=-1)
        B, Nq, _ = Q.shape
        Nk = K.shape[1]
        Q = Q.reshape(B, Nq, h, -1).transpose(1, 2)
        K = K.reshape(B, Nk, h, -1).transpose(1, 2)
        V = V.reshape(B, Nk, h, -1).transpose(1, 2)
        attn = torch.softmax((Q @ K.transpose(-1, -2)) * self.scale, dim=-1)
        out = (attn @ V).transpose(1, 2).reshape(B, Nq, -1)
        return self.to_out(out)


class FeedForward(nn.Module):
    def __init__(self, dim: int, mlp_ratio: float = 4.0) -> None:
        super().__init__()
        hidden = int(dim * mlp_ratio)
        self.net = nn.Sequential(
            nn.LayerNorm(dim),
            nn.Linear(dim, hidden),
            nn.GELU(),
            nn.Linear(hidden, dim),
        )

    def forward(self, x: torch.Tensor) -> torch.Tensor:
        return self.net(x)


class PerceiverResamplerLayer(nn.Module):
    def __init__(self, dim: int, num_heads: int, dim_head: int,
                 mlp_ratio: float) -> None:
        super().__init__()
        self.xattn = CrossAttention(dim, num_heads, dim_head)
        self.ff = FeedForward(dim, mlp_ratio)

    def forward(self, latents: torch.Tensor, video: torch.Tensor
                ) -> torch.Tensor:
        # Flamingo trick: keys/values are the concat of video tokens and the
        # latents themselves, so the latents can read both new evidence and
        # their cross-latent state.
        kv = torch.cat([video, latents], dim=1)
        latents = latents + self.xattn(latents, kv)
        latents = latents + self.ff(latents)
        return latents


class PerceiverResampler(nn.Module):
    """Compresses ``(B, N, D)`` patch tokens into ``(B, num_latents, D)``."""

    def __init__(self, dim: int = 768, num_latents: int = 64,
                 num_layers: int = 2, num_heads: int = 12,
                 dim_head: int = 64, mlp_ratio: float = 4.0) -> None:
        super().__init__()
        self.latents = nn.Parameter(torch.randn(num_latents, dim) * 0.02)
        self.layers = nn.ModuleList([
            PerceiverResamplerLayer(dim, num_heads, dim_head, mlp_ratio)
            for _ in range(num_layers)
        ])
        self.norm = nn.LayerNorm(dim)

    def forward(self, video_tokens: torch.Tensor) -> torch.Tensor:
        B = video_tokens.shape[0]
        latents = self.latents.unsqueeze(0).expand(B, -1, -1)
        for layer in self.layers:
            latents = layer(latents, video_tokens)
        return self.norm(latents)
