"""Cross-attention transformer used by the State-Prompt Encoder and the Action
Decoder (Paper §II-C, Table IV).

Each block = cross-attn(Q=stream, KV=context) -> self-attn(stream) -> FFN.
4 layers, d=768, 8 heads, MLP ratio 4 (the standard ViT FFN ratio).
"""

from __future__ import annotations

import torch
import torch.nn as nn

from .perceiver import CrossAttention, FeedForward


class SelfAttention(nn.Module):
    def __init__(self, dim: int, num_heads: int) -> None:
        super().__init__()
        self.num_heads = num_heads
        self.dim_head = dim // num_heads
        self.scale = self.dim_head ** -0.5
        self.norm = nn.LayerNorm(dim)
        self.qkv = nn.Linear(dim, dim * 3, bias=False)
        self.out = nn.Linear(dim, dim, bias=False)

    def forward(self, x: torch.Tensor) -> torch.Tensor:
        B, N, D = x.shape
        h = self.num_heads
        qkv = self.qkv(self.norm(x)).reshape(B, N, 3, h, self.dim_head)
        Q, K, V = qkv.permute(2, 0, 3, 1, 4)
        attn = torch.softmax((Q @ K.transpose(-1, -2)) * self.scale, dim=-1)
        out = (attn @ V).transpose(1, 2).reshape(B, N, D)
        return self.out(out)


class CrossAttnTransformerBlock(nn.Module):
    """xattn(Q=stream, KV=context) -> self-attn(stream) -> FFN, pre-norm."""

    def __init__(self, dim: int, num_heads: int, mlp_ratio: float = 4.0
                 ) -> None:
        super().__init__()
        dim_head = dim // num_heads
        self.cross_attn = CrossAttention(dim, num_heads, dim_head)
        self.self_attn = SelfAttention(dim, num_heads)
        self.ff = FeedForward(dim, mlp_ratio)

    def forward(self, stream: torch.Tensor, context: torch.Tensor
                ) -> torch.Tensor:
        stream = stream + self.cross_attn(stream, context)
        stream = stream + self.self_attn(stream)
        stream = stream + self.ff(stream)
        return stream


class CrossAttnTransformer(nn.Module):
    def __init__(self, dim: int = 768, num_layers: int = 4, num_heads: int = 8,
                 mlp_ratio: float = 4.0) -> None:
        super().__init__()
        self.layers = nn.ModuleList([
            CrossAttnTransformerBlock(dim, num_heads, mlp_ratio)
            for _ in range(num_layers)
        ])
        self.norm = nn.LayerNorm(dim)

    def forward(self, stream: torch.Tensor, context: torch.Tensor
                ) -> torch.Tensor:
        for layer in self.layers:
            stream = layer(stream, context)
        return self.norm(stream)
