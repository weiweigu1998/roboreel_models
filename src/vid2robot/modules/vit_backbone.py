"""Shared ViT-B/16 backbone (Paper §II-C, Table IV).

Wraps timm's ``vit_base_patch16_224`` and exposes per-patch tokens (no CLS) for
both the prompt and state encoders. Weights are shared between the two streams
(paper: "we use the same image encoder weights for both").

Input:  (B, T, H, W, C) uint8/float32 video in [0, 1] (model normalizes
        internally with the ViT's ImageNet mean/std).
Output: (B, T, 196, 768) patch features.
"""

from __future__ import annotations

import torch
import torch.nn as nn


class ViTBackbone(nn.Module):
    def __init__(self, model_name: str = "vit_base_patch16_224",
                 pretrained: bool = True, hidden_dim: int = 768) -> None:
        super().__init__()
        import timm
        # ``num_classes=0, global_pool=''`` keeps timm's per-token output.
        self.vit = timm.create_model(model_name, pretrained=pretrained,
                                     num_classes=0, global_pool="")
        # ImageNet-mean/std from timm's default cfg — store as buffers so the
        # normalization is part of the module and tracks the device.
        mean = self.vit.default_cfg.get("mean", (0.485, 0.456, 0.406))
        std = self.vit.default_cfg.get("std", (0.229, 0.224, 0.225))
        self.register_buffer("mean", torch.tensor(mean).view(1, 3, 1, 1))
        self.register_buffer("std", torch.tensor(std).view(1, 3, 1, 1))
        self.hidden_dim = hidden_dim

    def forward(self, video: torch.Tensor) -> torch.Tensor:
        """``video``: (B, T, H, W, C) float in [0,1] -> (B, T, 196, 768)."""
        B, T, H, W, C = video.shape
        x = video.reshape(B * T, H, W, C).permute(0, 3, 1, 2).contiguous()  # (BT, C, H, W)
        x = (x - self.mean) / self.std
        feats = self.vit.forward_features(x)  # (BT, 197, 768) — CLS + 196 patches
        if feats.ndim != 3:
            raise RuntimeError(
                f"timm ViT returned shape {feats.shape}; expected (BT, N+1, D).")
        # Strip CLS token, keep 196 patches.
        patches = feats[:, 1:, :]
        return patches.reshape(B, T, patches.shape[1], patches.shape[2])
