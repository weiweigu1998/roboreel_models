"""Vid2Robot end-to-end policy (PyTorch).

Wires the four modules (Paper §II-C / Fig. 3):
  1. Prompt Encoder  — shared ViT-B/16 + Perceiver Resampler psi_p -> 64 tokens
  2. State Encoder   — same ViT (tied) + Perceiver Resampler psi_s -> 64 tokens
  3. State-Prompt Encoder — 4-layer xattn(Q=state, KV=prompt)
  4. Action Decoder       — 4-layer xattn(Q=pos-emb, KV=state|prompt) + 256-bin head

plus side heads for the auxiliary losses (Paper §II-E):
  * AlignmentProjector + per-frame mean-pooled ViT features for TCC
  * AttentionPool over the prompt-resampled tokens for SigLIP video embeddings
"""

from __future__ import annotations

from dataclasses import dataclass

import torch
import torch.nn as nn

from .config import Vid2RobotConfig
from .losses.tcc import AlignmentProjector
from .modules.action_head import ActionHead
from .modules.attention_pool import AttentionPool
from .modules.cross_attn import CrossAttnTransformer
from .modules.perceiver import PerceiverResampler
from .modules.vit_backbone import ViTBackbone


@dataclass
class Vid2RobotForward:
    """Container for everything a single forward returns.

    Keeping the auxiliary signals as named fields keeps the training loop
    explicit about which loss is consuming which signal.
    """
    # Main action prediction.
    action_logits: torch.Tensor  # (B, chunk, action_dim, num_bins)
    # Per-frame projected features for TCC.
    prompt_frame_proj: torch.Tensor  # (B, T_prompt, proj_dim)
    robot_frame_proj: torch.Tensor   # (B, T_robot, proj_dim) — for the *full-clip* robot video
    # Video-level embeddings for SigLIP (VVCL + VTCL).
    prompt_video_embed: torch.Tensor  # (B, D)
    robot_video_embed: torch.Tensor   # (B, D)


class Vid2Robot(nn.Module):
    def __init__(self, cfg: Vid2RobotConfig) -> None:
        super().__init__()
        self.cfg = cfg

        # --- shared ViT backbone (paper: same weights for prompt & state) ---
        self.vit = ViTBackbone(model_name=cfg.vit_name,
                               pretrained=cfg.vit_pretrained,
                               hidden_dim=cfg.vit_hidden_dim)
        if cfg.gradient_checkpointing:
            # timm's ViT block-wise checkpointing — recomputes per-block
            # activations during backward to fit batch 16 on a single 80 GB A100.
            self.vit.vit.set_grad_checkpointing(True)

        # --- two separate Perceiver Resamplers (same shape) ---
        self.prompt_resampler = PerceiverResampler(
            dim=cfg.vit_hidden_dim,
            num_latents=cfg.resampler_num_latents,
            num_layers=cfg.resampler_num_layers,
            num_heads=cfg.resampler_num_heads,
            dim_head=cfg.resampler_dim_head,
            mlp_ratio=cfg.resampler_mlp_ratio,
        )
        self.state_resampler = PerceiverResampler(
            dim=cfg.vit_hidden_dim,
            num_latents=cfg.resampler_num_latents,
            num_layers=cfg.resampler_num_layers,
            num_heads=cfg.resampler_num_heads,
            dim_head=cfg.resampler_dim_head,
            mlp_ratio=cfg.resampler_mlp_ratio,
        )

        # --- 4-layer cross-attention fusion / decoder ---
        self.state_prompt_encoder = CrossAttnTransformer(
            dim=cfg.transformer_dim,
            num_layers=cfg.state_prompt_layers,
            num_heads=cfg.transformer_num_heads,
            mlp_ratio=cfg.transformer_mlp_ratio,
        )
        self.action_decoder = CrossAttnTransformer(
            dim=cfg.transformer_dim,
            num_layers=cfg.action_decoder_layers,
            num_heads=cfg.transformer_num_heads,
            mlp_ratio=cfg.transformer_mlp_ratio,
        )
        self.action_head = ActionHead(
            dim=cfg.transformer_dim,
            action_dim=cfg.action_dim,
            chunk_size=cfg.chunk_size,
            num_bins=cfg.num_bins,
        )

        # --- side heads for auxiliary losses ---
        self.alignment_projector = AlignmentProjector(
            in_dim=cfg.vit_hidden_dim,
            hidden_dim=cfg.tcc_proj_hidden_dim,
            out_dim=cfg.tcc_proj_out_dim,
        )
        # Single attention pool reused for both prompt + robot videos
        # (paper: same prompt-encoder branch is applied to both).
        self.video_attn_pool = AttentionPool(
            dim=cfg.transformer_dim,
            num_heads=cfg.transformer_num_heads,
        )

    # ---- encoders ----------------------------------------------------------

    def _frame_features(self, video: torch.Tensor) -> torch.Tensor:
        """(B, T, H, W, C) -> (B, T, 196, 768) patch tokens."""
        return self.vit(video)

    def _flat_tokens(self, frame_feats: torch.Tensor) -> torch.Tensor:
        """(B, T, N, D) -> (B, T*N, D) for the Perceiver Resampler input."""
        B, T, N, D = frame_feats.shape
        return frame_feats.reshape(B, T * N, D)

    def _mean_pool_frames(self, frame_feats: torch.Tensor) -> torch.Tensor:
        """(B, T, N, D) -> (B, T, D) per-frame spatial mean-pool (paper §II-E (2))."""
        return frame_feats.mean(dim=2)

    # ---- staged inference (used at eval time) ------------------------------

    @torch.no_grad()
    def encode_prompt(self, prompt_video: torch.Tensor) -> torch.Tensor:
        """Run the prompt branch once per demo. Returns ``z_prompt: (B, 64, 768)``.

        At eval the prompt video doesn't change during a rollout; caching this
        avoids redundant ViT-B/16 forwards on the 16-frame clip every step.
        """
        feats = self._frame_features(prompt_video)
        return self.prompt_resampler(self._flat_tokens(feats))

    @torch.no_grad()
    def predict_action_logits(self, z_prompt: torch.Tensor,
                              state_video: torch.Tensor) -> torch.Tensor:
        """Per-step path: state encoder + fusion + decoder + head.

        ``z_prompt``: (B, 64, 768) from ``encode_prompt``.
        ``state_video``: (B, T_s, H, W, C) k-step history (current frame last).
        Returns ``(B, chunk, action_dim, num_bins)`` logits.
        """
        state_feats = self._frame_features(state_video)
        z_state = self.state_resampler(self._flat_tokens(state_feats))
        z_state_prompt = self.state_prompt_encoder(z_state, z_prompt)
        B = z_state_prompt.shape[0]
        queries = self.action_head.queries(B)
        decoded = self.action_decoder(queries, z_state_prompt)
        return self.action_head.decode(decoded)

    # ---- main forward ------------------------------------------------------

    def forward(self, prompt_video: torch.Tensor, state_video: torch.Tensor,
                robot_full_video: torch.Tensor | None = None
                ) -> Vid2RobotForward:
        """Single forward pass.

        Args:
            prompt_video: (B, T_p, H, W, C) human/demonstrator prompt video.
            state_video:  (B, T_s, H, W, C) k-step robot history (current at -1).
            robot_full_video: optional (B, T_p, H, W, C) robot demo sampled at
                the same length as ``prompt_video`` — used only for VVCL/VTCL
                and TCC; if None, the auxiliary fields are filled with zeros so
                only ``action_logits`` is meaningful.
        """
        # Patch features (shared ViT).
        prompt_feats = self._frame_features(prompt_video)   # (B, T_p, 196, 768)
        state_feats = self._frame_features(state_video)     # (B, T_s, 196, 768)

        # Resamplers compress space-time tokens to 64 latents each.
        z_prompt = self.prompt_resampler(self._flat_tokens(prompt_feats))  # (B, 64, 768)
        z_state = self.state_resampler(self._flat_tokens(state_feats))    # (B, 64, 768)

        # State-Prompt fusion: state is Q, prompt is K/V.
        z_state_prompt = self.state_prompt_encoder(z_state, z_prompt)     # (B, 64, 768)

        # Action decoder: learnable queries cross-attend to state|prompt.
        B = z_state_prompt.shape[0]
        queries = self.action_head.queries(B)
        decoded = self.action_decoder(queries, z_state_prompt)
        action_logits = self.action_head.decode(decoded)  # (B, chunk, action_dim, num_bins)

        # ---- auxiliary signals ----------------------------------------------
        # Prompt-side: pool prompt resampler tokens to a single video embedding.
        prompt_video_embed = self.video_attn_pool(z_prompt)               # (B, 768)
        prompt_frame_proj = self.alignment_projector(
            self._mean_pool_frames(prompt_feats))                          # (B, T_p, P)

        if robot_full_video is not None:
            robot_feats = self._frame_features(robot_full_video)
            z_robot = self.prompt_resampler(self._flat_tokens(robot_feats))  # tied weights
            robot_video_embed = self.video_attn_pool(z_robot)
            robot_frame_proj = self.alignment_projector(
                self._mean_pool_frames(robot_feats))
        else:
            robot_video_embed = torch.zeros_like(prompt_video_embed)
            robot_frame_proj = torch.zeros_like(prompt_frame_proj)

        return Vid2RobotForward(
            action_logits=action_logits,
            prompt_frame_proj=prompt_frame_proj,
            robot_frame_proj=robot_frame_proj,
            prompt_video_embed=prompt_video_embed,
            robot_video_embed=robot_video_embed,
        )
