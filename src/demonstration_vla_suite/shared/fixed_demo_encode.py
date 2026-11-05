"""Encode a fixed demonstration trajectory into perceptual / recurrent tensors for LfO Pi0.

Uses the same token layouts as the ``DemonstrationBuffer`` stack, but as a single forward
pass over the full demo (no incremental ``add_buffer``).
"""

from __future__ import annotations

from typing import Any, Callable

import numpy as np

from demonstration_vla_suite.shared.demonstration_buffer import (
    DemonstrationBuffer,
    DemonstrationBufferRecurrent,
)


def encode_pi0_demonstration_tensors(
    demonstration_cfg: Any,
    vision_enc_fn: Callable[..., Any],
    images: np.ndarray,
    states: np.ndarray,
) -> dict[str, np.ndarray]:
    """Build ``static_*`` or ``recur_*`` arrays for ``HistAugObservation`` / ``DemoVLAInputs``.

    Args:
        demonstration_cfg: Loaded demonstration YAML (``representation_type``, etc.).
        vision_enc_fn: JIT vision encoder ``(T, V, H, W, C) -> embeddings``.
        images: ``(T, num_views, 224, 224, 3)`` uint8.
        states: ``(T, state_dim)`` float32 (robot state per demo frame).

    Returns:
        Dict with either ``static_image_emb``, ``static_pos_emb``, ``static_state_emb``,
        ``static_mask`` or ``recur_image_emb``, ``recur_pos_emb``, ``recur_state_emb``,
        ``recur_mask``.
    """
    t = images.shape[0]
    step_indices = list(range(t))
    last_idx = t - 1

    if demonstration_cfg.representation_type == "recurrent":
        buf = DemonstrationBufferRecurrent(
            num_views=demonstration_cfg.num_views,
            img_emb_dim=demonstration_cfg.demonstration_feature.img.input_dim,
            pos_emb_dim=demonstration_cfg.demonstration_feature.pos.input_dim,
            state_emb_dim=demonstration_cfg.demonstration_feature.state.input_dim,
            input_obs_horizon=demonstration_cfg.streaming_obs_horizon,
            max_recur_steps=demonstration_cfg.recurrent_demonstration.max_recur_steps,
            max_video_steps=demonstration_cfg.recurrent_demonstration.max_pretraj_steps,
            max_steps=t,
            prepare_buffer=True,
            vision_enc_fn=vision_enc_fn,
        )
        buf.add_buffer(images, states, step_indices)
        gather = buf.default_demonstration_feats_gather_fn
        ri, rp, rs, rm = buf.prepare_token_recurrent(last_idx, 0, gather)
        return {
            "recur_image_emb": ri,
            "recur_pos_emb": rp,
            "recur_state_emb": rs,
            "recur_mask": rm,
        }

    if demonstration_cfg.representation_type != "perceptual":
        raise ValueError(f"Unsupported representation_type: {demonstration_cfg.representation_type}")

    buf = DemonstrationBuffer(
        num_views=demonstration_cfg.num_views,
        img_emb_dim=demonstration_cfg.demonstration_feature.img.input_dim,
        pos_emb_dim=demonstration_cfg.demonstration_feature.pos.input_dim,
        state_emb_dim=demonstration_cfg.demonstration_feature.state.input_dim,
        max_steps=t,
        compute_token_drop_score=str(demonstration_cfg.demonstration_sampling_type) == "token_dropping",
        token_drop_stride=demonstration_cfg.streaming_obs_horizon // 2,
        prepare_buffer=True,
        vision_enc_fn=vision_enc_fn,
    )
    buf.add_buffer(images, states, step_indices)
    gather = buf.default_demonstration_feats_gather_fn
    budget = demonstration_cfg.budget
    if str(demonstration_cfg.demonstration_sampling_type) == "token_dropping":
        si, sp, ss, sm = buf.prepare_token_dropping(last_idx, budget, gather)
    else:
        tpi = demonstration_cfg.token_per_image
        si, sp, ss, sm = buf.prepare_frame_sampling(last_idx, budget, tpi, gather)
    return {
        "static_image_emb": si,
        "static_pos_emb": sp,
        "static_state_emb": ss,
        "static_mask": sm,
    }
