"""Vid2Robot model configuration.

Defaults follow Tables IV and V of the paper, with the LfO action-space
adaptation: 8-D pd_joint_pos (drop the paper's mode + base dims).
"""

from __future__ import annotations

from dataclasses import dataclass, field


@dataclass
class Vid2RobotConfig:
    # --- video shapes (Table V) ---
    image_size: int = 224
    num_prompt_frames: int = 16
    num_state_frames: int = 8
    in_channels: int = 3

    # --- ViT-B/16 backbone (Table IV) ---
    vit_name: str = "vit_base_patch16_224"
    vit_pretrained: bool = True
    vit_hidden_dim: int = 768
    vit_num_patches: int = 196  # 224/16 = 14, 14*14 = 196

    # --- Perceiver Resamplers (Table IV) ---
    resampler_num_latents: int = 64
    resampler_num_layers: int = 2
    resampler_num_heads: int = 12
    resampler_dim_head: int = 64
    resampler_mlp_ratio: float = 4.0

    # --- State-Prompt Encoder + Action Decoder (Table IV) ---
    transformer_dim: int = 768
    transformer_num_heads: int = 8
    state_prompt_layers: int = 4
    action_decoder_layers: int = 4
    transformer_mlp_ratio: float = 4.0

    # --- action head ---
    # LfO benchmark uses pd_joint_pos: 7 arm joints + 1 gripper close.
    action_dim: int = 8
    chunk_size: int = 4
    num_bins: int = 256

    # --- losses ---
    use_tcc: bool = True
    use_vvcl: bool = True
    use_vtcl: bool = True
    text_embed_dim: int = 768  # SigLIP text-tower output dim
    tcc_proj_hidden_dim: int = 256
    tcc_proj_out_dim: int = 128
    siglip_init_temperature: float = 10.0  # SigLIP defaults; init log_t = log(10)
    siglip_init_bias: float = -10.0

    # --- training (Table V) ---
    seed: int = 42
    # If True, enable timm's per-block gradient checkpointing on the ViT
    # backbone. Roughly halves activation memory at the cost of ~30% slower
    # backward — needed to fit batch 32 on a single 80 GB A100.
    gradient_checkpointing: bool = False

    # Per-task task-text strings used by the text encoder for VTCL.
    task_text_template: str = "{task}"

    def __post_init__(self) -> None:
        # Sanity: ViT-B/16 at 224 -> 14x14 = 196 patches.
        side = self.image_size // 16
        if side * side != self.vit_num_patches:
            raise ValueError(
                f"image_size={self.image_size} (patch16) -> {side*side} patches, "
                f"but vit_num_patches={self.vit_num_patches}.")
