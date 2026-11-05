"""Vid2Robot (Jain et al. 2024) — PyTorch re-implementation for the LfO benchmark.

End-to-end video-conditioned policy: ViT-B/16 image encoder shared across a
16-frame prompt video and an 8-frame robot-state video, two Perceiver Resamplers
(64 latents each), a 4-layer cross-attention state-prompt encoder, and a
4-layer cross-attention action decoder with learnable position embeddings.

This port replaces the paper's 11-D Google-Robot action space with the LfO
benchmark's 8-D ``pd_joint_pos`` (7 arm joints + 1 gripper close in [-1, 1]).
chunk_size = 4 (Table V "Prediction Horizon"), 256 bins per dim.
"""
