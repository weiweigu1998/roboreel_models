"""
3D posemb_sincos, adapted from Qwen-2.5VL M-RoPE.

Assumes raw image patches are 16x16. Spatial indices are grid-point based,
adjusted for pooled resolutions (8x8, 4x4, 2x2). Temporal indices use
frame step positions.
"""

import jax.numpy as jnp
import einops


class PosEmb3D:
    def __init__(self, dim: int, temporal_base: int = 10_000, spatial_base: int = 1_000):
        self.dim = dim
        assert dim % 6 == 0, "dim must be divisible by 6"
        width = dim // 6

        omega = jnp.arange(width) / (width - 1)
        self.temporal_omega = 1.0 / (temporal_base ** omega)
        self.spatial_omega = 1.0 / (spatial_base ** omega)

        self.spatial_pe8x8 = self._compute_spatial_pe(8)
        self.spatial_pe4x4 = self._compute_spatial_pe(4)
        self.spatial_pe2x2 = self._compute_spatial_pe(2)

        self.temporal_pe = self._compute_temporal_pe(max_length=2048)

    def _compute_spatial_pe(self, size):
        stride = 16 // size
        offset = stride // 2
        y, x = jnp.mgrid[:size, :size]
        y = stride * y + offset
        x = stride * x + offset
        y = jnp.einsum("m,d->md", y.flatten(), self.spatial_omega)
        x = jnp.einsum("m,d->md", x.flatten(), self.spatial_omega)
        return jnp.concatenate([jnp.sin(y), jnp.cos(y), jnp.sin(x), jnp.cos(x)], axis=-1)

    def _compute_temporal_pe(self, max_length: int = 2048):
        pos = jnp.arange(max_length)
        sinusoid_input = jnp.einsum("m,d->md", pos, self.temporal_omega)
        return jnp.concatenate([jnp.sin(sinusoid_input), jnp.cos(sinusoid_input)], axis=-1)

    def __call__(self, pos, spatial_size: int):
        """
        Generate 3D positional embeddings.

        Args:
            pos: 1D array of temporal positions (frame indices)
            spatial_size: Spatial resolution (8, 4, or 2)

        Returns:
            3D positional embeddings of shape (T, H*W, D)
        """
        max_length = pos.max() + 1
        if max_length > self.temporal_pe.shape[0]:
            self.temporal_pe = self._compute_temporal_pe(max_length * 2)
        assert spatial_size in [8, 4, 2], "spatial_size must be 8, 4, or 2"

        if spatial_size == 8:
            spatial_pe = self.spatial_pe8x8
        elif spatial_size == 4:
            spatial_pe = self.spatial_pe4x4
        else:
            spatial_pe = self.spatial_pe2x2

        temporal_pe = self.temporal_pe[pos]
        temporal_pe_repeat = einops.repeat(temporal_pe, "t d -> t hw d", hw=spatial_size * spatial_size)
        spatial_pe_repeat = einops.repeat(spatial_pe, "hw d -> t hw d", t=pos.shape[0])
        pe_3d = jnp.concatenate([temporal_pe_repeat, spatial_pe_repeat], axis=-1)
        return pe_3d
