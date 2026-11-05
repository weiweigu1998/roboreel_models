import flax.nnx as nnx
import jax.numpy as jnp

import openpi.shared.array_typing as at
from demonstration_vla_suite.models.representation.feature_encoder import FeatureEncoder


class PerceptualDemonstrationEncoder(nnx.Module):
    def __init__(self, config, rngs: nnx.Rngs, dtype: at.DTypeLike = jnp.float32):
        self.config = config
        self.dtype = dtype

        self.feature_encoder = FeatureEncoder(
            rngs=rngs,
            dtype=dtype,
            image_input_dim=self.config.demonstration_feature.img.input_dim,
            pos_input_dim=self.config.demonstration_feature.pos.input_dim,
            state_input_dim=self.config.demonstration_feature.state.input_dim,
            pos_output_dim=self.config.demonstration_feature.pos.hidden_dim,
            state_output_dim=self.config.demonstration_feature.state.hidden_dim,
            ouput_dim_for_recur=None,
            output_dim_for_percep=self.config.demonstration_token_dim,
            use_pos_emb=self.config.use_pos_emb,
            use_state_emb=self.config.use_state_emb,
        )

    def __call__(
        self,
        static_image_emb: at.Float[at.Array, "b l d1"],
        static_pos_emb: at.Float[at.Array, "b l d2"],
        static_state_emb: at.Float[at.Array, "b l d3"],
    ):
        assert static_image_emb.shape[1] == self.config.budget
        hidden_states = self.feature_encoder.encode_perceptual_demonstration(
            static_image_emb, static_pos_emb, static_state_emb
        )
        return hidden_states, None, None
