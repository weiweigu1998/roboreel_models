import flax.nnx as nnx
import jax
import jax.numpy as jnp
import einops
import openpi.shared.array_typing as at
import openpi.training.sharding as sharding
from functools import partial
from jax import vmap
from demonstration_vla_suite.models.representation.utils import (
    kernel_init,
    kernel_init_out_proj_rmt,
    precompute_freqs_cis,
    apply_rotary_emb,
)


class RMTLayer(nnx.Module):
    def __init__(self, config, rngs: nnx.Rngs):
        self.config = config
        self.demonstration_slots = config.budget
        self.hidden_dim = config.recurrent_demonstration.hidden_dim
        self.num_heads = config.recurrent_demonstration.rmt_config.num_attn_heads
        self.num_kv_heads = config.recurrent_demonstration.rmt_config.num_kv_heads
        self.mini_batch_size = self.config.recurrent_demonstration.mini_batch_size
        self.head_dim = self.hidden_dim // self.num_heads

        self.dtype = dtype = jnp.float32
        self.output_stats = config.recurrent_demonstration.output_stats

        self.q_proj = nnx.Linear(self.hidden_dim, self.hidden_dim, use_bias=False, rngs=rngs, dtype=dtype, kernel_init=kernel_init)
        self.k_proj = nnx.Linear(self.hidden_dim, self.num_kv_heads * self.head_dim, use_bias=False, rngs=rngs, dtype=dtype, kernel_init=kernel_init)
        self.v_proj = nnx.Linear(self.hidden_dim, self.num_kv_heads * self.head_dim, use_bias=False, rngs=rngs, dtype=dtype, kernel_init=kernel_init)

        self.pre_norm = nnx.LayerNorm(self.hidden_dim, rngs=rngs, dtype=dtype)
        self.to_out = nnx.Linear(self.hidden_dim, self.hidden_dim, use_bias=False, rngs=rngs, dtype=dtype, kernel_init=kernel_init_out_proj_rmt)

        self.demonstration_state = nnx.Param(
            jax.random.normal(rngs.params(), (self.demonstration_slots, self.hidden_dim))
            * jnp.sqrt(1 / (self.hidden_dim)),
        )

        self.post_norm = nnx.LayerNorm(self.hidden_dim, rngs=rngs, dtype=dtype)
        self.freqs_cis = nnx.Variable(
            precompute_freqs_cis(self.head_dim, (self.mini_batch_size + self.demonstration_slots) * 2),
        )

    def __call__(self, hidden_states, mask, demonstration_state):
        @partial(vmap, axis_name="batch")
        def update_batch(x, mask):
            def compute_mini_batch(carry, inputs):
                x_mb = inputs["X"]
                mask_mb = inputs["mask"]
                carry_new, stats_dict = self.process_mini_batch(x_mb, mask_mb, carry)
                return carry_new, stats_dict

            inputs = {"X": x, "mask": mask}
            demonstration_tokens, stats_dict = jax.lax.scan(f=compute_mini_batch, init=demonstration_state, xs=inputs)
            return demonstration_tokens, stats_dict
        demonstration_state_new, stats_dict = update_batch(hidden_states, mask)
        return demonstration_state_new, demonstration_state_new, stats_dict

    def reset(self):
        return jax.tree.map(
            lambda x: x.value if isinstance(x, nnx.Param) else x, self.demonstration_state
        )

    def process_mini_batch(self, x_mb, mask_mb, demonstration_state):
        def cross_attention():
            demonstration_dtype = demonstration_state.dtype
            demonstration_slot_count = demonstration_state.shape[0]
            input_len = x_mb.shape[0]

            inputs = jnp.concatenate([x_mb, demonstration_state], axis=0)
            inputs = self.pre_norm(inputs)
            q_vec = self.q_proj(inputs)
            k_vec = self.k_proj(inputs)
            v_vec = self.v_proj(inputs)

            q_vec = einops.rearrange(q_vec, "s (nh d) -> s nh d", nh=self.num_heads)
            k_vec = einops.rearrange(k_vec, "s (nh d) -> s nh d", nh=self.num_kv_heads)
            v_vec = einops.rearrange(v_vec, "s (nh d) -> s nh d", nh=self.num_kv_heads)

            freqs_cis = jnp.take(self.freqs_cis.value, jnp.arange(q_vec.shape[0]), axis=0)
            q_vec, k_vec = apply_rotary_emb(q_vec[None, :], k_vec[None, :], freqs_cis)

            encoded = jax.nn.dot_product_attention(
                query=q_vec[0].astype(jnp.bfloat16),
                key=k_vec[0].astype(jnp.bfloat16),
                value=v_vec.astype(jnp.bfloat16),
            )

            encoded = einops.rearrange(encoded, "s nh d -> s (nh d)")
            encoded = encoded.astype(self.dtype)

            out = self.to_out(encoded[-demonstration_slot_count:])
            out = sharding.activation_sharding_constraint(out)

            if self.output_stats:
                stats_dict = {
                    "x_v_norm": jnp.linalg.norm(v_vec[:input_len], axis=-1).mean().astype(jnp.bfloat16),
                    "demonstration_v_norm": jnp.linalg.norm(v_vec[-demonstration_slot_count:], axis=-1).mean().astype(jnp.bfloat16),
                    "demonstration_in_norm": jnp.linalg.norm(demonstration_state, axis=-1).mean().astype(jnp.bfloat16),
                    "demonstration_out_norm": jnp.linalg.norm(out, axis=-1).mean().astype(jnp.bfloat16),
                    "mask": mask_mb.astype(jnp.bool_),
                }
            else:
                stats_dict = None
            new_demonstration_state = self.post_norm(out + demonstration_state)
            return new_demonstration_state.astype(demonstration_dtype), stats_dict

        def keep_original():
            if self.output_stats:
                stats_dict = {
                    "x_v_norm": jnp.array(0.0).astype(jnp.bfloat16),
                    "demonstration_v_norm": jnp.array(0.0).astype(jnp.bfloat16),
                    "demonstration_in_norm": jnp.linalg.norm(demonstration_state, axis=-1).mean().astype(jnp.bfloat16),
                    "demonstration_out_norm": jnp.array(0.0).astype(jnp.bfloat16),
                    "mask": mask_mb.astype(jnp.bool_),
                }
            else:
                stats_dict = None
            return demonstration_state, stats_dict

        demonstration_state_new, stats_dict = jax.lax.cond(mask_mb, cross_attention, keep_original)
        return demonstration_state_new, stats_dict
