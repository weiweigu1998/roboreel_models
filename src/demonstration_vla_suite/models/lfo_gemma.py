import einops
import flax.linen as nn
import jax
import jax.numpy as jnp

from collections.abc import Sequence
from typing import Literal, TypeAlias

from openpi.models.gemma import PALIGEMMA_VOCAB_SIZE
from openpi.models.gemma import Config
from openpi.models.gemma import Embedder
from openpi.models.gemma import KVCache
from openpi.models.gemma import RMSNorm
from openpi.models.gemma import _apply_rope
from openpi.models.gemma import _gated_residual
from openpi.models.gemma import Attention
import openpi.models.lora as lora
import openpi.shared.array_typing as at
import openpi.training.sharding as sharding

from demonstration_vla_suite.models.representation.utils import kernel_init_out_proj
from demonstration_vla_suite.models.integration_utils import _name
from demonstration_vla_suite.models.integration_utils import Attention_with_DemonstrationExpert
from demonstration_vla_suite.models.integration_utils import get_config, Variant


@at.typecheck
class DemonstrationRMSNorm(nn.Module):
    @nn.compact
    def __call__(self, x, cond=None):
        dtype = x.dtype
        var = jnp.mean(jnp.square(x.astype(jnp.float32)), axis=-1, keepdims=True)
        normed_inputs = jnp.asarray(x * jnp.reciprocal(jnp.sqrt(var + 1e-06)))
        if cond is None:
            scale = self.param("scale", nn.initializers.zeros_init(), (x.shape[-1]))
            normed_inputs = normed_inputs * (1 + scale)
            return normed_inputs.astype(dtype)

        modulation = nn.Dense(x.shape[-1] * 2, kernel_init=kernel_init_out_proj, dtype=dtype)(cond)
        scale, shift = jnp.split(modulation, 2, axis=-1)
        normed_inputs = normed_inputs * (1 + scale) + shift
        return normed_inputs.astype(dtype)


class DemonstrationAttention(nn.Module):
    """Cross-attention for demonstration modulation."""
    @nn.compact
    def __call__(self, x, demonstration_seq, demonstration_mask):
        B, demonstration_len, demonstration_width = demonstration_seq.shape
        B, x_len, x_width = x.shape
        num_heads, num_kv_heads, head_dim, width = (4, 1, 256, 1024)
        assert demonstration_width == x_width == width
        q_einsum = lora.Einsum(
            shape=(num_heads, width, head_dim), name="q_einsum_demonstration",
            init_fn=nn.initializers.lecun_normal(in_axis=-2, out_axis=-1, batch_axis=(0,)),
        )
        kv_einsum = lora.Einsum(
            shape=(2, num_kv_heads, width, head_dim), name="kv_einsum_demonstration",
            init_fn=nn.initializers.lecun_normal(in_axis=-2, out_axis=-1, batch_axis=(0, 1)),
        )
        rms_norm = DemonstrationRMSNorm(name="demonstration_rms_norm")
        x = rms_norm(x)
        q = q_einsum("BTD,NDH->BTNH", x)

        demonstration_seq = rms_norm(demonstration_seq)
        k, v = kv_einsum("BSD,2KDH->2BSKH", demonstration_seq)

        q_positions = einops.repeat(
            jnp.arange(demonstration_len, x_len + demonstration_len), "t -> b t", b=B,
        )
        k_positions = einops.repeat(jnp.arange(demonstration_len), "t -> b t", b=B)

        q = _apply_rope(q, positions=q_positions)
        q *= head_dim ** -0.5
        k = _apply_rope(k, positions=k_positions)
        q = einops.rearrange(q, "B T (K G) H -> B T K G H", K=num_kv_heads)

        logits = jnp.einsum("BTKGH,BSKH->BKGTS", q, k, preferred_element_type=jnp.float32)
        attn_mask = demonstration_mask[:, None, None, None, :]
        masked_logits = jnp.where(attn_mask, logits, -2.3819763e38)
        probs = jax.nn.softmax(masked_logits, axis=-1).astype(x.dtype)
        encoded = jnp.einsum("BKGTS,BSKH->BTKGH", probs, v)
        encoded = einops.rearrange(encoded, "B T K G H -> B T (K G) H")

        out_einsum = lora.Einsum(
            shape=(num_heads, head_dim, width), name="out_einsum_demonstration",
            init_fn=nn.initializers.lecun_normal(in_axis=(-3, -2), out_axis=-1),
        )
        return out_einsum("BTNH,NHD->BTD", encoded)


@at.typecheck
class LfOBlock(nn.Module):
    """Transformer block with demonstration integration support."""

    configs: tuple[Config, ...]
    dropout: float = 0.0
    dropout_bdims: tuple[int, ...] = ()
    integration_type: str | None = None

    @nn.compact
    def __call__(
        self,
        xs,
        kv_cache,
        positions,
        attn_mask,
        adarms_cond,
        demonstration_seq,
        demonstration_mask,
        deterministic=True,
    ):
        if self.integration_type == "modulation":
            demonstration_attn = DemonstrationAttention(name="demonstration_attn")

        xs = sharding.activation_sharding_constraint(xs)
        drop = nn.Dropout(self.dropout, self.dropout_bdims) if self.dropout else lambda x, _: x

        if self.integration_type == "expert":
            attn = Attention_with_DemonstrationExpert(configs=self.configs, name="attn")
        else:
            attn = Attention(configs=self.configs, name="attn")

        pre_attn = []
        gates = []
        for i, x in enumerate(xs):
            if x is not None:
                name = _name("pre_attention_norm", i) if self.integration_type != "expert" else _name("pre_attention_norm", i - 1)
                x, gate = RMSNorm(name=name)(x, adarms_cond[i])
            pre_attn.append(x)
            gates.append(gate if x is not None else None)

        pre_attn = sharding.activation_sharding_constraint(pre_attn)
        post_attn, kv_cache = attn(pre_attn, positions, attn_mask, kv_cache)
        post_attn = jax.tree.map(lambda x: drop(x, deterministic), post_attn)
        post_attn = sharding.activation_sharding_constraint(post_attn)
        xs = [_gated_residual(x, y, gate) for x, y, gate in zip(xs, post_attn, gates, strict=True)]
        xs = sharding.activation_sharding_constraint(xs)

        out = []
        gates = []
        for i, (x, config) in enumerate(zip(xs, self.configs, strict=True)):
            if x is not None:
                if i == len(xs) - 1 and self.integration_type == "modulation":
                    demonstration_mod_vec = demonstration_attn(x, demonstration_seq[-1], demonstration_mask[-1])
                    x = DemonstrationRMSNorm(name="demonstration_rms_norm_ffn")(x, demonstration_mod_vec)

                name = _name("pre_ffw_norm", i) if self.integration_type != "expert" else _name("pre_ffw_norm", i - 1)
                x, gate = RMSNorm(name=name)(x, adarms_cond[i])

                name = _name("mlp", i) if self.integration_type != "expert" else _name("mlp", i - 1)
                x = lora.FeedForward(
                    features=config.width, hidden_dim=config.mlp_dim, name=name,
                    lora_config=config.lora_configs.get("ffn"),
                )(x)

            out.append(x)
            gates.append(gate if x is not None else None)

        out = sharding.activation_sharding_constraint(out)
        out = jax.tree.map(lambda x: drop(x, deterministic), out)
        xs = [_gated_residual(x, y, gate) for x, y, gate in zip(xs, out, gates, strict=True)]
        xs = sharding.activation_sharding_constraint(xs)

        return xs, kv_cache


KVCache: TypeAlias = tuple[
    at.Float[at.Array, "l b _t _k _h"], at.Float[at.Array, "l b _t _v _h"]
]


@at.typecheck
class Module(nn.Module):
    """Transformer model with demonstration integration support."""

    configs: Sequence[Config]
    embed_dtype: str
    dropout: float = 0.0
    dropout_bdims: tuple[int, ...] = ()
    adarms: bool = False
    integration_type: str | None = None

    def setup(self):
        assert all(config.depth == self.configs[0].depth for config in self.configs)
        embed_dim = self.configs[0].width if self.integration_type != "expert" else self.configs[1].width
        self.embedder = Embedder(
            vocab_size=PALIGEMMA_VOCAB_SIZE, embed_dim=embed_dim, name="embedder",
        )
        block_cls = nn.remat(
            LfOBlock, prevent_cse=False, static_argnums=(7,),
            policy=jax.checkpoint_policies.nothing_saveable,
        )
        self.layers = nn.scan(
            block_cls,
            variable_axes={"params": 0},
            split_rngs={"params": True, "dropout": True},
            in_axes=(0, nn.broadcast, nn.broadcast, nn.broadcast, nn.broadcast, nn.broadcast, nn.broadcast),
            length=self.configs[0].depth,
        )(
            configs=self.configs,
            dropout=self.dropout,
            dropout_bdims=self.dropout_bdims,
            integration_type=self.integration_type,
        )
        self.final_norms = [
            RMSNorm(name=_name("final_norm", i) if self.integration_type != "expert" else _name("final_norm", i - 1))
            for i in range(len(self.configs))
        ]

    @at.typecheck
    def embed(self, tokens: at.Int[at.Array, "b t"]) -> at.Float[at.Array, "b t d"]:
        return self.embedder.encode(tokens).astype(self.embed_dtype)

    @at.typecheck
    def __call__(
        self,
        embedded: Sequence[at.Float[at.Array, "b _t _d"] | None],
        positions: at.Int[at.Array, "b t"],
        mask: at.Bool[at.Array, "b t s"],
        adarms_cond: Sequence[at.Float[at.Array, "b _d"] | None] | None = None,
        *,
        kv_cache: KVCache | None = None,
        demonstration_seq: Sequence[at.Float[at.Array, "b ldemo _d"] | None] | None = None,
        demonstration_mask: Sequence[at.Bool[at.Array, "b ldemo"] | None] | None = None,
        deterministic: bool = True,
    ) -> tuple[Sequence[at.Float[at.Array, "b _t _d"] | None], KVCache]:
        embedded = jax.tree.map(lambda e: e.astype(self.embed_dtype), embedded)
        mask = jnp.asarray(mask)[:, None, :, :]
        if adarms_cond is None:
            adarms_cond = [None] * len(self.configs)

        embedded, kv_cache = self.layers(
            embedded, kv_cache, positions, mask, adarms_cond,
            demonstration_seq, demonstration_mask, deterministic,
        )

        assert all(e.dtype == jnp.dtype(self.embed_dtype) for e in embedded if e is not None)

        return [
            f(e, a)[0] if e is not None else e
            for f, e, a in zip(self.final_norms, embedded, adarms_cond, strict=True)
        ], kv_cache

    def init(self, use_adarms: Sequence[bool], demonstration_mods: Sequence[bool]):
        """Convenience method for initializing all parameters."""
        self.embed(jnp.zeros((1, 1), dtype=jnp.int32))
        self(
            [jnp.zeros((1, 1, c.width)) for c in self.configs],
            jnp.zeros((1, len(self.configs)), dtype=jnp.int32),
            jnp.zeros((1, len(self.configs), len(self.configs)), dtype=bool),
            adarms_cond=[
                jnp.zeros((1, c.width)) if u else None
                for u, c in zip(use_adarms, self.configs, strict=True)
            ],
            demonstration_seq=[
                jnp.zeros((1, 4, c.width)) if m else None
                for c, m in zip(self.configs, demonstration_mods, strict=True)
            ],
            demonstration_mask=[
                jnp.ones((1, 4), dtype=bool) if m else None for m in demonstration_mods
            ],
        )
