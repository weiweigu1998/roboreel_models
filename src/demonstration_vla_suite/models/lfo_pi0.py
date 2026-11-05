import dataclasses
import logging
from typing import Any

import einops
import flax.nnx as nnx
import flax.nnx.bridge as nnx_bridge
import jax
import jax.numpy as jnp
from typing_extensions import override

from openpi.models.model import Actions
from openpi.models.model import BaseModel
from openpi.models.pi0_config import Pi0Config
from openpi.models.pi0 import posemb_sincos
import openpi.models.siglip as _siglip
import openpi.shared.array_typing as at
import openpi.shared.nnx_utils as nnx_utils

from demonstration_vla_suite.models.lfo_observation import (
    HistAugObservation,
    preprocess_observation,
)
from demonstration_vla_suite.models import lfo_gemma as _gemma
from demonstration_vla_suite.models.config.utils import get_demonstration_config


logger = logging.getLogger("lfo-pi0")


def make_attn_mask(input_mask, mask_ar, mask_na=None):
    """Build attention mask supporting prefix-lm and demonstration-aware patterns."""
    mask_ar = jnp.broadcast_to(mask_ar, input_mask.shape)
    cumsum = jnp.cumsum(mask_ar, axis=1)
    attn_mask = cumsum[:, None, :] <= cumsum[:, :, None]
    valid_mask = input_mask[:, None, :] * input_mask[:, :, None]
    if mask_na is not None:
        mask_na = jnp.broadcast_to(mask_na, input_mask.shape)
        mask_not_attend = jnp.logical_and(
            jnp.logical_or(mask_na[:, None, :], mask_na[:, :, None]),
            einops.repeat(
                jnp.cumsum(mask_na, axis=1) <= 0, "b k -> b s k", s=mask_na.shape[1]
            ),
        )
        return jnp.where(mask_not_attend, False, jnp.logical_and(attn_mask, valid_mask))
    else:
        return jnp.logical_and(attn_mask, valid_mask)


@dataclasses.dataclass(frozen=True)
class LfOPi0Config(Pi0Config):
    demonstration_expert_variant: _gemma.Variant = "gemma_150m"
    #: Mirroring ``HistoryPi0Config.use_history``: when False, build plain π₀.₅ (no LfO modules).
    use_demonstration: bool = False
    demonstration_config: str | None = None
    max_token_len: int = 64

    @override
    def create(self, rng: at.KeyArrayLike) -> "LfOPi0":
        # Like ``HistoryPi0Config.create``: only load YAML when the feature is on *and* a path is set.
        if self.use_demonstration and self.demonstration_config is not None:
            loaded_config = get_demonstration_config(self.demonstration_config)
            config_loaded = dataclasses.replace(self, demonstration_config=loaded_config)

            print("max_token_len: ", config_loaded.max_token_len)

            return LfOPi0(config_loaded, rngs=nnx.Rngs(rng))

        cfg_plain = dataclasses.replace(self, use_demonstration=False, demonstration_config=None)
        return LfOPi0(cfg_plain, rngs=nnx.Rngs(rng))

    def inputs_spec(self, *, batch_size: int = 1) -> tuple[HistAugObservation, Actions]:
        base_obs_spec, action_spec = super().inputs_spec(batch_size=batch_size)
        with at.disable_typechecking():
            # Like ``HistoryPi0Config.inputs_spec`` + ``if not self.use_history: base_obs_spec``.
            if not self.use_demonstration or self.demonstration_config is None:
                observation_spec = base_obs_spec
            else:
                dc = self.demonstration_config
                if isinstance(dc, str):
                    dc = get_demonstration_config(dc)
                if dc.representation_type == "perceptual":
                    observation_spec = HistAugObservation.from_base_obs(
                        base_obs_spec,
                        static_image_emb=jax.ShapeDtypeStruct(
                            [batch_size, dc.budget, dc.demonstration_feature.img.input_dim],
                            jnp.float32,
                        ),
                        static_mask=jax.ShapeDtypeStruct(
                            [batch_size, dc.budget], jnp.bool_,
                        ),
                        static_pos_emb=jax.ShapeDtypeStruct(
                            [batch_size, dc.budget, dc.demonstration_feature.pos.input_dim],
                            jnp.float32,
                        ),
                        static_state_emb=jax.ShapeDtypeStruct(
                            [batch_size, dc.budget, dc.demonstration_feature.state.input_dim],
                            jnp.float32,
                        ),
                    )
                elif dc.representation_type == "recurrent":
                    observation_spec = HistAugObservation.from_base_obs(
                        base_obs_spec,
                        recur_image_emb=jax.ShapeDtypeStruct(
                            [batch_size, dc.recurrent_demonstration.max_recur_steps,
                             dc.num_views, dc.token_per_image,
                             dc.demonstration_feature.img.input_dim],
                            jnp.float32,
                        ),
                        recur_mask=jax.ShapeDtypeStruct(
                            [batch_size, dc.recurrent_demonstration.max_recur_steps], jnp.bool_,
                        ),
                        recur_pos_emb=jax.ShapeDtypeStruct(
                            [batch_size, dc.recurrent_demonstration.max_recur_steps,
                             dc.num_views, dc.token_per_image,
                             dc.demonstration_feature.pos.input_dim],
                            jnp.float32,
                        ),
                        recur_state_emb=jax.ShapeDtypeStruct(
                            [batch_size, dc.recurrent_demonstration.max_recur_steps,
                             dc.demonstration_feature.state.input_dim],
                            jnp.float32,
                        ),
                    )
                else:
                    raise ValueError(f"Not supported representation type: {dc.representation_type}")

        return observation_spec, action_spec

    def get_freeze_filter(self) -> nnx.filterlib.Filter:
        filters = []
        has_lora = False
        gemma_params_filter = nnx_utils.PathRegex(".*llm.*")
        action_expert_params_filter = nnx_utils.PathRegex(".*llm.*_1.*")
        if "lora" in self.paligemma_variant:
            filters.append(gemma_params_filter)
            if "lora" not in self.action_expert_variant:
                filters.append(nnx.Not(action_expert_params_filter))
            has_lora = True
        elif "lora" in self.action_expert_variant:
            filters.append(action_expert_params_filter)
            has_lora = True

        if has_lora:
            filters.extend([
                nnx.Not(nnx_utils.PathRegex(".*lora.*")),
                nnx.Not(nnx_utils.PathRegex(".*(mem|demonstration).*")),
            ])
            return nnx.Any(nnx.All(*filters), nnx_utils.PathRegex(".*img.*"))
        else:
            return nnx_utils.PathRegex(".*img.*")


class LfOPi0(BaseModel):
    def __init__(self, config: LfOPi0Config, rngs: nnx.Rngs):
        super().__init__(config.action_dim, config.action_horizon, config.max_token_len)
        self.pi05 = config.pi05
        paligemma_config = _gemma.get_config(config.paligemma_variant)
        action_expert_config = _gemma.get_config(config.action_expert_variant)

        self.config = config
        self.use_demonstration = config.use_demonstration
        if self.use_demonstration:
            self.demonstration_config = config.demonstration_config
            self.integration_type = config.demonstration_config.integration_type
            self.representation_type = config.demonstration_config.representation_type
            assert self.integration_type in ["context", "modulation", "expert"]
            assert self.representation_type in ["perceptual", "recurrent"]

            if self.representation_type == "perceptual":
                from demonstration_vla_suite.models.representation.perceptual_demonstration_encoder import (
                    PerceptualDemonstrationEncoder,
                )
                self.demonstration_encoder = PerceptualDemonstrationEncoder(
                    config=self.demonstration_config, rngs=rngs, dtype=config.dtype,
                )
            elif self.representation_type == "recurrent":
                from demonstration_vla_suite.models.representation.recurrent_demonstration_encoder import (
                    RecurrentDemonstrationEncoder,
                )
                self.demonstration_encoder = RecurrentDemonstrationEncoder(
                    config=self.demonstration_config, rngs=rngs, dtype=config.dtype,
                )
            else:
                raise ValueError(f"Not supported representation type: {self.representation_type}")

            print(f"====== LfO Pi0: representation={self.representation_type}, integration={self.integration_type} ======")

            if self.integration_type == "expert":
                demonstration_expert_config = _gemma.get_config(config.demonstration_expert_variant)
                llm = nnx_bridge.ToNNX(
                    _gemma.Module(
                        configs=[demonstration_expert_config, paligemma_config, action_expert_config],
                        embed_dtype=config.dtype, adarms=config.pi05, integration_type=self.integration_type,
                    )
                )
                llm.lazy_init(
                    rngs=rngs, method="init",
                    use_adarms=([False, False, True] if config.pi05 else [False, False, False]),
                    demonstration_mods=[False, False, False],
                )
            else:
                llm = nnx_bridge.ToNNX(
                    _gemma.Module(
                        configs=[paligemma_config, action_expert_config],
                        embed_dtype=config.dtype, adarms=config.pi05, integration_type=self.integration_type,
                    )
                )
                llm.lazy_init(
                    rngs=rngs, method="init",
                    use_adarms=[False, True] if config.pi05 else [False, False],
                    demonstration_mods=[False, True] if self.integration_type == "modulation" else [False, False],
                )
        else:
            self.demonstration_config = self.integration_type = self.representation_type = None
            llm = nnx_bridge.ToNNX(
                _gemma.Module(
                    configs=[paligemma_config, action_expert_config],
                    embed_dtype=config.dtype, adarms=config.pi05, integration_type=self.integration_type,
                )
            )
            llm.lazy_init(
                rngs=rngs, method="init",
                use_adarms=[False, True] if config.pi05 else [False, False],
                demonstration_mods=[False, False],
            )

        img = nnx_bridge.ToNNX(
            _siglip.Module(
                num_classes=paligemma_config.width, variant="So400m/14",
                pool_type="none", scan=True, dtype_mm=config.dtype,
            )
        )
        img.lazy_init(next(iter(config.fake_obs().images.values())), train=False, rngs=rngs)

        self.PaliGemma = nnx.Dict(llm=llm, img=img)
        self.action_in_proj = nnx.Linear(config.action_dim, action_expert_config.width, rngs=rngs)

        if config.pi05:
            self.time_mlp_in = nnx.Linear(action_expert_config.width, action_expert_config.width, rngs=rngs)
            self.time_mlp_out = nnx.Linear(action_expert_config.width, action_expert_config.width, rngs=rngs)
        else:
            self.state_proj = nnx.Linear(config.action_dim, action_expert_config.width, rngs=rngs)
            self.action_time_mlp_in = nnx.Linear(2 * action_expert_config.width, action_expert_config.width, rngs=rngs)
            self.action_time_mlp_out = nnx.Linear(action_expert_config.width, action_expert_config.width, rngs=rngs)
        self.action_out_proj = nnx.Linear(action_expert_config.width, config.action_dim, rngs=rngs)

        self.deterministic = True

    @at.typecheck
    def embed_demonstration(self, obs: HistAugObservation):
        if not self.use_demonstration:
            return None, None, None, None, None
        if self.representation_type == "perceptual":
            tokens, _, stats = self.demonstration_encoder(obs.static_image_emb, obs.static_pos_emb, obs.static_state_emb)
            input_mask = obs.static_mask
            ar_mask = [False] * tokens.shape[1]
            na_mask = [False] * tokens.shape[1]
        elif self.representation_type == "recurrent":
            (tokens, input_mask), _, stats = self.demonstration_encoder(
                obs.recur_image_emb, obs.recur_mask, obs.recur_pos_emb, obs.recur_state_emb
            )
            ar_mask = [False] * tokens.shape[1]
            na_mask = [False] * tokens.shape[1]
        else:
            tokens = input_mask = ar_mask = na_mask = stats = None
        return tokens, input_mask, ar_mask, na_mask, stats

    @at.typecheck
    def embed_prefix(self, obs: HistAugObservation):
        input_mask = []
        ar_mask = []
        tokens = []
        na_mask = []

        # Match ``HistoryPi0.embed_prefix``: memory/demonstration prefix only in context mode when enabled.
        if self.use_demonstration and self.integration_type == "context":
            demonstration_tokens, demonstration_input_mask, demonstration_ar_mask, demonstration_na_mask, stats = self.embed_demonstration(obs)
            if demonstration_tokens is not None:
                tokens.append(demonstration_tokens)
                input_mask.append(demonstration_input_mask)
                ar_mask += demonstration_ar_mask
                na_mask += demonstration_na_mask
        else:
            stats = None

        for i, name in enumerate(obs.images):
            image_tokens, _ = self.PaliGemma.img(obs.images[name], train=False)
            tokens.append(image_tokens)
            input_mask.append(
                einops.repeat(obs.image_masks[name], "b -> b s", s=image_tokens.shape[1])
            )
            if i == 0:
                ar_mask += [True] + ([False] * (image_tokens.shape[1] - 1))
            else:
                ar_mask += [False] * image_tokens.shape[1]
            na_mask += [True] * image_tokens.shape[1]
        # Line 468 to line 477 from robomme_policy_learning/history_pi0.py is removed as we don't use the symbolic representation.
        if obs.tokenized_prompt is not None:
            tokenized_inputs = self.PaliGemma.llm(obs.tokenized_prompt, method="embed")
            tokens.append(tokenized_inputs)
            input_mask.append(obs.tokenized_prompt_mask)
            ar_mask += [False] * tokenized_inputs.shape[1]
            na_mask += [False] * tokenized_inputs.shape[1]

        tokens = jnp.concatenate(tokens, axis=1)
        input_mask = jnp.concatenate(input_mask, axis=1)
        ar_mask = jnp.array(ar_mask)
        na_mask = jnp.array(na_mask)

        return tokens, input_mask, ar_mask, na_mask, stats

    @at.typecheck
    def embed_suffix(self, obs: HistAugObservation, noisy_actions: Actions, timestep: at.Float[at.Array, " b"]):
        input_mask = []
        ar_mask = []
        tokens = []
        na_mask = []

        if not self.pi05:
            state_token = self.state_proj(obs.state)[:, None, :]
            tokens.append(state_token)
            input_mask.append(jnp.ones((obs.state.shape[0], 1), dtype=jnp.bool_))
            ar_mask += [True]
            na_mask += [False]

        action_tokens = self.action_in_proj(noisy_actions)
        time_emb = posemb_sincos(timestep, self.action_in_proj.out_features, min_period=4e-3, max_period=4.0)
        if self.pi05:
            time_emb = self.time_mlp_in(time_emb)
            time_emb = nnx.swish(time_emb)
            time_emb = self.time_mlp_out(time_emb)
            time_emb = nnx.swish(time_emb)
            action_expert_tokens = action_tokens
            adarms_cond = time_emb
        else:
            time_tokens = einops.repeat(time_emb, "b emb -> b s emb", s=self.action_horizon)
            action_time_tokens = jnp.concatenate([action_tokens, time_tokens], axis=-1)
            action_time_tokens = self.action_time_mlp_in(action_time_tokens)
            action_time_tokens = nnx.swish(action_time_tokens)
            action_time_tokens = self.action_time_mlp_out(action_time_tokens)
            action_expert_tokens = action_time_tokens
            adarms_cond = None

        tokens.append(action_expert_tokens)
        input_mask.append(jnp.ones(action_expert_tokens.shape[:2], dtype=jnp.bool_))
        ar_mask += [True] + ([False] * (self.action_horizon - 1))
        na_mask += [False] * self.action_horizon
        tokens = jnp.concatenate(tokens, axis=1)
        input_mask = jnp.concatenate(input_mask, axis=1)

        ar_mask = jnp.array(ar_mask)
        na_mask = jnp.array(na_mask)
        return tokens, input_mask, ar_mask, na_mask, adarms_cond

    @override
    def compute_loss(self, rng, observation, actions, *, train=False):
        preprocess_rng, noise_rng, time_rng = jax.random.split(rng, 3)
        observation = preprocess_observation(preprocess_rng, observation, train=train)

        batch_shape = actions.shape[:-2]
        noise = jax.random.normal(noise_rng, actions.shape)
        time = jax.random.beta(time_rng, 1.5, 1, batch_shape) * 0.999 + 0.001
        time_expanded = time[..., None, None]
        x_t = time_expanded * noise + (1 - time_expanded) * actions
        u_t = noise - actions

        prefix_tokens, prefix_mask, prefix_ar_mask, prefix_na_mask, stats = self.embed_prefix(observation)
        suffix_tokens, suffix_mask, suffix_ar_mask, suffix_na_mask, adarms_cond = self.embed_suffix(observation, x_t, time)

        if self.integration_type == "expert":
            demonstration_tokens, demonstration_input_mask, demonstration_ar_mask, demonstration_na_mask, stats = self.embed_demonstration(observation)
            demonstration_ar_mask = jnp.array(demonstration_ar_mask)
            demonstration_na_mask = jnp.array(demonstration_na_mask)
            input_mask = jnp.concatenate([demonstration_input_mask, prefix_mask, suffix_mask], axis=1)
            ar_mask = jnp.concatenate([demonstration_ar_mask, prefix_ar_mask, suffix_ar_mask], axis=0)
            na_mask = jnp.concatenate([demonstration_na_mask, prefix_na_mask, suffix_na_mask], axis=0)
        else:
            input_mask = jnp.concatenate([prefix_mask, suffix_mask], axis=1)
            ar_mask = jnp.concatenate([prefix_ar_mask, suffix_ar_mask], axis=0)
            na_mask = jnp.concatenate([prefix_na_mask, suffix_na_mask], axis=0)

        # Like ``HistoryPi0.compute_loss``: use na_mask when demonstrations participate as non-symbolic tokens.
        # This part is not used as we don't use the symbolic representation.
        if self.representation_type != "symbolic":
            attn_mask = make_attn_mask(input_mask, ar_mask, na_mask)
        else:
            attn_mask = make_attn_mask(input_mask, ar_mask)

        positions = jnp.cumsum(input_mask, axis=1) - 1

        if self.integration_type == "expert":
            (demonstration_out, prefix_out, suffix_out), _ = self.PaliGemma.llm(
                [demonstration_tokens, prefix_tokens, suffix_tokens],
                mask=attn_mask, positions=positions, adarms_cond=[None, None, adarms_cond],
            )
        elif self.integration_type == "modulation":
            demonstration_seq, demonstration_mask, _, _, stats = self.embed_demonstration(observation)
            (prefix_out, suffix_out), _ = self.PaliGemma.llm(
                [prefix_tokens, suffix_tokens],
                mask=attn_mask, positions=positions, adarms_cond=[None, adarms_cond],
                demonstration_seq=[None, demonstration_seq], demonstration_mask=[None, demonstration_mask],
            )
        else:
            (prefix_out, suffix_out), _ = self.PaliGemma.llm(
                [prefix_tokens, suffix_tokens],
                mask=attn_mask, positions=positions, adarms_cond=[None, adarms_cond],
            )

        v_t = self.action_out_proj(suffix_out[:, -self.action_horizon:])
        return jnp.mean(jnp.square(v_t - u_t), axis=-1), stats

    @override
    def sample_actions(self, rng, observation, *, num_steps=10, noise=None):
        observation = preprocess_observation(None, observation, train=False)
        dt = -1.0 / num_steps
        batch_size = observation.state.shape[0]
        if noise is None:
            noise = jax.random.normal(rng, (batch_size, self.action_horizon, self.action_dim))

        if self.integration_type == "expert":
            demonstration_tokens, demonstration_input_mask, demonstration_ar_mask, demonstration_na_mask, _ = self.embed_demonstration(observation)
            vlm_tokens, vlm_mask, vlm_ar_mask, vlm_na_mask, _ = self.embed_prefix(observation)
            demonstration_ar_mask = jnp.array(demonstration_ar_mask)
            demonstration_na_mask = jnp.array(demonstration_na_mask)
            prefix_mask = jnp.concatenate([demonstration_input_mask, vlm_mask], axis=1)
            prefix_ar_mask = jnp.concatenate([demonstration_ar_mask, vlm_ar_mask], axis=0)
            prefix_na_mask = jnp.concatenate([demonstration_na_mask, vlm_na_mask], axis=0)
            prefix_attn_mask = make_attn_mask(prefix_mask, prefix_ar_mask, prefix_na_mask)
            positions = jnp.cumsum(prefix_mask, axis=1) - 1
            _, kv_cache = self.PaliGemma.llm(
                [demonstration_tokens, vlm_tokens, None], mask=prefix_attn_mask, positions=positions,
            )
        elif self.integration_type == "modulation":
            prefix_tokens, prefix_mask, prefix_ar_mask, _, _ = self.embed_prefix(observation)
            prefix_attn_mask = make_attn_mask(prefix_mask, prefix_ar_mask)
            positions = jnp.cumsum(prefix_mask, axis=1) - 1
            _, kv_cache = self.PaliGemma.llm(
                [prefix_tokens, None], mask=prefix_attn_mask, positions=positions,
            )
            demonstration_seq, demonstration_mask, _, _, _ = self.embed_demonstration(observation)
        else:
            prefix_tokens, prefix_mask, prefix_ar_mask, prefix_na_mask, _ = self.embed_prefix(observation)
            if self.use_demonstration and self.integration_type == "context":
                prefix_attn_mask = make_attn_mask(prefix_mask, prefix_ar_mask, prefix_na_mask)
            else:
                prefix_attn_mask = make_attn_mask(prefix_mask, prefix_ar_mask)
            positions = jnp.cumsum(prefix_mask, axis=1) - 1
            _, kv_cache = self.PaliGemma.llm(
                [prefix_tokens, None], mask=prefix_attn_mask, positions=positions,
            )

        def step(carry):
            x_t, time = carry
            suffix_tokens, suffix_mask, suffix_ar_mask, _, adarms_cond = self.embed_suffix(
                observation, x_t, jnp.broadcast_to(time, batch_size)
            )
            suffix_attn_mask = make_attn_mask(suffix_mask, suffix_ar_mask)
            prefix_attn_mask = einops.repeat(prefix_mask, "b p -> b s p", s=suffix_tokens.shape[1])
            full_attn_mask = jnp.concatenate([prefix_attn_mask, suffix_attn_mask], axis=-1)
            assert full_attn_mask.shape == (
                batch_size, suffix_tokens.shape[1], prefix_mask.shape[1] + suffix_tokens.shape[1],
            )
            positions = (
                jnp.sum(prefix_mask, axis=-1)[:, None] + jnp.cumsum(suffix_mask, axis=-1) - 1
            )

            if self.integration_type == "expert":
                (demonstration_out, prefix_out, suffix_out), _ = self.PaliGemma.llm(
                    [None, None, suffix_tokens], mask=full_attn_mask, positions=positions,
                    kv_cache=kv_cache, adarms_cond=[None, None, adarms_cond],
                )
                assert demonstration_out is None
            elif self.integration_type == "modulation":
                (prefix_out, suffix_out), _ = self.PaliGemma.llm(
                    [None, suffix_tokens], mask=full_attn_mask, positions=positions,
                    kv_cache=kv_cache, adarms_cond=[None, adarms_cond],
                    demonstration_seq=[None, demonstration_seq], demonstration_mask=[None, demonstration_mask],
                )
            else:
                (prefix_out, suffix_out), _ = self.PaliGemma.llm(
                    [None, suffix_tokens], mask=full_attn_mask, positions=positions,
                    kv_cache=kv_cache, adarms_cond=[None, adarms_cond],
                )

            assert prefix_out is None
            v_t = self.action_out_proj(suffix_out[:, -self.action_horizon:])
            return x_t + dt * v_t, time + dt

        def cond(carry):
            x_t, time = carry
            return time >= -dt / 2

        x_0, _ = jax.lax.while_loop(cond, step, (noise, 1.0))
        return x_0

    def vision_encode(self, images: at.Float[at.Array, "k v 224 224 3"]):
        shape = images.shape
        assert shape[-3:] == (224, 224, 3), f"x.shape: {shape}, should be (B, 224, 224, 3)"
        x = einops.rearrange(images, "k v h w c -> (k v) h w c")
        out, _ = self.PaliGemma.img(x, train=False)
        out = einops.rearrange(out, "(k v) p d -> k v p d", k=shape[0], v=shape[1])
        return out
