from collections.abc import Sequence
import time
from typing import Any

import jax
import jax.numpy as jnp
import numpy as np
from typing_extensions import override

from openpi import transforms as _transforms
from openpi.shared import nnx_utils

from demonstration_vla_suite.models.lfo_observation import HistAugObservation
from demonstration_vla_suite.models.lfo_pi0 import LfOPi0
from demonstration_vla_suite.shared.fixed_demo_encode import encode_pi0_demonstration_tensors


class DemoVLAPolicy:
    def __init__(
        self,
        model: LfOPi0,
        *,
        seed: int = 42,
        transforms: Sequence[_transforms.DataTransformFn] = (),
        output_transforms: Sequence[_transforms.DataTransformFn] = (),
        sample_kwargs: dict[str, Any] | None = None,
        metadata: dict[str, Any] | None = None,
        norm_stats: dict[str, _transforms.NormStats] | None = None,
        use_quantiles: bool = False,
    ):
        self._model = model
        self._seed = seed
        self._input_transform = _transforms.compose(transforms)
        self._output_transform = _transforms.compose(output_transforms)
        self._sample_kwargs = sample_kwargs or {}
        self._metadata = metadata or {}

        self._sample_actions = nnx_utils.module_jit(model.sample_actions)
        self._vision_encode = nnx_utils.module_jit(model.vision_encode)

        self.config = model.demonstration_config
        self._demo_fields: dict[str, np.ndarray] | None = None

        self.state_norm_stats = norm_stats["state"]
        self.use_quantiles = use_quantiles

        self.reset()

    @override
    def infer(self, obs: dict) -> dict:
        if self.config is not None:
            has_inline = "static_image_emb" in obs or "recur_image_emb" in obs
            assert has_inline or self._demo_fields is not None, (
                "LfO Pi0 requires demonstration tensors: pass static_* / recur_* on obs, "
                "or call set_demonstration(demo_images, demo_states) once per episode."
            )

        inputs = jax.tree.map(lambda x: x, obs)
        inputs = self._attach_demonstration(inputs)
        inputs = self._input_transform(inputs)
        observation = HistAugObservation.from_dict(
            jax.tree.map(lambda x: jnp.asarray(x)[np.newaxis, ...], inputs)
        )
        self._rng, sample_rng = jax.random.split(self._rng)

        start_time = time.monotonic()
        outputs = {
            "state": observation.state,
            "actions": self._sample_actions(sample_rng, observation, **self._sample_kwargs),
        }
        model_time = time.monotonic() - start_time
        outputs = jax.tree.map(lambda x: np.asarray(x[0, ...]), outputs)
        outputs = self._output_transform(outputs)
        outputs["infer_time_ms"] = model_time * 1000

        return outputs

    @override
    def reset(self) -> None:
        self._demo_fields = None
        self._rng = jax.random.key(self._seed)

    def set_demonstration(self, images: np.ndarray, states: np.ndarray) -> None:
        """Encode a full expert trajectory and cache tensors for subsequent ``infer`` calls.

        Args:
            images: ``(T, num_views, 224, 224, 3)`` uint8 (same layout as former ``add_buffer``).
            states: ``(T, D)`` float32, one state vector per demo frame.
        """
        if self.config is None:
            return
        self._demo_fields = encode_pi0_demonstration_tensors(
            self.config, self._vision_encode, images, states,
        )

    def _normalize_state_np(self, state: np.ndarray) -> np.ndarray:
        if self.use_quantiles:
            return (state - self.state_norm_stats.q01) / (
                self.state_norm_stats.q99 - self.state_norm_stats.q01 + 1e-6
            ) * 2.0 - 1.0
        return (state - self.state_norm_stats.mean) / (self.state_norm_stats.std + 1e-6)

    def _attach_demonstration(self, inputs: dict) -> dict:
        if self.config is None:
            return inputs

        if self.config.representation_type == "perceptual":
            if "static_image_emb" not in inputs:
                assert self._demo_fields is not None
                inputs = {**inputs, **self._demo_fields}
            if inputs.get("static_state_emb") is not None:
                inputs = {
                    **inputs,
                    "static_state_emb": self._normalize_state_np(inputs["static_state_emb"]),
                }
        elif self.config.representation_type == "recurrent":
            if "recur_image_emb" not in inputs:
                assert self._demo_fields is not None
                inputs = {**inputs, **self._demo_fields}
            if inputs.get("recur_state_emb") is not None:
                inputs = {
                    **inputs,
                    "recur_state_emb": self._normalize_state_np(inputs["recur_state_emb"]),
                }
        else:
            raise ValueError(f"Not supported representation type: {self.config.representation_type}")

        return inputs

    @property
    def metadata(self) -> dict[str, Any]:
        return self._metadata
