import sys
from pathlib import Path

import jax
import jax.numpy as jnp
import einops
import math
import heapq
from copy import deepcopy
import numpy as np
from typing import Callable
import cv2
from collections import defaultdict

# Allow running from repo without setting PYTHONPATH.
# Needed so `openpi.*` (vendored under `src/external_models/openpi/`) can be imported.
_REPO_ROOT = Path(__file__).resolve().parents[3]
_EXTERNAL_MODELS_ROOT = _REPO_ROOT / "src" / "external_models"
_OPENPI_PKG_DIR = _EXTERNAL_MODELS_ROOT / "openpi"
if _OPENPI_PKG_DIR.exists():
    # sys.path must contain the directory *containing* the `openpi/` package.
    _s = str(_EXTERNAL_MODELS_ROOT)
    if _s not in sys.path:
        sys.path.insert(0, _s)

from openpi.shared import image_tools
from demonstration_vla_suite.shared.data_utils import (
    even_sampling_indices,
    right_padding_token_emb,
    left_padding_token_emb,
    pool_tokens_to_size,
)


def create_dict(indices):
    result = defaultdict(lambda: defaultdict(list))
    for step_idx, view_idx, patch_idx in indices:
        result[step_idx][view_idx].append(patch_idx)
    return result


class DemonstrationBuffer:
    """Demonstration buffer for perceptual demonstration (token dropping and frame sampling)."""

    def __init__(
        self,
        num_views: int = 1,
        img_emb_dim: int = 2048,
        pos_emb_dim: int = 768,
        state_emb_dim: int = 8,
        max_steps: int = 4096,
        compute_token_drop_score: bool = False,
        token_drop_keptsize: int = 2048,
        token_drop_stride: int = 8,
        prepare_buffer: bool = False,
        pool_type: str = "mean",
        vision_enc_fn: Callable | None = None,
    ):
        self.num_views = num_views
        self.img_emb_dim = img_emb_dim
        self.pos_emb_dim = pos_emb_dim
        self.state_emb_dim = state_emb_dim

        self.pool_type = pool_type
        self.token_drop_keptsize = token_drop_keptsize
        self.token_drop_stride = token_drop_stride
        self.token_drop_last_frame = -1
        self.scored_token_heap = []
        self.compute_token_drop_score = compute_token_drop_score

        self._demonstration_feats = {}

        if prepare_buffer:
            from demonstration_vla_suite.shared.posemb_3d import PosEmb3D

            if vision_enc_fn is None:
                # Match RoboMME default: use SigLIP tokenizer when caller doesn't provide a vision encoder.
                from demonstration_vla_suite.shared.siglip_tokenizer import SigLipTokenizer

                siglip_tokenizer = SigLipTokenizer()
                self.vision_enc = jax.jit(siglip_tokenizer.__call__)
            else:
                self.vision_enc = vision_enc_fn
            _cpu = next((d for d in jax.local_devices() if d.platform == "cpu"), None)
            if _cpu is not None:
                with jax.default_device(_cpu):
                    pos_embedder = PosEmb3D(dim=pos_emb_dim)
                    ranges = jnp.arange(max_steps)
                    pe8 = np.asarray(pos_embedder(ranges, 8), dtype=np.float32)
                    pe4 = np.asarray(pos_embedder(ranges, 4), dtype=np.float32)
                    pe2 = np.asarray(pos_embedder(ranges, 2), dtype=np.float32)
            else:
                pos_embedder = PosEmb3D(dim=pos_emb_dim)
                ranges = jnp.arange(max_steps)
                pe8 = np.asarray(pos_embedder(ranges, 8), dtype=np.float32)
                pe4 = np.asarray(pos_embedder(ranges, 4), dtype=np.float32)
                pe2 = np.asarray(pos_embedder(ranges, 2), dtype=np.float32)
            self.pos_emb_dict = {"8x8": pe8, "4x4": pe4, "2x2": pe2}
        else:
            self.vision_enc = None
            self.pos_emb_dict = None

    def add_buffer(
        self,
        images,         # (t v h w 3), np.uint8
        states,         # (t d), np.float32
        step_idx_list: list[int],
    ):
        assert self.vision_enc is not None, "encode is not initialized"

        t, v, _, _, _ = images.shape
        assert v == self.num_views

        image_jnp = jnp.array(images.astype(np.float32) / 255.0 * 2.0 - 1.0)
        image_jnp = einops.rearrange(image_jnp, "t v h w c -> (t v) h w c")
        image_jnp = image_tools.resize_with_pad(image_jnp, 224, 224)
        image_jnp = einops.rearrange(image_jnp, "(t v) h w c -> t v h w c", t=t, v=v)
        output_emb = self.vision_enc(image_jnp)  # (t, v, 64, 2048)

        pooled_emb_8x8 = pool_tokens_to_size(output_emb, 64)
        pooled_emb_4x4 = pool_tokens_to_size(output_emb, 16)
        pooled_emb_2x2 = pool_tokens_to_size(output_emb, 4)

        for i, step_idx in enumerate(step_idx_list):
            image_emb_8x8 = jax.device_get(pooled_emb_8x8)[i]
            image_emb_4x4 = jax.device_get(pooled_emb_4x4)[i]
            image_emb_2x2 = jax.device_get(pooled_emb_2x2)[i]

            pos_emb_8x8 = self.pos_emb_dict["8x8"][
                step_idx * self.num_views : (step_idx + 1) * self.num_views
            ]
            pos_emb_4x4 = self.pos_emb_dict["4x4"][
                step_idx * self.num_views : (step_idx + 1) * self.num_views
            ]
            pos_emb_2x2 = self.pos_emb_dict["2x2"][
                step_idx * self.num_views : (step_idx + 1) * self.num_views
            ]

            token_emb_to_save = {
                "image_pixels": images[i].copy(),
                "image_emb_8x8": image_emb_8x8,
                "image_emb_4x4": image_emb_4x4,
                "image_emb_2x2": image_emb_2x2,
                "pos_emb_8x8": pos_emb_8x8,
                "pos_emb_4x4": pos_emb_4x4,
                "pos_emb_2x2": pos_emb_2x2,
                "state_emb": states[i],
            }
            assert step_idx not in self._demonstration_feats, f"step_idx {step_idx} already in buffer"
            self._demonstration_feats[step_idx] = token_emb_to_save

            if self.compute_token_drop_score:
                self._process_token_drop_score(step_idx)

    def get_demonstration_feats(self, step_idx: int, remove_image_pixels: bool = True):
        if remove_image_pixels:
            return {k: v for k, v in self._demonstration_feats[step_idx].items() if k != "image_pixels"}
        return self._demonstration_feats[step_idx]

    def _process_token_drop_score(self, step_idx):
        if step_idx == 0:
            for patch_idx in range(64):
                for view_idx in range(self.num_views):
                    heapq.heappush(self.scored_token_heap, (1000.0, step_idx, view_idx, patch_idx))

        if step_idx == self.token_drop_last_frame + self.token_drop_stride:
            prev_img = self._demonstration_feats[max(0, self.token_drop_last_frame)]["image_pixels"]
            curr_img = self._demonstration_feats[step_idx]["image_pixels"]

            prev_img = prev_img.astype(np.float32) / 255.0 * 2.0 - 1.0
            curr_img = curr_img.astype(np.float32) / 255.0 * 2.0 - 1.0

            prev_img_tokens = einops.rearrange(prev_img, "v (ph h) (pw w) c -> v (ph pw) (h w c)", ph=8, pw=8)
            curr_img_tokens = einops.rearrange(curr_img, "v (ph h) (pw w) c -> v (ph pw) (h w c)", ph=8, pw=8)

            for view_idx in range(self.num_views):
                difference = np.abs(prev_img_tokens[view_idx] - curr_img_tokens[view_idx]).mean(axis=-1)
                for patch_idx in range(64):
                    if difference[patch_idx] < 1e-4:
                        continue
                    heapq.heappush(
                        self.scored_token_heap,
                        (difference[patch_idx], step_idx, view_idx, patch_idx),
                    )
                    if len(self.scored_token_heap) > self.token_drop_keptsize:
                        heapq.heappop(self.scored_token_heap)

            self.token_drop_last_frame += self.token_drop_stride

    def clear(self):
        self.scored_token_heap.clear()
        self.token_drop_last_frame = -1
        self._demonstration_feats.clear()

    def get_token_dropping_indices(self):
        selected_tokens_heap = deepcopy(self.scored_token_heap)
        kept_indices = []
        while selected_tokens_heap:
            _, buffer_idx, view_idx, patch_idx = heapq.heappop(selected_tokens_heap)
            kept_indices.append((buffer_idx, view_idx, patch_idx))
        return kept_indices

    @staticmethod
    def filter_token_dropping_indices(kept_indices, step_idx, token_budget, is_sorted=True):
        kept_indices_filtered = [item for item in kept_indices if item[0] <= step_idx]
        kept_indices = kept_indices_filtered[-token_budget:]
        if is_sorted:
            kept_indices = sorted(kept_indices)
        return kept_indices

    def _prepare_token_dropping(self, demonstration_feats, sorted_kept_indices, token_budget):
        img_emb = np.zeros((token_budget, self.img_emb_dim), dtype=np.float32)
        pos_emb = np.zeros((token_budget, self.pos_emb_dim), dtype=np.float32)
        state_emb = np.zeros((token_budget, self.state_emb_dim), dtype=np.float32)
        mask = np.zeros((token_budget), dtype=np.bool_)

        for idx, (buffer_idx, view_idx, patch_idx) in enumerate(sorted_kept_indices[:token_budget]):
            img_emb[idx] = demonstration_feats[buffer_idx]["image_emb_8x8"][view_idx][patch_idx]
            pos_emb[idx] = demonstration_feats[buffer_idx]["pos_emb_8x8"][view_idx][patch_idx]
            state_emb[idx] = demonstration_feats[buffer_idx]["state_emb"]
            mask[idx] = True

        return img_emb, pos_emb, state_emb, mask

    def prepare_token_dropping(self, step_idx, token_budget, demonstration_feats_gather_fn, kept_indices=None, *args, **kwargs):
        if kept_indices is None:
            indices_to_load = self.get_token_dropping_indices()
        else:
            indices_to_load = kept_indices

        sorted_kept_indices = self.filter_token_dropping_indices(
            indices_to_load, step_idx, token_budget, is_sorted=True
        )

        indices_to_load = sorted(set([item[0] for item in sorted_kept_indices]))
        demonstration_feats = demonstration_feats_gather_fn(indices_to_load, *args, **kwargs)

        return self._prepare_token_dropping(demonstration_feats, sorted_kept_indices, token_budget)

    def get_frame_sampling_indices(self, step_idx, token_budget, token_per_image):
        max_size = token_budget // (token_per_image * self.num_views)
        return even_sampling_indices(step_idx, max_size)

    def _prepare_frame_sampling(self, demonstration_feats, indices_to_load, token_budget, token_per_image):
        spatial_size = str(int(math.sqrt(token_per_image)))
        spatial_key = f"{spatial_size}x{spatial_size}"
        max_size = token_budget // (token_per_image * self.num_views)

        sampled_img_emb = self._load_emb(demonstration_feats, indices_to_load, f"image_emb_{spatial_key}")
        sampled_pos_emb = self._load_emb(demonstration_feats, indices_to_load, f"pos_emb_{spatial_key}")
        sampled_state_emb = self._load_emb(demonstration_feats, indices_to_load, "state_emb")
        mask = np.ones((sampled_img_emb.shape[0]), dtype=np.bool_)

        sampled_img_emb, sampled_pos_emb, sampled_state_emb, mask = right_padding_token_emb(
            sampled_img_emb, sampled_pos_emb, sampled_state_emb, mask, max_size
        )

        img_emb = np.reshape(sampled_img_emb, (-1, self.img_emb_dim))
        pos_emb = np.reshape(sampled_pos_emb, (-1, self.pos_emb_dim))
        mask = np.repeat(mask, self.num_views * token_per_image)
        state_emb = np.repeat(sampled_state_emb, self.num_views * token_per_image, axis=0)

        return img_emb, pos_emb, state_emb, mask

    def prepare_frame_sampling(self, step_idx, token_budget, token_per_image, demonstration_feats_gather_fn, *args, **kwargs):
        indices_to_load = self.get_frame_sampling_indices(step_idx, token_budget, token_per_image)
        demonstration_feats = demonstration_feats_gather_fn(indices_to_load, *args, **kwargs)
        return self._prepare_frame_sampling(demonstration_feats, indices_to_load, token_budget, token_per_image)

    @staticmethod
    def _load_emb(demonstration_feats: dict, indices_to_load: list[int], key: str):
        return np.stack([demonstration_feats[idx][key] for idx in indices_to_load], axis=0)

    def default_demonstration_feats_gather_fn(self, indices_to_load, *args, **kwargs):
        return {idx: self._demonstration_feats[idx] for idx in indices_to_load}


class DemonstrationBufferRecurrent(DemonstrationBuffer):
    """Demonstration buffer for recurrent demonstration (TTT and RMT)."""

    def __init__(
        self,
        input_obs_horizon: int = 8,
        max_recur_steps: int = 64,
        max_video_steps: int = 40,
        *args,
        **kwargs,
    ):
        super().__init__(*args, **kwargs)
        self.input_obs_horizon = input_obs_horizon
        self.max_recur_steps = max_recur_steps
        self.max_video_steps = max_video_steps

    def add_buffer(
        self,
        images,         # (t v h w 3), np.uint8
        state,          # (t d), np.float32
        step_idx_list: list[int],
    ):
        assert self.vision_enc is not None, "encode is not initialized"

        t, v, _, _, _ = images.shape
        assert v == self.num_views

        image_jnp = jnp.array(images.astype(np.float32) / 255.0 * 2.0 - 1.0)
        image_jnp = einops.rearrange(image_jnp, "t v h w c -> (t v) h w c")
        image_jnp = image_tools.resize_with_pad(image_jnp, 224, 224)
        image_jnp = einops.rearrange(image_jnp, "(t v) h w c -> t v h w c", t=t, v=v)
        output_emb = self.vision_enc(image_jnp)  # (t, v, 64, 2048)

        pooled_emb_8x8 = pool_tokens_to_size(output_emb, 64)

        for i, step_idx in enumerate(step_idx_list):
            image_emb_8x8 = jax.device_get(pooled_emb_8x8)[i]

            pos_emb_8x8 = self.pos_emb_dict["8x8"][
                step_idx * self.num_views : (step_idx + 1) * self.num_views
            ]

            token_emb_to_save = {
                "image_emb_8x8": image_emb_8x8,
                "pos_emb_8x8": pos_emb_8x8,
                "state_emb": state[i],
            }
            assert step_idx not in self._demonstration_feats, f"step_idx {step_idx} already in buffer"
            self._demonstration_feats[step_idx] = token_emb_to_save

    def get_token_recurrent_indices(self, step_idx, exec_start_idx):
        input_obs_horizon = self.input_obs_horizon
        max_recur_steps = self.max_recur_steps
        max_video_steps = self.max_video_steps

        assert step_idx >= 0 and step_idx >= exec_start_idx

        if exec_start_idx == 0:
            if step_idx < input_obs_horizon:
                indices_to_load = [step_idx]
            else:
                start_idx = step_idx % input_obs_horizon
                indices_to_load = list(range(start_idx, step_idx + 1, input_obs_horizon))
                indices_to_load = indices_to_load[-max_recur_steps:]
        else:
            if exec_start_idx <= input_obs_horizon * 2:
                video_indices = list(range(0, exec_start_idx, input_obs_horizon // 2))
            elif exec_start_idx <= max_video_steps * input_obs_horizon:
                video_indices = list(range(0, exec_start_idx, input_obs_horizon))
            else:
                video_indices = np.linspace(0, exec_start_idx - 1, max_video_steps, dtype=int).tolist()

            if step_idx - exec_start_idx < input_obs_horizon:
                rest_indices = [step_idx]
            else:
                start_idx = (step_idx - exec_start_idx) % input_obs_horizon + exec_start_idx
                rest_indices = list(range(start_idx, step_idx + 1, input_obs_horizon))

            combined_indices = video_indices + rest_indices
            indices_to_load = combined_indices[-max_recur_steps:]

        assert len(indices_to_load) > 0 and len(indices_to_load) <= max_recur_steps

        return indices_to_load

    def _prepare_token_recurrent(self, demonstration_feats, indices_to_load, padding=True):
        recur_image_embs = []
        recur_pos_embs = []
        recur_state_embs = []
        recur_masks = []

        for i in indices_to_load:
            recur_image_embs.append(demonstration_feats[i]["image_emb_8x8"])
            recur_pos_embs.append(demonstration_feats[i]["pos_emb_8x8"])
            recur_state_embs.append(demonstration_feats[i]["state_emb"])
            recur_masks.append(True)

        recur_image_emb = np.stack(recur_image_embs, axis=0)
        recur_pos_emb = np.stack(recur_pos_embs, axis=0)
        recur_state_emb = np.stack(recur_state_embs, axis=0)
        recur_mask = np.stack(recur_masks, axis=0)

        if padding:
            recur_image_emb, recur_pos_emb, recur_state_emb, recur_mask = left_padding_token_emb(
                recur_image_emb, recur_pos_emb, recur_state_emb, recur_mask, self.max_recur_steps
            )
        return recur_image_emb, recur_pos_emb, recur_state_emb, recur_mask

    def prepare_token_recurrent(self, step_idx, exec_start_idx, demonstration_feats_gather_fn, *args, **kwargs):
        indices_to_load = self.get_token_recurrent_indices(step_idx, exec_start_idx)
        assert len(indices_to_load) > 0 and len(indices_to_load) <= self.max_recur_steps
        demonstration_feats = demonstration_feats_gather_fn(indices_to_load, *args, **kwargs)
        return self._prepare_token_recurrent(demonstration_feats, indices_to_load)
