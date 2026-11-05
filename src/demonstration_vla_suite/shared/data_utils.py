import math
import re
from pathlib import Path

import einops
import numpy as np

# ``flax.nnx`` is imported lazily inside ``pool_tokens_to_size`` because the
# non-JAX consumers use other helpers in this module (``get_demonstration_tokens``
# in particular) without ever needing JAX. Keeping the import top-level would force
# every consumer of ``data_utils`` to also install ``flax``.


def even_sampling_indices(step_idx: int, token_budget: int) -> list[int]:
    """Generate evenly spaced indices for sampling frames."""
    if step_idx < token_budget:
        return list(range(step_idx + 1))
    else:
        return np.linspace(0, step_idx, token_budget, dtype=np.int32).tolist()


def right_padding_token_emb(
    sampled_img_emb,    # (l v p d1)
    sampled_pos_emb,    # (l v p d2)
    sampled_state_emb,  # (l d3)
    mask,               # (l)
    max_size: int,
):
    if sampled_img_emb.shape[0] < max_size:
        # Match input dtypes when zero-padding — ``np.zeros`` defaults to float64
        # which silently upcasts the float32 inputs and breaks downstream
        # ``F.linear(...)`` calls (mat1 Double vs mat2 Float).
        sampled_img_emb = np.concatenate(
            [sampled_img_emb,
             np.zeros((max_size - sampled_img_emb.shape[0], *sampled_img_emb.shape[1:]),
                      dtype=sampled_img_emb.dtype)],
            axis=0,
        )
        sampled_pos_emb = np.concatenate(
            [sampled_pos_emb,
             np.zeros((max_size - sampled_pos_emb.shape[0], *sampled_pos_emb.shape[1:]),
                      dtype=sampled_pos_emb.dtype)],
            axis=0,
        )
        sampled_state_emb = np.concatenate(
            [sampled_state_emb,
             np.zeros((max_size - sampled_state_emb.shape[0], *sampled_state_emb.shape[1:]),
                      dtype=sampled_state_emb.dtype)],
            axis=0,
        )
        mask = np.concatenate([mask, np.zeros((max_size - mask.shape[0]), dtype=np.bool_)], axis=0)
    else:
        sampled_img_emb = sampled_img_emb[:max_size]
        sampled_pos_emb = sampled_pos_emb[:max_size]
        sampled_state_emb = sampled_state_emb[:max_size]
        mask = mask[:max_size]
    return sampled_img_emb, sampled_pos_emb, sampled_state_emb, mask


def left_padding_token_emb(
    recur_image_emb,    # (l v p d1)
    recur_pos_emb,      # (l v p d2)
    recur_state_emb,    # (l d3)
    recur_mask,         # (l)
    max_recur_steps: int,
):
    if recur_image_emb.shape[0] < max_recur_steps:
        recur_image_emb = np.concatenate(
            [np.zeros((max_recur_steps - recur_image_emb.shape[0], *recur_image_emb.shape[1:]),
                      dtype=recur_image_emb.dtype),
             recur_image_emb],
            axis=0,
        )
        recur_pos_emb = np.concatenate(
            [np.zeros((max_recur_steps - recur_pos_emb.shape[0], *recur_pos_emb.shape[1:]),
                      dtype=recur_pos_emb.dtype),
             recur_pos_emb],
            axis=0,
        )
        recur_state_emb = np.concatenate(
            [np.zeros((max_recur_steps - recur_state_emb.shape[0], *recur_state_emb.shape[1:]),
                      dtype=recur_state_emb.dtype),
             recur_state_emb],
            axis=0,
        )
        recur_mask = np.concatenate(
            [np.zeros(max_recur_steps - recur_mask.shape[0], dtype=np.bool_), recur_mask],
            axis=0,
        )
    else:
        recur_image_emb = recur_image_emb[-max_recur_steps:]
        recur_pos_emb = recur_pos_emb[-max_recur_steps:]
        recur_state_emb = recur_state_emb[-max_recur_steps:]
        recur_mask = recur_mask[-max_recur_steps:]

    return recur_image_emb, recur_pos_emb, recur_state_emb, recur_mask


def pool_tokens_to_size(
    tokens,             # (b v p d) or (b p d)
    target_size: int = 64,
    pool_type: str = "mean",
):
    if len(tokens.shape) == 4:
        b, v, p, d = tokens.shape
    elif len(tokens.shape) == 3:
        b, p, d = tokens.shape
    else:
        raise ValueError(f"Invalid tokens shape: {tokens.shape}")
    if p == target_size:
        return tokens

    h = w = int(math.sqrt(p))
    if len(tokens.shape) == 4:
        tokens_2d = einops.rearrange(tokens, "b v (h w) d -> (b v) h w d", h=h, w=w)
    else:
        tokens_2d = einops.rearrange(tokens, "b (h w) d -> b h w d", h=h, w=w)

    pool_size = int(math.sqrt(p // target_size))
    # Lazy import — ``flax`` is JAX-only; non-JAX consumers don't need it.
    import flax.nnx as nnx
    if pool_type == "mean":
        pool_func = nnx.avg_pool
    elif pool_type == "max":
        pool_func = nnx.max_pool
    else:
        raise ValueError(f"Invalid pool type: {pool_type}")

    pooled = pool_func(tokens_2d, window_shape=(pool_size, pool_size), strides=(pool_size, pool_size))
    if len(tokens.shape) == 4:
        out = einops.rearrange(pooled, "(b v) h w d -> b v (h w) d", b=b, v=v)
    else:
        out = einops.rearrange(pooled, "b h w d -> b (h w) d", b=b)

    return out

import numpy as np
import re
from pathlib import Path
import math

_TOKEN_EMB_RE = re.compile(r"^token_emb_(\d+)_(.+)\.npy$")
_TRAJ_DIR_RE = re.compile(r"^\d{4}$")

def _uniform_offset_indices(last_index: int, max_frames: int) -> list[int]:
    """Evenly spaced offsets in ``[0, last_index]``."""
    if max_frames <= 0 or last_index < 0:
        return []
    if last_index + 1 <= max_frames:
        return list(range(last_index + 1))
    return np.linspace(0, last_index, max_frames, dtype=np.int32).tolist()


def sample_demonstration_timesteps(last_index: int, max_frames: int,
                                   demonstration_sampling_type: str,
                                   key_frames: list[int] | None = None,
                                   rng: "np.random.Generator | None" = None,
                                   ) -> list[int]:
    """Pick demonstration-frame offsets in ``[0, last_index]``.

    Returned values are offsets (the caller adds ``t_lo`` to map back to the
    cached demo's absolute timesteps). At most ``max_frames`` entries.

    Modes:

    * ``"uniform_frame_sampling"`` — evenly spaced. ``key_frames`` ignored.
    * ``"keyframe_based_sampling"`` — use ``key_frames`` directly. The caller
      must pre-filter / shift them to the same offset space as ``last_index``
      (i.e. relative to ``t_lo``). Out-of-range entries and duplicates are
      dropped here.

      * If ``len(keys) > max_frames``: uniformly subsample the (sorted)
        keyframe list down (preserves first + last).
      * If ``len(keys) == max_frames``: return them as-is.
      * If ``len(keys) < max_frames``: **always** keep all keyframes, then
        fill the remaining ``max_frames - len(keys)`` slots from the
        non-keyframe offsets by weighted random sampling without
        replacement. Each candidate's weight is ``1 / distance_to_nearest_keyframe``,
        so frames close to keyframes are more likely picks — the extras
        densify around the keyframes rather than filling empty gaps.
        ``rng`` controls the draw; if ``None`` we derive a stable seed
        from ``(last_index, max_frames, keys)`` so a given demo's fill is
        reproducible across epochs.

      Raises ``ValueError`` if ``key_frames`` is missing / empty / has no
      entries in-range — no silent fallback to uniform, so partially
      annotated datasets fail loudly at train time.
    """
    if demonstration_sampling_type == "uniform_frame_sampling":
        return _uniform_offset_indices(last_index, max_frames)
    if demonstration_sampling_type == "keyframe_based_sampling":
        if max_frames <= 0 or last_index < 0:
            return []
        if not key_frames:
            raise ValueError(
                "keyframe_based_sampling requires non-empty key_frames; "
                "got None / empty. The demo likely has no keyframe annotation "
                "JSON under <data_root>/bpp_keyframe_annotations/."
            )
        keys = sorted({int(k) for k in key_frames if 0 <= int(k) <= last_index})
        if not keys:
            raise ValueError(
                f"keyframe_based_sampling: no key_frames fell within the demo's "
                f"timestep span [0, {last_index}] (raw key_frames={list(key_frames)}). "
                "Check that the JSON's keyframe_indices match the demo's video frames."
            )
        if len(keys) > max_frames:
            # More keyframes than budget — uniformly downsample the (already
            # sorted) keyframe list. Guaranteed to keep first + last because
            # ``np.linspace(0, n-1, max_frames)`` hits both endpoints.
            picks = np.linspace(0, len(keys) - 1, max_frames).round().astype(np.int32)
            seen, out = set(), []
            for i in picks.tolist():
                if i in seen:
                    continue
                seen.add(i)
                out.append(keys[i])
            return out

        # len(keys) <= max_frames — guarantee every keyframe is in, then
        # fill the remaining slots from the non-keyframe offsets via weighted
        # random sampling without replacement. The weight of each candidate
        # is inversely proportional to its distance to the nearest keyframe,
        # so frames close to keyframes are more likely picks (densifying
        # around important moments rather than spreading into empty gaps).
        # Determinism: when no rng is supplied we derive a stable seed from
        # ``(last_index, max_frames, keys)`` so each demo's fill is
        # reproducible across epochs.
        selected_set: set[int] = set(keys)
        candidates = np.fromiter(
            (i for i in range(last_index + 1) if i not in selected_set),
            dtype=np.int64,
        )
        budget_left = max_frames - len(keys)
        if budget_left <= 0 or candidates.size == 0:
            return sorted(selected_set)

        sel = np.asarray(keys, dtype=np.int64)
        # min distance from each candidate to the keyframe set; >= 1 because
        # all keyframes are already excluded from ``candidates``.
        dists_to_kf = np.min(np.abs(candidates[:, None] - sel[None, :]),
                             axis=1).astype(np.float64)
        weights = 1.0 / dists_to_kf
        weights /= weights.sum()

        n_to_pick = int(min(budget_left, candidates.size))
        if rng is None:
            seed = abs(hash((int(last_index), int(max_frames),
                             tuple(int(k) for k in keys)))) & 0xFFFFFFFF
            rng = np.random.default_rng(seed)
        picked = rng.choice(candidates.size, size=n_to_pick,
                            replace=False, p=weights)
        for i in picked.tolist():
            selected_set.add(int(candidates[i]))
        return sorted(selected_set)
    raise NotImplementedError(
        f"Demonstration sampling type {demonstration_sampling_type!r} not implemented "
        f"(supported: 'uniform_frame_sampling', 'keyframe_based_sampling')."
    )

_TOKEN_EMB_RE = re.compile(r"^token_emb_(\d+)_(.+)\.npy$")


def _load_token_emb_dict(path: Path) -> dict:
    raw = np.load(path, allow_pickle=True)
    feats = raw.item() if raw.shape == () else raw
    if not isinstance(feats, dict):
        raise TypeError(f"Expected dict in {path}, got {type(feats)}")
    return feats


def load_cached_demo_sample(
    demo_dir: Path | str,
    demo_cameras: list[str] | tuple[str, ...] | str,
    spatial: str,
) -> dict[str, np.ndarray]:
    """Build the dataset-style demo dict from cached SigLIP token features.

    Supports two on-disk formats:

    * **Consolidated** (current — written by the post-data-fix pipeline): one
      ``<camera>.npy`` per demo containing a single dict with keys
      ``image_emb_<spatial>`` ``(T, P, D)``, ``pos_emb_<spatial>`` ``(T, P, Dpos)``,
      ``state_emb`` ``(T, D_state)``, ``timesteps`` ``(T,)``.
    * **Per-timestep** (legacy): one ``token_emb_<t>_<camera>.npy`` per
      ``(timestep, camera)`` containing a single-row version of the same dict.

    For both formats the return shape matches ``LfOVLADataset.__getitem__``:
    ``demo_image_emb_<spatial>`` has shape ``(T, len(demo_cameras), P, D)`` so it
    feeds straight into ``get_demonstration_tokens``. A single string is
    accepted for ``demo_cameras`` (wrapped in a list).
    """
    demo_dir = Path(demo_dir)
    if isinstance(demo_cameras, str):
        demo_cameras = [demo_cameras]
    demo_cameras = list(demo_cameras)
    if not demo_cameras:
        raise ValueError("demo_cameras must contain at least one camera name.")

    img_key = f"image_emb_{spatial}"
    pos_key = f"pos_emb_{spatial}"

    # --- new consolidated format -------------------------------------------
    if (demo_dir / f"{demo_cameras[0]}.npy").is_file():
        per_cam: list[dict] = []
        for cam in demo_cameras:
            p = demo_dir / f"{cam}.npy"
            if not p.is_file():
                raise FileNotFoundError(f"Missing demo feature file: {p}")
            per_cam.append(_load_token_emb_dict(p))
        # Each per_cam[i][img_key] is (T, P, D); stack across views.
        img_stack = np.stack(
            [np.asarray(feats[img_key]) for feats in per_cam], axis=1
        )  # (T, num_views, P, D)
        pos_stack = np.stack(
            [np.asarray(feats[pos_key]) for feats in per_cam], axis=1
        )  # (T, num_views, P, Dpos)
        # state_emb is camera-independent; take from the first view.
        state_stack = np.asarray(per_cam[0]["state_emb"])  # (T, D_state)
        timesteps = np.asarray(per_cam[0]["timesteps"], dtype=np.int64)
        return {
            f"demo_image_emb_{spatial}": img_stack.astype(np.float32),
            f"demo_pos_emb_{spatial}": pos_stack.astype(np.float32),
            "demo_state_emb": state_stack.astype(np.float32),
            "demo_timesteps": timesteps,
        }

    # --- legacy per-timestep format ----------------------------------------
    ts_list: list[int] = []
    for p in demo_dir.glob(f"token_emb_*_{demo_cameras[0]}.npy"):
        m = _TOKEN_EMB_RE.fullmatch(p.name)
        if m:
            ts_list.append(int(m.group(1)))
    if not ts_list:
        raise FileNotFoundError(
            f"No demo features under {demo_dir} — expected either "
            f"{demo_cameras[0]}.npy (consolidated) or token_emb_*_{demo_cameras[0]}.npy (legacy)"
        )
    ts_list.sort()

    img_stack_list: list[np.ndarray] = []
    pos_stack_list: list[np.ndarray] = []
    state_stack_list: list[np.ndarray] = []
    for t in ts_list:
        view_imgs: list[np.ndarray] = []
        view_pos: list[np.ndarray] = []
        state_emb: np.ndarray | None = None
        for cam in demo_cameras:
            path = demo_dir / f"token_emb_{int(t)}_{cam}.npy"
            if not path.is_file():
                raise FileNotFoundError(f"Missing demo token file: {path}")
            feats = _load_token_emb_dict(path)
            view_imgs.append(np.asarray(feats[img_key]))   # (1, P, D)
            view_pos.append(np.asarray(feats[pos_key]))    # (1, P, Dpos)
            if state_emb is None:  # state_emb is camera-independent
                state_emb = np.asarray(feats["state_emb"])
        img_stack_list.append(np.concatenate(view_imgs, axis=0))   # (num_views, P, D)
        pos_stack_list.append(np.concatenate(view_pos, axis=0))    # (num_views, P, Dpos)
        state_stack_list.append(state_emb)

    return {
        f"demo_image_emb_{spatial}": np.stack(img_stack_list, axis=0).astype(np.float32),  # (T, num_views, P, D)
        f"demo_pos_emb_{spatial}": np.stack(pos_stack_list, axis=0).astype(np.float32),
        "demo_state_emb": np.stack(state_stack_list, axis=0).astype(np.float32),
        "demo_timesteps": np.asarray(ts_list, dtype=np.int64),
    }


def get_demonstration_tokens(sample: dict, demo_cfg) -> dict[str, np.ndarray]:
    """Map ``LfODataset`` stacks to perceptual ``static_*`` tensors (``uniform_frame_sampling`` layout).

    ``demo_cfg.frame_budget`` is the target number of **demonstration timesteps** (frames) after
    uniform sampling along ``demo_timesteps``. YAML ``budget`` must still be the total **token**
    count ``frame_budget * token_per_image * num_views`` so ``PerceptualDemonstrationEncoder`` and
    ``LfOPi0.inputs_spec`` stay aligned with the flattened sequence.

    ``sample_demonstration_timesteps`` picks up to ``frame_budget`` targets in timestep space; each
    maps to the loaded row with nearest ``demo_timesteps`` value. ``right_padding_token_emb`` pads
    to ``frame_budget`` rows.
    """

    tpi = int(demo_cfg.token_per_image)
    nv = int(demo_cfg.num_views)
    representation_type = str(demo_cfg.representation_type)
    demonstration_sampling_type = str(demo_cfg.demonstration_sampling_type)
    g = int(round(math.sqrt(tpi)))
    spatial = f"{g}x{g}"

    img = np.asarray(sample[f"demo_image_emb_{spatial}"], dtype=np.float32)
    pos = np.asarray(sample[f"demo_pos_emb_{spatial}"], dtype=np.float32)
    state = np.asarray(sample["demo_state_emb"], dtype=np.float32)

    if img.ndim == 3:
        img = img[:, None, :, :]
        pos = pos[:, None, :, :]
    if state.ndim == 1:
        state = state[:, None]
    if state.shape[0] != img.shape[0]:
        raise ValueError(
            f"demo_state_emb length {state.shape[0]} != number of demo frames {img.shape[0]}"
        )

    frame_budget = int(demo_cfg.frame_budget)
    token_budget_cfg = int(demo_cfg.budget)
    expected_tokens = frame_budget * tpi * nv
    # The perceptual encoder / inputs_spec are sized off ``budget`` (the flattened
    # frame×view×patch token count), so the YAML must keep it consistent.
    if token_budget_cfg != expected_tokens:
        raise ValueError(
            f"demonstration YAML: budget ({token_budget_cfg}) must equal "
            f"frame_budget * token_per_image * num_views = {frame_budget} * {tpi} * {nv} = {expected_tokens}"
        )
    # The number of demo views loaded into the sample (== len(demo_cameras)) must
    # match the YAML's num_views, else the flattened sequence won't match ``budget``.
    if img.shape[1] != nv:
        raise ValueError(
            f"loaded {img.shape[1]} demo view(s) but the demonstration YAML has num_views={nv}; "
            "set num_views (and budget = frame_budget * token_per_image * num_views) to match len(demo_cameras)"
        )
    if img.shape[2] != tpi:
        raise ValueError(
            f"offline demo has {img.shape[2]} patches / frame but token_per_image={tpi} "
            "(set LfODataset demo_emb_spatial to match the demonstration YAML, e.g. 4x4 for 16 tokens)"
        )

    demo_ts = np.asarray(sample["demo_timesteps"], dtype=np.int64)

    t_lo = int(np.min(demo_ts))
    t_hi = int(np.max(demo_ts))
    span = t_hi - t_lo
    if span < 0:
        raise ValueError("demo_timesteps: max < min")
    # Pull this demo's keyframe annotation if the dataset injected one (see
    # LfOVLADataset). Frames are stored as absolute original-video indices;
    # convert to offsets relative to ``t_lo`` so they live in the same space
    # as the offsets ``sample_demonstration_timesteps`` returns.
    raw_kf = sample.get("demo_key_frame_indices")
    key_frame_offsets: list[int] | None = None
    if raw_kf is not None:
        kf_arr = np.asarray(raw_kf, dtype=np.int64).reshape(-1)
        if kf_arr.size > 0:
            key_frame_offsets = [int(k - t_lo) for k in kf_arr.tolist()
                                 if t_lo <= int(k) <= t_hi]
    off_list = sample_demonstration_timesteps(
        span, frame_budget, demonstration_sampling_type,
        key_frames=key_frame_offsets,
    )
    if not off_list:
        raise ValueError("sample_demonstration_timesteps returned empty selection")
    offs = np.asarray(off_list, dtype=np.int64)
    targets = t_lo + offs
    sel_rows = [int(np.argmin(np.abs(demo_ts - tgt))) for tgt in targets]
    dedup: list[int] = []
    for r in sel_rows:
        if not dedup or dedup[-1] != r:
            dedup.append(r)
    sel = np.asarray(dedup, dtype=np.int64)
    img = img[sel]
    pos = pos[sel]
    state = state[sel]

    mask = np.ones((img.shape[0],), dtype=np.bool_)
    # for recur_representation and mask, we just directly use the sampled_img and mask without flattening
    sampled_img, sampled_pos, sampled_state, mask = right_padding_token_emb(
        img, pos, state, mask, frame_budget,
    )
    img_emb = np.reshape(sampled_img, (-1, demo_cfg.demonstration_feature.img.input_dim))
    pos_emb = np.reshape(sampled_pos, (-1, demo_cfg.demonstration_feature.pos.input_dim))
    mask_flat = np.repeat(mask, nv * tpi)
    state_emb = np.repeat(sampled_state, nv * tpi, axis=0)
    # Use the same image, pos, and state embeddings for static and recurrent demonstrations because the recurrent representation comes from the same static demonstration and does not get updated over demonstration.
    return {
            "static_image_emb": img_emb,
            "static_pos_emb": pos_emb,
            "static_state_emb": state_emb,
            "static_mask": mask_flat,
            "recur_image_emb": sampled_img,
            "recur_pos_emb": sampled_pos,
            "recur_state_emb": sampled_state,
            "recur_mask": mask
        }