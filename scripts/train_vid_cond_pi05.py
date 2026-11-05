"""
Train the demonstration-augmented VLA policy on ManiSkill demonstration data.

For initial testing, freezes everything except the action expert to avoid OOM.

Reads the unified LfO layout via ``LfODataset`` (alias of ``LfOVLADataset``):

  data_dir/
    training_trajectories/<task>/<traj_id>/*.h5   ← robot BC rollouts
    video_demonstrations/<task>/<0000-0099>/token_emb_<t>_<camera>.npy

Offline SigLIP patch tokens are packed into ``static_*`` when the demonstration YAML is
**perceptual** and ``demonstration_sampling_type`` is ``uniform_frame_sampling`` (see ``_lfo_offline_to_pi0_static``).

Observation preprocessing matches ``eval_maniskill_vla.py``: 224² resize via
``demonstration_vla_suite.shared.maniskill_io.resize_image`` (INTER_LINEAR), and 8-D state via
``pack_robot_state_8d`` (qpos[:7] joints + mean(qpos[7:9]) gripper).

Usage:
  python scripts/train_vid_cond_pi05.py --config /path/to/train_config.json [--gpu N]

``--gpu N`` (alias ``--cuda-device``; accepts a comma-separated list like ``0,1,2,3`` for
multi-GPU FSDP) pins ``CUDA_VISIBLE_DEVICES`` before JAX initializes — useful for launching
many runs across a multi-GPU machine.

Config fields (all optional except checkpoint_dir and data_dir):
  checkpoint_dir    Path to base Pi0.5 checkpoint (required)
  data_dir          Root with ``training_trajectories`` and ``video_demonstrations`` (required)
  demonstration_config    Demonstration YAML under models/config/, or "none" to disable
                    (default: perceptual-framesamp-context.yaml)
  use_demonstration     When false, π₀.₅ is built without LfO (YAML path kept but ignored; default: true)
  output_dir        Output directory for checkpoints (default: runs/ckpts/vla_train)
  prompt            Task prompt (default: "complete the task")
  batch_size        Batch size (default: 2)
  num_train_steps   Total training steps (default: 1000)
  peak_lr           Peak learning rate (default: 5e-5)
  warmup_steps      LR warmup steps (default: 100)
  ema_decay         EMA decay factor (default: 0.999)
  grad_clip         Gradient clip norm (default: 1.0)
  seed              Random seed (default: 42)
  fsdp_devices      Number of FSDP devices (default: 1)
  log_interval      Steps between log lines (default: 10)
  save_interval     Steps between checkpoint saves (default: 500)
  wandb             Enable wandb logging (default: false)
  demo_cameras      One or more human-demo camera names for the token_emb files (default: ["front"])
  robot_cameras     One or more robot-side camera names for obs/sensor_data; each becomes an
                    ``image`` slot for the policy, in order
                    (default: ["base_camera", "left_camera", "right_camera", "hand_camera"])
  lfo_max_demo_index   Max numeric video folder id (default: 99)
"""

from __future__ import annotations

import dataclasses
import functools
import importlib.util
import json
import logging
import time
import math
import os
import platform
import sys
import types
from pathlib import Path

os.environ.setdefault("MUJOCO_GL", "egl")


def _pin_cuda_device_from_argv() -> None:
    """``--gpu``/``--cuda-device <id>`` → ``CUDA_VISIBLE_DEVICES``, applied before JAX init.

    Parsed here (not in ``_parse_cli``) because ``import jax`` below grabs the device list.
    """
    import argparse
    p = argparse.ArgumentParser(add_help=False)
    p.add_argument("--gpu", "--cuda-device", dest="gpu", default=None)
    gpu = p.parse_known_args()[0].gpu
    if gpu is not None:
        os.environ["CUDA_VISIBLE_DEVICES"] = str(gpu)


_pin_cuda_device_from_argv()

import jax
import jax.numpy as jnp
import numpy as np
import optax
import flax.nnx as nnx
import flax.traverse_util as traverse_util
from flax.training import common_utils

import tqdm_loggable.auto as tqdm

# ---------------------------------------------------------------------------
# Path setup
# ---------------------------------------------------------------------------
_PROJECT_ROOT = Path(__file__).resolve().parents[1]
for _p in (_PROJECT_ROOT, _PROJECT_ROOT / "src", _PROJECT_ROOT / "src" / "external_models"):
    _s = str(_p)
    if _s not in sys.path:
        sys.path.insert(0, _s)

# _DEMO_UTILS_PATH = _PROJECT_ROOT / "roboreel_env" / "src" / "utils" / "demonstration_utils.py"
# if not _DEMO_UTILS_PATH.is_file():
#     raise FileNotFoundError(
#         f"Expected {_DEMO_UTILS_PATH} for sample_demonstration_timesteps; "
#         "ensure roboreel_env is present under the project root."
#     )
# _spec_du = importlib.util.spec_from_file_location("lfo_benchmark_demonstration_utils", _DEMO_UTILS_PATH)
# if _spec_du is None or _spec_du.loader is None:
#     raise RuntimeError(f"Cannot load demonstration utils from {_DEMO_UTILS_PATH}")
# _demonstration_utils = importlib.util.module_from_spec(_spec_du)
# _spec_du.loader.exec_module(_demonstration_utils)
# sample_demonstration_timesteps = _demonstration_utils.sample_demonstration_timesteps

import openpi.shared.array_typing as at
import openpi.shared.nnx_utils as nnx_utils
import openpi.training.checkpoints as _checkpoints
import openpi.training.optimizer as _optimizer
import openpi.training.sharding as sharding
import openpi.training.utils as training_utils
import openpi.training.weight_loaders as _weight_loaders
import openpi.models.model as _model

from config_paths import expand_paths  # noqa: E402
from demonstration_vla_suite.dataset.lfo_vla_dataset import LfODataset
from demonstration_vla_suite.models.lfo_pi0 import LfOPi0Config
from demonstration_vla_suite.models.lfo_observation import HistAugObservation
from demonstration_vla_suite.models.config.utils import get_demonstration_config
from demonstration_vla_suite.shared.data_utils import get_demonstration_tokens
logger = logging.getLogger("train_vla")


# ---------------------------------------------------------------------------
# Freeze filter: freeze everything EXCEPT the action expert
# ---------------------------------------------------------------------------

def make_action_expert_only_freeze_filter():
    """Freeze everything except:
    - action_in_proj, action_out_proj (action projection heads)
    - time_mlp_in, time_mlp_out (Pi0.5 time MLP)
    - llm.*_1.* (action expert Gemma blocks)
    - demonstration_encoder.* (demonstration representation modules)

    This keeps the VLM backbone (PaliGemma), SigLip vision encoder, and
    language Gemma blocks frozen.
    """
    # What we want trainable:
    action_expert_gemma = nnx_utils.PathRegex(".*llm.*_1.*")
    action_projections = nnx_utils.PathRegex(".*(action_in_proj|action_out_proj|time_mlp_in|time_mlp_out).*")
    demonstration_encoder = nnx_utils.PathRegex(".*(mem_encoder|demonstration_encoder).*")

    trainable = nnx.Any(action_expert_gemma, action_projections, demonstration_encoder)
    # freeze_filter = everything that is NOT trainable
    return nnx.Not(trainable)


# ---------------------------------------------------------------------------
# LfODataset → Pi0 static demonstration tensors
# ---------------------------------------------------------------------------



# ---------------------------------------------------------------------------
# Data pipeline: transforms + batching
# ---------------------------------------------------------------------------


def _stack_leaf(vals):
    """Stack a list of leaf values into a batched array."""
    if vals[0] is None:
        return None
    if isinstance(vals[0], dict):
        return {k: _stack_leaf([v[k] for v in vals]) for k in vals[0]}
    if isinstance(vals[0], np.ndarray):
        return jnp.array(np.stack(vals))
    if isinstance(vals[0], (np.bool_, bool)):
        # Broadcast scalar mask to (batch,) array
        return jnp.array(np.array(vals, dtype=np.bool_))
    if isinstance(vals[0], (int, float, np.integer, np.floating)):
        return jnp.array(np.array(vals))
    return vals[0]


def collate_to_jax(samples: list[dict]) -> tuple[HistAugObservation, jnp.ndarray]:
    """Collate a list of transformed sample dicts into a (HistAugObservation, actions) batch."""
    batch = {}
    for key in samples[0]:
        vals = [s[key] for s in samples]
        batch[key] = _stack_leaf(vals)

    actions = batch.pop("actions")
    observation = HistAugObservation.from_dict(batch)
    return observation, actions


class SimpleDataLoader:
    """Minimal data loader that wraps a dataset with shuffling and batching."""

    def __init__(self, dataset, transform_fn, batch_size: int, seed: int = 0):
        self.dataset = dataset
        self.transform_fn = transform_fn
        self.batch_size = batch_size
        self.rng = np.random.RandomState(seed)
        self.indices = np.arange(len(dataset))

    def __iter__(self):
        while True:
            self.rng.shuffle(self.indices)
            for start in range(0, len(self.indices) - self.batch_size + 1, self.batch_size):
                batch_indices = self.indices[start : start + self.batch_size]
                samples = []
                for idx in batch_indices:
                    raw = self.dataset[idx]
                    transformed = self.transform_fn(raw)
                    samples.append(transformed)
                yield collate_to_jax(samples)


# ---------------------------------------------------------------------------
# Weight loading
# ---------------------------------------------------------------------------

def load_and_merge_weights(
    weight_loader: _weight_loaders.WeightLoader,
    params_shape: at.Params,
) -> at.Params:
    """Load base model weights and merge, allowing missing keys (new params)."""
    loaded_params = weight_loader.load(params_shape)
    at.check_pytree_equality(
        expected=params_shape, got=loaded_params, check_shapes=True, check_dtypes=True
    )
    return traverse_util.unflatten_dict({
        k: v
        for k, v in traverse_util.flatten_dict(loaded_params).items()
        if not isinstance(v, jax.ShapeDtypeStruct)
    })


# ---------------------------------------------------------------------------
# Training state init
# ---------------------------------------------------------------------------

def init_train_state(
    model_config: LfOPi0Config,
    freeze_filter,
    trainable_filter,
    tx: optax.GradientTransformation,
    weight_loader: _weight_loaders.WeightLoader,
    init_rng,
    mesh,
    ema_decay: float | None = None,
):
    def init(rng, partial_params=None):
        rng, model_rng = jax.random.split(rng)
        model = model_config.create(model_rng)

        if partial_params is not None:
            graphdef, state = nnx.split(model)
            state.replace_by_pure_dict(partial_params)
            model = nnx.merge(graphdef, state)

        params = nnx.state(model)
        # Frozen params to bfloat16
        params = nnx_utils.state_map(
            params, freeze_filter,
            lambda p: p.replace(p.value.astype(jnp.bfloat16)),
        )

        total_size = sum(x.size for x in jax.tree_util.tree_leaves(params))
        trainable_size = sum(x.size for x in jax.tree_util.tree_leaves(params.filter(trainable_filter)))
        logging.info(f"Total params: {total_size / 1e6:.1f}M, Trainable: {trainable_size / 1e6:.1f}M")

        return training_utils.TrainState(
            step=0,
            params=params,
            model_def=nnx.graphdef(model),
            tx=tx,
            opt_state=tx.init(params.filter(trainable_filter)),
            ema_decay=ema_decay,
            ema_params=None if ema_decay is None else params,
        )

    train_state_shape = jax.eval_shape(init, init_rng)
    state_sharding = sharding.fsdp_sharding(train_state_shape, mesh, log=True)

    partial_params = load_and_merge_weights(weight_loader, train_state_shape.params.to_pure_dict())

    replicated_sharding = jax.sharding.NamedSharding(mesh, jax.sharding.PartitionSpec())
    train_state = jax.jit(
        init,
        donate_argnums=(1,),
        in_shardings=replicated_sharding,
        out_shardings=state_sharding,
    )(init_rng, partial_params)

    return train_state, state_sharding


# ---------------------------------------------------------------------------
# Train step
# ---------------------------------------------------------------------------

def train_step(
    trainable_filter,
    rng,
    state: training_utils.TrainState,
    batch: tuple[HistAugObservation, jnp.ndarray],
):
    model = nnx.merge(state.model_def, state.params)
    model.train()

    def loss_fn(model, rng, observation, actions):
        chunked_loss, aux = model.compute_loss(rng, observation, actions, train=True)
        return jnp.mean(chunked_loss), aux

    train_rng = jax.random.fold_in(rng, state.step)
    observation, actions = batch

    diff_state = nnx.DiffState(0, trainable_filter)
    (loss, _), grads = nnx.value_and_grad(loss_fn, argnums=diff_state, has_aux=True)(
        model, train_rng, observation, actions
    )

    params = state.params.filter(trainable_filter)
    updates, new_opt_state = state.tx.update(grads, state.opt_state, params)
    new_params = optax.apply_updates(params, updates)

    nnx.update(model, new_params)
    new_params = nnx.state(model)

    new_state = dataclasses.replace(
        state, step=state.step + 1, params=new_params, opt_state=new_opt_state
    )

    if state.ema_decay is not None:
        new_state = dataclasses.replace(
            new_state,
            ema_params=jax.tree.map(
                lambda old, new: state.ema_decay * old + (1 - state.ema_decay) * new,
                state.ema_params, new_params,
            ),
        )

    info = {
        "loss": loss,
        "grad_norm": optax.global_norm(grads),
    }
    return new_state, info


# ---------------------------------------------------------------------------
# Training config builder
# ---------------------------------------------------------------------------

def build_train_config(args):
    """Build a TrainConfig-like object with data pipeline from CLI args."""
    from demonstration_vla_suite.policies.transforms import DemoVLAInputs, DemoVLAOutputs
    import openpi.transforms as transforms
    import openpi.models.model as model_mod

    model_config = LfOPi0Config(
        pi05=True,
        action_horizon=20,
        action_dim=32,   # matches Pi0.5 base checkpoint; PadStatesAndActions pads our 8-dim actions
        demonstration_config=args.demonstration_config if args.demonstration_config != "none" else None,
        use_demonstration=bool(getattr(args, "use_demonstration", True)),
        discrete_state_input=False,
        max_token_len=64,
    )

    # Repack transforms
    repack_transform = transforms.Group(
        inputs=[
            transforms.RepackTransform({
                "observation/image": "image",
                "observation/state": "state",
                "actions": "actions",
                "prompt": "prompt",
                "static_image_emb": "static_image_emb",
                "static_pos_emb": "static_pos_emb",
                "static_state_emb": "static_state_emb",
                "static_mask": "static_mask",
                "recur_image_emb": "recur_image_emb",
                "recur_pos_emb": "recur_pos_emb",
                "recur_state_emb": "recur_state_emb",
                "recur_mask": "recur_mask",
            })
        ]
    )

    data_transforms = transforms.Group(
        inputs=[DemoVLAInputs(model_type=model_mod.ModelType.PI05, robot_cameras=tuple(args.robot_cameras))],
        outputs=[DemoVLAOutputs()],
    )
    delta_action_mask = transforms.make_bool_mask(7, -1)
    data_transforms = data_transforms.push(
        inputs=[transforms.DeltaActions(delta_action_mask)],
        outputs=[transforms.AbsoluteActions(delta_action_mask)],
    )

    # Model transforms (tokenization)
    from openpi.shared import download
    import sentencepiece

    class PaligemmaTokenizer:
        def __init__(self, max_len: int = 64):
            self._max_len = max_len
            path = download.maybe_download("gs://big_vision/paligemma_tokenizer.model", gs={"token": "anon"})
            with path.open("rb") as f:
                self._tokenizer = sentencepiece.SentencePieceProcessor(model_proto=f.read())

        def tokenize(self, prompt, state=None, subgoal=None):
            cleaned = prompt.strip().replace("_", " ").replace("\n", " ")
            tokens = self._tokenizer.encode(cleaned, add_bos=True) + self._tokenizer.encode("\n")
            tokens_len = len(tokens)
            if tokens_len < self._max_len:
                padding = [False] * (self._max_len - tokens_len)
                mask = [True] * tokens_len + padding
                tokens = tokens + padding
            else:
                tokens = tokens[: self._max_len]
                mask = [True] * self._max_len
            return np.asarray(tokens), np.asarray(mask)

    model_transforms = transforms.Group(
        inputs=[
            transforms.ResizeImages(224, 224),
            transforms.TokenizePrompt(PaligemmaTokenizer(model_config.max_token_len)),
            transforms.PadStatesAndActions(model_config.action_dim),
        ],
    )

    return model_config, repack_transform, data_transforms, model_transforms


# ---------------------------------------------------------------------------
# CLI
# ---------------------------------------------------------------------------

_TRAIN_DEFAULTS = {
    "demonstration_config": "perceptual-framesamp-context.yaml",
    "use_demonstration": True,
    "prompt": "complete the task",
    "batch_size": 2,
    "num_train_steps": 1000,
    "peak_lr": 5e-5,
    "warmup_steps": 100,
    "ema_decay": 0.999,
    "grad_clip": 1.0,
    "seed": 42,
    "fsdp_devices": 1,
    "log_interval": 10,
    "save_interval": 500,
    "wandb": False,
    "lfo_max_demo_index": 99,
    "include_video_types": ["without_distraction"],
    "keyframe_annotation_dir": None,
    "demo_cameras": ["front"],
    "robot_cameras": ["base_camera", "left_camera", "right_camera", "hand_camera"],
}

_TRAIN_REQUIRED = {"checkpoint_dir", "data_dir"}


def load_config(config_path: str):
    import argparse
    with open(config_path) as f:
        cfg = expand_paths(json.load(f))
    if "history_config" in cfg and "demonstration_config" not in cfg:
        cfg["demonstration_config"] = cfg.pop("history_config")
    elif "history_config" in cfg:
        cfg.pop("history_config", None)
    # Back-compat: accept the deprecated singular ``demo_camera``.
    if "demo_camera" in cfg and "demo_cameras" not in cfg:
        cfg["demo_cameras"] = [cfg.pop("demo_camera")]
    else:
        cfg.pop("demo_camera", None)
    missing = _TRAIN_REQUIRED - set(cfg)
    if missing:
        raise ValueError(f"Missing required config fields: {missing}")
    merged = {**_TRAIN_DEFAULTS, "output_dir": str(_PROJECT_ROOT / "runs" / "ckpts" / "vla_train"), **cfg}
    merged["data_dir"] = os.path.expanduser(os.path.normpath(merged["data_dir"]))
    return argparse.Namespace(**merged)


def _parse_cli():
    import argparse
    p = argparse.ArgumentParser(description="Train demo-augmented VLA on action bench data")
    p.add_argument("--config", required=True, help="Path to JSON config file.")
    p.add_argument("--gpu", "--cuda-device", dest="gpu", default=None,
                   help="CUDA device id(s) to run on, e.g. '3' or '0,1,2,3' (sets CUDA_VISIBLE_DEVICES; "
                        "applied at import time, before JAX initializes).")
    return load_config(p.parse_args().config)


# ---------------------------------------------------------------------------
# Checkpointing (OpenPI save_state expects a DataLoader for norm_stats assets)
# ---------------------------------------------------------------------------

class _NoNormStatsDataLoaderStub:
    """``checkpoints.save_state`` calls ``data_loader.data_config()``; ManiSkill script has no OpenPI dataloader."""

    def data_config(self):
        return types.SimpleNamespace(norm_stats=None, asset_id=None)


# ---------------------------------------------------------------------------
# Main
# ---------------------------------------------------------------------------

def main():
    args = _parse_cli()

    logging.basicConfig(
        level=logging.INFO,
        format="%(asctime)s [%(levelname)s] %(message)s",
        datefmt="%H:%M:%S",
    )
    logging.info(f"Running on: {platform.node()}")
    logging.info(f"Devices: {jax.devices()}")
    logging.info(f"Args: {args}")

    if args.batch_size % jax.device_count() != 0:
        raise ValueError(
            f"Batch size {args.batch_size} must be divisible by device count {jax.device_count()}"
        )

    rng = jax.random.key(args.seed)
    train_rng, init_rng = jax.random.split(rng)

    # --- Build config and transforms ---
    model_config, _, data_transforms, model_transforms = build_train_config(args)

    import openpi.transforms as transforms

    use_lfo_demonstration = (
        args.demonstration_config not in (None, "none", "None")
        and bool(getattr(args, "use_demonstration", True))
    )

    demo_cfg = None
    spatial = "8x8"
    if use_lfo_demonstration:
        demo_cfg = get_demonstration_config(args.demonstration_config)
        g = int(round(math.sqrt(demo_cfg.token_per_image)))
        spatial = f"{g}x{g}"

    # Skip repack_transform: the dataset already provides observation/* keys
    # that DemoVLAInputs expects directly.
    base_input_transform = transforms.compose([
        *data_transforms.inputs,
        *model_transforms.inputs,
    ])

    def input_transform(raw):
        if demo_cfg is not None:
            raw = dict(raw)
            raw.update(get_demonstration_tokens(raw, demo_cfg))
        return base_input_transform(raw)

    dataset = LfODataset(
        args.data_dir,
        action_horizon=model_config.action_horizon,
        prompt=args.prompt,
        demo_cameras=args.demo_cameras,
        robot_cameras=args.robot_cameras,
        max_demo_index=int(args.lfo_max_demo_index),
        include_video_types=args.include_video_types,
        seed=args.seed,
        demo_emb_spatial=spatial,
        keyframe_annotation_dir=args.keyframe_annotation_dir,
    )
    logger.info(
        "LfODataset: demo_emb_spatial=%s (temporal subsampling in train_vid_cond_pi05 transform)",
        spatial,
    )

    # --- Freeze filter ---
    freeze_filter = make_action_expert_only_freeze_filter()
    trainable_filter = nnx.All(nnx.Param, nnx.Not(freeze_filter))
    # --- Optimizer ---
    lr_schedule = _optimizer.CosineDecaySchedule(
        warmup_steps=args.warmup_steps,
        peak_lr=args.peak_lr,
        decay_steps=args.num_train_steps,
        decay_lr=args.peak_lr * 0.1,
    )
    optimizer_config = _optimizer.AdamW(clip_gradient_norm=args.grad_clip)
    tx = _optimizer.create_optimizer(optimizer_config, lr_schedule, weight_decay_mask=None)

    # --- Weight loader ---
    checkpoint_path = Path(args.checkpoint_dir)
    params_path = checkpoint_path / "params"
    if not params_path.exists():
        # Maybe the checkpoint_dir already points to the params dir
        params_path = checkpoint_path
    weight_loader = _weight_loaders.CheckpointWeightLoader(str(params_path))

    # Wrap with missing key tolerance for new demonstration encoder params
    class MergeWeightLoader(_weight_loaders.WeightLoader):
        def __init__(self, inner):
            self._inner = inner

        def load(self, params):
            from openpi.training.weight_loaders import _merge_params
            loaded = _model.restore_params(str(params_path), restore_type=np.ndarray)
            return _merge_params(loaded, params, missing_regex=".*")

    weight_loader = MergeWeightLoader(weight_loader)

    # --- Mesh and train state ---
    mesh = sharding.make_mesh(args.fsdp_devices)
    data_sharding = jax.sharding.NamedSharding(mesh, jax.sharding.PartitionSpec(sharding.DATA_AXIS))
    replicated_sharding = jax.sharding.NamedSharding(mesh, jax.sharding.PartitionSpec())

    print(f"[milestone] Initializing train state (bs={args.batch_size}, fsdp_devices={args.fsdp_devices})...", flush=True)
    _t_init = time.time()
    train_state, state_sharding = init_train_state(
        model_config, freeze_filter, trainable_filter, tx, weight_loader,
        init_rng, mesh, ema_decay=args.ema_decay,
    )
    jax.block_until_ready(train_state)
    print(f"[milestone] Train state initialized in {time.time() - _t_init:.1f}s", flush=True)

    print("[milestone] Building data loader...", flush=True)
    _t_dl = time.time()
    data_loader = SimpleDataLoader(dataset, input_transform, batch_size=args.batch_size, seed=args.seed)
    data_iter = iter(data_loader)

    print("[milestone] Prefetching first batch...", flush=True)
    batch = next(data_iter)
    print(f"[milestone] First batch ready in {time.time() - _t_dl:.1f}s, actions shape={batch[1].shape}", flush=True)

    # --- Checkpointing ---
    output_dir = Path(args.output_dir)
    output_dir.mkdir(parents=True, exist_ok=True)

    checkpoint_manager, _ = _checkpoints.initialize_checkpoint_dir(
        output_dir, keep_period=args.save_interval, overwrite=True, resume=False,
    )
    checkpoint_data_loader = _NoNormStatsDataLoaderStub()

    # Persist the resolved run config (+ the task list seen by the dataset) so eval can be
    # driven by just the checkpoint path. Must come AFTER initialize_checkpoint_dir, which
    # wipes output_dir when overwrite=True; written before the training loop so it exists
    # even for runs that crash before the first checkpoint.
    _tasks = sorted({task for task, _h5, _t in dataset.samples})
    with open(output_dir / "train_config.json", "w") as _f:
        json.dump({**vars(args), "tasks": _tasks}, _f, indent=2)
    logging.info("Wrote train config (tasks=%s) → %s", _tasks, output_dir / "train_config.json")

    # Record the demonstration YAML name beside the checkpoint, as provenance
    # for which demonstration config the run was trained with.
    if args.demonstration_config not in (None, "none", "None") and bool(
        getattr(args, "use_demonstration", True)
    ):
        (output_dir / "demonstration_config.txt").write_text(args.demonstration_config)

    # --- JIT compile train step ---
    ptrain_step = jax.jit(
        functools.partial(train_step, trainable_filter),
        in_shardings=(replicated_sharding, state_sharding, data_sharding),
        out_shardings=(state_sharding, replicated_sharding),
        donate_argnums=(1,),
    )

    # --- Wandb ---
    if args.wandb:
        import wandb
        wandb.init(
            project="action_bench_vla",
            name=f"vla_{args.demonstration_config}",
            config=vars(args),
        )
    else:
        import wandb
        wandb.init(mode="disabled")

    # --- Training loop ---
    logging.info(f"Starting training for {args.num_train_steps} steps...")
    pbar = tqdm.tqdm(range(args.num_train_steps), dynamic_ncols=True)
    infos = []

    for step in pbar:
        with sharding.set_mesh(mesh):
            train_state, info = ptrain_step(train_rng, train_state, batch)
        infos.append(info)

        if step % args.log_interval == 0 and step > 0:
            stacked = common_utils.stack_forest(infos)
            reduced = jax.device_get(jax.tree.map(jnp.mean, stacked))
            info_str = ", ".join(f"{k}={v:.4f}" for k, v in reduced.items())
            pbar.write(f"Step {step}: {info_str}")
            wandb.log(reduced, step=step)
            infos = []

        batch = next(data_iter)

        if (step % args.save_interval == 0 and step > 0) or step == args.num_train_steps - 1:
            _checkpoints.save_state(checkpoint_manager, train_state, checkpoint_data_loader, step)
            logging.info(f"Saved checkpoint at step {step}")

    logging.info("Training complete.")
    checkpoint_manager.wait_until_finished()


if __name__ == "__main__":
    main()
