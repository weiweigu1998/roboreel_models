#!/usr/bin/env python3
"""Full fine-tune (FFT) lerobot's Pi0.5 on the LfO robot trajectories.

This is the *plain* (language-conditioned, no demo-video) Pi0.5 baseline. The
policy is fully fine-tuned from ``lerobot/pi05_base`` on our
``postprocessed_robot_trajectories`` — base + hand (wrist) camera, robot state,
and the action chunk. The language instruction for each robot sample is drawn
from the VLM autoannotations of the *same task*
(``human_video_annotation_qwen3-vl-4b.json``). ``--annotations`` can point at
any annotations file in the same format.

FFT (full fine-tune) — every parameter trains; no PEFT/LoRA wrap.
``policy.save_pretrained(<step>/)`` dumps the full model + config at each
save interval, so eval loads it back with ``PI05Policy.from_pretrained``
directly (no base + adapter merge).

Run in the ``lerobot_pi05`` conda env (lerobot 0.5.2 + ``[pi]``, python
3.12). The ``lerobot`` library lives outside this repo at ``$LEROBOT_ROOT`` (a clone kept outside this repo).
"""

from __future__ import annotations

import argparse
import json
import os
import pickle
import random
import time
from collections import defaultdict
from pathlib import Path

import numpy as np
import torch
from torch.utils.data import DataLoader, Dataset

from lerobot.configs.policies import PreTrainedConfig
from lerobot.configs.types import FeatureType, NormalizationMode, PolicyFeature
from lerobot.policies.pi05 import PI05Policy, make_pi05_pre_post_processors
from lerobot.utils.constants import ACTION, OBS_IMAGES, OBS_STATE

ROBOT_SUBDIR = "postprocessed_robot_trajectories"


# ---------------------------------------------------------------------------
# data
# ---------------------------------------------------------------------------
def _infer_entry_bucket(entry: dict) -> str:
    """Fallback bucket label for legacy index entries without a "bucket" field."""
    return entry.get("bucket") or "without_distraction"


def _load_annotations(path: Path, video_types, max_demo_index: int
                      ) -> dict[str, dict[str, dict[str, str]]]:
    """Return ``{bucket: {task: {demo_id: instruction}}}`` from an annotations
    json, filtered to ``video_types`` and capped at ``max_demo_index``.

    Key format in the annotation file: ``<bucket>/<task>/<demo>``.
    """
    data = json.loads(path.read_text())
    out: dict[str, dict[str, dict[str, str]]] = defaultdict(lambda: defaultdict(dict))
    allowed = set(video_types)
    for key, text in data.get("annotations", {}).items():
        parts = key.split("/")
        if len(parts) == 3:
            bucket, task, demo = parts
        elif len(parts) == 2:
            # Flat bucket — task layer omitted; use bucket name as synthetic task.
            bucket, demo = parts
            task = bucket
        else:
            continue
        if bucket not in allowed:
            continue
        try:
            if int(demo) > max_demo_index:
                continue
        except ValueError:
            continue
        out[bucket][task][demo] = text
    return {b: {t: dict(d) for t, d in by_task.items()} for b, by_task in out.items()}


class LfORobotPi05Dataset(Dataset):
    """Yields lerobot-style raw batches from postprocessed_robot_trajectories.

    Each item: per-camera image tensors (C,H,W float [0,1]), ``observation.state``,
    the ``action`` chunk, and a ``task`` instruction. The instruction is a
    random annotation drawn from the same ``(bucket, task)``.

    The robot index is filtered by ``include_video_types`` so a bucket-specific
    condition (with-distraction only, say) doesn't pull the other bucket's
    pkls into the training set.
    """

    def __init__(self, data_root: str, robot_cameras: list[str],
                 annotations: dict[str, dict[str, dict[str, str]]],
                 include_video_types: list[str] | tuple[str, ...] = ("without_distraction",),
                 seed: int = 0):
        self.samples_root = Path(data_root) / ROBOT_SUBDIR
        self.data_dir = self.samples_root / "data"
        self.robot_cameras = list(robot_cameras)
        self.annotations = annotations
        self.include_video_types = tuple(include_video_types)
        self._rng = random.Random(seed)

        allowed = set(self.include_video_types)
        index_path = self.samples_root / "meta" / "index.jsonl"
        self.index: list[dict] = []
        with open(index_path) as f:
            for line in f:
                line = line.strip()
                if not line:
                    continue
                e = json.loads(line)
                if _infer_entry_bucket(e) not in allowed:
                    continue
                self.index.append(e)
        if not self.index:
            raise RuntimeError(f"No index entries for buckets {list(allowed)}")

        # Sanity: every (bucket, task) we'll sample must have annotations.
        seen = set()
        for e in self.index:
            seen.add((_infer_entry_bucket(e), e["task"]))
        missing = [(b, t) for (b, t) in sorted(seen)
                   if not annotations.get(b, {}).get(t)]
        if missing:
            raise RuntimeError(f"Missing annotations for (bucket, task): {missing}")

    def __len__(self) -> int:
        return len(self.index)

    def __getitem__(self, i: int) -> dict:
        entry = self.index[i]
        with open(self.data_dir / f"{entry['idx']}.pkl", "rb") as f:
            pkl = pickle.load(f)
        item: dict = {}
        for cam in self.robot_cameras:
            rgb = np.asarray(pkl[f"observation/{cam}"], dtype=np.float32) / 255.0
            item[f"{OBS_IMAGES}.{cam}"] = torch.from_numpy(rgb).permute(2, 0, 1).contiguous()
        item[OBS_STATE] = torch.from_numpy(np.asarray(pkl["state"], dtype=np.float32))
        item[ACTION] = torch.from_numpy(np.asarray(pkl["actions"], dtype=np.float32))
        bucket = _infer_entry_bucket(entry)
        anns = list(self.annotations[bucket][entry["task"]].values())
        item["task"] = anns[self._rng.randrange(len(anns))]
        return item


# ---------------------------------------------------------------------------
# normalization stats (Pi0.5 normalizes state/action with q01/q99 quantiles)
# ---------------------------------------------------------------------------
def compute_norm_stats(dataset: LfORobotPi05Dataset, num_samples: int,
                       cache_path: Path) -> dict:
    if cache_path.is_file():
        raw = json.loads(cache_path.read_text())
        return {k: {s: torch.tensor(v) for s, v in d.items()} for k, d in raw.items()}

    n = min(num_samples, len(dataset))
    idxs = random.Random(0).sample(range(len(dataset)), n)
    states, actions = [], []
    for j in idxs:
        item = dataset[j]
        states.append(item[OBS_STATE].numpy())
        actions.append(item[ACTION].numpy().reshape(-1, item[ACTION].shape[-1]))
    states = np.stack(states)               # (n, state_dim)
    actions = np.concatenate(actions, 0)    # (n*chunk, action_dim)

    def _stats(arr):
        return {
            "mean": arr.mean(0), "std": arr.std(0) + 1e-8,
            "min": arr.min(0), "max": arr.max(0),
            "q01": np.quantile(arr, 0.01, axis=0),
            "q99": np.quantile(arr, 0.99, axis=0),
        }

    stats = {OBS_STATE: _stats(states), ACTION: _stats(actions)}
    cache_path.write_text(json.dumps(
        {k: {s: v.tolist() for s, v in d.items()} for k, d in stats.items()}, indent=2))
    return {k: {s: torch.tensor(v, dtype=torch.float32) for s, v in d.items()}
            for k, d in stats.items()}


# ---------------------------------------------------------------------------
# main
# ---------------------------------------------------------------------------
def parse_args() -> argparse.Namespace:
    p = argparse.ArgumentParser(description=__doc__.splitlines()[0])
    p.add_argument("--data-root", default=os.environ.get("ROBOREEL_DATA"),
                   help="Dataset root (default: $ROBOREEL_DATA).")
    p.add_argument("--annotations", default=None,
                   help="Annotations json. Default "
                        "<data-root>/human_video_annotation_qwen3-vl-4b.json.")
    p.add_argument("--pretrained", default="lerobot/pi05_base")
    p.add_argument("--output-dir",
                   default=(os.path.join(os.environ["ROBOREEL_CKPT"], "lerobot_pi05_fft")
                            if os.environ.get("ROBOREEL_CKPT") else None),
                   help="Where to write checkpoints (default: $ROBOREEL_CKPT/lerobot_pi05_fft).")
    p.add_argument("--robot-cameras", nargs="+", default=["base_camera", "hand_camera"])
    p.add_argument("--include-video-types", nargs="+", default=["without_distraction"],
                   choices=["without_distraction", "with_distraction"],
                   help="Human-demo distraction buckets to draw instructions "
                        "from. Filters both the robot index and the annotation "
                        "pool.")
    p.add_argument("--max-demo-index", type=int, default=79,
                   help="Cap on demo id (e.g. 79 for the 80/20 split).")
    p.add_argument("--batch-size", type=int, default=32)
    p.add_argument("--num-train-steps", type=int, default=30000)
    p.add_argument("--lr", type=float, default=2.5e-5,
                   help="FFT LR. ~10× smaller than the LoRA equivalent because every "
                        "parameter updates each step.")
    p.add_argument("--num-workers", type=int, default=8)
    p.add_argument("--num-stats-samples", type=int, default=5000)
    p.add_argument("--save-interval", type=int, default=10000)
    p.add_argument("--log-interval", type=int, default=50)
    p.add_argument("--device", default="cuda:0")
    p.add_argument("--dtype", default="bfloat16", choices=["bfloat16", "float32"],
                   help="Compute dtype for the Pi0.5 base. bfloat16 is ~2× faster "
                        "and half the VRAM of float32 (pi05_base ships float32). FFT "
                        "needs more VRAM than LoRA — bfloat16 is strongly recommended.")
    p.add_argument("--gradient-checkpointing", action=argparse.BooleanOptionalAction,
                   default=True,
                   help="Recompute activations in the backward pass — large VRAM "
                        "saving on the 4B base, modest speed cost. On for FFT by "
                        "default; can be disabled if you have spare VRAM.")
    p.add_argument("--seed", type=int, default=42)
    return p.parse_args()


def main() -> None:
    args = parse_args()
    torch.manual_seed(args.seed)
    random.seed(args.seed)
    np.random.seed(args.seed)
    out_dir = Path(args.output_dir)
    out_dir.mkdir(parents=True, exist_ok=True)

    ann_path = Path(args.annotations) if args.annotations else \
        Path(args.data_root) / "human_video_annotation_qwen3-vl-4b.json"
    variant = "gt" if "ground_truth" in ann_path.name else "vlm"
    print(f"[pi05-fft] annotations <- {ann_path}  "
          f"(variant={variant}, types={args.include_video_types}, "
          f"max_demo_index={args.max_demo_index})", flush=True)
    annotations = _load_annotations(ann_path, args.include_video_types, args.max_demo_index)

    dataset = LfORobotPi05Dataset(
        args.data_root, args.robot_cameras, annotations,
        include_video_types=args.include_video_types,
        seed=args.seed,
    )
    print(f"[pi05-fft] {len(dataset)} robot samples, cameras={args.robot_cameras}", flush=True)

    print("[pi05-fft] computing/loading normalization stats ...", flush=True)
    stats = compute_norm_stats(dataset, args.num_stats_samples,
                               out_dir / "norm_stats.json")
    state_dim = stats[OBS_STATE]["mean"].shape[0]
    action_dim = stats[ACTION]["mean"].shape[0]
    chunk = int(dataset[0][ACTION].shape[0])
    print(f"[pi05-fft] state_dim={state_dim}, action_dim={action_dim}, chunk={chunk}", flush=True)

    # Config: load pi05_base's config (PreTrainedConfig dispatches on the
    # ``type`` discriminator to PI05Config), then point its features at our data.
    config = PreTrainedConfig.from_pretrained(args.pretrained)
    config.chunk_size = chunk
    config.n_action_steps = chunk
    config.device = args.device
    config.input_features = {
        f"{OBS_IMAGES}.{cam}": PolicyFeature(type=FeatureType.VISUAL, shape=(3, 224, 224))
        for cam in args.robot_cameras
    }
    config.input_features[OBS_STATE] = PolicyFeature(type=FeatureType.STATE, shape=(state_dim,))
    config.output_features = {ACTION: PolicyFeature(type=FeatureType.ACTION, shape=(action_dim,))}
    config.normalization_mapping = {
        "VISUAL": NormalizationMode.IDENTITY,
        "STATE": NormalizationMode.QUANTILES,
        "ACTION": NormalizationMode.QUANTILES,
    }
    config.pretrained_path = args.pretrained
    config.gradient_checkpointing = args.gradient_checkpointing
    config.dtype = args.dtype

    print(f"[pi05-fft] loading {args.pretrained} ...", flush=True)
    policy = PI05Policy.from_pretrained(args.pretrained, config=config)
    # FFT — train every parameter (no PEFT/LoRA wrap).
    for p_ in policy.parameters():
        p_.requires_grad_(True)
    policy = policy.to(args.device)
    policy.train()
    n_trainable = sum(p.numel() for p in policy.parameters() if p.requires_grad)
    n_total = sum(p.numel() for p in policy.parameters())
    print(f"[pi05-fft] FFT trainable params: {n_trainable/1e6:.2f}M / "
          f"{n_total/1e6:.1f}M (should be ~100%)", flush=True)

    preprocessor, _post = make_pi05_pre_post_processors(config, dataset_stats=stats)

    loader = DataLoader(dataset, batch_size=args.batch_size, shuffle=True,
                        num_workers=args.num_workers, drop_last=True, pin_memory=True)

    trainable = [p for p in policy.parameters() if p.requires_grad]
    optimizer = torch.optim.AdamW(trainable, lr=args.lr, betas=config.optimizer_betas,
                                  eps=config.optimizer_eps,
                                  weight_decay=config.optimizer_weight_decay)
    scheduler = torch.optim.lr_scheduler.CosineAnnealingLR(
        optimizer, T_max=args.num_train_steps, eta_min=args.lr * 0.1)

    with open(out_dir / "train_config.json", "w") as f:
        json.dump({**vars(args), "state_dim": state_dim, "action_dim": action_dim,
                   "chunk": chunk, "variant": variant,
                   "training_method": "fft"}, f, indent=2)

    print("[pi05-fft] training ...", flush=True)
    t0 = time.time()
    step = 0
    data_iter = iter(loader)
    while step < args.num_train_steps:
        try:
            batch = next(data_iter)
        except StopIteration:
            data_iter = iter(loader)
            batch = next(data_iter)
        step += 1

        batch = preprocessor(batch)
        loss, _ = policy.forward(batch)
        optimizer.zero_grad(set_to_none=True)
        loss.backward()
        torch.nn.utils.clip_grad_norm_(trainable, config.optimizer_grad_clip_norm)
        optimizer.step()
        scheduler.step()

        if step % args.log_interval == 0 or step == 1:
            print(f"[pi05-fft] step {step}/{args.num_train_steps}  loss={loss.item():.5f}  "
                  f"lr={scheduler.get_last_lr()[0]:.2e}  "
                  f"({(time.time()-t0)/step:.2f}s/step)", flush=True)
        if step % args.save_interval == 0 or step == args.num_train_steps:
            ckpt = out_dir / f"{step:06d}"
            # save_pretrained dumps the full model (config.json + weights) —
            # the eval wrapper loads it back with PI05Policy.from_pretrained.
            policy.save_pretrained(ckpt)
            print(f"[pi05-fft] checkpoint -> {ckpt}", flush=True)
    print("[pi05-fft] done.", flush=True)


if __name__ == "__main__":
    main()
