#!/usr/bin/env python3
"""Train Vid2Robot (Jain et al. 2024) on the LfO benchmark.

PyTorch re-implementation. The model follows Tables IV / V of the paper with
two LfO adaptations:
  * action head -> 8-D ``pd_joint_pos`` (7 arm joints + 1 gripper close in
    [-1, 1]), chunk_size = 4, 256 bins per dim (paper's 11-D Google-Robot
    space doesn't apply here since the LfO env is base-static).
  * prompt videos come from ``human_demos/<split>/<type>/<task>/<demo>/
    front.mp4`` and the robot history from ``observation/base_camera``,

Losses (all averaged): action CE, TCC, prompt-robot SigLIP, video-text SigLIP
(paper §II-E). The dataset encodes each task description with the frozen
SigLIP text tower at init time and returns the embedding per-item; the trainer
just reads ``batch["text_embedding"]``.

Config: JSON, see ``configs/train_vid2robot/nd.json``. Run inside the
``vid2robot`` env.
"""

from __future__ import annotations

import argparse
import json
import os
import sys
import time

import numpy as np
import torch
import torch.nn.functional as F
from torch.utils.data import DataLoader

REPO = os.path.abspath(os.path.join(os.path.dirname(__file__), ".."))
for p in (REPO, os.path.join(REPO, "src")):
    if p not in sys.path:
        sys.path.insert(0, p)

# numpy 2.x -> 1.x unpickle shim (the Stage-0a pkls were saved in numpy 2.x;
# torch envs may resolve numpy at 1.x or 2.x). See src/np2_pickle_compat.py.
from config_paths import expand_paths  # noqa: E402
import np2_pickle_compat  # noqa: E402,F401

from vid2robot.config import Vid2RobotConfig  # noqa: E402
from vid2robot.data.dataset import Vid2RobotDataset  # noqa: E402
from vid2robot.losses.siglip import SigLIPLoss  # noqa: E402
from vid2robot.losses.tcc import tcc_loss  # noqa: E402
from vid2robot.model import Vid2Robot  # noqa: E402


def parse_args() -> argparse.Namespace:
    p = argparse.ArgumentParser(description="Train Vid2Robot on the LfO benchmark.")
    p.add_argument("--config", required=True, help="Path to JSON training config.")
    p.add_argument("--smoke-test", action="store_true",
                   help="Run a single forward+backward pass and exit.")
    p.add_argument("--gpu", default=None, help="CUDA device id (sets CUDA_VISIBLE_DEVICES).")
    return p.parse_args()


def _to_device(batch: dict, device: torch.device) -> dict:
    out = {}
    for k, v in batch.items():
        if isinstance(v, torch.Tensor):
            out[k] = v.to(device, non_blocking=True)
        else:
            out[k] = v
    return out


def collate(batch: list[dict]) -> dict:
    """Custom collate: stack tensor fields, keep ``task`` and paths as lists."""
    out: dict = {"task": [b["task"] for b in batch]}
    out["prompt_video_path"] = [b.get("prompt_video_path") for b in batch]
    for key in ("prompt_video", "state_video", "action_bins", "text_embedding"):
        out[key] = torch.from_numpy(np.stack([b[key] for b in batch]))
    if "robot_full_video" in batch[0]:
        out["robot_full_video"] = torch.from_numpy(
            np.stack([b["robot_full_video"] for b in batch]))
    return out


def build_cosine_with_warmup(optimizer, total_steps: int, warmup_steps: int,
                             base_lr: float, final_lr: float):
    """Cosine schedule with linear warmup — matches Table V (8e-5 -> 1e-6)."""
    def lr_at(step: int) -> float:
        if step < warmup_steps:
            return base_lr * (step + 1) / max(warmup_steps, 1)
        progress = (step - warmup_steps) / max(total_steps - warmup_steps, 1)
        progress = min(max(progress, 0.0), 1.0)
        cos = 0.5 * (1.0 + np.cos(np.pi * progress))
        return final_lr + (base_lr - final_lr) * cos

    def step(step_idx: int) -> float:
        lr = lr_at(step_idx)
        for g in optimizer.param_groups:
            g["lr"] = lr
        return lr

    return step


def main() -> None:
    args = parse_args()
    if args.gpu is not None:
        os.environ["CUDA_VISIBLE_DEVICES"] = str(args.gpu)
    with open(args.config) as f:
        cfg = expand_paths(json.load(f))

    torch.manual_seed(int(cfg.get("seed", 42)))
    np.random.seed(int(cfg.get("seed", 42)))

    device = torch.device("cuda" if torch.cuda.is_available() else "cpu")
    print(f"[v2r] device={device}", flush=True)

    model_cfg = Vid2RobotConfig(
        action_dim=int(cfg.get("action_dim", 8)),
        chunk_size=int(cfg.get("chunk_size", 4)),
        num_bins=int(cfg.get("num_bins", 256)),
        num_prompt_frames=int(cfg.get("num_prompt_frames", 16)),
        num_state_frames=int(cfg.get("num_state_frames", 8)),
        image_size=int(cfg.get("image_size", 224)),
        seed=int(cfg.get("seed", 42)),
        gradient_checkpointing=bool(cfg.get("gradient_checkpointing", False)),
    )
    print(f"[v2r] config: {model_cfg}", flush=True)

    dataset = Vid2RobotDataset(
        data_root=cfg["data_root"],
        tasks=cfg["tasks"],
        samples_subdir=cfg.get("samples_subdir", "postprocessed_robot_trajectories"),
        robot_camera=cfg.get("robot_camera", "base_camera"),
        image_size=model_cfg.image_size,
        num_prompt_frames=model_cfg.num_prompt_frames,
        num_state_frames=model_cfg.num_state_frames,
        chunk_size=model_cfg.chunk_size,
        include_video_types=tuple(cfg.get("include_video_types", ["without_distraction"])),
        max_demo_index=cfg.get("max_demo_index"),
        seed=model_cfg.seed,
        with_robot_video=True,
    )
    print(f"[v2r] dataset: {len(dataset)} (demo, t) items across "
          f"{len(dataset.demos)} demos, {len(cfg['tasks'])} tasks", flush=True)
    print(f"[v2r] action codec low={dataset.codec.low.tolist()}", flush=True)
    print(f"[v2r] action codec high={dataset.codec.high.tolist()}", flush=True)

    model = Vid2Robot(model_cfg).to(device)
    n_params = sum(p.numel() for p in model.parameters() if p.requires_grad)
    print(f"[v2r] model params: {n_params/1e6:.1f}M trainable", flush=True)

    # Hyperparameters per Table V.
    base_lr = float(cfg.get("learning_rate", 8e-5))
    final_lr = float(cfg.get("final_learning_rate", 1e-6))
    warmup = int(cfg.get("warmup_steps", 2000))
    num_steps = 1 if args.smoke_test else int(cfg["num_train_steps"])

    siglip_vv = SigLIPLoss(init_temperature=model_cfg.siglip_init_temperature,
                           init_bias=model_cfg.siglip_init_bias).to(device)
    siglip_vt = SigLIPLoss(init_temperature=model_cfg.siglip_init_temperature,
                           init_bias=model_cfg.siglip_init_bias).to(device)

    params = list(model.parameters()) + list(siglip_vv.parameters()) + list(siglip_vt.parameters())
    optimizer = torch.optim.AdamW(params, lr=base_lr,
                                  weight_decay=float(cfg.get("weight_decay", 0.05)))
    schedule = build_cosine_with_warmup(optimizer, num_steps, warmup, base_lr, final_lr)

    output_dir = cfg.get("output_dir")
    if output_dir and not args.smoke_test:
        os.makedirs(output_dir, exist_ok=True)
        with open(os.path.join(output_dir, "train_config.json"), "w") as f:
            json.dump(cfg, f, indent=2)
        with open(os.path.join(output_dir, "action_codec.json"), "w") as f:
            json.dump(dataset.codec.to_dict(), f, indent=2)

    loader = DataLoader(
        dataset,
        batch_size=int(cfg.get("batch_size", 8)),
        shuffle=True,
        num_workers=int(cfg.get("num_workers", 0)),
        collate_fn=collate,
        pin_memory=device.type == "cuda",
        drop_last=True,
    )

    log_interval = int(cfg.get("log_interval", 10))
    save_interval = int(cfg.get("save_interval", 5000))
    grad_accum = max(1, int(cfg.get("grad_accum", 1)))

    # fp16 mixed-precision (use_amp=true is the default). The GradScaler is a
    # no-op on CPU. Disabling falls back to fp32.
    use_amp = bool(cfg.get("use_amp", True)) and device.type == "cuda"
    amp_dtype = torch.float16
    scaler = torch.amp.GradScaler("cuda", enabled=use_amp)
    print(f"[v2r] amp={use_amp} dtype={amp_dtype}", flush=True)

    model.train()
    siglip_vv.train()
    siglip_vt.train()

    print(f"[v2r] starting training: {num_steps} steps, "
          f"batch={cfg.get('batch_size')}, grad_accum={grad_accum}", flush=True)
    step = 0
    t0 = time.time()

    data_iter = iter(loader)
    while step < num_steps:
        try:
            batch = next(data_iter)
        except StopIteration:
            data_iter = iter(loader)
            batch = next(data_iter)
        batch = _to_device(batch, device)

        with torch.autocast(device_type="cuda", dtype=amp_dtype, enabled=use_amp):
            out = model(prompt_video=batch["prompt_video"],
                        state_video=batch["state_video"],
                        robot_full_video=batch.get("robot_full_video"))

            # 1) Action CE.
            logits = out.action_logits  # (B, chunk, action_dim, num_bins)
            targets = batch["action_bins"]  # (B, chunk, action_dim)
            loss_ce = F.cross_entropy(logits.reshape(-1, logits.shape[-1]),
                                      targets.reshape(-1))

            # 2) TCC.
            loss_tcc = tcc_loss(out.prompt_frame_proj, out.robot_frame_proj)

            # 3) Prompt-robot SigLIP (VVCL).
            loss_vvcl = siglip_vv(out.prompt_video_embed, out.robot_video_embed)

            # 4) Video-text SigLIP (VTCL).
            text = batch["text_embedding"]
            loss_vtcl = 0.5 * (siglip_vt(out.prompt_video_embed, text)
                               + siglip_vt(out.robot_video_embed, text))

            loss = (loss_ce + loss_tcc + loss_vvcl + loss_vtcl) / 4.0

        scaler.scale(loss / grad_accum).backward()

        if (step + 1) % grad_accum == 0:
            scaler.unscale_(optimizer)
            torch.nn.utils.clip_grad_norm_(params, max_norm=1.0)
            lr = schedule(step)
            scaler.step(optimizer)
            scaler.update()
            optimizer.zero_grad(set_to_none=True)
        else:
            lr = schedule(step)

        step += 1
        if step % log_interval == 0 or step == 1 or args.smoke_test:
            elapsed = time.time() - t0
            print(f"[v2r] step {step}/{num_steps}  loss={loss.item():.4f}  "
                  f"ce={loss_ce.item():.4f}  tcc={loss_tcc.item():.4f}  "
                  f"vv={loss_vvcl.item():.4f}  vt={loss_vtcl.item():.4f}  "
                  f"lr={lr:.2e}  ({elapsed/step:.2f}s/step)", flush=True)
        if not args.smoke_test and output_dir and step % save_interval == 0:
            ckpt = os.path.join(output_dir, f"vid2robot_step{step}.pt")
            torch.save({
                "model": model.state_dict(),
                "siglip_vv": siglip_vv.state_dict(),
                "siglip_vt": siglip_vt.state_dict(),
                "optimizer": optimizer.state_dict(),
                "scaler": scaler.state_dict(),
                "step": step,
                "codec": dataset.codec.to_dict(),
                "config": cfg,
            }, ckpt)
            print(f"[v2r] checkpoint -> {ckpt}", flush=True)

    if args.smoke_test:
        print(f"[v2r] smoke test ok  total_loss={loss.item():.4f}  "
              f"logits.shape={tuple(out.action_logits.shape)}", flush=True)
        if device.type == "cuda":
            peak_mib = torch.cuda.max_memory_allocated(device) / (1024 ** 2)
            reserved_mib = torch.cuda.max_memory_reserved(device) / (1024 ** 2)
            print(f"[v2r] peak VRAM: allocated={peak_mib:.0f} MiB  "
                  f"reserved={reserved_mib:.0f} MiB  "
                  f"batch_size={int(cfg.get('batch_size', 8))}", flush=True)
    print("[v2r] done.", flush=True)


if __name__ == "__main__":
    main()
