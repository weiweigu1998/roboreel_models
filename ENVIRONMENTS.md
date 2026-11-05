# Environments — minimal conda recipes

Four conda environments, one per method. The
first four training envs run their training script(s) **and**
the environment smoke check below; `lerobot_pi05` (§3) is
training-only; `vid2robot` (§4) trains *and* evals (the env is built to host
both). All target a clean
Linux + NVIDIA GPU machine.

The version pins below are taken from the working `benchmark_env_vla`
(Python 3.11) and `benchmark_env_uniskill` (Python 3.12) conda envs on the
dev box, then split per track so each environment carries only what it
actually imports.

## Prerequisites (any track)

- Linux x86_64 (tested on Ubuntu 22.04 / 24.04).
- NVIDIA driver ≥ 525 (CUDA 12.x runtime; check with `nvidia-smi`).
- Miniconda or Mamba in user space — no `sudo` required.
- Git with submodule support; clone the repo with submodules:
  ```bash
  git clone --recurse-submodules <repo-url> action_understanding_bench
  cd action_understanding_bench
  ```
  If you cloned without `--recurse-submodules`:
  ```bash
  git submodule update --init --recursive
  ```

The PyTorch and JAX CUDA wheels bundle their own CUDA runtime libraries, so
no system-level CUDA toolkit install is needed.

## Common verification (after each env install)

Once an env is created and activated, set `PYTHONPATH` so the bench code
imports cleanly, then run the LfO env smoke test:

```bash
export REPO=$(pwd)
export PYTHONPATH="${REPO}:${REPO}/src:${REPO}/src/external_models:${REPO}/src/external_models/lerobot/src:${REPO}/roboreel_env/src"

python -c "import mani_skill.envs, envs, gymnasium as gym; print(gym.make('PickCube-v2'))"
```

Expect 200 random steps to run in ~0.2s with `truncated=True` at step 199
and the final `info` reporting `is_grasped`, `is_lifted`, `is_robot_static`,
`success`, `elapsed_steps`.

The training-smoke commands at the end of each section assume the same
`PYTHONPATH` plus a `data_dir` reachable from the host (the default config
points at `~/Downloads/sample_data`; edit the JSON or symlink as needed).

---

## 1. π₀.₅ — OpenPI (JAX)

```bash
conda create -n pi05 python=3.11 -y
conda activate pi05

# JAX with CUDA 12 wheels.
pip install "jax[cuda12]==0.5.3"

# Flow-matching + diffusion utilities used by openpi.
pip install \
    "flax==0.10.2" \
    "optax==0.2.8" \
    "orbax-checkpoint==0.11.13" \
    "equinox==0.13.6" \
    "jaxtyping==0.2.36" \
    "treescope==0.1.10" \
    "augmax==0.4.1" \
    "chex==0.1.90"

# IMPORTANT: ``chex``'s resolver can transitively bump ``jax`` to 0.10.x,
# which breaks orbax 0.11.13. Re-pin explicitly to be safe.
pip install "jax==0.5.3" "jaxlib==0.5.3"

# Torch (openpi has a small torch-dependent code path; still needed even
# though the flow-matching loop is JAX).
pip install --index-url https://download.pytorch.org/whl/cu126 \
    "torch==2.7.1"

# Tokenizer + transformers utilities.
pip install \
    "transformers==4.57.6" \
    "sentencepiece==0.2.1" \
    "huggingface-hub>=0.25"

# ManiSkill / SAPIEN / LfO env smoke. ``setuptools<81`` is required by
# sapien (it imports ``pkg_resources``, dropped in setuptools 81+).
# Install mani_skill WITHOUT gymnasium first: mani_skill hard-pins
# gymnasium==0.29.1, and pip's resolver refuses to co-resolve that with the
# 1.3.0 the rest of the stack wants in one command (ResolutionImpossible).
# Override gymnasium in a separate step afterwards.
pip install \
    "setuptools<81" \
    "mani_skill==3.0.0b22" \
    "trimesh==4.11.5" \
    "sapien==3.0.2"
pip install "gymnasium==1.3.0"

# Numerical + utilities used by the Pi0.5 scripts.
pip install \
    "einops==0.8.2" \
    "dm-tree==0.1.10" \
    "ml_collections==1.0.0" \
    "numpydantic>=1.6" \
    "fsspec[gcs]>=2024.6" \
    "h5py==3.13.0" \
    "opencv-python-headless>=4.10" \
    "pillow==12.2.0" \
    "imageio==2.37.3" \
    "mediapy==1.2.6" \
    "wandb==0.21.4" \
    "tyro==1.0.12" \
    "tqdm-loggable>=0.2" \
    "loguru==0.7.3" \
    "rich==14.3.3" \
    "hydra-core>=1.3" \
    "beartype==0.19.0" \
    "filelock>=3.16" \
    "typing-extensions>=4.12" \
    "polars>=1.30" \
    "pandas>=2.2" \
    "etils[epath]>=1.0" \
    "imageio-ffmpeg>=0.5"

# openpi-client provides image_tools used by openpi.transforms. Note:
# this pulls numpy<2 transitively, which the rest of the stack accepts.
pip install "openpi-client>=0.1.2"
```

> **Resolver order matters.** Two non-obvious traps:
> 1. ``chex`` doesn't pin ``jax`` strictly, so ``pip install chex`` after
>    JAX can silently upgrade JAX to 0.10.x, breaking orbax 0.11.13. The
>    explicit ``jax==0.5.3 jaxlib==0.5.3`` re-pin above defends against it.
> 2. ``mani_skill 3.0.0b22`` pulls ``gymnasium==0.29.1``; the ``gymnasium==1.3.0``
>    pin in this section overrides that. Runtime works at 1.3.0 (matches
>    benchmark_env_vla).
>
> **Three non-obvious deps that are easy to miss:**
> - ``setuptools<81`` — sapien imports ``pkg_resources``.
> - ``numpydantic`` — used by ``openpi.shared.normalize``.
> - ``openpi-client`` — used by ``openpi.transforms`` for image utilities.

**Verify env smoke:** `python -c "import mani_skill.envs, envs, gymnasium as gym; print(gym.make('PickCube-v2'))"`.

**Verify training smoke:**

```bash
# JAX wants ~all of GPU memory for Pi0.5 unfrozen — set this for single-GPU.
export XLA_PYTHON_CLIENT_MEM_FRACTION=0.95

python scripts/train_vid_cond_pi05.py \
    --config configs/train_vid_cond_pi05/nd-perceptual-context.json
```

Expect 5 training steps to complete (loss ~0.6 → 0.5 range) and an Orbax
checkpoint dir saved at the path in the JSON's `output_dir`. The full
matrix configs live at
`configs/train_vid_cond_pi05/{perceptual,recurrent}-{context,modul,expert}.json`.

The Pi0.5 base checkpoint is downloaded from the `openpi-assets` bucket on
first run; cache location is controlled by `OPENPI_DATA_HOME`
(default `~/.cache/openpi/`).

---

## 2. UniSkill — IDM/FSD pretraining + skill extraction + policy training

The UniSkill pipeline is multi-step. One environment supports all phases.

```bash
conda create -n uniskill python=3.12 -y
conda activate uniskill

# PyTorch (CUDA 12.8 wheels match torch 2.10).
pip install --index-url https://download.pytorch.org/whl/cu128 \
    "torch==2.10.0" "torchvision==0.25.0"

# IDM / FSD diffusion stack (instruct-pix2pix UNet + Depth-Anything-V2).
pip install \
    "diffusers==0.37.0" \
    "transformers==5.3.0" \
    "accelerate==1.13.0" \
    "safetensors==0.7.0" \
    "huggingface-hub>=0.25"

# ManiSkill / SAPIEN / LfO env smoke.
pip install \
    "mani_skill==3.0.0b22" \
    "gymnasium==0.29.1" \
    "trimesh==4.11.1" \
    "sapien==3.0.2"

# ManiSkill / SAPIEN / LfO env smoke. ``setuptools<81`` is required by
# sapien (it imports ``pkg_resources``, dropped in setuptools 81+).
pip install \
    "setuptools<81" \
    "loguru==0.7.3"

# UniSkill IDM dataset + numerical utilities.
#
# This env runs on numpy 2.x. Stage 0a (process_training_robot_trajectories.py)
# pickles the per-sample postprocessed_robot_trajectories pkls under numpy
# 2.x, which renamed its internal `numpy.core` package to `numpy._core`. A
# numpy-1.x interpreter (the pi05 env, locked to numpy<2 by openpi-client)
# can't resolve `numpy._core.*` when unpickling those arrays and raises
# ModuleNotFoundError("numpy._core..."). Rather than pinning every env down
# to numpy 1.x, `src/np2_pickle_compat.py` aliases `numpy._core` onto the
# still-present `numpy.core` in the consuming numpy-1.x envs so `pickle.load`
# resolves — import that shim before unpickling there. So this env stays on
# the current numpy 2.x; opencv-python-headless 4.13+ (which declares
# numpy>=2) is fine here.
pip install \
    "numpy==2.4.2" \
    "scipy==1.17.0" \
    "h5py==3.15.1" \
    "opencv-python-headless==4.13.0.92" \
    "pillow==12.1.0" \
    "einops==0.8.2" \
    "decord==0.6.0" \
    "packaging>=24.0"

# UniSkill-Policy robomimic deps.
pip install \
    "hydra-core>=1.3" \
    "omegaconf==2.3.0" \
    "easydict==1.13" \
    "cloudpickle==3.1.2" \
    "tensorboard==2.20.0" \
    "tensorboardX==2.6" \
    "imageio==2.37.2" \
    "imageio-ffmpeg>=0.5" \
    "wandb==0.25.0" \
    "tqdm==4.67.3" \
    "psutil==7.2.2" \
    "termcolor==3.3.0" \
    "tianshou==0.4.10" \
    "bddl==3.6.0" \
    "gym==0.25.2" \
    "PyOpenGL==3.1.10"

# UniSkill-Policy is a proper python package (provides ``robomimic`` plus
# the ManiSkill-side helper scripts). Editable-install it so the
# ``robomimic`` module resolves and our
# ``maniskill/convert_lfo_to_robomimic.py`` is usable.
pip install --no-deps -e src/external_models/UniSkill-Policy
```

> The UniSkill submodule itself (`src/external_models/UniSkill/`) is
> **not** a pip-installable package — it has no `setup.py` /
> `pyproject.toml`. Add it to `PYTHONPATH` instead so
> `from dynamics.idm import IDM` resolves at runtime:
>
> ```bash
> export PYTHONPATH="$PWD:$PWD/src:$PWD/src/external_models:$PWD/src/external_models/UniSkill:$PWD/src/external_models/UniSkill-Policy:$PWD/roboreel_env/src"
> ```
>
> `LIBERO/`, `robosuite/`, `robocasa/` under `src/external_models/UniSkill-Policy/`
> are upstream submodules that are **not** initialised by default. Our
> `lfo-layout` branch made the LIBERO and robosuite imports lazy in
> `train_utils.py`, so the maniskill path runs without them. If you want
> the LIBERO eval path too, run
> `git -C src/external_models/UniSkill-Policy submodule update --init`
> and pip-install each of those three editable.
>
> **Two non-obvious deps that are easy to miss:**
> - `setuptools<81` — sapien imports `pkg_resources` (dropped in setuptools 81).
> - `loguru` — required by `roboreel_env/src/envs/close_drawer.py` at import time (so it's needed for the env smoke too, not just for training).

**Verify env smoke:** `python -c "import mani_skill.envs, envs, gymnasium as gym; print(gym.make('PickCube-v2'))"`.

**Verify multi-step pipeline:** the phases are launched individually. The
inverse-skill-dynamics model (ISD/IDM) is **not** trained here — the paper uses
the UniSkill authors' released checkpoint unmodified. Point `$UNISKILL_ISD` at
it and only the diffusion policy over its latents is trained.

```bash
# Phase 0 — post-process raw state-mode trajectories into sensor_data-mode h5s.
python scripts/process_training_robot_trajectories.py --data-root "$ROBOREEL_DATA"

# Phase 1 — convert LfO trajectories to robomimic-format HDF5 (one per task).
python src/external_models/UniSkill-Policy/maniskill/convert_lfo_to_robomimic.py \
    --data-dir "$ROBOREEL_DATA" \
    --output-dir "$ROBOREEL_DATA/robomimic_hdf5"

# Phase 2 — extract skill latents with the off-the-shelf ISD.
#   robot trajectories (policy training input)
python scripts/extract_uniskill_robot_skill.py \
    --samples-root "$ROBOREEL_DATA/postprocessed_robot_trajectories" \
    --idm-checkpoint "$UNISKILL_ISD" \
    --skill-dir "$ROBOREEL_DATA/UniSkill_skill_representations" \
    --aug-num 5
#   human demonstration videos (what evaluation conditions on)
python scripts/extract_uniskill_human_skill.py \
    --idm-checkpoint "$UNISKILL_ISD" \
    --output-root "$ROBOREEL_DATA/uniskill_skills" --device cuda:0

# Phase 3 — skill-conditioned policy training.
PYTHONPATH=src/external_models/UniSkill-Policy:$PYTHONPATH python \
    src/external_models/UniSkill-Policy/robomimic/scripts/train.py \
    --config configs/train_uniskill_policy/nd_pretrained_isd.json
```

`configs/train_uniskill_policy/{nd,wd}_pretrained_isd.json` carry the
Robomimic hyperparameters; their `train.data[*].path` and `train.skill_dir`
are written against `$ROBOREEL_DATA`.

---

## 3. lerobot Pi0.5 — full fine-tune (`lerobot/pi05_base`, PyTorch)

The *plain* (language-conditioned, no demo-video) Pi0.5 baseline — **full
fine-tune** (FFT) with lerobot's own Pi0.5 implementation
(`scripts/train_vlm_pi05.py`, eval wrapper
`src/evaluation/lerobot_pi05_fft_wrapper.py`). Used to train the two FFT
Used to train the paper's **VLM + π₀.₅** baseline, whose instructions come
from Qwen3-VL-4B. This is a **separate track from §1**: §1 is the OpenPI/JAX
*demonstration* Pi0.5; this is lerobot/PyTorch, a different model identity and
framework.

Training needs a GPU host; evaluation additionally needs Vulkan/SAPIEN for the
simulator. The env recipe is the same for both.

This env installs lerobot from a **fresh clone kept outside the repo**, so
the `src/external_models/lerobot` submodule (used by the Pi0.5 tracks)
stays untouched.

```bash
conda create -n lerobot_pi05 python=3.12 -y      # lerobot 0.5.x requires >=3.12
conda activate lerobot_pi05

# Clone current lerobot OUTSIDE the repo, editable-install with the
# pi (Pi0.5) extras. FFT doesn't need peft — only `[pi]` is required.
git clone https://github.com/huggingface/lerobot "$LEROBOT_ROOT"
cd "$LEROBOT_ROOT"
pip install -e ".[pi]"
```

> **Gated tokenizer.** lerobot's Pi0.5 processor loads the PaliGemma
> tokenizer `google/paligemma-3b-pt-224` — a gated HF repo. Set `HF_TOKEN`
> (env var or `~/.bashrc`) to a token whose HuggingFace account has accepted
> the gate at <https://huggingface.co/google/paligemma-3b-pt-224>. A
> read-scope token is sufficient.

**Verify env:**

```bash
python -c "
from lerobot.policies.pi05 import PI05Policy, make_pi05_pre_post_processors
print('lerobot pi05 import OK')
"
```

**Verify training smoke** (run from the repo, with `HF_TOKEN` set):

```bash
python scripts/train_vlm_pi05.py --num-train-steps 2 --batch-size 2 \
    --num-stats-samples 300 --device cuda:0 --output-dir /tmp/pi05_fft_smoke
```

Expect `pi05_base` to load ("All keys loaded successfully"), FFT trainable
params ≈ total params (every parameter trains; ~4.1 B), 2 training steps,
and a full-weight checkpoint saved under `/tmp/pi05_fft_smoke/<step>/`.

---

## 4. Vid2Robot — video-conditioned policy (PyTorch, train + eval)

Vid2Robot (Jain et al. 2024) is implemented end-to-end in this repo at
`src/vid2robot/` and trained via `scripts/train_vid2robot.py`. The env hosts
**both** training and ManiSkill rollout eval (the `vid2robot_runner` invoked by
`scripts/evaluation.py`) so there's no separate eval env.

`numpy<2` matches the pi05 env (§1) so the postprocessed-trajectory pkls (which
were written under numpy 2.x in the `uniskill` env at Stage 0a) unpickle via
`src/np2_pickle_compat.py` the same way the pi05 track does.

```bash
conda create -n vid2robot python=3.11 -y
conda activate vid2robot

# PyTorch (CUDA 12.6 wheels). Same versions as the lerobot track (§3) so the wheel
# index and ABI agree if you co-host this env on the same box.
pip install --index-url https://download.pytorch.org/whl/cu126 \
    "torch==2.7.1" "torchvision==0.22.1"

# Vid2Robot model stack — ViT-B/16 backbone via timm; SigLIP text tower via
# transformers (used by the VTCL contrastive loss; ``--fake-text`` skips it
# for offline smoke tests).
pip install \
    "timm==1.0.11" \
    "transformers==4.57.6" \
    "huggingface-hub>=0.25" \
    "sentencepiece==0.2.1" \
    "protobuf>=4.21" \
    "einops==0.8.2"

# ManiSkill / SAPIEN / LfO env smoke + the eval runner deps. h5py is pulled
# by demonstration_vla_suite.shared.maniskill_io which the eval runner imports.
# Install mani_skill WITHOUT gymnasium first (mani_skill pins 0.29.1, the rest
# wants 1.3.0 — same trick as §1).
pip install \
    "setuptools<81" \
    "mani_skill==3.0.0b22" \
    "trimesh==4.11.5" \
    "sapien==3.0.2" \
    "h5py==3.13.0"
pip install "gymnasium==1.3.0"

# Numerical + image / mp4 utilities. numpy<2 to match the pi05 env (§1) so the
# Stage-0a pkls unpickle without any version-specific drama. opencv-python-headless
# (no GUI libs) + the numpy-1-compatible 4.11.x build. ``np2_pickle_compat``
# (auto-installed by the train / eval scripts) handles the numpy-2 pkls.
pip install "numpy<2"
pip uninstall -y opencv-python
pip install "opencv-python-headless==4.11.0.86"
pip install \
    "pillow==12.2.0" \
    "decord==0.6.0" \
    "tqdm==4.67.3" \
    "imageio==2.37.3" \
    "imageio-ffmpeg>=0.5"
```

**Verify env smoke:** `python -c "import mani_skill.envs, envs, gymnasium as gym; print(gym.make('PickCube-v2'))"`.

**Verify training smoke** (downloads the SigLIP text tower on first run, ~600 MB):

```bash
python scripts/train_vid2robot.py \
    --config configs/train_vid2robot/nd.json \
    --smoke-test
```

Expect one step to complete with `logits.shape=(2, 4, 8, 256)` and the four
losses (CE / TCC / VVCL / VTCL) all populated.

**Optional pre-extraction step** (one-time; speeds up training by mmap-loading
the human prompt videos instead of decoding mp4s each `__getitem__`):

```bash
python scripts/downsample_demo_resolution.py --data-root "$ROBOREEL_DATA"
```

Writes `downsampled_demonstration_videos/<video_type>/<task>/<demo>/front.npy`,
shape `(T, 224, 224, 3) uint8`. Temporal sampling stays at the dataset layer so
the paper's 16-frame "first + last + sorted-middle" sampler re-randomises each
training step.

**Eval config shape** (place under `configs/eval_vid2robot/<name>.json`):

```json
{ "model": "vid2robot",
  "checkpoint_path": "/path/to/vid2robot_step100000.pt",
  "scope": "full",
  "save_video": false,
  "chunk_size": 1 }
```

Then `python scripts/evaluation.py --config configs/eval_vid2robot/<name>.json`.

---

## Choosing CUDA wheels for your driver

| Driver `nvidia-smi` reports | PyTorch wheel index URL |
|---|---|
| CUDA 12.1 / 12.2 / 12.3 / 12.4 | `https://download.pytorch.org/whl/cu124` |
| CUDA 12.5 / 12.6 / 12.7 (recommended for the Pi0.5 tracks here) | `https://download.pytorch.org/whl/cu126` |
| CUDA 12.8 / 12.9 (recommended for UniSkill here) | `https://download.pytorch.org/whl/cu128` |
| CUDA 13.0+ | use latest from <https://pytorch.org/get-started/locally/> |

The driver in `nvidia-smi` is forward-compatible with older CUDA wheels —
the table above shows the **highest** CUDA wheel that works. If
`torch.cuda.is_available()` returns False after install, the wheel CUDA
version is too high for your driver; try the next index URL down.

JAX's `jax[cuda12]` extras handle CUDA discovery automatically — no
matching index URL needed.

---

## Sanity-check script (optional)

After creating an env, this one-liner verifies everything that the
training scripts depend on:

```bash
python -c "
import torch, mani_skill, gymnasium as gym
print('torch:', torch.__version__, '| cuda:', torch.cuda.is_available())
print('mani_skill:', mani_skill.__version__)
import sapien; print('sapien:', sapien.__version__)
"
```

For Pi0.5 add `import jax; print('jax:', jax.__version__, '|',
jax.devices())`.



For UniSkill add `from dynamics.idm import IDM; print('uniskill IDM: OK')`
(this requires the `pip install -e src/external_models/UniSkill` step).


