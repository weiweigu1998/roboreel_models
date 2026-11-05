# RoboReel — Policy Learning and Evaluation

Training and evaluation code for **"Monkey See, Can Monkey Do? A Benchmark for
Evaluating Robot Skill Learning by Observation"**.

[Paper](https://arxiv.org/abs/2609.08209) ·
[Project page](https://roboreel.github.io) ·
[Dataset](https://huggingface.co/datasets/LogosRoboticsLab/roboreel) ·
[Checkpoints](https://huggingface.co/LogosRoboticsLab/roboreel_models)

RoboReel asks whether a robot can acquire a manipulation skill by *watching a
human do it*. A policy is conditioned on a human demonstration video — not on
teleoperated robot trajectories of the same scene — and evaluated in
simulation across four test suites that vary visual distraction and object
pose.

## Outline

- [Installation](#installation)
- [Quick start](#quick-start)
- [Repository structure](#repository-structure)
- [Download](#download)
- [Data preparation](#data-preparation)
- [Model training](#model-training)
- [Evaluation](#evaluation)
- [Troubleshooting](#troubleshooting)
- [Acknowledgement](#acknowledgement)
- [Citation](#citation)

## Methods

| Paper label | Internal name | Train entrypoint | Eval runner |
|---|---|---|---|
| VLM + π₀.₅ | `vlm_pi05` | `scripts/train_vlm_pi05.py` | `lerobot_pi05_fft` |
| Vid. Cond. π₀.₅ (Uniform) | `vid_cond_pi05_uniform` | `scripts/train_vid_cond_pi05.py` | `demo_pi05` |
| Vid. Cond. π₀.₅ (Keyframes) | `vid_cond_pi05_keyframes` | `scripts/train_vid_cond_pi05.py` | `demo_pi05` |
| Vid2Robot | `vid2robot` | `scripts/train_vid2robot.py` | `vid2robot` |
| UniSkill | `uniskill_policy` | `scripts/train_uniskill_policy.py` | `uniskill` |

**UniSkill uses the pre-trained ISD.** Its inverse-skill-dynamics model is the
UniSkill authors' released checkpoint, used off the shelf and **not
fine-tuned** on RoboReel — only the diffusion policy over its skill latents is
trained here. That is why there is no ISD training config in this repository,
and why the ISD is not redistributed with our checkpoints: point
`$UNISKILL_ISD` at the authors' release.

**SeeDo** and **PointPolicy**, the paper's other two methods, will be added
later.

**Test suites.** `nd` = no distraction (clean train and eval). `wd` = with
distraction (distractors in both). `ed` = clean training, distracted
evaluation — it reuses the `nd` checkpoints, so there is nothing extra to
train or download. A single evaluation run sweeps the 2×2 of (human demo type
× environment type), which covers all three.

## Installation

Each method has its own dependency set, and the π₀.₅ tracks split across JAX
and PyTorch, so **one conda environment per method** is the supported setup.
See [ENVIRONMENTS.md](ENVIRONMENTS.md) for the recipes.

```bash
git clone --recurse-submodules https://github.com/weiweigu1998/roboreel_models.git
cd roboreel_models

# or, if you already cloned without submodules:
git submodule update --init --recursive
```

Point the code at your data and checkpoints — configs are written against
these two variables rather than absolute paths, and an unset one is reported
immediately instead of failing later as a missing file:

```bash
export ROBOREEL_DATA=/path/to/roboreel          # dataset root
export ROBOREEL_CKPT=/path/to/roboreel_models   # checkpoint root
export OPENPI_DATA_HOME=~/.cache/openpi         # π₀.₅ base weights
```

**ManiSkill assets.** The simulation environments need PartNet-Mobility.
Download it from the [SAPIEN downloads page](https://sapien.ucsd.edu/downloads)
and arrange it under `~/.maniskill/data/`:

```
~/.maniskill/data/
├── configs/                                  → symlink to roboreel_env/assets configs
└── urdfs/partnet_mobility/dataset/<model_id>/...
```

Verify the environments load before anything else:

```bash
python -c "import mani_skill.envs, envs, gymnasium as gym; print(gym.make('PickCube-v2'))"
```

## Quick start

Reproduce a reported number from a released checkpoint, without training:

```bash
export ROBOREEL_DATA=/path/to/roboreel ROBOREEL_CKPT=/path/to/ckpt

# 1. human-demo RGB for the evaluation episodes (~11 GB); this also links the
#    layout the code reads
python scripts/download_dataset.py --subset eval --dest "$ROBOREEL_DATA"

# 2. one policy (~6 GB)
python scripts/download_checkpoints.py \
    --method vid_cond_pi05_keyframes --condition nd --dest "$ROBOREEL_CKPT"

# 3. precompute what this runner conditions on — it reads cached features and
#    keyframe indices, not the mp4s
python scripts/compute_vision_features_for_demo.py --data-root "$ROBOREEL_DATA"
python scripts/extract_keyframes.py --video <demo>/front.mp4 --task pick_cube

# 4. evaluate
python scripts/evaluation.py --config configs/eval/nd/vid_cond_pi05_keyframes.json --gpu 0
```

Step 3 is not optional: see [Data preparation](#data-preparation) for which
artifact each method needs at evaluation time.

Train the same model from scratch:

```bash
python scripts/download_dataset.py --dest "$ROBOREEL_DATA"      # ~104 GB
python scripts/compute_vision_features_for_demo.py --data-root "$ROBOREEL_DATA"
python scripts/process_training_robot_trajectories.py \
    --data-root "$ROBOREEL_DATA" --model vid_cond_pi05
python scripts/train_vid_cond_pi05.py \
    --config configs/train_vid_cond_pi05/nd-perceptual-keyframe-context.json --gpu 0
```

## Repository structure

```
roboreel_models/
├── configs/
│   ├── eval/{nd,wd}/<method>.json          # evaluation, one per method × suite
│   ├── train_vid_cond_pi05/                # Vid. Cond. π₀.₅ (uniform + keyframes)
│   ├── train_vid2robot/                    # Vid2Robot
│   ├── train_uniskill_policy/              # UniSkill diffusion policy
│   └── uniskill.json                       # UniSkill architecture
├── scripts/
│   ├── download_dataset.py                    # dataset from HuggingFace
│   ├── download_checkpoints.py                # checkpoints from HuggingFace
│   ├── process_training_robot_trajectories.py # raw trajectories → per-sample form
│   ├── extract_keyframes.py                   # VLM keyframe annotation
│   ├── annotate_human_videos.py               # VLM instructions for VLM + π₀.₅
│   ├── compute_vision_features_for_demo.py    # cached demo video features
│   ├── extract_uniskill_human_skill.py        # UniSkill latents, human videos
│   ├── extract_uniskill_robot_skill.py        # UniSkill latents, robot trajectories
│   ├── train_vid_cond_pi05.py                 # Vid. Cond. π₀.₅
│   ├── train_vlm_pi05.py                      # VLM + π₀.₅
│   ├── train_vid2robot.py                     # Vid2Robot
│   ├── train_uniskill_policy.py               # UniSkill diffusion policy
│   ├── evaluation.py                          # single evaluation entrypoint
│   └── evaluate.sh                            # 3 seed passes of evaluation.py
├── src/
│   ├── demonstration_vla_suite/            # video-conditioned π₀.₅
│   ├── keyframes/                          # keyframe selection pipeline
│   ├── vid2robot/                          # Vid2Robot model + data
│   ├── wrappers/uniskill.py                # UniSkill inference pipeline
│   ├── evaluation/                         # per-method runners + protocol driver
│   ├── lfo_data.py                         # demo discovery / video loading
│   ├── config_paths.py                     # $VAR expansion for configs
│   └── external_models/
│       ├── openpi/                         # vendored, see its PROVENANCE.md
│       ├── lerobot/                        # submodule
│       ├── UniSkill/                       # submodule
│       └── UniSkill-Policy/                # submodule
└── roboreel_env/                           # RoboReel simulation environments (submodule, v1)
```

## Download

### Dataset

```bash
python scripts/download_dataset.py --dest "$ROBOREEL_DATA"              # default
python scripts/download_dataset.py --subset eval --dest "$ROBOREEL_DATA"
python scripts/download_dataset.py --subset full --dry-run
```

| Subset | Contents | Size |
|---|---|---|
| `exclude-depth` *(default)* | everything but the human-demo `depths.h5` | ~104 GB |
| `eval` | human-demo RGB under `human_demos/eval/` only | ~11 GB |
| `full` | everything, depth included | ~113 GB |

Depth is only needed to work with the raw depth video or camera calibration;
none of the methods here consume it. The `eval` subset is enough to reproduce
the reported numbers from released checkpoints.

### Checkpoints

```bash
python scripts/download_checkpoints.py --dest "$ROBOREEL_CKPT"                     # all, ~50 GB
python scripts/download_checkpoints.py --method vid2robot --condition nd --dry-run
```

Each method directory holds the **evaluated** checkpoint as `ckpt/`, `ckpt.pt`
or `ckpt.pth`, plus the run's own `train_config.json`.

### Base weights

`pi05_base` (for both π₀.₅ tracks) is fetched into `$OPENPI_DATA_HOME` on first
use. The UniSkill inverse-skill-dynamics model is used **off the shelf,
un-finetuned** — take it from the UniSkill authors' release and point
`$UNISKILL_ISD` at it.

## Data preparation

**Evaluation needs this too — it is not training-only.** Four of the five
methods condition on *precomputed* human-demonstration artifacts rather than
on the mp4s, and there is no fallback that computes them on the fly: the
runners raise `FileNotFoundError` if they are missing.

| Method | Reads at eval time | Produced by |
|---|---|---|
| Vid. Cond. π₀.₅ (Uniform) | `video_demonstration_features/<bucket>/<task>/<demo>/` | `compute_vision_features_for_demo.py` |
| Vid. Cond. π₀.₅ (Keyframes) | the above **+** `bpp_keyframe_annotations/<bucket>/<task>/<demo>.json` | **+** `extract_keyframes.py` |
| VLM + π₀.₅ | `human_video_annotation_qwen3-vl-4b.json` | `annotate_human_videos.py` |
| UniSkill | `uniskill_skills/<bucket>/<task>/<demo>/<camera>.npy` | `extract_uniskill_human_skill.py` |
| Vid2Robot | `raw_video_demonstrations/<bucket>/<task>/<demo>/<camera>.mp4` | — (raw video) |

Only the **robot-trajectory** steps (`process_training_robot_trajectories.py`,
`convert_lfo_to_robomimic.py`, `extract_uniskill_robot_skill.py`) are
training-only. Everything derived from the human demos is needed for both.

**On-disk layout.** The scripts and runners read
`raw_video_demonstrations/<bucket>/<task>/<demo_id>/<camera>.mp4` and
`raw_training_trajectories/<bucket>/<task>/<demo_id>/`, with `<bucket>` spelled
`without_distraction` / `with_distraction`. The HuggingFace dataset publishes
`human_demos/{train,eval}/{no_distraction,with_distraction}/...` and
`robot_trajectories/...` instead, so `download_dataset.py` links the two
together with relative symlinks once the download finishes — nothing is
copied. Re-run it with `--layout-only` on an existing tree to rebuild the
links.

### Human demonstrations — needed by all four methods, for training *and* evaluation

```bash
# cached demonstration video features (Vid. Cond. π₀.₅, both variants)
python scripts/compute_vision_features_for_demo.py --data-root "$ROBOREEL_DATA"

# keyframe indices (Vid. Cond. π₀.₅ keyframes variant) — needs OPENAI_API_KEY
python scripts/extract_keyframes.py --video <demo>/front.mp4 --task pick_cube

# natural-language instructions (VLM + π₀.₅)
python scripts/annotate_human_videos.py --data-root "$ROBOREEL_DATA"

# UniSkill skill latents over the human videos (UniSkill)
python scripts/extract_uniskill_human_skill.py \
    --idm-checkpoint "$UNISKILL_ISD" \
    --output-root "$ROBOREEL_DATA/uniskill_skills" --device cuda:0
```

Run the step belonging to each method you intend to use. All four are read at
**evaluation** time as well — they are not a training-only cost.

### Robot trajectories — training only

```bash
# raw trajectories → the shared per-sample pkls + the index each method reads
python scripts/process_training_robot_trajectories.py --data-root "$ROBOREEL_DATA"

# UniSkill skill latents over the robot trajectories
python scripts/extract_uniskill_robot_skill.py \
    --samples-root "$ROBOREEL_DATA/postprocessed_robot_trajectories" \
    --idm-checkpoint "$UNISKILL_ISD" \
    --skill-dir "$ROBOREEL_DATA/UniSkill_skill_representations" --aug-num 5

# UniSkill policy inputs: robomimic HDF5, one per task
python src/external_models/UniSkill-Policy/maniskill/convert_lfo_to_robomimic.py \
    --data-dir "$ROBOREEL_DATA" --output-dir "$ROBOREEL_DATA/robomimic_hdf5"

# optional, Vid2Robot training speed: pre-extract the human prompt videos to
# per-demo npy bundles so the dataset stops decoding mp4s every __getitem__
python scripts/downsample_demo_resolution.py --data-root "$ROBOREEL_DATA"
```

Evaluation never touches these. A released checkpoint is rolled out in the
simulator conditioned on a *human* demonstration, so no robot trajectory is
involved.

`process_training_robot_trajectories.py` writes one shared set of per-sample pkls;
what differs per method is the index file that selects which timesteps it
samples. `--model` picks which to produce (repeatable, default all):

| `--model` | Writes | Read by |
|---|---|---|
| `vid_cond_pi05` | `meta/index.jsonl` | `LfOVLADataset` |
| `vlm_pi05` | `meta/index.jsonl` | `train_vlm_pi05.py` |
| `uniskill` | `meta/index.jsonl` | `extract_uniskill_robot_skill.py` |
| `vid2robot` | the above **+** `meta/{full,vid2robot}_index.jsonl` | `src/vid2robot/data/dataset.py` |

Vid2Robot samples shorter action chunks than π₀.₅, so it can use timesteps the
π₀.₅ action horizon trims off each demo's tail. Selecting it (or `all`) runs
that extra pass automatically, rather than letting training fail later on a
missing `vid2robot_index.jsonl`.

`extract_uniskill_robot_skill.py` must run **after**
`process_training_robot_trajectories.py`: it reads
`postprocessed_robot_trajectories/meta/index.jsonl` and the per-sample pkls
that step writes.

## Model training

Each config fixes one method and one training suite. Replace `nd` with `wd` for
the distracted condition.

```bash
# Vid. Cond. π₀.₅ (Uniform)
python scripts/train_vid_cond_pi05.py \
    --config configs/train_vid_cond_pi05/nd-perceptual-context.json --gpu 0

# Vid. Cond. π₀.₅ (Keyframes)
python scripts/train_vid_cond_pi05.py \
    --config configs/train_vid_cond_pi05/nd-perceptual-keyframe-context.json --gpu 0

# VLM + π₀.₅
python scripts/train_vlm_pi05.py \
    --data-root "$ROBOREEL_DATA" --output-dir "$ROBOREEL_CKPT/nd/vlm_pi05" \
    --include-video-types without_distraction

# Vid2Robot
python scripts/train_vid2robot.py --config configs/train_vid2robot/nd.json --gpu 0
```

```bash
# UniSkill — the diffusion policy only; the ISD is the authors' checkpoint
python scripts/train_uniskill_policy.py \
    --config configs/train_uniskill_policy/nd_pretrained_isd.json --gpu 0
```

Extract the skill latents first (see [Data preparation](#data-preparation)).
Robomimic does not expand environment variables, so this launcher resolves the
config before handing it over.

Hyperparameters match the paper's appendix and are recorded in each config, and
again in the `train_config.json` shipped beside every released checkpoint.

## Evaluation

One entrypoint for every method; the `model` field in the config selects the
runner.

```bash
python scripts/evaluation.py --config configs/eval/nd/vid_cond_pi05_uniform.json --gpu 0
```

A run sweeps the 2×2 of (human demonstration type × environment type) over all
10 tasks and writes a per-cell summary JSON. Read the suites off the cells:

| Suite | Cell |
|---|---|
| ND | `nd/` checkpoint, (without_distraction, without_distraction) |
| WD | `wd/` checkpoint, (with_distraction, with_distraction) |
| ED | `nd/` checkpoint, (with_distraction, with_distraction) |

Each pass writes `<run>/seed_offset_<N>/summary.json`, holding the per-cell
counts in `table` and every rollout in `per_trial` (`{task, demo_id,
success}`) — enough to recompute per-task rates and the mean ± SEM across the
three passes.

`scripts/evaluate.sh` repeats an evaluation over three environment-seed
passes, which is how the reported means were produced.

## Troubleshooting

- **`config references undefined environment variable(s)`** — export
  `ROBOREEL_DATA` / `ROBOREEL_CKPT` (and `UNISKILL_ISD` for UniSkill).
- **Environments fail to load, or Vulkan/EGL errors** — run
  the environment smoke check above first. ManiSkill needs a working Vulkan setup and
  the PartNet-Mobility assets in `~/.maniskill/data/`.
- **CUDA out of memory in training** — lower `batch_size` in the config;
  the π₀.₅ tracks also support gradient checkpointing (on by default).
- **Mixing environments** — the JAX (`vid_cond_pi05_*`) and PyTorch
  (`vlm_pi05`, `vid2robot`, `uniskill_policy`) tracks must not share one conda
  env; they pin conflicting CUDA runtimes.

## Acknowledgement

Built on [openpi](https://github.com/Physical-Intelligence/openpi) (π₀.₅),
[LeRobot](https://github.com/huggingface/lerobot),
[UniSkill](https://github.com/Kicksaw-Consulting/UniSkill),
[Vid2Robot](https://vid2robot.github.io),
[robomimic](https://robomimic.github.io) and
[ManiSkill](https://github.com/haosulab/ManiSkill).

The vendored copy of openpi is Apache-2.0 and its Gemma-derived files are
additionally governed by the [Gemma Terms of Use](LICENSE_GEMMA.txt); see
[`src/external_models/openpi/PROVENANCE.md`](src/external_models/openpi/PROVENANCE.md).

## Citation

```bibtex
@article{roboreel2026,
  title   = {Monkey See, Can Monkey Do? A Benchmark for Evaluating Robot Skill Learning by Observation},
  journal = {arXiv preprint arXiv:2609.08209},
  year    = {2026}
}
```
