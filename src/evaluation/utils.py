"""Shared evaluation utilities — demo discovery, env construction, the 2×2 protocol driver,
plus a few legacy bookkeeping helpers (config loading / summary writing / StopWatch).

Consolidates what used to live in ``demonstration_vla_suite.shared.{eval_demo_discovery,
eval_loop,eval_protocol}``.

Data layout (under ``data_dir``)::

    video_demonstration_features/<with_distraction|without_distraction>/<task>/<0000-0099>/token_emb_<t>_<camera>.npy
    raw_video_demonstrations/<with_distraction|without_distraction>/<task>/<0000-0099>/<camera>.mp4

A *complete* evaluation iterates the 2×2 grid {human-demo distractor variant} ×
{task-env distractor} over a demo scope (``full`` = demos 0000-0099, ``last20`` =
0080-0099) and over every task the model was trained on; the task env is reset with
``seed = demo_id`` for each demo video. ``run_eval_protocol`` returns/prints a 2×2
success-rate table.

A *runner* (one per model — see ``evaluation.{demo_pi05,uniskill,vid2robot}_runner``) supplies the
model-specific bits:

    class Runner:
        name: str
        data_dir: str
        tasks: list[str]
        action_horizon: int                       # default chunk size if the config doesn't override
        robot_cameras: list[str]                   # ManiSkill camera keys fed to the policy (one image slot each)
        robot_camera: str                          # = robot_cameras[0]; the camera used for the rendered video
        def load_demo_static(self, demo_distract, task, demo_id) -> Any: ...
        def rollout(self, env, obs, demo_static, max_rollout_steps, chunk_size, video_frames) -> bool: ...
"""

from __future__ import annotations

import argparse
import json
import os
import logging
import pickle
import time
from dataclasses import dataclass, field
from pathlib import Path

from config_paths import expand_paths

logger = logging.getLogger("evaluation")


# ===========================================================================
# Demo discovery / task → env mapping
# ===========================================================================

#: Demo IDs per eval scope for the demo_distract=False (no-distraction) cells.
#: ``full`` runs all 100; ``last20`` the last 20 (0080-0099); ``one`` just demo
#: 0000 (smoke test).
EVAL_DEMO_SCOPES: dict[str, range] = {
    "full":   range(0, 100),
    "last20": range(80, 100),
    "one":    range(0, 1),
}

#: Demo IDs for the demo_distract=True (with-distraction) cells. The protocol's
#: ``last20`` with-distraction slice is a fresh held-out set, 0100-0119 — not
#: the 0080-0099 used for the no-distraction cells. Other scopes are unchanged.
EVAL_DEMO_SCOPES_DISTRACT: dict[str, range] = {
    "full":   range(0, 100),
    "last20": range(100, 120),
    "one":    range(0, 1),
}

#: Sub-directory names for the two human-demo distractor variants.
DEMO_DISTRACTION_DIRS = {True: "with_distraction", False: "without_distraction"}

#: Reverse mapping: type-string -> distract boolean. Used to parse
#: ``human_demo_and_env_types`` from eval configs (which write the type strings
#: ``"without_distraction"`` / ``"with_distraction"`` directly rather than booleans
#: so the JSON reads as the actual data-dir suffixes).
DISTRACT_TYPE_TO_BOOL: dict[str, bool] = {
    "without_distraction": False,
    "with_distraction": True,
}

#: Default list of (human_demo_type, env_type) tuples — the canonical 2×2 grid
#: used by every protocol-conformant eval. Override via the eval config's
#: ``human_demo_and_env_types`` field to evaluate a subset.
DEFAULT_HUMAN_DEMO_AND_ENV_TYPES: list[tuple[str, str]] = [
    ("without_distraction", "without_distraction"),
    ("without_distraction", "with_distraction"),
    ("with_distraction",    "without_distraction"),
    ("with_distraction",    "with_distraction"),
]

#: Per-task ManiSkill env id. Single source of truth for ``gym.make``.
TASK_TO_ENV_ID: dict[str, str] = {
    "bowl_in_plate":  "BowlOnPlate-v1",
    "close_drawer":   "CloseDrawer-v1",
    "empty_basket":   "EmptyBasket-v1",
    "open_lid":       "OpenLid-v1",
    "pick_cube":      "PickCube-v2",
    "press_toaster":  "Toaster-v1",
    "push_cube":      "PushCube-v2",
    "stack_cups":     "StackCups-v1",
    "toys_in_basket": "PutBasket-v1",
    "toys_in_drawer": "ObjectInDrawer-v1",
}

#: Per-task control mode. The dataset produces 8-dim actions (7 joint targets + 1
#: gripper) under ``pd_joint_pos``; tasks collected under a different controller override here.
TASK_TO_CONTROL_MODE: dict[str, str] = {}

DEFAULT_CONTROL_MODE = "pd_joint_pos"


def demo_ids_for_scope(scope: str, demo_distract: bool = False) -> list[int]:
    """Return the integer demo IDs for an eval scope.

    ``demo_distract`` picks the demo set: the with-distraction cells use a
    separate held-out slice (``EVAL_DEMO_SCOPES_DISTRACT``).
    """
    table = EVAL_DEMO_SCOPES_DISTRACT if demo_distract else EVAL_DEMO_SCOPES
    if scope not in table:
        raise ValueError(f"Unknown eval scope {scope!r}. Expected one of {sorted(table)}.")
    return list(table[scope])


def demo_feature_dir(data_dir: str | Path, demo_distract: bool, task: str, demo_id: int) -> Path:
    """Directory holding ``token_emb_<t>_<camera>.npy`` for one human demo."""
    return (
        Path(data_dir)
        / "video_demonstration_features"
        / DEMO_DISTRACTION_DIRS[bool(demo_distract)]
        / task
        / f"{int(demo_id):04d}"
    )


def demo_video_dir(data_dir: str | Path, demo_distract: bool, task: str, demo_id: int) -> Path:
    """Directory holding the raw ``<camera>.mp4`` files for one human demo."""
    return (
        Path(data_dir)
        / "raw_video_demonstrations"
        / DEMO_DISTRACTION_DIRS[bool(demo_distract)]
        / task
        / f"{int(demo_id):04d}"
    )


def env_id_for_task(task: str) -> str:
    try:
        return TASK_TO_ENV_ID[task]
    except KeyError as exc:
        raise KeyError(f"No env_id registered for task {task!r}. Add it to TASK_TO_ENV_ID.") from exc


def control_mode_for_task(task: str) -> str:
    return TASK_TO_CONTROL_MODE.get(task, DEFAULT_CONTROL_MODE)


def make_eval_env(task: str, distract: bool, max_episode_steps: int | None = None):
    """Build the ManiSkill env for ``task`` with the distractor flag set. Returns ``(env, env_id)``.

    Uses the ``panda_wristcam_hd`` robot (the same one the training trajectories were generated
    with — note the ``panda_wristcam_hd`` articulation in the h5s) so the obs exposes the wrist
    ``hand_camera``; the LfO envs' ``_default_sensor_configs`` add ``base_camera`` / ``left_camera``
    / ``right_camera`` / ``overhead_camera``. That matches the camera set ``LfOVLADataset`` reads
    from the h5s, so a multi-``robot_cameras`` checkpoint can be evaluated.

    ``max_episode_steps`` overrides the per-env registration default (200 for most LfO tasks,
    400 for ObjectInDrawer-v1, 600 for Play-v*). Without this, gymnasium's TimeLimit wrapper
    returns ``truncated=True`` at the registered cap regardless of the runner's
    ``max_rollout_steps`` — i.e. the runner's outer loop never reaches its declared horizon for
    long-trajectory tasks (empty_basket / stack_cups / toys_* mean trajectory length 400-510).
    """
    import gymnasium as gym
    env_id = env_id_for_task(task)
    kwargs: dict = dict(
        obs_mode="sensor_data",
        num_envs=1,
        render_mode="sensors",
        robot_uids="panda_wristcam_hd",
        enable_shadow=False,
        control_mode=control_mode_for_task(task),
        distract=bool(distract),
    )
    if max_episode_steps is not None:
        kwargs["max_episode_steps"] = int(max_episode_steps)
    env = gym.make(env_id, **kwargs)
    return env, env_id


#: Repo-root path where per-(task, distract bucket) eval env config banks live.
#: Each ``<bucket>/<task>.pkl`` is a ``list[dict]`` of ``{"seed": int, "state": dict}``
#: snapshots in the standard ManiSkill 3 ``BaseEnv.get_state_dict()`` format.
#: Frozen per-task evaluation scene configs. Shipped by the environment
#: submodule (``roboreel_env/init_states/<bucket>/<task>.pkl``) so the envs and
#: the states they are evaluated from stay versioned together. The legacy
#: in-repo ``eval_env_configs/`` is still honoured if present.
_REPO_ROOT = Path(__file__).resolve().parents[2]
EVAL_ENV_CONFIG_ROOT = _REPO_ROOT / "roboreel_env" / "init_states"
if not EVAL_ENV_CONFIG_ROOT.is_dir() and (_REPO_ROOT / "eval_env_configs").is_dir():
    EVAL_ENV_CONFIG_ROOT = _REPO_ROOT / "eval_env_configs"


def load_eval_env_bank(task: str, env_distract: bool) -> list[dict]:
    """Load the pre-recorded env-config bank for ``(task, env_distract)``.

    Each entry is ``{"seed": int, "state": dict}``. The protocol driver replays
    one entry per rollout instead of sampling fresh env state from a seed, so the
    three protocol passes (``seed_offset_base`` ∈ {0, 100000, 200000}) can be made
    deterministic *and* disjoint by indexing into different slices of the bank.
    """
    bucket = DEMO_DISTRACTION_DIRS[bool(env_distract)]
    p = EVAL_ENV_CONFIG_ROOT / bucket / f"{task}.pkl"
    if not p.is_file():
        raise FileNotFoundError(
            f"No eval env config bank for task={task!r} env_distract={env_distract}: "
            f"expected {p}. Re-extract eval_pkl.zip into the repo root."
        )
    with p.open("rb") as f:
        bank = pickle.load(f)
    if not isinstance(bank, list) or not all(
        isinstance(e, dict) and "seed" in e and "state" in e for e in bank
    ):
        raise TypeError(
            f"{p}: expected list[dict] with 'seed' + 'state' keys per entry; "
            f"got {type(bank).__name__} of {type(bank[0]).__name__ if bank else 'empty'}"
        )
    return bank


# ===========================================================================
# Eval config + train-config lookup
# ===========================================================================

@dataclass
class EvalConfig:
    """The per-run eval config (``configs/eval_maniskill_*/*.json``).

    ``model`` selects the runner: ``"demo_pi05"`` (checkpoint-path driven) or
    ``"uniskill"`` (idm + policy checkpoints + an explicit ``data_dir``/``tasks``).
    """

    model: str                              # "demo_pi05" | "uniskill" | "lerobot_pi05_fft" | "vid2robot"
    scope: str = "full"                     # "full" (0000-0099) | "last20" (off=0080-0099 / on=0100-0119)
    seed_offset_base: int = 0               # added to the env reset seed; run a 3-pass eval as 3 invocations with offsets 0/100000/200000 (evaluate.sh)
    save_video: bool = False
    max_rollout_steps: int | None = None    # optional override; default = env's max_episode_steps
    chunk_size: int | None = None            # optional override; default = runner.action_horizon
    output_dir: str | None = None            # optional; default depends on the runner
    # Which (human-demo type, env type) combinations to evaluate. Each tuple is
    # two strings from {"without_distraction", "with_distraction"}. Default None
    # → the canonical 4-cell 2×2 sweep (DEFAULT_HUMAN_DEMO_AND_ENV_TYPES). Set to
    # a subset to skip cells; the per-cell summary JSONs are dumped only for the
    # cells listed here.
    human_demo_and_env_types: list[tuple[str, str]] | None = None
    # Optional override of the runner's task list. Default None → use the
    # runner's declared self.tasks (which is normally all 10 LfO tasks, or
    # whatever the train_config recorded). Set this to a subset of LfO task
    # names to restrict the eval to a subset. Validated against
    # TASK_TO_ENV_ID's keys at load time.
    tasks: list[str] | None = None
    raw: dict = field(default_factory=dict)  # the full parsed config (model-specific fields live here)


def load_eval_config(path: str | Path) -> EvalConfig:
    d = expand_paths(json.loads(Path(path).read_text()))
    if "model" not in d:
        raise ValueError(f"eval config {path} missing required field 'model' (demo_pi05|uniskill|lerobot_pi05_fft|vid2robot)")
    # human_demo_and_env_types comes from JSON as a list of 2-element lists
    # (JSON has no tuple type); normalize to list[tuple[str, str]] and validate
    # each element is a known distract-type string.
    raw_types = d.get("human_demo_and_env_types")
    if raw_types is None:
        hd_env_types = None
    else:
        hd_env_types = []
        for i, pair in enumerate(raw_types):
            if not (isinstance(pair, (list, tuple)) and len(pair) == 2):
                raise ValueError(
                    f"eval config {path}: human_demo_and_env_types[{i}] must be a 2-element "
                    f"[human_demo_type, env_type] list; got {pair!r}"
                )
            hd, ev = str(pair[0]), str(pair[1])
            if hd not in DISTRACT_TYPE_TO_BOOL or ev not in DISTRACT_TYPE_TO_BOOL:
                raise ValueError(
                    f"eval config {path}: human_demo_and_env_types[{i}] strings must be in "
                    f"{sorted(DISTRACT_TYPE_TO_BOOL)} (got ({hd!r}, {ev!r}))"
                )
            hd_env_types.append((hd, ev))
    # Optional task-subset override — validate against TASK_TO_ENV_ID's keys.
    raw_tasks = d.get("tasks")
    if raw_tasks is None:
        tasks_override = None
    else:
        tasks_override = [str(t) for t in raw_tasks]
        unknown = [t for t in tasks_override if t not in TASK_TO_ENV_ID]
        if unknown:
            raise ValueError(
                f"eval config {path}: tasks override contains unknown task(s) {unknown}; "
                f"known tasks: {sorted(TASK_TO_ENV_ID)}"
            )
    return EvalConfig(
        model=str(d["model"]).lower(),
        scope=d.get("scope", "full"),
        seed_offset_base=int(d.get("seed_offset_base", 0)),
        save_video=bool(d.get("save_video", False)),
        max_rollout_steps=d.get("max_rollout_steps"),
        chunk_size=d.get("chunk_size"),
        output_dir=d.get("output_dir"),
        human_demo_and_env_types=hd_env_types,
        tasks=tasks_override,
        raw=d,
    )


def load_train_config(checkpoint_dir: str | Path) -> dict:
    """The training-run config persisted beside the checkpoint (``<output_dir>/train_config.json``).

    ``checkpoint_dir`` is normally ``<output_dir>/<step>``, so the config is one level up;
    we also accept it directly inside ``checkpoint_dir``.
    """
    cd = Path(checkpoint_dir)
    for cand in (cd / "train_config.json", cd.parent / "train_config.json"):
        if cand.is_file():
            # Released checkpoints record data paths as ${ROBOREEL_DATA}; the
            # runners read data_root/data_dir straight out of this dict.
            return expand_paths(json.loads(cand.read_text()))
    raise FileNotFoundError(
        f"No train_config.json next to {checkpoint_dir} (checked it and its parent). "
        "Re-run training with the updated train scripts so the run config is persisted."
    )


# ===========================================================================
# The protocol driver
# ===========================================================================

def _success_from_info(info: dict) -> bool:
    """Extract the env's per-step success flag (handles batched torch tensors)."""
    import torch
    v = info.get("success", False)
    if isinstance(v, torch.Tensor):
        return bool(v.reshape(-1)[0].item())
    return bool(v)


def _video_subdir(demo_distract: bool, env_distract: bool, task: str) -> str:
    dd = "demoDistract" if demo_distract else "demoClean"
    ed = "envDistract" if env_distract else "envClean"
    return f"{dd}__{ed}/{task}"


def run_eval_protocol(eval_cfg: EvalConfig, runner) -> dict:
    """Run the protocol with ``runner``; print + write the summary table(s).

    The set of (human-demo type, env type) cells to evaluate is taken from
    ``eval_cfg.human_demo_and_env_types`` — defaults to the canonical 2×2 grid
    (``DEFAULT_HUMAN_DEMO_AND_ENV_TYPES``). For each requested cell, a per-cell
    JSON is written to ``<output_dir>/summary_human_<demo>_env_<env>.json``.
    The combined ``summary.json`` (all evaluated cells in one file, same schema
    as before) is also written for downstream tooling.

    Returns the combined summary dict.
    """
    # eval_cfg.tasks (if set) overrides runner.tasks — lets a per-experiment
    # eval config restrict the task subset without retraining the model. We
    # intersect rather than replace blindly to catch typos: any requested
    # task that the runner can't actually serve raises.
    if eval_cfg.tasks:
        runner_set = set(runner.tasks)
        missing = [t for t in eval_cfg.tasks if t not in runner_set]
        if missing:
            raise ValueError(
                f"eval_cfg.tasks requested {missing} but runner.tasks only has {sorted(runner_set)}"
            )
        tasks = list(eval_cfg.tasks)
        logger.info("eval_cfg.tasks override: restricting to %d task(s) %s "
                    "(runner.tasks had %d)", len(tasks), tasks, len(runner.tasks))
    else:
        tasks = list(runner.tasks)
    if not tasks:
        raise ValueError("tasks list is empty — nothing to evaluate.")
    # Demo slice differs by the demo_distract axis: no-distraction cells use the
    # scope's range (last20 -> 0080-0099); with-distraction cells use a separate
    # held-out slice (last20 -> 0100-0119). Env reset seed follows demo_id.
    demo_ids_clean = demo_ids_for_scope(eval_cfg.scope, demo_distract=False)
    demo_ids_distract = demo_ids_for_scope(eval_cfg.scope, demo_distract=True)
    chunk_size = int(eval_cfg.chunk_size or runner.action_horizon)

    # Resolve which cells to run. Each element is (demo_type_str, env_type_str)
    # taken verbatim from the config (or the default list); we also derive the
    # boolean form (demo_distract, env_distract) used everywhere downstream.
    requested_types: list[tuple[str, str]] = (
        eval_cfg.human_demo_and_env_types
        if eval_cfg.human_demo_and_env_types is not None
        else list(DEFAULT_HUMAN_DEMO_AND_ENV_TYPES)
    )
    requested_cells: list[tuple[bool, bool, str, str]] = [
        (DISTRACT_TYPE_TO_BOOL[hd], DISTRACT_TYPE_TO_BOOL[ev], hd, ev)
        for (hd, ev) in requested_types
    ]

    if eval_cfg.output_dir:
        output_dir = Path(eval_cfg.output_dir)
    else:
        output_dir = Path(getattr(runner, "default_output_dir", Path.cwd() / "eval_results")) / eval_cfg.scope
    output_dir.mkdir(parents=True, exist_ok=True)
    video_root = (output_dir / "videos") if eval_cfg.save_video else None
    if video_root is not None:
        video_root.mkdir(parents=True, exist_ok=True)

    logger.info(
        "Eval protocol: runner=%s, scope=%s, tasks=%s, demos/combo=%d clean / %d distract, "
        "chunk_size=%d, save_video=%s, cells=%s, out=%s",
        runner.name, eval_cfg.scope, tasks, len(demo_ids_clean), len(demo_ids_distract),
        chunk_size, eval_cfg.save_video, requested_types, output_dir,
    )

    # Group requested cells by env_distract so we only build each ManiSkill env
    # once. ``results[(demo_distract, env_distract)]`` -> list[(task, demo_id, success)].
    cells_by_env: dict[bool, list[tuple[bool, str, str]]] = {}
    for dd_bool, ed_bool, hd_str, ev_str in requested_cells:
        cells_by_env.setdefault(ed_bool, []).append((dd_bool, hd_str, ev_str))

    results: dict[tuple[bool, bool], list[tuple[str, int, bool]]] = {}

    # ``seed_offset_base`` is repurposed as the bank-slice selector across the
    # three protocol passes: pass 0 → bank[0:N], pass 1 → bank[N:2N], pass 2 →
    # bank[2N:3N], where N = len(demo_ids_<bucket>). The canonical values are
    # {0, 100000, 200000} → pass_idx ∈ {0, 1, 2}.
    pass_idx = int(eval_cfg.seed_offset_base) // 100000

    for env_distract, dd_list in cells_by_env.items():
        for task in tasks:
            # Pass max_rollout_steps to make_eval_env so it overrides the env's
            # registered max_episode_steps (200 for most LfO tasks). Without
            # this, the TimeLimit wrapper truncates rollouts at 200 frames even
            # if the runner's outer loop cap is higher.
            env, _env_id = make_eval_env(
                task, env_distract,
                max_episode_steps=eval_cfg.max_rollout_steps,
            )
            max_steps = int(
                eval_cfg.max_rollout_steps
                or getattr(getattr(env, "spec", None), "max_episode_steps", None)
                or 200
            )
            # Pre-recorded env-config bank for this (task, env_distract). The
            # bank is shared across demo_distract variants — env init depends
            # only on env_distract; demo_distract changes only which human
            # video conditions the policy, not the env state.
            bank = load_eval_env_bank(task, env_distract)
            for demo_distract, _hd_str, _ev_str in dd_list:
                ids = list(demo_ids_distract if demo_distract else demo_ids_clean)
                slice_lo = pass_idx * len(ids)
                slice_hi = slice_lo + len(ids)
                if slice_hi > len(bank):
                    raise ValueError(
                        f"Bank for task={task} env_distract={env_distract} has only "
                        f"{len(bank)} entries; need at least {slice_hi} for "
                        f"pass_idx={pass_idx} (slice_size={len(ids)}). "
                        f"Re-generate the .pkl with more configs or use a smaller scope."
                    )
                slice_seeds = [bank[slice_lo + i]["seed"] for i in range(len(ids))]
                logger.info(
                    "  bank task=%s env_distract=%s pass_idx=%d → bank[%d:%d], seeds=%s",
                    task, env_distract, pass_idx, slice_lo, slice_hi, slice_seeds,
                )
                for i, demo_id in enumerate(ids):
                    cfg = bank[slice_lo + i]
                    demo_static = runner.load_demo_static(demo_distract, task, int(demo_id))
                    # When env_distract=True, force scene reconfigure with the
                    # bank's recorded distractor identities + count so the env's
                    # actor registry exposes the same distractor_<i> slots that
                    # cfg["state"]["actors"] keys reference. The bank's
                    # distractor_names list is the source of asset identity;
                    # if its length disagrees with the number of distractor_
                    # slots actually recorded in state, the state slot count
                    # wins (bank-gen sometimes records more names than slots).
                    # See the roboreel_env submodule for the env-side override hook.
                    reset_opts: dict = {}
                    if env_distract:
                        n_slots = sum(
                            1 for k in cfg["state"]["actors"] if str(k).startswith("distractor")
                        )
                        names = list(cfg.get("distractor_names") or [])[:n_slots]
                        if len(names) < n_slots and names:
                            # defensive pad — repeat last asset if names < slots
                            names = names + [names[-1]] * (n_slots - len(names))
                        if names:
                            reset_opts = {"reconfigure": True, "distractor_names": names}
                    obs, _ = env.reset(seed=int(cfg["seed"]), options=reset_opts or None)
                    env.unwrapped.set_state_dict(cfg["state"])
                    obs = env.unwrapped.get_obs()
                    video_frames: list | None = [] if video_root is not None else None
                    t0 = time.time()
                    success = bool(runner.rollout(env, obs, demo_static, max_steps, chunk_size, video_frames))
                    results.setdefault((demo_distract, env_distract), []).append((task, int(demo_id), success))
                    if video_root is not None and video_frames:
                        import imageio
                        vdir = video_root / _video_subdir(demo_distract, env_distract, task)
                        vdir.mkdir(parents=True, exist_ok=True)
                        imageio.mimwrite(
                            str(vdir / f"{int(demo_id):04d}_{'ok' if success else 'fail'}.mp4"),
                            video_frames, fps=20, quality=8,
                        )
                    logger.info(
                        "  envDistract=%s demoDistract=%s task=%s demo=%04d → %s (%.1fs)",
                        env_distract, demo_distract, task, int(demo_id),
                        "OK" if success else "fail", time.time() - t0,
                    )
            env.close()

    table: dict[tuple[bool, bool], dict] = {}
    for (dd, ed), rows in results.items():
        n = len(rows)
        s = sum(1 for _, _, ok in rows if ok)
        table[(dd, ed)] = {"success": s, "total": n, "rate": (s / n if n else 0.0)}

    _log_table(table, runner.name, eval_cfg.scope, tasks)

    # Per-cell JSON dumps — one file per (human_demo_type, env_type) tuple that
    # was actually evaluated. Self-contained so each file can be consumed alone.
    common_header = {
        "model": eval_cfg.model,
        "runner": runner.name,
        "scope": eval_cfg.scope,
        "seed_offset_base": eval_cfg.seed_offset_base,
        "tasks": tasks,
        "chunk_size": chunk_size,
        "max_rollout_steps": eval_cfg.max_rollout_steps,
    }
    for dd_bool, ed_bool, hd_str, ev_str in requested_cells:
        rows = results.get((dd_bool, ed_bool), [])
        cell_summary = {
            **common_header,
            "human_demo_type": hd_str,
            "env_type": ev_str,
            "demo_ids": demo_ids_distract if dd_bool else demo_ids_clean,
            "table": table.get((dd_bool, ed_bool), {"success": 0, "total": 0, "rate": 0.0}),
            "per_trial": [
                {"task": t, "demo_id": di, "success": ok} for (t, di, ok) in rows
            ],
        }
        cell_path = output_dir / f"summary_human_{hd_str}_env_{ev_str}.json"
        cell_path.write_text(json.dumps(cell_summary, indent=2))
        logger.info("Wrote %s", cell_path)

    summary = {
        **common_header,
        "demo_ids_no_distract": demo_ids_clean,
        "demo_ids_distract": demo_ids_distract,
        "human_demo_and_env_types": requested_types,
        "table": {f"demo_distract={dd}|env_distract={ed}": v for (dd, ed), v in table.items()},
        "per_trial": {
            f"demo_distract={dd}|env_distract={ed}": [
                {"task": t, "demo_id": di, "success": ok} for (t, di, ok) in rows
            ]
            for (dd, ed), rows in results.items()
        },
    }
    (output_dir / "summary.json").write_text(json.dumps(summary, indent=2))
    logger.info("Wrote %s", output_dir / "summary.json")
    return summary


def _log_table(table: dict, runner_name: str, scope: str, tasks: list[str]) -> None:
    def cell(dd: bool, ed: bool) -> str:
        if (dd, ed) not in table:
            return "    --"
        c = table[(dd, ed)]
        return f"{c['rate'] * 100:5.1f}% ({c['success']}/{c['total']})"

    for ln in (
        f"=== Eval summary (runner={runner_name}, scope={scope}, tasks={tasks}) ===",
        f"{'':>20} | {'env: clean':>16} | {'env: distract':>16}",
        f"{'demo: clean':>20} | {cell(False, False):>16} | {cell(False, True):>16}",
        f"{'demo: distract':>20} | {cell(True, False):>16} | {cell(True, True):>16}",
    ):
        logger.info(ln)


# ===========================================================================
# Legacy bookkeeping helpers (kept from the old eval_loop; not used by the new
# protocol driver, but retained for any external callers / smoke tests).
# ===========================================================================

EVAL_DEFAULTS_COMMON: dict = {
    "demo_source": "video",
    "prompt": "complete the task",
    "num_episodes_per_demo": 1,
    "horizon": 200,
    "chunk_size": 16,
    "obs_horizon": 16,
    "seed": 42,
    "eval_seed_start": 100,
    "max_demos": None,
    "save_video": False,
}

EVAL_REQUIRED_COMMON: set[str] = {"data_dir", "task", "condition", "output_dir"}


def load_config(config_path: str, *, extra_defaults: dict | None = None,
                extra_required: set[str] | None = None) -> argparse.Namespace:
    with open(config_path) as f:
        cfg = json.load(f)
    required = set(EVAL_REQUIRED_COMMON)
    if extra_required:
        required |= extra_required
    missing = required - set(cfg)
    if missing:
        raise ValueError(f"Missing required config fields: {sorted(missing)}")
    merged = {**EVAL_DEFAULTS_COMMON}
    if extra_defaults:
        merged.update(extra_defaults)
    merged.update(cfg)
    return argparse.Namespace(**merged)


def parse_eval_cli(extra_defaults: dict | None = None, extra_required: set[str] | None = None) -> argparse.Namespace:
    p = argparse.ArgumentParser()
    p.add_argument("--config", required=True, help="Path to JSON eval config.")
    args = p.parse_args()
    return load_config(args.config, extra_defaults=extra_defaults, extra_required=extra_required)


def condition_output_dir(output_root: str | Path, task: str, condition: str) -> Path:
    out = Path(output_root) / task / condition
    out.mkdir(parents=True, exist_ok=True)
    return out


@dataclass
class TrialResult:
    demo_id: str
    episode_index: int
    seed: int
    success: bool
    steps: int
    elapsed_seconds: float
    info: dict = field(default_factory=dict)


def write_summary(out_dir: Path, *, task: str, condition: str, checkpoint_dir: str,
                  trials: list["TrialResult"], eval_seed_range: tuple[int, int],
                  elapsed_seconds: float, extra: dict | None = None) -> dict:
    successes = [t for t in trials if t.success]
    by_demo: dict[str, dict] = {}
    for t in trials:
        slot = by_demo.setdefault(t.demo_id, {"demo_id": t.demo_id, "trials": []})
        slot["trials"].append({
            "episode_index": t.episode_index, "seed": t.seed, "success": t.success,
            "steps": t.steps, "elapsed_seconds": round(t.elapsed_seconds, 2), "info": t.info,
        })
    (out_dir / "results.json").write_text(json.dumps(list(by_demo.values()), indent=2))
    summary = {
        "task": task, "condition": condition, "checkpoint_dir": checkpoint_dir,
        "num_demos": len(by_demo),
        "num_episodes_per_demo": (len(trials) // len(by_demo)) if by_demo else 0,
        "total_trials": len(trials), "success_count": len(successes),
        "success_rate": (len(successes) / len(trials)) if trials else 0.0,
        "eval_seed_range": list(eval_seed_range), "elapsed_seconds": round(elapsed_seconds, 1),
    }
    if extra:
        summary.update(extra)
    (out_dir / "summary.json").write_text(json.dumps(summary, indent=2))
    return summary


class StopWatch:
    """Tiny context manager: ``with StopWatch() as sw: ...; sw.elapsed``."""

    def __enter__(self) -> "StopWatch":
        self._t0 = time.time()
        return self

    def __exit__(self, *exc) -> None:
        self.elapsed = time.time() - self._t0
