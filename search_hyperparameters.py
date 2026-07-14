#!/usr/bin/env python3
"""
Optuna-driven hyperparameter and architecture search for the Magic Pong DQN.

Runs a series of training trials via train_optimized.py with sampled
hyperparameters/architectures, scores each trial by evaluating it against
builtin AIs, and ranks the best resulting checkpoints (plus builtin AIs) with
an Elo tournament (magic_pong.ai.tournament) so the final ranking reflects
head-to-head strength, not just performance against a single opponent.

Each trial trains through train_optimized.py's full pipeline mode (curriculum
-> reward annealing -> self-play, with phase3 dense reward shaping) rather
than a single training phase, so a trial's score reflects the same multi-stage
procedure a real production model would get. This is significantly more
expensive per trial than a single phase -- roughly
2 * --curriculum-episodes + --anneal-episodes + --selfplay-episodes episodes
in total -- by design, in exchange for a ranking that's trustworthy relative
to real training. Each stage's episode budget is independently controllable
(see --curriculum-episodes/--anneal-episodes/--selfplay-episodes below) since
they are not interchangeable: curriculum builds basic interception/return
skills that annealing and self-play then build on, so an imbalanced split
(e.g. a tiny curriculum budget next to huge annealing/self-play budgets)
under-prepares the agent for the harder later stages.

Usage:
    python search_hyperparameters.py --study-name my_search --n-trials 50

Re-run the final tournament only (e.g. with a different --top-n), without
further training:
    python search_hyperparameters.py --study-name my_search --tournament-only

Parallel workers (automatic, single machine):
    python search_hyperparameters.py --study-name my_search --n-trials 50 \
        --n-workers 4
This spawns 4 subprocess workers -- each an independent OS process, since
training mutates process-global config (ai_config/game_config), so this must
be separate processes, never in-process threads or Optuna's n_jobs -- splits
--n-trials across them, and has them all train against the same sqlite-backed
study concurrently. Each worker's training output is written to
<output-root>/<study-name>/worker_<i>.log; the parent prints concise
started/finished/exit-code lines only. If a GPU is present, workers default
to CPU-only (dqn_ai.py auto-detects a single GPU with no coordination across
processes, so sharing it risks contention/OOM for comparatively little
benefit given how small these networks are) -- pass --allow-gpu-workers to
opt back into GPU sharing.

Scaling beyond one machine (manual, multiple shell invocations sharing one
sqlite study over a shared filesystem) is still possible by running this
script directly (with --n-workers omitted/1) from several hosts/shells
pointed at the same --output-root/--study-name.

Requires the "search" extra: pip install -e ".[search,ai]"
"""

from __future__ import annotations

import argparse
import functools
import os
import statistics
import subprocess
import sys
from pathlib import Path
from typing import Any

import optuna

import train_optimized
from magic_pong.ai import evaluation
from magic_pong.ai import tournament
from magic_pong.ai.models.dqn_ai import device as dqn_device

DEFAULT_OPPONENTS = ("random", "follow_ball", "defensive", "aggressive", "predictive")
ARCHITECTURE_HIDDEN_SIZES = (64, 128, 256, 512)
ARCHITECTURE_LAYER_WIDTHS = (32, 64, 128, 256, 512)


def split_trial_count(n_trials: int, n_workers: int) -> list[int]:
    """Split n_trials as evenly as possible across n_workers workers.

    Returns a list of length n_workers giving each worker's trial count. The
    remainder is front-loaded onto the first `n_trials % n_workers` workers,
    so counts differ by at most 1. If n_trials < n_workers, trailing workers
    get 0 -- callers should skip launching workers with a 0 count.
    """
    base, remainder = divmod(n_trials, n_workers)
    return [base + 1 if i < remainder else base for i in range(n_workers)]


def build_search_space(trial: optuna.Trial) -> dict[str, Any]:
    """Sample one trial's hyperparameters, including network architecture.

    Architecture is searched via a categorical switch: "geometric" trials
    tune hidden_size/layer_size (the network's existing halving scheme),
    "explicit" trials instead sample an arbitrary per-layer width list.

    gamma and the epsilon-greedy schedule (epsilon_decay/epsilon_min) are
    deliberately not searched: episodes here end after a single point, so
    gamma rarely swings outcomes much, and the exploration schedule is a
    secondary effect compared to learning_rate/architecture. Both are left
    at train_optimized.py's own CLI defaults (0.99 / 0.998 / 0.05) by simply
    not passing --gamma/--epsilon_decay/--epsilon_min in build_trial_argv.
    """
    params: dict[str, Any] = {
        "learning_rate": trial.suggest_float("learning_rate", 1e-5, 1e-2, log=True),
        "batch_size": trial.suggest_categorical("batch_size", [32, 64, 128, 256]),
        "memory_size": trial.suggest_categorical("memory_size", [10000, 20000, 50000, 100000]),
        "train_frequency": trial.suggest_int("train_frequency", 1, 8),
        "architecture_mode": trial.suggest_categorical(
            "architecture_mode", ["geometric", "explicit"]
        ),
    }

    if params["architecture_mode"] == "geometric":
        params["hidden_size"] = trial.suggest_categorical(
            "hidden_size", list(ARCHITECTURE_HIDDEN_SIZES)
        )
        params["layer_size"] = trial.suggest_int("layer_size", 1, 4)
    else:
        n_layers = trial.suggest_int("n_layers", 1, 4)
        params["widths"] = [
            trial.suggest_categorical(f"layer{i}_width", list(ARCHITECTURE_LAYER_WIDTHS))
            for i in range(n_layers)
        ]

    return params


def build_trial_argv(
    params: dict[str, Any],
    *,
    trial_dir: Path,
    curriculum_episodes: int,
    anneal_episodes: int,
    selfplay_episodes: int,
    max_steps_per_episode: int,
) -> list[str]:
    """Translate sampled hyperparameters into train_optimized.py CLI arguments.

    Uses --mode pipeline (curriculum -> reward annealing -> self-play) rather
    than a bare single-opponent phase, so a trial's score reflects the same
    multi-stage training procedure a real production model would get. Each
    stage's episode budget is passed explicitly (--pipeline_curriculum_episodes
    is a per-sub-phase count, used twice internally by curriculum training, so
    it costs 2 * curriculum_episodes) rather than relying on
    train_optimized.py's --total_episodes, which only feeds the annealing and
    self-play stages and leaves curriculum pinned to a small, unrelated
    default -- the exact mismatch this function exists to avoid.
    """
    argv = [
        "--mode",
        "pipeline",
        "--output_dir",
        str(trial_dir),
        "--model_prefix",
        "dqn",
        "--pipeline_curriculum_episodes",
        str(curriculum_episodes),
        "--pipeline_anneal_episodes",
        str(anneal_episodes),
        "--pipeline_selfplay_episodes",
        str(selfplay_episodes),
        "--max_steps_per_episode",
        str(max_steps_per_episode),
        "--learning_rate",
        str(params["learning_rate"]),
        "--batch_size",
        str(params["batch_size"]),
        "--memory_size",
        str(params["memory_size"]),
        "--train_frequency",
        str(params["train_frequency"]),
        "--headless",
        # Dense reward shaping (conservative default weights) so the DQN gets
        # a useful learning signal within a search trial's short episode budget,
        # instead of only sparse point/goal rewards.
        "--phase3_reward_shaping",
    ]
    if params["architecture_mode"] == "geometric":
        argv += [
            "--hidden_size",
            str(params["hidden_size"]),
            "--layer_size",
            str(params["layer_size"]),
        ]
    else:
        argv += ["--hidden_layers", ",".join(str(width) for width in params["widths"])]
    return argv


def run_trial_training(argv: list[str]) -> Path:
    """Train one DQN agent through the full pipeline, returning its final checkpoint path.

    Reimplements the small setup `train_optimized.main()` normally does before
    dispatching to a training-mode function (output dir creation, checkpoint
    dir defaulting) rather than calling `main()` itself, so we can disable plot
    generation by mutating the parsed Namespace -- there is no CLI flag for
    that, and `main()` re-parses argv internally, so mutating a separately
    parsed Namespace would have no effect on it.
    """
    args = train_optimized.parse_arguments(argv)
    args.save_plots = False

    Path(args.output_dir).mkdir(parents=True, exist_ok=True)
    if args.checkpoint_dir is None:
        args.checkpoint_dir = str(Path(args.output_dir) / "checkpoints")
    if args.checkpoint_interval > 0:
        Path(args.checkpoint_dir).mkdir(parents=True, exist_ok=True)

    agent = train_optimized.create_agent_from_args(args)
    _agent, training_results = train_optimized.train_pipeline(agent, args)
    return Path(training_results["final_model_path"])


def objective(
    trial: optuna.Trial,
    *,
    curriculum_episodes: int,
    anneal_episodes: int,
    selfplay_episodes: int,
    max_steps_per_episode: int,
    eval_episodes: int,
    output_root: Path,
) -> float:
    """Train and score one hyperparameter/architecture configuration.

    The score is the mean win rate against a fixed set of builtin AIs
    (`evaluation.evaluate_checkpoint`) -- Elo ranking against other trials only
    becomes meaningful once the whole study is done, so it's computed
    separately by `run_final_tournament`, not per-trial.
    """
    params = build_search_space(trial)
    trial_dir = output_root / f"trial_{trial.number:04d}"
    argv = build_trial_argv(
        params,
        trial_dir=trial_dir,
        curriculum_episodes=curriculum_episodes,
        anneal_episodes=anneal_episodes,
        selfplay_episodes=selfplay_episodes,
        max_steps_per_episode=max_steps_per_episode,
    )

    checkpoint_path = run_trial_training(argv)

    result = evaluation.evaluate_checkpoint(
        checkpoint_path,
        evaluation.EvaluationConfig(
            opponents=DEFAULT_OPPONENTS,
            episodes=eval_episodes,
            eval_epsilon=0.0,
            seed=trial.number,
        ),
    )
    mean_win_rate = statistics.mean(opponent.win_rate for opponent in result.opponents)

    trial.set_user_attr("checkpoint_path", str(checkpoint_path))
    trial.set_user_attr("eval_result", evaluation.result_to_dict(result))

    return mean_win_rate


def run_final_tournament(
    study: optuna.Study,
    *,
    top_n: int,
    episodes_per_match: int,
    output_root: Path,
) -> dict[str, tournament.EloRating]:
    """Rank the study's best checkpoints (plus builtin AIs) via an Elo tournament."""
    completed = [trial for trial in study.trials if trial.state == optuna.trial.TrialState.COMPLETE]
    top_trials = sorted(
        completed,
        key=lambda trial: trial.value if trial.value is not None else float("-inf"),
        reverse=True,
    )[:top_n]

    participants = [
        tournament.Participant(
            id=f"trial_{trial.number:04d}",
            kind="dqn",
            checkpoint_path=trial.user_attrs["checkpoint_path"],
        )
        for trial in top_trials
    ] + [
        tournament.Participant(id=f"builtin:{name}", kind="builtin", builtin_type=name)
        for name in tournament.BUILTIN_AI_TYPES
    ]

    return tournament.run_tournament(
        participants,
        episodes_per_match=episodes_per_match,
        output_dir=output_root / "final_tournament",
    )


def parse_search_args(argv: list[str] | None = None) -> argparse.Namespace:
    parser = argparse.ArgumentParser(
        description="Optuna hyperparameter/architecture search for the Magic Pong DQN"
    )
    parser.add_argument("--study-name", required=True)
    parser.add_argument("--output-root", default="search_runs")
    parser.add_argument("--n-trials", type=int, default=50)
    parser.add_argument("--timeout-seconds", type=float, default=None)
    parser.add_argument(
        "--curriculum-episodes",
        type=int,
        default=200,
        help=(
            "Episodes per curriculum sub-phase (used twice internally, so "
            "total curriculum cost is 2x this value). Builds basic "
            "interception/return skills before annealing and self-play."
        ),
    )
    parser.add_argument(
        "--anneal-episodes",
        type=int,
        default=1000,
        help="Episode budget for the reward-annealing (dense -> sparse) phase.",
    )
    parser.add_argument(
        "--selfplay-episodes",
        type=int,
        default=1000,
        help="Episode budget for the self-play (pure goal rewards) phase.",
    )
    parser.add_argument("--max-steps-per-episode", type=int, default=500)
    parser.add_argument("--eval-episodes", type=int, default=20)
    parser.add_argument("--top-n", type=int, default=8)
    parser.add_argument("--episodes-per-match", type=int, default=20)
    parser.add_argument("--seed", type=int, default=0)
    parser.add_argument(
        "--tournament-only",
        action="store_true",
        help="Skip training and just (re-)run the final tournament for an existing study.",
    )
    parser.add_argument(
        "--n-workers",
        type=int,
        default=1,
        help=(
            "Number of parallel OS-process workers to spawn for training trials "
            "(each worker re-invokes this script as a subprocess against the "
            "same sqlite-backed study). Default 1 trains in-process, unchanged "
            "from single-worker behavior."
        ),
    )
    parser.add_argument(
        "--allow-gpu-workers",
        action="store_true",
        help=(
            "When used with --n-workers > 1 and a GPU is present, allow workers "
            "to share it (default is to force worker subprocesses to CPU-only, "
            "since dqn_ai.py's GPU auto-detect has no cross-process coordination "
            "and these networks are small enough that GPU contention/OOM risk "
            "usually isn't worth it). No effect when --n-workers <= 1 or no GPU "
            "is present."
        ),
    )
    parser.add_argument("--skip-tournament", action="store_true", help=argparse.SUPPRESS)
    return parser.parse_args(argv)


def run_study_optimize(study: optuna.Study, args: argparse.Namespace, output_root: Path) -> None:
    """Run study.optimize in-process with args' training-relevant settings.

    Used both by the single-process path (--n-workers <= 1) and by each
    worker subprocess spawned by run_parallel_workers (which calls main()
    again with its own slice of --n-trials and offset --seed already baked
    into its own argv).
    """
    study.optimize(
        functools.partial(
            objective,
            curriculum_episodes=args.curriculum_episodes,
            anneal_episodes=args.anneal_episodes,
            selfplay_episodes=args.selfplay_episodes,
            max_steps_per_episode=args.max_steps_per_episode,
            eval_episodes=args.eval_episodes,
            output_root=output_root,
        ),
        n_trials=args.n_trials,
        timeout=args.timeout_seconds,
    )


def build_worker_argv(args: argparse.Namespace, *, n_trials: int, seed: int) -> list[str]:
    """Build one worker subprocess's argv: the parent's training flags, with
    --n-trials/--seed overridden and --skip-tournament forwarded so only the
    parent runs the final tournament once all workers finish."""
    argv = [
        "--study-name",
        args.study_name,
        "--output-root",
        args.output_root,
        "--n-trials",
        str(n_trials),
        "--curriculum-episodes",
        str(args.curriculum_episodes),
        "--anneal-episodes",
        str(args.anneal_episodes),
        "--selfplay-episodes",
        str(args.selfplay_episodes),
        "--max-steps-per-episode",
        str(args.max_steps_per_episode),
        "--eval-episodes",
        str(args.eval_episodes),
        "--seed",
        str(seed),
        "--skip-tournament",
    ]
    if args.timeout_seconds is not None:
        argv += ["--timeout-seconds", str(args.timeout_seconds)]
    return argv


def run_parallel_workers(args: argparse.Namespace, output_root: Path) -> None:
    """Spawn args.n_workers subprocess workers, each training a slice of
    args.n_trials against the shared sqlite-backed study, then wait for all
    to finish. Does not run the final tournament itself -- the caller (main())
    is responsible for that once this returns."""
    trial_counts = split_trial_count(args.n_trials, args.n_workers)
    launched = sum(1 for count in trial_counts if count > 0)
    if launched < args.n_workers:
        print(
            f"[search] only {launched}/{args.n_workers} worker(s) have trials to "
            f"run ({args.n_trials} trials split across {args.n_workers} workers)."
        )

    gpu_present = dqn_device.type == "cuda"
    force_cpu_for_workers = gpu_present and not args.allow_gpu_workers
    if force_cpu_for_workers:
        print(
            "[search] GPU detected; defaulting workers to CPU-only to avoid "
            "context/memory contention (pass --allow-gpu-workers to override)."
        )

    processes: list[tuple[int, subprocess.Popen, Path]] = []
    for i, n_trials in enumerate(trial_counts):
        if n_trials == 0:
            continue
        worker_env = os.environ.copy()
        if force_cpu_for_workers:
            worker_env["CUDA_VISIBLE_DEVICES"] = ""
        worker_argv = build_worker_argv(args, n_trials=n_trials, seed=args.seed + i)
        log_path = output_root / f"worker_{i}.log"
        print(
            f"[search] launching worker {i} ({n_trials} trials, seed={args.seed + i}) -> log: {log_path}"
        )
        with log_path.open("w") as log_file:
            proc = subprocess.Popen(
                [sys.executable, __file__, *worker_argv],
                stdout=log_file,
                stderr=subprocess.STDOUT,
                env=worker_env,
            )
        processes.append((i, proc, log_path))

    failed: list[tuple[int, Path]] = []
    for i, proc, log_path in processes:
        exit_code = proc.wait()
        status = "OK" if exit_code == 0 else f"FAILED (exit {exit_code})"
        print(f"[search] worker {i} finished: {status}")
        if exit_code != 0:
            failed.append((i, log_path))

    if failed:
        print(
            f"[search] WARNING: {len(failed)}/{len(processes)} worker(s) failed -- "
            "proceeding to tournament with whatever trials completed. Failed worker logs:"
        )
        for i, log_path in failed:
            print(f"  worker {i}: {log_path}")


def main(argv: list[str] | None = None) -> None:
    args = parse_search_args(argv)
    if args.n_workers < 1:
        raise SystemExit("--n-workers must be >= 1")
    output_root = Path(args.output_root) / args.study_name
    output_root.mkdir(parents=True, exist_ok=True)

    study = optuna.create_study(
        study_name=args.study_name,
        storage=f"sqlite:///{output_root / f'{args.study_name}.db'}",
        direction="maximize",
        load_if_exists=True,
        sampler=optuna.samplers.TPESampler(seed=args.seed),
    )

    if not args.tournament_only:
        if args.n_workers > 1:
            run_parallel_workers(args, output_root)
        else:
            run_study_optimize(study, args, output_root)

    if not args.skip_tournament:
        ratings = run_final_tournament(
            study,
            top_n=args.top_n,
            episodes_per_match=args.episodes_per_match,
            output_root=output_root,
        )
        print(tournament.format_leaderboard(ratings))


if __name__ == "__main__":
    main()
