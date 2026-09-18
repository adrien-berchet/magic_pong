"""Reproducible fixed-frame gamma pilots, with held-out evaluation untouched.

Run separate processes with the same arguments and different ``--seed-filter``
values to parallelize seeds. Completed seeds are verified and reused; partial
seeds require a new output directory. This runner never resumes partial RL.
"""

from __future__ import annotations

import argparse
import hashlib
import json
import math
import os
import platform
import tempfile
import time
from collections import Counter
from collections.abc import Iterator
from contextlib import contextmanager
from dataclasses import asdict
from dataclasses import dataclass
from pathlib import Path
from typing import Any

import numpy as np
import torch

from magic_pong.ai.competitive import CompetitiveConfig
from magic_pong.ai.competitive import competitive_result_to_dict
from magic_pong.ai.competitive import evaluate_competitive_agent
from magic_pong.ai.evaluation import _seed_everything
from magic_pong.ai.evaluation import preserve_global_rng_state
from magic_pong.ai.league import OpponentLeague
from magic_pong.ai.models import dqn_ai
from magic_pong.ai.models.dqn_ai import DQNAgent
from magic_pong.ai.simulator_pretraining import SimulatorPretrainer
from magic_pong.core.game_engine import GameEngine
from magic_pong.utils.config import AIConfig
from magic_pong.utils.config import GameConfig
from magic_pong.utils.config import _restore_config
from magic_pong.utils.config import ai_config
from magic_pong.utils.config import game_config

SCHEMA_VERSION = "magic_pong.pilot.v1"
STRATA = ((1, False), (2, False), (1, True), (2, True))


@dataclass(frozen=True)
class PilotConfig:
    seeds: tuple[int, ...] = (11, 22, 33)
    gammas: tuple[float, ...] = (0.99, 0.999)
    env_steps: int = 1_000_000
    pretraining_labels: int = 5000
    hidden_layers: tuple[int, ...] = (256, 128)
    learning_rate: float = 0.0003
    batch_size: int = 64
    memory_size: int = 20000
    train_frequency: int = 10
    min_replay_size: int = 1000
    target_update_frequency: int = 1000
    max_episode_steps: int = 10000
    training_max_score: int = 1
    epsilon_start: float = 0.3
    epsilon_end: float = 0.05
    epsilon_decay_fraction: float = 0.8
    shaping_decay_fraction: float = 0.5
    snapshot_interval: int = 100000
    midpoint_validation: bool = True
    eval_seeds: tuple[int, ...] = (101, 102, 103)
    held_out_seeds: tuple[int, ...] = (10001, 10002, 10003)
    eval_max_steps: int = 18000
    eval_max_score: int = 11
    teacher_horizon: int = 90
    teacher_action_steps: int = 6
    teacher_rollout_steps: int = 120
    pretraining_batch_labels: int = 256
    threads: int = 1

    def __post_init__(self) -> None:
        for name in (
            "seeds",
            "gammas",
            "hidden_layers",
            "eval_seeds",
            "held_out_seeds",
        ):
            object.__setattr__(self, name, tuple(getattr(self, name)))
        if not self.seeds or len(set(self.seeds)) != len(self.seeds):
            raise ValueError("seeds must be nonempty and unique")
        if any(type(seed) is not int or not 0 <= seed < 2**32 for seed in self.seeds):
            raise ValueError("training seeds must be unsigned 32-bit integers")
        if not self.gammas or len(set(self.gammas)) != len(self.gammas):
            raise ValueError("gammas must be nonempty and unique")
        if any(not math.isfinite(gamma) or not 0 < gamma < 1 for gamma in self.gammas):
            raise ValueError("gammas must lie strictly between zero and one")
        if type(self.env_steps) is not int or self.env_steps < 4 or self.env_steps % 4:
            raise ValueError("env_steps must be a positive multiple of four for balanced strata")
        if type(self.pretraining_labels) is not int or self.pretraining_labels < 0:
            raise ValueError("pretraining_labels must be a nonnegative integer")
        for name in (
            "batch_size",
            "memory_size",
            "train_frequency",
            "min_replay_size",
            "target_update_frequency",
            "max_episode_steps",
            "training_max_score",
            "threads",
            "teacher_horizon",
            "teacher_action_steps",
            "teacher_rollout_steps",
            "pretraining_batch_labels",
        ):
            if type(getattr(self, name)) is not int or getattr(self, name) <= 0:
                raise ValueError(f"{name} must be a positive integer")
        if self.memory_size < self.batch_size or self.min_replay_size > self.memory_size:
            raise ValueError("replay capacity must accommodate batch and warmup sizes")
        if not self.hidden_layers or any(
            type(width) is not int or width <= 0 for width in self.hidden_layers
        ):
            raise ValueError("hidden_layers must contain positive widths")
        if not math.isfinite(self.learning_rate) or self.learning_rate <= 0:
            raise ValueError("learning_rate must be finite and positive")
        if not 0 <= self.epsilon_end <= self.epsilon_start <= 1:
            raise ValueError("epsilon must satisfy 0 <= end <= start <= 1")
        if not 0 < self.epsilon_decay_fraction <= 1 or not 0 < self.shaping_decay_fraction <= 1:
            raise ValueError("decay fractions must be in (0, 1]")
        if type(self.snapshot_interval) is not int or self.snapshot_interval < 0:
            raise ValueError("snapshot_interval must be a nonnegative integer")
        self.evaluation_config()

    def evaluation_config(self) -> CompetitiveConfig:
        return CompetitiveConfig(
            seeds=self.eval_seeds,
            held_out_seeds=self.held_out_seeds,
            max_steps=self.eval_max_steps,
            max_score=self.eval_max_score,
        )


def epsilon_at_step(config: PilotConfig, completed_steps: int) -> float:
    progress = min(
        1.0,
        max(0, completed_steps) / (config.env_steps * config.epsilon_decay_fraction),
    )
    return config.epsilon_start + progress * (config.epsilon_end - config.epsilon_start)


def shaping_at_step(config: PilotConfig, completed_steps: int) -> float:
    return max(
        0.0,
        1.0 - max(0, completed_steps) / (config.env_steps * config.shaping_decay_fraction),
    )


def _sha256(path: Path) -> str:
    return hashlib.sha256(path.read_bytes()).hexdigest()


def _network_hash(agent: DQNAgent) -> str:
    digest = hashlib.sha256()
    for name, tensor in sorted(agent.q_network.state_dict().items()):
        digest.update(name.encode())
        digest.update(str((tuple(tensor.shape), tensor.dtype)).encode())
        digest.update(tensor.detach().cpu().contiguous().numpy().tobytes())
    return digest.hexdigest()


def _json_data(value: Any) -> Any:
    return json.loads(json.dumps(value, sort_keys=True))


def _write_json(path: Path, value: Any) -> None:
    """Create an artifact exclusively: a pilot never overwrites previous work."""
    with path.open("x", encoding="utf-8") as stream:
        json.dump(value, stream, indent=2, sort_keys=True)
        stream.write("\n")


def _manifest(config: PilotConfig) -> dict[str, Any]:
    package = Path(__file__).resolve().parents[1]
    sources = sorted(package.rglob("*.py"))
    return _json_data(
        {
            "schema_version": SCHEMA_VERSION,
            "config": asdict(config),
            "evaluation_protocol": asdict(config.evaluation_config()),
            "held_out_evaluation_executed": False,
            "training_game_config": GameConfig(MAX_SCORE=config.training_max_score).model_dump(
                mode="json"
            ),
            "training_ai_config": AIConfig(
                HEADLESS_MODE=True,
                FAST_MODE_MULTIPLIER=1.0,
                REWARD_SHAPING_MODE="phase3",
            ).model_dump(mode="json"),
            "schedules": {
                "unit": "completed environment frames before the next action",
                "epsilon": {
                    "shape": "linear_then_constant",
                    "start": config.epsilon_start,
                    "end": config.epsilon_end,
                    "end_step": config.env_steps * config.epsilon_decay_fraction,
                    "agent_epsilon_decay": 1.0,
                },
                "auxiliary_reward_scale": {
                    "shape": "linear_then_zero",
                    "start": 1.0,
                    "end": 0.0,
                    "end_step": config.env_steps * config.shaping_decay_fraction,
                    "score_rewards_scaled": False,
                },
            },
            "runtime": {
                "python": platform.python_version(),
                "torch": torch.__version__,
                "numpy": np.__version__,
                "device": "cpu",
                "threads": config.threads,
            },
            "source_sha256": {str(path.relative_to(package)): _sha256(path) for path in sources},
        }
    )


def _ensure_manifest(output_dir: Path, manifest: dict[str, Any]) -> None:
    output_dir.mkdir(parents=True, exist_ok=True)
    destination = output_dir / "manifest.json"
    # Hard-link publication is atomic and exclusive, including concurrent seed workers.
    with tempfile.NamedTemporaryFile(mode="w", encoding="utf-8", dir=output_dir) as stream:
        json.dump(manifest, stream, indent=2, sort_keys=True)
        stream.write("\n")
        stream.flush()
        os.fsync(stream.fileno())
        try:
            os.link(stream.name, destination)
        except FileExistsError:
            if json.loads(destination.read_text(encoding="utf-8")) != manifest:
                raise ValueError(
                    "output manifest differs from this pilot configuration or source"
                ) from None


@contextmanager
def _isolated_training(config: PilotConfig) -> Iterator[None]:
    original_game, original_ai = game_config.model_copy(deep=True), ai_config.model_copy(deep=True)
    original_device = dqn_ai.device
    original_threads = torch.get_num_threads()
    deterministic = torch.are_deterministic_algorithms_enabled()
    warn_only = torch.is_deterministic_algorithms_warn_only_enabled()
    with preserve_global_rng_state():
        try:
            _restore_config(game_config, GameConfig(MAX_SCORE=config.training_max_score))
            _restore_config(
                ai_config,
                AIConfig(
                    HEADLESS_MODE=True,
                    FAST_MODE_MULTIPLIER=1.0,
                    REWARD_SHAPING_MODE="phase3",
                ),
            )
            dqn_ai.device = torch.device("cpu")
            torch.set_num_threads(config.threads)
            torch.use_deterministic_algorithms(True)
            yield
        finally:
            _restore_config(game_config, original_game)
            _restore_config(ai_config, original_ai)
            dqn_ai.device = original_device
            torch.set_num_threads(original_threads)
            torch.use_deterministic_algorithms(deterministic, warn_only=warn_only)


def _new_agent(config: PilotConfig, gamma: float) -> DQNAgent:
    return DQNAgent(
        gamma=gamma,
        hidden_layers=list(config.hidden_layers),
        lr=config.learning_rate,
        batch_size=config.batch_size,
        memory_size=config.memory_size,
        train_frequency=config.train_frequency,
        min_replay_size=config.min_replay_size,
        target_update_frequency=config.target_update_frequency,
        epsilon=config.epsilon_start,
        epsilon_min=config.epsilon_end,
        epsilon_decay=1.0,
    )


def _save_agent(agent: DQNAgent, path: Path) -> None:
    # DQN save_model writes directly; reserve its unique pathname first.
    with path.open("xb"):
        pass
    agent.save_model(str(path))


def _evaluate(
    agent: DQNAgent,
    config: PilotConfig,
    path: Path,
    checkpoint_path: Path,
) -> dict[str, Any]:
    result = evaluate_competitive_agent(agent, config.evaluation_config(), split="validation")
    payload = competitive_result_to_dict(result)
    payload["network_sha256"] = _network_hash(agent)
    payload["checkpoint_path"] = str(checkpoint_path)
    _write_json(path, payload)
    return {
        "report": str(path),
        "rank": payload["rank"],
        "total_matches": payload["total_matches"],
        "network_sha256": payload["network_sha256"],
        "checkpoint_path": str(checkpoint_path),
    }


def _pretrain(agent: DQNAgent, config: PilotConfig, seed: int) -> dict[str, Any]:
    trainer = SimulatorPretrainer(
        seed=seed,
        teacher_horizon=config.teacher_horizon,
        teacher_action_steps=config.teacher_action_steps,
        rollout_steps=config.teacher_rollout_steps,
        supervised_batch_size=config.batch_size,
    )
    totals: Counter[str] = Counter()
    counts: Counter[str] = Counter()
    losses = []
    quotas = {
        1: (config.pretraining_labels + 1) // 2,
        2: config.pretraining_labels // 2,
    }
    side_labels = {1: 0, 2: 0}
    while any(quotas.values()):
        for side in (1, 2):
            labels = min(quotas[side], config.pretraining_batch_labels)
            if labels == 0:
                continue
            stats = trainer.pretraining_step(agent, player_id=side, num_steps=labels)
            quotas[side] -= stats["steps"]
            side_labels[side] += stats["steps"]
            counts.update(stats["scenario_counts"])
            losses.append(stats["average_loss"])
            for key in (
                "steps",
                "physics_steps",
                "supervised_updates",
                "learner_decisions",
                "goals_for",
                "goals_against",
            ):
                totals[key] += stats[key]
            print(
                f"Pilot seed {seed}: teacher labels {totals['steps']}/{config.pretraining_labels}, "
                f"batch loss {stats['average_loss']:.5f}",
                flush=True,
            )
    return {
        "labels": totals["steps"],
        "labels_by_side": side_labels,
        "scenario_counts": dict(counts),
        "mean_batch_loss": sum(losses) / len(losses) if losses else 0.0,
        **dict(totals),
    }


def _train_branch(
    agent: DQNAgent,
    config: PilotConfig,
    seed: int,
    directory: Path,
    pretrained_path: Path,
    initial_evaluation: dict[str, Any],
) -> dict[str, Any]:
    league = OpponentLeague(directory / "league", seed=seed)
    reference = league.add_checkpoint(pretrained_path, role="pilot_initial_reference")
    initial_hash = _network_hash(agent)
    initial_replay_size = len(agent.memory)
    initial_optimizer_size = len(agent.optimizer.state)
    _save_agent(agent, directory / "initial.pth")
    _seed_everything(seed)  # Loading the same reference cannot perturb the paired training stream.
    stratum_steps = dict.fromkeys(STRATA, 0)
    quota = config.env_steps // len(STRATA)
    total_steps = episodes = wins = timeouts = goals_for = goals_against = 0
    total_reward = 0.0
    midpoint = config.env_steps // 2 if config.midpoint_validation else None
    curve = [{"env_steps": 0, **initial_evaluation}]
    started = time.monotonic()
    try:
        with (directory / "episodes.jsonl").open("x", encoding="utf-8") as log:
            while total_steps < config.env_steps:
                for side, bonuses in STRATA:
                    remaining = quota - stratum_steps[side, bonuses]
                    if not remaining:
                        continue
                    game_config.BONUSES_ENABLED = bonuses
                    opponent_id, opponent = league.next_opponent()
                    engine = GameEngine(headless=True)
                    players = (agent, opponent) if side == 1 else (opponent, agent)
                    engine.set_players(*players)
                    engine.start_game()
                    limit = min(remaining, config.max_episode_steps)
                    engine.ai_environment.max_steps = limit
                    episode_steps = episode_for = episode_against = 0
                    episode_reward = 0.0
                    start_step = total_steps
                    loss_start = len(agent.loss_history)
                    winner = 0
                    try:
                        while engine.is_running() and episode_steps < limit:
                            agent.epsilon = epsilon_at_step(config, total_steps)
                            ai_config.MICRO_REWARD_SCALE = shaping_at_step(config, total_steps)
                            result = engine.update(
                                dt=game_config.GAME_SPEED_MULTIPLIER / game_config.FPS
                            )
                            total_steps += 1
                            episode_steps += 1
                            stratum_steps[side, bonuses] += 1
                            episode_reward += result["rewards"][f"player{side}"]
                            goals = result["events"].get("goals", [])
                            episode_for += sum(goal["player"] == side for goal in goals)
                            episode_against += sum(goal["player"] != side for goal in goals)
                            winner = int(result["info"].get("winner", 0))
                            if (
                                config.snapshot_interval
                                and total_steps % config.snapshot_interval == 0
                            ):
                                league.add_snapshot(agent, metadata={"env_steps": total_steps})
                            if total_steps == midpoint:
                                print(
                                    f"Pilot seed {seed}, gamma {agent.gamma}: midpoint validation at {total_steps} frames",
                                    flush=True,
                                )
                                _save_agent(agent, directory / "midpoint.pth")
                                report = _evaluate(
                                    agent,
                                    config,
                                    directory / "midpoint.validation.json",
                                    directory / "midpoint.pth",
                                )
                                curve.append({"env_steps": total_steps, **report})
                            if total_steps % 50000 == 0:
                                print(
                                    f"Pilot seed {seed}, gamma {agent.gamma}: {total_steps}/{config.env_steps} frames, "
                                    f"epsilon={agent.epsilon:.3f}, shaping={ai_config.MICRO_REWARD_SCALE:.3f}",
                                    flush=True,
                                )
                            if result["done"]:
                                break
                    finally:
                        if engine.is_running():
                            engine.stop_game()
                    if episode_steps == 0:
                        raise RuntimeError("training environment made no progress")
                    timeout = winner == 0
                    league.record_outcome(opponent_id, won=winner == side, timeout=timeout)
                    episodes += 1
                    wins += int(winner == side)
                    timeouts += int(timeout)
                    goals_for += episode_for
                    goals_against += episode_against
                    total_reward += episode_reward
                    losses = agent.loss_history[loss_start:]
                    row = {
                        "episode": episodes,
                        "opponent": opponent_id,
                        "player_side": side,
                        "bonuses_enabled": bonuses,
                        "steps": episode_steps,
                        "start_env_step": start_step,
                        "end_env_step": total_steps,
                        "winner": winner,
                        "timeout": timeout,
                        "goals_for": episode_for,
                        "terminated": bool(result["info"].get("terminated", winner != 0)),
                        "truncated": bool(result["info"].get("truncated", timeout)),
                        "stratum_budget_cutoff": timeout
                        and episode_steps == limit
                        and limit < config.max_episode_steps,
                        "goals_against": episode_against,
                        "reward": episode_reward,
                        "updates": len(losses),
                        "mean_loss": sum(losses) / len(losses) if losses else None,
                        "epsilon_start": epsilon_at_step(config, start_step),
                        "epsilon_end": epsilon_at_step(config, total_steps),
                        "shaping_scale_start": shaping_at_step(config, start_step),
                        "shaping_scale_end": shaping_at_step(config, total_steps),
                    }
                    log.write(json.dumps(row) + "\n")
                    log.flush()
        training_seconds = time.monotonic() - started
        agent.epsilon = epsilon_at_step(config, total_steps)
        ai_config.MICRO_REWARD_SCALE = shaping_at_step(config, total_steps)
        _save_agent(agent, directory / "final.pth")
        print(
            f"Pilot seed {seed}, gamma {agent.gamma}: final validation at {total_steps} frames",
            flush=True,
        )
        report = _evaluate(
            agent, config, directory / "final.validation.json", directory / "final.pth"
        )
        curve.append({"env_steps": total_steps, **report})
        return {
            "gamma": agent.gamma,
            "initial_network_sha256": initial_hash,
            "initial_replay_size": initial_replay_size,
            "initial_optimizer_state_size": initial_optimizer_size,
            "reference_sha256": league.entries[reference].metadata["sha256"],
            "league_directory": str(directory / "league"),
            "initial_checkpoint": str(directory / "initial.pth"),
            "final_checkpoint": str(directory / "final.pth"),
            "learning_curve": curve,
            "training": {
                "env_steps": total_steps,
                "episodes": episodes,
                "wins": wins,
                "losses": episodes - wins - timeouts,
                "timeouts": timeouts,
                "goals_for": goals_for,
                "goals_against": goals_against,
                "reward": total_reward,
                "updates": agent.training_step,
                "mean_loss": (
                    sum(agent.loss_history) / len(agent.loss_history)
                    if agent.loss_history
                    else None
                ),
                "final_epsilon": agent.epsilon,
                "final_shaping_scale": ai_config.MICRO_REWARD_SCALE,
                "seconds_including_midpoint_validation": training_seconds,
                "strata": [
                    {"player_side": side, "bonuses_enabled": bonuses, "steps": steps}
                    for (side, bonuses), steps in stratum_steps.items()
                ],
            },
        }
    finally:
        league.cleanup()


def _run_seed(config: PilotConfig, seed: int, directory: Path) -> dict[str, Any]:
    _seed_everything(seed)
    agent = _new_agent(config, config.gammas[0])
    started = time.monotonic()
    _save_agent(agent, directory / "untrained.pth")
    untrained_hash = _network_hash(agent)
    print(f"Pilot seed {seed}: validating untrained policy", flush=True)
    baseline = _evaluate(
        agent,
        config,
        directory / "untrained.validation.json",
        directory / "untrained.pth",
    )
    print(
        f"Pilot seed {seed}: collecting {config.pretraining_labels} teacher labels",
        flush=True,
    )
    pretraining_started = time.monotonic()
    pretraining = _pretrain(agent, config, seed)
    pretraining["seconds"] = time.monotonic() - pretraining_started
    pretrained_path = directory / "pretrained.pth"
    _save_agent(agent, pretrained_path)
    pretrained_hash = _network_hash(agent)
    print(f"Pilot seed {seed}: validating pretrained policy", flush=True)
    pretrained_evaluation = _evaluate(
        agent,
        config,
        directory / "pretrained.validation.json",
        pretrained_path,
    )
    branches = {}
    for gamma in config.gammas:
        branch_dir = directory / f"gamma_{gamma}"
        branch_dir.mkdir()
        _seed_everything(seed)
        branch = _new_agent(config, gamma)
        branch.q_network.load_state_dict(agent.q_network.state_dict())
        branch.update_target_network()
        print(
            f"Pilot seed {seed}, gamma {gamma}: training {config.env_steps} frames",
            flush=True,
        )
        branches[str(gamma)] = _train_branch(
            branch,
            config,
            seed,
            branch_dir,
            pretrained_path,
            pretrained_evaluation,
        )
        _write_json(branch_dir / "summary.json", branches[str(gamma)])
    artifacts = {
        str(path.relative_to(directory)): _sha256(path)
        for path in sorted(directory.rglob("*"))
        if path.is_file()
    }
    result = _json_data(
        {
            "status": "complete",
            "seed": seed,
            "schema_version": SCHEMA_VERSION,
            "untrained_network_sha256": untrained_hash,
            "pretrained_network_sha256": pretrained_hash,
            "pretrained_checkpoint_sha256": _sha256(pretrained_path),
            "untrained_evaluation": baseline,
            "pretrained_evaluation": pretrained_evaluation,
            "pretraining": pretraining,
            "branches": branches,
            "seconds": time.monotonic() - started,
            "artifacts": artifacts,
        }
    )
    _write_json(directory / "result.json", result)
    return result


def run_pilot(
    config: PilotConfig,
    output_dir: str | Path,
    *,
    seed_filter: tuple[int, ...] | None = None,
) -> dict[str, Any]:
    """Run selected seeds; validation and pretraining do not consume RL budgets."""
    selected = tuple(seed_filter) if seed_filter is not None else config.seeds
    if not selected or len(selected) != len(set(selected)) or set(selected) - set(config.seeds):
        raise ValueError("seed_filter must contain unique configured seeds")
    output = Path(output_dir).resolve()
    _ensure_manifest(output, _manifest(config))
    results = {}
    with _isolated_training(config):
        for seed in selected:
            directory = output / f"seed_{seed}"
            if directory.exists():
                result_path = directory / "result.json"
                if not result_path.exists():
                    raise FileExistsError(
                        f"incomplete pilot seed at {directory}; choose a new output directory"
                    )
                result = json.loads(result_path.read_text(encoding="utf-8"))
                if result.get("status") != "complete":
                    raise FileExistsError(f"incomplete pilot result at {directory}")
                for relative, digest in result["artifacts"].items():
                    artifact = directory / relative
                    if not artifact.is_file() or _sha256(artifact) != digest:
                        raise ValueError(f"completed pilot artifact changed: {artifact}")
            else:
                directory.mkdir()
                try:
                    result = _run_seed(config, seed, directory)
                except BaseException as error:
                    _write_json(
                        directory / "failure.json",
                        {"status": "incomplete", "error": str(error)},
                    )
                    raise
            results[str(seed)] = result
    return {"manifest": str(output / "manifest.json"), "seeds": results}


def main(argv: list[str] | None = None) -> int:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--output-dir", type=Path, required=True)
    defaults = PilotConfig()
    for name in ("seeds", "eval_seeds", "held_out_seeds", "hidden_layers"):
        parser.add_argument(
            "--" + name.replace("_", "-"),
            nargs="+",
            type=int,
            default=getattr(defaults, name),
        )
    parser.add_argument("--gammas", nargs="+", type=float, default=defaults.gammas)
    parser.add_argument("--seed-filter", nargs="+", type=int)
    for name in (
        "env_steps",
        "pretraining_labels",
        "batch_size",
        "memory_size",
        "train_frequency",
        "min_replay_size",
        "target_update_frequency",
        "max_episode_steps",
        "training_max_score",
        "snapshot_interval",
        "eval_max_steps",
        "eval_max_score",
        "teacher_horizon",
        "teacher_action_steps",
        "teacher_rollout_steps",
        "pretraining_batch_labels",
        "threads",
    ):
        parser.add_argument(
            "--" + name.replace("_", "-"), type=int, default=getattr(defaults, name)
        )
    for name in (
        "learning_rate",
        "epsilon_start",
        "epsilon_end",
        "epsilon_decay_fraction",
        "shaping_decay_fraction",
    ):
        parser.add_argument(
            "--" + name.replace("_", "-"), type=float, default=getattr(defaults, name)
        )
    parser.add_argument("--no-midpoint-validation", action="store_true")
    args = vars(parser.parse_args(argv))
    output = args.pop("output_dir")
    seeds = args.pop("seed_filter")
    args["midpoint_validation"] = not args.pop("no_midpoint_validation")
    result = run_pilot(PilotConfig(**args), output, seed_filter=tuple(seeds) if seeds else None)
    print(f"Pilot complete: {', '.join(result['seeds'])}; manifest: {result['manifest']}")
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
