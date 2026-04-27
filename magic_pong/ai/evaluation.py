"""Reusable evaluation harness for Magic Pong AI agents."""

from __future__ import annotations

import argparse
import csv
import hashlib
import io
import json
import math
import os
import random
import statistics
from collections.abc import Callable
from collections.abc import Mapping
from contextlib import contextmanager
from dataclasses import asdict
from dataclasses import dataclass
from pathlib import Path
from typing import Any
from typing import cast

import numpy as np

os.environ.setdefault("PYGAME_HIDE_SUPPORT_PROMPT", "1")

from magic_pong.ai.models.simple_ai import create_ai  # noqa: E402
from magic_pong.core.game_engine import TrainingManager  # noqa: E402
from magic_pong.utils.config import game_config  # noqa: E402

SCHEMA_VERSION = "magic_pong.evaluation.v1"
DEFAULT_OPPONENTS = ("follow_ball",)
NON_DETERMINISTIC_OPPONENTS = frozenset({"training_dummy"})


@dataclass(frozen=True)
class EvaluationConfig:
    """Configuration for deterministic agent evaluation."""

    opponents: tuple[str, ...] | str = DEFAULT_OPPONENTS
    episodes: int = 10
    max_steps: int = 1000
    seed: int = 0
    eval_epsilon: float = 0.0
    max_score: int | None = None
    bonuses_enabled: bool | None = None
    ball_direction: int = 0
    ball_angle_deg: float | None = None
    include_episodes: bool = False

    def __post_init__(self) -> None:
        object.__setattr__(self, "opponents", _normalize_opponents(self.opponents))

        if not self.opponents:
            raise ValueError("EvaluationConfig.opponents must contain at least one opponent")
        non_deterministic = sorted(set(self.opponents) & NON_DETERMINISTIC_OPPONENTS)
        if non_deterministic:
            names = ", ".join(non_deterministic)
            raise ValueError(
                f"Evaluation opponent {names!r} is not deterministic because it uses wall-clock "
                "time. Use a deterministic opponent for evaluation."
            )
        if self.episodes < 0:
            raise ValueError("EvaluationConfig.episodes must be non-negative")
        if self.max_steps <= 0:
            raise ValueError("EvaluationConfig.max_steps must be positive")
        if not 0.0 <= self.eval_epsilon <= 1.0:
            raise ValueError("EvaluationConfig.eval_epsilon must be between 0.0 and 1.0")
        if self.max_score is not None and self.max_score <= 0:
            raise ValueError("EvaluationConfig.max_score must be positive when set")
        if self.ball_direction not in (-1, 0, 1):
            raise ValueError("EvaluationConfig.ball_direction must be -1, 0, or 1")


@dataclass(frozen=True)
class EpisodeEvaluation:
    """Metrics captured for a single evaluated episode."""

    opponent: str
    episode_index: int
    seed: int
    winner: int
    outcome: str
    agent_won: bool
    timeout: bool
    steps: int
    total_reward_agent: float
    total_reward_opponent: float
    goals_for: int
    goals_against: int
    goal_diff: int


@dataclass(frozen=True)
class OpponentEvaluation:
    """Aggregate metrics for one opponent."""

    opponent: str
    episodes: int
    wins: int
    losses: int
    draws: int
    timeouts: int
    win_rate: float
    avg_reward: float
    std_reward: float
    min_reward: float
    max_reward: float
    avg_steps: float
    median_steps: float
    avg_goal_diff: float


@dataclass(frozen=True)
class EvaluationResult:
    """Top-level evaluation result."""

    config: EvaluationConfig
    opponents: tuple[OpponentEvaluation, ...]
    episodes: tuple[EpisodeEvaluation, ...] = ()
    checkpoint_path: str | None = None
    checkpoint_metadata: dict[str, Any] | None = None
    schema_version: str = SCHEMA_VERSION


OpponentFactory = Callable[..., Any]
TrainingManagerFactory = Callable[..., Any]


def evaluate_agent(
    agent: Any,
    config: EvaluationConfig,
    opponent_factory: OpponentFactory = create_ai,
    training_manager_factory: TrainingManagerFactory = TrainingManager,
) -> EvaluationResult:
    """Evaluate an already-created agent against configured opponents."""
    all_episodes: list[EpisodeEvaluation] = []
    opponents = _normalize_opponents(config.opponents)

    with (
        _temporary_game_config(config),
        _temporary_agent_state(agent),
        _temporary_eval_mode(agent, config.eval_epsilon),
    ):
        for opponent_index, opponent_name in enumerate(opponents):
            for episode_index in range(config.episodes):
                episode_seed = derive_episode_seed(
                    config.seed, opponent_name, opponent_index, episode_index
                )
                _seed_everything(episode_seed)

                opponent = opponent_factory(
                    opponent_name,
                    name=f"EvaluationOpponent_{opponent_name}_{episode_index}",
                )
                manager = _create_training_manager(training_manager_factory, config)
                try:
                    stats = manager.train_episode(agent, opponent, max_steps=config.max_steps)
                finally:
                    cleanup = getattr(manager, "cleanup", None)
                    if callable(cleanup):
                        cleanup()

                all_episodes.append(
                    _episode_from_stats(
                        stats,
                        opponent=opponent_name,
                        episode_index=episode_index,
                        seed=episode_seed,
                    )
                )

    return EvaluationResult(
        config=config,
        opponents=_aggregate_by_opponent(opponents, all_episodes),
        episodes=tuple(all_episodes) if config.include_episodes else (),
    )


def evaluate_checkpoint(path: str | Path, config: EvaluationConfig) -> EvaluationResult:
    """Load a strict current DQN checkpoint and evaluate it."""
    from magic_pong.ai.models.dqn_ai import DQNAgent

    checkpoint_path = Path(path)
    agent = DQNAgent(name="DQN Evaluation Agent")
    agent.load_model(str(checkpoint_path))

    result = evaluate_agent(agent, config)
    return EvaluationResult(
        config=result.config,
        opponents=result.opponents,
        episodes=result.episodes,
        checkpoint_path=str(checkpoint_path),
        checkpoint_metadata=_checkpoint_metadata(checkpoint_path),
    )


def derive_episode_seed(
    base_seed: int, opponent: str, opponent_index: int, episode_index: int
) -> int:
    """Derive a stable 32-bit seed for one opponent episode."""
    payload = f"{base_seed}:{opponent_index}:{opponent}:{episode_index}".encode()
    return int.from_bytes(hashlib.sha256(payload).digest()[:8], "big") % (2**32)


def _normalize_opponents(opponents: tuple[str, ...] | str) -> tuple[str, ...]:
    if isinstance(opponents, str):
        return (opponents,)
    return tuple(opponents)


def result_to_dict(
    result: EvaluationResult, *, include_episodes: bool | None = None
) -> dict[str, Any]:
    """Convert an evaluation result to a nested JSON-compatible dictionary."""
    include_episode_rows = (
        result.config.include_episodes if include_episodes is None else include_episodes
    )
    data: dict[str, Any] = {
        "schema_version": result.schema_version,
        "config": _json_safe(asdict(result.config)),
        "checkpoint_path": result.checkpoint_path,
        "checkpoint_metadata": _json_safe(result.checkpoint_metadata),
        "opponents": [asdict(opponent) for opponent in result.opponents],
    }
    if include_episode_rows:
        data["episodes"] = [asdict(episode) for episode in result.episodes]
    return data


def result_to_json(
    result: EvaluationResult,
    output_path: str | Path | None = None,
    *,
    include_episodes: bool | None = None,
) -> str:
    """Serialize an evaluation result as nested JSON and optionally write it."""
    payload = json.dumps(result_to_dict(result, include_episodes=include_episodes), indent=2)
    if output_path is not None:
        Path(output_path).write_text(payload + "\n", encoding="utf-8")
    return payload


def result_to_csv(
    result: EvaluationResult,
    output_path: str | Path | None = None,
    *,
    include_episodes: bool | None = None,
) -> str:
    """Serialize an evaluation result as scalar CSV rows and optionally write it."""
    rows = _csv_rows(result, include_episodes=include_episodes)
    stream = io.StringIO(newline="")
    writer = csv.DictWriter(stream, fieldnames=_csv_fieldnames())
    writer.writeheader()
    writer.writerows(rows)
    payload = stream.getvalue()
    if output_path is not None:
        Path(output_path).write_text(payload, encoding="utf-8")
    return payload


def main(argv: list[str] | None = None) -> int:
    """Command-line entry point for checkpoint evaluation."""
    parser = _build_parser()
    args = parser.parse_args(argv)
    config = EvaluationConfig(
        opponents=tuple(args.opponents or DEFAULT_OPPONENTS),
        episodes=args.episodes,
        max_steps=args.max_steps,
        seed=args.seed,
        eval_epsilon=args.eval_epsilon,
        max_score=args.max_score,
        bonuses_enabled=args.bonuses_enabled,
        ball_direction=args.ball_direction,
        ball_angle_deg=args.ball_angle_deg,
        include_episodes=args.include_episodes,
    )

    result = evaluate_checkpoint(args.checkpoint, config)
    if args.format == "json":
        payload = result_to_json(result, args.output)
    else:
        payload = result_to_csv(result, args.output)

    if args.output is None:
        print(payload)
    return 0


def _build_parser() -> argparse.ArgumentParser:
    parser = argparse.ArgumentParser(description="Evaluate a Magic Pong DQN checkpoint")
    parser.add_argument("--checkpoint", required=True, help="Path to a strict DQN checkpoint")
    parser.add_argument(
        "--opponent",
        dest="opponents",
        action="append",
        help="Opponent AI type. May be repeated.",
    )
    parser.add_argument("--episodes", type=int, default=10)
    parser.add_argument("--max-steps", type=int, default=1000)
    parser.add_argument("--seed", type=int, default=0)
    parser.add_argument("--eval-epsilon", type=float, default=0.0)
    parser.add_argument("--max-score", type=int)

    bonus_group = parser.add_mutually_exclusive_group()
    bonus_group.add_argument("--bonuses", dest="bonuses_enabled", action="store_true", default=None)
    bonus_group.add_argument("--no-bonuses", dest="bonuses_enabled", action="store_false")

    parser.add_argument("--ball-direction", type=int, choices=(-1, 0, 1), default=0)
    parser.add_argument("--ball-angle-deg", type=float)
    parser.add_argument("--format", choices=("json", "csv"), default="json")
    parser.add_argument("--output", help="Optional output file path")
    parser.add_argument("--include-episodes", action="store_true")
    return parser


def _create_training_manager(
    training_manager_factory: TrainingManagerFactory, config: EvaluationConfig
) -> Any:
    return training_manager_factory(
        headless=True,
        initial_ball_direction=config.ball_direction,
        initial_ball_angle=(
            math.radians(config.ball_angle_deg) if config.ball_angle_deg is not None else None
        ),
    )


def _seed_everything(seed: int) -> None:
    random.seed(seed)
    np.random.seed(seed)

    try:
        import torch
    except ImportError:
        return

    torch.manual_seed(seed)
    if torch.cuda.is_available():
        torch.cuda.manual_seed_all(seed)


@contextmanager
def _temporary_eval_mode(agent: Any, eval_epsilon: float) -> Any:
    missing = object()
    original_training_enabled = getattr(agent, "training_enabled", missing)
    original_exploration_enabled = getattr(agent, "exploration_enabled", missing)
    original_epsilon = getattr(agent, "epsilon", missing)

    try:
        set_training_mode = getattr(agent, "set_training_mode", None)
        if callable(set_training_mode):
            set_training_mode(False)
        elif original_training_enabled is not missing:
            agent.training_enabled = False

        if original_epsilon is not missing:
            agent.epsilon = eval_epsilon

        set_exploration_mode = getattr(agent, "set_exploration_mode", None)
        if callable(set_exploration_mode):
            set_exploration_mode(eval_epsilon > 0.0)
        elif original_exploration_enabled is not missing:
            agent.exploration_enabled = eval_epsilon > 0.0

        yield
    finally:
        set_training_mode = getattr(agent, "set_training_mode", None)
        if original_training_enabled is not missing:
            if callable(set_training_mode):
                set_training_mode(bool(original_training_enabled))
            else:
                agent.training_enabled = original_training_enabled

        set_exploration_mode = getattr(agent, "set_exploration_mode", None)
        if original_exploration_enabled is not missing:
            if callable(set_exploration_mode):
                set_exploration_mode(bool(original_exploration_enabled))
            else:
                agent.exploration_enabled = original_exploration_enabled

        if original_epsilon is not missing:
            agent.epsilon = original_epsilon


@contextmanager
def _temporary_agent_state(agent: Any) -> Any:
    attr_snapshots = {
        attr_name: _copy_agent_attr(getattr(agent, attr_name))
        for attr_name in (
            "current_episode_reward",
            "episode_rewards",
            "episode_buffer",
            "last_state",
            "last_action",
            "training_step",
            "step_count",
            "loss_history",
            "reward_history",
        )
        if hasattr(agent, attr_name)
    }
    replay_snapshot = _snapshot_replay_memory(agent)

    try:
        yield
    finally:
        for attr_name, value in attr_snapshots.items():
            _restore_agent_attr(agent, attr_name, value)
        _restore_replay_memory(agent, replay_snapshot)


@contextmanager
def _temporary_game_config(config: EvaluationConfig) -> Any:
    touched: dict[str, Any] = {}
    try:
        if config.max_score is not None and hasattr(game_config, "MAX_SCORE"):
            touched["MAX_SCORE"] = game_config.MAX_SCORE
            game_config.MAX_SCORE = config.max_score
        if config.bonuses_enabled is not None and hasattr(game_config, "BONUSES_ENABLED"):
            touched["BONUSES_ENABLED"] = game_config.BONUSES_ENABLED
            game_config.BONUSES_ENABLED = config.bonuses_enabled
        yield
    finally:
        for name, value in touched.items():
            setattr(game_config, name, value)


def _copy_agent_attr(value: Any) -> Any:
    if isinstance(value, np.ndarray):
        return value.copy()
    if isinstance(value, list):
        return list(value)
    if isinstance(value, tuple):
        return tuple(value)
    return value


def _restore_agent_attr(agent: Any, attr_name: str, value: Any) -> None:
    current_value = getattr(agent, attr_name, None)
    if isinstance(current_value, list) and isinstance(value, list):
        current_value[:] = value
    else:
        setattr(agent, attr_name, _copy_agent_attr(value))


def _snapshot_replay_memory(agent: Any) -> dict[str, Any]:
    memory = getattr(agent, "memory", None)
    if memory is None:
        return {}

    snapshot: dict[str, Any] = {"memory": memory}
    buffer = getattr(memory, "buffer", None)
    if buffer is not None and hasattr(buffer, "__iter__"):
        snapshot["buffer"] = list(buffer)

    priorities = getattr(memory, "priorities", None)
    if priorities is not None and hasattr(priorities, "__iter__"):
        snapshot["priorities"] = list(priorities)

    return snapshot


def _restore_replay_memory(agent: Any, snapshot: dict[str, Any]) -> None:
    memory = snapshot.get("memory")
    if memory is None or getattr(agent, "memory", None) is not memory:
        return

    if "buffer" in snapshot:
        _replace_mutable_sequence(getattr(memory, "buffer", None), snapshot["buffer"])
    if "priorities" in snapshot:
        _replace_mutable_sequence(getattr(memory, "priorities", None), snapshot["priorities"])


def _replace_mutable_sequence(target: Any, values: list[Any]) -> None:
    clear = getattr(target, "clear", None)
    extend = getattr(target, "extend", None)
    if callable(clear) and callable(extend):
        clear()
        extend(values)


def _episode_from_stats(
    stats: Mapping[str, Any], *, opponent: str, episode_index: int, seed: int
) -> EpisodeEvaluation:
    winner = int(stats.get("winner", 0))
    goals_for = int(stats.get("goals_for", _count_goals(stats.get("events", []), player=1)))
    goals_against = int(stats.get("goals_against", _count_goals(stats.get("events", []), player=2)))
    timeout = winner == 0
    return EpisodeEvaluation(
        opponent=opponent,
        episode_index=episode_index,
        seed=seed,
        winner=winner,
        outcome=_outcome(winner),
        agent_won=winner == 1,
        timeout=timeout,
        steps=int(stats.get("steps", 0)),
        total_reward_agent=float(
            stats.get("total_reward_agent", stats.get("total_reward_p1", 0.0))
        ),
        total_reward_opponent=float(
            stats.get("total_reward_opponent", stats.get("total_reward_p2", 0.0))
        ),
        goals_for=goals_for,
        goals_against=goals_against,
        goal_diff=goals_for - goals_against,
    )


def _count_goals(events: Any, *, player: int) -> int:
    if not isinstance(events, list | tuple):
        return 0
    return sum(
        1
        for event in events
        if isinstance(event, Mapping) and int(event.get("player", 0)) == player
    )


def _outcome(winner: int) -> str:
    if winner == 1:
        return "win"
    if winner == 2:
        return "loss"
    return "timeout"


def _aggregate_by_opponent(
    opponents: tuple[str, ...], episodes: list[EpisodeEvaluation]
) -> tuple[OpponentEvaluation, ...]:
    aggregates: list[OpponentEvaluation] = []
    for opponent in opponents:
        opponent_episodes = [episode for episode in episodes if episode.opponent == opponent]
        rewards = [episode.total_reward_agent for episode in opponent_episodes]
        steps = [episode.steps for episode in opponent_episodes]
        goal_diffs = [episode.goal_diff for episode in opponent_episodes]
        episode_count = len(opponent_episodes)
        wins = sum(1 for episode in opponent_episodes if episode.winner == 1)
        losses = sum(1 for episode in opponent_episodes if episode.winner == 2)
        timeouts = sum(1 for episode in opponent_episodes if episode.timeout)

        aggregates.append(
            OpponentEvaluation(
                opponent=opponent,
                episodes=episode_count,
                wins=wins,
                losses=losses,
                draws=timeouts,
                timeouts=timeouts,
                win_rate=wins / episode_count if episode_count else 0.0,
                avg_reward=_mean(rewards),
                std_reward=float(statistics.pstdev(rewards)) if len(rewards) > 1 else 0.0,
                min_reward=min(rewards) if rewards else 0.0,
                max_reward=max(rewards) if rewards else 0.0,
                avg_steps=_mean(steps),
                median_steps=float(statistics.median(steps)) if steps else 0.0,
                avg_goal_diff=_mean(goal_diffs),
            )
        )
    return tuple(aggregates)


def _mean(values: list[int] | list[float]) -> float:
    return float(sum(values) / len(values)) if values else 0.0


def _checkpoint_metadata(checkpoint_path: Path) -> dict[str, Any] | None:
    from magic_pong.ai.models.dqn_checkpoint import get_dqn_checkpoint_validation_info
    from magic_pong.ai.models.dqn_checkpoint import safe_torch_load

    try:
        checkpoint = safe_torch_load(str(checkpoint_path), map_location="cpu")
    except Exception:
        return None
    metadata = _json_safe(get_dqn_checkpoint_validation_info(checkpoint).as_dict())
    return cast(dict[str, Any], metadata)


def _json_safe(value: Any) -> Any:
    if isinstance(value, Mapping):
        return {str(key): _json_safe(item) for key, item in value.items()}
    if isinstance(value, list | tuple):
        return [_json_safe(item) for item in value]
    if isinstance(value, np.generic):
        return value.item()
    if isinstance(value, Path):
        return str(value)
    return value


def _csv_fieldnames() -> list[str]:
    return [
        "schema_version",
        "checkpoint_path",
        "row_type",
        "opponent",
        "episodes",
        "wins",
        "losses",
        "draws",
        "timeouts",
        "win_rate",
        "avg_reward",
        "std_reward",
        "min_reward",
        "max_reward",
        "avg_steps",
        "median_steps",
        "avg_goal_diff",
        "episode_index",
        "seed",
        "winner",
        "outcome",
        "agent_won",
        "timeout",
        "steps",
        "total_reward_agent",
        "total_reward_opponent",
        "goals_for",
        "goals_against",
        "goal_diff",
    ]


def _csv_rows(
    result: EvaluationResult, *, include_episodes: bool | None = None
) -> list[dict[str, Any]]:
    rows: list[dict[str, Any]] = []
    base_row = {
        "schema_version": result.schema_version,
        "checkpoint_path": result.checkpoint_path or "",
    }

    for opponent in result.opponents:
        row = dict.fromkeys(_csv_fieldnames(), "")
        row.update(base_row)
        row.update(asdict(opponent))
        row["row_type"] = "opponent"
        rows.append(row)

    include_episode_rows = (
        result.config.include_episodes if include_episodes is None else include_episodes
    )
    if include_episode_rows:
        for episode in result.episodes:
            row = dict.fromkeys(_csv_fieldnames(), "")
            row.update(base_row)
            row.update(asdict(episode))
            row["row_type"] = "episode"
            rows.append(row)

    return rows
