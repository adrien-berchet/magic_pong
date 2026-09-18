"""Fixed full-match benchmarks for checkpoint selection and held-out reporting.

A seed is one match *per opponent, player side and bonus mode*. Time limits are
reported as non-wins. Defaults are a screening budget, not evidence of 99% play.
"""

from __future__ import annotations

import argparse
import copy
import json
import math
from collections.abc import Callable
from collections.abc import Iterator
from collections.abc import Mapping
from contextlib import contextmanager
from dataclasses import asdict
from dataclasses import dataclass
from dataclasses import field
from dataclasses import replace
from pathlib import Path
from typing import Any
from typing import Literal

from magic_pong.ai.evaluation import _count_goals
from magic_pong.ai.evaluation import _seed_everything
from magic_pong.ai.evaluation import _temporary_agent_state
from magic_pong.ai.evaluation import _temporary_eval_mode
from magic_pong.ai.evaluation import preserve_global_rng_state
from magic_pong.ai.models.simple_ai import BUILTIN_OPPONENTS
from magic_pong.ai.models.simple_ai import create_ai
from magic_pong.core.game_engine import TrainingManager
from magic_pong.utils.config import AIConfig
from magic_pong.utils.config import GameConfig
from magic_pong.utils.config import _restore_config
from magic_pong.utils.config import ai_config
from magic_pong.utils.config import game_config

SCHEMA_VERSION = "magic_pong.competitive.v1"
Split = Literal["validation", "held_out"]


@dataclass(frozen=True)
class CompetitiveConfig:
    """Explicit benchmark protocol; seed counts are matches per matchup cell.

    Validation uses standard gameplay defaults, independent of training's
    randomized field or curriculum. ``game_overrides`` declares custom rules.
    Additional opponent names require factories returning fixed opponents;
    use checkpoint identities in those names when evaluating a neural league.
    """

    opponents: tuple[str, ...] = BUILTIN_OPPONENTS
    seeds: tuple[int, ...] = (101, 102, 103)
    held_out_seeds: tuple[int, ...] = (10001, 10002, 10003)
    player_sides: tuple[int, ...] = (1, 2)
    bonus_modes: tuple[bool, ...] = (False, True)
    max_steps: int = 18000
    max_score: int = 11
    confidence_z: float = 1.959963984540054
    game_overrides: dict[str, Any] = field(default_factory=dict)

    def __post_init__(self) -> None:
        for name in (
            "opponents",
            "seeds",
            "held_out_seeds",
            "player_sides",
            "bonus_modes",
        ):
            values = tuple(getattr(self, name))
            object.__setattr__(self, name, values)
            if not values or len(values) != len(set(values)):
                raise ValueError(f"{name} must be non-empty and contain unique values")
        if set(self.seeds) & set(self.held_out_seeds):
            raise ValueError("validation and held-out seeds must be disjoint")
        if any(
            type(seed) is not int or not 0 <= seed < 2**32
            for seed in self.seeds + self.held_out_seeds
        ):
            raise ValueError("seeds must be 32-bit non-negative integers")
        if any(side not in (1, 2) for side in self.player_sides):
            raise ValueError("player_sides must contain 1 and/or 2")
        if any(type(mode) is not bool for mode in self.bonus_modes):
            raise ValueError("bonus_modes must contain booleans")
        if self.max_steps <= 0 or self.max_score <= 0:
            raise ValueError("max_steps and max_score must be positive")
        if not math.isfinite(self.confidence_z) or self.confidence_z <= 0:
            raise ValueError("confidence_z must be finite and positive")
        unknown = set(self.game_overrides) - set(GameConfig.model_fields)
        if unknown:
            raise ValueError(f"unknown game_overrides: {sorted(unknown)}")
        if {"MAX_SCORE", "BONUSES_ENABLED"} & self.game_overrides.keys():
            raise ValueError("use max_score and bonus_modes instead of game_overrides")
        object.__setattr__(self, "game_overrides", copy.deepcopy(self.game_overrides))
        GameConfig(**self.game_overrides)

    def match_seeds(self, split: Split) -> tuple[int, ...]:
        if split == "validation":
            return self.seeds
        if split == "held_out":
            return self.held_out_seeds
        raise ValueError("split must be 'validation' or 'held_out'")


@dataclass(frozen=True)
class CompetitiveMatch:
    opponent: str
    player_side: int
    bonuses_enabled: bool
    seed: int
    winner: int
    agent_won: bool
    timeout: bool
    steps: int
    goals_for: int
    goals_against: int


@dataclass(frozen=True)
class CompetitiveMatchup:
    opponent: str
    player_side: int
    bonuses_enabled: bool
    matches: int
    wins: int
    losses: int
    timeouts: int
    win_rate: float
    wilson_lower_bound: float
    avg_goal_diff: float


@dataclass(frozen=True)
class CompetitiveResult:
    config: CompetitiveConfig
    split: Split
    matchups: tuple[CompetitiveMatchup, ...]
    matches: tuple[CompetitiveMatch, ...]
    resolved_game_config: dict[str, Any]
    resolved_ai_config: dict[str, Any]
    checkpoint_path: str | None = None
    schema_version: str = SCHEMA_VERSION


def wilson_lower_bound(wins: int, matches: int, z: float = 1.959963984540054) -> float:
    """Lower end of the two-sided 95% Wilson interval by default."""
    if matches <= 0:
        return 0.0
    if not 0 <= wins <= matches:
        raise ValueError("wins must be between zero and matches")
    rate = wins / matches
    z2 = z * z
    return max(
        0.0,
        (
            rate
            + z2 / (2 * matches)
            - z * math.sqrt(rate * (1 - rate) / matches + z2 / (4 * matches * matches))
        )
        / (1 + z2 / matches),
    )


@contextmanager
def _standard_configuration(
    config: CompetitiveConfig,
) -> Iterator[tuple[dict[str, Any], dict[str, Any]]]:
    original_game = game_config.model_copy(deep=True)
    original_ai = ai_config.model_copy(deep=True)
    resolved_game = GameConfig(**config.game_overrides, MAX_SCORE=config.max_score)
    resolved_ai = AIConfig(HEADLESS_MODE=True, FAST_MODE_MULTIPLIER=1.0)
    try:
        _restore_config(game_config, resolved_game)
        _restore_config(ai_config, resolved_ai)
        yield resolved_game.model_dump(mode="json"), resolved_ai.model_dump(mode="json")
    finally:
        _restore_config(game_config, original_game)
        _restore_config(ai_config, original_ai)


def evaluate_competitive_agent(
    agent: Any,
    config: CompetitiveConfig,
    *,
    split: Split = "validation",
    opponent_factories: Mapping[str, Callable[[], Any]] | None = None,
    training_manager_factory: Callable[..., Any] = TrainingManager,
) -> CompetitiveResult:
    """Evaluate fixed opponents with no exploration, preserving training state.

    A factory may return an existing frozen neural agent: both players' runtime
    state and module modes are restored after every match. Seed reset occurs
    after construction, so factories' RNG consumption cannot alter a serve.
    """
    seeds = config.match_seeds(split)
    factories = dict(opponent_factories or {})
    unknown = set(config.opponents) - set(BUILTIN_OPPONENTS) - factories.keys()
    if unknown:
        raise ValueError(f"missing opponent factories: {sorted(unknown)}")
    matches = []
    matchups = []
    with preserve_global_rng_state(), _standard_configuration(config) as resolved:
        for opponent_name in config.opponents:
            for side in config.player_sides:
                for bonuses_enabled in config.bonus_modes:
                    rows = []
                    for seed in seeds:
                        # Restore complete rules even if an opponent changes globals.
                        _restore_config(game_config, GameConfig(**resolved[0]))
                        _restore_config(ai_config, AIConfig(**resolved[1]))
                        game_config.BONUSES_ENABLED = bonuses_enabled
                        _seed_everything(seed)
                        factory = factories.get(opponent_name)
                        opponent = factory() if factory else create_ai(opponent_name)
                        if opponent is agent:
                            raise ValueError("candidate and opponent must be distinct objects")
                        with (
                            _temporary_agent_state(agent),
                            _temporary_agent_state(opponent),
                            _temporary_eval_mode(agent, 0.0),
                            _temporary_eval_mode(opponent, 0.0),
                        ):
                            manager = training_manager_factory(headless=True)
                            try:
                                _seed_everything(seed)
                                players = (agent, opponent) if side == 1 else (opponent, agent)
                                stats = manager.train_episode(*players, max_steps=config.max_steps)
                            finally:
                                cleanup = getattr(manager, "cleanup", None)
                                if callable(cleanup):
                                    cleanup()
                        winner = int(stats.get("winner", 0))
                        # Manager goal summaries, when supplied, refer to player 1.
                        p1_goals = int(
                            stats.get(
                                "goals_for",
                                _count_goals(stats.get("events", []), player=1),
                            )
                        )
                        p2_goals = int(
                            stats.get(
                                "goals_against",
                                _count_goals(stats.get("events", []), player=2),
                            )
                        )
                        row = CompetitiveMatch(
                            opponent=opponent_name,
                            player_side=side,
                            bonuses_enabled=bonuses_enabled,
                            seed=seed,
                            winner=winner,
                            agent_won=winner == side,
                            timeout=winner == 0,
                            steps=int(stats.get("steps", 0)),
                            goals_for=p1_goals if side == 1 else p2_goals,
                            goals_against=p2_goals if side == 1 else p1_goals,
                        )
                        rows.append(row)
                    wins = sum(row.agent_won for row in rows)
                    timeouts = sum(row.timeout for row in rows)
                    matchups.append(
                        CompetitiveMatchup(
                            opponent=opponent_name,
                            player_side=side,
                            bonuses_enabled=bonuses_enabled,
                            matches=len(rows),
                            wins=wins,
                            losses=len(rows) - wins - timeouts,
                            timeouts=timeouts,
                            win_rate=wins / len(rows),
                            wilson_lower_bound=wilson_lower_bound(
                                wins, len(rows), config.confidence_z
                            ),
                            avg_goal_diff=sum(row.goals_for - row.goals_against for row in rows)
                            / len(rows),
                        )
                    )
                    matches.extend(rows)
    return CompetitiveResult(
        config=config,
        split=split,
        matchups=tuple(matchups),
        matches=tuple(matches),
        resolved_game_config=resolved[0],
        resolved_ai_config=resolved[1],
    )


def evaluate_competitive_checkpoint(
    path: str | Path, config: CompetitiveConfig, **kwargs: Any
) -> CompetitiveResult:
    """Load a DQN checkpoint without disturbing training random streams."""
    from magic_pong.ai.models.dqn_ai import DQNAgent

    with preserve_global_rng_state():
        agent = DQNAgent(name="Competitive evaluation candidate")
        agent.load_model(str(path))
        result = evaluate_competitive_agent(agent, config, **kwargs)
    return replace(result, checkpoint_path=str(path))


def competitive_rank(result: CompetitiveResult) -> tuple[float, float, float, float]:
    """Rank weakest matchup first; average metrics only break ties."""
    cells = result.matchups
    if not cells or any(cell.matches <= 0 for cell in cells):
        raise ValueError("competitive ranking requires a non-empty evaluated protocol")
    return (
        min(cell.wilson_lower_bound for cell in cells),
        sum(cell.wilson_lower_bound for cell in cells) / len(cells),
        sum(cell.avg_goal_diff for cell in cells) / len(cells),
        sum(cell.win_rate for cell in cells) / len(cells),
    )


def is_better(candidate: CompetitiveResult, incumbent: CompetitiveResult | None) -> bool:
    """Compare like protocols; reevaluate incumbents when the league changes."""
    if incumbent is None:
        return True
    if (
        candidate.config != incumbent.config
        or candidate.split != incumbent.split
        or candidate.resolved_game_config != incumbent.resolved_game_config
        or candidate.resolved_ai_config != incumbent.resolved_ai_config
        or candidate.schema_version != incumbent.schema_version
    ):
        raise ValueError("checkpoint comparison requires an identical evaluation protocol")
    return competitive_rank(candidate) > competitive_rank(incumbent)


def competitive_result_to_dict(result: CompetitiveResult) -> dict[str, Any]:
    payload = asdict(result)
    payload["rank"] = list(competitive_rank(result))
    payload["matches_per_matchup"] = len(result.config.match_seeds(result.split))
    payload["total_matches"] = len(result.matches)
    payload["timing"] = {
        "physics_dt_seconds": result.resolved_game_config["GAME_SPEED_MULTIPLIER"]
        / result.resolved_game_config["FPS"],
        "headless_changes_physics": False,
    }
    return payload


def main(argv: list[str] | None = None) -> int:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("checkpoint")
    parser.add_argument("--output", type=Path, required=True)
    parser.add_argument("--split", choices=("validation", "held_out"), default="held_out")
    parser.add_argument("--opponents", nargs="+", default=list(BUILTIN_OPPONENTS))
    parser.add_argument(
        "--seeds",
        type=int,
        nargs="+",
        default=[101, 102, 103],
        help="Validation seeds, each used once per opponent/side/bonus cell",
    )
    parser.add_argument("--held-out-seeds", type=int, nargs="+", default=[10001, 10002, 10003])
    parser.add_argument("--max-steps", type=int, default=18000)
    parser.add_argument("--max-score", type=int, default=11)
    args = parser.parse_args(argv)
    config = CompetitiveConfig(
        opponents=tuple(args.opponents),
        seeds=tuple(args.seeds),
        held_out_seeds=tuple(args.held_out_seeds),
        max_steps=args.max_steps,
        max_score=args.max_score,
    )
    result = evaluate_competitive_checkpoint(args.checkpoint, config, split=args.split)
    args.output.parent.mkdir(parents=True, exist_ok=True)
    args.output.write_text(
        json.dumps(competitive_result_to_dict(result), indent=2) + "\n",
        encoding="utf-8",
    )
    print(f"Saved {len(result.matches)} matches to {args.output}; rank={competitive_rank(result)}")
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
