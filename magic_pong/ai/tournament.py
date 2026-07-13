"""Round-robin tournament runner and Elo rating engine for Magic Pong agents.

Used to rank trained DQN checkpoints (and builtin rule-based AIs) by genuine
relative strength rather than by win rate against a single fixed opponent.
"""

from __future__ import annotations

import json
import zlib
from dataclasses import asdict
from dataclasses import dataclass
from pathlib import Path
from typing import Any
from typing import Literal

from magic_pong.ai.evaluation import EvaluationConfig
from magic_pong.ai.evaluation import evaluate_agent
from magic_pong.ai.models.simple_ai import create_ai

BUILTIN_AI_TYPES = ("random", "follow_ball", "defensive", "aggressive", "predictive")

DEFAULT_K_FACTOR = 32.0
DEFAULT_INITIAL_RATING = 1200.0
DEFAULT_EPISODES_PER_MATCH = 20
DEFAULT_MAX_STEPS = 1000


@dataclass(frozen=True)
class Participant:
    """One tournament entrant: either a builtin rule-based AI or a DQN checkpoint."""

    id: str
    kind: Literal["builtin", "dqn"]
    builtin_type: str | None = None
    checkpoint_path: str | None = None

    def __post_init__(self) -> None:
        if self.kind == "builtin" and not self.builtin_type:
            raise ValueError(f"Participant {self.id!r} has kind='builtin' but no builtin_type")
        if self.kind == "dqn" and not self.checkpoint_path:
            raise ValueError(f"Participant {self.id!r} has kind='dqn' but no checkpoint_path")


@dataclass(frozen=True)
class MatchResult:
    """Outcome of one round-robin pairing."""

    participant_a: str
    participant_b: str
    episodes: int
    wins_a: int
    wins_b: int
    draws: int


@dataclass
class EloRating:
    """Elo rating for one participant after a tournament."""

    participant_id: str
    rating: float
    matches_played: int


def _build_agent(participant: Participant) -> Any:
    """Construct a ready-to-play, non-learning agent for a participant."""
    if participant.kind == "builtin":
        assert participant.builtin_type is not None
        return create_ai(participant.builtin_type, name=participant.id)

    from magic_pong.ai.models.dqn_ai import DQNAgent

    assert participant.checkpoint_path is not None
    agent = DQNAgent(name=participant.id)
    agent.load_model(participant.checkpoint_path)
    agent.set_training_mode(False)
    return agent


def _pair_seed(base_seed: int, id_a: str, id_b: str) -> int:
    """Derive a stable seed for one match pairing."""
    payload = f"{base_seed}:{id_a}:{id_b}".encode()
    return zlib.crc32(payload)


def play_match(
    agent_a: Any,
    participant_a_id: str,
    agent_b: Any,
    participant_b_id: str,
    *,
    episodes: int,
    max_steps: int,
    seed: int,
) -> MatchResult:
    """Play a deterministic head-to-head match between two agents.

    Reuses `evaluate_agent` (rather than driving `TrainingManager` directly) so
    seeding and game-config handling stay identical to single-checkpoint
    evaluation. `agent_b` is injected as the opponent for every episode via
    `opponent_factory`, which is the extension point `evaluate_agent` already
    exposes for exactly this purpose.
    """
    config = EvaluationConfig(
        opponents=(participant_b_id,),
        episodes=episodes,
        max_steps=max_steps,
        seed=seed,
        eval_epsilon=0.0,
    )
    result = evaluate_agent(
        agent_a,
        config,
        opponent_factory=lambda _name, **_kwargs: agent_b,
    )
    opponent_eval = result.opponents[0]
    draws = opponent_eval.draws + opponent_eval.timeouts
    return MatchResult(
        participant_a=participant_a_id,
        participant_b=participant_b_id,
        episodes=opponent_eval.episodes,
        wins_a=opponent_eval.wins,
        wins_b=opponent_eval.losses,
        draws=draws,
    )


def run_round_robin(
    participants: list[Participant],
    *,
    episodes_per_match: int = DEFAULT_EPISODES_PER_MATCH,
    max_steps: int = DEFAULT_MAX_STEPS,
    base_seed: int = 0,
) -> list[MatchResult]:
    """Play every unordered pair of participants exactly once (no self-matches)."""
    agent_cache: dict[str, Any] = {}

    def get_agent(participant: Participant) -> Any:
        cache_key = participant.checkpoint_path or participant.id
        if cache_key not in agent_cache:
            agent_cache[cache_key] = _build_agent(participant)
        return agent_cache[cache_key]

    results: list[MatchResult] = []
    for i, participant_a in enumerate(participants):
        for j, participant_b in enumerate(participants):
            if i >= j:
                continue
            seed = _pair_seed(base_seed, participant_a.id, participant_b.id)
            results.append(
                play_match(
                    get_agent(participant_a),
                    participant_a.id,
                    get_agent(participant_b),
                    participant_b.id,
                    episodes=episodes_per_match,
                    max_steps=max_steps,
                    seed=seed,
                )
            )
    return results


def elo_update(
    rating_a: float,
    rating_b: float,
    actual_score_a: float,
    k: float = DEFAULT_K_FACTOR,
) -> tuple[float, float]:
    """Return updated (rating_a, rating_b) after one match with the given outcome.

    `actual_score_a` is the fraction of the match won by A (1.0 win, 0.5 draw,
    0.0 loss). Standard logistic Elo update.
    """
    expected_a = 1.0 / (1.0 + 10.0 ** ((rating_b - rating_a) / 400.0))
    delta = k * (actual_score_a - expected_a)
    return rating_a + delta, rating_b - delta


def compute_elo(
    matches: list[MatchResult],
    participants: list[Participant],
    *,
    k_factor: float = DEFAULT_K_FACTOR,
    initial_rating: float = DEFAULT_INITIAL_RATING,
) -> dict[str, EloRating]:
    """Compute Elo ratings from a batch of round-robin match results.

    Each match contributes a single batched rating update (actual score = win
    fraction of that match) rather than one update per individual game, so the
    result doesn't depend on an arbitrary within-match game ordering.
    """
    ratings = {participant.id: initial_rating for participant in participants}
    matches_played = dict.fromkeys(ratings, 0)

    for match in matches:
        actual_score_a = (match.wins_a + 0.5 * match.draws) / match.episodes
        new_a, new_b = elo_update(
            ratings[match.participant_a],
            ratings[match.participant_b],
            actual_score_a,
            k_factor,
        )
        ratings[match.participant_a] = new_a
        ratings[match.participant_b] = new_b
        matches_played[match.participant_a] += 1
        matches_played[match.participant_b] += 1

    return {
        participant_id: EloRating(
            participant_id=participant_id,
            rating=rating,
            matches_played=matches_played[participant_id],
        )
        for participant_id, rating in ratings.items()
    }


def format_leaderboard(ratings: dict[str, EloRating]) -> str:
    """Render Elo ratings as a human-readable ranked table."""
    ordered = sorted(ratings.values(), key=lambda entry: entry.rating, reverse=True)
    lines = [f"{'Rank':<5}{'Participant':<30}{'Elo':>8}{'Matches':>10}"]
    for rank, entry in enumerate(ordered, start=1):
        lines.append(
            f"{rank:<5}{entry.participant_id:<30}{entry.rating:>8.1f}{entry.matches_played:>10}"
        )
    return "\n".join(lines)


def tournament_results_to_dict(
    matches: list[MatchResult], ratings: dict[str, EloRating]
) -> dict[str, Any]:
    """Build a JSON-serializable summary of a completed tournament."""
    return {
        "matches": [asdict(match) for match in matches],
        "ratings": [asdict(rating) for rating in ratings.values()],
    }


def run_tournament(
    participants: list[Participant],
    *,
    episodes_per_match: int = DEFAULT_EPISODES_PER_MATCH,
    max_steps: int = DEFAULT_MAX_STEPS,
    k_factor: float = DEFAULT_K_FACTOR,
    initial_rating: float = DEFAULT_INITIAL_RATING,
    base_seed: int = 0,
    output_dir: str | Path | None = None,
) -> dict[str, EloRating]:
    """Run a full round robin among `participants` and return final Elo ratings.

    Writes `tournament_results.json` and `tournament_leaderboard.txt` to
    `output_dir` when given.
    """
    matches = run_round_robin(
        participants,
        episodes_per_match=episodes_per_match,
        max_steps=max_steps,
        base_seed=base_seed,
    )
    ratings = compute_elo(matches, participants, k_factor=k_factor, initial_rating=initial_rating)

    if output_dir is not None:
        output_path = Path(output_dir)
        output_path.mkdir(parents=True, exist_ok=True)
        (output_path / "tournament_results.json").write_text(
            json.dumps(tournament_results_to_dict(matches, ratings), indent=2) + "\n",
            encoding="utf-8",
        )
        (output_path / "tournament_leaderboard.txt").write_text(
            format_leaderboard(ratings) + "\n", encoding="utf-8"
        )

    return ratings
