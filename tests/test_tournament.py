"""Tests for the Elo tournament module."""

from __future__ import annotations

from typing import Any

import pytest

from magic_pong.ai import tournament
from magic_pong.ai.tournament import MatchResult
from magic_pong.ai.tournament import Participant
from magic_pong.ai.tournament import compute_elo
from magic_pong.ai.tournament import elo_update
from magic_pong.ai.tournament import format_leaderboard
from magic_pong.ai.tournament import run_round_robin
from magic_pong.ai.tournament import run_tournament


def test_participant_builtin_requires_builtin_type() -> None:
    with pytest.raises(ValueError, match="builtin_type"):
        Participant(id="x", kind="builtin")


def test_participant_dqn_requires_checkpoint_path() -> None:
    with pytest.raises(ValueError, match="checkpoint_path"):
        Participant(id="x", kind="dqn")


def test_elo_update_symmetric_draw_leaves_ratings_unchanged() -> None:
    new_a, new_b = elo_update(1200.0, 1200.0, actual_score_a=0.5)
    assert new_a == pytest.approx(1200.0)
    assert new_b == pytest.approx(1200.0)


def test_elo_update_conserves_total_rating() -> None:
    new_a, new_b = elo_update(1000.0, 1400.0, actual_score_a=1.0, k=32.0)
    assert (new_a + new_b) == pytest.approx(1000.0 + 1400.0)


def test_elo_update_upset_gains_more_than_expected_win() -> None:
    # Lower-rated A beating higher-rated B should gain more than the reverse.
    upset_a, _ = elo_update(1000.0, 1400.0, actual_score_a=1.0, k=32.0)
    expected_winner_a, _ = elo_update(1400.0, 1000.0, actual_score_a=1.0, k=32.0)

    upset_gain = upset_a - 1000.0
    expected_gain = expected_winner_a - 1400.0
    assert upset_gain > expected_gain > 0


def test_elo_update_k_factor_scales_delta_linearly() -> None:
    a_k16, _ = elo_update(1000.0, 1200.0, actual_score_a=1.0, k=16.0)
    a_k32, _ = elo_update(1000.0, 1200.0, actual_score_a=1.0, k=32.0)
    assert (a_k32 - 1000.0) == pytest.approx(2 * (a_k16 - 1000.0))


def test_compute_elo_batches_one_update_per_match() -> None:
    participants = [
        Participant(id="a", kind="builtin", builtin_type="random"),
        Participant(id="b", kind="builtin", builtin_type="random"),
    ]
    matches = [
        MatchResult(participant_a="a", participant_b="b", episodes=10, wins_a=8, wins_b=2, draws=0)
    ]

    ratings = compute_elo(matches, participants)

    assert ratings["a"].rating > ratings["b"].rating
    assert ratings["a"].matches_played == 1
    assert ratings["b"].matches_played == 1


def test_run_round_robin_has_no_self_or_duplicate_pairings(monkeypatch: pytest.MonkeyPatch) -> None:
    participants = [
        Participant(id=f"p{i}", kind="builtin", builtin_type="random") for i in range(4)
    ]
    invoked_pairs: list[tuple[str, str]] = []

    def fake_build_agent(participant: Participant) -> str:
        return participant.id

    def fake_play_match(
        agent_a: Any,
        participant_a_id: str,
        agent_b: Any,
        participant_b_id: str,
        *,
        episodes: int,
        max_steps: int,
        seed: int,
    ) -> MatchResult:
        invoked_pairs.append((participant_a_id, participant_b_id))
        return MatchResult(
            participant_a=participant_a_id,
            participant_b=participant_b_id,
            episodes=episodes,
            wins_a=1,
            wins_b=0,
            draws=0,
        )

    monkeypatch.setattr(tournament, "_build_agent", fake_build_agent)
    monkeypatch.setattr(tournament, "play_match", fake_play_match)

    results = run_round_robin(participants, episodes_per_match=5, max_steps=10)

    n = len(participants)
    assert len(results) == n * (n - 1) // 2
    assert len(invoked_pairs) == len(set(invoked_pairs))
    for id_a, id_b in invoked_pairs:
        assert id_a != id_b


def test_format_leaderboard_ranks_by_rating_descending() -> None:
    ratings = compute_elo(
        [
            MatchResult(
                participant_a="a", participant_b="b", episodes=10, wins_a=10, wins_b=0, draws=0
            )
        ],
        [
            Participant(id="a", kind="builtin", builtin_type="random"),
            Participant(id="b", kind="builtin", builtin_type="random"),
        ],
    )
    leaderboard = format_leaderboard(ratings)
    lines = leaderboard.splitlines()
    assert lines[1].strip().startswith("1")
    assert "a" in lines[1]
    assert "b" in lines[2]


def test_run_tournament_end_to_end_builtins_only(tmp_path: Any) -> None:
    participants = [
        Participant(id="builtin:random", kind="builtin", builtin_type="random"),
        Participant(id="builtin:dummy", kind="builtin", builtin_type="dummy"),
        Participant(id="builtin:follow_ball", kind="builtin", builtin_type="follow_ball"),
    ]

    ratings = run_tournament(
        participants,
        episodes_per_match=2,
        max_steps=50,
        output_dir=tmp_path,
    )

    assert set(ratings) == {p.id for p in participants}
    for rating in ratings.values():
        assert rating.matches_played == len(participants) - 1

    assert (tmp_path / "tournament_results.json").exists()
    assert (tmp_path / "tournament_leaderboard.txt").exists()


def test_run_tournament_with_real_dqn_checkpoint(tmp_path: Any) -> None:
    pytest.importorskip("torch")
    from magic_pong.ai.models.dqn_ai import DQNAgent

    checkpoint_path = tmp_path / "tiny_dqn.pth"
    agent = DQNAgent(hidden_size=16, layer_size=1, name="TinyDQN")
    agent.save_model(str(checkpoint_path))

    participants = [
        Participant(id="dqn:tiny", kind="dqn", checkpoint_path=str(checkpoint_path)),
        Participant(id="builtin:random", kind="builtin", builtin_type="random"),
    ]

    ratings = run_tournament(participants, episodes_per_match=2, max_steps=50)

    assert set(ratings) == {"dqn:tiny", "builtin:random"}
    assert ratings["dqn:tiny"].matches_played == 1
    assert ratings["builtin:random"].matches_played == 1
