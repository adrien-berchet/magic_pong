"""Checks for the statistics used by the read-only learner investigation."""

import numpy as np
import pytest

from docs.ai.run_value_diagnostics import aliasing_probe
from docs.ai.run_value_diagnostics import batch_event_probability
from docs.ai.run_value_diagnostics import discount_returns
from docs.ai.run_value_diagnostics import replay_episode_labels
from docs.ai.run_value_diagnostics import summarize_matches


def test_discount_returns_keeps_signed_goals_and_frame_units():
    assert discount_returns([0.1, 0.0, -1.0], 0.5) == pytest.approx([-0.15, -0.5, -1])
    assert discount_returns([], 0.99).size == 0


def test_uniform_batch_probability_is_without_replacement():
    assert batch_event_probability(4, 1, 2) == pytest.approx(0.5)
    assert batch_event_probability(4, 0, 2) == 0
    assert batch_event_probability(4, 3, 2) == 1
    with pytest.raises(ValueError):
        batch_event_probability(4, 5, 2)


def test_replay_labels_include_partial_oldest_episode_and_stratum_tail():
    episodes = [
        {"episode": 1, "end_frames": 4, "frames": 4, "side": 1, "bonuses": False},
        {"episode": 2, "end_frames": 10, "frames": 6, "side": 2, "bonuses": True},
        {"episode": 3, "end_frames": 12, "frames": 2, "side": 1, "bonuses": True},
    ]
    labels = replay_episode_labels(episodes, 12, 9)
    assert [row["episode"] for row in labels] == [1, 2, 2, 2, 2, 2, 2, 3, 3]
    assert labels[0]["side"] == 1
    assert labels[-1]["bonuses"]
    episodes[1]["frames"] = 5
    with pytest.raises(ValueError, match="contiguous"):
        replay_episode_labels(episodes, 12, 9)


def test_match_summary_counts_timeouts_and_reports_point_margin():
    rows = [
        {"agent_won": True, "timeout": False, "goals_for": 11, "goals_against": 4, "steps": 100},
        {"agent_won": False, "timeout": True, "goals_for": 3, "goals_against": 5, "steps": 200},
    ]
    result = summarize_matches(rows)
    assert result["wins"] == 1
    assert result["timeouts"] == 1
    assert result["goal_diff_mean"] == pytest.approx(2.5)
    assert result["point_win_fraction"] == pytest.approx(14 / 23)
    assert np.isfinite(result["goals_for_per_1000_frames"])


def test_same_encoded_goal_has_different_terminal_rules():
    report = aliasing_probe()
    assert report["before_vectors_equal"]
    assert report["after_vectors_equal"]
    assert [r["terminated"] for r in report["cases"]] == [True, False, True]
    assert [r["goal_reward"] for r in report["cases"]] == [1, 1, 1]
