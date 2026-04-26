"""
Regression tests for protocol reward calculators.
"""

import pytest

from magic_pong.ai.interfaces.reward import DenseRewardCalculator


def test_dense_reward_consumes_canonical_dict_events() -> None:
    calculator = DenseRewardCalculator()
    events = {
        "goals": [{"player": 1, "score": [1, 0]}],
        "paddle_hits": [{"player": 1}],
        "wall_bounces": ["top"],
        "bonus_collected": [{"player": 1, "type": "enlarge_paddle"}],
    }

    reward = calculator.calculate_reward(events, {}, player_id=1)

    assert reward == pytest.approx(1.25)


def test_dense_reward_tolerates_legacy_event_shapes() -> None:
    calculator = DenseRewardCalculator()
    events = {
        "goals": [{"player": 2, "score": [0, 1]}],
        "paddle_hits": [1],
        "bonus_collected": [(1, "enlarge_paddle")],
    }

    reward = calculator.calculate_reward(events, {}, player_id=1)

    assert reward == pytest.approx(-0.8)
