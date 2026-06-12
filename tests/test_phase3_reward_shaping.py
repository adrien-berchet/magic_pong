"""Tests for opt-in Phase 3 reward shaping on the active GameEnvironment path."""

import pytest

from magic_pong.ai.interface import RewardCalculator
from magic_pong.utils.config import ai_config
from magic_pong.utils.config import ai_config_tmp


def player1_state(
    *,
    player_y: float,
    previous_player_y: float,
    ball_velocity: tuple[float, float] = (-300.0, 0.0),
) -> dict:
    return {
        "ball_position": (400.0, 300.0),
        "ball_velocity": ball_velocity,
        "player1_position": (20.0, player_y),
        "player2_position": (765.0, 260.0),
        "player1_last_position": (20.0, previous_player_y),
        "player2_last_position": (765.0, 260.0),
        "player1_paddle_size": 80.0,
        "player2_paddle_size": 80.0,
        "active_bonuses": [],
        "rotating_paddles": [],
        "score": [0, 0],
        "time_elapsed": 0.0,
        "field_bounds": (0.0, 800.0, 0.0, 600.0),
    }


def test_phase3_reward_shaping_disabled_preserves_legacy_reward_values() -> None:
    calculator = RewardCalculator()
    state = player1_state(player_y=260.0, previous_player_y=310.0)
    events = {"paddle_hits": [{"player": 1}]}

    with ai_config_tmp(REWARD_SHAPING_MODE="legacy", USE_PROXIMITY_REWARD=False):
        reward = calculator.calculate_reward(state, events, player_id=1)

    assert reward == pytest.approx(ai_config.PADDLE_HIT_REWARD)


def test_phase3_reward_improves_when_paddle_moves_toward_predicted_intercept() -> None:
    close_to_intercept = player1_state(player_y=260.0, previous_player_y=310.0)
    moving_away_far = player1_state(player_y=360.0, previous_player_y=310.0)

    with ai_config_tmp(
        REWARD_SHAPING_MODE="phase3",
        PHASE3_INTERCEPT_PROGRESS_REWARD=0.02,
        PHASE3_INTERCEPT_DISTANCE_PENALTY=0.02,
    ):
        closer_reward = RewardCalculator().calculate_reward(close_to_intercept, {}, player_id=1)
        far_reward = RewardCalculator().calculate_reward(moving_away_far, {}, player_id=1)

    assert closer_reward > far_reward
    assert closer_reward > 0
    assert far_reward < 0


def test_phase3_successful_return_reward_requires_ball_moving_to_opponent_side() -> None:
    events = {"paddle_hits": [{"player": 1}]}
    successful_return = player1_state(
        player_y=260.0, previous_player_y=260.0, ball_velocity=(300.0, 0.0)
    )
    failed_return = player1_state(
        player_y=260.0, previous_player_y=260.0, ball_velocity=(-300.0, 0.0)
    )

    with ai_config_tmp(REWARD_SHAPING_MODE="phase3", PHASE3_SUCCESSFUL_RETURN_REWARD=0.05):
        successful_reward = RewardCalculator().calculate_reward(
            successful_return, events, player_id=1
        )
        failed_reward = RewardCalculator().calculate_reward(failed_return, events, player_id=1)

    assert successful_reward == pytest.approx(ai_config.PADDLE_HIT_REWARD + 0.05)
    assert successful_reward > failed_reward


def test_phase3_reward_shaping_preserves_terminal_goal_reward() -> None:
    calculator = RewardCalculator()
    state = player1_state(player_y=360.0, previous_player_y=310.0)
    events = {"goals": [{"player": 1, "score": [1, 0]}]}

    with ai_config_tmp(REWARD_SHAPING_MODE="phase3"):
        reward = calculator.calculate_reward(state, events, player_id=1)

    assert reward == pytest.approx(ai_config.SCORE_REWARD)


def test_phase3_terminal_goal_reward_ignores_legacy_proximity_shaping() -> None:
    calculator = RewardCalculator()
    state = player1_state(player_y=360.0, previous_player_y=310.0)
    events = {"goals": [{"player": 1, "score": [1, 0]}]}

    with ai_config_tmp(
        REWARD_SHAPING_MODE="phase3",
        USE_PROXIMITY_REWARD=True,
        PROXIMITY_REWARD_FACTOR=0.5,
        PROXIMITY_PENALTY_FACTOR=0.5,
    ):
        calculator.calculate_reward(state, {}, player_id=1)
        reward = calculator.calculate_reward(state, events, player_id=1)

    assert reward == pytest.approx(ai_config.SCORE_REWARD)


def test_phase3_reward_shaping_clears_stale_optimal_point_when_ball_not_approaching() -> None:
    calculator = RewardCalculator()

    with ai_config_tmp(REWARD_SHAPING_MODE="phase3"):
        calculator.calculate_reward(
            player1_state(player_y=260.0, previous_player_y=310.0),
            {},
            player_id=1,
        )
        assert calculator.get_optimal_points()[1] is not None

        calculator.calculate_reward(
            player1_state(
                player_y=260.0,
                previous_player_y=260.0,
                ball_velocity=(300.0, 0.0),
            ),
            {},
            player_id=1,
        )

    assert calculator.get_optimal_points() == {}
