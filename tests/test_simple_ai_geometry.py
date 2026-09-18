"""Built-in opponents must reason about the paddle centre, not its top-left corner."""

from typing import Any

import pytest

from magic_pong.ai.models.simple_ai import BUILTIN_OPPONENTS
from magic_pong.ai.models.simple_ai import AggressiveAI
from magic_pong.ai.models.simple_ai import DefensiveAI
from magic_pong.ai.models.simple_ai import FollowBallAI
from magic_pong.ai.models.simple_ai import PredictiveAI
from magic_pong.ai.models.simple_ai import UnitObservationMixin
from magic_pong.ai.models.simple_ai import create_ai
from magic_pong.utils.config import game_config


def _observation(
    ball: tuple[float, float],
    paddle_top_left: tuple[float, float],
    *,
    velocity: tuple[float, float] = (0.0, 0.0),
    paddle_height: float = 80.0,
    normalized: bool = False,
) -> dict[str, Any]:
    scale_x = 1.0 / 800.0 if normalized else 1.0
    scale_y = 1.0 / 600.0 if normalized else 1.0
    scale_v = 1.0 / game_config.MAX_BALL_SPEED if normalized else 1.0
    return {
        "ball_pos": [ball[0] * scale_x, ball[1] * scale_y],
        "player_pos": [paddle_top_left[0] * scale_x, paddle_top_left[1] * scale_y],
        "opponent_pos": [765.0 * scale_x, 260.0 * scale_y],
        "ball_vel": [velocity[0] * scale_v, velocity[1] * scale_v],
        "player_paddle_size": paddle_height,
        "opponent_paddle_size": 80.0,
        "bonuses": [],
        "rotating_paddles": [],
        "field_width": 800.0,
        "field_height": 600.0,
        "positions_normalized": normalized,
        "max_ball_speed": game_config.MAX_BALL_SPEED,
        "time_elapsed": 0.0,
    }


@pytest.mark.parametrize("normalized", [False, True])
def test_unit_observation_reports_the_paddle_centre(normalized: bool) -> None:
    observation = _observation((400.0, 300.0), (20.0, 260.0), normalized=normalized)
    unit = UnitObservationMixin()._unit_observation(observation)
    half_width = game_config.PADDLE_WIDTH / 2
    assert unit.player_pos == pytest.approx(((20.0 + half_width) / 800.0, 300.0 / 600.0))
    assert unit.paddle_height == pytest.approx(80.0 / 600.0)


def test_unit_observation_uses_current_paddle_height_for_the_centre() -> None:
    observation = _observation((400.0, 300.0), (20.0, 200.0), paddle_height=120.0)
    unit = UnitObservationMixin()._unit_observation(observation)
    assert unit.player_pos[1] == pytest.approx(260.0 / 600.0)


def test_unit_observation_falls_back_to_configured_paddle_height() -> None:
    observation = _observation((400.0, 300.0), (20.0, 260.0))
    del observation["player_paddle_size"]
    unit = UnitObservationMixin()._unit_observation(observation)
    assert unit.player_pos[1] == pytest.approx((260.0 + game_config.PADDLE_HEIGHT / 2) / 600.0)


@pytest.mark.parametrize("ai_player", [FollowBallAI(), AggressiveAI()])
def test_chasing_ais_stop_when_the_ball_is_at_the_paddle_centre(ai_player: Any) -> None:
    centre = (20.0 + game_config.PADDLE_WIDTH / 2, 300.0)
    action = ai_player.get_action(_observation(centre, (20.0, 260.0)))
    assert action.move_x == pytest.approx(0.0)
    assert action.move_y == pytest.approx(0.0)


def test_follow_ball_moves_up_when_the_ball_is_above_the_paddle_centre() -> None:
    action = FollowBallAI().get_action(_observation((27.5, 280.0), (20.0, 260.0)))
    assert action.move_y < 0
    assert action.move_x == pytest.approx(0.0)


def test_predictive_ai_holds_position_at_a_stationary_ball_centre() -> None:
    centre = (20.0 + game_config.PADDLE_WIDTH / 2, 300.0)
    action = PredictiveAI().get_action(_observation(centre, (20.0, 260.0)))
    assert action.move_x == pytest.approx(0.0)
    assert action.move_y == pytest.approx(0.0)


def test_defensive_ai_measures_its_guard_line_from_the_paddle_centre() -> None:
    # Top-left corner exactly on the guard column (x = 0.1 * 800): the centre is 7.5 px
    # beyond it, so the paddle must move back, and it is level with the ball.
    action = DefensiveAI().get_action(_observation((400.0, 300.0), (80.0, 260.0)))
    assert action.move_x < 0
    assert action.move_y == pytest.approx(0.0, abs=1e-6)


def test_tracker_opponents_are_registered_built_ins() -> None:
    assert "tracker" in BUILTIN_OPPONENTS
    assert "striker" in BUILTIN_OPPONENTS
    assert create_ai("tracker").name
    assert create_ai("striker").name


@pytest.mark.parametrize("name", ["tracker", "striker"])
@pytest.mark.parametrize("normalized", [False, True])
def test_goal_line_opponents_retreat_and_align_with_the_ball(name: str, normalized: bool) -> None:
    ai_player = create_ai(name)
    # Paddle forward of its goal line, ball approaching from the right, above the centre.
    observation = _observation(
        (400.0, 100.0), (120.0, 260.0), velocity=(-300.0, 0.0), normalized=normalized
    )
    action = ai_player.get_action(observation)
    assert action.move_x == pytest.approx(-1.0)
    assert action.move_y < 0


def test_goal_line_opponents_use_their_own_side_as_home() -> None:
    for name in ("tracker", "striker"):
        observation = _observation((400.0, 300.0), (600.0, 260.0), velocity=(300.0, 0.0))
        observation["opponent_pos"] = [20.0, 260.0]
        assert create_ai(name).get_action(observation).move_x == pytest.approx(1.0)


def test_tracker_moves_exactly_onto_the_ball_when_within_one_frame() -> None:
    step = game_config.PADDLE_SPEED / game_config.FPS
    observation = _observation((27.5, 300.0 + step / 2), (20.0, 260.0), velocity=(-300.0, 0.0))
    action = create_ai("tracker").get_action(observation)
    assert action.move_y == pytest.approx(0.5)


def test_striker_aims_with_the_paddle_edge_toward_the_far_half() -> None:
    # Ball heading straight at the centre in the upper half: hit with the lower part of the
    # paddle so the return crosses to the lower half, i.e. the paddle centre moves up.
    observation = _observation((400.0, 200.0), (20.0, 160.0), velocity=(-300.0, 0.0))
    striker = create_ai("striker")
    action = striker.get_action(observation)
    assert action.move_y < 0
    lower = _observation((400.0, 400.0), (20.0, 360.0), velocity=(-300.0, 0.0))
    assert striker.get_action(lower).move_y > 0


def test_striker_predicts_wall_reflections_before_aiming() -> None:
    # Ball at (400, 300) moving (-300, -300): it reaches y = 8 after 292/300 s, reflects,
    # and arrives at the paddle face x = 43 at t = 357/300 s with y = 8 + 65 = 73.
    observation = _observation((400.0, 300.0), (20.0, 260.0), velocity=(-300.0, -300.0))
    action = create_ai("striker").get_action(observation)
    assert action.move_y == pytest.approx(-1.0)


def test_goal_line_opponents_return_to_the_centre_when_the_ball_moves_away() -> None:
    for name in ("tracker", "striker"):
        observation = _observation((400.0, 50.0), (20.0, 20.0), velocity=(300.0, 0.0))
        action = create_ai(name).get_action(observation)
        assert action.move_x == pytest.approx(-1.0)
        assert action.move_y > 0
