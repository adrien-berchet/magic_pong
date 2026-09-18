"""Regressions for simulation units, field geometry and observation contracts."""

import numpy as np
import pytest

from magic_pong.ai.agent_adapter import to_player1_dqn_observation
from magic_pong.ai.interface import GameEnvironment
from magic_pong.ai.models.dqn_ai import DQNAgent
from magic_pong.ai.models.simple_ai import UnitObservationMixin
from magic_pong.core.collision import CollisionDetector
from magic_pong.core.collision import apply_paddle_bounce
from magic_pong.core.entities import Action
from magic_pong.core.entities import Ball
from magic_pong.core.entities import Paddle
from magic_pong.core.entities import Vector2D
from magic_pong.core.physics import PhysicsEngine
from magic_pong.utils.config import ai_config
from magic_pong.utils.config import game_config


@pytest.mark.parametrize("dt", [1 / 60, 1 / 6])
def test_paddle_bounce_speed_change_is_independent_of_timestep(monkeypatch, dt):
    monkeypatch.setattr(game_config, "MAX_BALL_SPEED", 2000.0)
    paddle = Paddle(100, 100, 1)
    paddle.prev_position.x = paddle.position.x - 200 * dt
    ball = Ball(120, 140, -300, 0)
    apply_paddle_bounce(ball, paddle, Vector2D(1, 0), dt=dt)
    assert ball.velocity.x == pytest.approx(300 * game_config.BALL_SPEED_INCREASE)


def test_moving_paddle_catches_slower_ball_from_behind(monkeypatch):
    monkeypatch.setattr(game_config, "MAX_BALL_SPEED", 2000.0)
    dt = 1 / 60
    paddle = Paddle(100, 100, 1)
    paddle.prev_position.x = 100 - 500 * dt
    ball = Ball(119, 140, 100, 0)
    ball.prev_position = Vector2D(119 - 100 * dt, 140)
    assert not CollisionDetector().check_ball_paddle(ball, paddle, dt)
    assert ball.velocity.to_tuple() == (100, 0)
    assert ball.position.x > paddle.position.x + paddle.width + ball.radius


@pytest.mark.parametrize("player_id", [1, 2])
def test_dqn_raw_and_normalized_observations_encode_identically(monkeypatch, player_id):
    env = GameEnvironment(PhysicsEngine(1000, 750))
    state = env.physics_engine.get_game_state()
    state["active_bonuses"] = [(300.0, 450.0, "enlarge_paddle")]
    state["rotating_paddles"] = [(700.0, 250.0, 0.2)]
    agent = DQNAgent(hidden_size=8, layer_size=1)

    monkeypatch.setattr(ai_config, "NORMALIZE_POSITIONS", False)
    raw = env.observation_processor.process_game_state(state, player_id)
    monkeypatch.setattr(ai_config, "NORMALIZE_POSITIONS", True)
    normalized = env.observation_processor.process_game_state(state, player_id)
    # Deliberately process stored observations under the opposite global flag.
    raw_state = agent._observation_to_state(to_player1_dqn_observation(raw, player_id))
    monkeypatch.setattr(ai_config, "NORMALIZE_POSITIONS", False)
    normalized_state = agent._observation_to_state(
        to_player1_dqn_observation(normalized, player_id)
    )
    np.testing.assert_allclose(raw_state, normalized_state, atol=1e-7)
    assert raw_state[0] == pytest.approx(0.5)
    assert raw_state[1] == pytest.approx(0.5)
    assert raw_state[15] == pytest.approx(0.3 if player_id == 1 else 0.7)


def test_configured_geometry_controls_bounds_resets_and_observations(monkeypatch):
    env = GameEnvironment(PhysicsEngine(800, 600))
    monkeypatch.setattr(game_config, "FIELD_HEIGHT", 655)
    env.configure_geometry(1000, 655)
    obs, _ = env.reset()
    physics = env.physics_engine
    assert physics.field_width == physics.bonus_spawner.field_width == 1000
    assert physics.field_height == physics.bonus_spawner.field_height == 655
    assert obs["field_width"] == 1000
    assert obs["field_height"] == 655
    assert physics.ball.position.to_tuple() == (500, 327.5)
    assert physics.player1.max_y == 655 - physics.player1.height
    assert physics.player2.max_x == 1000 - physics.player2.width - game_config.PADDLE_MARGIN

    # Existing engines retain their dimensions after global randomization ends.
    monkeypatch.setattr(game_config, "FIELD_HEIGHT", 600)
    physics.player1.apply_size_effect(2, 1)
    assert physics.player1.max_y == 655 - physics.player1.height
    physics.reset_game()
    assert physics.player1.max_y == 655 - physics.player1.height


@pytest.mark.parametrize("normalized", [False, True])
def test_builtin_observation_respects_explicit_units_near_zero(monkeypatch, normalized):
    env = GameEnvironment(PhysicsEngine(800, 600))
    state = env.physics_engine.get_game_state()
    state["ball_position"] = (0.5, 0.25)
    state["ball_velocity"] = (0.5, 50.0)
    state["active_bonuses"] = [(0.5, 0.25, "enlarge_paddle")]
    monkeypatch.setattr(ai_config, "NORMALIZE_POSITIONS", normalized)
    observation = env.observation_processor.process_game_state(state, 1)
    unit = UnitObservationMixin()._unit_observation(observation)
    assert unit.ball_pos == pytest.approx((0.5 / 800, 0.25 / 600))
    assert unit.ball_vel == pytest.approx(
        (0.5 / game_config.MAX_BALL_SPEED, 50.0 / game_config.MAX_BALL_SPEED)
    )
    assert unit.bonuses[0][:2] == pytest.approx((0.5 / 800, 0.25 / 600))


def test_large_physics_step_matches_repeated_gameplay_steps(monkeypatch):
    monkeypatch.setattr(game_config, "BONUSES_ENABLED", False)
    one_step = PhysicsEngine(800, 600)
    repeated = PhysicsEngine(800, 600)
    for physics in (one_step, repeated):
        physics.ball = Ball(400, 20, 100, -300)
    events = one_step.update(6 / game_config.FPS, Action(0, 1), Action(0, 0))
    for _ in range(6):
        repeated.update(1 / game_config.FPS, Action(0, 1), Action(0, 0))
    assert one_step.ball.position.to_tuple() == pytest.approx(repeated.ball.position.to_tuple())
    assert one_step.ball.velocity.to_tuple() == pytest.approx(repeated.ball.velocity.to_tuple())
    assert one_step.player1.position.to_tuple() == pytest.approx(
        repeated.player1.position.to_tuple()
    )
    assert events["wall_bounces"] == ["top"]


def test_large_physics_step_stops_at_match_end(monkeypatch):
    monkeypatch.setattr(game_config, "BONUSES_ENABLED", False)
    monkeypatch.setattr(game_config, "MAX_SCORE", 1)
    physics = PhysicsEngine(800, 600)
    physics.ball = Ball(790, 100, 100, 0)
    events = physics.update(10.0, Action(0, 0), Action(0, 0))
    assert physics.game_time < 0.1
    assert physics.score == [1, 0]
    assert len(events["goals"]) == 1
