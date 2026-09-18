"""Ball speed and position-only paddle pushing contracts."""

import math

import pytest

from magic_pong.ai.pretraining import create_pretrainer
from magic_pong.core.collision import CollisionDetector
from magic_pong.core.collision import apply_paddle_bounce
from magic_pong.core.collision import circle_rect_collision
from magic_pong.core.entities import Action
from magic_pong.core.entities import Ball
from magic_pong.core.entities import Paddle
from magic_pong.core.entities import Vector2D
from magic_pong.core.physics import PhysicsEngine
from magic_pong.utils.config import game_config


@pytest.mark.parametrize("side", [1, 2])
@pytest.mark.parametrize("velocity", [(300, 0), (240, 180), (0, 0)])
def test_repeated_push_preserves_velocity_and_releases_immediately(monkeypatch, side, velocity):
    monkeypatch.setattr(game_config, "BONUSES_ENABLED", False)
    engine = PhysicsEngine(800, 600)
    paddle = getattr(engine, f"player{side}")
    paddle.position = Vector2D(100 if side == 1 else 600, 200)
    paddle.prev_position = paddle.position.copy()
    sign = 1 if side == 1 else -1
    front = paddle.position.x + (paddle.width if side == 1 else 0)
    engine.ball = Ball(
        front + sign * (game_config.BALL_RADIUS + 1), 240, sign * velocity[0], velocity[1]
    )
    engine.ball.last_paddle_hit = side  # A prior hit must not suppress pushing.
    original_velocity = engine.ball.velocity.to_tuple()
    push = Action(sign, 0)
    actions = (push, Action(0, 0)) if side == 1 else (Action(0, 0), push)
    for _ in range(10):
        events = engine.update(1 / 60, *actions)
        assert events["paddle_hits"] == []
        assert engine.ball.velocity.to_tuple() == original_velocity
        assert not circle_rect_collision(engine.ball, paddle.get_rect())
        front = paddle.position.x + (paddle.width if side == 1 else 0)
        assert sign * (engine.ball.position.x - front) == pytest.approx(
            engine.ball.radius, abs=1e-5
        )
    previous = engine.ball.position.copy()
    events = engine.update(1 / 60, Action(0, 0), Action(0, 0))
    assert events["paddle_hits"] == []
    assert engine.ball.velocity.to_tuple() == original_velocity
    assert engine.ball.position.to_tuple() == pytest.approx(
        (previous + engine.ball.velocity * (1 / 60)).to_tuple(), abs=1e-8
    )


@pytest.mark.parametrize("offset", [-30, 0, 30])
@pytest.mark.parametrize("motion", [(0, 0), (500, 0), (0, 500), (500, 500)])
def test_bounce_scales_speed_by_increase_factor_for_all_paddle_motion(offset, motion):
    paddle = Paddle(100, 200, 1)
    paddle.prev_position = paddle.position - Vector2D(*motion) * (1 / 60)
    ball = Ball(121, 240 + offset, -240, 180)
    original = ball.velocity.magnitude()
    apply_paddle_bounce(ball, paddle, dt=1 / 60)
    expected = original * game_config.BALL_SPEED_INCREASE
    assert ball.velocity.magnitude() == pytest.approx(expected, abs=1e-9)


def test_bounce_speed_is_capped_at_max_ball_speed(monkeypatch):
    monkeypatch.setattr(game_config, "MAX_BALL_SPEED", 310.0)
    paddle = Paddle(100, 200, 1)
    ball = Ball(121, 240, -300, 0)
    apply_paddle_bounce(ball, paddle, dt=1 / 60)
    assert ball.velocity.magnitude() == pytest.approx(310.0, abs=1e-9)
    apply_paddle_bounce(ball, paddle, dt=1 / 60)
    assert ball.velocity.magnitude() == pytest.approx(310.0, abs=1e-9)


def test_rally_accelerates_on_every_paddle_bounce_and_serve_resets_speed(monkeypatch):
    monkeypatch.setattr(game_config, "BONUSES_ENABLED", False)
    increase = game_config.BALL_SPEED_INCREASE
    engine = PhysicsEngine(800, 600)
    engine.player1.position = Vector2D(20, 200)
    engine.player1.prev_position = engine.player1.position.copy()
    engine.player2.position = Vector2D(765, 200)
    engine.player2.prev_position = engine.player2.position.copy()
    engine.ball = Ball(60, 240, -300, 0)
    hits = []
    for _ in range(300):
        events = engine.update(1 / 60, Action(0, 0), Action(0, 0))
        hits.extend(hit["player"] for hit in events["paddle_hits"])
        if len(hits) == 2:
            break
    assert hits == [1, 2]
    assert engine.ball.velocity.magnitude() == pytest.approx(300 * increase**2, abs=1e-6)
    engine.player1.position = Vector2D(20, 500)
    engine.player1.prev_position = engine.player1.position.copy()
    engine.ball = Ball(12, 240, -300 * increase**2, 0)
    events = engine.update(1 / 60, Action(0, 0), Action(0, 0))
    assert events["goals"] == [{"player": 2, "score": [0, 1]}]
    assert engine.ball.velocity.magnitude() == pytest.approx(game_config.BALL_SPEED)


@pytest.mark.parametrize("side", [1, 2])
def test_head_on_moving_paddle_still_bounces(monkeypatch, side):
    monkeypatch.setattr(game_config, "BONUSES_ENABLED", False)
    engine = PhysicsEngine(800, 600)
    paddle = getattr(engine, f"player{side}")
    paddle.position = Vector2D(100 if side == 1 else 600, 200)
    paddle.prev_position = paddle.position.copy()
    sign = 1 if side == 1 else -1
    front = paddle.position.x + (paddle.width if side == 1 else 0)
    engine.ball = Ball(front + sign * (game_config.BALL_RADIUS + 2), 240, -sign * 300, 0)
    actions = (Action(sign, 0), Action(0, 0)) if side == 1 else (Action(0, 0), Action(sign, 0))
    events = engine.update(1 / 60, *actions)
    assert events["paddle_hits"] == [{"player": side}]
    expected = (sign * 300 * game_config.BALL_SPEED_INCREASE, 0)
    assert engine.ball.velocity.to_tuple() == pytest.approx(expected)
    assert not circle_rect_collision(engine.ball, paddle.get_rect())


def test_already_overlapping_outgoing_ball_is_only_separated():
    paddle = Paddle(100, 200, 1)
    ball = Ball(120, 240, 300, 0)
    ball.prev_position = Vector2D(118, 240)
    assert not CollisionDetector().check_ball_paddle(ball, paddle, 1 / 60)
    assert ball.velocity.to_tuple() == (300, 0)
    assert not circle_rect_collision(ball, paddle.get_rect())


def test_swept_collision_uses_entry_side_for_bounce_and_separation():
    paddle = Paddle(400, 200, 2)
    ball = Ball(450, 240, 300, 0)
    ball.prev_position = Vector2D(380, 240)
    assert CollisionDetector().check_ball_paddle(ball, paddle, 1 / 60)
    assert ball.velocity.to_tuple() == pytest.approx((-300 * game_config.BALL_SPEED_INCREASE, 0))
    assert ball.position.x < paddle.position.x - ball.radius


def test_tangential_push_near_corner_is_not_misclassified_by_spin():
    paddle = Paddle(100, 200, 1)
    paddle.prev_position = Vector2D(100 - 500 / 60, 200)
    ball = Ball(121, 270, 0, -300)
    ball.prev_position = Vector2D(121, 275)
    assert not CollisionDetector().check_ball_paddle(ball, paddle, 1 / 60)
    assert ball.velocity.to_tuple() == (0, -300)
    assert not circle_rect_collision(ball, paddle.get_rect())


def test_diagonal_corner_push_preserves_both_velocity_components():
    paddle = Paddle(100, 200, 1)
    paddle.prev_position = Vector2D(100 - 500 / 60, 200 - 500 / 60)
    ball = Ball(120, 285, 0, 300)
    ball.prev_position = Vector2D(120, 280)
    assert not CollisionDetector().check_ball_paddle(ball, paddle, 1 / 60)
    assert ball.velocity.to_tuple() == (0, 300)
    assert not circle_rect_collision(ball, paddle.get_rect())


@pytest.mark.parametrize("direction", [-1, 1])
def test_vertical_push(direction):
    paddle = Paddle(100, 200, 1)
    paddle.prev_position = Vector2D(100, 200 - direction * 500 / 60)
    edge = paddle.position.y + (paddle.height if direction > 0 else 0)
    ball = Ball(107.5, edge + direction * 6, 0, direction * 300)
    ball.prev_position = ball.position - ball.velocity * (1 / 60)
    assert not CollisionDetector().check_ball_paddle(ball, paddle, 1 / 60)
    assert ball.velocity.to_tuple() == (0, direction * 300)
    assert not circle_rect_collision(ball, paddle.get_rect())


def test_grazing_bounce_spin_cannot_send_ball_back_into_paddle():
    paddle = Paddle(100, 200, 1)
    ball = Ball(121, 270, -10, math.sqrt(300**2 - 10**2))
    apply_paddle_bounce(ball, paddle)
    assert ball.velocity.x > 0
    expected = 300 * game_config.BALL_SPEED_INCREASE
    assert ball.velocity.magnitude() == pytest.approx(expected, abs=1e-9)


def test_new_incoming_contact_on_same_paddle_can_bounce():
    paddle = Paddle(100, 200, 1)
    ball = Ball(95, 240, 300, 0)
    ball.prev_position = Vector2D(85, 240)
    ball.last_paddle_hit = 1
    assert CollisionDetector().check_ball_paddle(ball, paddle, 1 / 60)
    assert ball.velocity.to_tuple() == pytest.approx((-300 * game_config.BALL_SPEED_INCREASE, 0))


def test_incoming_ball_starting_exactly_at_contact_bounces_this_frame():
    paddle = Paddle(100, 200, 1)
    ball = Ball(118, 240, -300, 0)
    ball.prev_position = Vector2D(123, 240)
    assert CollisionDetector().check_ball_paddle(ball, paddle, 1 / 60)
    assert ball.velocity.to_tuple() == pytest.approx((300 * game_config.BALL_SPEED_INCREASE, 0))


def test_touching_ball_separating_from_retreating_paddle_does_not_bounce():
    paddle = Paddle(100, 200, 1)
    paddle.prev_position = Vector2D(100 + 500 / 60, 200)
    ball = Ball(123 + 500 / 60 - 5, 240, -300, 0)
    ball.prev_position = Vector2D(123 + 500 / 60, 240)
    original = ball.position.to_tuple()
    assert not CollisionDetector().check_ball_paddle(ball, paddle, 1 / 60)
    assert ball.velocity.to_tuple() == (-300, 0)
    assert ball.position.to_tuple() == original


def test_size_expiry_only_separates_new_overlap(monkeypatch):
    monkeypatch.setattr(game_config, "BONUSES_ENABLED", False)
    engine = PhysicsEngine(800, 600)
    paddle = engine.player1
    paddle.apply_size_effect(0.5, 0.001)
    paddle.position = Vector2D(100, 560)
    paddle.prev_position = paddle.position.copy()
    engine.ball = Ball(110, 545, 0, 300)
    events = engine.update(1 / 60, Action(0, 0), Action(0, 0))
    assert paddle.height == paddle.original_height
    assert paddle.position.to_tuple() == paddle.prev_position.to_tuple()
    assert events["paddle_hits"] == []
    assert engine.ball.velocity.to_tuple() == (0, 300)
    assert not circle_rect_collision(engine.ball, paddle.get_rect())


def test_wall_bounces_preserve_speed(monkeypatch):
    monkeypatch.setattr(game_config, "BONUSES_ENABLED", False)
    engine = PhysicsEngine(800, 600)
    engine.ball = Ball(400, 300, 0, game_config.BALL_SPEED)
    bounces = 0
    for _ in range(500):
        events = engine.update(1 / 60, Action(0, 0), Action(0, 0))
        bounces += len(events["wall_bounces"])
        assert engine.ball.velocity.magnitude() == pytest.approx(game_config.BALL_SPEED)
    assert bounces > 0


@pytest.mark.parametrize("side", [1, 2])
@pytest.mark.parametrize("speed", [180.0, 300.0, 500.0])
def test_pretraining_scenarios_use_configured_total_ball_speed(monkeypatch, side, speed):
    monkeypatch.setattr(game_config, "BALL_SPEED", speed)
    pretrainer = create_pretrainer(mode="simulator", seed=19)
    modes = set()
    for _ in range(60):
        pretrainer._new_scenario(side)
        modes.add(pretrainer.current_mode)
        velocity = pretrainer.physics.ball.velocity
        assert math.hypot(velocity.x, velocity.y) == pytest.approx(speed, abs=1e-10)
    assert modes == set(pretrainer.scenario_modes)
