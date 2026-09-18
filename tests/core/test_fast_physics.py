"""The numba rollout must reproduce the Python physics engine frame by frame."""

import math
import random

import numpy as np
import pytest

pytest.importorskip("numba")

from magic_pong.ai.fast_teacher import FastPhysicsTeacher  # noqa: E402
from magic_pong.ai.physics_teacher import PhysicsTeacher  # noqa: E402
from magic_pong.core import fast_physics  # noqa: E402
from magic_pong.core.entities import Action  # noqa: E402
from magic_pong.core.entities import Ball  # noqa: E402
from magic_pong.core.entities import Bonus  # noqa: E402
from magic_pong.core.entities import BonusType  # noqa: E402
from magic_pong.core.entities import RotatingPaddle  # noqa: E402
from magic_pong.core.entities import Vector2D  # noqa: E402
from magic_pong.core.physics import PhysicsEngine  # noqa: E402
from magic_pong.utils.config import game_config  # noqa: E402
from magic_pong.utils.config import game_config_tmp  # noqa: E402

DT = 1 / 60


def _random_engine(rng: random.Random, *, with_effects: bool) -> PhysicsEngine:
    engine = PhysicsEngine(800, 600)
    for paddle in (engine.player1, engine.player2):
        paddle.position = Vector2D(
            rng.uniform(paddle.min_x, paddle.max_x), rng.uniform(paddle.min_y, paddle.max_y)
        )
        paddle.prev_position = paddle.position.copy()
        if with_effects and rng.random() < 0.5:
            paddle.apply_size_effect(rng.choice((0.6, 1.5)), rng.uniform(0.05, 2.0))
            paddle.prev_position = paddle.position.copy()
    speed = rng.uniform(150, game_config.MAX_BALL_SPEED)
    angle = rng.uniform(-math.pi, math.pi)
    # Start near a paddle often enough to exercise contacts within a short horizon.
    if rng.random() < 0.5:
        paddle = rng.choice((engine.player1, engine.player2))
        x = paddle.position.x + (paddle.width + 40 if paddle.player_id == 1 else -40)
        y = paddle.position.y + paddle.height / 2 + rng.uniform(-60, 60)
    else:
        x, y = rng.uniform(50, 750), rng.uniform(20, 580)
    engine.ball = Ball(x, y, speed * math.cos(angle), speed * math.sin(angle))
    if with_effects:
        for _ in range(rng.randint(0, 3)):
            engine.bonuses.append(
                Bonus(
                    rng.uniform(40, 740),
                    rng.uniform(20, 560),
                    rng.choice((BonusType.ENLARGE_PADDLE, BonusType.SHRINK_OPPONENT)),
                )
            )
        for _ in range(rng.randint(0, 2)):
            rotating = RotatingPaddle(
                rng.uniform(100, 700), rng.uniform(60, 540), rng.choice((1, 2))
            )
            rotating.angle = rng.uniform(0, 2 * math.pi)
            rotating.lifetime = rng.uniform(0.2, 5.0)
            engine.rotating_paddles.append(rotating)
    return engine


def _random_actions(rng: random.Random, frames: int) -> np.ndarray:
    choices = (-1.0, 0.0, 1.0)
    return np.array(
        [(rng.choice(choices), rng.choice(choices)) for _ in range(frames)], dtype=np.float64
    )


def _python_trace(engine: PhysicsEngine, actions1: np.ndarray, actions2: np.ndarray) -> dict:
    counts = np.zeros(fast_physics.N_EVENTS, dtype=np.int64)
    frames = 0
    for a1, a2 in zip(actions1, actions2):
        events = engine.update(DT, Action(*a1), Action(*a2))
        frames += 1
        for hit in events["paddle_hits"]:
            counts[fast_physics.EV_HIT1 + hit["player"] - 1] += 1
        counts[fast_physics.EV_WALL] += len(events["wall_bounces"])
        counts[fast_physics.EV_ROTATING] += len(events["rotating_paddle_hits"])
        for collected in events["bonus_collected"]:
            counts[fast_physics.EV_BONUS1 + collected["player"] - 1] += 1
        if events["goals"]:
            counts[fast_physics.EV_GOAL] = events["goals"][0]["player"]
            break
    return {"frames": frames, "counts": counts}


@pytest.mark.parametrize("seed", range(40))
@pytest.mark.parametrize("with_effects", [False, True])
def test_fast_simulation_matches_python_engine(seed: int, with_effects: bool) -> None:
    rng = random.Random(seed)
    frames = 120
    with game_config_tmp(BONUSES_ENABLED=True, BONUS_SPAWN_INTERVAL=1e9):
        engine = _random_engine(rng, with_effects=with_effects)
        state = fast_physics.snapshot(engine)
        actions1 = _random_actions(rng, frames)
        actions2 = _random_actions(rng, frames)
        result = fast_physics.simulate(*state, actions1, actions2)
        expected = _python_trace(engine, actions1, actions2)
    assert result.frames == expected["frames"]
    np.testing.assert_array_equal(result.counts, expected["counts"])
    if expected["counts"][fast_physics.EV_GOAL]:
        return  # The Python engine serves a fresh ball after a goal; the kernel stops.
    ball = engine.ball
    assert result.state[fast_physics.BALL_X : fast_physics.BALL_Y + 1] == pytest.approx(
        (ball.position.x, ball.position.y), abs=1e-6
    )
    assert result.state[fast_physics.BALL_VX : fast_physics.BALL_VY + 1] == pytest.approx(
        (ball.velocity.x, ball.velocity.y), abs=1e-6
    )
    for base, paddle in ((fast_physics.P1, engine.player1), (fast_physics.P2, engine.player2)):
        assert result.state[base + fast_physics.PX] == pytest.approx(paddle.position.x, abs=1e-6)
        assert result.state[base + fast_physics.PY] == pytest.approx(paddle.position.y, abs=1e-6)
        assert result.state[base + fast_physics.PH] == pytest.approx(paddle.height, abs=1e-9)
        assert result.state[base + fast_physics.PTIMER] == pytest.approx(
            paddle.size_effect_timer, abs=1e-9
        )


def test_snapshot_does_not_alias_engine_state() -> None:
    engine = PhysicsEngine(800, 600)
    state = fast_physics.snapshot(engine)
    fast_physics.simulate(*state, _random_actions(random.Random(1), 30), np.zeros((30, 2)))
    assert engine.ball.position.to_tuple() == (400.0, 300.0)


@pytest.mark.parametrize("seed", range(12))
@pytest.mark.parametrize("player_id", [1, 2])
def test_fast_teacher_reproduces_python_teacher_values(seed: int, player_id: int) -> None:
    rng = random.Random(100 + seed)
    with game_config_tmp(BONUSES_ENABLED=False):
        engine = _random_engine(rng, with_effects=False)
        slow = PhysicsTeacher(horizon_steps=45, action_steps=6).label(engine, player_id)
        fast = FastPhysicsTeacher(horizon_steps=45, action_steps=6).label(engine, player_id)
    np.testing.assert_allclose(fast.values, slow.values, rtol=1e-6, atol=1e-6)
    assert fast.action == slow.action


def test_fast_teacher_y_only_restricts_actions() -> None:
    engine = PhysicsEngine(800, 600)
    decision = FastPhysicsTeacher(y_only=True).label(engine, 1)
    assert decision.action in (0, 1, 2)
    assert np.isneginf(decision.values[3:]).all()


def test_action_table_matches_dqn_action_mapping() -> None:
    from magic_pong.ai.models.dqn_ai import ACTION_MAPPING

    for index, action in ACTION_MAPPING.items():
        assert tuple(fast_physics.ACTION_TABLE[index]) == (action.move_x, action.move_y)
