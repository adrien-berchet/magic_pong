"""numba built-in policies must match the Python opponents on random states."""

import math
import random

import numpy as np
import pytest

pytest.importorskip("numba")

from magic_pong.ai.fast_policies import POLICY_KINDS  # noqa: E402
from magic_pong.ai.fast_policies import builtin_action  # noqa: E402
from magic_pong.ai.interface import ObservationProcessor  # noqa: E402
from magic_pong.ai.models.simple_ai import create_ai  # noqa: E402
from magic_pong.ai.physics_teacher import continuation_action  # noqa: E402
from magic_pong.core import fast_physics  # noqa: E402
from magic_pong.core.entities import Ball  # noqa: E402
from magic_pong.core.entities import Bonus  # noqa: E402
from magic_pong.core.entities import BonusType  # noqa: E402
from magic_pong.core.entities import Vector2D  # noqa: E402
from magic_pong.core.physics import PhysicsEngine  # noqa: E402
from magic_pong.utils.config import game_config  # noqa: E402

PYTHON_NAMES = (
    "dummy",
    "training_dummy",
    "follow_ball",
    "defensive",
    "aggressive",
    "predictive",
    "tracker",
    "striker",
)


def _engine(rng: random.Random) -> PhysicsEngine:
    engine = PhysicsEngine(800, 600)
    for paddle in (engine.player1, engine.player2):
        paddle.position = Vector2D(
            rng.uniform(paddle.min_x, paddle.max_x), rng.uniform(paddle.min_y, paddle.max_y)
        )
        paddle.prev_position = paddle.position.copy()
        if rng.random() < 0.3:
            paddle.apply_size_effect(rng.choice((0.6, 1.5)), 3.0)
    speed = rng.uniform(100, 800)
    angle = rng.uniform(-math.pi, math.pi)
    engine.ball = Ball(
        rng.uniform(30, 770), rng.uniform(20, 580), speed * math.cos(angle), speed * math.sin(angle)
    )
    for _ in range(rng.randint(0, 3)):
        engine.bonuses.append(
            Bonus(rng.uniform(40, 740), rng.uniform(20, 560), rng.choice(list(BonusType)))
        )
    engine.game_time = rng.uniform(0, 100)
    return engine


@pytest.mark.parametrize("name", PYTHON_NAMES)
@pytest.mark.parametrize("seed", range(25))
def test_fast_policy_matches_python_opponent(name: str, seed: int) -> None:
    rng = random.Random(seed)
    engine = _engine(rng)
    side = rng.choice((1, 2))
    observation = ObservationProcessor(800, 600).process_game_state(engine.get_game_state(), side)
    expected = create_ai(name).get_action(observation)
    state, bonuses, rotating, params = fast_physics.snapshot(engine)
    ax, ay = builtin_action(
        POLICY_KINDS[name], state, bonuses, params, side, fast_physics.no_random()
    )
    assert ax == pytest.approx(expected.move_x, abs=1e-9)
    assert ay == pytest.approx(expected.move_y, abs=1e-9)


@pytest.mark.parametrize("seed", range(10))
def test_fast_continuation_matches_teacher_policy(seed: int) -> None:
    rng = random.Random(100 + seed)
    engine = _engine(rng)
    side = rng.choice((1, 2))
    expected = continuation_action(engine, side, 1 / game_config.FPS)
    state, bonuses, rotating, params = fast_physics.snapshot(engine)
    ax, ay = builtin_action(
        POLICY_KINDS["continuation"], state, bonuses, params, side, fast_physics.no_random()
    )
    assert (ax, ay) == pytest.approx((expected.move_x, expected.move_y), abs=1e-9)


def test_random_policy_uses_the_supplied_draws() -> None:
    engine = PhysicsEngine(800, 600)
    state, bonuses, rotating, params = fast_physics.snapshot(engine)
    rand = np.zeros(fast_physics.N_RAND)
    rand[fast_physics.RAND_OPP_X] = 0.75
    rand[fast_physics.RAND_OPP_Y] = 0.0
    assert builtin_action(POLICY_KINDS["random"], state, bonuses, params, 1, rand) == pytest.approx(
        (0.5, -1.0)
    )
