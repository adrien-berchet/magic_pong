"""
Regression tests for the AI game environment boundary.
"""

import pytest

from magic_pong.ai.interface import GameEnvironment
from magic_pong.core.entities import Action
from magic_pong.core.physics import PhysicsEngine
from magic_pong.utils.config import ai_config
from magic_pong.utils.config import game_config


def test_environment_explicit_dt_is_threaded_to_physics() -> None:
    """An explicit dt from GameEngine/GameEnvironment callers should drive physics."""
    physics = PhysicsEngine(800, 600)
    physics.ball.velocity.x = 100.0
    physics.ball.velocity.y = 0.0
    start_x = physics.ball.position.x

    env = GameEnvironment(physics, headless=True)
    env.step(Action(0, 0), Action(0, 0), dt=0.5)

    assert physics.game_time == pytest.approx(0.5)
    assert physics.ball.position.x == pytest.approx(start_x + 50.0)


def test_environment_headless_constructor_enables_default_fast_dt() -> None:
    """Default dt should honor the environment headless flag, not only the global config."""
    physics = PhysicsEngine(800, 600)
    physics.ball.velocity.x = 0.0
    physics.ball.velocity.y = 0.0
    env = GameEnvironment(physics, headless=True)

    original_headless = ai_config.HEADLESS_MODE
    try:
        ai_config.HEADLESS_MODE = False
        env.step(Action(0, 0), Action(0, 0))
    finally:
        ai_config.HEADLESS_MODE = original_headless

    expected_dt = (
        game_config.GAME_SPEED_MULTIPLIER / game_config.FPS * ai_config.FAST_MODE_MULTIPLIER
    )
    assert physics.game_time == pytest.approx(expected_dt)


def test_environment_max_steps_done_and_info_step_count() -> None:
    """The first step should finish when max_steps is one and report the current count."""
    physics = PhysicsEngine(800, 600)
    env = GameEnvironment(physics, headless=True)
    env.max_steps = 1

    *_, done, info = env.step(Action(0, 0), Action(0, 0), dt=0.01)

    assert done is True
    assert info["step_count"] == 1
