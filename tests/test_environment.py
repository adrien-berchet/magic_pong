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


@pytest.mark.parametrize("headless", [False, True])
@pytest.mark.parametrize("global_headless", [False, True])
def test_environment_headless_mode_preserves_gameplay_dt(
    monkeypatch, headless: bool, global_headless: bool
) -> None:
    """Rendering and fast-mode settings must not change the control problem."""
    physics = PhysicsEngine(800, 600)
    physics.ball.velocity.x = 0.0
    physics.ball.velocity.y = 0.0
    env = GameEnvironment(physics, headless=headless)
    monkeypatch.setattr(ai_config, "HEADLESS_MODE", global_headless)
    monkeypatch.setattr(ai_config, "FAST_MODE_MULTIPLIER", 25.0)
    start_y = physics.player1.position.y
    env.step(Action(0, 1), Action(0, 0))

    expected_dt = game_config.GAME_SPEED_MULTIPLIER / game_config.FPS
    assert physics.game_time == pytest.approx(expected_dt)
    assert physics.player1.position.y - start_y == pytest.approx(
        game_config.PADDLE_SPEED * expected_dt
    )


def test_environment_max_steps_done_and_info_step_count() -> None:
    """The first step should finish when max_steps is one and report the current count."""
    physics = PhysicsEngine(800, 600)
    env = GameEnvironment(physics, headless=True)
    env.max_steps = 1

    *_, done, info = env.step(Action(0, 0), Action(0, 0), dt=0.01)

    assert done is True
    assert info["step_count"] == 1
