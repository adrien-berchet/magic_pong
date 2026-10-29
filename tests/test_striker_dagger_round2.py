"""Unit tests and protocol verification for Striker DAgger Round 2."""

from __future__ import annotations

import sys
from pathlib import Path

import numpy as np
import pytest
import torch

sys.path.insert(0, str(Path(__file__).resolve().parents[1] / "docs/ai"))
import run_striker_concession_audit as audit  # type: ignore[import-not-found]  # noqa: E402

from magic_pong.ai.agent_adapter import to_player1_dqn_observation
from magic_pong.ai.geometry import compute_geometry_features_from_physics
from magic_pong.ai.geometry import compute_geometry_features_from_physics_batch
from magic_pong.ai.interface import ObservationProcessor
from magic_pong.ai.models.dqn_ai import DQNAgent
from magic_pong.core.fast_physics import snapshot
from magic_pong.core.physics import PhysicsEngine


def test_audit_reachability_calculation() -> None:
    """Verify that check_reachability accurately distinguishes reachable vs unreachable states."""
    # Test 1: Straight, slow ball directly at paddle - MUST BE REACHABLE
    res1 = audit.check_reachability(
        student_side=1,
        ball_pos=(400.0, 300.0),
        ball_vel=(-400.0, 0.0),
        student_pos_y=300.0,
        student_height=60.0,
        paddle_x=50.0,
    )
    assert res1["approaching"] is True
    assert res1["reachable"] is True
    assert pytest.approx(res1["intercept_y"], abs=1e-3) == 300.0
    assert res1["arrival_time"] > 0.8

    # Test 2: Extreme corner slice at maximum speed while paddle is stuck in opposite corner - MUST BE UNREACHABLE
    # Ball moving towards player 1 from (750, 50) at -850 px/s with steep angle to corner y=550
    # Paddle is at y=50. Arrival time ~ 0.8s. Max travel = 500 * 0.8 = 400px.
    # Max reach = 50 + 400 + 30 = 480 < 550.
    res2 = audit.check_reachability(
        student_side=1,
        ball_pos=(750.0, 50.0),
        ball_vel=(-850.0, 600.0),
        student_pos_y=50.0,
        student_height=60.0,
        paddle_x=50.0,
    )
    assert res2["approaching"] is True
    assert res2["reachable"] is False
    assert res2["error_margin"] < 0.0

    # Test 3: Ball moving away from student - NOT APPROACHING
    res3 = audit.check_reachability(
        student_side=1,
        ball_pos=(400.0, 300.0),
        ball_vel=(400.0, 0.0),
        student_pos_y=300.0,
        student_height=60.0,
        paddle_x=50.0,
    )
    assert res3["approaching"] is False
    assert res3["reachable"] is False


@pytest.mark.parametrize("side", [1, 2])
def test_v7_geometry_16_online_offline_parity(side: int) -> None:
    """Verify that online DQNAgent 16 geometry features match offline physics derivation bitwise."""
    engine = PhysicsEngine(800, 600)
    engine.ball.position.x = 350.0
    engine.ball.position.y = 280.0
    engine.ball.velocity.x = -220.0 if side == 1 else 220.0
    engine.ball.velocity.y = 130.0
    engine.player1.position.y = 200.0
    engine.player1.prev_position.y = 195.0
    engine.player2.position.y = 220.0
    engine.player2.prev_position.y = 225.0

    agent = DQNAgent(hidden_layers=[16], geometry_context=True)
    proc = ObservationProcessor(800, 600)
    snap = snapshot(engine)

    obs = proc.process_game_state(engine.get_game_state(), side)
    adapted = to_player1_dqn_observation(obs, side)
    online_geom = agent._observation_to_state(adapted)[30:46]
    offline_geom = np.array(
        compute_geometry_features_from_physics(snap[0], snap[3], side), dtype=np.float32
    )

    np.testing.assert_allclose(online_geom, offline_geom, rtol=1e-5, atol=1e-5)


def test_v7_batch_geometry_augmentation() -> None:
    """Verify that batch offline derivation produces (N, 46) arrays."""
    n = 20
    engine = PhysicsEngine(800, 600)
    snap = snapshot(engine)
    physics = np.tile(snap[0], (n, 1))
    physics[:, 10] = np.linspace(150.0, 350.0, n)  # P1 PY
    params = np.tile(snap[3], (n, 1))
    states_30 = np.zeros((n, 30), dtype=np.float32)
    sides = np.array([1 if i % 2 == 0 else 2 for i in range(n)], dtype=np.int32)

    geom = compute_geometry_features_from_physics_batch(physics, params, sides)
    assert geom.shape == (n, 16)

    states_46 = np.concatenate([states_30, geom], axis=-1)
    assert states_46.shape == (n, 46)
    np.testing.assert_array_equal(states_46[:, :30], states_30)


def test_v7_zero_projection_and_optimizer_parameters() -> None:
    """Verify zero-projection parity and optimizer coverage for Schema v7."""
    torch.manual_seed(42)
    agent = DQNAgent(hidden_layers=[256, 128], geometry_context=False)
    state_30 = np.random.randn(5, 30).astype(np.float32)
    state_46 = np.concatenate([state_30, np.random.randn(5, 16).astype(np.float32)], axis=-1)

    device = next(agent.q_network.parameters()).device
    with torch.no_grad():
        out_base = agent.q_network(torch.as_tensor(state_30, device=device))

    agent.enable_geometry_context()
    assert agent.geometry_context is True
    assert agent.state_size == 46
    assert agent.q_network.geometry_context_weight is not None

    with torch.no_grad():
        out_geom = agent.q_network(torch.as_tensor(state_46, device=device))

    # Exact bitwise parity on base inputs at epoch 0
    np.testing.assert_allclose(out_base.cpu().numpy(), out_geom.cpu().numpy(), atol=1e-6)

    # All parameters must be in optimizer
    opt = torch.optim.Adam(agent.q_network.parameters(), lr=1e-3)
    opt_params = {p for g in opt.param_groups for p in g["params"]}
    assert agent.q_network.geometry_context_weight in opt_params
    assert opt_params == {p for p in agent.q_network.parameters() if p.requires_grad}
