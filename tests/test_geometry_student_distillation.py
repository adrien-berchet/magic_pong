"""Guardrails and unit tests for geometry-context student-state distillation."""

import copy
import sys
from pathlib import Path

import numpy as np
import pytest
import torch

sys.path.insert(0, str(Path(__file__).resolve().parents[1] / "docs/ai"))
import run_geometry_student_distillation as run  # noqa: E402

from magic_pong.ai.agent_adapter import to_player1_dqn_observation
from magic_pong.ai.geometry import compute_geometry_features_from_physics
from magic_pong.ai.interface import ObservationProcessor
from magic_pong.ai.models.dqn_ai import DQNAgent
from magic_pong.core.fast_physics import snapshot
from magic_pong.core.physics import PhysicsEngine


def test_splits_disjointness():
    # Valid default settings must pass
    run.assert_splits(run.SETTINGS)

    # Must reject overlap with 2026-09-21 seeds
    stale = copy.deepcopy(run.SETTINGS)
    stale["validation_seeds"] = [690100, 690101]
    with pytest.raises(ValueError, match="Reused"):
        run.assert_splits(stale)

    # Must reject internal overlap between validation and heldout
    overlap = copy.deepcopy(run.SETTINGS)
    overlap["heldout_seeds"][0] = overlap["validation_seeds"][0]
    with pytest.raises(ValueError, match="Overlapping"):
        run.assert_splits(overlap)

    # Must reject internal overlap between heldout and scoring
    overlap_score = copy.deepcopy(run.SETTINGS)
    overlap_score["scoring_seeds"][0] = overlap_score["heldout_seeds"][0]
    with pytest.raises(ValueError, match="Overlapping"):
        run.assert_splits(overlap_score)


@pytest.mark.parametrize("side", [1, 2])
def test_offline_online_geometry_parity(side):
    """Test 2: Online observation features must match offline physics derivation bitwise/numerically."""
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


def test_augment_dataset_with_geometry():
    """Verify that batch offline derivation produces (N, 34) arrays and validates bounds."""
    n = 20
    engine = PhysicsEngine(800, 600)
    snap = snapshot(engine)
    physics = np.tile(snap[0], (n, 1))
    # Give non-zero movement so variance check passes
    physics[:, 10] = np.linspace(150.0, 350.0, n)  # P1 PY
    physics[:, 11] = physics[:, 10] - 2.0  # P1 PPY
    params = np.tile(snap[3], (n, 1))
    states = np.zeros((n, 30), dtype=np.float32)

    raw_data = {"states": states, "physics": physics, "params": params}
    sides = np.array([1 if i % 2 == 0 else 2 for i in range(n)], dtype=np.int32)

    augmented = run.augment_dataset_with_geometry(raw_data, sides)
    assert augmented["states"].shape == (n, 34)
    np.testing.assert_array_equal(augmented["states"][:, :30], states)

    # Assert bounded values
    assert np.all(augmented["states"][:, 30] >= 0.0)
    assert np.all(augmented["states"][:, 30] <= 1.0)
    assert np.all(augmented["states"][:, 31] > 0.0)
    assert np.all(augmented["states"][:, 31] <= 1.0)
    assert np.all(augmented["states"][:, 32] >= -1.0)
    assert np.all(augmented["states"][:, 32] <= 1.0)
    assert np.all(augmented["states"][:, 33] >= -1.0)
    assert np.all(augmented["states"][:, 33] <= 1.0)


def test_compute_scoring_metrics():
    matches = [
        {
            "opponent": "tracker",
            "agent_won": True,
            "timeout": False,
            "goals_for": 3,
            "goals_against": 1,
            "steps": 1500,
        },
        {
            "opponent": "tracker",
            "agent_won": False,
            "timeout": True,
            "goals_for": 0,
            "goals_against": 0,
            "steps": 6000,
        },
        {
            "opponent": "striker",
            "agent_won": False,
            "timeout": False,
            "goals_for": 1,
            "goals_against": 3,
            "steps": 2500,
        },
    ]
    # Tracker metrics
    tracker = run.compute_scoring_metrics(matches, "tracker")
    assert tracker["matches"] == 2
    assert tracker["wins"] == 1
    assert tracker["losses"] == 0
    assert tracker["timeouts"] == 1
    assert tracker["goals_for"] == 3
    assert tracker["steps"] == 7500
    # 3 goals / 7500 frames * 1000 = 0.4 goals/1k frames
    assert pytest.approx(tracker["goals_for_per_1000_frames"]) == 0.4
    # 7500 frames / 3 goals = 2500 frames per goal
    assert pytest.approx(tracker["frames_per_goal_for"]) == 2500.0

    # Total metrics
    total = run.compute_scoring_metrics(matches)
    assert total["matches"] == 3
    assert total["goals_for"] == 4
    assert total["steps"] == 10000
    assert pytest.approx(total["goals_for_per_1000_frames"]) == 0.4


def test_alias_resolution_in_metric_aggregation(tmp_path):
    """Test 3: Alias resolution must not drop seeds or double count."""
    dir_11 = tmp_path / "seed_11"
    dir_11.mkdir()
    res_a = {
        "matches": [
            {
                "opponent": "tracker",
                "agent_won": True,
                "timeout": False,
                "goals_for": 3,
                "goals_against": 0,
                "steps": 1000,
            }
        ]
    }
    path_a = dir_11 / "heldout_control_geom.json"
    run.write_json(path_a, res_a)

    # Policy B is an alias of Policy A
    index = {
        "control_geom": {
            "arm": "control_geom",
            "epoch": 10,
            "artifact": str(path_a),
            "alias_of": None,
        },
        "alias_policy": {
            "arm": "alias_policy",
            "epoch": 10,
            "artifact": str(path_a),
            "alias_of": "control_geom",
        },
    }

    # When resolving via spec["artifact"], both point to valid results
    data_a = json_loads(Path(index["control_geom"]["artifact"]).read_text())
    data_b = json_loads(Path(index["alias_policy"]["artifact"]).read_text())
    assert data_a == data_b
    assert data_b["matches"][0]["goals_for"] == 3


def test_zero_projection_initial_parity_and_optimizer_param_coverage():
    """Verify epoch 0 bitwise forward pass parity and complete optimizer coverage."""
    torch.manual_seed(123)
    agent = DQNAgent(hidden_layers=[32, 16], geometry_context=False)
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

    # Epoch 0 predictions must be bitwise/numerically identical on the base inputs
    np.testing.assert_allclose(out_base.cpu().numpy(), out_geom.cpu().numpy(), atol=1e-6)

    # Optimizer must include geometry_context_weight
    opt = torch.optim.Adam(agent.q_network.parameters(), lr=1e-3)
    opt_params = {p for g in opt.param_groups for p in g["params"]}
    assert agent.q_network.geometry_context_weight in opt_params
    assert opt_params == {p for p in agent.q_network.parameters() if p.requires_grad}


def json_loads(text):
    import json

    return json.loads(text)
