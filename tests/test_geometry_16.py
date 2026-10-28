"""Comprehensive verification suite for 16-feature geometry_context (schema v7)."""

import sys
from pathlib import Path

import numpy as np
import pytest
import torch

from magic_pong.ai.agent_adapter import to_player1_dqn_observation
from magic_pong.ai.geometry import FEATURE_NAMES
from magic_pong.ai.geometry import GEOMETRY_CONTEXT_FEATURES
from magic_pong.ai.geometry import GEOMETRY_CONTEXT_HORIZON
from magic_pong.ai.geometry import GEOMETRY_CONTEXT_VERSION
from magic_pong.ai.geometry import ContactGeometryConfig
from magic_pong.ai.geometry import compute_contact_features
from magic_pong.ai.interface import ObservationProcessor
from magic_pong.ai.models.converter import convert_probe_to_dqn_checkpoint
from magic_pong.ai.models.dqn_ai import DQNAgent
from magic_pong.ai.models.dqn_ai import DQNNetwork
from magic_pong.ai.models.dqn_checkpoint import FULL_CONTEXT_STATE_SIZE
from magic_pong.ai.models.dqn_checkpoint import GEOMETRY_CONTEXT_STATE_SIZE
from magic_pong.ai.models.dqn_checkpoint import DQNCheckpointError
from magic_pong.ai.models.dqn_checkpoint import validate_dqn_checkpoint
from magic_pong.ai.vector_env import encode_observation
from magic_pong.core.fast_physics import snapshot
from magic_pong.core.physics import PhysicsEngine
from magic_pong.utils.config import game_config

sys.path.insert(0, str(Path(__file__).resolve().parents[1] / "docs/ai"))
from planner_contact_features import ContactGeometryConfig as ProbeConfig
from planner_contact_features import contact_features as probe_contact_features
from run_planner_geometry_cost import ProbeNetwork


def test_feature_tuple_matches_probe_specification():
    assert len(GEOMETRY_CONTEXT_FEATURES) == 16
    assert GEOMETRY_CONTEXT_FEATURES == FEATURE_NAMES
    assert GEOMETRY_CONTEXT_HORIZON == 1.5
    assert GEOMETRY_CONTEXT_VERSION == 2
    assert GEOMETRY_CONTEXT_STATE_SIZE == 46
    assert FULL_CONTEXT_STATE_SIZE == 49


def test_pinned_constants_match_game_config():
    cfg = ContactGeometryConfig.from_game_config()
    assert cfg.width == float(game_config.FIELD_WIDTH)
    assert cfg.height == float(game_config.FIELD_HEIGHT)
    assert cfg.max_ball_speed == float(game_config.MAX_BALL_SPEED)
    assert cfg.paddle_width == float(game_config.PADDLE_WIDTH)
    assert cfg.paddle_height == float(game_config.PADDLE_HEIGHT)
    assert cfg.paddle_speed == float(game_config.PADDLE_SPEED)
    assert cfg.ball_radius == float(game_config.BALL_RADIUS)
    assert cfg.horizon_seconds == 1.5


def test_exact_parity_with_probe_reference_across_random_states():
    cfg = ContactGeometryConfig.from_game_config()
    probe_cfg = ProbeConfig(
        width=cfg.width,
        height=cfg.height,
        max_ball_speed=cfg.max_ball_speed,
        paddle_width=cfg.paddle_width,
        paddle_height=cfg.paddle_height,
        paddle_speed=cfg.paddle_speed,
        ball_radius=cfg.ball_radius,
        horizon_frames=90,
        fps=60.0,
    )

    np.random.seed(1337)
    n = 2000
    states = np.zeros((n, 30), dtype=np.float32)
    states[:, 0] = np.random.uniform(0.05, 0.95, n)
    states[:, 1] = np.random.uniform(0.05, 0.95, n)
    states[:, 2] = np.random.uniform(-1.0, 1.0, n)
    states[:, 3] = np.random.uniform(-1.0, 1.0, n)
    states[:, 4] = np.random.uniform(0.02, 0.05, n)
    states[:, 5] = np.random.uniform(0.05, 0.95, n)
    states[:, 6] = np.random.uniform(0.92, 0.95, n)
    states[:, 7] = np.random.uniform(0.05, 0.95, n)
    states[:, 10] = np.random.uniform(0.0, 4.0, n)
    states[:, 11] = np.random.uniform(0.0, 4.0, n)

    expected = probe_contact_features(states, probe_cfg)
    actual = compute_contact_features(states, cfg)

    assert np.array_equal(expected, actual)


def test_edge_cases_semantics():
    cfg = ContactGeometryConfig.from_game_config()
    states = np.zeros((4, 30), dtype=np.float32)

    # Row 0: Ball moving away (vx > 0)
    states[0, 0] = 0.5
    states[0, 1] = 0.5
    states[0, 2] = 0.4
    states[0, 10] = 1.0
    states[0, 11] = 1.0

    # Row 1: Zero horizontal velocity (|vx| <= 1e-6)
    states[1, 0] = 0.5
    states[1, 1] = 0.5
    states[1, 2] = 0.0
    states[1, 10] = 1.0
    states[1, 11] = 1.0

    # Row 2: Ball already past paddle face (gap < 0)
    states[2, 0] = 0.01  # Behind left paddle
    states[2, 1] = 0.5
    states[2, 2] = -0.5
    states[2, 4] = 0.05  # paddle x is ahead of ball
    states[2, 10] = 1.0
    states[2, 11] = 1.0

    # Row 3: Contact beyond horizon (time > 1.5s -> normalized time capped at 1.0, valid = 0)
    states[3, 0] = 0.9  # Far away
    states[3, 1] = 0.5
    states[3, 2] = -0.05  # Slow approach: time = (0.9 - 0.05) * 800 / (0.05 * 1000) = 13.6s
    states[3, 4] = 0.025
    states[3, 10] = 1.0
    states[3, 11] = 1.0

    feats = compute_contact_features(states, cfg)

    # Row 0: incoming=0, own_time=0, own_valid=0, intercept=0, outgoing=1
    assert feats[0, 4] == 0.0
    assert feats[0, 5] == 0.0
    assert feats[0, 6] == 0.0
    assert feats[0, 7] == 0.0
    assert feats[0, 11] == 1.0

    # Row 1: incoming=0, outgoing=0, valid=0
    assert feats[1, 4] == 0.0
    assert feats[1, 6] == 0.0
    assert feats[1, 11] == 0.0

    # Row 2: gap < 0 -> valid=0, own_time=0
    assert feats[2, 6] == 0.0
    assert feats[2, 5] == 0.0

    # Row 3: On course but beyond horizon: incoming=1, own_time=1.0 (capped), own_valid=0, intercept=0
    assert feats[3, 4] == 1.0
    assert feats[3, 5] == 1.0
    assert feats[3, 6] == 0.0
    assert feats[3, 7] == 0.0


def test_three_way_parity_vector_env_agent_and_numpy_batch():
    engine = PhysicsEngine(800, 600)
    engine.ball.position.x = 350.0
    engine.ball.position.y = 280.0
    engine.ball.velocity.x = -220.0
    engine.ball.velocity.y = 130.0
    engine.player1.position.y = 200.0
    engine.player2.position.y = 220.0

    snap = snapshot(engine)
    proc = ObservationProcessor(800, 600)
    obs = to_player1_dqn_observation(proc.process_game_state(engine.get_game_state(), 1), 1)

    # 1. Python DQNAgent
    agent = DQNAgent(hidden_layers=[16], geometry_context=True)
    state_agent = agent._observation_to_state(obs)
    assert state_agent.shape == (46,)
    geom_agent = state_agent[30:46]

    # 2. VectorEnv Numba encode_observation
    out_vector = np.zeros(46, dtype=np.float32)
    encode_observation(
        snap[0],
        snap[1],
        snap[2],
        snap[3],
        1,
        False,
        True,
        out_vector,
    )
    geom_vector = out_vector[30:46]

    # 3. Batch compute_contact_features
    geom_batch = compute_contact_features(state_agent[:30])

    # Three-way exact bitwise equality
    assert np.array_equal(geom_agent, geom_vector)
    assert np.array_equal(geom_agent, geom_batch)


def test_symmetry_side1_and_side2():
    engine = PhysicsEngine(800, 600)
    # Symmetrical configuration
    engine.ball.position.x = 250.0
    engine.ball.position.y = 280.0
    engine.ball.velocity.x = -200.0
    engine.ball.velocity.y = 100.0
    engine.player1.position.y = 220.0
    engine.player2.position.y = 220.0

    proc = ObservationProcessor(800, 600)
    obs1 = to_player1_dqn_observation(proc.process_game_state(engine.get_game_state(), 1), 1)

    # Mirrored game state for side 2
    engine_mirrored = PhysicsEngine(800, 600)
    engine_mirrored.ball.position.x = 800.0 - 250.0
    engine_mirrored.ball.position.y = 280.0
    engine_mirrored.ball.velocity.x = 200.0
    engine_mirrored.ball.velocity.y = 100.0
    engine_mirrored.player1.position.y = 220.0
    engine_mirrored.player2.position.y = 220.0

    obs2 = to_player1_dqn_observation(
        proc.process_game_state(engine_mirrored.get_game_state(), 2), 2
    )

    agent = DQNAgent(hidden_layers=[16], geometry_context=True)
    state1 = agent._observation_to_state(obs1)
    state2 = agent._observation_to_state(obs2)

    np.testing.assert_allclose(state1[30:46], state2[30:46], atol=1e-5)


def test_zero_projection_preserves_initial_q_values_and_rng():
    torch.manual_seed(42)
    base = DQNAgent(hidden_layers=[64, 32], geometry_context=False)
    rng_base = torch.get_rng_state()

    torch.manual_seed(42)
    geom = DQNAgent(hidden_layers=[64, 32], geometry_context=True)
    assert torch.equal(torch.get_rng_state(), rng_base)

    device = next(base.q_network.parameters()).device
    x_30 = torch.randn(10, 30, device=device)
    x_46 = torch.cat([x_30, torch.randn(10, 16, device=device)], dim=-1)

    with torch.no_grad():
        q_base = base.q_network(x_30)
        q_geom = geom.q_network(x_46)

    assert torch.equal(q_base, q_geom)


def test_gradient_flow_to_all_parameters():
    agent = DQNAgent(hidden_layers=[32, 16], geometry_context=True)
    device = next(agent.q_network.parameters()).device
    x = torch.randn(4, 46, device=device)
    out = agent.q_network(x)
    loss = out.sum()
    loss.backward()

    # Geometry context weights must receive non-zero gradients when features are non-zero
    assert agent.q_network.geometry_context_weight.grad is not None
    assert torch.count_nonzero(agent.q_network.geometry_context_weight.grad) > 0

    for layer in agent.q_network.fc_layers:
        assert layer.weight.grad is not None
        assert torch.count_nonzero(layer.weight.grad) > 0
    assert agent.q_network.output_layer.weight.grad is not None


def test_layout_refresh_on_migration():
    # Build a score-context network (33 inputs)
    net = DQNNetwork(input_size=33, hidden_size=64, layer_size=2, score_context=True)
    assert net.layout.score_slice == slice(30, 33)
    assert net.layout.geometry_slice is None
    assert net.layout.state_size == 33

    # Enable geometry context dynamically
    net.enable_geometry_context()
    assert net.geometry_context is True
    assert net.layout.score_slice == slice(30, 33)
    assert net.layout.geometry_slice == slice(33, 49)
    assert net.layout.state_size == 49
    assert net.geometry_context_weight.shape == (net.fc_layers[0].out_features, 16)


def test_v6_checkpoint_rejection_and_v7_roundtrip(tmp_path):
    # 1. Verify v6 rejection
    v6_metadata = {
        "schema_version": 6,
        "state_schema": "magic_pong.dqn_state.v6",
        "model_type": "magic_pong.dqn",
        "observation_frame": "player1_dqn_frame",
        "action_frame": "player1_dqn_frame",
        "state_size": 34,
        "action_size": 9,
    }
    hp = {
        "state_size": 34,
        "action_size": 9,
        "lr": 0.001,
        "gamma": 0.99,
        "epsilon_min": 0.05,
        "epsilon_decay": 0.995,
        "batch_size": 32,
    }
    with pytest.raises(DQNCheckpointError, match="retired"):
        validate_dqn_checkpoint(
            {
                "metadata": v6_metadata,
                "hyperparameters": hp,
                "q_network_state_dict": {},
                "target_network_state_dict": {},
                "optimizer_state_dict": {},
                "epsilon": 0.0,
                "training_step": 0,
                "loss_history": [],
                "reward_history": [],
            }
        )

    # 2. Verify v7 save and roundtrip
    agent = DQNAgent(hidden_layers=[16], geometry_context=True)
    save_path = tmp_path / "v7_model.pth"
    agent.save_model(save_path)

    loaded = DQNAgent()
    loaded.load_model(save_path)
    assert loaded.geometry_context is True
    assert loaded.state_size == 46
    assert loaded.layout.geometry_slice == slice(30, 46)


def test_probe_network_converter_and_exact_q_parity(tmp_path):
    probe_path = Path("models/student_state_distillation_2026-09-21/seed_11/student_10_epoch_15.pt")
    if not probe_path.exists():
        pytest.skip("Historical student_16 checkpoint not found on disk")

    converted_path = tmp_path / "student_16_converted_v7.pth"
    convert_probe_to_dqn_checkpoint(probe_path, converted_path, hidden_layers=[256, 128])

    agent = DQNAgent()
    agent.load_model(converted_path)
    assert agent.geometry_context is True
    assert agent.state_size == 46

    # Compare against ProbeNetwork directly
    probe_weights = torch.load(probe_path, weights_only=True)
    dummy_base = DQNAgent(hidden_layers=[256, 128])
    probe = ProbeNetwork(dummy_base.q_network)
    probe.load_state_dict(probe_weights)
    probe.eval()

    device = next(agent.q_network.parameters()).device
    probe = probe.to(device)

    test_states = torch.randn(50, 46, device=device)
    with torch.no_grad():
        q_probe = probe(test_states)
        q_agent = agent.q_network(test_states)

    diff = (q_probe - q_agent).abs().max().item()
    assert diff < 1e-6
