"""Geometry-context observations, zero-projection migration, and checkpoint contract tests."""

import copy
import io

import numpy as np
import pytest
import torch

from magic_pong.ai.agent_adapter import to_player1_dqn_observation
from magic_pong.ai.geometry import GEOMETRY_CONTEXT_HORIZON
from magic_pong.ai.geometry import compute_geometry_features
from magic_pong.ai.geometry import get_state_layout
from magic_pong.ai.interface import ObservationProcessor
from magic_pong.ai.models.dqn_ai import DQNAgent
from magic_pong.ai.models.dqn_checkpoint import GEOMETRY_CONTEXT_SCHEMA_VERSION
from magic_pong.ai.models.dqn_checkpoint import GEOMETRY_CONTEXT_STATE_SCHEMA
from magic_pong.ai.models.dqn_checkpoint import DQNCheckpointError
from magic_pong.ai.models.dqn_checkpoint import validate_dqn_checkpoint
from magic_pong.core.entities import Ball
from magic_pong.core.physics import PhysicsEngine
from magic_pong.utils.config import ai_config_tmp
from magic_pong.utils.config import game_config


def make_observation(
    side: int = 1,
    ball_pos=(400.0, 300.0),
    ball_vel=(-250.0, 100.0),
    player_pos=(20.0, 250.0),
    player_prev_pos=(20.0, 245.0),
    opponent_pos=(760.0, 250.0),
    opponent_prev_pos=(760.0, 255.0),
    paddle_height=100.0,
    field_width=800.0,
    field_height=600.0,
    frame_dt=1.0 / 60.0,
):
    raw = {
        "positions_normalized": False,
        "ball_pos": list(ball_pos),
        "ball_vel": list(ball_vel),
        "player_pos": list(player_pos),
        "player_previous_pos": list(player_prev_pos),
        "opponent_pos": list(opponent_pos),
        "opponent_previous_pos": list(opponent_prev_pos),
        "player_paddle_size": paddle_height,
        "opponent_paddle_size": paddle_height,
        "field_width": field_width,
        "field_height": field_height,
        "max_ball_speed": 1000.0,
        "frame_dt": frame_dt,
        "bonuses": [],
        "rotating_paddles": [],
        "score_diff": 0.0,
        "time_elapsed": 10.0,
    }
    return to_player1_dqn_observation(raw, side)


@pytest.mark.parametrize("side", [1, 2])
@pytest.mark.parametrize("normalized", [False, True])
def test_geometry_context_is_side_relative_and_paddle_centered(side, normalized):
    with ai_config_tmp(NORMALIZE_POSITIONS=normalized):
        engine = PhysicsEngine(800, 600)
        # Move ball towards side 1 (vx < 0) or side 2 (vx > 0)
        vx = -250.0 if side == 1 else 250.0
        engine.ball = Ball(400.0, 300.0, vx, 80.0)
        engine.player1.position.y = 200.0
        engine.player1.prev_position.y = 195.0
        engine.player2.position.y = 220.0
        engine.player2.prev_position.y = 225.0

        obs_proc = ObservationProcessor(800, 600)
        obs = to_player1_dqn_observation(
            obs_proc.process_game_state(engine.get_game_state(), side), side
        )

    base_agent = DQNAgent(hidden_layers=[16], geometry_context=False)
    geom_agent = DQNAgent(hidden_layers=[16], geometry_context=True)

    base_state = base_agent._observation_to_state(obs)
    geom_state = geom_agent._observation_to_state(obs)

    assert base_state.shape == (30,)
    assert geom_state.shape == (34,)
    np.testing.assert_array_equal(geom_state[:30], base_state)

    # In canonical player-1 frame, ball is incoming (approaching) towards own paddle
    features = geom_state[30:34]
    pred_y, tti, own_vel, opp_vel = features

    assert 0.0 <= pred_y <= 1.0
    assert 0.0 < tti <= 1.0
    # Check own paddle velocity sign matches setup:
    # player1 moved down (200 - 195 = +5), player2 moved up (220 - 225 = -5)
    if side == 1:
        assert own_vel > 0.0
        assert opp_vel < 0.0
    else:
        assert own_vel < 0.0
        assert opp_vel > 0.0


def test_sentinel_when_ball_not_approaching():
    # Ball moving away (vx > 0 in player 1 canonical frame)
    obs = make_observation(
        side=1,
        ball_pos=(400.0, 300.0),
        ball_vel=(250.0, 50.0),  # moving right (away from left paddle)
        player_pos=(20.0, 200.0),
        player_prev_pos=(20.0, 200.0),
        paddle_height=120.0,
        field_height=600.0,
    )
    agent = DQNAgent(hidden_layers=[8], geometry_context=True)
    state = agent._observation_to_state(obs)
    pred_y, tti, own_vel, opp_vel = state[30:34]

    # Paddle center is (200 + 120 / 2) / 600 = 260 / 600
    expected_center = 260.0 / 600.0
    assert pred_y == pytest.approx(expected_center, abs=1e-6)
    assert tti == 1.0
    assert own_vel == pytest.approx(0.0, abs=1e-6)


def test_zero_projection_preserves_rng_and_exact_initial_q_values():
    torch.manual_seed(999)
    old = DQNAgent(hidden_layers=[32, 16], geometry_context=False)
    rng = torch.get_rng_state()

    torch.manual_seed(999)
    new = DQNAgent(hidden_layers=[32, 16], geometry_context=True)
    assert torch.equal(torch.get_rng_state(), rng)

    obs_list = [
        make_observation(side=1, ball_pos=(x, y), ball_vel=(vx, vy))
        for x in (200.0, 400.0, 600.0)
        for y in (150.0, 350.0)
        for vx in (-300.0, 300.0)
        for vy in (-100.0, 100.0)
    ]
    old_x = torch.tensor(np.array([old._observation_to_state(o) for o in obs_list]))
    new_x = torch.tensor(np.array([new._observation_to_state(o) for o in obs_list]))

    old.q_network.cpu()
    new.q_network.cpu()

    # Exact bitwise match on initial predictions
    assert torch.equal(old.q_network(old_x), new.q_network(new_x))

    # Weight perturbation changes predictions
    new.q_network.geometry_context_weight.data.fill_(0.02)
    assert not torch.equal(old.q_network(old_x), new.q_network(new_x))


def test_context_combinations_and_slice_ordering():
    layout_base = get_state_layout(False, False)
    assert layout_base.state_size == 30
    assert layout_base.input_blocks == ("base",)

    layout_score = get_state_layout(True, False)
    assert layout_score.state_size == 33
    assert layout_score.input_blocks == ("base", "score_context")
    assert layout_score.score_slice == slice(30, 33)

    layout_geom = get_state_layout(False, True)
    assert layout_geom.state_size == 34
    assert layout_geom.input_blocks == ("base", "geometry_context")
    assert layout_geom.geometry_slice == slice(30, 34)

    layout_both = get_state_layout(True, True)
    assert layout_both.state_size == 37
    assert layout_both.input_blocks == ("base", "score_context", "geometry_context")
    assert layout_both.score_slice == slice(30, 33)
    assert layout_both.geometry_slice == slice(33, 37)


def test_checkpoint_roundtrip_and_upgrades(tmp_path):
    # 1. Base agent (30 inputs)
    base = DQNAgent(hidden_layers=[16, 8])
    base_path = tmp_path / "base.pth"
    base.save_model(base_path)

    # 2. Upgrade base -> geometry_context (34 inputs)
    migrated = DQNAgent()
    migrated.load_model(base_path)
    migrated.enable_geometry_context()
    assert migrated.geometry_context and migrated.state_size == 34
    assert torch.count_nonzero(migrated.q_network.geometry_context_weight) == 0

    geom_path = tmp_path / "geom.pth"
    migrated.save_model(geom_path)

    # 3. Reload v6 checkpoint
    restored = DQNAgent()
    restored.load_model(geom_path)
    assert restored.geometry_context and restored.state_size == 34
    val = validate_dqn_checkpoint(torch.load(geom_path, weights_only=True, map_location="cpu"))
    assert val.valid
    assert val.metadata["schema_version"] == GEOMETRY_CONTEXT_SCHEMA_VERSION
    assert val.metadata["state_schema"] == GEOMETRY_CONTEXT_STATE_SCHEMA
    assert val.metadata["input_blocks"] == ["base", "geometry_context"]

    # 4. Upgrade score_context (33 inputs) -> full context (37 inputs)
    score_agent = DQNAgent(hidden_layers=[16, 8], score_context=True)
    score_path = tmp_path / "score.pth"
    score_agent.save_model(score_path)

    full_agent = DQNAgent()
    full_agent.load_model(score_path)
    full_agent.enable_geometry_context()
    assert full_agent.score_context and full_agent.geometry_context
    assert full_agent.state_size == 37
    assert torch.count_nonzero(full_agent.q_network.geometry_context_weight) == 0

    full_path = tmp_path / "full.pth"
    full_agent.save_model(full_path)
    val_full = validate_dqn_checkpoint(torch.load(full_path, weights_only=True, map_location="cpu"))
    assert val_full.valid
    assert val_full.metadata["schema_version"] == GEOMETRY_CONTEXT_SCHEMA_VERSION
    assert val_full.metadata["input_blocks"] == ["base", "score_context", "geometry_context"]


def test_missing_or_non_finite_context_fails():
    obs = make_observation()
    del obs["player_previous_pos"]
    # Base agent succeeds without player_previous_pos
    DQNAgent()._observation_to_state(obs)
    # Geometry context agent fails loudly
    with pytest.raises(ValueError, match="Missing required geometry context key"):
        DQNAgent(geometry_context=True)._observation_to_state(obs)

    obs_nan = make_observation(ball_vel=(float("nan"), 100.0))
    with pytest.raises(ValueError, match="Non-finite values"):
        DQNAgent(geometry_context=True)._observation_to_state(obs_nan)


def test_validation_rejects_corrupted_v6_metadata():
    stream = io.BytesIO()
    DQNAgent(geometry_context=True).save_model(stream)
    stream.seek(0)
    checkpoint = torch.load(stream, weights_only=True, map_location="cpu")
    validate_dqn_checkpoint(checkpoint)

    # 1. Invalid input_blocks order or elements
    bad1 = copy.deepcopy(checkpoint)
    bad1["metadata"]["input_blocks"] = ["geometry_context", "base"]
    with pytest.raises(DQNCheckpointError, match="input_blocks"):
        validate_dqn_checkpoint(bad1)

    # 2. Mismatched horizon
    bad2 = copy.deepcopy(checkpoint)
    bad2["metadata"]["geometry_context_horizon"] = 999.0
    with pytest.raises(DQNCheckpointError, match="geometry_context_horizon"):
        validate_dqn_checkpoint(bad2)

    # 3. Missing weight parameter
    bad3 = copy.deepcopy(checkpoint)
    del bad3["q_network_state_dict"]["geometry_context_weight"]
    with pytest.raises(DQNCheckpointError, match="missing network weights"):
        validate_dqn_checkpoint(bad3)


def test_cross_check_against_physics_engine_reflection():
    """Verify reflection formula against actual PhysicsEngine simulation."""
    physics = PhysicsEngine(800.0, 600.0)
    # Move paddles far away so ball does not collide with them before contact face
    physics.player1.position.y = -500.0
    physics.player2.position.y = -500.0

    # Launch ball from center towards left wall with multiple bounces
    start_x, start_y = 500.0, 300.0
    vx, vy = -300.0, 200.0
    physics.ball = Ball(start_x, start_y, vx, vy)

    contact_x = game_config.PADDLE_MARGIN + game_config.PADDLE_WIDTH + game_config.BALL_RADIUS
    features = compute_geometry_features(
        ball_x_px=start_x,
        ball_y_px=start_y,
        vx_px=vx,
        vy_px=vy,
        own_x_px=game_config.PADDLE_MARGIN,
        own_y_px=250.0,
        own_prev_y_px=250.0,
        opp_y_px=250.0,
        opp_prev_y_px=250.0,
        own_height_px=100.0,
        field_width=800.0,
        field_height=600.0,
        paddle_width=game_config.PADDLE_WIDTH,
        ball_radius=game_config.BALL_RADIUS,
        paddle_speed=game_config.PADDLE_SPEED,
        frame_dt=1.0 / 60.0,
        horizon_seconds=GEOMETRY_CONTEXT_HORIZON,
    )
    predicted_y_norm, arrival_norm, _, _ = features
    predicted_y_px = predicted_y_norm * 600.0

    # Step physics with tiny dt to find exact y when crossing contact_x
    dt = 0.0001
    while physics.ball.position.x > contact_x:
        physics.ball.update(dt)
        # Handle wall bounces like PhysicsEngine does
        if physics.ball.position.y - physics.ball.radius < 0:
            physics.ball.position.y = physics.ball.radius
            physics.ball.velocity.y *= -1
        elif physics.ball.position.y + physics.ball.radius > 600:
            physics.ball.position.y = 600 - physics.ball.radius
            physics.ball.velocity.y *= -1

    actual_y_px = physics.ball.position.y
    assert actual_y_px == pytest.approx(predicted_y_px, abs=1.0)


def test_gradient_step_updates_geometry_weights():
    agent = DQNAgent(hidden_layers=[16], geometry_context=True)
    obs = make_observation()
    state = agent._observation_to_state(obs)

    initial_weight = agent.q_network.geometry_context_weight.clone()
    assert torch.count_nonzero(initial_weight) == 0

    x = torch.tensor(state[None, :], device=agent.q_network.fc_layers[0].weight.device)
    loss = agent.q_network(x).sum()
    agent.optimizer.zero_grad()
    loss.backward()
    agent.optimizer.step()

    assert not torch.equal(agent.q_network.geometry_context_weight, initial_weight)
    assert torch.count_nonzero(agent.q_network.geometry_context_weight) > 0


def test_gui_and_scheduled_training_geometry_context_integration(tmp_path):
    import train_optimized
    from magic_pong.gui.game_app import MagicPongApp

    # 1. Check MagicPongApp GUI loading
    path = tmp_path / "geometry.pth"
    initial = DQNAgent(geometry_context=True, hidden_layers=[8])
    initial.save_model(path)
    app = MagicPongApp.__new__(MagicPongApp)
    info = app._load_model_info(str(path))
    assert info["valid"] is True
    app.model_info_by_path = {str(path): info}
    playable = app._create_trained_ai(str(path))
    assert playable is not None
    assert playable.geometry_context is True
    assert playable.state_size == 34
    assert not playable.training_enabled

    # 2. Check train_optimized CLI with --frame_geometry_context
    directory = tmp_path / "run"
    flags = [
        "--mode",
        "frame_pipeline",
        "--output_dir",
        str(directory),
        "--load_model",
        str(path),
        "--pipeline_curriculum_frames",
        "24",
        "--pipeline_anneal_frames",
        "40",
        "--pipeline_selfplay_frames",
        "32",
        "--memory_size",
        "8",
        "--batch_size",
        "4",
        "--min_replay_size",
        "4",
        "--max_steps_per_episode",
        "8",
        "--validation_opponents",
        "dummy",
        "--validation_seeds",
        "101",
        "--validation_max_steps",
        "8",
        "--validation_max_score",
        "1",
        "--frame_validation_interval",
        "24",
        "--frame_regression_patience",
        "0",
        "--frame_geometry_context",
        "--frame_selfplay_opponent",
        "follow_ball",
    ]
    result = train_optimized.main(flags)
    assert result["status"] == "complete"
    assert result["config"]["geometry_context"] is True
    assert result["config"]["score_context"] is False

    bundle = torch.load(directory / "latest.training.pt", map_location="cpu", weights_only=False)
    assert bundle["replay"]["states"].shape[1] == 34
    assert bundle["config"]["geometry_context"] is True


def test_vector_training_geometry_context_eval():
    from magic_pong.ai.vector_training import evaluate_network

    agent = DQNAgent(hidden_layers=[16], geometry_context=True)
    device = next(agent.q_network.parameters()).device
    report = evaluate_network(
        agent.q_network,
        device,
        opponents=("dummy",),
        seeds=1,
        sides=(1,),
        bonus_modes=(False,),
        match_target=1,
        max_frames=100,
        geometry_context=True,
    )
    assert "rank" in report
    assert "matches" in report
