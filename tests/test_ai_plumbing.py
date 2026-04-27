"""Regression tests for DQN AI plumbing boundaries."""

import math
from typing import Any

import numpy as np
import pytest

from magic_pong.ai.agent_adapter import adapt_action_for_world
from magic_pong.ai.agent_adapter import to_player1_dqn_observation
from magic_pong.ai.environment.factory import EnvironmentFactory
from magic_pong.ai.interfaces.observation import VectorObservationBuilder
from magic_pong.core.entities import Action
from magic_pong.core.entities import Player
from magic_pong.core.game_engine import TrainingManager
from magic_pong.core.physics import PhysicsEngine
from magic_pong.utils.config import ai_config


def _canonical_observation() -> dict[str, Any]:
    return {
        "ball_pos": [160.0, 240.0],
        "player_pos": [20.0, 250.0],
        "opponent_pos": [765.0, 260.0],
        "opponent_previous_pos": [760.0, 260.0],
        "field_width": 800.0,
        "field_height": 600.0,
        "ball_vel": [-3.0, 0.5],
        "player_paddle_size": 80.0,
        "opponent_paddle_size": 90.0,
        "bonuses": [[220.0, 200.0, 1.0]],
        "rotating_paddles": [[300.0, 320.0, 0.25]],
        "score_diff": 2,
        "time_elapsed": 12.0,
    }


def _mirrored_player2_observation() -> dict[str, Any]:
    return {
        "ball_pos": [640.0, 240.0],
        "player_pos": [765.0, 250.0],
        "opponent_pos": [20.0, 260.0],
        "opponent_previous_pos": [25.0, 260.0],
        "field_width": 800.0,
        "field_height": 600.0,
        "ball_vel": [3.0, 0.5],
        "player_paddle_size": 80.0,
        "opponent_paddle_size": 90.0,
        "bonuses": [[580.0, 200.0, 1.0]],
        "rotating_paddles": [[500.0, 320.0, math.pi - (4 * math.pi + 0.25)]],
        "score_diff": 2,
        "time_elapsed": 12.0,
    }


def _dqn_observation(ball_x: float, ball_vx: float, player_y: float) -> dict[str, Any]:
    return {
        "ball_pos": [ball_x, 0.5],
        "ball_vel": [ball_vx, 0.1],
        "player_pos": [0.05, player_y],
        "opponent_pos": [0.95, 0.55],
        "opponent_previous_pos": [0.95, 0.56],
        "field_width": 1.0,
        "field_height": 1.0,
        "player_paddle_size": 1.0,
        "opponent_paddle_size": 1.0,
        "score_diff": 0.0,
        "time_elapsed": 0.0,
        "bonuses": [],
        "rotating_paddles": [],
    }


class RecordingPlayer(Player):
    def __init__(self) -> None:
        super().__init__("RecordingPlayer")
        self.done_flags: list[bool] = []

    def get_action(self, observation: dict[str, Any] | None) -> Action:
        return Action(move_x=0.0, move_y=0.0)

    def on_step(
        self,
        observation: dict[str, Any],
        action: Action,
        reward: float,
        done: bool,
        info: dict[str, Any],
    ) -> None:
        self.done_flags.append(done)


def test_player2_dqn_adapter_matches_equivalent_player1_observation(
    monkeypatch: Any,
) -> None:
    monkeypatch.setattr(ai_config, "NORMALIZE_POSITIONS", False)

    canonical = _canonical_observation()
    assert to_player1_dqn_observation(canonical, player_id=1) is canonical

    adapted = to_player1_dqn_observation(_mirrored_player2_observation(), player_id=2)

    assert adapted == canonical

    player1_action = Action(move_x=-0.75, move_y=0.5)
    player2_world_action = adapt_action_for_world(player1_action, player_id=2, agent=None)
    assert player2_world_action.move_x == pytest.approx(-0.75)

    class DQNLike:
        uses_player1_dqn_frame = True

    identity_action = adapt_action_for_world(player1_action, player_id=1, agent=DQNLike())
    assert identity_action is player1_action

    mirrored_action = adapt_action_for_world(player1_action, player_id=2, agent=DQNLike())
    assert mirrored_action.move_x == pytest.approx(0.75)
    assert mirrored_action.move_y == pytest.approx(0.5)


def test_training_manager_max_steps_drives_terminal_step() -> None:
    manager = TrainingManager(headless=True)
    player1 = RecordingPlayer()
    player2 = RecordingPlayer()

    stats = manager.train_episode(player1, player2, max_steps=1)

    assert stats["steps"] == 1
    assert player1.done_flags == [True]
    assert player2.done_flags == [True]


def test_vector_observation_builder_uses_shared_left_frame_semantics() -> None:
    builder = VectorObservationBuilder(field_width=800.0, field_height=600.0)
    player1_state = {
        "ball_position": (160.0, 240.0),
        "ball_velocity": (-300.0, 50.0),
        "player1_position": (20.0, 250.0),
        "player2_position": (765.0, 260.0),
    }
    player2_state = {
        "ball_position": (640.0, 240.0),
        "ball_velocity": (300.0, 50.0),
        "player1_position": (20.0, 260.0),
        "player2_position": (765.0, 250.0),
    }

    np.testing.assert_array_equal(
        builder.build_observation(player1_state, player_id=1),
        builder.build_observation(player2_state, player_id=2),
    )


def test_environment_wrapper_mirrors_player2_agent_action_to_world() -> None:
    physics = PhysicsEngine(800.0, 600.0)
    env = EnvironmentFactory.create_default(physics, headless=True, player_id=2)
    start_x = physics.player2.position.x

    env.step(Action(move_x=1.0, move_y=0.0))

    assert physics.player2.position.x < start_x


def test_dqn_transition_uses_decision_time_state_and_stores_terminal(
    monkeypatch: Any,
) -> None:
    pytest.importorskip("torch")
    from magic_pong.ai.models.dqn_ai import DQNAgent

    agent = DQNAgent(
        epsilon=0.0,
        batch_size=8,
        min_replay_size=1000,
        training_mode="step_by_step",
    )
    monkeypatch.setattr(agent, "act", lambda *_args, **_kwargs: 5)

    decision_obs = _dqn_observation(ball_x=0.4, ball_vx=-0.2, player_y=0.3)
    post_step_obs = _dqn_observation(ball_x=0.3, ball_vx=-0.2, player_y=0.4)
    decision_state = agent._observation_to_state(decision_obs)
    post_step_state = agent._observation_to_state(post_step_obs)

    action = agent.get_action(decision_obs, explore=False)
    agent.on_step(post_step_obs, action, reward=1.25, done=False, info={})

    first_transition = agent.memory.buffer[0]
    np.testing.assert_array_equal(first_transition.state, decision_state)
    np.testing.assert_array_equal(first_transition.next_state, post_step_state)
    assert first_transition.action == 5
    assert first_transition.done is False

    terminal_obs = _dqn_observation(ball_x=0.2, ball_vx=-0.2, player_y=0.45)
    terminal_action = agent.get_action(post_step_obs, explore=False)
    agent.on_step(terminal_obs, terminal_action, reward=-1.0, done=True, info={})

    assert agent.memory.buffer[-1].done is True
    assert agent.last_state is None
    assert agent.last_action is None


def test_dqn_eval_mode_is_greedy_and_does_not_copy_target_network() -> None:
    torch = pytest.importorskip("torch")
    from magic_pong.ai.models.dqn_ai import ACTION_MAPPING
    from magic_pong.ai.models.dqn_ai import DQNAgent

    agent = DQNAgent(epsilon=1.0)
    with torch.no_grad():
        for parameter in agent.q_network.parameters():
            parameter.zero_()
        agent.q_network.output_layer.bias[4] = 1.0
        for parameter in agent.target_network.parameters():
            parameter.fill_(0.25)

    target_before = [parameter.detach().clone() for parameter in agent.target_network.parameters()]

    agent.set_training_mode(False)
    agent.set_training_mode(True)

    for before, after in zip(target_before, agent.target_network.parameters(), strict=False):
        assert torch.equal(before, after)

    agent.set_training_mode(False)
    observation = _dqn_observation(ball_x=0.4, ball_vx=-0.2, player_y=0.3)
    actions = [agent.get_action(observation) for _ in range(5)]

    assert actions == [ACTION_MAPPING[4]] * 5


def test_dqn_eval_can_explicitly_explore_without_learning(monkeypatch: Any) -> None:
    pytest.importorskip("torch")
    from magic_pong.ai.models.dqn_ai import ACTION_MAPPING
    from magic_pong.ai.models.dqn_ai import DQNAgent

    agent = DQNAgent(epsilon=1.0, training_mode="step_by_step")
    agent.set_training_mode(False)
    agent.set_exploration_mode(True)
    monkeypatch.setattr("magic_pong.ai.models.dqn_ai.random.random", lambda: 0.0)
    monkeypatch.setattr("magic_pong.ai.models.dqn_ai.random.randrange", lambda _size: 2)

    observation = _dqn_observation(ball_x=0.4, ball_vx=-0.2, player_y=0.3)
    action = agent.get_action(observation)
    agent.on_step(observation, action, reward=1.0, done=False, info={})

    assert action == ACTION_MAPPING[2]
    assert agent.training_enabled is False
    assert len(agent.memory) == 0


def test_dqn_rotating_paddle_angle_uses_unwrapped_radians() -> None:
    pytest.importorskip("torch")
    from magic_pong.ai.models.dqn_ai import DQNAgent

    agent = DQNAgent(epsilon=0.0)
    wrapped = _dqn_observation(ball_x=0.4, ball_vx=-0.2, player_y=0.3)
    unwrapped = _dqn_observation(ball_x=0.4, ball_vx=-0.2, player_y=0.3)
    wrapped["rotating_paddles"] = [[0.5, 0.5, 0.25]]
    unwrapped["rotating_paddles"] = [[0.5, 0.5, 4 * math.pi + 0.25]]

    np.testing.assert_array_equal(
        agent._observation_to_state(wrapped),
        agent._observation_to_state(unwrapped),
    )


def test_tactical_reward_direction_uses_left_side_canonical_frame() -> None:
    pytest.importorskip("torch")
    from magic_pong.ai.models.dqn_ai import HybridRewardCalculator

    calculator = HybridRewardCalculator()
    action = Action(move_x=1.0, move_y=1.0)
    approaching = {
        "ball_pos": [0.12, 0.5],
        "player_pos": [0.05, 0.5],
        "ball_vel": [-0.4, 0.0],
    }
    moving_away = {
        "ball_pos": [0.12, 0.5],
        "player_pos": [0.05, 0.5],
        "ball_vel": [0.4, 0.0],
    }

    assert calculator.calculate_tactical_reward(approaching, action, 0.0) > 0.0
    assert calculator.calculate_tactical_reward(moving_away, action, 0.0) == pytest.approx(0.0)

    raw_approaching = {
        "ball_pos": [96.0, 300.0],
        "player_pos": [20.0, 300.0],
        "ball_vel": [-200.0, 0.0],
        "field_width": 800.0,
        "field_height": 600.0,
    }
    raw_moving_away = {
        "ball_pos": [96.0, 300.0],
        "player_pos": [20.0, 300.0],
        "ball_vel": [200.0, 0.0],
        "field_width": 800.0,
        "field_height": 600.0,
    }

    assert calculator.calculate_tactical_reward(raw_approaching, action, 0.0) > 0.0
    assert calculator.calculate_tactical_reward(raw_moving_away, action, 0.0) == pytest.approx(0.0)


def test_strategic_defensive_reward_uses_left_side_canonical_frame() -> None:
    pytest.importorskip("torch")
    from magic_pong.ai.models.dqn_ai import HybridRewardCalculator

    calculator = HybridRewardCalculator()
    actions = [Action(move_x=0.0, move_y=0.0)] * 9
    rewards = [0.0] * 9
    defensive_observations = [
        {
            "ball_pos": [0.8, 0.4],
            "player_pos": [0.05, 0.5],
            "ball_vel": [0.2, 0.0],
        }
        for _ in range(9)
    ]
    approaching_observations = [
        {
            "ball_pos": [0.2, 0.4],
            "player_pos": [0.05, 0.5],
            "ball_vel": [-0.2, 0.0],
        }
        for _ in range(9)
    ]

    defensive_reward = calculator.calculate_strategic_reward(
        rewards, defensive_observations, actions
    )
    approaching_reward = calculator.calculate_strategic_reward(
        rewards, approaching_observations, actions
    )

    assert defensive_reward[-1] - approaching_reward[-1] == pytest.approx(0.4)

    raw_defensive_observations = [
        {
            "ball_pos": [640.0, 240.0],
            "player_pos": [20.0, 300.0],
            "ball_vel": [200.0, 0.0],
            "field_width": 800.0,
            "field_height": 600.0,
        }
        for _ in range(9)
    ]
    raw_approaching_observations = [
        {
            "ball_pos": [160.0, 240.0],
            "player_pos": [20.0, 300.0],
            "ball_vel": [-200.0, 0.0],
            "field_width": 800.0,
            "field_height": 600.0,
        }
        for _ in range(9)
    ]

    raw_defensive_reward = calculator.calculate_strategic_reward(
        rewards, raw_defensive_observations, actions
    )
    raw_approaching_reward = calculator.calculate_strategic_reward(
        rewards, raw_approaching_observations, actions
    )

    assert raw_defensive_reward[-1] - raw_approaching_reward[-1] == pytest.approx(0.4)


def test_strategic_rally_reward_uses_runtime_event_buckets() -> None:
    pytest.importorskip("torch")
    from magic_pong.ai.models.dqn_ai import HybridRewardCalculator

    calculator = HybridRewardCalculator()
    actions = [Action(move_x=0.0, move_y=0.0)] * 5
    rewards = [0.0] * 5
    observations = [
        {
            "ball_pos": [0.5, 0.5],
            "player_pos": [0.05, 0.5],
            "ball_vel": [-0.2, 0.0],
            "events": {},
        }
        for _ in range(5)
    ]
    observations[3]["events"] = {"paddle_hits": [{"player": 1}]}

    strategic_rewards = calculator.calculate_strategic_reward(rewards, observations, actions)

    assert strategic_rewards[0] > 0.0
