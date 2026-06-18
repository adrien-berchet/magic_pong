"""Regression tests for DQN AI plumbing boundaries."""

import math
from typing import Any

import numpy as np
import pytest

from magic_pong.ai.agent_adapter import adapt_action_for_world
from magic_pong.ai.agent_adapter import adapt_info_for_agent
from magic_pong.ai.agent_adapter import to_player1_dqn_observation
from magic_pong.ai.environment.factory import EnvironmentFactory
from magic_pong.ai.interface import ObservationProcessor
from magic_pong.ai.interfaces.observation import VectorObservationBuilder
from magic_pong.ai.models.simple_ai import AggressiveAI
from magic_pong.ai.models.simple_ai import DefensiveAI
from magic_pong.ai.models.simple_ai import FollowBallAI
from magic_pong.ai.models.simple_ai import PredictiveAI
from magic_pong.ai.models.simple_ai import TrainingDummyAI
from magic_pong.ai.pretraining import create_pretrainer
from magic_pong.core.entities import Action
from magic_pong.core.entities import Player
from magic_pong.core.game_engine import GameEngine
from magic_pong.core.game_engine import TrainingManager
from magic_pong.core.physics import PhysicsEngine
from magic_pong.utils.config import ai_config
from magic_pong.utils.config import game_config


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


def _canonical_game_state() -> dict[str, Any]:
    return {
        "ball_position": (160.0, 240.0),
        "ball_velocity": (-300.0, 50.0),
        "player1_position": (20.0, 250.0),
        "player2_position": (765.0, 260.0),
        "player1_last_position": (20.0, 250.0),
        "player2_last_position": (760.0, 260.0),
        "player1_paddle_size": 80.0,
        "player2_paddle_size": 90.0,
        "active_bonuses": [(220.0, 200.0, "enlarge_paddle")],
        "rotating_paddles": [(300.0, 320.0, 0.25)],
        "score": [4, 2],
        "time_elapsed": 12.0,
        "field_bounds": (0.0, 800.0, 0.0, 600.0),
    }


def _mirrored_player2_game_state() -> dict[str, Any]:
    return {
        "ball_position": (640.0, 240.0),
        "ball_velocity": (300.0, 50.0),
        "player1_position": (20.0, 260.0),
        "player2_position": (765.0, 250.0),
        "player1_last_position": (25.0, 260.0),
        "player2_last_position": (765.0, 250.0),
        "player1_paddle_size": 90.0,
        "player2_paddle_size": 80.0,
        "active_bonuses": [(580.0, 200.0, "enlarge_paddle")],
        "rotating_paddles": [(500.0, 320.0, math.pi - 0.25)],
        "score": [2, 4],
        "time_elapsed": 12.0,
        "field_bounds": (0.0, 800.0, 0.0, 600.0),
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


class RecordingDQNLikePlayer(Player):
    uses_player1_dqn_frame = True

    def __init__(self) -> None:
        super().__init__("RecordingDQNLikePlayer")
        self.action_observations: list[dict[str, Any] | None] = []
        self.step_observations: list[dict[str, Any]] = []
        self.step_actions: list[Action] = []
        self.step_infos: list[dict[str, Any]] = []

    def get_action(self, observation: dict[str, Any] | None) -> Action:
        self.action_observations.append(observation)
        return Action(move_x=1.0, move_y=0.0)

    def on_step(
        self,
        observation: dict[str, Any],
        action: Action,
        reward: float,
        done: bool,
        info: dict[str, Any],
    ) -> None:
        self.step_observations.append(observation)
        self.step_actions.append(action)
        self.step_infos.append(info)


class RecordingPretrainingAgent:
    def __init__(self) -> None:
        self.observations: list[dict[str, Any]] = []
        self.memory: list[Any] = []
        self.batch_size = 8
        self.epsilon = 0.0
        self.training_step = 0
        self._last_chosen_action: int | None = None

    def set_training_mode(self, training: bool) -> None:
        self.training = training

    def _observation_to_state(self, observation: dict[str, Any]) -> np.ndarray:
        self.observations.append(observation)
        return np.array(
            [
                observation["ball_pos"][0],
                observation["ball_vel"][0],
                observation["player_pos"][0],
            ],
            dtype=np.float32,
        )

    def _extend_state(self, state: np.ndarray, prev_action: int | None) -> np.ndarray:
        return state

    def act(self, state: np.ndarray, training: bool = True) -> int:
        action = 4
        self._last_chosen_action = action
        return action

    def remember(
        self, state: np.ndarray, action: int, reward: float, next_state: np.ndarray, done: bool
    ) -> None:
        self.memory.append((state, action, reward, next_state, done))

    def replay(self) -> None:
        return None


def _raw_heuristic_observation() -> dict[str, Any]:
    return {
        "ball_pos": [480.0, 180.0],
        "player_pos": [160.0, 300.0],
        "ball_vel": [240.0, -120.0],
        "bonuses": [[200.0, 315.0, 1.0]],
        "field_width": 800.0,
        "field_height": 600.0,
    }


def _normalized_heuristic_observation() -> dict[str, Any]:
    return {
        "ball_pos": [480.0 / 800.0, 180.0 / 600.0],
        "player_pos": [160.0 / 800.0, 300.0 / 600.0],
        "ball_vel": [240.0 / game_config.MAX_BALL_SPEED, -120.0 / game_config.MAX_BALL_SPEED],
        "bonuses": [[200.0 / 800.0, 315.0 / 600.0, 1.0]],
        "field_width": 800.0,
        "field_height": 600.0,
    }


def _set_checkpoint_value(checkpoint: dict[str, Any], dotted_key: str, value: Any) -> None:
    target = checkpoint
    parts = dotted_key.split(".")
    for part in parts[:-1]:
        target = target[part]
    target[parts[-1]] = value


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


def test_player2_dqn_adapter_matches_equivalent_player1_normalized_observation(
    monkeypatch: Any,
) -> None:
    monkeypatch.setattr(ai_config, "NORMALIZE_POSITIONS", True)
    processor = ObservationProcessor(field_width=800.0, field_height=600.0)

    canonical = processor.process_game_state(_canonical_game_state(), player_id=1)
    mirrored_player2 = processor.process_game_state(_mirrored_player2_game_state(), player_id=2)
    adapted = to_player1_dqn_observation(mirrored_player2, player_id=2)

    assert adapted["ball_pos"] == pytest.approx(canonical["ball_pos"])
    assert adapted["player_pos"] == pytest.approx(canonical["player_pos"])
    assert adapted["opponent_pos"] == pytest.approx(canonical["opponent_pos"])
    assert adapted["opponent_previous_pos"] == pytest.approx(canonical["opponent_previous_pos"])
    assert adapted["ball_vel"] == pytest.approx(canonical["ball_vel"])
    assert adapted["bonuses"][0] == pytest.approx(canonical["bonuses"][0])
    assert adapted["rotating_paddles"][0] == pytest.approx(canonical["rotating_paddles"][0])
    assert adapted["score_diff"] == canonical["score_diff"]
    assert adapted["field_width"] == canonical["field_width"]
    assert adapted["field_height"] == canonical["field_height"]


def test_game_engine_player2_dqn_mount_uses_canonical_hook_frame(monkeypatch: Any) -> None:
    monkeypatch.setattr(ai_config, "NORMALIZE_POSITIONS", False)
    engine = GameEngine(headless=True)
    player2 = RecordingDQNLikePlayer()
    engine.set_players(None, player2)
    engine.start_game()
    engine.physics_engine.reset_ball(direction=-1, angle=0.0)

    start_x = engine.physics_engine.player2.position.x
    initial_world_observation = engine.ai_environment.observation_processor.process_game_state(
        engine.physics_engine.get_game_state(), 2
    )

    result = engine.update(dt=1.0 / 60.0)

    assert engine.physics_engine.player2.position.x < start_x
    assert player2.action_observations == [
        to_player1_dqn_observation(initial_world_observation, player_id=2)
    ]
    assert player2.step_observations == [
        to_player1_dqn_observation(result["observations"]["player2"], player_id=2)
    ]
    assert player2.step_actions[0].move_x == pytest.approx(1.0)
    assert player2.step_actions[0].move_y == pytest.approx(0.0)


def test_game_engine_player2_real_dqn_mount_mirrors_action_to_world() -> None:
    torch = pytest.importorskip("torch")
    from magic_pong.ai.models.dqn_ai import DQNAgent

    agent = DQNAgent(epsilon=0.0)
    with torch.no_grad():
        for parameter in agent.q_network.parameters():
            parameter.zero_()
        agent.q_network.output_layer.bias[4] = 1.0
    agent.set_training_mode(False)

    engine = GameEngine(headless=True)
    engine.set_players(None, agent)
    engine.start_game()
    start_x = engine.physics_engine.player2.position.x

    engine.update(dt=1.0 / 60.0)

    assert engine.physics_engine.player2.position.x < start_x


def test_player2_dqn_info_filters_contacts_to_canonical_agent_frame() -> None:
    player2 = RecordingDQNLikePlayer()
    info = {
        "events": {
            "paddle_hits": [{"player": 1, "kind": "opponent"}, {"player": 2, "kind": "agent"}],
            "rotating_paddle_hits": [{"player": 2, "kind": "agent_rotate"}],
            "goals": [{"player": 1}],
        },
        "winner": 0,
    }

    adapted = adapt_info_for_agent(info, player_id=2, agent=player2)

    assert adapted is not info
    assert adapted["events"]["paddle_hits"] == [{"player": 1, "kind": "agent"}]
    assert adapted["events"]["rotating_paddle_hits"] == [{"player": 1, "kind": "agent_rotate"}]
    assert adapted["events"]["goals"] == [{"player": 1}]
    assert info["events"]["paddle_hits"][1]["player"] == 2


@pytest.mark.parametrize(
    "ai_player",
    [
        TrainingDummyAI(),
        FollowBallAI(),
        DefensiveAI(),
        AggressiveAI(),
        PredictiveAI(),
    ],
)
def test_heuristic_ai_raw_and_normalized_observations_match_direction(
    ai_player: Any, monkeypatch: Any
) -> None:
    monkeypatch.setattr("magic_pong.ai.models.simple_ai.time.time", lambda: 123.0)

    raw_action = ai_player.get_action(_raw_heuristic_observation())
    normalized_action = ai_player.get_action(_normalized_heuristic_observation())

    assert raw_action.move_x == pytest.approx(normalized_action.move_x)
    assert raw_action.move_y == pytest.approx(normalized_action.move_y)
    assert -1.0 <= raw_action.move_x <= 1.0
    assert -1.0 <= raw_action.move_y <= 1.0
    assert -1.0 <= normalized_action.move_x <= 1.0
    assert -1.0 <= normalized_action.move_y <= 1.0


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


def test_pretraining_player2_uses_canonical_dqn_observation_and_world_action(
    monkeypatch: Any,
) -> None:
    monkeypatch.setattr(ai_config, "NORMALIZE_POSITIONS", False)
    pretrainer = create_pretrainer()
    agent = RecordingPretrainingAgent()

    pretrainer.pretraining_step(agent, player_id=2, num_steps=1)

    initial_observation, next_observation = agent.observations
    assert initial_observation["ball_vel"][0] < 0.0
    assert initial_observation["player_pos"][0] < 400.0
    assert next_observation["player_pos"][0] > initial_observation["player_pos"][0]
    assert agent.memory[0][1] == 4


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

    # With include_prev_action_in_state=True (the default), the replay buffer holds
    # states extended with the previous-action one-hot.  On the first ever step the
    # previous action is None → one-hot is all zeros.  On the next-state side the
    # "previous action" is the action that was just taken (5).
    expected_first_state = agent._extend_state(decision_state, None)
    expected_first_next_state = agent._extend_state(post_step_state, 5)

    action = agent.get_action(decision_obs, explore=False)
    agent.on_step(post_step_obs, action, reward=1.25, done=False, info={})

    first_transition = agent.memory.buffer[0]
    np.testing.assert_array_equal(first_transition.state, expected_first_state)
    np.testing.assert_array_equal(first_transition.next_state, expected_first_next_state)
    assert first_transition.action == 5
    assert first_transition.done is False

    terminal_obs = _dqn_observation(ball_x=0.2, ball_vx=-0.2, player_y=0.45)
    terminal_action = agent.get_action(post_step_obs, explore=False)
    agent.on_step(terminal_obs, terminal_action, reward=-1.0, done=True, info={})

    assert agent.memory.buffer[-1].done is True
    assert agent.last_state is None
    assert agent.last_action is None


def test_dqn_checkpoint_valid_metadata_loads_and_resets_pending_buffers(tmp_path: Any) -> None:
    pytest.importorskip("torch")
    from magic_pong.ai.models.dqn_ai import DQNAgent

    model_path = tmp_path / "valid_dqn.pth"
    saved_agent = DQNAgent(epsilon=0.25, batch_size=8, min_replay_size=1000)
    saved_agent.training_step = 7
    saved_agent.loss_history = [0.5]
    saved_agent.reward_history = [1.5]
    saved_agent.save_model(str(model_path))

    loading_agent = DQNAgent(epsilon=1.0, batch_size=8, min_replay_size=1000)
    loading_agent.last_state = np.ones(32, dtype=np.float32)
    loading_agent.last_action = 3
    loading_agent.episode_buffer.append({"state": "pending"})
    loading_agent.episode_rewards.append(1.0)

    loading_agent.load_model(str(model_path))

    assert loading_agent.epsilon == pytest.approx(0.25)
    assert loading_agent.training_step == 7
    assert loading_agent.loss_history == [0.5]
    assert loading_agent.reward_history == [1.5]
    assert loading_agent.last_state is None
    assert loading_agent.last_action is None
    assert loading_agent.episode_buffer == []
    assert loading_agent.episode_rewards == []


@pytest.mark.parametrize(
    ("dotted_key", "bad_value", "error_match"),
    [
        ("metadata.schema_version", 999, "schema_version"),
        ("metadata.model_type", "other.dqn", "model_type"),
        ("metadata.state_schema", "other_state", "state_schema"),
        ("metadata.observation_frame", "world_frame", "observation_frame"),
        ("metadata.action_frame", "world_frame", "action_frame"),
        ("metadata.state_size", 31, "state_size"),
        ("metadata.action_size", 8, "action_size"),
        ("hyperparameters.state_size", 31, "hyperparameters.state_size"),
        ("hyperparameters.action_size", 8, "hyperparameters.action_size"),
    ],
)
def test_dqn_checkpoint_incompatible_metadata_rejected_before_pending_reset(
    tmp_path: Any, dotted_key: str, bad_value: Any, error_match: str
) -> None:
    torch = pytest.importorskip("torch")
    from magic_pong.ai.models.dqn_ai import DQNAgent
    from magic_pong.ai.models.dqn_checkpoint import DQNCheckpointError
    from magic_pong.ai.models.dqn_checkpoint import safe_torch_load

    valid_path = tmp_path / "valid_dqn.pth"
    invalid_path = tmp_path / "invalid_dqn.pth"
    DQNAgent(epsilon=0.25).save_model(str(valid_path))
    checkpoint = safe_torch_load(str(valid_path), map_location="cpu")
    _set_checkpoint_value(checkpoint, dotted_key, bad_value)
    torch.save(checkpoint, str(invalid_path))

    loading_agent = DQNAgent(epsilon=1.0)
    loading_agent.last_state = np.ones(32, dtype=np.float32)
    loading_agent.last_action = 3
    loading_agent.episode_buffer.append({"state": "pending"})

    with pytest.raises(DQNCheckpointError, match=error_match):
        loading_agent.load_model(str(invalid_path))

    assert loading_agent.epsilon == pytest.approx(1.0)
    assert loading_agent.last_state is not None
    assert loading_agent.last_action == 3
    assert loading_agent.episode_buffer == [{"state": "pending"}]


def test_dqn_checkpoint_missing_metadata_is_rejected_as_legacy(tmp_path: Any) -> None:
    torch = pytest.importorskip("torch")
    from magic_pong.ai.models.dqn_ai import DQNAgent
    from magic_pong.ai.models.dqn_checkpoint import DQNCheckpointError
    from magic_pong.ai.models.dqn_checkpoint import safe_torch_load

    valid_path = tmp_path / "valid_dqn.pth"
    legacy_path = tmp_path / "legacy_dqn.pth"
    DQNAgent(epsilon=0.25).save_model(str(valid_path))
    checkpoint = safe_torch_load(str(valid_path), map_location="cpu")
    checkpoint.pop("metadata")
    torch.save(checkpoint, str(legacy_path))

    loading_agent = DQNAgent(epsilon=1.0)

    with pytest.raises(DQNCheckpointError, match="Legacy DQN checkpoint without metadata"):
        loading_agent.load_model(str(legacy_path))


def test_dqn_checkpoint_malformed_optimizer_state_is_rejected_before_load(
    tmp_path: Any,
) -> None:
    torch = pytest.importorskip("torch")
    from magic_pong.ai.models.dqn_ai import DQNAgent
    from magic_pong.ai.models.dqn_checkpoint import DQNCheckpointError
    from magic_pong.ai.models.dqn_checkpoint import safe_torch_load

    valid_path = tmp_path / "valid_dqn.pth"
    invalid_path = tmp_path / "invalid_optimizer_dqn.pth"
    DQNAgent(epsilon=0.25).save_model(str(valid_path))
    checkpoint = safe_torch_load(str(valid_path), map_location="cpu")
    checkpoint["optimizer_state_dict"] = {"state": {}, "param_groups": []}
    torch.save(checkpoint, str(invalid_path))

    loading_agent = DQNAgent(epsilon=1.0)
    loading_agent.last_state = np.ones(32, dtype=np.float32)
    loading_agent.last_action = 3

    with pytest.raises(DQNCheckpointError, match="expected one parameter group"):
        loading_agent.load_model(str(invalid_path))

    assert loading_agent.epsilon == pytest.approx(1.0)
    assert loading_agent.last_state is not None
    assert loading_agent.last_action == 3


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


def test_strategic_rally_reward_ignores_opponent_contact_events() -> None:
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
    observations[3]["events"] = {"paddle_hits": [{"player": 2}]}

    strategic_rewards = calculator.calculate_strategic_reward(rewards, observations, actions)

    assert strategic_rewards == pytest.approx([0.0] * 5)
