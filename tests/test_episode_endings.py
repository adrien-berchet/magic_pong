"""Episode ending contract tests for termination versus truncation."""

from __future__ import annotations

from typing import Any

import numpy as np
import pytest

from magic_pong.ai.interface import GameEnvironment
from magic_pong.ai.models.dqn_ai import DQNAgent
from magic_pong.core.entities import Action
from magic_pong.core.entities import Player
from magic_pong.core.game_engine import GameEngine
from magic_pong.core.game_engine import TrainingManager
from magic_pong.core.physics import PhysicsEngine
from magic_pong.utils.config import game_config


def _dqn_observation(*, ball_x: float, player_y: float) -> dict[str, Any]:
    return {
        "ball_pos": [ball_x, 0.5],
        "player_pos": [0.05, player_y],
        "opponent_pos": [0.95, 0.5],
        "opponent_previous_pos": [0.95, 0.5],
        "field_width": 800.0,
        "field_height": 600.0,
        "positions_normalized": True,
        "ball_vel": [-0.4, 0.0],
        "player_paddle_size": 80.0,
        "opponent_paddle_size": 80.0,
        "bonuses": [],
        "rotating_paddles": [],
        "score_diff": 0,
        "time_elapsed": 0.0,
    }


class FixedActionPlayer(Player):
    def __init__(self) -> None:
        self.ended = 0

    def get_action(self, observation: dict[str, Any]) -> Action:
        return Action(0.0, 0.0)

    def on_episode_end(self) -> None:
        self.ended += 1


def test_environment_timeout_sets_truncated_without_terminal() -> None:
    env = GameEnvironment(PhysicsEngine(game_config.FIELD_WIDTH, game_config.FIELD_HEIGHT))
    env.max_steps = 1
    initial_obs, _ = env.reset()

    obs, _, _, _, done, info = env.step(Action(0, 0), Action(0, 0))

    assert done is True
    assert info["terminated"] is False
    assert info["truncated"] is True
    assert info["winner"] == 0
    assert info["step_count"] == 1
    assert obs["ball_pos"] != initial_obs["ball_pos"]


def test_true_match_end_wins_over_same_frame_cutoff() -> None:
    physics = PhysicsEngine(game_config.FIELD_WIDTH, game_config.FIELD_HEIGHT)
    env = GameEnvironment(physics)
    env.max_steps = 1
    env.reset()

    def score_on_update(dt: float, action1: Action, action2: Action) -> dict[str, list]:
        physics.score = [game_config.MAX_SCORE, 0]
        return {"goals": [{"player": 1}]}

    physics.update = score_on_update  # type: ignore[method-assign]

    _, _, _, _, done, info = env.step(Action(0, 0), Action(0, 0))

    assert done is True
    assert info["terminated"] is True
    assert info["truncated"] is False
    assert info["winner"] == 1


def test_training_manager_reports_truncation() -> None:
    manager = TrainingManager(headless=True)
    try:
        stats = manager.train_episode(FixedActionPlayer(), FixedActionPlayer(), max_steps=1)
    finally:
        manager.cleanup()

    assert stats["winner"] == 0
    assert stats["terminated"] is False
    assert stats["truncated"] is True


def test_dqn_bootstraps_on_truncation_but_not_termination(
    monkeypatch: pytest.MonkeyPatch,
) -> None:
    agent = DQNAgent(epsilon=0.0, batch_size=8, min_replay_size=1000)
    monkeypatch.setattr(agent, "act", lambda *_args, **_kwargs: 3)

    decision_obs = _dqn_observation(ball_x=0.5, player_y=0.4)
    timeout_obs = _dqn_observation(ball_x=0.45, player_y=0.42)
    agent.get_action(decision_obs, explore=False)
    agent.on_step(
        timeout_obs,
        Action(0, 0),
        reward=0.0,
        done=True,
        info={"terminated": False, "truncated": True},
    )

    timeout_transition = agent.memory.buffer[-1]
    assert timeout_transition.done is False
    np.testing.assert_array_equal(
        timeout_transition.next_state,
        agent._observation_to_state(timeout_obs),
    )

    terminal_obs = _dqn_observation(ball_x=0.4, player_y=0.45)
    agent.get_action(timeout_obs, explore=False)
    agent.on_step(
        terminal_obs,
        Action(0, 0),
        reward=1.0,
        done=True,
        info={"terminated": True, "truncated": False},
    )

    assert agent.memory.buffer[-1].done is True
    assert agent.last_state is None
    assert agent.last_action is None


@pytest.mark.parametrize(
    ("done", "info", "expected_done"),
    [
        (True, {"terminated": False, "truncated": True}, False),
        (True, {"terminated": True, "truncated": False}, True),
        (True, {"truncated": False}, True),
        (True, {}, True),
        (False, {"truncated": False}, False),
    ],
)
def test_dqn_done_semantics_for_partial_and_legacy_info(
    monkeypatch: pytest.MonkeyPatch,
    done: bool,
    info: dict[str, bool],
    expected_done: bool,
) -> None:
    agent = DQNAgent(epsilon=0.0, batch_size=8, min_replay_size=1000)
    monkeypatch.setattr(agent, "act", lambda *_args, **_kwargs: 1)
    agent.get_action(_dqn_observation(ball_x=0.5, player_y=0.4), explore=False)

    agent.on_step(
        _dqn_observation(ball_x=0.45, player_y=0.42),
        Action(0, 0),
        reward=0.0,
        done=done,
        info=info,
    )

    assert agent.memory.buffer[-1].done is expected_done


def test_episode_start_clears_pending_dqn_transition(
    monkeypatch: pytest.MonkeyPatch,
) -> None:
    agent = DQNAgent(epsilon=0.0, batch_size=8, min_replay_size=1000)
    monkeypatch.setattr(agent, "act", lambda *_args, **_kwargs: 2)
    agent.get_action(_dqn_observation(ball_x=0.5, player_y=0.4), explore=False)
    assert agent.last_state is not None
    assert agent.last_action == 2

    engine = GameEngine(headless=True)
    engine.set_players(agent, FixedActionPlayer())
    engine.start_game()

    assert agent.last_state is None
    assert agent.last_action is None
    engine.stop_game()


@pytest.mark.parametrize("terminal_on_goal", [False, True])
@pytest.mark.parametrize("scorer", [1, 2])
@pytest.mark.parametrize(
    "ending", ["ongoing", "terminated", "truncated", "no_goal", "truncated_no_goal"]
)
def test_goal_terminal_mask_is_learner_only(terminal_on_goal, scorer, ending):
    import copy

    agent = DQNAgent(terminal_on_goal=terminal_on_goal, epsilon=0.0)
    observation = _dqn_observation(ball_x=0.5, player_y=0.4)
    info = {
        "events": {
            "goals": [] if ending in ("no_goal", "truncated_no_goal") else [{"player": scorer}]
        },
        "terminated": ending == "terminated",
        "truncated": ending in ("truncated", "truncated_no_goal"),
    }
    original = copy.deepcopy(info)
    agent.get_action(observation, explore=False)
    agent.on_step(
        observation,
        Action(0, 0),
        0.0,
        ending in ("terminated", "truncated", "truncated_no_goal"),
        info,
    )
    assert agent.memory.buffer[-1].done is (
        ending == "terminated"
        or (terminal_on_goal and ending not in ("no_goal", "truncated_no_goal"))
    )
    assert info == original
    # A learning ending must still accept the following serve as a fresh decision.
    agent.get_action(observation, explore=False)
    agent.on_step(observation, Action(0, 0), 1.0, False, {"events": {"goals": []}})
    assert agent.memory.buffer[-1].done is False  # reward alone never triggers the mask
    assert len(agent.memory) == 2


def test_goal_terminal_setting_roundtrips_and_legacy_defaults_false(tmp_path):
    import torch

    path = tmp_path / "model.pth"
    DQNAgent(terminal_on_goal=True).save_model(path)
    loaded = DQNAgent()
    loaded.load_model(path)
    assert loaded.terminal_on_goal is True
    checkpoint = torch.load(path, weights_only=True, map_location="cpu")
    del checkpoint["hyperparameters"]["terminal_on_goal"]
    torch.save(checkpoint, path)
    loaded.load_model(path)
    assert loaded.terminal_on_goal is False
