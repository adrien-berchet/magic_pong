"""Behavioral checks for simulator labels and coherent imitation warm starts."""

import random

import numpy as np
import pytest
import torch

from magic_pong.ai.models.dqn_ai import ACTION_MAPPING
from magic_pong.ai.models.dqn_ai import DQNAgent
from magic_pong.ai.physics_teacher import PhysicsTeacher
from magic_pong.ai.pretraining import create_pretrainer
from magic_pong.core.entities import Ball
from magic_pong.core.entities import Bonus
from magic_pong.core.entities import BonusType
from magic_pong.core.entities import Vector2D
from magic_pong.core.physics import PhysicsEngine
from magic_pong.utils.config import game_config


@pytest.fixture
def physics(monkeypatch):
    monkeypatch.setattr(game_config, "BONUSES_ENABLED", False)
    return PhysicsEngine(game_config.FIELD_WIDTH, game_config.FIELD_HEIGHT)


@pytest.mark.parametrize("player_id", [1, 2])
def test_teacher_recovers_toward_guard_when_ball_heads_away(physics, player_id):
    paddle = getattr(physics, f"player{player_id}")
    paddle.position.y = 20
    paddle.prev_position = paddle.position.copy()
    physics.ball = Ball(400, 300, 300 if player_id == 1 else -300, 0)
    decision = PhysicsTeacher(horizon_steps=24, action_steps=6).label(physics, player_id)
    assert ACTION_MAPPING[decision.action].move_y > 0


def test_teacher_prefers_real_save_over_conceding(physics):
    physics.player1.position = Vector2D(30, 250)
    physics.player1.prev_position = physics.player1.position.copy()
    physics.ball = Ball(70, 230, -400, 0)
    decision = PhysicsTeacher(horizon_steps=30, action_steps=6).label(physics, 1)
    assert decision.outcomes[0].goals_against == 1  # Staying misses the paddle.
    assert decision.outcomes[decision.action].paddle_hits > 0
    assert decision.outcomes[decision.action].goals_against == 0


def test_teacher_selects_scoring_contact_against_forward_opponent(physics):
    physics.player1.position = Vector2D(288.6, 94.7)
    physics.player2.position = Vector2D(452.7, 403.0)
    for paddle in (physics.player1, physics.player2):
        paddle.prev_position = paddle.position.copy()
    physics.ball = Ball(366.3, 158.6, -347.9, 0)
    decision = PhysicsTeacher(horizon_steps=90, action_steps=6).label(physics, 1)
    assert decision.outcomes[0].goals_for == 0
    assert decision.outcomes[decision.action].goals_for == 1
    assert decision.outcomes[decision.action].paddle_hits > 0


def test_teacher_considers_actual_bonus_collection(physics, monkeypatch):
    monkeypatch.setattr(game_config, "BONUSES_ENABLED", True)
    physics.player1.position = Vector2D(30, 250)
    physics.player1.prev_position = physics.player1.position.copy()
    physics.ball = Ball(400, 300, 300, 0)
    physics.bonuses = [Bonus(30, 200, BonusType.ENLARGE_PADDLE)]
    decision = PhysicsTeacher(horizon_steps=12, action_steps=6).label(physics, 1)
    assert decision.outcomes[0].bonuses == 0
    assert decision.outcomes[decision.action].bonuses == 1


def test_teacher_rollouts_leave_live_physics_and_randomness_unchanged(physics):
    before = physics.get_game_state()
    random_before = random.getstate()
    first = PhysicsTeacher(horizon_steps=12).label(physics, 2)
    assert physics.get_game_state() == before
    assert random.getstate() == random_before
    second = PhysicsTeacher(horizon_steps=12).label(physics, 2)
    np.testing.assert_array_equal(first.values, second.values)


def test_simulator_pretraining_visits_all_scenarios_and_learns_without_rl_replay():
    agent = DQNAgent(hidden_size=16, layer_size=1, batch_size=4)
    initial = [parameter.detach().clone() for parameter in agent.q_network.parameters()]
    trainer = create_pretrainer(
        mode="simulator",
        teacher_horizon=12,
        teacher_action_steps=2,
        rollout_steps=4,
        learner_probability=1.0,
        supervised_batch_size=4,
        seed=12,
    )
    stats = trainer.run_pretraining_phase(agent, total_steps=24, steps_per_batch=12, verbose=False)
    assert set(stats["scenario_counts"]) == set(trainer.scenario_modes)
    assert stats["learner_decisions"] == 24
    assert stats["physics_steps"] >= 24
    assert stats["supervised_updates"] > 0
    assert len(agent.memory) == 0
    assert agent.training_step == 0
    assert agent.epsilon == 1.0
    assert any(not torch.equal(a, b) for a, b in zip(initial, agent.q_network.parameters()))
    for online, target in zip(agent.q_network.parameters(), agent.target_network.parameters()):
        assert torch.equal(online, target)


def test_labels_follow_one_continuous_learner_trajectory(monkeypatch):
    trainer = create_pretrainer(
        mode="simulator",
        teacher_horizon=6,
        teacher_action_steps=2,
        rollout_steps=120,
        learner_probability=1,
        seed=5,
    )
    agent = DQNAgent(hidden_size=16, layer_size=1)
    visited = []
    label = trainer.teacher.label

    def capture(physics, player_id):
        visited.append(physics.get_game_state())
        return label(physics, player_id)

    monkeypatch.setattr(trainer.teacher, "label", capture)
    trainer.pretraining_step(agent, num_steps=3)
    assert [state["time_elapsed"] for state in visited] == pytest.approx(
        [
            0,
            2 * game_config.GAME_SPEED_MULTIPLIER / game_config.FPS,
            4 * game_config.GAME_SPEED_MULTIPLIER / game_config.FPS,
        ]
    )
    assert len({state["ball_position"] for state in visited}) == 3
    assert trainer.scenario_index == 1


@pytest.mark.parametrize("scenario_index", [4, 5])
def test_scenarios_exercise_small_paddles_and_bonus_objects(scenario_index, monkeypatch):
    trainer = create_pretrainer(mode="simulator", teacher_horizon=6, seed=15)
    trainer.scenario_index = scenario_index
    agent = DQNAgent(hidden_size=16, layer_size=1)
    visited = []
    label = trainer.teacher.label

    def capture(physics, player_id):
        visited.append(physics.get_game_state())
        return label(physics, player_id)

    monkeypatch.setattr(trainer.teacher, "label", capture)
    trainer.pretraining_step(agent, num_steps=1)
    if scenario_index == 4:
        assert visited[0]["player1_paddle_size"] == game_config.PADDLE_HEIGHT / 2
        assert visited[0]["player2_paddle_size"] == game_config.PADDLE_HEIGHT / 2
    else:
        assert len(visited[0]["active_bonuses"]) == 1


@pytest.mark.parametrize(
    "kwargs",
    [
        {"teacher_horizon": 0},
        {"teacher_action_steps": 0},
        {"rollout_steps": 0},
        {"learner_probability": 1.1},
        {"supervised_batch_size": 0},
    ],
)
def test_simulator_pretrainer_rejects_invalid_budgets(kwargs):
    with pytest.raises(ValueError):
        create_pretrainer(mode="simulator", **kwargs)
