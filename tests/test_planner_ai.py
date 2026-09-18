"""The planner plays through the engine binding and repeats decisions when asked."""

import random

import numpy as np
import pytest
import torch

from magic_pong.ai.models.planner_ai import PlannerAI
from magic_pong.ai.models.planner_ai import make_teacher
from magic_pong.ai.models.simple_ai import create_ai
from magic_pong.ai.physics_teacher import PhysicsTeacher
from magic_pong.core.entities import Action
from magic_pong.core.entities import Ball
from magic_pong.core.entities import Vector2D
from magic_pong.core.game_engine import GameEngine
from magic_pong.core.game_engine import TrainingManager
from magic_pong.utils.config import game_config


class _Recorder:
    def __init__(self) -> None:
        self.bound = None

    def bind_physics(self, physics, player_id):
        self.bound = (physics, player_id)

    def get_action(self, observation):
        return Action(0.0, 0.0)


def test_game_engine_binds_the_live_physics_to_both_players() -> None:
    engine = GameEngine(headless=True)
    first, second = _Recorder(), _Recorder()
    engine.set_players(first, second)
    assert first.bound == (engine.physics_engine, 1)
    assert second.bound == (engine.physics_engine, 2)


def test_planner_refuses_to_act_without_a_bound_engine() -> None:
    planner = PlannerAI(backend="python")
    with pytest.raises(RuntimeError):
        planner.get_action({"ball_pos": [0.0, 0.0]})
    assert planner.get_action(None) == Action(0.0, 0.0)


def test_make_teacher_backends() -> None:
    assert isinstance(make_teacher(backend="python"), PhysicsTeacher)
    pytest.importorskip("numba")
    from magic_pong.ai.fast_teacher import FastPhysicsTeacher

    assert isinstance(make_teacher(backend="fast"), FastPhysicsTeacher)
    assert isinstance(make_teacher(backend="auto"), FastPhysicsTeacher)
    with pytest.raises(ValueError):
        make_teacher(backend="gpu")


def test_planner_repeats_its_decision_for_the_configured_interval(monkeypatch) -> None:
    planner = PlannerAI(decision_interval=4, backend="python", horizon_steps=6, action_steps=3)
    engine = GameEngine(headless=True)
    engine.set_players(planner, create_ai("dummy"))
    calls = []
    original = planner.teacher.label

    def counting_label(physics, player_id):
        calls.append(player_id)
        return original(physics, player_id)

    monkeypatch.setattr(planner.teacher, "label", counting_label)
    observation = engine.ai_environment.observation_processor.process_game_state(
        engine.physics_engine.get_game_state(), 1
    )
    actions = [planner.get_action(observation) for _ in range(8)]
    assert calls == [1, 1]
    assert actions[:4] == [actions[0]] * 4 and actions[4:] == [actions[4]] * 4


def test_player_two_planner_returns_home_when_the_ball_moves_away() -> None:
    engine = GameEngine(headless=True)
    planner = PlannerAI(backend="python", horizon_steps=30, action_steps=6)
    engine.set_players(create_ai("dummy"), planner)
    physics = engine.physics_engine
    physics.player2.position = Vector2D(physics.player2.min_x, 260.0)
    physics.player2.prev_position = physics.player2.position.copy()
    physics.ball = Ball(300.0, 300.0, -300.0, 0.0)
    observation = engine.ai_environment.observation_processor.process_game_state(
        physics.get_game_state(), 2
    )
    action = planner.get_action(observation)
    assert action.move_x == 1.0  # world frame: toward player 2's own goal line


def test_planner_defends_its_goal_against_a_static_opponent(monkeypatch) -> None:
    monkeypatch.setattr(game_config, "BONUSES_ENABLED", False)
    monkeypatch.setattr(game_config, "MAX_SCORE", 1)
    random.seed(3)
    np.random.seed(3)
    torch.manual_seed(3)
    planner = PlannerAI(horizon_steps=60, action_steps=6, decision_interval=2)
    manager = TrainingManager(headless=True)
    stats = manager.train_episode(planner, create_ai("dummy"), max_steps=900)
    goals_against = sum(goal["player"] == 2 for goal in stats["events"])
    assert goals_against == 0
    assert planner.last_decision is not None
