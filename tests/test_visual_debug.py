"""Tests for AI visual debug capture helpers."""

from __future__ import annotations

import json
from typing import Any

from magic_pong.ai.models.simple_ai import DummyAI
from magic_pong.ai.pretraining import create_pretrainer
from magic_pong.ai.visual_debug import capture_pretraining_run
from magic_pong.ai.visual_debug import capture_training_episode
from magic_pong.ai.visual_debug import capture_trajectory_debug
from magic_pong.ai.visual_debug import draw_training_replay_frame
from magic_pong.ai.visual_debug import load_debug_json
from magic_pong.ai.visual_debug import mark_reward_frames
from magic_pong.ai.visual_debug import save_debug_json
from magic_pong.core.entities import Action
from magic_pong.core.entities import Player


class FixedActionPlayer(Player):
    def __init__(self, action: Action) -> None:
        super().__init__("FixedAction")
        self.action = action

    def get_action(self, observation: dict[str, Any] | None) -> Action:
        return self.action


def test_trajectory_capture_includes_path_and_optimal_point() -> None:
    data = capture_trajectory_debug(seed=123, player_id=1)

    assert data["kind"] == "trajectory"
    assert data["player_id"] == 1
    assert len(data["trajectory"]) > 1
    assert data["optimal_point"] is not None
    assert data["optimal_point"]["position"]
    assert data["optimal_point"]["ball_interception_position"]
    assert data["optimal_point"]["paddle_target_position"] == data["optimal_point"]["position"]
    end_x = data["trajectory"][-1]["position"][0]
    radius = data["config"]["ball_radius"]
    field_width = data["config"]["field_width"]
    assert end_x == radius or end_x == field_width - radius
    json.dumps(data)


def test_trajectory_capture_uses_pretraining_reward_trajectory() -> None:
    pretrainer = create_pretrainer()
    data = capture_trajectory_debug(pretrainer=pretrainer, seed=123, player_id=1)
    game_state = data["game_state"]
    expected = pretrainer.reward_calculator._simulate_ball_trajectory(
        tuple(game_state["ball_position"]),
        tuple(game_state["ball_velocity"]),
        tuple(game_state["field_bounds"]),
        max_time=None,
    )

    assert data["trajectory"] == [
        {"position": [position[0], position[1]], "time": time_step}
        for position, time_step in expected
    ]


def test_pretraining_run_capture_records_action_and_reward() -> None:
    agent = FixedActionPlayer(Action(move_x=0.0, move_y=1.0))
    data = capture_pretraining_run(agent, seed=123, player_id=1)

    assert data["kind"] == "pretraining_run"
    assert data["action"]["index"] == 2
    assert data["action"]["label"] == "2:down"
    assert isinstance(data["reward"], float)
    assert data["initial_game_state"]["player1_position"]
    assert data["final_game_state"]["player1_position"]
    assert len(data["trajectory"]) > 1
    json.dumps(data)


def test_mark_reward_frames_detects_rewards_and_events() -> None:
    trace = {
        "frames": [
            {"rewards": {"player1": 0.0, "player2": 0.0}, "events": {}},
            {"rewards": {"player1": 0.5, "player2": 0.0}, "events": {}},
            {
                "rewards": {"player1": 0.0, "player2": 0.0},
                "events": {"goals": [{"player": 1}]},
            },
        ]
    }

    mark_reward_frames(trace)

    assert trace["reward_frames"] == [1, 2]
    assert trace["frames"][0]["is_reward_frame"] is False
    assert trace["frames"][1]["is_reward_frame"] is True
    assert trace["frames"][2]["is_reward_frame"] is True


def test_training_capture_round_trips_json(tmp_path) -> None:
    agent = DummyAI(name="Agent")
    opponent = DummyAI(name="Opponent")
    trace = capture_training_episode(agent, opponent, seed=5, max_steps=3)

    assert trace["kind"] == "training_replay"
    assert trace["metadata"]["dt"] > 0
    assert trace["summary"]["steps"] == len(trace["frames"])
    assert len(trace["frames"]) <= 3
    assert "actions" in trace["frames"][0]
    assert "events" in trace["frames"][0]

    path = save_debug_json(trace, tmp_path / "trace.json")
    loaded = load_debug_json(path)

    assert loaded["summary"] == trace["summary"]
    assert loaded["reward_frames"] == trace["reward_frames"]


def test_replay_frame_draws_to_pygame_surface() -> None:
    import pygame

    agent = DummyAI(name="Agent")
    opponent = DummyAI(name="Opponent")
    trace = capture_training_episode(agent, opponent, seed=5, max_steps=1)

    pygame.font.init()
    surface = pygame.Surface((800, 600))
    draw_training_replay_frame(surface, trace, 0)

    assert surface.get_at((20, 20)) != pygame.Color(6, 10, 16, 255)
    pygame.quit()
