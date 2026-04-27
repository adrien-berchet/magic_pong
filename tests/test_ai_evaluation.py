"""Tests for the reusable AI evaluation harness."""

import csv
import io
import json
import os
import random
import subprocess
import sys
from collections import deque
from typing import Any

import numpy as np
import pytest

from magic_pong.ai.evaluation import EvaluationConfig
from magic_pong.ai.evaluation import EvaluationResult
from magic_pong.ai.evaluation import OpponentEvaluation
from magic_pong.ai.evaluation import derive_episode_seed
from magic_pong.ai.evaluation import evaluate_agent
from magic_pong.ai.evaluation import main
from magic_pong.ai.evaluation import result_to_csv
from magic_pong.ai.evaluation import result_to_json
from magic_pong.utils.config import game_config


class FakeAgent:
    pass


class FakeDQNLikeAgent:
    def __init__(self) -> None:
        self.training_enabled = True
        self.exploration_enabled = False
        self.epsilon = 0.9
        self.training_mode_calls: list[bool] = []
        self.exploration_mode_calls: list[bool] = []

    def set_training_mode(self, training: bool) -> None:
        self.training_mode_calls.append(training)
        self.training_enabled = training
        self.exploration_enabled = training

    def set_exploration_mode(self, explore: bool) -> None:
        self.exploration_mode_calls.append(explore)
        self.exploration_enabled = explore


class FakeStatefulDQNLikeAgent(FakeDQNLikeAgent):
    def __init__(self) -> None:
        super().__init__()
        self.current_episode_reward = 7.0
        self.episode_rewards = [1.0, 2.0]
        self.episode_buffer = [{"state": "pending"}]
        self.last_state = np.array([1.0, 2.0])
        self.last_action = 3
        self.training_step = 11
        self.step_count = 13
        self.loss_history = [0.25]
        self.reward_history = [1.5]
        self.memory = type(
            "FakeReplayMemory",
            (),
            {
                "buffer": deque(["transition-1"], maxlen=5),
                "priorities": deque([0.5], maxlen=5),
            },
        )()


def fake_opponent_factory(opponent_type: str, **kwargs: Any) -> dict[str, Any]:
    return {"type": opponent_type, **kwargs}


def manager_factory_for(stats: list[dict[str, Any]]) -> type:
    class FakeTrainingManager:
        instances: list["FakeTrainingManager"] = []

        def __init__(self, **kwargs: Any) -> None:
            self.kwargs = kwargs
            self.cleaned_up = False
            self.instances.append(self)

        def train_episode(self, _agent: Any, opponent: Any, max_steps: int) -> dict[str, Any]:
            episode_stats = dict(stats.pop(0))
            episode_stats["opponent_type"] = opponent["type"]
            episode_stats["max_steps"] = max_steps
            return episode_stats

        def cleanup(self) -> None:
            self.cleaned_up = True

    return FakeTrainingManager


def test_aggregation_with_fake_training_manager_returns_expected_metrics() -> None:
    stats = [
        {
            "winner": 1,
            "steps": 10,
            "total_reward_p1": 4.0,
            "total_reward_p2": -1.0,
            "events": [{"player": 1}, {"player": 1}],
        },
        {
            "winner": 2,
            "steps": 20,
            "total_reward_p1": -2.0,
            "total_reward_p2": 1.0,
            "events": [{"player": 2}],
        },
        {
            "winner": 0,
            "steps": 30,
            "total_reward_p1": 1.0,
            "total_reward_p2": 0.5,
            "events": [],
        },
    ]
    manager_factory = manager_factory_for(stats)
    config = EvaluationConfig(opponents=("random",), episodes=3, include_episodes=True)

    result = evaluate_agent(
        FakeAgent(),
        config,
        opponent_factory=fake_opponent_factory,
        training_manager_factory=manager_factory,
    )

    aggregate = result.opponents[0]
    assert aggregate.opponent == "random"
    assert aggregate.episodes == 3
    assert aggregate.wins == 1
    assert aggregate.losses == 1
    assert aggregate.draws == 1
    assert aggregate.timeouts == 1
    assert aggregate.win_rate == pytest.approx(1 / 3)
    assert aggregate.avg_reward == pytest.approx(1.0)
    assert aggregate.min_reward == pytest.approx(-2.0)
    assert aggregate.max_reward == pytest.approx(4.0)
    assert aggregate.avg_steps == pytest.approx(20.0)
    assert aggregate.median_steps == pytest.approx(20.0)
    assert aggregate.avg_goal_diff == pytest.approx(1 / 3)
    assert [episode.goal_diff for episode in result.episodes] == [2, -1, 0]
    assert all(instance.cleaned_up for instance in manager_factory.instances)


def test_same_seed_produces_same_episode_seeds_and_fake_rows() -> None:
    class RandomBackedManager:
        def __init__(self, **_kwargs: Any) -> None:
            pass

        def train_episode(self, _agent: Any, _opponent: Any, max_steps: int) -> dict[str, Any]:
            assert max_steps == 1000
            return {
                "winner": 1,
                "steps": random.randint(1, 100),
                "total_reward_p1": float(np.random.random()),
                "total_reward_p2": 0.0,
                "events": [{"player": 1}],
            }

    config = EvaluationConfig(
        opponents=("random", "dummy"), episodes=2, seed=123, include_episodes=True
    )

    result_a = evaluate_agent(
        FakeAgent(),
        config,
        opponent_factory=fake_opponent_factory,
        training_manager_factory=RandomBackedManager,
    )
    result_b = evaluate_agent(
        FakeAgent(),
        config,
        opponent_factory=fake_opponent_factory,
        training_manager_factory=RandomBackedManager,
    )

    assert [episode.seed for episode in result_a.episodes] == [
        episode.seed for episode in result_b.episodes
    ]
    assert result_a.episodes == result_b.episodes
    assert result_a.episodes[0].seed == derive_episode_seed(123, "random", 0, 0)


def test_eval_mode_snapshot_and_restore_for_dqn_like_agent() -> None:
    agent = FakeDQNLikeAgent()

    class AssertingManager:
        def __init__(self, **_kwargs: Any) -> None:
            pass

        def train_episode(
            self, active_agent: Any, _opponent: Any, max_steps: int
        ) -> dict[str, Any]:
            assert max_steps == 1000
            assert active_agent.training_enabled is False
            assert active_agent.exploration_enabled is True
            assert active_agent.epsilon == pytest.approx(0.25)
            return {
                "winner": 1,
                "steps": 1,
                "total_reward_p1": 1.0,
                "total_reward_p2": 0.0,
                "events": [{"player": 1}],
            }

    config = EvaluationConfig(opponents=("random",), episodes=1, eval_epsilon=0.25)

    evaluate_agent(
        agent,
        config,
        opponent_factory=fake_opponent_factory,
        training_manager_factory=AssertingManager,
    )

    assert agent.training_enabled is True
    assert agent.exploration_enabled is False
    assert agent.epsilon == pytest.approx(0.9)
    assert agent.training_mode_calls == [False, True]
    assert agent.exploration_mode_calls == [True, False]


def test_evaluation_restores_dqn_like_episode_and_replay_state() -> None:
    agent = FakeStatefulDQNLikeAgent()
    original_last_state = agent.last_state.copy()

    class MutatingManager:
        def __init__(self, **_kwargs: Any) -> None:
            pass

        def train_episode(
            self, active_agent: Any, _opponent: Any, max_steps: int
        ) -> dict[str, Any]:
            assert max_steps == 1000
            active_agent.current_episode_reward = 99.0
            active_agent.episode_rewards.append(42.0)
            active_agent.episode_buffer.append({"state": "eval"})
            active_agent.last_state = np.array([9.0, 9.0])
            active_agent.last_action = 8
            active_agent.training_step += 5
            active_agent.step_count += 6
            active_agent.loss_history.append(4.5)
            active_agent.reward_history.append(5.5)
            active_agent.memory.buffer.append("transition-eval")
            active_agent.memory.priorities.append(9.0)
            return {
                "winner": 1,
                "steps": 1,
                "total_reward_p1": 1.0,
                "total_reward_p2": 0.0,
                "events": [{"player": 1}],
            }

    evaluate_agent(
        agent,
        EvaluationConfig(opponents=("random",), episodes=1),
        opponent_factory=fake_opponent_factory,
        training_manager_factory=MutatingManager,
    )

    assert agent.current_episode_reward == pytest.approx(7.0)
    assert agent.episode_rewards == [1.0, 2.0]
    assert agent.episode_buffer == [{"state": "pending"}]
    np.testing.assert_array_equal(agent.last_state, original_last_state)
    assert agent.last_action == 3
    assert agent.training_step == 11
    assert agent.step_count == 13
    assert agent.loss_history == [0.25]
    assert agent.reward_history == [1.5]
    assert list(agent.memory.buffer) == ["transition-1"]
    assert list(agent.memory.priorities) == [0.5]


def test_game_config_restored_after_exception() -> None:
    original_max_score = game_config.MAX_SCORE
    original_bonuses_enabled = game_config.BONUSES_ENABLED

    class RaisingManager:
        def __init__(self, **_kwargs: Any) -> None:
            pass

        def train_episode(self, _agent: Any, _opponent: Any, max_steps: int) -> dict[str, Any]:
            assert max_steps == 1000
            assert game_config.MAX_SCORE == original_max_score + 1
            assert game_config.BONUSES_ENABLED is (not original_bonuses_enabled)
            raise RuntimeError("episode failed")

    config = EvaluationConfig(
        opponents=("random",),
        episodes=1,
        max_score=original_max_score + 1,
        bonuses_enabled=not original_bonuses_enabled,
    )

    with pytest.raises(RuntimeError, match="episode failed"):
        evaluate_agent(
            FakeAgent(),
            config,
            opponent_factory=fake_opponent_factory,
            training_manager_factory=RaisingManager,
        )

    assert game_config.MAX_SCORE == original_max_score
    assert game_config.BONUSES_ENABLED is original_bonuses_enabled


def test_training_dummy_is_rejected_as_non_deterministic_opponent() -> None:
    with pytest.raises(ValueError, match="training_dummy.*not deterministic"):
        EvaluationConfig(opponents=("training_dummy",))


def test_winner_zero_is_timeout_draw_not_loss() -> None:
    stats = [
        {
            "winner": 0,
            "steps": 5,
            "total_reward_p1": -0.25,
            "total_reward_p2": 0.25,
            "events": [],
        }
    ]
    config = EvaluationConfig(opponents=("dummy",), episodes=1, max_steps=5, include_episodes=True)

    result = evaluate_agent(
        FakeAgent(),
        config,
        opponent_factory=fake_opponent_factory,
        training_manager_factory=manager_factory_for(stats),
    )

    episode = result.episodes[0]
    aggregate = result.opponents[0]
    assert episode.outcome == "timeout"
    assert episode.agent_won is False
    assert episode.timeout is True
    assert aggregate.losses == 0
    assert aggregate.draws == 1
    assert aggregate.timeouts == 1


def test_json_and_csv_outputs_are_parseable_and_writable(tmp_path: Any) -> None:
    stats = [
        {
            "winner": 1,
            "steps": 3,
            "total_reward_p1": 2.5,
            "total_reward_p2": -0.5,
            "events": [{"player": 1}],
        }
    ]
    config = EvaluationConfig(opponents=("random",), episodes=1, include_episodes=True)
    result = evaluate_agent(
        FakeAgent(),
        config,
        opponent_factory=fake_opponent_factory,
        training_manager_factory=manager_factory_for(stats),
    )
    json_path = tmp_path / "evaluation.json"
    csv_path = tmp_path / "evaluation.csv"

    json_payload = result_to_json(result, json_path)
    csv_payload = result_to_csv(result, csv_path)

    parsed_json = json.loads(json_payload)
    assert parsed_json["schema_version"] == "magic_pong.evaluation.v1"
    assert parsed_json["episodes"][0]["outcome"] == "win"
    assert json.loads(json_path.read_text()) == parsed_json

    rows = list(csv.DictReader(io.StringIO(csv_payload)))
    assert [row["row_type"] for row in rows] == ["opponent", "episode"]
    assert rows[1]["total_reward_agent"] == "2.5"
    assert list(csv.DictReader(io.StringIO(csv_path.read_text()))) == rows


def test_evaluation_import_does_not_emit_pygame_prompt() -> None:
    env = os.environ.copy()
    env.pop("PYGAME_HIDE_SUPPORT_PROMPT", None)

    result = subprocess.run(
        [sys.executable, "-c", "import magic_pong.ai.evaluation"],
        check=True,
        capture_output=True,
        text=True,
        env=env,
    )

    assert result.stdout == ""


def test_cli_stdout_json_is_parseable_without_output(monkeypatch: Any, capsys: Any) -> None:
    def fake_evaluate_checkpoint(path: str, config: EvaluationConfig) -> EvaluationResult:
        return EvaluationResult(
            config=config,
            checkpoint_path=path,
            opponents=(
                OpponentEvaluation(
                    opponent=config.opponents[0],
                    episodes=1,
                    wins=1,
                    losses=0,
                    draws=0,
                    timeouts=0,
                    win_rate=1.0,
                    avg_reward=1.0,
                    std_reward=0.0,
                    min_reward=1.0,
                    max_reward=1.0,
                    avg_steps=2.0,
                    median_steps=2.0,
                    avg_goal_diff=1.0,
                ),
            ),
        )

    monkeypatch.setattr("magic_pong.ai.evaluation.evaluate_checkpoint", fake_evaluate_checkpoint)

    exit_code = main(["--checkpoint", "model.pth", "--opponent", "dummy", "--episodes", "1"])

    assert exit_code == 0
    parsed = json.loads(capsys.readouterr().out)
    assert parsed["checkpoint_path"] == "model.pth"
    assert parsed["opponents"][0]["opponent"] == "dummy"


def test_cli_smoke_uses_checkpoint_evaluator_monkeypatch(monkeypatch: Any, tmp_path: Any) -> None:
    captured: dict[str, Any] = {}

    def fake_evaluate_checkpoint(path: str, config: EvaluationConfig) -> EvaluationResult:
        captured["path"] = path
        captured["config"] = config
        return EvaluationResult(
            config=config,
            checkpoint_path=path,
            checkpoint_metadata={"training_step": 12},
            opponents=(
                OpponentEvaluation(
                    opponent=config.opponents[0],
                    episodes=1,
                    wins=1,
                    losses=0,
                    draws=0,
                    timeouts=0,
                    win_rate=1.0,
                    avg_reward=1.0,
                    std_reward=0.0,
                    min_reward=1.0,
                    max_reward=1.0,
                    avg_steps=2.0,
                    median_steps=2.0,
                    avg_goal_diff=1.0,
                ),
            ),
        )

    monkeypatch.setattr("magic_pong.ai.evaluation.evaluate_checkpoint", fake_evaluate_checkpoint)
    output_path = tmp_path / "cli.json"

    exit_code = main(
        [
            "--checkpoint",
            "model.pth",
            "--opponent",
            "dummy",
            "--episodes",
            "1",
            "--max-steps",
            "2",
            "--seed",
            "99",
            "--eval-epsilon",
            "0.1",
            "--max-score",
            "3",
            "--no-bonuses",
            "--ball-direction",
            "1",
            "--ball-angle-deg",
            "15",
            "--format",
            "json",
            "--output",
            str(output_path),
            "--include-episodes",
        ]
    )

    assert exit_code == 0
    assert captured["path"] == "model.pth"
    assert captured["config"] == EvaluationConfig(
        opponents=("dummy",),
        episodes=1,
        max_steps=2,
        seed=99,
        eval_epsilon=0.1,
        max_score=3,
        bonuses_enabled=False,
        ball_direction=1,
        ball_angle_deg=15.0,
        include_episodes=True,
    )
    assert json.loads(output_path.read_text())["checkpoint_metadata"] == {"training_step": 12}
