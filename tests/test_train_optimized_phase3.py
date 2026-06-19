"""Tests for Phase 3 DQN reward shaping and evaluation gate plumbing."""

import json
from pathlib import Path
from types import SimpleNamespace
from typing import Any

import pytest

import train_optimized
from magic_pong.ai.evaluation import EvaluationConfig
from magic_pong.ai.evaluation import EvaluationResult
from magic_pong.ai.evaluation import OpponentEvaluation
from magic_pong.utils.config import ai_config


class FakeAgent:
    epsilon = 0.5

    def on_episode_start(self) -> None:
        pass

    def get_training_stats(self) -> dict[str, Any]:
        return {"memory_size": 0}


class FakeOpponent:
    name = "fixed"


def train_args(**overrides: Any) -> SimpleNamespace:
    args = {
        "mode": "single",
        "mixed_opponents": False,
        "training_max_score": 1,
        "verbose": False,
        "max_steps_per_episode": 8,
        "checkpoint_interval": 0,
        "checkpoint_dir": ".",
        "early_stopping": 0,
        "log_interval": 100,
        "ball_direction": "random",
        "ball_angle": None,
        "headless": True,
        "evaluate_checkpoints": False,
        "checkpoint_eval_episodes": 0,
    }
    args.update(overrides)
    return SimpleNamespace(**args)


def test_phase3_cli_parses_reward_shaping_and_eval_gates() -> None:
    args = train_optimized.parse_arguments(
        [
            "--mode",
            "single",
            "--phase3_reward_shaping",
            "--eval_gate",
            "aggressive:win_rate:0.65",
            "--stop_when_eval_gates_pass",
        ]
    )

    assert args.reward_shaping == "phase3"
    assert args.parsed_eval_gates == (
        train_optimized.EvaluationGate("aggressive", "win_rate", 0.65),
    )
    assert "aggressive" in train_optimized._checkpoint_eval_opponents(args)
    assert args.stop_when_eval_gates_pass is True


def test_eval_gate_cli_rejects_invalid_gate(capsys: Any) -> None:
    with pytest.raises(SystemExit):
        train_optimized.parse_arguments(
            ["--mode", "single", "--eval_gate", "defensive:win_rate:65"]
        )

    assert "0.0 to 1.0" in capsys.readouterr().err

    with pytest.raises(SystemExit):
        train_optimized.parse_arguments(
            ["--mode", "single", "--eval_gate", "defensive:avg_reward:0.5"]
        )

    assert "Unknown evaluation gate metric" in capsys.readouterr().err


def test_checkpoint_evaluation_json_includes_gate_results(monkeypatch: Any, tmp_path: Path) -> None:
    captured: dict[str, Any] = {}

    def fake_evaluate_checkpoint(path: str | Path, config: EvaluationConfig) -> EvaluationResult:
        captured["reward_shaping_mode"] = ai_config.REWARD_SHAPING_MODE
        opponents = []
        for opponent in config.opponents:
            win_rate = 0.75 if opponent == "defensive" else 0.5
            opponents.append(
                OpponentEvaluation(
                    opponent=opponent,
                    episodes=2,
                    wins=int(win_rate * 2),
                    losses=2 - int(win_rate * 2),
                    draws=0,
                    timeouts=0,
                    win_rate=win_rate,
                    avg_reward=1.0,
                    std_reward=0.0,
                    min_reward=1.0,
                    max_reward=1.0,
                    avg_steps=2.0,
                    median_steps=2.0,
                    avg_goal_diff=1.0,
                )
            )
        return EvaluationResult(
            config=config, checkpoint_path=str(path), opponents=tuple(opponents)
        )

    monkeypatch.setattr(
        train_optimized.ai_evaluation, "evaluate_checkpoint", fake_evaluate_checkpoint
    )
    checkpoint_path = tmp_path / "fine_tuned_final.pth"
    args = train_args(
        eval_gates=["defensive:win_rate:0.50", "random:win_rate:0.65"],
        parsed_eval_gates=train_optimized.parse_eval_gates(
            ["defensive:win_rate:0.50", "random:win_rate:0.65"]
        ),
        checkpoint_eval_episodes=1,
        checkpoint_eval_max_steps=8,
        checkpoint_eval_seed=123,
        checkpoint_eval_epsilon=0.0,
        checkpoint_eval_max_score=3,
        checkpoint_eval_bonuses_enabled=False,
        checkpoint_eval_include_episodes=False,
        checkpoint_eval_opponents=None,
        checkpoint_eval_opponents_csv=None,
        reward_shaping="phase3",
        phase3_intercept_progress_reward=0.02,
        phase3_intercept_distance_penalty=0.02,
        phase3_successful_return_reward=0.05,
    )

    output_path = train_optimized.maybe_save_checkpoint_evaluation(
        checkpoint_path,
        args,
        {"candidate_type": "final_model"},
    )

    assert captured["reward_shaping_mode"] == "legacy"
    payload = json.loads(Path(output_path).read_text(encoding="utf-8"))
    assert payload["reward_shaping"] == {
        "evaluation_mode": "legacy",
        "training_mode": "phase3",
    }
    assert payload["eval_gates"]["passed"] is False
    assert payload["eval_gates"]["gates"][0] == {
        "opponent": "defensive",
        "metric": "win_rate",
        "comparison": ">=",
        "threshold": 0.5,
        "actual": 0.75,
        "passed": True,
        "error": None,
    }
    assert payload["eval_gates"]["gates"][1]["opponent"] == "random"
    assert payload["eval_gates"]["gates"][1]["actual"] == 0.5
    assert payload["eval_gates"]["gates"][1]["passed"] is False
    assert payload["training_context"]["candidate_type"] == "final_model"


def test_reward_shaping_config_restores_after_training_error(monkeypatch: Any) -> None:
    monkeypatch.setattr(ai_config, "REWARD_SHAPING_MODE", "legacy")
    monkeypatch.setattr(ai_config, "PHASE3_INTERCEPT_PROGRESS_REWARD", 0.02)

    class RaisingManager:
        def set_ball_initial_direction(self, *_args: Any) -> None:
            pass

        def train_episode(self, _agent: Any, _opponent: Any, max_steps: int) -> dict[str, Any]:
            assert max_steps == 8
            assert ai_config.REWARD_SHAPING_MODE == "phase3"
            assert ai_config.PHASE3_INTERCEPT_PROGRESS_REWARD == pytest.approx(0.123)
            raise RuntimeError("training failed")

    with pytest.raises(RuntimeError, match="training failed"):
        train_optimized.train_phase(
            FakeAgent(),
            FakeOpponent(),
            "phase3 config restoration",
            1,
            RaisingManager(),
            train_args(
                reward_shaping="phase3",
                phase3_intercept_progress_reward=0.123,
                phase3_intercept_distance_penalty=0.02,
                phase3_successful_return_reward=0.05,
            ),
        )

    assert ai_config.REWARD_SHAPING_MODE == "legacy"
    assert ai_config.PHASE3_INTERCEPT_PROGRESS_REWARD == pytest.approx(0.02)
