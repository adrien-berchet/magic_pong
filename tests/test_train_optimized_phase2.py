"""Tests for Phase 2 DQN fine-tuning plumbing in train_optimized."""

import json
import random
from pathlib import Path
from types import SimpleNamespace
from typing import Any

import numpy as np
import pytest

import train_optimized
from magic_pong.ai.evaluation import EvaluationConfig
from magic_pong.ai.evaluation import EvaluationResult
from magic_pong.ai.evaluation import OpponentEvaluation
from magic_pong.utils.config import ai_config
from magic_pong.utils.config import game_config


class FakeAgent:
    epsilon = 0.5
    training_enabled = True
    exploration_enabled = True

    def on_episode_start(self) -> None:
        pass

    def get_training_stats(self) -> dict[str, Any]:
        return {"dual_scale_training": False, "training_mode": "step_by_step", "memory_size": 0}

    def save_model(self, path: str) -> None:
        Path(path).write_text("fake checkpoint", encoding="utf-8")

    def set_training_mode(self, enabled: bool) -> None:
        self.training_enabled = enabled

    def set_exploration_mode(self, enabled: bool) -> None:
        self.exploration_enabled = enabled


class FakeOpponent:
    def __init__(self, name: str):
        self.name = name


def phase_args(**overrides: Any) -> SimpleNamespace:
    args = {
        "mode": "continue",
        "mixed_opponents": True,
        "opponent_mix": "dummy:1",
        "parsed_opponent_mix": train_optimized.parse_opponent_mix("dummy:1"),
        "opponent_mix_seed": 7,
        "training_score_mix": "3,5,11",
        "parsed_training_score_targets": train_optimized.parse_training_score_targets("3,5,11"),
        "fine_tune_bonuses": False,
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


def test_opponent_mix_normalizes_and_samples_deterministically() -> None:
    mix = train_optimized.parse_opponent_mix("defensive:2,random:1")

    assert [entry.name for entry in mix] == ["defensive", "random"]
    assert sum(entry.weight for entry in mix) == pytest.approx(1.0)
    assert mix[0].weight == pytest.approx(2 / 3)

    rng_a = random.Random(123)
    rng_b = random.Random(123)
    samples_a = [train_optimized.sample_weighted_opponent(mix, rng_a).name for _ in range(3)]
    samples_b = [train_optimized.sample_weighted_opponent(mix, rng_b).name for _ in range(3)]
    assert samples_a == samples_b

    with pytest.raises(ValueError, match="Unknown opponent"):
        train_optimized.parse_opponent_mix("not_real:1")
    with pytest.raises(ValueError, match="positive"):
        train_optimized.parse_opponent_mix("dummy:0")


def test_training_score_targets_parse_and_sample_deterministically() -> None:
    scores = train_optimized.parse_training_score_targets("3,5,11")

    assert scores == (3, 5, 11)
    rng_a = random.Random(5)
    rng_b = random.Random(5)
    assert [train_optimized.sample_training_score_target(scores, rng_a) for _ in range(6)] == [
        train_optimized.sample_training_score_target(scores, rng_b) for _ in range(6)
    ]

    with pytest.raises(ValueError, match="positive"):
        train_optimized.parse_training_score_targets("3,0")


def test_warm_start_copies_legacy_overlap_without_prev_action_leak(tmp_path: Path) -> None:
    torch = pytest.importorskip("torch")
    from magic_pong.ai.models.dqn_ai import DQNAgent

    def legacy_state_dict() -> dict[str, Any]:
        first_weight = torch.arange(32, dtype=torch.float32).expand(512, 32).clone()
        return {
            "fc_layers.0.weight": first_weight,
            "fc_layers.0.bias": torch.full((512,), 2.0),
            "fc_layers.1.weight": torch.full((256, 512), 3.0),
            "fc_layers.1.bias": torch.full((256,), 4.0),
            "fc_layers.2.weight": torch.full((128, 256), 5.0),
            "fc_layers.2.bias": torch.full((128,), 6.0),
            "layer_norms.0.weight": torch.full((512,), 5.0),
            "layer_norms.0.bias": torch.full((512,), 6.0),
            "layer_norms.1.weight": torch.full((256,), 7.0),
            "layer_norms.1.bias": torch.full((256,), 8.0),
            "layer_norms.2.weight": torch.full((128,), 11.0),
            "layer_norms.2.bias": torch.full((128,), 12.0),
            "output_layer.weight": torch.full((9, 128), 9.0),
            "output_layer.bias": torch.full((9,), 10.0),
        }

    checkpoint_path = tmp_path / "legacy_v1.pth"
    torch.save(
        {
            "q_network_state_dict": legacy_state_dict(),
            "target_network_state_dict": legacy_state_dict(),
            "epsilon": 0.123,
            "training_step": 42,
            "step_count": 7,
            "loss_history": [1.5],
            "reward_history": [2.5],
            "metadata": {"state_size": 32},
            "hyperparameters": {"state_size": 32},
        },
        checkpoint_path,
    )

    agent = DQNAgent(epsilon=1.0, batch_size=8, min_replay_size=1000)
    before = {key: value.detach().clone() for key, value in agent.q_network.state_dict().items()}

    train_optimized.warm_start_from_v1_checkpoint(agent, str(checkpoint_path))

    after = agent.q_network.state_dict()
    expected_columns = torch.tensor(
        list(range(10)) + list(range(12, 32)), dtype=after["fc_layers.0.weight"].dtype
    )
    assert torch.all(after["fc_layers.0.weight"][:, :30] == expected_columns)
    assert torch.allclose(after["fc_layers.0.weight"][:, 30:], before["fc_layers.0.weight"][:, 30:])
    assert torch.all(after["fc_layers.0.bias"] == 2.0)
    assert torch.all(after["fc_layers.1.weight"] == 3.0)
    assert torch.all(after["fc_layers.1.bias"] == 4.0)
    assert torch.allclose(after["output_layer.weight"], before["output_layer.weight"])
    assert torch.all(after["output_layer.bias"] == 10.0)
    assert agent.epsilon == pytest.approx(0.123)
    assert agent.training_step == 42
    assert agent.step_count == 7

    current_checkpoint_path = tmp_path / "current_schema.pth"
    agent.save_model(str(current_checkpoint_path))
    DQNAgent().load_model(str(current_checkpoint_path))


def test_warm_start_rejects_missing_state_dict_before_mutating(tmp_path: Path) -> None:
    torch = pytest.importorskip("torch")
    from magic_pong.ai.models.dqn_ai import DQNAgent

    checkpoint_path = tmp_path / "invalid_legacy.pth"
    torch.save({"q_network_state_dict": {}}, checkpoint_path)
    agent = DQNAgent(batch_size=8, min_replay_size=1000)
    before = {key: value.detach().clone() for key, value in agent.q_network.state_dict().items()}

    with pytest.raises(ValueError, match="target_network_state_dict"):
        train_optimized.warm_start_from_v1_checkpoint(agent, str(checkpoint_path))

    after = agent.q_network.state_dict()
    for key, before_value in before.items():
        assert torch.equal(after[key], before_value)


def test_progressive_curriculum_uses_paddle_hit_reward_config(
    monkeypatch: Any, tmp_path: Path
) -> None:
    original_paddle_hit_reward = ai_config.PADDLE_HIT_REWARD
    observed_rewards: list[tuple[int, float]] = []

    class FakeTrainingManager:
        def __init__(self, *_args: Any, **_kwargs: Any) -> None:
            pass

        def cleanup(self) -> None:
            pass

    class SavingAgent(FakeAgent):
        epsilon_min = 0.05

    def fake_train_phase(
        _agent: Any,
        _opponent: Any,
        _phase_name: str,
        _episodes: int,
        _training_manager: Any,
        _args: Any,
        *,
        phase_num: int,
        total_phases: int,
        start_episode: int = 0,
    ) -> tuple[list[float], int, int]:
        assert total_phases == 4
        assert start_episode >= 0
        observed_rewards.append((phase_num, ai_config.PADDLE_HIT_REWARD))
        return [1.0], 1, 1

    monkeypatch.setattr(train_optimized, "TrainingManager", FakeTrainingManager)
    monkeypatch.setattr(train_optimized, "get_opponent", lambda name: FakeOpponent(name))
    monkeypatch.setattr(train_optimized, "train_phase", fake_train_phase)

    args = phase_args(
        mode="progressive",
        mixed_opponents=False,
        progressive_episodes_per_phase=1,
        progressive_epsilon_reduction=0.7,
        phase1_training_opponent="dummy",
        output_dir=str(tmp_path),
        model_prefix="progressive_smoke",
        save_phase_models=False,
        save_plots=False,
        fast_gui=True,
        eval_gates=[],
        parsed_eval_gates=(),
    )

    _agent, results = train_optimized.train_progressive_curriculum(SavingAgent(), args)

    assert observed_rewards[0] == (1, 1)
    assert observed_rewards[1] == (2, 1)
    assert results["total_episodes"] == 4
    assert ai_config.PADDLE_HIT_REWARD == pytest.approx(original_paddle_hit_reward)


def test_train_phase_samples_opponent_and_score_and_restores_config(monkeypatch: Any) -> None:
    monkeypatch.setattr(game_config, "MAX_SCORE", 11)
    monkeypatch.setattr(game_config, "BONUSES_ENABLED", True)
    monkeypatch.setattr(
        train_optimized, "get_opponent", lambda opponent_name: FakeOpponent(opponent_name)
    )
    observed: list[tuple[str, int, bool]] = []

    class RecordingManager:
        def set_ball_initial_direction(self, *_args: Any) -> None:
            pass

        def train_episode(self, _agent: Any, opponent: Any, max_steps: int) -> dict[str, Any]:
            assert max_steps == 8
            observed.append((opponent.name, game_config.MAX_SCORE, game_config.BONUSES_ENABLED))
            return {"total_reward_p1": 1.0, "winner": 1}

    rewards, wins, episodes = train_optimized.train_phase(
        FakeAgent(),
        FakeOpponent("fixed"),
        "mixed fine tuning",
        3,
        RecordingManager(),
        phase_args(),
    )

    assert rewards == [1.0, 1.0, 1.0]
    assert wins == 3
    assert episodes == 3
    assert {opponent for opponent, _score, _bonuses in observed} == {"dummy"}
    assert {score for _opponent, score, _bonuses in observed} <= {3, 5, 11}
    assert all(bonuses is False for _opponent, _score, bonuses in observed)
    assert game_config.MAX_SCORE == 11
    assert game_config.BONUSES_ENABLED is True


def test_train_phase_restores_config_after_episode_error(monkeypatch: Any) -> None:
    monkeypatch.setattr(game_config, "MAX_SCORE", 11)
    monkeypatch.setattr(game_config, "BONUSES_ENABLED", False)
    monkeypatch.setattr(
        train_optimized, "get_opponent", lambda opponent_name: FakeOpponent(opponent_name)
    )

    class RaisingManager:
        def set_ball_initial_direction(self, *_args: Any) -> None:
            pass

        def train_episode(self, _agent: Any, _opponent: Any, max_steps: int) -> dict[str, Any]:
            assert max_steps == 8
            assert game_config.MAX_SCORE in {3, 5, 11}
            assert game_config.BONUSES_ENABLED is True
            raise RuntimeError("episode failed")

    with pytest.raises(RuntimeError, match="episode failed"):
        train_optimized.train_phase(
            FakeAgent(),
            FakeOpponent("fixed"),
            "mixed fine tuning",
            1,
            RaisingManager(),
            phase_args(fine_tune_bonuses=True),
        )

    assert game_config.MAX_SCORE == 11
    assert game_config.BONUSES_ENABLED is False


def test_train_phase_uses_training_max_score_when_mix_disabled(monkeypatch: Any) -> None:
    monkeypatch.setattr(game_config, "MAX_SCORE", 11)
    monkeypatch.setattr(game_config, "BONUSES_ENABLED", True)
    observed: list[tuple[str, int, bool]] = []

    class RecordingManager:
        def set_ball_initial_direction(self, *_args: Any) -> None:
            pass

        def train_episode(self, _agent: Any, opponent: Any, max_steps: int) -> dict[str, Any]:
            assert max_steps == 8
            observed.append((opponent.name, game_config.MAX_SCORE, game_config.BONUSES_ENABLED))
            return {"total_reward_p1": 1.0, "winner": 2}

    train_optimized.train_phase(
        FakeAgent(),
        FakeOpponent("fixed"),
        "single opponent",
        2,
        RecordingManager(),
        phase_args(mixed_opponents=False, training_max_score=7),
    )

    assert observed == [("fixed", 7, True), ("fixed", 7, True)]
    assert game_config.MAX_SCORE == 11
    assert game_config.BONUSES_ENABLED is True


def test_checkpoint_evaluation_writes_json_with_training_context(
    monkeypatch: Any, tmp_path: Path
) -> None:
    captured: dict[str, Any] = {}

    def fake_evaluate_checkpoint(path: str | Path, config: EvaluationConfig) -> EvaluationResult:
        random.seed(12345)
        np.random.seed(12345)
        captured["path"] = str(path)
        captured["config"] = config
        return EvaluationResult(
            config=config,
            checkpoint_path=str(path),
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

    monkeypatch.setattr(
        train_optimized.ai_evaluation, "evaluate_checkpoint", fake_evaluate_checkpoint
    )
    checkpoint_path = tmp_path / "checkpoint_ep100_phase1_test.pth"
    random.seed(42)
    np.random.seed(42)
    python_rng_state = random.getstate()
    numpy_rng_state = np.random.get_state()
    args = phase_args(
        checkpoint_eval_episodes=2,
        checkpoint_eval_max_steps=None,
        checkpoint_eval_seed=99,
        checkpoint_eval_epsilon=0.0,
        checkpoint_eval_max_score=5,
        checkpoint_eval_bonuses_enabled=None,
        checkpoint_eval_include_episodes=False,
        checkpoint_eval_opponents=None,
        checkpoint_eval_opponents_csv=None,
    )

    output_path = train_optimized.maybe_save_checkpoint_evaluation(
        checkpoint_path,
        args,
        {"episode": 100, "opponent_mix": [{"opponent": "dummy", "weight": 1.0}]},
    )

    assert output_path == str(tmp_path / "checkpoint_ep100_phase1_test.eval.json")
    assert captured["path"] == str(checkpoint_path)
    assert captured["config"] == EvaluationConfig(
        opponents=train_optimized.DEFAULT_CHECKPOINT_EVAL_OPPONENTS,
        episodes=2,
        max_steps=8,
        seed=99,
        eval_epsilon=0.0,
        max_score=5,
        bonuses_enabled=False,
    )
    payload = json.loads(Path(output_path).read_text())
    assert payload["checkpoint_path"] == str(checkpoint_path)
    assert payload["training_context"]["episode"] == 100
    assert payload["training_context"]["opponent_mix"][0]["opponent"] == "dummy"
    assert random.getstate() == python_rng_state
    restored_numpy_state = np.random.get_state()
    assert restored_numpy_state[0] == numpy_rng_state[0]
    np.testing.assert_array_equal(restored_numpy_state[1], numpy_rng_state[1])
    assert restored_numpy_state[2:] == numpy_rng_state[2:]


def test_train_single_phase_evaluates_final_model_when_checkpoint_eval_enabled(
    monkeypatch: Any, tmp_path: Path
) -> None:
    monkeypatch.setattr(game_config, "MAX_SCORE", 11)
    monkeypatch.setattr(game_config, "BONUSES_ENABLED", True)
    monkeypatch.setattr(train_optimized, "get_opponent", lambda name: FakeOpponent(name))
    captured: dict[str, Any] = {}

    class RecordingManager:
        def set_ball_initial_direction(self, *_args: Any) -> None:
            pass

        def train_episode(self, _agent: Any, opponent: Any, max_steps: int) -> dict[str, Any]:
            assert opponent.name == "defensive"
            assert max_steps == 8
            return {"total_reward_p1": 2.0, "winner": 1}

        def cleanup(self) -> None:
            captured["cleanup"] = True

    def fake_training_manager(**_kwargs: Any) -> RecordingManager:
        return RecordingManager()

    def fake_checkpoint_eval(path: str | Path, _args: Any, context: dict[str, Any]) -> str:
        captured["eval_path"] = str(path)
        captured["context"] = context
        return str(Path(path).with_suffix(".eval.json"))

    monkeypatch.setattr(train_optimized, "TrainingManager", fake_training_manager)
    monkeypatch.setattr(train_optimized, "maybe_save_checkpoint_evaluation", fake_checkpoint_eval)

    args = phase_args(
        mode="continue",
        mixed_opponents=False,
        opponent="defensive",
        total_episodes=1,
        output_dir=str(tmp_path),
        model_prefix="fine_tuned",
        save_plots=False,
        checkpoint_eval_episodes=1,
        evaluate_checkpoints=True,
    )

    _agent, results = train_optimized.train_single_phase(FakeAgent(), args)

    assert results["final_model_path"] == str(tmp_path / "fine_tuned_final.pth")
    assert captured["cleanup"] is True
    assert captured["eval_path"] == str(tmp_path / "fine_tuned_final.pth")
    assert captured["context"]["candidate_type"] == "final_model"
    assert captured["context"]["label"] == "single_phase_final"
    assert captured["context"]["win_rate"] == 100


def test_evaluate_final_performance_restores_score_when_setup_fails(monkeypatch: Any) -> None:
    monkeypatch.setattr(game_config, "MAX_SCORE", 11)
    monkeypatch.setattr(train_optimized, "get_opponent", lambda name: FakeOpponent(name))

    def raising_training_manager(**_kwargs: Any) -> None:
        raise RuntimeError("manager failed")

    monkeypatch.setattr(train_optimized, "TrainingManager", raising_training_manager)
    args = phase_args(
        eval_episodes=1,
        eval_max_score=5,
        eval_epsilon=0.0,
        headless=True,
        max_steps_per_episode=8,
        output_dir=".",
        model_prefix="eval",
        save_eval_results=False,
    )

    with pytest.raises(RuntimeError, match="manager failed"):
        train_optimized.evaluate_final_performance(FakeAgent(), args)

    assert game_config.MAX_SCORE == 11


def test_mixed_opponents_cli_requires_fine_tuning_mode_and_loaded_model(capsys: Any) -> None:
    with pytest.raises(SystemExit):
        train_optimized.parse_arguments(
            ["--mode", "curriculum", "--load_model", "model.pth", "--mixed_opponents"]
        )
    assert "--mixed_opponents can only be used" in capsys.readouterr().err

    with pytest.raises(SystemExit):
        train_optimized.parse_arguments(["--mode", "continue", "--mixed_opponents"])
    assert "requires --load_model" in capsys.readouterr().err

    args = train_optimized.parse_arguments(
        ["--mode", "continue", "--load_model", "model.pth", "--mixed_opponents"]
    )
    assert args.mixed_opponents is True


def test_checkpoint_eval_max_steps_zero_is_rejected_when_enabled(capsys: Any) -> None:
    with pytest.raises(SystemExit):
        train_optimized.parse_arguments(
            [
                "--mode",
                "continue",
                "--load_model",
                "model.pth",
                "--mixed_opponents",
                "--evaluate_checkpoints",
                "--checkpoint_eval_max_steps",
                "0",
            ]
        )

    assert "max_steps must be positive" in capsys.readouterr().err


def test_phase2_cli_flags_are_discoverable(capsys: Any) -> None:
    with pytest.raises(SystemExit):
        train_optimized.parse_arguments(["--help"])

    help_text = capsys.readouterr().out
    assert "--mixed_opponents" in help_text
    assert "--training_max_scores" in help_text
    assert "--checkpoint_eval_episodes" in help_text
