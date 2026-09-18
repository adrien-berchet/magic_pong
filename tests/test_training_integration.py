import json
import random
from types import SimpleNamespace

import pytest
import torch

import train_optimized
from magic_pong.core.game_engine import TrainingManager
from magic_pong.utils.config import ai_config
from magic_pong.utils.config import ai_config_tmp
from magic_pong.utils.config import game_config


def test_domain_randomization_updates_all_geometry_and_restores_after_failure():
    manager = TrainingManager(headless=True)
    original = game_config.model_copy(deep=True)
    rng_state = random.getstate()
    try:
        random.seed(17)
        with pytest.raises(RuntimeError, match="episode failed"):
            with train_optimized._domain_rand_context(
                SimpleNamespace(domain_rand_noise=0.1), manager
            ):
                manager.game_engine.start_game()
                physics = manager.game_engine.physics_engine
                processor = manager.game_engine.ai_environment.observation_processor
                assert physics.field_height == game_config.FIELD_HEIGHT
                assert processor.field_height == game_config.FIELD_HEIGHT
                assert physics.player1.max_y + physics.player1.height == game_config.FIELD_HEIGHT
                assert physics.bonus_spawner.field_height == game_config.FIELD_HEIGHT
                raise RuntimeError("episode failed")
        assert game_config == original
        assert manager.game_engine.physics_engine.field_height == original.FIELD_HEIGHT
        assert (
            manager.game_engine.ai_environment.observation_processor.field_height
            == original.FIELD_HEIGHT
        )
    finally:
        random.setstate(rng_state)
        manager.cleanup()


def test_domain_randomization_clamps_extreme_samples(monkeypatch):
    monkeypatch.setattr(game_config, "MAX_BALL_SPEED", 500.0)
    manager = TrainingManager(headless=True)
    original = game_config.model_copy(deep=True)
    samples = iter([0.99, 0.0, -0.99])
    monkeypatch.setattr(train_optimized.random, "uniform", lambda low, high: next(samples))
    try:
        with train_optimized._domain_rand_context(SimpleNamespace(domain_rand_noise=0.99), manager):
            assert game_config.BALL_SPEED == game_config.MAX_BALL_SPEED
            assert game_config.PADDLE_SPEED > 0
            assert game_config.FIELD_HEIGHT == int(game_config.PADDLE_HEIGHT + 50)
    finally:
        manager.cleanup()
    assert game_config == original


@pytest.mark.parametrize("control", ["--micro_reward_scale", "--anneal_micro_rewards"])
def test_training_phase_restores_original_shaping_scale_on_failure(control, tmp_path):
    class FailingManager:
        def set_ball_initial_direction(self, *args):
            pass

        def train_episode(self, *args, **kwargs):
            raise RuntimeError("episode failed")

    flags = [control, "0.0"] if control == "--micro_reward_scale" else [control]
    args = train_optimized.parse_arguments(
        ["--output_dir", str(tmp_path), "--competitive_eval_interval", "0", *flags]
    )
    agent = train_optimized.create_agent_from_args(args)
    with ai_config_tmp(MICRO_REWARD_SCALE=0.4):
        with pytest.raises(RuntimeError, match="episode failed"):
            train_optimized.train_phase(
                agent, train_optimized.get_opponent("dummy"), "test", 1, FailingManager(), args
            )
        assert ai_config.MICRO_REWARD_SCALE == pytest.approx(0.4)


def test_pipeline_load_best_requires_current_phase_result(tmp_path):
    class Agent:
        def __init__(self):
            self.loaded = []

        def load_model(self, path):
            self.loaded.append(path)

    agent = Agent()
    stale = tmp_path / "stale_best.pth"
    stale.write_text("old best")
    assert train_optimized._pipeline_load_best(agent, {}, "stale") is None
    assert agent.loaded == []
    assert train_optimized._pipeline_load_best(agent, {"best_model_path": str(stale)}, "fresh")
    assert agent.loaded == [str(stale)]


def test_single_phase_cleans_manager_and_league_on_failure(monkeypatch, tmp_path):
    cleaned = []
    league_cleaned = []

    class FakeManager:
        def __init__(self, **kwargs):
            pass

        def cleanup(self):
            cleaned.append("manager")

    class FakeLeague:
        def snapshot_references(self):
            return ("existing",)

    class FakeSelfPlaySampler:
        def __init__(self, **kwargs):
            self.league = FakeLeague()

        def cleanup(self):
            league_cleaned.append("league")

    monkeypatch.setattr(train_optimized, "TrainingManager", FakeManager)
    monkeypatch.setattr(train_optimized, "SelfPlaySampler", FakeSelfPlaySampler)
    monkeypatch.setattr(
        train_optimized,
        "train_phase",
        lambda *args, **kwargs: (_ for _ in ()).throw(RuntimeError("episode failed")),
    )
    args = train_optimized.parse_arguments(
        [
            "--mode",
            "single",
            "--output_dir",
            str(tmp_path),
            "--self_play_interval",
            "1",
            "--no-save_plots",
            "--no-save_metrics",
        ]
    )
    with pytest.raises(RuntimeError, match="episode failed"):
        train_optimized.train_single_phase(object(), args)
    assert cleaned == ["manager"]
    assert league_cleaned == ["league"]


def test_simulator_pretraining_league_and_validation_run_through_cli(tmp_path):
    threads = torch.get_num_threads()
    torch.set_num_threads(1)
    try:
        train_optimized.main(
            [
                "--mode",
                "single",
                "--total_episodes",
                "2",
                "--output_dir",
                str(tmp_path),
                "--model_prefix",
                "smoke",
                "--pretrain",
                "--pretrain_steps",
                "6",
                "--teacher_horizon",
                "6",
                "--teacher_action_steps",
                "2",
                "--teacher_rollout_steps",
                "2",
                "--self_play_interval",
                "1",
                "--max_steps_per_episode",
                "24",
                "--validation_opponents",
                "dummy",
                "--validation_seeds",
                "101",
                "--test_seeds",
                "10001",
                "--validation_max_steps",
                "24",
                "--validation_max_score",
                "1",
                "--competitive_eval_interval",
                "1",
                "--checkpoint_interval",
                "0",
                "--no-save_plots",
                "--no-save_metrics",
                "--no-track_q_values",
                "--hidden_size",
                "16",
                "--batch_size",
                "4",
                "--min_replay_size",
                "4",
                "--memory_size",
                "32",
                "--train_frequency",
                "1",
            ]
        )
    finally:
        torch.set_num_threads(threads)
    pretraining = json.loads((tmp_path / "smoke_pretrained.pretraining.json").read_text())
    assert pretraining["total_steps"] == 6
    assert (tmp_path / "smoke_best.pth").exists()
    assert (tmp_path / "smoke_final.pth").exists()
    evaluation = json.loads((tmp_path / "smoke_best.validation.json").read_text())
    assert evaluation["split"] == "validation"
    assert {cell["player_side"] for cell in evaluation["matchups"]} == {1, 2}
    assert {cell["bonuses_enabled"] for cell in evaluation["matchups"]} == {False, True}
    assert any("sha256:" in cell["opponent"] for cell in evaluation["matchups"])
    assert len(list((tmp_path / "league").glob("*.pth"))) >= 2


@pytest.mark.parametrize(
    "arguments",
    [
        ["--validation_seeds", "101", "--test_seeds", "101"],
        ["--teacher_action_steps", "100", "--teacher_horizon", "3"],
        ["--domain_rand_noise", "1.1"],
        ["--competitive_eval_interval", "-1"],
    ],
)
def test_training_rejects_incoherent_protocol(arguments):
    with pytest.raises(SystemExit):
        train_optimized.parse_arguments(arguments)


def test_training_seed_controls_initialization_for_independent_league_runs():
    first = train_optimized.create_agent_from_args(
        train_optimized.parse_arguments(["--seed", "23"])
    )
    repeat = train_optimized.create_agent_from_args(
        train_optimized.parse_arguments(["--seed", "23"])
    )
    other = train_optimized.create_agent_from_args(
        train_optimized.parse_arguments(["--seed", "24"])
    )
    first_weights = next(first.q_network.parameters())
    assert torch.equal(first_weights, next(repeat.q_network.parameters()))
    assert not torch.equal(first_weights, next(other.q_network.parameters()))
