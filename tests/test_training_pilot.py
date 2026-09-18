"""A controlled pilot compares gammas using identical budgets and warm starts."""

import json
import random
from pathlib import Path

import numpy as np
import pytest
import torch

from magic_pong.ai import pilot
from magic_pong.ai.models.simple_ai import BUILTIN_OPPONENTS
from magic_pong.ai.pilot import PilotConfig
from magic_pong.ai.pilot import epsilon_at_step
from magic_pong.ai.pilot import run_pilot
from magic_pong.ai.pilot import shaping_at_step
from magic_pong.utils.config import ai_config
from magic_pong.utils.config import game_config


def tiny_config(**changes):
    values = {
        "seeds": (11,),
        "env_steps": 16,
        "pretraining_labels": 2,
        "hidden_layers": (8,),
        "batch_size": 2,
        "min_replay_size": 2,
        "memory_size": 16,
        "max_episode_steps": 3,
        "snapshot_interval": 8,
        "eval_seeds": (101,),
        "eval_max_steps": 2,
        "teacher_horizon": 2,
        "teacher_action_steps": 1,
        "teacher_rollout_steps": 2,
        "pretraining_batch_labels": 1,
    }
    values.update(changes)
    return PilotConfig(**values)


def test_schedules_depend_on_frames_and_finish_before_budget():
    config = tiny_config(env_steps=100)
    assert epsilon_at_step(config, 0) == pytest.approx(0.3)
    assert epsilon_at_step(config, 40) == pytest.approx(0.175)
    assert epsilon_at_step(config, 80) == pytest.approx(0.05)
    assert epsilon_at_step(config, 100) == pytest.approx(0.05)
    assert shaping_at_step(config, 0) == 1
    assert shaping_at_step(config, 25) == pytest.approx(0.5)
    assert shaping_at_step(config, 50) == 0


def test_tiny_pilot_pairs_warmstarts_and_uses_exact_balanced_step_budgets(tmp_path, monkeypatch):
    calls = []
    decision_schedules = []
    original_evaluate = pilot.evaluate_competitive_agent
    original_engine = pilot.GameEngine

    class TrackedEngine(original_engine):
        def update(self, dt=None):
            agent = (
                self.player1 if getattr(self.player1, "training_enabled", False) else self.player2
            )
            decision_schedules.append((agent.epsilon, ai_config.MICRO_REWARD_SCALE))
            return super().update(dt)

    def tracked_evaluate(agent, config, **kwargs):
        calls.append((config.opponents, config.seeds, kwargs.get("split", "validation")))
        return original_evaluate(agent, config, **kwargs)

    monkeypatch.setattr(pilot, "evaluate_competitive_agent", tracked_evaluate)
    monkeypatch.setattr(pilot, "GameEngine", TrackedEngine)
    config = tiny_config()
    result = run_pilot(config, tmp_path / "pilot")
    seed_result = result["seeds"]["11"]
    assert seed_result["status"] == "complete"
    assert seed_result["pretraining"]["labels"] == 2
    assert seed_result["pretrained_network_sha256"] != seed_result["untrained_network_sha256"]
    expected_schedule = [
        (epsilon_at_step(config, step), shaping_at_step(config, step)) for step in range(16)
    ]
    assert decision_schedules == expected_schedule * 2
    assert len(calls) == 6  # untrained, pretrained, midpoint/final for each gamma
    assert all(
        len(opponents) == len(BUILTIN_OPPONENTS) and seeds == (101,) and split == "validation"
        for opponents, seeds, split in calls
    )
    branches = seed_result["branches"]
    assert branches["0.99"]["initial_network_sha256"] == branches["0.999"]["initial_network_sha256"]
    for gamma, branch in branches.items():
        assert branch["gamma"] == float(gamma)
        assert branch["initial_network_sha256"] == seed_result["pretrained_network_sha256"]
        assert branch["initial_replay_size"] == 0
        assert branch["initial_optimizer_state_size"] == 0
        assert branch["training"]["env_steps"] == 16
        assert {row["steps"] for row in branch["training"]["strata"]} == {4}
        assert len(branch["training"]["strata"]) == 4
        assert branch["training"]["timeouts"] > 0
        assert branch["training"]["final_epsilon"] == pytest.approx(config.epsilon_end)
        assert branch["training"]["final_shaping_scale"] == 0
        assert branch["reference_sha256"] == seed_result["pretrained_checkpoint_sha256"]
        assert [point["env_steps"] for point in branch["learning_curve"]] == [0, 8, 16]
        for point in branch["learning_curve"]:
            payload = json.loads(Path(point["report"]).read_text())
            assert Path(payload["checkpoint_path"]).is_file()
            assert payload["network_sha256"] == point["network_sha256"]
    assert branches["0.99"]["league_directory"] != branches["0.999"]["league_directory"]
    before = len(calls)
    assert run_pilot(config, tmp_path / "pilot") == result
    assert len(calls) == before
    checkpoint = tmp_path / "pilot" / "seed_11" / "gamma_0.99" / "final.pth"
    checkpoint.write_bytes(b"changed artifact")
    with pytest.raises(ValueError, match="artifact changed"):
        run_pilot(config, tmp_path / "pilot")


def test_pilot_restores_global_configs_rng_device_and_threads_on_error(tmp_path, monkeypatch):
    saved_game, saved_ai = game_config.model_dump(), ai_config.model_dump()
    random_state, numpy_state = random.getstate(), np.random.get_state()
    torch_state, threads = torch.get_rng_state(), torch.get_num_threads()

    def fail(*args, **kwargs):
        game_config.BONUSES_ENABLED = not game_config.BONUSES_ENABLED
        ai_config.MICRO_REWARD_SCALE = 0.12
        random.random()
        np.random.random()
        torch.rand(1)
        raise RuntimeError("validation failed")

    monkeypatch.setattr(pilot, "evaluate_competitive_agent", fail)
    with pytest.raises(RuntimeError, match="validation failed"):
        run_pilot(tiny_config(), tmp_path / "pilot")
    assert game_config.model_dump() == saved_game
    assert ai_config.model_dump() == saved_ai
    assert random.getstate() == random_state
    np.testing.assert_array_equal(np.random.get_state()[1], numpy_state[1])
    assert torch.equal(torch.get_rng_state(), torch_state)
    assert torch.get_num_threads() == threads
    with pytest.raises(FileExistsError, match="incomplete"):
        run_pilot(tiny_config(), tmp_path / "pilot")


def test_manifest_rejects_conflicting_configuration_and_seed_filter(tmp_path, monkeypatch):
    # An invalid filter must fail before starting any workload.
    with pytest.raises(ValueError, match="seed_filter"):
        run_pilot(tiny_config(), tmp_path / "pilot", seed_filter=(99,))
    output = tmp_path / "conflicting"
    output.mkdir()
    (output / "manifest.json").write_text(json.dumps({"different": "experiment"}))
    with pytest.raises(ValueError, match="manifest"):
        run_pilot(tiny_config(), output)


@pytest.mark.parametrize(
    "kwargs",
    [
        {"env_steps": 3},
        {"pretraining_labels": -1},
        {"gammas": (1.0,)},
        {"seeds": ()},
        {"threads": 0},
        {"epsilon_decay_fraction": 0},
        {"shaping_decay_fraction": -1},
        {"eval_seeds": (10001,)},
    ],
)
def test_invalid_pilot_budget_or_protocol_is_rejected(kwargs):
    with pytest.raises(ValueError):
        tiny_config(**kwargs)
