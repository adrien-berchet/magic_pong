"""DAgger distillation collects planner labels from real matches and fits the network."""

import random

import numpy as np
import pytest
import torch

from magic_pong.ai.distillation import PlannerDistiller
from magic_pong.ai.distillation import _DaggerPlayer
from magic_pong.ai.distillation import main
from magic_pong.ai.models.dqn_ai import ACTION_MAPPING
from magic_pong.ai.models.dqn_ai import DQNAgent
from magic_pong.ai.physics_teacher import TeacherDecision
from magic_pong.ai.pretraining import create_pretrainer
from magic_pong.core.game_engine import GameEngine
from magic_pong.utils.config import game_config


def _agent() -> DQNAgent:
    torch.manual_seed(0)
    return DQNAgent(hidden_layers=[32, 16])


def test_create_pretrainer_supports_distill_mode() -> None:
    pretrainer = create_pretrainer(mode="distill", opponents=("dummy",), backend="python")
    assert isinstance(pretrainer, PlannerDistiller)
    with pytest.raises(ValueError):
        create_pretrainer(mode="distill", opponents=("nobody",))


def test_distiller_rejects_invalid_schedules() -> None:
    with pytest.raises(ValueError):
        PlannerDistiller(beta_start=0.2, beta_end=0.5, backend="python")
    with pytest.raises(ValueError):
        PlannerDistiller(match_steps=0, backend="python")


def test_dagger_player_executes_planner_action_when_beta_is_one(monkeypatch) -> None:
    distiller = PlannerDistiller(
        opponents=("dummy",), backend="python", teacher_horizon=6, teacher_action_steps=3
    )
    agent = _agent()
    distiller._attach(agent)
    monkeypatch.setattr(
        distiller.teacher,
        "label",
        lambda physics, player_id: TeacherDecision(4, np.eye(9, dtype=np.float32)[4], {}),
    )
    engine = GameEngine(headless=True)
    learner = _DaggerPlayer(distiller, beta=1.0)
    engine.set_players(_DaggerPlayer(distiller, beta=1.0), learner)
    observation = engine.ai_environment.observation_processor.process_game_state(
        engine.physics_engine.get_game_state(), 2
    )
    action = learner.get_action(observation)
    # Action 4 is "right" in the player-1 frame, i.e. toward the goal line for player 2.
    assert action.move_x == -ACTION_MAPPING[4].move_x
    assert action.move_y == ACTION_MAPPING[4].move_y
    assert len(distiller.samples) == 1
    assert distiller.samples[0][0].shape == (agent.state_size,)
    assert distiller.samples[0][1].shape == (9,)


def test_run_pretraining_phase_collects_labels_and_updates_the_network(monkeypatch) -> None:
    monkeypatch.setattr(game_config, "BONUSES_ENABLED", False)
    random.seed(1)
    np.random.seed(1)
    agent = _agent()
    before = [p.detach().clone() for p in agent.q_network.parameters()]
    distiller = PlannerDistiller(
        opponents=("dummy", "tracker"),
        backend="python",
        teacher_horizon=6,
        teacher_action_steps=3,
        match_steps=40,
        match_score=1,
        batch_size=16,
        epochs_per_match=0.5,
        final_epochs=1,
        seed=3,
    )
    stats = distiller.run_pretraining_phase(
        agent, total_steps=100, steps_per_batch=50, verbose=False
    )
    assert stats["mode"] == "distill"
    assert stats["total_steps"] >= 100
    assert stats["matches"] >= 3
    assert set(stats["opponent_counts"]) == {"dummy", "tracker"}
    assert len(distiller.samples) == stats["total_steps"]
    assert 0.0 <= stats["label_agreement"] <= 1.0
    assert np.isnan(stats["decisive_agreement"]) or 0.0 <= stats["decisive_agreement"] <= 1.0
    assert any(
        not torch.equal(old, new.detach()) for old, new in zip(before, agent.q_network.parameters())
    )
    for key, value in agent.q_network.state_dict().items():
        assert torch.equal(value, agent.target_network.state_dict()[key])
    sides = {record["side"] for record in stats["match_log"]}
    assert sides == {1, 2}
    assert game_config.MAX_SCORE == 11  # match_score override is restored


def test_soft_targets_are_sharp_for_decisive_states_and_flat_for_ties() -> None:
    distiller = PlannerDistiller(backend="python", temperature=1.0, tie_break_weight=0.0)
    values = torch.tensor([[0.0] * 9, [1000.0] + [0.0] * 8])
    targets = distiller._targets(values)
    assert torch.allclose(targets[0], torch.full((9,), 1 / 9))
    assert targets[1, 0] > 0.999


def test_tie_break_weight_keeps_mass_on_the_planner_choice_when_values_tie() -> None:
    distiller = PlannerDistiller(backend="python", temperature=0.03, tie_break_weight=0.1)
    values = torch.tensor([[0.0] * 9])
    targets = distiller._targets(values)
    assert targets[0, 0] == pytest.approx(0.9 / 9 + 0.1)
    assert targets[0, 1] == pytest.approx(0.9 / 9)
    assert targets.sum().item() == pytest.approx(1.0)
    with pytest.raises(ValueError):
        PlannerDistiller(backend="python", tie_break_weight=1.5)


def test_decisive_agreement_ignores_tied_states() -> None:
    distiller = PlannerDistiller(backend="python")
    agent = _agent()
    distiller._attach(agent)
    state = np.zeros(agent.state_size, dtype=np.float32)
    distiller.samples.append((state, np.zeros(9, dtype=np.float32)))  # a tie
    assert np.isnan(distiller.agreement(decisive_gap=0.1))
    decisive = np.zeros(9, dtype=np.float32)
    decisive[distiller.network_action(state)] = 5.0
    distiller.samples.append((state, decisive))
    assert distiller.agreement(decisive_gap=0.1) == pytest.approx(1.0)


def test_beta_anneals_linearly() -> None:
    distiller = PlannerDistiller(beta_start=0.8, beta_end=0.2, backend="python")
    assert distiller.beta(0.0) == pytest.approx(0.8)
    assert distiller.beta(0.5) == pytest.approx(0.5)
    assert distiller.beta(2.0) == pytest.approx(0.2)


def test_cli_writes_checkpoint_and_stats(tmp_path, monkeypatch) -> None:
    monkeypatch.setattr(game_config, "BONUSES_ENABLED", False)
    output = tmp_path / "distilled.pth"
    code = main(
        [
            "--labels",
            "60",
            "--output",
            str(output),
            "--hidden_layers",
            "16,8",
            "--opponents",
            "dummy",
            "--teacher_horizon",
            "6",
            "--teacher_action_steps",
            "3",
            "--match_steps",
            "30",
            "--match_score",
            "1",
            "--batch_size",
            "8",
            "--final_epochs",
            "1",
        ]
    )
    assert code == 0
    assert output.exists()
    assert output.with_suffix(".distillation.json").exists()
    loaded = DQNAgent()
    loaded.load_model(str(output))
    assert loaded.hidden_layers == [16, 8]
