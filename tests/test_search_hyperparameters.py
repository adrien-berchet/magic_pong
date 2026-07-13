"""Tests for the Optuna hyperparameter/architecture search script.

All tests mock training and evaluation -- real end-to-end training is covered
by the manual verification steps in the implementation plan, not here, to
keep this suite fast.
"""

from __future__ import annotations

from pathlib import Path
from typing import Any
from unittest.mock import MagicMock

import pytest

optuna = pytest.importorskip("optuna")

import search_hyperparameters as search_module  # noqa: E402
import train_optimized  # noqa: E402
from magic_pong.ai import tournament  # noqa: E402

BASE_TRIAL_PARAMS = {
    "learning_rate": 0.001,
    "batch_size": 64,
    "memory_size": 20000,
    "train_frequency": 4,
}


def test_build_search_space_geometric_mode() -> None:
    trial = optuna.trial.FixedTrial(
        {
            **BASE_TRIAL_PARAMS,
            "architecture_mode": "geometric",
            "hidden_size": 128,
            "layer_size": 2,
        }
    )

    params = search_module.build_search_space(trial)

    assert params["architecture_mode"] == "geometric"
    assert params["hidden_size"] == 128
    assert params["layer_size"] == 2
    assert "widths" not in params
    # gamma/epsilon_decay/epsilon_min are deliberately not part of the search space.
    assert "gamma" not in params
    assert "epsilon_decay" not in params
    assert "epsilon_min" not in params

    argv = search_module.build_trial_argv(
        params,
        trial_dir=Path("/tmp/trial_0000"),
        curriculum_episodes=10,
        anneal_episodes=20,
        selfplay_episodes=30,
        max_steps_per_episode=50,
    )
    assert "--gamma" not in argv
    assert "--epsilon_decay" not in argv
    assert "--epsilon_min" not in argv
    # Each pipeline stage's episode budget must be independently controllable
    # (not silently pinned to a shared/unrelated default).
    assert "--total_episodes" not in argv

    args = train_optimized.parse_arguments(argv)
    assert args.mode == "pipeline"
    assert args.pipeline_curriculum_episodes == 10
    assert args.pipeline_anneal_episodes == 20
    assert args.pipeline_selfplay_episodes == 30
    assert args.hidden_size == 128
    assert args.layer_size == 2
    assert args.hidden_layers is None
    # Left at train_optimized.py's own CLI defaults.
    assert args.gamma == pytest.approx(0.99)
    assert args.epsilon_decay == pytest.approx(0.998)
    assert args.epsilon_min == pytest.approx(0.05)


def test_build_search_space_explicit_mode() -> None:
    trial = optuna.trial.FixedTrial(
        {
            **BASE_TRIAL_PARAMS,
            "architecture_mode": "explicit",
            "n_layers": 2,
            "layer0_width": 128,
            "layer1_width": 64,
        }
    )

    params = search_module.build_search_space(trial)

    assert params["architecture_mode"] == "explicit"
    assert params["widths"] == [128, 64]

    argv = search_module.build_trial_argv(
        params,
        trial_dir=Path("/tmp/trial_0000"),
        curriculum_episodes=10,
        anneal_episodes=10,
        selfplay_episodes=10,
        max_steps_per_episode=50,
    )
    args = train_optimized.parse_arguments(argv)
    assert args.hidden_layers == [128, 64]


def test_objective_wires_training_and_evaluation(
    monkeypatch: pytest.MonkeyPatch, tmp_path: Path
) -> None:
    checkpoint_path = tmp_path / "trial_0000" / "dqn_final.pth"

    def fake_run_trial_training(argv: list[str]) -> Path:
        checkpoint_path.parent.mkdir(parents=True, exist_ok=True)
        checkpoint_path.touch()
        return checkpoint_path

    fake_opponent = MagicMock()
    fake_opponent.win_rate = 0.75
    fake_result = MagicMock()
    fake_result.opponents = [fake_opponent]

    monkeypatch.setattr(search_module, "run_trial_training", fake_run_trial_training)
    monkeypatch.setattr(
        search_module.evaluation, "evaluate_checkpoint", lambda path, config: fake_result
    )
    monkeypatch.setattr(search_module.evaluation, "result_to_dict", lambda result: {"fake": True})

    trial = optuna.trial.FixedTrial(
        {
            **BASE_TRIAL_PARAMS,
            "architecture_mode": "geometric",
            "hidden_size": 128,
            "layer_size": 2,
        }
    )

    score = search_module.objective(
        trial,
        curriculum_episodes=10,
        anneal_episodes=10,
        selfplay_episodes=10,
        max_steps_per_episode=50,
        eval_episodes=5,
        output_root=tmp_path,
    )

    assert score == pytest.approx(0.75)
    assert trial.user_attrs["checkpoint_path"] == str(checkpoint_path)
    assert trial.user_attrs["eval_result"] == {"fake": True}


def test_run_final_tournament_selects_top_n_and_builtins(
    monkeypatch: pytest.MonkeyPatch, tmp_path: Path
) -> None:
    study = optuna.create_study(direction="maximize")
    for number, value in enumerate((0.9, 0.5, 0.7)):
        study.add_trial(
            optuna.trial.create_trial(
                state=optuna.trial.TrialState.COMPLETE,
                value=value,
                params={},
                distributions={},
                user_attrs={"checkpoint_path": f"/models/trial_{number}.pth"},
            )
        )

    captured: dict[str, Any] = {}

    def fake_run_tournament(
        participants: list[tournament.Participant], **kwargs: Any
    ) -> dict[str, Any]:
        captured["participants"] = participants
        return {}

    monkeypatch.setattr(search_module.tournament, "run_tournament", fake_run_tournament)

    search_module.run_final_tournament(study, top_n=2, episodes_per_match=5, output_root=tmp_path)

    participants = captured["participants"]
    dqn_participants = [p for p in participants if p.kind == "dqn"]
    builtin_participants = [p for p in participants if p.kind == "builtin"]

    # Values are 0.9 (trial 0), 0.5 (trial 1), 0.7 (trial 2) -> top 2 are trials 0 and 2.
    assert {p.checkpoint_path for p in dqn_participants} == {
        "/models/trial_0.pth",
        "/models/trial_2.pth",
    }
    assert len(builtin_participants) == len(tournament.BUILTIN_AI_TYPES)
