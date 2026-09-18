"""Tests for the Optuna hyperparameter/architecture search script.

All tests mock training and evaluation -- real end-to-end training is covered
by the manual verification steps in the implementation plan, not here, to
keep this suite fast.
"""

from __future__ import annotations

import sys
from pathlib import Path
from typing import Any
from unittest.mock import MagicMock

import pytest
import torch

optuna = pytest.importorskip("optuna")

import search_hyperparameters as search_module  # noqa: E402
import train_optimized  # noqa: E402
from magic_pong.ai import tournament  # noqa: E402

BASE_TRIAL_PARAMS = {
    "learning_rate": 0.001,
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
    # gamma/epsilon_decay/epsilon_min/batch_size/memory_size are deliberately
    # not part of the search space.
    assert "gamma" not in params
    assert "epsilon_decay" not in params
    assert "epsilon_min" not in params
    assert "batch_size" not in params
    assert "memory_size" not in params

    argv = search_module.build_trial_argv(
        params,
        trial_dir=Path("/tmp/trial_0000"),
        curriculum_episodes=10,
        anneal_episodes=20,
        selfplay_episodes=30,
        max_steps_per_episode=50,
        early_stopping=150,
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
    assert args.early_stopping == 150
    assert args.hidden_size == 128
    assert args.layer_size == 2
    assert args.hidden_layers is None
    # Left at train_optimized.py's own CLI defaults.
    assert args.gamma == pytest.approx(0.99)
    assert args.epsilon_decay == pytest.approx(0.998)
    assert args.epsilon_min == pytest.approx(0.05)
    # Fixed to the empirically best values, not searched.
    assert args.batch_size == search_module.FIXED_BATCH_SIZE
    assert args.memory_size == search_module.FIXED_MEMORY_SIZE


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
    # early_stopping defaults to 0 (disabled) when omitted -- no behavior
    # change for callers that don't pass it.
    assert args.early_stopping == 0


def test_objective_wires_training_and_evaluation(
    monkeypatch: pytest.MonkeyPatch, tmp_path: Path
) -> None:
    checkpoint_path = tmp_path / "trial_0000" / "dqn_final.pth"

    def fake_run_trial_training(argv: list[str]) -> Path:
        checkpoint_path.parent.mkdir(parents=True, exist_ok=True)
        checkpoint_path.touch()
        return checkpoint_path

    fake_result = MagicMock()
    configs = []

    def evaluate(path, config):
        configs.append(config)
        return fake_result

    monkeypatch.setattr(search_module, "run_trial_training", fake_run_trial_training)
    monkeypatch.setattr(
        search_module.competitive,
        "evaluate_competitive_checkpoint",
        evaluate,
    )
    monkeypatch.setattr(
        search_module.competitive, "competitive_result_to_dict", lambda result: {"fake": True}
    )
    monkeypatch.setattr(
        search_module.competitive, "competitive_rank", lambda result: (0.55, 0.75, 1.0, 0.9)
    )

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
        early_stopping=100,
    )

    assert score == pytest.approx(0.55)
    assert trial.user_attrs["checkpoint_path"] == str(checkpoint_path)
    assert trial.user_attrs["eval_result"] == {"fake": True}
    assert configs[0].seeds == (101, 102, 103, 104, 105)
    assert configs[0].player_sides == (1, 2)
    assert configs[0].bonus_modes == (False, True)
    assert configs[0].max_score == 11


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


def test_split_trial_count_even() -> None:
    assert search_module.split_trial_count(100, 4) == [25, 25, 25, 25]


def test_split_trial_count_remainder_goes_to_first_workers() -> None:
    assert search_module.split_trial_count(10, 3) == [4, 3, 3]


def test_split_trial_count_single_worker() -> None:
    assert search_module.split_trial_count(7, 1) == [7]


def test_split_trial_count_more_workers_than_trials() -> None:
    assert search_module.split_trial_count(2, 5) == [1, 1, 0, 0, 0]


def _make_fake_popen(
    exit_codes: list[int] | None = None,
) -> tuple[list[dict[str, Any]], Any]:
    """Build a fake subprocess.Popen substitute that records every call's
    argv/env instead of spawning a real process."""
    calls: list[dict[str, Any]] = []

    class _FakeProcess:
        def __init__(self, code: int) -> None:
            self._code = code

        def wait(self) -> int:
            return self._code

    def fake_popen(argv: list[str], *, stdout: Any, stderr: Any, env: dict[str, str]) -> Any:
        index = len(calls)
        calls.append({"argv": argv, "stdout": stdout, "stderr": stderr, "env": env})
        code = exit_codes[index] if exit_codes is not None else 0
        return _FakeProcess(code)

    return calls, fake_popen


def test_main_with_n_workers_spawns_subprocesses_and_runs_tournament_once(
    monkeypatch: pytest.MonkeyPatch, tmp_path: Path
) -> None:
    calls, fake_popen = _make_fake_popen()
    monkeypatch.setattr(search_module.subprocess, "Popen", fake_popen)
    monkeypatch.setattr(search_module, "dqn_device", torch.device("cpu"))

    tournament_calls: list[dict[str, Any]] = []
    monkeypatch.setattr(
        search_module,
        "run_final_tournament",
        lambda study, **kwargs: tournament_calls.append(kwargs) or {},
    )

    search_module.main(
        [
            "--study-name",
            "s",
            "--output-root",
            str(tmp_path),
            "--n-trials",
            "10",
            "--n-workers",
            "3",
            "--early-stopping",
            "200",
        ]
    )

    assert len(calls) == 3
    for call in calls:
        assert call["argv"][0] == sys.executable
        assert call["argv"][1] == search_module.__file__

    n_trials_per_call = [call["argv"][call["argv"].index("--n-trials") + 1] for call in calls]
    assert n_trials_per_call == ["4", "3", "3"]

    seeds = [call["argv"][call["argv"].index("--seed") + 1] for call in calls]
    assert seeds == ["0", "1", "2"]

    for call in calls:
        assert "--skip-tournament" in call["argv"]
        assert "--n-workers" not in call["argv"]
        assert call["argv"][call["argv"].index("--early-stopping") + 1] == "200"

    assert len(tournament_calls) == 1


def test_run_parallel_workers_forces_cpu_by_default_when_gpu_present(
    monkeypatch: pytest.MonkeyPatch, tmp_path: Path
) -> None:
    calls, fake_popen = _make_fake_popen()
    monkeypatch.setattr(search_module.subprocess, "Popen", fake_popen)
    monkeypatch.setattr(search_module, "dqn_device", torch.device("cuda"))
    monkeypatch.delenv("CUDA_VISIBLE_DEVICES", raising=False)

    args = search_module.parse_search_args(
        [
            "--study-name",
            "s",
            "--output-root",
            str(tmp_path),
            "--n-trials",
            "4",
            "--n-workers",
            "2",
        ]
    )
    output_root = tmp_path / "s"
    output_root.mkdir()
    search_module.run_parallel_workers(args, output_root)

    assert len(calls) == 2
    for call in calls:
        assert call["env"]["CUDA_VISIBLE_DEVICES"] == ""


def test_run_parallel_workers_allow_gpu_workers_leaves_env_untouched(
    monkeypatch: pytest.MonkeyPatch, tmp_path: Path
) -> None:
    calls, fake_popen = _make_fake_popen()
    monkeypatch.setattr(search_module.subprocess, "Popen", fake_popen)
    monkeypatch.setattr(search_module, "dqn_device", torch.device("cuda"))
    monkeypatch.delenv("CUDA_VISIBLE_DEVICES", raising=False)

    args = search_module.parse_search_args(
        [
            "--study-name",
            "s",
            "--output-root",
            str(tmp_path),
            "--n-trials",
            "4",
            "--n-workers",
            "2",
            "--allow-gpu-workers",
        ]
    )
    output_root = tmp_path / "s"
    output_root.mkdir()
    search_module.run_parallel_workers(args, output_root)

    assert len(calls) == 2
    for call in calls:
        assert "CUDA_VISIBLE_DEVICES" not in call["env"]


def test_run_parallel_workers_no_gpu_leaves_env_untouched(
    monkeypatch: pytest.MonkeyPatch, tmp_path: Path
) -> None:
    calls, fake_popen = _make_fake_popen()
    monkeypatch.setattr(search_module.subprocess, "Popen", fake_popen)
    monkeypatch.setattr(search_module, "dqn_device", torch.device("cpu"))
    monkeypatch.delenv("CUDA_VISIBLE_DEVICES", raising=False)

    args = search_module.parse_search_args(
        [
            "--study-name",
            "s",
            "--output-root",
            str(tmp_path),
            "--n-trials",
            "4",
            "--n-workers",
            "2",
        ]
    )
    output_root = tmp_path / "s"
    output_root.mkdir()
    search_module.run_parallel_workers(args, output_root)

    assert len(calls) == 2
    for call in calls:
        assert "CUDA_VISIBLE_DEVICES" not in call["env"]


def test_run_parallel_workers_pins_thread_count_per_worker(
    monkeypatch: pytest.MonkeyPatch, tmp_path: Path
) -> None:
    calls, fake_popen = _make_fake_popen()
    monkeypatch.setattr(search_module.subprocess, "Popen", fake_popen)
    monkeypatch.setattr(search_module, "dqn_device", torch.device("cpu"))
    monkeypatch.setattr(search_module.os, "cpu_count", lambda: 20)

    args = search_module.parse_search_args(
        [
            "--study-name",
            "s",
            "--output-root",
            str(tmp_path),
            "--n-trials",
            "4",
            "--n-workers",
            "4",
        ]
    )
    output_root = tmp_path / "s"
    output_root.mkdir()
    search_module.run_parallel_workers(args, output_root)

    assert len(calls) == 4
    for call in calls:
        for var in search_module.THREAD_LIMIT_ENV_VARS:
            assert call["env"][var] == "5"  # 20 cpus // 4 workers


def test_run_parallel_workers_thread_pinning_rounds_down_to_at_least_one(
    monkeypatch: pytest.MonkeyPatch, tmp_path: Path
) -> None:
    calls, fake_popen = _make_fake_popen()
    monkeypatch.setattr(search_module.subprocess, "Popen", fake_popen)
    monkeypatch.setattr(search_module, "dqn_device", torch.device("cpu"))
    monkeypatch.setattr(search_module.os, "cpu_count", lambda: 2)

    args = search_module.parse_search_args(
        [
            "--study-name",
            "s",
            "--output-root",
            str(tmp_path),
            "--n-trials",
            "8",
            "--n-workers",
            "8",
        ]
    )
    output_root = tmp_path / "s"
    output_root.mkdir()
    search_module.run_parallel_workers(args, output_root)

    assert len(calls) == 8
    for call in calls:
        assert call["env"]["OMP_NUM_THREADS"] == "1"  # max(1, 2 // 8)


def test_run_parallel_workers_reports_failed_worker(
    monkeypatch: pytest.MonkeyPatch, tmp_path: Path, capsys: pytest.CaptureFixture[str]
) -> None:
    calls, fake_popen = _make_fake_popen(exit_codes=[0, 1])
    monkeypatch.setattr(search_module.subprocess, "Popen", fake_popen)
    monkeypatch.setattr(search_module, "dqn_device", torch.device("cpu"))

    args = search_module.parse_search_args(
        [
            "--study-name",
            "s",
            "--output-root",
            str(tmp_path),
            "--n-trials",
            "4",
            "--n-workers",
            "2",
        ]
    )
    output_root = tmp_path / "s"
    output_root.mkdir()
    search_module.run_parallel_workers(args, output_root)

    del calls
    captured = capsys.readouterr()
    assert "worker 1" in captured.out
    assert "FAILED" in captured.out
    assert str(output_root / "worker_1.log") in captured.out


def test_main_runs_tournament_despite_failed_worker(
    monkeypatch: pytest.MonkeyPatch, tmp_path: Path
) -> None:
    _calls, fake_popen = _make_fake_popen(exit_codes=[1, 0])
    monkeypatch.setattr(search_module.subprocess, "Popen", fake_popen)
    monkeypatch.setattr(search_module, "dqn_device", torch.device("cpu"))

    tournament_calls: list[dict[str, Any]] = []
    monkeypatch.setattr(
        search_module,
        "run_final_tournament",
        lambda study, **kwargs: tournament_calls.append(kwargs) or {},
    )

    search_module.main(
        [
            "--study-name",
            "s",
            "--output-root",
            str(tmp_path),
            "--n-trials",
            "4",
            "--n-workers",
            "2",
        ]
    )

    assert len(tournament_calls) == 1


def test_main_single_worker_never_spawns_subprocess(
    monkeypatch: pytest.MonkeyPatch, tmp_path: Path
) -> None:
    def fail_if_called(*args: Any, **kwargs: Any) -> Any:
        raise AssertionError("subprocess.Popen should not be called when --n-workers <= 1")

    monkeypatch.setattr(search_module.subprocess, "Popen", fail_if_called)

    optimize_calls: list[int] = []
    monkeypatch.setattr(
        search_module,
        "run_study_optimize",
        lambda study, args, output_root: optimize_calls.append(1),
    )
    monkeypatch.setattr(search_module, "run_final_tournament", lambda study, **kwargs: {})

    search_module.main(["--study-name", "s1", "--output-root", str(tmp_path), "--n-trials", "3"])
    search_module.main(
        [
            "--study-name",
            "s2",
            "--output-root",
            str(tmp_path),
            "--n-trials",
            "3",
            "--n-workers",
            "1",
        ]
    )

    assert len(optimize_calls) == 2


def test_main_rejects_zero_workers(tmp_path: Path) -> None:
    with pytest.raises(SystemExit):
        search_module.main(
            [
                "--study-name",
                "s",
                "--output-root",
                str(tmp_path),
                "--n-trials",
                "3",
                "--n-workers",
                "0",
            ]
        )


def test_study_resume_rejects_legacy_objective_and_changed_protocol() -> None:
    legacy = optuna.create_study(direction="maximize")
    legacy.add_trial(optuna.trial.create_trial(value=0.95))
    with pytest.raises(ValueError, match="new study"):
        search_module.ensure_study_protocol(legacy, eval_episodes=3)

    study = optuna.create_study(direction="maximize")
    search_module.ensure_study_protocol(study, eval_episodes=3)
    search_module.ensure_study_protocol(study, eval_episodes=3)
    with pytest.raises(ValueError, match="new study"):
        search_module.ensure_study_protocol(study, eval_episodes=10)


def test_search_rejects_empty_validation_before_training() -> None:
    with pytest.raises(SystemExit):
        search_module.parse_search_args(["--study-name", "invalid", "--eval-episodes", "0"])
