from pathlib import Path
from types import SimpleNamespace

import pytest

from magic_pong.ai import training_validation as validation


class SavingAgent:
    def __init__(self):
        self.version = "initial"

    def save_model(self, path):
        Path(path).write_text(self.version)


def fake_results(monkeypatch, ranks):
    calls = []
    remaining = iter(ranks)

    def evaluate(agent, config, **kwargs):
        calls.append((agent, config, kwargs))
        return SimpleNamespace(rank=next(remaining))

    monkeypatch.setattr(validation.competitive, "evaluate_competitive_agent", evaluate)
    monkeypatch.setattr(validation.competitive, "competitive_rank", lambda r: r.rank)
    monkeypatch.setattr(validation.competitive, "is_better", lambda a, b: a.rank > b.rank)
    monkeypatch.setattr(
        validation.competitive, "competitive_result_to_dict", lambda r: {"rank": r.rank}
    )
    return calls


def test_selection_uses_validation_and_keeps_better_incumbent(monkeypatch, tmp_path):
    calls = fake_results(monkeypatch, [(0.4, 0.5), (0.2, 0.9), (0.5, 0.6)])
    path = tmp_path / "best.pth"
    selector = validation.CheckpointSelector(path, validation.competitive.CompetitiveConfig(), 5)
    agent = SavingAgent()
    assert selector.consider(agent, 0, force=True)
    agent.version = "worse"
    assert not selector.consider(agent, 4)
    assert not selector.consider(agent, 5)
    assert path.read_text() == "initial"
    agent.version = "stronger"
    assert selector.consider(agent, 10)
    assert path.read_text() == "stronger"
    assert selector.last_improved_episode == 10
    assert len(calls) == 3
    assert path.with_suffix(".validation.json").exists()


def test_phase_end_evaluation_is_not_repeated(monkeypatch, tmp_path):
    calls = fake_results(monkeypatch, [(0.4,), (0.5,)])
    selector = validation.CheckpointSelector(
        tmp_path / "best.pth", validation.competitive.CompetitiveConfig(), 5
    )
    agent = SavingAgent()
    selector.consider(agent, 0, force=True)
    selector.consider(agent, 2, force=True)
    selector.consider(agent, 2, force=True)
    assert len(calls) == 2


def test_existing_checkpoint_is_revalidated_before_replacement(monkeypatch, tmp_path):
    fake_results(monkeypatch, [(0.3,)])
    path = tmp_path / "best.pth"
    path.write_text("existing champion")
    seen = []

    def evaluate_checkpoint(path, config, **kwargs):
        seen.append(path)
        return SimpleNamespace(rank=(0.9,))

    monkeypatch.setattr(
        validation.competitive, "evaluate_competitive_checkpoint", evaluate_checkpoint
    )
    selector = validation.CheckpointSelector(path, validation.competitive.CompetitiveConfig(), 5)
    assert not selector.consider(SavingAgent(), 0, force=True)
    assert path.read_text() == "existing champion"
    assert seen == [path]
    report = path.with_suffix(".validation.json")
    assert report.exists()
    assert '"incumbent": true' in report.read_text()
    assert '"checked_training_episode": 0' in report.read_text()


def test_incumbent_uses_phase_local_patience_baseline(monkeypatch, tmp_path):
    fake_results(monkeypatch, [(0.3,), (0.2,)])
    path = tmp_path / "best.pth"
    path.write_text("existing champion")
    monkeypatch.setattr(
        validation.competitive,
        "evaluate_competitive_checkpoint",
        lambda path, config, **kwargs: SimpleNamespace(rank=(0.9,)),
    )
    selector = validation.CheckpointSelector(
        path,
        validation.competitive.CompetitiveConfig(),
        5,
        patience_baseline_episode=20,
    )
    assert not selector.consider(SavingAgent(), 20, force=True)
    assert selector.last_improved_episode == 20
    assert not selector.consider(SavingAgent(), 25)
    assert selector.last_improved_episode == 20


def test_config_separates_validation_and_heldout_seeds():
    args = SimpleNamespace(
        validation_seeds="101,102",
        test_seeds="9001,9002",
        validation_opponents="aggressive,dummy",
    )
    config = validation.validation_config_from_args(args, ("historical:1",))
    assert config.seeds == (101, 102)
    assert config.held_out_seeds == (9001, 9002)
    assert config.opponents == ("aggressive", "dummy", "historical:1")
    assert config.player_sides == (1, 2)
    assert config.bonus_modes == (False, True)
    assert config.max_score == 11
    args.test_seeds = "101"
    with pytest.raises(ValueError):
        validation.validation_config_from_args(args)
