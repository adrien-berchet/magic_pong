"""Gameplay selection, fresh panels and fail-closed artifact boundaries."""

import importlib.util
import json
import sys
from pathlib import Path

import numpy as np
import pytest
import torch

SCRIPTS = Path(__file__).resolve().parents[1] / "docs" / "ai"
sys.path.insert(0, str(SCRIPTS))
spec = importlib.util.spec_from_file_location(
    "combo_runner", SCRIPTS / "run_planner_geometry_combo.py"
)
experiment = importlib.util.module_from_spec(spec)
spec.loader.exec_module(experiment)


def counts(epoch, wins=30, losses=10, margin=0, matches=64):
    return {
        "epoch": epoch,
        "wins": wins,
        "losses": losses,
        "goal_margin": margin,
        "matches": matches,
    }


def test_gameplay_rank_uses_wins_then_losses_then_margin_then_earliest():
    candidates = [
        counts(5, 29, 0, 100),
        counts(10, 30, 11, 100),
        counts(20, 30, 10, 1),
        counts(15, 30, 10, 1),
        counts(17, 30, 10, 0),
    ]
    selected = experiment.select_candidate(candidates, counts(0))
    assert selected["best_trained_epoch"] == 15
    assert selected["promotion_epoch"] == 15
    assert selected["gate_passed"] is True


@pytest.mark.parametrize("candidate", [counts(10, 29, 9), counts(10, 31, 11)])
def test_gate_failure_preserves_trained_candidate_for_diagnostic(candidate):
    result = experiment.select_candidate([candidate], counts(0))
    assert result["best_trained_epoch"] == 10
    assert result["promotion_epoch"] == 0
    assert result["gate_passed"] is False
    assert result["selected_validation"] == candidate


def test_selector_rejects_original_duplicates_and_incomparable_panels():
    for rows in ([], [counts(0)], [counts(5), counts(5)], [counts(5, matches=128)]):
        with pytest.raises(ValueError):
            experiment.select_candidate(rows, counts(0))


def test_candidate_grid_includes_dev_best_without_duplicates_or_epoch_zero():
    assert experiment.candidate_epochs(17) == [5, 10, 15, 17, 20]
    assert experiment.candidate_epochs(20) == [5, 10, 15, 20]
    assert experiment.candidate_epochs(0) == [5, 10, 15, 20]


def panel(split):
    settings = experiment.SETTINGS
    seeds = settings["validation_seeds"] if split == "validation" else settings["heldout_seeds"]
    return {
        "split": split,
        "matches": [
            {"opponent": o, "player_side": s, "bonuses_enabled": b, "seed": k}
            for o in settings["opponents"]
            for s in settings["sides"]
            for b in settings["bonuses"]
            for k in seeds
        ],
    }


def test_validation_and_heldout_panels_are_disjoint_and_exact():
    validation, heldout = panel("validation"), panel("held_out")
    experiment.assert_panel(validation, "validation")
    experiment.assert_panel(heldout, "held_out")
    assert len(validation["matches"]) == 64
    assert len(heldout["matches"]) == 128
    old_seeds = {row["seed"] for row in experiment.previous.test_matches()}
    old_seeds.update(experiment.previous.SETTINGS["gameplay_seeds"])
    old_seeds.update(row["seed"] for row in experiment.prior.match_specs())
    new_seeds = set(experiment.SETTINGS["validation_seeds"] + experiment.SETTINGS["heldout_seeds"])
    assert old_seeds.isdisjoint(new_seeds)
    assert set(experiment.SETTINGS["validation_seeds"]).isdisjoint(
        experiment.SETTINGS["heldout_seeds"]
    )
    with pytest.raises(ValueError):
        experiment.assert_panel(heldout, "validation")
    validation["matches"][-1] = validation["matches"][0]
    with pytest.raises(ValueError):
        experiment.assert_panel(validation, "validation")


def test_heldout_is_blocked_until_all_selections_are_sealed(tmp_path, monkeypatch):
    (tmp_path / "manifest.json").write_text("{}")
    for seed in (11, 22):
        directory = tmp_path / f"seed_{seed}"
        directory.mkdir()
        selection = directory / "selection.json"
        selection.write_text("{}")
        experiment.seal(tmp_path, directory / "selection_integrity.json", [selection])
    monkeypatch.setattr(
        experiment, "context", lambda output, seed: ({}, None, None, output / f"seed_{seed}")
    )
    monkeypatch.setattr(
        experiment,
        "evaluate_games",
        lambda *args: pytest.fail("Heldout opened before all selection"),
    )
    with pytest.raises(FileNotFoundError):
        experiment.heldout(tmp_path, 11)


def test_closure_rejects_missing_required_artifacts_and_mutation(tmp_path):
    (tmp_path / "manifest.json").write_text("{}")
    data = tmp_path / "selection.json"
    data.write_text("original")
    closure = tmp_path / "selection_integrity.json"
    experiment.seal(tmp_path, closure, [data])
    experiment.verify_closure(tmp_path, closure, [data])
    with pytest.raises(ValueError, match="Incomplete"):
        experiment.verify_closure(tmp_path, closure, [tmp_path / "unknown.json"])
    data.write_text("mutated")
    with pytest.raises(ValueError):
        experiment.verify_closure(tmp_path, closure, [data])


def test_geometry_ce_reproduction_checks_weights_and_complete_history(tmp_path):
    directory = tmp_path / "seed_11"
    directory.mkdir()
    weights = {"projection": torch.zeros(3, 2)}
    torch.save(weights, directory / "geometry.pt")
    history = [{"epoch": 0, "train": {"count": 2}, "dev": {"count": 1}}]
    (directory / "geometry_fit.json").write_text(
        json.dumps({"selected_epoch": 0, "history": history})
    )
    record = {"dev_best_epoch": 0, "history": history}
    result = experiment.check_ce_reproduction(record, weights, 11, {"parent_root": str(tmp_path)})
    assert result["exact_weights"] and result["exact_history"]
    with pytest.raises(AssertionError, match="weights mismatch"):
        experiment.check_ce_reproduction(
            record, {"projection": torch.ones(3, 2)}, 11, {"parent_root": str(tmp_path)}
        )


def test_manifest_rejects_changed_input_and_missing_archive(tmp_path):
    archive = tmp_path / "archive" / "magic_pong"
    archive.mkdir(parents=True)
    source = archive / "__init__.py"
    source.write_text("# pinned archive\n")
    manifest = {
        "schema": experiment.SCHEMA,
        "settings": experiment.SETTINGS,
        "feature_names": list(experiment.previous.FEATURE_NAMES),
        "application_archive": str(archive.parent),
        "runtime": {
            "torch": torch.__version__,
            "numpy": np.__version__,
            "numba": experiment.importlib.metadata.version("numba"),
        },
        "input_sha256": {str(source.resolve()): experiment.sha(source)},
    }
    (tmp_path / "manifest.json").write_text(json.dumps(manifest))
    experiment.read_manifest(tmp_path)
    source.write_text("# changed source\n")
    with pytest.raises(ValueError):
        experiment.read_manifest(tmp_path)


def test_game_counts_include_timeouts_goals_and_frames():
    result = {
        "matches": [
            {"agent_won": True, "timeout": False, "goals_for": 3, "goals_against": 2, "steps": 100},
            {
                "agent_won": False,
                "timeout": False,
                "goals_for": 1,
                "goals_against": 3,
                "steps": 200,
            },
            {
                "agent_won": False,
                "timeout": True,
                "goals_for": 0,
                "goals_against": 1,
                "steps": 6000,
            },
        ]
    }
    assert experiment.summarize_games(result) == {
        "matches": 3,
        "wins": 1,
        "losses": 1,
        "timeouts": 1,
        "goals_for": 4,
        "goals_against": 6,
        "goal_margin": -2,
        "frames": 6300,
    }
