"""Match pairing and selection/provenance checks for combo experiment exports."""

import copy
import importlib.metadata
import itertools
import json

import pytest

from docs.ai import analyze_planner_geometry_combo as analysis


def game(side, winner, seed=620000):
    won, timeout = winner == side, winner == 0
    return {
        "opponent": "tracker",
        "player_side": side,
        "bonuses_enabled": False,
        "seed": seed,
        "winner": winner,
        "agent_won": won,
        "timeout": timeout,
        "goals_for": 3 if won else 1,
        "goals_against": 1 if won or timeout else 3,
        "steps": 6000 if timeout else 1000,
    }


def test_pairing_uses_side_as_well_as_seed_and_preserves_raw_effects():
    original = [game(1, 2), game(2, 0)]
    improved = [game(1, 1), game(2, 2)]
    pair = analysis.pair_panels(original, improved)
    assert len(pair["matches"]) == 2
    assert pair["total_delta"] == {
        "win_delta": 2,
        "loss_delta": -1,
        "timeout_delta": -1,
        "goal_margin_delta": 6,
    }
    assert pair["matches"][0]["reference"] == original[0]
    assert pair["matches"][1]["challenger"] == improved[1]


def test_summary_keeps_timeouts_separate_and_rejects_invalid_identity():
    result = analysis.summarize_panel([game(1, 1), game(2, 1), game(2, 0, 620001)])
    assert (
        result["overall"]["wins"]
        == result["overall"]["losses"]
        == result["overall"]["timeouts"]
        == 1
    )
    assert result["side"]["2"]["timeouts"] == 1
    assert len(result["cells"]) == 2
    with pytest.raises(ValueError, match="Duplicate match"):
        analysis.summarize_panel([game(1, 1), game(1, 1)])
    bad = game(1, 1)
    bad["timeout"] = True
    with pytest.raises(ValueError, match="Outcome flags"):
        analysis.summarize_panel([bad])
    with pytest.raises(ValueError, match="pair exactly"):
        analysis.pair_panels([game(1, 1)], [game(1, 1, 620001)])


def panel(seeds, wins, losses):
    rows = []
    for index, (opponent, side, bonuses, seed) in enumerate(
        itertools.product(analysis.OPPONENTS, (1, 2), (False, True), seeds)
    ):
        winner = side if index < wins else 3 - side if index < wins + losses else 0
        rows.append({**game(side, winner, seed), "opponent": opponent, "bonuses_enabled": bonuses})
    return rows


def test_rank_trained_first_then_gate_without_substituting_a_lower_ranked_candidate():
    initial = panel(range(610000, 610004), 32, 16)
    winner = panel(range(610000, 610004), 33, 17)
    lower_ranked_gate_pass = panel(range(610000, 610004), 32, 15)
    result = analysis.reconstruct_selection(initial, {5: winner, 10: lower_ranked_gate_pass})
    assert result["best_trained_epoch"] == result["selected_validation"]["epoch"] == 5
    assert result["promotion_epoch"] == 0
    assert result["gate_passed"] is False
    # Ranking has no original-policy candidate, even when every trained model is worse.
    worse = panel(range(610000, 610004), 10, 20)
    result = analysis.reconstruct_selection(initial, {10: worse, 5: worse})
    assert result["best_trained_epoch"] == 5
    assert result["promotion_epoch"] == 0


def test_selection_ties_use_losses_then_goal_margin_then_earlier_epoch():
    initial = panel(range(610000, 610004), 10, 20)
    more_losses = panel(range(610000, 610004), 20, 16)
    fewer_losses = panel(range(610000, 610004), 20, 15)
    more_goals = copy.deepcopy(fewer_losses)
    more_goals[0]["goals_against"] = 0
    result = analysis.reconstruct_selection(
        initial, {5: more_losses, 10: fewer_losses, 15: more_goals, 20: more_goals}
    )
    assert result["best_trained_epoch"] == result["promotion_epoch"] == 15
    assert result["gate_passed"] is True
    with pytest.raises(ValueError, match="positive"):
        analysis.reconstruct_selection(initial, {0: initial})


@pytest.fixture
def completed_fixture(tmp_path):
    """Use the real runner's protocol and hash verifier on synthetic completed panels."""
    runner = analysis.runner_module()
    root = tmp_path / "experiment"
    source = root / "archive/magic_pong/example.py"
    source.parent.mkdir(parents=True)
    source.write_text("# Synthetic archived source\n")
    manifest = {
        "schema": runner.SCHEMA,
        "settings": runner.SETTINGS,
        "feature_names": list(runner.previous.FEATURE_NAMES),
        "application_archive": str(root / "archive"),
        "runtime": {
            "torch": runner.torch.__version__,
            "numpy": runner.np.__version__,
            "numba": importlib.metadata.version("numba"),
        },
        "input_sha256": {str(source): analysis.sha(source)},
    }
    analysis.write_json(root / "manifest.json", manifest)
    for seed in analysis.SEEDS:
        directory = root / f"seed_{seed}"
        training = {"seed": seed, "arms": {}}
        initial = {"matches": panel(range(610000, 610004), 32, 16)}
        analysis.write_json(directory / "validation_initial.json", initial)
        selection = {"seed": seed, "reference": runner.summarize_games(initial), "arms": {}}
        for arm in analysis.ARMS:
            epochs = [5, 7, 10, 15, 20]
            history = [
                {
                    "epoch": epoch,
                    **{
                        split: {
                            "all": {"mean_regret": 50 / (epoch + 1)},
                            "decisive": {"agreement": min(0.9, 0.1 + epoch * 0.05)},
                        }
                        for split in ("train", "dev")
                    },
                }
                for epoch in range(21)
            ]
            training["arms"][arm] = {
                "history": history,
                "updates": 2000,
                "dev_best_epoch": 7,
                "candidate_epochs": epochs,
                "feature_matrices_sha256": {"train": "synthetic", "dev": "synthetic"},
            }
            candidates = []
            for epoch in epochs:
                wins, losses = {5: (33, 17), 7: (31, 16), 10: (32, 15), 15: (20, 20), 20: (10, 30)}[
                    epoch
                ]
                if arm == "geometry_cost" and epoch == 10:
                    wins, losses = 35, 15
                result = {"matches": panel(range(610000, 610004), wins, losses)}
                analysis.write_json(directory / f"validation_{arm}_epoch_{epoch}.json", result)
                candidates.append({"epoch": epoch, **runner.summarize_games(result)})
            selection["arms"][arm] = runner.select_candidate(candidates, selection["reference"])
        analysis.write_json(directory / "training.json", training)
        analysis.write_json(directory / "selection.json", selection)
        for name, wins, losses in (
            ("initial", 64, 32),
            ("geometry_ce", 65, 34),
            ("geometry_cost", 66, 31),
        ):
            analysis.write_json(
                directory / f"heldout_{name}.json",
                {
                    "matches": panel(range(620000, 620008), wins, losses),
                    "selection": {
                        "arm": name,
                        "epoch": 0
                        if name == "initial"
                        else selection["arms"][name]["best_trained_epoch"],
                        "selection_sha256": analysis.sha(directory / "selection.json"),
                    },
                },
            )
        files = sorted(directory.glob("*.json"))
        runner.seal(root, directory / "selection_integrity.json", files)
        runner.seal(
            root, directory / "completion.json", [*files, directory / "selection_integrity.json"]
        )
    return root


def test_completed_schema_selection_pairing_and_unique_work_counts(completed_fixture):
    manifest, protected = analysis.verified_inputs(completed_fixture)
    runs = analysis.load_runs(completed_fixture, protected)
    summary = analysis.build_summary(
        manifest, runs, analysis.sha(completed_fixture / "manifest.json")
    )
    assert summary["work"] == {
        "new_supervised_labels": 0,
        "unique_validation_panels": 33,
        "unique_validation_matches": 2112,
        "unique_heldout_panels": 9,
        "unique_heldout_matches": 1152,
        "supervised_optimizer_updates": 12000,
    }
    ce = next(row for row in summary["summary"] if row["arm"] == "geometry_ce")
    assert ce["heldout_evaluated_epoch"] == 5
    assert ce["reporting_promotion_epoch"] == 0
    assert ce["heldout"]["overall"]["wins"] == 65  # Actual candidate, not original fallback.
    assert len(summary["paired_heldout"]) == 9
    assert all(len(pair["matches"]) == 128 for pair in summary["paired_heldout"])
    assert summary["seed_aggregates"]["geometry_cost"]["wins"] == {
        "valid_seeds": 3,
        "mean": 66,
        "min": 66,
        "max": 66,
    }


def test_loader_rejects_unsealed_files_wrong_grid_and_wrong_selection(completed_fixture):
    _, protected = analysis.verified_inputs(completed_fixture)
    path = completed_fixture / "seed_11/selection.json"
    data = json.loads(path.read_text())
    data["arms"]["geometry_ce"]["promotion_epoch"] = 5
    analysis.write_json(path, data)
    with pytest.raises(ValueError, match="selection does not reproduce"):
        analysis.load_runs(completed_fixture, protected)
    del protected[str(path)]
    with pytest.raises(ValueError, match="outside verified closures"):
        analysis.load_runs(completed_fixture, protected)
    with pytest.raises(ValueError, match="declared.*grid"):
        analysis.verify_panel_grid(panel(range(620000, 620008), 50, 50), range(610000, 610004))


@pytest.mark.parametrize(
    ("arm", "field", "value"),
    [
        ("initial", "epoch", 5),
        ("initial", "arm", "geometry_ce"),
        ("geometry_ce", "epoch", 0),  # Reporting fallback is not the evaluated candidate.
        ("geometry_ce", "arm", "geometry_cost"),
        ("geometry_cost", "epoch", 5),  # CE checkpoint must not masquerade as cost.
        ("geometry_cost", "selection_sha256", "stale-selection-hash"),
    ],
)
def test_loader_rejects_stale_or_swapped_heldout_metadata(completed_fixture, arm, field, value):
    _, protected = analysis.verified_inputs(completed_fixture)
    path = completed_fixture / f"seed_11/heldout_{arm}.json"
    data = json.loads(path.read_text())
    data["selection"][field] = value
    analysis.write_json(path, data)
    with pytest.raises(ValueError, match="Held-out selection metadata"):
        analysis.load_runs(completed_fixture, protected)


def test_verifier_requires_all_seeds_and_unchanged_source(completed_fixture):
    source = completed_fixture / "archive/magic_pong/example.py"
    source.write_text("# Changed source\n")
    with pytest.raises(ValueError, match="changed"):
        analysis.verified_inputs(completed_fixture)


def test_verifier_rejects_missing_completion(completed_fixture):
    (completed_fixture / "seed_33/completion.json").unlink()
    with pytest.raises((FileNotFoundError, ValueError)):
        analysis.verified_inputs(completed_fixture)


def test_export_is_reproducible_read_only_and_hashes_its_outputs(completed_fixture, tmp_path):
    prefix = tmp_path / "exports/result"
    _, before = analysis.verified_inputs(completed_fixture)
    result = analysis.export(completed_fixture, prefix)
    integrity_path = prefix.with_name(prefix.name + "_integrity.json")
    integrity = json.loads(integrity_path.read_text())
    assert integrity["posthoc_exporter"] is True
    assert integrity["input_sha256"] == before
    assert result["work"]["unique_heldout_matches"] == 1152
    analysis.check_hashes(integrity["export_sha256"])
    analysis.check_hashes(before)
    first_hash = analysis.sha(integrity_path)
    analysis.export(completed_fixture, prefix)
    assert analysis.sha(integrity_path) == first_hash
    analysis.check_hashes(integrity["export_sha256"])


def test_export_detects_input_mutation_during_plotting(completed_fixture, tmp_path, monkeypatch):
    source = completed_fixture / "archive/magic_pong/example.py"
    monkeypatch.setattr(analysis, "draw_plot", lambda *_: source.write_text("changed\n"))
    monkeypatch.setattr(analysis, "draw_imitation_plot", lambda *_: None)
    prefix = tmp_path / "mutated/result"
    with pytest.raises(ValueError, match="Protected artifact changed"):
        analysis.export(completed_fixture, prefix)
    assert not prefix.with_name(prefix.name + "_integrity.json").exists()
