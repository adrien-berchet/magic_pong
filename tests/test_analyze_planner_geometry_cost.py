"""Paired identities and provenance checks for the posthoc exporter."""

from __future__ import annotations

import copy
import json
from pathlib import Path

import pytest

from docs.ai import analyze_planner_geometry_cost as analysis


def group(rows, regret, agreement=0.5):
    return {
        "all": {"count": rows, "mean_regret": regret, "max_regret": regret * 2},
        "decisive": {"count": rows // 2, "agreement": agreement},
        "tail": {"fraction": 0.25, "count": rows // 4},
    }


def game(side, winner):
    won, timeout = winner == side, winner == 0
    return {
        "opponent": "tracker",
        "player_side": side,
        "bonuses_enabled": False,
        "seed": 520000,
        "winner": winner,
        "agent_won": won,
        "timeout": timeout,
        "steps": 6000 if timeout else 1000,
        "goals_for": 3 if won else 1,
        "goals_against": 1 if won or timeout else 3,
    }


def make_run(seed):
    records = [
        {"index": i, "seed": 510000 + i, "opponent": "tracker", "side": i + 1, "bonuses": False}
        for i in range(2)
    ]
    run = {"seed": seed, "collection": {"seed": seed, "matches": records, "rows": 10}, "arms": {}}
    for name, regret, winners in (
        ("initial", 10, (1, 2)),
        ("control", 4, (2, 0)),
        ("geometry", 2, (1, 2)),
        ("cost", 3, (1, 1)),
    ):
        matches = [game(side, winner) for side, winner in enumerate(winners, 1)]
        run["arms"][name] = {
            "selected_epoch": 0 if name == "initial" else 2,
            "train": group(12, regret),
            "dev": group(5, regret),
            "test": {
                "overall": group(10, regret),
                "category": {"incoming_contact": group(4, regret + 1)},
                "by_match": {"0": group(2, regret), "1": group(8, regret + 1)},
            },
            "fit": None if name == "initial" else {"updates": 2},
            "gameplay_matches": matches,
            "gameplay": analysis.gameplay_summary(matches),
        }
    return run


def manifest(tmp_path):
    return {
        "settings": analysis.runner.SETTINGS,
        "geometry": {},
        "feature_names": [],
        "input_sha256": {},
        "prior_root": str(tmp_path / "prior"),
        "application_archive": str(tmp_path / "baseline/source_snapshot"),
    }


def test_pairing_retains_full_game_identity_and_matches_state_populations():
    run = make_run(11)
    result = analysis.paired_matches(run, "geometry")
    assert len(result["gameplay_matches"]) == 2  # Same RNG seed, different player sides.
    assert result["mean_match_mean_regret_delta"] == -2
    assert [r["rows"] for r in result["supervised_matches"]] == [2, 8]
    assert result["gameplay_total_delta"] == {
        "win_delta": 2,
        "loss_delta": -1,
        "timeout_delta": -1,
        "goal_margin_delta": 6,
    }
    run["arms"]["geometry"]["test"]["by_match"]["1"]["decisive"]["count"] += 1
    with pytest.raises(ValueError, match="different row populations"):
        analysis.paired_matches(run, "geometry")


def test_pairing_rejects_unmatched_game_seeds_and_duplicate_outcomes():
    run = make_run(11)
    run["arms"]["cost"]["gameplay_matches"][0]["seed"] += 1
    with pytest.raises(ValueError, match="do not pair exactly"):
        analysis.paired_matches(run, "cost")
    row = game(1, 1)
    with pytest.raises(ValueError, match="Duplicate gameplay"):
        analysis.gameplay_summary([row, row])
    row["timeout"] = True
    with pytest.raises(ValueError, match="outcome fields disagree"):
        analysis.gameplay_summary([row])


def test_summary_has_four_arms_and_reuses_control_without_double_counting(tmp_path):
    runs = [make_run(seed) for seed in (11, 22, 33)]
    summary = analysis.build_summary(manifest(tmp_path), runs, "manifest")
    assert len(summary["summary"]) == 12
    assert len(summary["paired_comparisons"]) == 15
    assert summary["work"] == {
        "new_test_labels": 30,
        "test_collection_matches": 6,
        "unique_gameplay_matches": 24,
        "supervised_optimizer_updates": 18,
    }
    aggregate = summary["seed_aggregates"]["geometry"]["test"]["mean_regret"]
    assert aggregate == {"valid_seeds": 3, "mean": 2, "min": 2, "max": 2}
    geometry = summary["summary"][2]
    assert geometry["gameplay"]["overall"]["wins"] == 2
    assert geometry["gameplay"]["side"]["1"]["goals_for"] == 3
    assert geometry["gameplay"]["bonuses"]["False"]["timeouts"] == 0


def test_verification_requires_all_six_stage_completions(monkeypatch, tmp_path):
    monkeypatch.setattr(analysis.runner, "verify", lambda _: [])
    monkeypatch.setattr(analysis.runner, "read_manifest", lambda _: manifest(tmp_path))
    with pytest.raises(ValueError, match="All three geometry"):
        analysis.verified_inputs(tmp_path)


def test_load_runs_rejects_artifacts_outside_verified_closure(tmp_path):
    with pytest.raises(ValueError, match="outside verified closure"):
        analysis.load_runs(tmp_path, {})


def test_export_reproducibility_and_input_output_hashes(monkeypatch, tmp_path):
    root = tmp_path / "model"
    root.mkdir()
    meta = manifest(tmp_path)
    analysis.runner.write_json(root / "manifest.json", meta)
    protected = {
        str(root / "manifest.json"): analysis.runner.sha(root / "manifest.json"),
        str(Path(analysis.__file__).resolve()): analysis.runner.sha(analysis.__file__),
        str(Path(__file__).resolve()): analysis.runner.sha(__file__),
    }
    calls = []

    def verified(path):
        calls.append(path)
        analysis.runner.prior.check_hashes(protected)
        return copy.deepcopy(meta), protected.copy()

    monkeypatch.setattr(analysis, "verified_inputs", verified)
    monkeypatch.setattr(analysis, "load_runs", lambda *_: [make_run(s) for s in (11, 22, 33)])
    prefix = tmp_path / "exports/geometry_cost"
    analysis.export(root, prefix)
    assert len(calls) == 2
    integrity = json.loads(prefix.with_name("geometry_cost_integrity.json").read_text())
    assert integrity["exporter_pinned_posthoc"] is True
    analysis.runner.prior.check_hashes(integrity["export_sha256"])
    analysis.export(root, prefix)
    analysis.runner.prior.check_hashes(integrity["export_sha256"])
    assert list(root.iterdir()) == [root / "manifest.json"]
    with pytest.raises(ValueError, match="outside model experiment"):
        analysis.export(root, root / "bad_export")


def test_changed_input_during_export_prevents_integrity_seal(monkeypatch, tmp_path):
    root = tmp_path / "model"
    root.mkdir()
    meta = manifest(tmp_path)
    analysis.runner.write_json(root / "manifest.json", meta)
    protected = {str(root / "manifest.json"): analysis.runner.sha(root / "manifest.json")}
    monkeypatch.setattr(analysis, "verified_inputs", lambda _: (meta, protected))
    monkeypatch.setattr(analysis, "load_runs", lambda *_: [make_run(s) for s in (11, 22, 33)])

    def mutate(_summary, path):
        path.write_bytes(b"plot")
        (root / "manifest.json").write_text("changed")

    monkeypatch.setattr(analysis, "draw_plot", mutate)
    prefix = tmp_path / "exports/geometry_cost"
    with pytest.raises(ValueError, match="Pinned input changed"):
        analysis.export(root, prefix)
    assert not prefix.with_name("geometry_cost_integrity.json").exists()
