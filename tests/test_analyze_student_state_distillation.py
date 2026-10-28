"""Posthoc exporter guards for student-state distillation."""

import copy
import json
import sys
from pathlib import Path

import pytest
import torch

sys.path.insert(0, str(Path(__file__).resolve().parents[1] / "docs/ai"))
import analyze_student_state_distillation as export  # noqa: E402


def fixture():
    selection = {
        "_sha256": "sealed",
        "seed": 11,
        "arms": {arm: {"best_trained_epoch": 15} for arm in export.run.ARMS},
    }
    training = {"seed": 11, "incumbent_epoch": 15}
    index, results = {}, {}
    for arm in export.POLICIES:
        owner = "incumbent" if arm == "control" else arm
        epoch = 0 if arm == "initial" else 15
        index[arm] = {
            "arm": arm,
            "epoch": epoch,
            "alias_of": "incumbent" if arm == "control" else None,
            "artifact": owner + ".json",
            "tensor_sha256": owner,
        }
        results[arm] = {
            "tensor_sha256": owner,
            "selection": {"arm": owner, "epoch": epoch, "selection_sha256": "sealed"},
        }
    return index, selection, training, results


def test_valid_alias_preserves_distinct_policy_view():
    export.validate_index(*fixture())


@pytest.mark.parametrize(
    "problem",
    (
        "missing_student",
        "epoch",
        "arm",
        "alias_tensor",
        "alias_artifact",
        "alias_chain",
        "result_tensor",
        "selection_hash",
        "selection_owner",
    ),
)
def test_stale_metadata_and_invalid_aliases_fail(problem):
    index, selection, training, results = copy.deepcopy(fixture())
    if problem == "missing_student":
        del index["student_10"]
    elif problem == "epoch":
        index["student_10"]["epoch"] = 10
    elif problem == "arm":
        index["student_10"]["arm"] = "teacher_10"
    elif problem == "alias_tensor":
        index["control"]["tensor_sha256"] = "different"
    elif problem == "alias_artifact":
        index["control"]["artifact"] = "stale.json"
    elif problem == "alias_chain":
        index["incumbent"]["alias_of"] = "initial"
    elif problem == "result_tensor":
        results["student_10"]["tensor_sha256"] = "different"
    elif problem == "selection_hash":
        results["student_10"]["selection"]["selection_sha256"] = "old"
    else:
        results["control"]["selection"]["arm"] = "control"
    with pytest.raises(ValueError):
        export.validate_index(index, selection, training, results)


@pytest.mark.parametrize("fault", [None, "stale_student_path", "wrong_teacher_tensor"])
def test_checkpoint_sources_match_fresh_selections(tmp_path, monkeypatch, fault):
    index, selection, training, _ = fixture()
    monkeypatch.setattr(export.run, "incumbent_path", lambda seed, epoch: tmp_path / "incumbent.pt")
    for i, (arm, spec) in enumerate(index.items()):
        path = export.expected_checkpoint(tmp_path, arm, 0 if arm == "initial" else 15, 11)
        path.parent.mkdir(parents=True, exist_ok=True)
        weights = {"w": torch.tensor([float(i)])}
        torch.save(weights, path)
        spec["weights"] = str(path)
        spec["tensor_sha256"] = export.run.tensor_digest(weights)
    if fault == "stale_student_path":
        index["student_10"]["weights"] = str(tmp_path / "student_10_epoch_10.pt")
    elif fault == "wrong_teacher_tensor":
        index["teacher_10"]["tensor_sha256"] = "old"
    if fault:
        with pytest.raises(ValueError, match="checkpoint"):
            export.validate_checkpoint_sources(tmp_path, index, training, selection)
    else:
        export.validate_checkpoint_sources(tmp_path, index, training, selection)


def match(steps=5, won=True, timeout=False, goals_for=3, goals_against=0):
    return {
        "agent_won": won,
        "timeout": timeout,
        "goals_for": goals_for,
        "goals_against": goals_against,
        "steps": steps,
    }


def write_source(directory, source, stem, cell, steps, labels=2):
    npz = directory / f"{stem}.npz"
    npz.write_bytes(f"{source}-{stem}".encode())
    record = directory / f"{stem}.json"
    record.write_text(
        json.dumps(
            {
                "cell": cell,
                "source_policy": source,
                "match": match(steps=steps),
                "trace_sha256": export.run.sha(npz),
                "replay_exact": True,
                "verification_frames": steps,
                "verification_labels": labels,
                "canonical_observations_checked": steps,
            }
        )
    )
    return record


def collection_pair(teacher_record, student_record, cell):
    return {
        "id": 0,
        "cell": cell,
        "start": 0,
        "length": 3,
        "teacher": {
            "frames": 4,
            "source_match": str(teacher_record.resolve()),
            "source_match_sha256": export.run.sha(teacher_record),
            "indices": [0, 1, 3],
        },
        "student": {
            "frames": 5,
            "source_match": str(student_record.resolve()),
            "source_match_sha256": export.run.sha(student_record),
            "indices": [0, 2, 4],
        },
    }


def write_collection(tmp_path, seed=11, student_suffix="a"):
    cell = {"side": 1, "bonuses": False, "match_seed": 680100}
    collector_epoch = export.run.SETTINGS["collector_epochs"][str(seed)]
    collector = tmp_path / "source" / f"seed_{seed}" / f"continuation_10_epoch_{collector_epoch}.pt"
    collector.parent.mkdir(parents=True, exist_ok=True)
    weights = {"w": torch.tensor([float(seed)])}
    torch.save(weights, collector)
    teacher_dir = tmp_path / "teacher"
    seed_dir = tmp_path / f"seed_{seed}"
    teacher_dir.mkdir(parents=True, exist_ok=True)
    seed_dir.mkdir(parents=True, exist_ok=True)
    teacher_record = write_source(teacher_dir, "teacher", "teacher_cell", cell, 4, labels=4)
    student_record = write_source(seed_dir, "student", f"student_cell_{student_suffix}", cell, 5)
    collection = {
        "seed": seed,
        "collector_epoch": collector_epoch,
        "collector_weights": str(collector.resolve()),
        "collector_tensor_sha256": export.run.tensor_digest(weights),
        "pairs": [collection_pair(teacher_record, student_record, cell)],
        "rows_per_arm": 3,
        "all_outcomes_retained": True,
        "teacher_matches_shared_across_seeds": True,
    }
    path = seed_dir / "collection.json"
    path.write_text(json.dumps(collection))
    return seed_dir, path, cell


def test_collection_sources_are_validated_and_count_teacher_once(tmp_path, monkeypatch):
    monkeypatch.setattr(
        export.run,
        "collection_cells",
        lambda: [{"side": 1, "bonuses": False, "match_seed": 680100}],
    )
    monkeypatch.setitem(export.run.SETTINGS, "rows_per_pair_cap", 3)
    monkeypatch.setattr(export.run, "SOURCE", tmp_path / "source")
    first_dir, first_path, _ = write_collection(tmp_path, 11, "a")
    second_dir, second_path, _ = write_collection(tmp_path, 22, "b")
    first = export.validate_collection_sources(
        first_dir, {"seed": 11, "collection_source": str(first_path), "new_rows_per_arm": 3}
    )
    second = export.validate_collection_sources(
        second_dir, {"seed": 22, "collection_source": str(second_path), "new_rows_per_arm": 3}
    )
    work = export.collection_work({"11": first, "22": second})
    assert work["teacher_collection_matches"] == 1
    assert work["student_collection_matches"] == 2
    assert work["new_collection_matches"] == 3
    assert work["new_collection_frames"] == 14
    assert work["offline_planner_labels"] == 12
    assert work["teacher_acting_labels"] == 4
    assert work["teacher_replay_acting_labels"] == 4
    assert work["teacher_replay_observer_labels"] == 4
    assert work["student_replay_observer_labels"] == 4


@pytest.mark.parametrize(
    "fault,match_text",
    [
        ("rows", "row"),
        ("source_hash", "hash"),
        ("trace_hash", "trace"),
        ("indices", "indices"),
        ("cell", "cell"),
        ("frames", "frame"),
        ("collector_epoch", "Collector epoch"),
        ("collector_tensor", "Collector checkpoint tensor"),
    ],
)
def test_collection_sources_reject_stale_or_unpaired_inputs(
    tmp_path, monkeypatch, fault, match_text
):
    monkeypatch.setattr(
        export.run,
        "collection_cells",
        lambda: [{"side": 1, "bonuses": False, "match_seed": 680100}],
    )
    monkeypatch.setitem(export.run.SETTINGS, "rows_per_pair_cap", 3)
    monkeypatch.setattr(export.run, "SOURCE", tmp_path / "source")
    seed_dir, collection_path, _ = write_collection(tmp_path, 11, "bad")
    collection = json.loads(collection_path.read_text())
    if fault == "rows":
        collection["rows_per_arm"] = 4
    elif fault == "source_hash":
        collection["pairs"][0]["student"]["source_match_sha256"] = "stale"
    elif fault == "trace_hash":
        record = Path(collection["pairs"][0]["student"]["source_match"])
        data = json.loads(record.read_text())
        data["trace_sha256"] = "stale"
        record.write_text(json.dumps(data))
        collection["pairs"][0]["student"]["source_match_sha256"] = export.run.sha(record)
    elif fault == "indices":
        collection["pairs"][0]["student"]["indices"] = [0, 1, 4]
    elif fault == "frames":
        collection["pairs"][0]["student"]["frames"] = 99
    else:
        if fault == "collector_epoch":
            collection["collector_epoch"] = 999
            collection_path.write_text(json.dumps(collection))
            with pytest.raises(ValueError, match=match_text):
                export.validate_collection_sources(
                    seed_dir,
                    {
                        "seed": 11,
                        "collection_source": str(collection_path),
                        "new_rows_per_arm": 3,
                    },
                )
            return
        if fault == "collector_tensor":
            collection["collector_tensor_sha256"] = "stale"
            collection_path.write_text(json.dumps(collection))
            with pytest.raises(ValueError, match=match_text):
                export.validate_collection_sources(
                    seed_dir,
                    {
                        "seed": 11,
                        "collection_source": str(collection_path),
                        "new_rows_per_arm": 3,
                    },
                )
            return
        record = Path(collection["pairs"][0]["student"]["source_match"])
        data = json.loads(record.read_text())
        data["cell"] = {"side": 2, "bonuses": False, "match_seed": 680100}
        record.write_text(json.dumps(data))
        collection["pairs"][0]["student"]["source_match_sha256"] = export.run.sha(record)
    collection_path.write_text(json.dumps(collection))
    with pytest.raises(ValueError, match=match_text):
        export.validate_collection_sources(
            seed_dir, {"seed": 11, "collection_source": str(collection_path), "new_rows_per_arm": 3}
        )


def test_collection_work_deduplicates_per_source_teacher_records():
    one = {
        "matches": {
            "matches": 1,
            "wins": 1,
            "losses": 0,
            "timeouts": 0,
            "goals_for": 3,
            "goals_against": 0,
            "goal_margin": 3,
            "frames": 10,
        },
        "verification_frames": 10,
        "verification_labels": 2,
        "canonical_observations_checked": 10,
    }
    seed_record = {
        "rows_per_arm": 4,
        "teacher_sources": {"teacher_a.json": one, "teacher_b.json": one},
        "student": {
            "matches": {**one["matches"], "matches": 2, "frames": 20},
            "verification_frames": 20,
            "verification_labels": 4,
        },
    }
    work = export.collection_work({"11": seed_record, "22": seed_record, "33": seed_record})
    assert work["teacher_collection_matches"] == 2
    assert work["student_collection_matches"] == 6
    assert work["new_collection_matches"] == 8
    assert work["teacher_acting_labels"] == 20
    assert work["teacher_replay_acting_labels"] == 20
    assert work["teacher_replay_observer_labels"] == 4
    assert work["offline_planner_labels"] == 24


def test_training_sources_require_inherited_control_bytes(tmp_path, monkeypatch):
    source = tmp_path / "source"
    parent = source / "seed_11"
    dest = tmp_path / "dest"
    parent.mkdir(parents=True)
    dest.mkdir()
    monkeypatch.setattr(export.run, "SOURCE", source)
    monkeypatch.setattr(
        export.run.previous,
        "verify_closure",
        lambda output, path, required=(): {
            "artifact_sha256": {str((parent / "control_epoch_5.pt").resolve()): "x"}
        },
    )
    record = {"candidate_epochs": [5], "history": [{"epoch": 0}]}
    parent_record = {
        **record,
        "provenance": {"newly_trained": False, "source_arm": "older_control"},
    }
    (parent / "training.json").write_text(json.dumps({"arms": {"control": parent_record}}))
    torch.save({"w": torch.tensor([1.0])}, parent / "control_epoch_5.pt")
    torch.save({"w": torch.tensor([1.0])}, dest / "control_epoch_5.pt")
    training = {
        "seed": 11,
        "arms": {
            "control": {
                **record,
                "provenance": {
                    "newly_trained": False,
                    "source_arm": "control",
                    "source_training_sha256": export.run.sha(parent / "training.json"),
                    "source_weights_sha256": {"5": export.run.sha(parent / "control_epoch_5.pt")},
                },
            },
            "teacher_10": {"provenance": {"newly_trained": True}},
            "student_10": {"provenance": {"newly_trained": True}},
        },
    }
    export.validate_training_sources(dest, training)
    torch.save({"w": torch.tensor([2.0])}, dest / "control_epoch_5.pt")
    with pytest.raises(ValueError, match="byte"):
        export.validate_training_sources(dest, training)


def test_paired_goal_win_loss_deltas_and_duplicate_rejection():
    common = {"opponent": "striker", "player_side": 2, "bonuses_enabled": True, "seed": 700100}
    reference = {
        "matches": [
            {**common, "agent_won": False, "timeout": False, "goals_for": 1, "goals_against": 3}
        ]
    }
    challenger = {
        "matches": [
            {**common, "agent_won": True, "timeout": False, "goals_for": 3, "goals_against": 0}
        ]
    }
    assert export.paired(reference, challenger)["total_delta"] == {
        "win_delta": 1,
        "loss_delta": -1,
        "goal_margin_delta": 5,
    }
    with pytest.raises(ValueError, match="paired"):
        export.paired(reference, {"matches": challenger["matches"] * 2})


def test_initial_validation_source_checks_original_tensor(tmp_path):
    weights = {"w": torch.tensor([1.0])}
    torch.save(weights, tmp_path / "original.pt")
    result = {
        "candidate": {"arm": "initial", "epoch": 0},
        "tensor_sha256": export.run.tensor_digest(weights),
    }
    export.validate_original_validation_source(tmp_path, result)
    result["tensor_sha256"] = "stale"
    with pytest.raises(ValueError, match="Initial validation tensor"):
        export.validate_original_validation_source(tmp_path, result)


def test_source_training_curves_exclude_inherited_control():
    summary = {
        "training": {
            "11": {
                "arms": {
                    "control": {"continuation_train_history": [{"all": {"mean_regret": 99.0}}]},
                    "teacher_10": {"continuation_train_history": [{"all": {"mean_regret": 5.0}}]},
                    "student_10": {"continuation_train_history": [{"all": {"mean_regret": 7.0}}]},
                }
            }
        }
    }
    assert export.source_training_curves(summary, 11) == {
        "teacher_10": [5.0],
        "student_10": [7.0],
    }
