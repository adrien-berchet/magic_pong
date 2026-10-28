"""Guardrails for the student-state distillation experiment."""

import copy
import inspect
import sys
from pathlib import Path

import numpy as np
import pytest

sys.path.insert(0, str(Path(__file__).resolve().parents[1] / "docs/ai"))
import run_student_state_distillation as run  # noqa: E402

from tests import test_continuation_distillation as prior_tests  # noqa: E402


def _block(source, source_hash, frames=(0, 1, 2, 3), pair_id=0, match_seed=680100):
    n = len(frames)
    return {
        "states": np.zeros((n, 30), dtype=np.float32),
        "physics": np.zeros((n, 14), dtype=np.float32),
        "bonuses": np.zeros((n, 0), dtype=np.float32),
        "rotating": np.zeros((n, 0), dtype=np.float32),
        "params": np.zeros((n, 1), dtype=np.float32),
        "actions": np.full(n, 3, dtype=np.int8),
        "hashes": np.asarray([f"{source}-hash-{i}".encode() for i in range(n)]),
        "values": np.ones((n, 9), dtype=np.float32),
        "teacher": np.arange(n, dtype=np.int8) % 9,
        "pair_id": np.full(n, pair_id, dtype=np.int64),
        "frame": np.asarray(frames, dtype=np.int32),
        "match_seed": np.full(n, match_seed, dtype=np.int64),
        "source_policy": np.asarray([source] * n),
        "source_match_sha256": np.asarray([source_hash] * n),
    }


def _pair_records(frames=(0, 1, 2, 3), pair_id=0, match_seed=680100):
    return [
        {
            "id": pair_id,
            "cell": {"side": 1, "bonuses": False, "match_seed": match_seed},
            "start": 0,
            "length": len(frames),
            "teacher": {
                "frames": 4,
                "source_match": "teacher_side1_bonus0_seed680100.npz",
                "sampled_indices": list(frames),
                "indices": list(frames),
                "source_match_sha256": "teacher-hash",
            },
            "student": {
                "frames": 4,
                "source_match": "student_side1_bonus0_seed680100.npz",
                "sampled_indices": list(frames),
                "indices": list(frames),
                "source_match_sha256": "student-hash",
            },
        }
    ]


def _drop_last(data):
    return {key: value[:-1] for key, value in data.items()}


@pytest.mark.parametrize(
    "name",
    [
        "test_splits_reject_overlap_and_duplicate_seeds",
        "test_panels_reject_duplicate_missing_and_wrong_split",
        "test_tensor_alias_requires_exact_values_not_epoch",
        "test_heldout_barrier_requires_all_three_sealed_selections",
    ],
)
def test_inherited_guards(name, tmp_path, monkeypatch):
    monkeypatch.setattr(prior_tests, "run", run)
    function = getattr(prior_tests, name)
    args = {"tmp_path": tmp_path, "monkeypatch": monkeypatch}
    function(**{k: args[k] for k in inspect.signature(function).parameters})


def test_collection_cells_are_fresh_paired_and_disjoint():
    run.assert_splits(run.SETTINGS)
    assert tuple(run.ARMS) == ("control", "teacher_10", "student_10")
    assert run.SETTINGS["collection_seeds"] == [680100, 680101, 680102, 680103]
    assert run.SETTINGS["validation_seeds"] == [690100, 690101, 690102, 690103]
    assert run.SETTINGS["heldout_seeds"] == list(range(700100, 700108))

    cells = run.collection_cells()
    assert len(cells) == 16
    identities = {(c["match_seed"], c["side"], c["bonuses"]) for c in cells}
    assert len(identities) == 16
    assert {c["match_seed"] for c in cells} == set(run.SETTINGS["collection_seeds"])
    assert {c["side"] for c in cells} == {1, 2}
    assert {c["bonuses"] for c in cells} == {False, True}


@pytest.mark.parametrize("frames,k", [(1, 1), (2, 2), (10, 4), (1025, 1024)])
def test_sampled_indices_are_sorted_unique_and_endpoint_inclusive(frames, k):
    indices = np.asarray(run.sampled_indices(frames, k))
    assert indices.dtype.kind in "iu"
    assert len(indices) == k
    assert indices.tolist() == sorted(set(indices.tolist()))
    assert indices[0] == 0
    assert indices[-1] == frames - 1
    assert indices.min() >= 0
    assert indices.max() < frames
    if len(indices) > 2:
        gaps = np.diff(indices)
        assert gaps.min() >= 1
        assert gaps.max() - gaps.min() <= 1


@pytest.mark.parametrize("frames,k", [(0, 1), (5, 0), (4, 5), (5, 1)])
def test_sampled_indices_reject_impossible_requests(frames, k):
    with pytest.raises(ValueError):
        run.sampled_indices(frames, k)


def test_validate_pair_accepts_aligned_teacher_and_student_rows(monkeypatch):
    monkeypatch.setattr(
        run, "collection_cells", lambda: [{"side": 1, "bonuses": False, "match_seed": 680100}]
    )
    teacher = _block("teacher", "teacher-hash")
    student = _block("student", "student-hash")
    run.validate_pair(teacher, student, _pair_records())


@pytest.mark.parametrize(
    "mutate,match",
    [
        (lambda teacher, student: (teacher, _drop_last(student)), "count|row|frame|index"),
        (
            lambda teacher, student: (
                teacher,
                {**student, "pair_id": np.asarray([0, 0, 9, 0], dtype=np.int64)},
            ),
            "pair",
        ),
        (
            lambda teacher, student: (
                teacher,
                {**student, "frame": np.asarray([0, 1, 2, 4], dtype=np.int32)},
            ),
            "frame|index",
        ),
        (
            lambda teacher, student: (
                teacher,
                {**student, "frame": np.asarray([0, 1, 1, 3], dtype=np.int32)},
            ),
            "frame|index",
        ),
        (
            lambda teacher, student: (
                teacher,
                {**student, "source_policy": np.asarray(["teacher"] * 4)},
            ),
            "source|policy",
        ),
        (
            lambda teacher, student: (
                teacher,
                {**student, "source_match_sha256": np.asarray(["stale"] * 4)},
            ),
            "hash|source",
        ),
        (
            lambda teacher, student: (
                teacher,
                {**student, "values": np.asarray([[np.inf] * 9] * 4, dtype=np.float32)},
            ),
            "Nonfinite",
        ),
    ],
)
def test_validate_pair_rejects_unpaired_or_stale_rows(mutate, match, monkeypatch):
    monkeypatch.setattr(
        run, "collection_cells", lambda: [{"side": 1, "bonuses": False, "match_seed": 680100}]
    )
    teacher = _block("teacher", "teacher-hash")
    student = _block("student", "student-hash")
    bad_teacher, bad_student = mutate(teacher, student)
    with pytest.raises(ValueError, match=match):
        run.validate_pair(bad_teacher, bad_student, _pair_records())


def test_validate_pair_rejects_missing_pair_record(monkeypatch):
    monkeypatch.setattr(
        run, "collection_cells", lambda: [{"side": 1, "bonuses": False, "match_seed": 680100}]
    )
    with pytest.raises(ValueError, match="Missing paired match"):
        run.validate_pair(_block("teacher", "teacher-hash"), _block("student", "student-hash"), [])


def _training_arm(new_regret, replaced=27, draw="draw", counts=(27,), updates=2):
    return {
        "updates": updates,
        "total_examples": 273,
        "batch_sizes": [256, 17],
        "legacy_shuffle_sha256": "legacy",
        "initial_tensor_sha256": "initial",
        "optimizer": "Adam",
        "learning_rate": 0.001,
        "gradient_clip": 10.0,
        "replaced_examples": replaced,
        "sampled_trajectory_counts": list(counts),
        "continuation_draw_sha256": draw,
        "parent_draw_sha256": "parent",
        "parent_draw_count": 136,
        "dosage_fraction": replaced / 273,
        "legacy_rng_seed": 900011,
        "continuation_rng_seed": 910011,
        "history": [{"epoch": 0, "dev": {"all": {"mean_regret": 3.0}}}],
        "continuation_train_history": [{"epoch": 0, "all": {"mean_regret": new_regret}}],
    }


def test_training_pair_allows_distinct_new_dataset_metrics():
    control = {
        key: value
        for key, value in _training_arm(0.0, replaced=0, draw="unused", counts=()).items()
        if key
        not in (
            "sampled_trajectory_counts",
            "continuation_draw_sha256",
            "parent_draw_sha256",
            "parent_draw_count",
            "dosage_fraction",
            "legacy_rng_seed",
            "continuation_rng_seed",
        )
    }
    teacher = _training_arm(1.0)
    student = _training_arm(9.0)

    run.verify_training_pair(control, teacher, student)

    for field, value in (
        ("updates", 3),
        ("replaced_examples", 28),
        ("sampled_trajectory_counts", [26]),
        ("continuation_draw_sha256", "other"),
        ("parent_draw_sha256", "other"),
        ("parent_draw_count", 135),
        ("dosage_fraction", 0.5),
        ("legacy_rng_seed", 900012),
        ("continuation_rng_seed", 910012),
    ):
        bad = copy.deepcopy(student)
        bad[field] = value
        with pytest.raises(AssertionError):
            run.verify_training_pair(control, teacher, bad)
