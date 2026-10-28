"""Controls for the fixed-data 0/10/50% dosage experiment."""

import copy
import hashlib
import inspect
import json
import sys
from pathlib import Path

import pytest
import torch

sys.path.insert(0, str(Path(__file__).resolve().parents[1] / "docs/ai"))
import run_continuation_mixture as run  # noqa: E402

from tests import test_continuation_distillation as prior_tests  # noqa: E402


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


def test_old_evaluation_seeds_are_rejected():
    settings = copy.deepcopy(run.SETTINGS)
    settings["validation_seeds"][0] = run.previous.SETTINGS["heldout_seeds"][0]
    with pytest.raises(ValueError, match="previous"):
        run.assert_splits(settings)


def test_nested_sampler_is_global_subset_with_identical_shared_samples():
    starts, lengths = torch.tensor([0, 2, 17]), torch.tensor([2, 15, 3])
    old_rng = torch.Generator().manual_seed(73)
    new_rng = torch.Generator().manual_seed(73)
    sampler = run.NestedSampler()
    offset = retained = 0
    digest = hashlib.sha256()
    for size in (256, 256, 17, 1, 255, 8, 256):
        pos, rows, ids = run.previous.replacement_indices(size, starts, lengths, old_rng)
        keep = (torch.arange(len(rows)) + offset) % 5 == 4
        actual = sampler.draw(size, starts, lengths, new_rng)
        for a, e in zip(actual, (pos[keep], rows[keep], ids[keep]), strict=True):
            assert torch.equal(a, e)
        assert all(int(p) % 2 == 1 for p in actual[0])
        retained += len(actual[0])
        offset += len(rows)
        digest.update(rows.numpy().tobytes())
    assert retained == offset // 5
    assert sampler.parent_digest.hexdigest() == digest.hexdigest()
    assert torch.equal(old_rng.get_state(), new_rng.get_state())


@pytest.mark.parametrize("seed", [11, 22, 33])
def test_nested_sampler_replays_the_actual_parent_stream(seed):
    directory = run.SOURCE / f"seed_{seed}"
    if not (directory / "training.json").exists():
        pytest.skip("Local immutable parent experiment is unavailable")
    old = json.loads((directory / "training.json").read_text())["arms"]["continuation"]
    ts = json.loads((directory / "collection.json").read_text())["trajectories"]
    starts = torch.tensor([t["start"] for t in ts])
    lengths = torch.tensor([t["length"] for t in ts])
    generator = torch.Generator().manual_seed(old["continuation_rng_seed"])
    sampler = run.NestedSampler()
    retained = 0
    for size in old["batch_sizes"]:
        _, rows, _ = sampler.draw(size, starts, lengths, generator)
        retained += len(rows)
    assert sampler.parent_digest.hexdigest() == old["continuation_draw_sha256"]
    assert sampler.seen == old["replaced_examples"]
    assert retained == old["replaced_examples"] // 5


@pytest.mark.parametrize("fault", [None, "tamper", "missing", "unsealed"])
def test_inherited_candidates_require_sealed_exact_weights(tmp_path, monkeypatch, fault):
    source, dest = tmp_path / "source", tmp_path / "dest"
    parent = source / "seed_11"
    parent.mkdir(parents=True)
    dest.mkdir()
    (source / "manifest.json").write_text("{}")
    monkeypatch.setattr(run, "SOURCE", source)
    monkeypatch.setattr(run.previous, "read_manifest", lambda output: {})
    record = {"arms": {a: {"candidate_epochs": [5, 10]} for a in ("control", "continuation")}}
    run.write_json(parent / "training.json", record)
    files = [parent / "training.json"]
    for arm in record["arms"]:
        for epoch in (5, 10):
            p = run.previous.weights_path(parent, arm, epoch)
            torch.save({"weight": torch.tensor([epoch])}, p)
            files.append(p)
    target = files[-1]
    run.previous.seal(
        source, parent / "training_integrity.json", files[:-1] if fault == "unsealed" else files
    )
    if fault == "tamper":
        target.write_bytes(b"changed")
    elif fault == "missing":
        target.unlink()
    if fault:
        with pytest.raises((ValueError, FileNotFoundError)):
            run.inherit_arms(11, dest)
    else:
        records = run.inherit_arms(11, dest)
        for arm, old_arm in (("control", "control"), ("continuation_50", "continuation")):
            assert records[arm]["provenance"]["newly_trained"] is False
            for epoch in (5, 10):
                assert run.sha(run.weights_path(dest, arm, epoch)) == run.sha(
                    run.previous.weights_path(parent, old_arm, epoch)
                )


def test_budget_guard_rejects_wrong_dosage_stream_and_updates():
    common = {
        "updates": 2,
        "total_examples": 273,
        "batch_sizes": [256, 17],
        "legacy_shuffle_sha256": "x",
        "initial_tensor_sha256": "y",
        "optimizer": "Adam",
        "learning_rate": 0.001,
        "gradient_clip": 10.0,
        "history": [{"epoch": 0}],
        "continuation_train_history": [{"epoch": 0}],
    }
    control = {**common, "replaced_examples": 0}
    fifty = {**common, "replaced_examples": 136, "continuation_draw_sha256": "parent"}
    ten = {
        **common,
        "replaced_examples": 27,
        "parent_draw_sha256": "parent",
        "sampled_trajectory_counts": [27],
    }
    run.verify_training_pair(control, ten, fifty)
    for field, value in (
        ("updates", 3),
        ("replaced_examples", 28),
        ("parent_draw_sha256", "other"),
        ("sampled_trajectory_counts", [26]),
    ):
        with pytest.raises(AssertionError):
            run.verify_training_pair(control, {**ten, field: value}, fifty)


def test_fresh_reselection_and_five_policy_barrier_aliases(tmp_path, monkeypatch):
    (tmp_path / "manifest.json").write_text("{}")
    monkeypatch.setattr(run, "read_manifest", lambda output: {"application_archive": str(tmp_path)})
    monkeypatch.setattr(run.combo.prior, "assert_archive_imports", lambda path: None)
    monkeypatch.setattr(prior_tests, "run", run)

    def result(split, wins, digest):
        d = prior_tests.panel(split)
        d["tensor_sha256"] = digest
        for i, row in enumerate(d["matches"]):
            row.update(
                agent_won=i < wins,
                timeout=i >= wins,
                goals_for=3 if i < wins else 1,
                goals_against=0,
                steps=100,
            )
        return d

    for seed in run.SETTINGS["training_seeds"]:
        directory = tmp_path / f"seed_{seed}"
        directory.mkdir()
        torch.save({"weight": torch.tensor([0.0])}, directory / "original.pt")
        artifacts = []
        for i, arm in enumerate(run.ARMS):
            for epoch in (5, 10):
                p = run.weights_path(directory, arm, epoch)
                torch.save({"weight": torch.tensor([float(1 + i * 2 + (epoch == 10))])}, p)
                artifacts.append(p)
        inc_hash = run.tensor_digest(
            torch.load(run.weights_path(directory, "control", 5), weights_only=True)
        )
        run.write_json(
            directory / "training.json",
            {
                "incumbent_epoch": 5,
                "incumbent_tensor_sha256": inc_hash,
                "arms": {a: {"candidate_epochs": [5, 10]} for a in run.ARMS},
            },
        )
        run.seal(
            tmp_path,
            directory / "training_integrity.json",
            [directory / "training.json", *artifacts],
        )
        panels = []
        for arm, wins in (
            ("control", [40, 36]),
            ("continuation_10", [39, 44]),
            ("continuation_50", [30, 43]),
        ):
            for epoch, win in zip((5, 10), wins, strict=True):
                digest = run.tensor_digest(
                    torch.load(run.weights_path(directory, arm, epoch), weights_only=True)
                )
                d = result("validation", win, digest)
                d["candidate"] = {"arm": arm, "epoch": epoch}
                p = directory / f"validation_{arm}_epoch_{epoch}.json"
                run.write_json(p, d)
                panels.append(p)
        run.seal(tmp_path, directory / "validation_integrity.json", panels)
        run.select(tmp_path, seed)
        selection = json.loads((directory / "selection.json").read_text())
        assert selection["arms"]["control"]["best_trained_epoch"] == 5
        assert selection["arms"]["continuation_50"]["best_trained_epoch"] == 10
        assert selection["arms"]["continuation_10"]["best_trained_epoch"] == 10
        if seed != 33:
            with pytest.raises(FileNotFoundError):
                run.require_all_selections(tmp_path)
    monkeypatch.setattr(
        run, "context", lambda output, seed: ({}, None, None, 5, tmp_path / f"seed_{seed}")
    )
    monkeypatch.setattr(
        run,
        "incumbent_path",
        lambda seed, epoch: run.weights_path(tmp_path / f"seed_{seed}", "control", epoch),
    )
    calls = []

    def evaluate(path, original, geometry, split):
        calls.append(path)
        return result(split, 32, run.tensor_digest(torch.load(path, weights_only=True)))

    monkeypatch.setattr(run, "evaluate", evaluate)
    for seed in run.SETTINGS["training_seeds"]:
        run.heldout(tmp_path, seed)
        d = json.loads((tmp_path / f"seed_{seed}/heldout_index.json").read_text())
        assert len(d["policies"]) == 5
        assert d["policies"]["control"]["alias_of"] == "incumbent"
        for spec in d["policies"].values():
            assert (
                json.loads(Path(spec["artifact"]).read_text())["selection"]["selection_sha256"]
                == d["selection_sha256"]
            )
    assert len(calls) == 12
    run.verify(tmp_path, complete=True)
