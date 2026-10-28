"""Budget, split, alias and real archived trajectory checks."""

import copy
import os
import subprocess
import sys
from pathlib import Path

import pytest
import torch

sys.path.insert(0, str(Path(__file__).resolve().parents[1] / "docs/ai"))
import run_continuation_distillation as run  # noqa: E402


def test_splits_reject_overlap_and_duplicate_seeds():
    run.assert_splits(run.SETTINGS)
    settings = copy.deepcopy(run.SETTINGS)
    settings["heldout_seeds"][0] = settings["collection_seeds"][0]
    with pytest.raises(ValueError, match="Overlapping"):
        run.assert_splits(settings)
    settings = copy.deepcopy(run.SETTINGS)
    settings["validation_seeds"].append(settings["validation_seeds"][0])
    with pytest.raises(ValueError, match="Duplicate"):
        run.assert_splits(settings)


def panel(split):
    seeds = (
        run.SETTINGS["validation_seeds"] if split == "validation" else run.SETTINGS["heldout_seeds"]
    )
    return {
        "split": split,
        "matches": [
            {"opponent": o, "player_side": side, "bonuses_enabled": b, "seed": s}
            for o in run.SETTINGS["opponents"]
            for side in (1, 2)
            for b in (False, True)
            for s in seeds
        ],
    }


def test_panels_reject_duplicate_missing_and_wrong_split():
    valid = panel("validation")
    run.assert_panel(valid, "validation")
    for bad in (
        {**valid, "matches": valid["matches"][:-1]},
        {**valid, "matches": valid["matches"] + valid["matches"][:1]},
        {**valid, "split": "held_out"},
    ):
        with pytest.raises(ValueError, match="panel"):
            run.assert_panel(bad, "validation")


def test_tensor_alias_requires_exact_values_not_epoch(tmp_path):
    files = []
    for name, value in (
        ("initial", 1.0),
        ("incumbent", 2.0),
        ("control", 2.0),
        ("continuation", 3.0),
    ):
        p = tmp_path / f"{name}.pt"
        torch.save({"weight": torch.tensor([value])}, p)
        files.append((name, 15, p))
    specs = run.deduplicate_policies(files)
    assert [s["alias_of"] for s in specs] == [None, None, "incumbent", None]


def test_mixture_preserves_short_batch_count_and_trajectory_bounds():
    starts, lengths = torch.tensor([0, 2]), torch.tensor([2, 10])
    gen = torch.Generator().manual_seed(7)
    positions, rows, ids = run.replacement_indices(17, starts, lengths, gen)
    assert positions.tolist() == list(range(1, 17, 2))
    assert torch.all(rows >= starts[ids]) and torch.all(rows < starts[ids] + lengths[ids])
    # Trajectory-uniform, not frame-uniform: lengths differ fivefold.
    _, _, many = run.replacement_indices(20000, starts, lengths, gen)
    assert 0.47 < (many == 0).float().mean().item() < 0.53


def test_training_budget_guard_rejects_unequal_updates_and_sample_counts():
    common = {
        "updates": 2,
        "total_examples": 273,
        "batch_sizes": [256, 17],
        "legacy_shuffle_sha256": "x",
        "initial_tensor_sha256": "y",
        "optimizer": "Adam",
        "learning_rate": 0.001,
        "gradient_clip": 10.0,
    }
    control = {**common, "replaced_examples": 0}
    treatment = {**common, "replaced_examples": 136}
    run.verify_training_pair(control, treatment)
    with pytest.raises(AssertionError, match="budget"):
        run.verify_training_pair(control, {**treatment, "updates": 3})
    with pytest.raises(AssertionError, match="Mixture"):
        run.verify_training_pair(control, {**treatment, "replaced_examples": 137})


def test_heldout_barrier_requires_all_three_sealed_selections(tmp_path, monkeypatch):
    (tmp_path / "manifest.json").write_text("{}")
    monkeypatch.setattr(run, "read_manifest", lambda output: {})
    for seed in (11, 22):
        directory = tmp_path / f"seed_{seed}"
        directory.mkdir()
        p = directory / "selection.json"
        p.write_text("{}")
        run.seal(tmp_path, directory / "selection_integrity.json", [p])
    with pytest.raises(FileNotFoundError):
        run.require_all_selections(tmp_path)
    directory = tmp_path / "seed_33"
    directory.mkdir()
    p = directory / "selection.json"
    p.write_text("{}")
    run.seal(tmp_path, directory / "selection_integrity.json", [p])
    run.require_all_selections(tmp_path)
    p.write_text('{"changed": true}')
    # codespell:ignore-next-line
    with pytest.raises(ValueError, match="[Hh]ash|[Cc]hanged|[Ii]ntegrity"):
        run.require_all_selections(tmp_path)


def test_fake_pipeline_seals_selection_before_heldout_and_reuses_alias(tmp_path, monkeypatch):
    import json

    (tmp_path / "manifest.json").write_text("{}")
    monkeypatch.setattr(run, "read_manifest", lambda output: {"application_archive": str(tmp_path)})
    monkeypatch.setattr(run.combo.prior, "assert_archive_imports", lambda path: None)

    def result(split, wins, digest="inc"):
        d = panel(split)
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
        for arm, value in (("initial", 0.0), ("control", 1.0), ("continuation", 2.0)):
            path = (
                directory / "original.pt"
                if arm == "initial"
                else run.weights_path(directory, arm, 5)
            )
            torch.save({"weight": torch.tensor([value])}, path)
        training = {
            "incumbent_epoch": 5,
            "incumbent_tensor_sha256": "inc",
            "arms": {a: {"candidate_epochs": [5]} for a in run.ARMS},
        }
        run.write_json(directory / "training.json", training)
        run.seal(tmp_path, directory / "training_integrity.json", [directory / "training.json"])
        artifacts = []
        for arm, wins in (("control", 32), ("continuation", 33)):
            data = result("validation", wins)
            data["candidate"] = {"arm": arm, "epoch": 5}
            path = directory / f"validation_{arm}_epoch_5.json"
            run.write_json(path, data)
            artifacts.append(path)
        run.seal(tmp_path, directory / "validation_integrity.json", artifacts)
        run.select(tmp_path, seed)
        if seed != 33:
            with pytest.raises(FileNotFoundError):
                run.require_all_selections(tmp_path)
    run.require_all_selections(tmp_path)
    calls = []
    monkeypatch.setattr(
        run, "context", lambda output, seed: ({}, None, None, 5, tmp_path / f"seed_{seed}")
    )
    monkeypatch.setattr(
        run,
        "incumbent_path",
        lambda seed, epoch: run.weights_path(tmp_path / f"seed_{seed}", "control", epoch),
    )

    def evaluate(path, original, geometry, split):
        calls.append(path)
        return result(split, 32, run.tensor_digest(torch.load(path, weights_only=True)))

    monkeypatch.setattr(run, "evaluate", evaluate)
    for seed in run.SETTINGS["training_seeds"]:
        run.heldout(tmp_path, seed)
        index = json.loads((tmp_path / f"seed_{seed}/heldout_index.json").read_text())["policies"]
        assert index["control"]["alias_of"] == "incumbent"
        assert index["control"]["artifact"] == index["incumbent"]["artifact"]
    assert len(calls) == 9
    run.verify(tmp_path, complete=True)


def test_archived_continuation_encoding_label_and_branch_parity():
    script = """
import json,pickle,sys
from pathlib import Path
import numpy as np
sys.path.insert(0,str(Path("docs/ai").resolve()))
import run_continuation_distillation as r
d=r.diagnostic
manifest=r.combo.read_manifest(r.SOURCE)
r.combo.prior.activate_archive(manifest)
from magic_pong.ai.competitive import _standard_configuration
from magic_pong.ai.models.planner_ai import make_teacher
from magic_pong.utils.config import game_config
original=r.combo.prior.load_agent(Path(manifest["models"]["11"]))
geometry=r.combo.previous.ContactGeometryConfig(**manifest["geometry"])
selection=json.loads((r.SOURCE/"seed_11/selection.json").read_text())
agent,_=d.policy(original,geometry,selection,11,"geometry_cost")
stem="geometry_cost_side2_bonus1_seed620000"
source=d.ROOT/"seed_11"
anchor=pickle.loads((source/f"{stem}_anchors.pkl").read_bytes())[-1]
with np.load(source/f"{stem}.npz") as saved: trace={k:saved[k] for k in saved.files}
cell={"side":2,"bonuses":True,"match_seed":620000}
teacher=make_teacher(90,30,backend="fast")
with _standard_configuration(d.config(cell)):
 game_config.BONUSES_ENABLED=True
 before=d.rng_state()
 data,record=r.teacher_continuation(anchor,original,teacher)
 after=d.rng_state()
 assert before[0]==after[0] and np.array_equal(before[1][1],after[1][1])
 assert np.array_equal(before[2].numpy(),after[2].numpy())
 reference=d.branch(anchor,agent,teacher,"teacher",trace)
 for field in ("frames","goals_for","goals_against","own_hits","other_hits","final_state_hash"):
  assert record[field]==reference[field],field
 for i in (0,len(data["states"])//2,len(data["states"])-1):
  decision=teacher.label_state(*(data[k][i].copy() for k in ("physics","bonuses","rotating","params")),player_id=2)
  assert decision.action==data["teacher"][i]
  assert np.array_equal(decision.values,data["values"][i])
 assert data["states"].dtype==np.float32 and data["states"].shape[1]==30
 assert np.all(np.argmax(data["values"],axis=1)==data["teacher"])
 print(json.dumps({"integration_continuation_frames":record["frames"],"verification_replay_frames":reference["frames"],"teacher_parity":True}))
"""
    env = {
        **os.environ,
        "OMP_NUM_THREADS": "1",
        "MKL_NUM_THREADS": "1",
        "OPENBLAS_NUM_THREADS": "1",
        "PYTHONDONTWRITEBYTECODE": "1",
    }
    result = subprocess.run(
        [sys.executable, "-c", script],
        env=env,
        capture_output=True,
        text=True,
        timeout=180,
        check=False,
    )
    assert result.returncode == 0, result.stdout + result.stderr
    print(result.stdout)
