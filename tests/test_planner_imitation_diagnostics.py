"""The diagnostic preserves its inputs and measures match-disjoint imitation."""

from __future__ import annotations

import json
import os
import random
import shutil
import subprocess
import sys
from pathlib import Path

import numpy as np
import pytest
import torch

from docs.ai import run_planner_imitation_diagnostics as diagnostic


def test_match_level_splits_are_disjoint_and_cover_every_condition():
    specs = diagnostic.match_specs()
    assert len(specs) == 96
    assert len({row["seed"] for row in specs}) == 96
    sets = {
        split: {row["seed"] for row in specs if row["split"] == split}
        for split in ("train", "dev", "test")
    }
    assert [len(sets[split]) for split in ("train", "dev", "test")] == [64, 16, 16]
    assert not (
        sets["train"] & sets["dev"] or sets["train"] & sets["test"] or sets["dev"] & sets["test"]
    )
    assert all(300000 <= seed < 304000 for seeds in sets.values() for seed in seeds)
    for opponent in diagnostic.OPPONENTS:
        for side in (1, 2):
            for bonuses in (False, True):
                rows = [
                    r
                    for r in specs
                    if (r["opponent"], r["side"], r["bonuses"]) == (opponent, side, bonuses)
                ]
                assert [r["split"] for r in rows] == ["train"] * 4 + ["dev", "test"]


def test_regret_ties_and_decisive_metrics_have_analytic_values():
    values = np.array([[3, 1, 0], [2, 2, 0], [0, 0.01, 0]], dtype=np.float32)
    teacher = np.array([0, 0, 1])
    logits = np.array([[0, 1, 0], [0, 1, 0], [1, 0, 0]], dtype=np.float32)
    result = diagnostic.metrics(values, teacher, logits)
    assert result["all"]["mean_regret"] == pytest.approx(2.01 / 3)
    assert result["all"]["max_regret"] == 2
    assert result["all"]["near_tie_fraction"] == pytest.approx(1 / 3)
    assert result["all"]["agreement"] == 0
    assert result["decisive"]["count"] == 1
    assert result["decisive"]["mean_regret"] == 2
    empty = diagnostic.metrics(values[1:2], teacher[1:2], logits[1:2])["decisive"]
    assert empty["count"] == 0 and empty["mean_regret"] is None
    json.dumps(result, allow_nan=False)


def test_soft_targets_match_existing_distillation_objective():
    from magic_pong.ai.distillation import PlannerDistiller

    values = torch.tensor(
        [
            [2.0, 2.0, 0.0, 0.0, 0.0, 0.0, 0.0, 0.0, 0.0],
            [0.0, 0.01, 0.03, 0.2, 0.1, 0.0, 0.0, 0.0, 0.0],
        ]
    )
    expected = PlannerDistiller(backend="python")._targets(values)
    assert torch.equal(expected, diagnostic.soft_targets(values))


def test_reservoir_is_deterministic_and_preserves_all_rows_before_cap():
    first, second = [], []
    rng1, rng2 = random.Random(17), random.Random(17)
    for item in range(100):
        diagnostic.reservoir_add(first, item, item + 1, 7, rng1)
        diagnostic.reservoir_add(second, item, item + 1, 7, rng2)
        if item < 7:
            assert first == list(range(item + 1))
    assert first == second and len(set(first)) == 7
    assert max(first) > 20


@pytest.mark.parametrize("side", [1, 2])
def test_collected_snapshot_labels_match_live_planner_and_inputs_do_not_change(side):
    from magic_pong.ai.agent_adapter import to_player1_dqn_observation
    from magic_pong.ai.fast_teacher import FastPhysicsTeacher
    from magic_pong.ai.interface import ObservationProcessor
    from magic_pong.ai.models.dqn_ai import DQNAgent
    from magic_pong.ai.physics_teacher import PhysicsTeacher
    from magic_pong.core import fast_physics
    from magic_pong.core.physics import PhysicsEngine
    from magic_pong.utils.config import game_config
    from magic_pong.utils.config import game_config_tmp

    with game_config_tmp(BONUSES_ENABLED=False, MAX_SCORE=3):
        physics = PhysicsEngine(game_config.FIELD_WIDTH, game_config.FIELD_HEIGHT)
        paddle = physics.player1 if side == 1 else physics.player2
        sign = 1 if side == 1 else -1
        face = paddle.position.x + (paddle.width if side == 1 else 0)
        physics.ball.position.x = face + sign * 20
        physics.ball.position.y = paddle.position.y + paddle.height / 2
        physics.ball.velocity.x = -sign * game_config.BALL_SPEED * 0.8
        physics.ball.velocity.y = game_config.BALL_SPEED * 0.2
        agent = DQNAgent(hidden_layers=[256, 128])
        agent.q_network.cpu().eval()
        observation = ObservationProcessor(
            physics.field_width, physics.field_height
        ).process_game_state(physics.get_game_state(), side)
        collector = diagnostic.make_collector(agent, 19)
        collector.bind_physics(physics, side)
        collector.get_action(observation)
        frame, state, action, snapshot = collector.rows[0]
        expected_state = agent._observation_to_state(to_player1_dqn_observation(observation, side))
        np.testing.assert_array_equal(state, expected_state)
        assert frame == 0
        with torch.no_grad():
            assert action == int(agent.q_network(torch.as_tensor(expected_state)).argmax())
        teacher = FastPhysicsTeacher(horizon_steps=12, action_steps=4)
        before = [array.copy() for array in snapshot]
        saved = diagnostic.label_snapshot(teacher, snapshot, side)
        live = teacher.label(physics, side)
        python = PhysicsTeacher(horizon_steps=12, action_steps=4).label(physics, side)
        assert saved.action == live.action == python.action
        np.testing.assert_array_equal(saved.values, live.values)
        np.testing.assert_allclose(saved.values, python.values, atol=1e-5, rtol=1e-5)
        for old, new, current in zip(before, snapshot, fast_physics.snapshot(physics)):
            np.testing.assert_array_equal(old, new)
            np.testing.assert_array_equal(old, current)


def test_fit_can_learn_toy_problem_without_using_test_rows(monkeypatch):
    torch.set_num_threads(1)
    torch.manual_seed(44)
    states = np.tile(np.array([[-1.0, -1.0], [1.0, 1.0]], dtype=np.float32), (16, 1))
    values = np.tile(np.array([[1.0, 0.0], [0.0, 1.0]], dtype=np.float32), (16, 1))
    data = {"states": states, "values": values, "teacher": values.argmax(axis=1)}
    train = np.arange(len(states)) < 16
    dev = (np.arange(len(states)) >= 16) & (np.arange(len(states)) < 24)
    monkeypatch.setitem(diagnostic.SETTINGS, "learning_rate", 0.03)
    network = torch.nn.Sequential(torch.nn.Linear(2, 8), torch.nn.ReLU(), torch.nn.Linear(8, 2))
    control = torch.nn.Sequential(torch.nn.Linear(2, 8), torch.nn.ReLU(), torch.nn.Linear(8, 2))
    control.load_state_dict(network.state_dict())
    trained, result = diagnostic.fit(network, data, train, dev, epochs=20, seed=44)
    poisoned = {name: array.copy() for name, array in data.items()}
    poisoned["states"][24:] = 1e6
    poisoned["values"][24:] = poisoned["values"][24:, ::-1]
    selected_control, result_control = diagnostic.fit(
        control, poisoned, train, dev, epochs=20, seed=44
    )
    assert result == result_control
    for name, value in trained.state_dict().items():
        assert torch.equal(value, selected_control.state_dict()[name])
    assert diagnostic.evaluate(trained, data, train)["all"]["agreement"] == 1
    assert diagnostic.evaluate(trained, data, dev)["all"]["mean_regret"] == 0


def test_archive_guard_rejects_preloaded_live_application(tmp_path):
    import magic_pong

    assert magic_pong.__file__
    with pytest.raises(ValueError, match="outside pinned archive"):
        diagnostic.assert_archive_imports(tmp_path)


def test_complete_archived_smoke_preserves_models_sources_and_rejects_corruption(tmp_path):
    from magic_pong.ai.models import dqn_ai

    baseline = tmp_path / "baseline"
    archive = baseline / "source_snapshot"
    source = Path(dqn_ai.__file__).resolve().parents[2]
    shutil.copytree(source, archive / "magic_pong", ignore=shutil.ignore_patterns("__pycache__"))
    sources = {
        str(path.relative_to(archive)): diagnostic.sha(path)
        for path in (archive / "magic_pong").rglob("*.py")
    }
    diagnostic.write_json(baseline / "manifest.json", {"source_hashes": sources})
    for seed in diagnostic.SEEDS:
        directory = baseline / f"seed_{seed}"
        directory.mkdir()
        model = directory / "best.pth"
        dqn_ai.DQNAgent(hidden_layers=[256, 128]).save_model(model)
        diagnostic.write_json(
            directory / "completion.json",
            {"status": "complete", "artifact_sha256": {str(model): diagnostic.sha(model)}},
        )
    originals = {str(path): diagnostic.sha(path) for path in baseline.rglob("*") if path.is_file()}
    output = tmp_path / "diagnostic"
    script = """
import sys
from pathlib import Path
from docs.ai import run_planner_imitation_diagnostics as d
d.SETTINGS.update(match_frames=12, rows_per_match=3, teacher_horizon=8,
                  teacher_action_steps=4, epochs=1, tiny_epochs=1, tiny_rows=3)
specs = d.match_specs()
d.match_specs = lambda: [specs[i] for i in (0,4,5)]
baseline, output = map(Path, sys.argv[1:])
d.prepare(baseline, output)
d.run(11, output)
"""
    env = {
        **os.environ,
        "OMP_NUM_THREADS": "1",
        "MKL_NUM_THREADS": "1",
        "OPENBLAS_NUM_THREADS": "1",
        "PYTHONDONTWRITEBYTECODE": "1",
    }
    completed = subprocess.run(
        [sys.executable, "-c", script, str(baseline), str(output)],
        env=env,
        capture_output=True,
        text=True,
        timeout=120,
    )
    assert completed.returncode == 0, completed.stdout + completed.stderr
    diagnostic.check_hashes(originals)
    result = json.loads((output / "seed_11/result.json").read_text())
    assert set(result["arms"]) == {"from_rl", "fresh"}
    assert result["initial"]["train"]["overall"]["all"]["count"] == 3
    assert result["initial"]["test"]["overall"]["all"]["count"] == 3
    assert diagnostic.verify_completed(output) == [11]
    collection_path = output / "seed_11/collection.json"
    original_record = collection_path.read_bytes()
    collection_path.write_bytes(original_record + b" ")
    with pytest.raises(ValueError, match="Pinned input changed"):
        diagnostic.verify_collection(
            output / "seed_11", diagnostic.sha(output / "manifest.json"), 11
        )
    with pytest.raises(ValueError, match="Pinned input changed"):
        diagnostic.verify_completed(output)
    collection_path.write_bytes(original_record)
    assert diagnostic.verify_completed(output) == [11]
    manifest = json.loads((output / "manifest.json").read_text())
    model = Path(manifest["models"]["11"])
    model.write_bytes(model.read_bytes() + b"corrupt")
    with pytest.raises(ValueError, match="Pinned input changed"):
        diagnostic.check_hashes(manifest["input_sha256"])


def test_invalid_collection_split_and_duplicate_frames_fail():
    records = [
        {"index": 0, "seed": 1, "split": "train", "retained": 2},
        {"index": 1, "seed": 2, "split": "dev", "retained": 1},
        {"index": 2, "seed": 3, "split": "test", "retained": 1},
    ]
    data = {
        "states": np.zeros((4, 3), np.float32),
        "values": np.zeros((4, 9), np.float32),
        "match": np.array([0, 0, 1, 2]),
        "frame": np.array([0, 4, 0, 0]),
    }
    diagnostic.validate_data(data, records)
    masks = diagnostic.split_masks(data, records)
    assert [int(mask.sum()) for mask in masks.values()] == [2, 1, 1]
    data["frame"][1] = 0
    with pytest.raises(ValueError, match="Duplicate"):
        diagnostic.validate_data(data, records)
    records[2]["split"] = "train"
    with pytest.raises(ValueError, match="Every match-level split"):
        diagnostic.split_masks(data, records)


def _synthetic_export_run(seed: int, regret: float, count: int) -> dict:
    import copy

    metric = {
        "count": count,
        "agreement": 1 - regret / 10,
        "mean_regret": regret,
        "max_regret": regret * 2,
        "soft_cross_entropy": regret / 2,
        "exact_tie_fraction": 0.1,
        "near_tie_fraction": 0.1,
    }
    pair = {"all": metric, "decisive": {**metric, "count": count // 2}}
    detailed = {
        split: {
            "overall": pair,
            "opponent": dict.fromkeys(diagnostic.OPPONENTS, pair),
            "category": dict.fromkeys(diagnostic.CATEGORIES, pair),
            "hidden_context": dict.fromkeys(diagnostic.HIDDEN_CONTEXT, pair),
            "side": {"1": pair, "2": pair},
            "bonuses": {"false": pair, "true": pair},
            "rollout_opportunity": dict.fromkeys(("can_score", "can_concede", "can_contact"), pair),
        }
        for split in ("train", "dev", "test")
    }
    history = [{"epoch": epoch, "train": pair, "dev": pair} for epoch in range(3)]
    result = {
        "seed": seed,
        "initial": detailed,
        "arms": {
            arm: {"selected_epoch": 1, "selected": detailed, "history": history}
            for arm in ("from_rl", "fresh")
        },
        "tiny_overfit": {"selected_epoch": 2, "history": history, "final": pair},
    }
    return copy.deepcopy(
        {
            "seed": seed,
            "result": result,
            "collection": {
                "matches": [{}, {}, {}],
                "frames": count * 12,
                "candidates": count * 4,
                "rows": count * 3,
            },
        }
    )


def test_export_aggregates_seed_means_not_row_weighted_and_retains_empty_groups():
    from docs.ai import analyze_planner_imitation_diagnostics as analysis

    runs = [
        _synthetic_export_run(11, 1, 10000),
        _synthetic_export_run(22, 2, 10),
        _synthetic_export_run(33, 3, 10),
    ]
    artifact = analysis.build_export({"settings": diagnostic.SETTINGS}, runs, "manifest")
    assert len(artifact["rows"]) == 3 * 436
    overall = next(
        row
        for row in artifact["seed_aggregates"]
        if (row["arm"], row["split"], row["dimension"], row["stratum"])
        == ("initial", "test", "overall", "all")
    )
    assert overall["mean_regret"] == {"mean": 2, "min": 1, "max": 3, "valid_seeds": 3}
    rows = [row.copy() for row in artifact["rows"] if row["arm"] == "tiny_final"]
    rows[0]["mean_regret"] = None
    rows[0]["count"] = 0
    aggregate = analysis.aggregate_rows(rows)
    assert aggregate[0]["mean_regret"]["valid_seeds"] == 2
    with pytest.raises(ValueError, match="Duplicate seed"):
        analysis.aggregate_rows([rows[0], rows[0]])


def test_export_requires_all_three_completed_seeds(monkeypatch, tmp_path):
    from docs.ai import analyze_planner_imitation_diagnostics as analysis

    monkeypatch.setattr(diagnostic, "read_manifest", lambda root: {})
    monkeypatch.setattr(diagnostic, "verify_completed", lambda root: [11, 22])
    with pytest.raises(ValueError, match="All three diagnostic seeds"):
        analysis.load_verified(tmp_path)


def test_export_is_reproducible_and_hashes_outputs_without_model_writes(monkeypatch, tmp_path):
    from docs.ai import analyze_planner_imitation_diagnostics as analysis

    root = tmp_path / "model_experiment"
    root.mkdir()
    manifest = {"settings": diagnostic.SETTINGS, "baseline_root": str(tmp_path / "baseline")}
    diagnostic.write_json(root / "manifest.json", manifest)
    protected = {
        str(root / "manifest.json"): diagnostic.sha(root / "manifest.json"),
        str(Path(analysis.__file__).resolve()): diagnostic.sha(analysis.__file__),
    }
    runs = [
        _synthetic_export_run(seed, number, 10) for number, seed in enumerate(diagnostic.SEEDS, 1)
    ]
    calls = []

    def load_verified(path):
        calls.append(path)
        diagnostic.check_hashes(protected)
        return manifest, runs, protected.copy()

    monkeypatch.setattr(analysis, "load_verified", load_verified)
    prefix = tmp_path / "exports" / "diagnostic"
    analysis.export(root, prefix)
    assert len(calls) == 2
    integrity = json.loads(prefix.with_name("diagnostic_integrity.json").read_text())
    diagnostic.check_hashes(integrity["export_sha256"])
    assert len(integrity["export_sha256"]) == 3
    assert integrity["complete_seeds"] == [11, 22, 33]
    original_exports = integrity["export_sha256"].copy()
    analysis.export(root, prefix)
    diagnostic.check_hashes(original_exports)
    assert sorted(path.name for path in root.iterdir()) == ["manifest.json"]
    with pytest.raises(ValueError, match="outside immutable model"):
        analysis.export(root, root / "bad_export")


def test_export_rechecks_protected_inputs_after_drawing(monkeypatch, tmp_path):
    from docs.ai import analyze_planner_imitation_diagnostics as analysis

    root = tmp_path / "root"
    root.mkdir()
    manifest = {"settings": diagnostic.SETTINGS, "baseline_root": str(tmp_path / "baseline")}
    diagnostic.write_json(root / "manifest.json", manifest)
    protected = {
        str(root / "manifest.json"): diagnostic.sha(root / "manifest.json"),
        str(Path(analysis.__file__).resolve()): diagnostic.sha(analysis.__file__),
    }
    runs = [_synthetic_export_run(seed, 1, 10) for seed in diagnostic.SEEDS]
    monkeypatch.setattr(analysis, "load_verified", lambda _: (manifest, runs, protected))

    def changed_input(_runs, path):
        path.write_bytes(b"plot")
        (root / "manifest.json").write_text("changed")

    monkeypatch.setattr(analysis, "draw_plot", changed_input)
    with pytest.raises(ValueError, match="Pinned input changed"):
        analysis.export(root, tmp_path / "output" / "diagnostic")
    assert not (tmp_path / "output" / "diagnostic_integrity.json").exists()


def test_hidden_context_counts_use_active_flags_and_both_paddle_timers():
    from magic_pong.core import fast_physics as fp

    data = {
        "physics": np.zeros((6, fp.N_STATE)),
        "bonuses": np.zeros((6, fp.MAX_BONUSES, fp.N_BONUS_COLS)),
        "rotating": np.zeros((6, fp.MAX_ROTATING, fp.N_ROTATING_COLS)),
    }
    # Expired/inactive objects can retain coordinates; they do not spoil clean rows.
    data["bonuses"][0, :, fp.BX] = 42
    data["rotating"][0, :, fp.RLIFE] = -1
    data["physics"][1, fp.P1 + fp.PTIMER] = 1
    data["physics"][2, fp.P2 + fp.PTIMER] = 2
    data["bonuses"][3, :3, fp.BACTIVE] = 1
    data["bonuses"][4, :4, fp.BACTIVE] = 1
    data["rotating"][4, :2, fp.RLIFE] = 1
    data["rotating"][5, :3, fp.RLIFE] = 1
    before = {key: value.copy() for key, value in data.items()}
    masks = diagnostic.hidden_context_masks(data)
    assert tuple(masks) == diagnostic.HIDDEN_CONTEXT
    assert {key: int(value.sum()) for key, value in masks.items()} == {
        "clean": 1,
        "size_timer_active": 2,
        "bonuses_0": 4,
        "bonuses_1_to_3": 1,
        "bonuses_above_3": 1,
        "rotating_0": 4,
        "rotating_1_to_2": 1,
        "rotating_above_2": 1,
    }
    assert masks["clean"].tolist() == [True, False, False, False, False, False]
    for key, value in before.items():
        np.testing.assert_array_equal(data[key], value)
