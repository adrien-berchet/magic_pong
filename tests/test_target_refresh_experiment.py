"""Exact fork continuation and experiment-only target freezing."""

import io
from copy import deepcopy

import pytest
import torch

from docs.ai.audit_score_context_experiment import equal
from docs.ai.audit_score_context_experiment import load
from docs.ai.run_target_refresh_experiment import GUARDED_SCHEMA
from docs.ai.run_target_refresh_experiment import canonical_state
from docs.ai.run_target_refresh_experiment import execute
from docs.ai.run_target_refresh_experiment import identity_for
from docs.ai.run_target_refresh_experiment import inspect_context
from docs.ai.run_target_refresh_experiment import inspect_masks
from docs.ai.run_target_refresh_experiment import read
from docs.ai.run_target_refresh_experiment import unwrap_bundle
from magic_pong.ai import scheduled_training as training
from magic_pong.ai.models import dqn_ai
from tests.test_policy_anchor import make_bank
from tests.test_scheduled_training import initial_model
from tests.test_scheduled_training import small_config


@pytest.fixture(autouse=True)
def cpu(monkeypatch):
    monkeypatch.setattr(dqn_ai, "device", torch.device("cpu"))


@pytest.fixture
def fork(tmp_path):
    bank, sha = make_bank(tmp_path)
    config = small_config(
        score_context=True,
        selfplay_opponent="follow_ball",
        target_update_frequency=2,
        anchor_weight=0.05,
        anchor_bank_sha256=sha,
    )
    training.run_frame_pipeline(
        config,
        tmp_path / "fork",
        initial_checkpoint=initial_model(tmp_path),
        anchor_bank=bank,
        stop_after_episodes=13,
    )
    return config, tmp_path / "fork/latest.training.pt"


def test_normal_wrapper_matches_stock_full_resume(fork, tmp_path):
    config, path = fork
    state = load(path)
    identity = identity_for(state, "normal", "test", path)
    training.run_frame_pipeline(config, tmp_path / "stock", resume=path)
    execute(config, tmp_path / "normal", state, identity)
    a, b = load(tmp_path / "stock/latest.training.pt"), load(tmp_path / "normal/latest.training.pt")
    for key in ("replay", "rng", "progress", "episodes", "anchor", "phase_results"):
        equal(a[key], b[key])
    equal(load(io.BytesIO(a["model"])), load(io.BytesIO(b["model"])))
    equal(canonical_state(a), canonical_state(unwrap_bundle(b, identity)))
    assert b["schema"] == GUARDED_SCHEMA
    assert b["target_refresh"]["refresh_steps"]
    summary = read(tmp_path / "normal/training.json")
    assert inspect_context(summary, tmp_path / "normal")["state_size"] == 33
    assert inspect_masks(summary, tmp_path / "normal", True)["final_replay"]["transitions"] > 0


def test_freeze_preserves_loaded_target_and_exact_guarded_resume(fork, tmp_path):
    config, path = fork
    state = load(path)
    identity = identity_for(state, "frozen", "test", path)
    original = deepcopy(state)
    before_update = dqn_ai.DQNAgent.update_target_network
    execute(config, tmp_path / "full", state, identity)
    execute(config, tmp_path / "split", state, identity, stop_after_episodes=14)
    paused = load(tmp_path / "split/latest.training.pt")
    with pytest.raises(ValueError, match="schema"):
        training.run_frame_pipeline(
            config, tmp_path / "unsafe", resume=tmp_path / "split/latest.training.pt"
        )
    execute(config, tmp_path / "split", paused, identity, resume=True)
    a, b = load(tmp_path / "full/latest.training.pt"), load(tmp_path / "split/latest.training.pt")
    for key in ("replay", "rng", "progress", "episodes", "anchor", "target_refresh"):
        equal(a[key], b[key])
    final = load(io.BytesIO(a["model"]))
    equal(final, load(io.BytesIO(b["model"])))
    equal(
        final["target_network_state_dict"],
        load(io.BytesIO(state["model"]))["target_network_state_dict"],
    )
    assert any(
        not torch.equal(v, final["q_network_state_dict"][k])
        for k, v in final["target_network_state_dict"].items()
    )
    equal(state, original)
    assert dqn_ai.DQNAgent.update_target_network is before_update
    assert not list((tmp_path / "split").glob(".resume-*"))
    (tmp_path / "split/intervention.json").unlink()
    with pytest.raises(ValueError, match="sidecar"):
        execute(config, tmp_path / "split", b, identity, resume=True)


def test_guard_rejects_wrong_variant_and_changed_target(fork, tmp_path):
    config, path = fork
    state = load(path)
    identity = identity_for(state, "frozen", "test", path)
    execute(config, tmp_path / "split", state, identity, stop_after_episodes=14)
    guarded = load(tmp_path / "split/latest.training.pt")
    with pytest.raises(ValueError, match="identity"):
        unwrap_bundle(guarded, {**identity, "variant": "normal"})
    broken = deepcopy(guarded)
    broken["target_refresh"]["target_sha256"] = "bad"
    with pytest.raises(ValueError, match="target"):
        unwrap_bundle(broken, identity)


def test_wrapper_restores_methods_and_cleans_temporary_on_failure(fork, tmp_path, monkeypatch):
    config, path = fork
    state = load(path)
    identity = identity_for(state, "frozen", "test", path)
    original_update = dqn_ai.DQNAgent.update_target_network
    original_save = training._save_bundle
    original_load = training.load_model_bytes

    def fail(*args, **kwargs):
        raise RuntimeError("forced failure")

    monkeypatch.setattr(training, "run_frame_pipeline", fail)
    with pytest.raises(RuntimeError, match="forced failure"):
        execute(config, tmp_path / "failed", state, identity)
    assert dqn_ai.DQNAgent.update_target_network is original_update
    assert training._save_bundle is original_save
    assert training.load_model_bytes is original_load
    assert not list((tmp_path / "failed").glob(".resume-*"))
