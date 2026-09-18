"""Verify scheduled hard copies against stock training and exact guarded resume."""

import io
from copy import deepcopy
from dataclasses import asdict
from dataclasses import replace

import pytest
import torch

from docs.ai import run_slow_target_experiment as slow
from tests.test_target_refresh_experiment import fork as _fork_fixture

fork = _fork_fixture


@pytest.fixture(autouse=True)
def cpu(monkeypatch):
    monkeypatch.setattr(slow.prior.dqn_ai, "device", torch.device("cpu"))


def test_slow_matches_stock_longer_frequency_and_exact_resume(fork, tmp_path):
    config, path = fork
    state = slow.prior.load(path)
    identity = slow.identity_for(state, "test", path, period=4)
    original = deepcopy(state)
    slow.execute(config, tmp_path / "full", state, identity)
    slow.execute(config, tmp_path / "split", state, identity, stop_after_episodes=14)
    paused = slow.prior.load(tmp_path / "split/latest.training.pt")
    with pytest.raises(ValueError, match="schema"):
        slow.prior.training.run_frame_pipeline(
            config, tmp_path / "unsafe", resume=tmp_path / "split/latest.training.pt"
        )
    slow.execute(config, tmp_path / "split", paused, identity, resume=True)
    a = slow.prior.load(tmp_path / "full/latest.training.pt")
    b = slow.prior.load(tmp_path / "split/latest.training.pt")
    for key in ("replay", "rng", "anchor", "progress", "episodes", "slow_target"):
        slow.prior.equal(a[key], b[key])
    final = slow.prior.load(io.BytesIO(a["model"]))
    slow.prior.equal(final, slow.prior.load(io.BytesIO(b["model"])))
    # Independent reference: the stock runner with a longer configured frequency.
    reference_config = replace(config, target_update_frequency=4)
    reference = deepcopy(state)
    reference["config"] = asdict(reference_config)
    reference_path = tmp_path / "reference.training.pt"
    torch.save(reference, reference_path)
    slow.prior.training.run_frame_pipeline(
        reference_config, tmp_path / "stock", resume=reference_path
    )
    stock = slow.prior.load(tmp_path / "stock/latest.training.pt")
    model = slow.prior.load(io.BytesIO(stock["model"]))
    for key in (
        "q_network_state_dict",
        "target_network_state_dict",
        "optimizer_state_dict",
        "training_step",
        "step_count",
        "loss_history",
    ):
        slow.prior.equal(final[key], model[key])
    for key in ("replay", "rng", "anchor", "episodes", "progress"):
        slow.prior.equal(a[key], stock[key])
    events = a["slow_target"]["events"]
    assert events and any(e["applied"] for e in events) and any(not e["applied"] for e in events)
    assert all(e["applied"] == (e["step"] % 4 == 0) for e in events)
    last = [e for e in events if e["applied"]][-1]
    assert slow.prior.target_hash(final) == last["online_sha256"]
    slow.prior.equal(state, original)
    assert not list((tmp_path / "split").glob(".resume-*"))


def test_identity_and_event_corruption_fail_closed(fork, tmp_path):
    config, path = fork
    state = slow.prior.load(path)
    identity = slow.identity_for(state, "test", path, period=4)
    slow.execute(config, tmp_path / "run", state, identity)
    saved = slow.prior.load(tmp_path / "run/latest.training.pt")
    for key, value in (("copy_period", 8), ("fork_sha256", "bad"), ("variant", "normal")):
        with pytest.raises(ValueError, match="identity"):
            slow.unwrap_bundle(saved, {**identity, key: value})
    for schema in (slow.prior.GUARDED_SCHEMA, slow.prior.training.SCHEMA):
        with pytest.raises(ValueError, match="schema"):
            slow.unwrap_bundle({**saved, "schema": schema}, identity)
    for change in ("schedule", "hash", "missing"):
        broken = deepcopy(saved)
        events = broken["slow_target"]["events"]
        if change == "schedule":
            events[0]["applied"] = not events[0]["applied"]
        elif change == "hash":
            events[-1]["after_sha256"] = "bad"
        else:
            events.pop()
        with pytest.raises(ValueError, match="target"):
            slow.unwrap_bundle(broken, identity)
    (tmp_path / "run/intervention.json").unlink()
    with pytest.raises(ValueError, match="sidecar"):
        slow.execute(config, tmp_path / "run", saved, identity, resume=True)


def test_methods_restored_and_temp_removed_on_error(fork, tmp_path, monkeypatch):
    config, path = fork
    state = slow.prior.load(path)
    identity = slow.identity_for(state, "test", path, period=4)
    originals = (
        slow.prior.training.load_model_bytes,
        slow.prior.training._save_bundle,
        slow.prior.dqn_ai.DQNAgent.update_target_network,
    )

    def fail(*args, **kwargs):
        raise RuntimeError("forced failure")

    monkeypatch.setattr(slow.prior.training, "run_frame_pipeline", fail)
    with pytest.raises(RuntimeError, match="forced"):
        slow.execute(config, tmp_path / "run", state, identity)
    assert originals == (
        slow.prior.training.load_model_bytes,
        slow.prior.training._save_bundle,
        slow.prior.dqn_ai.DQNAgent.update_target_network,
    )
    assert not list((tmp_path / "run").glob(".resume-*"))


@pytest.mark.parametrize("artifact", ["export", "fork"])
def test_prior_hash_corruption_rejected_before_training(tmp_path, monkeypatch, artifact):
    export, integrity, amendment, fork_path = [
        tmp_path / name
        for name in ("export.json", "integrity.json", "amendment.json", "fork.training.pt")
    ]
    slow.prior.write(export, {"complete": True})
    amendment.write_text("pinned amendment")
    fork_path.write_bytes(b"pinned fork")
    slow.prior.write(
        integrity,
        {
            "complete": True,
            "audit_amendment_sha256": slow.prior.digest(amendment),
            "artifact_sha256": {str(fork_path): slow.prior.digest(fork_path)},
            "export_sha256": {str(export): slow.prior.digest(export)},
        },
    )
    monkeypatch.setattr(slow, "PRIOR_EXPORT", export)
    monkeypatch.setattr(slow, "PRIOR_INTEGRITY", integrity)
    monkeypatch.setattr(slow, "AMENDMENT", amendment)
    monkeypatch.setattr(slow, "verify_amendment", lambda: {})
    if artifact == "export":
        slow.prior.write(export, {"complete": True, "changed": True})
    else:
        fork_path.write_bytes(b"changed fork")
    with pytest.raises(ValueError, match="changed prior artifact"):
        slow.verified_prior()
