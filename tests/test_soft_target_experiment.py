"""Gradual targets: arithmetic, independent hard-copy limit, and active resume."""

import io
from copy import deepcopy

import pytest
import torch

from docs.ai import run_soft_target_experiment as soft
from tests.test_target_refresh_experiment import fork as _fork_fixture

fork = _fork_fixture


@pytest.fixture(autouse=True)
def cpu(monkeypatch):
    monkeypatch.setattr(soft.prior.dqn_ai, "device", torch.device("cpu"))


def test_blend_arithmetic_does_not_mutate_inputs():
    target = {"w": torch.tensor([0.0, 4.0, -8.0])}
    online = {"w": torch.tensor([8.0, -4.0, 0.0])}
    before = deepcopy((target, online))
    result = soft.blend_state(target, online, 1, 4)
    assert torch.equal(result["w"], torch.tensor([2.0, 2.0, -6.0]))
    soft.prior.equal((target, online), before)
    assert not result["w"].requires_grad
    with pytest.raises(ValueError, match="representation"):
        soft.blend_state({"w": torch.tensor([1])}, {"w": torch.tensor([2])}, 1, 4)


def test_tau_one_matches_stock_hard_copies(fork, tmp_path):
    config, path = fork
    state = soft.prior.load(path)
    identity = soft.identity_for(state, "test", path, numerator=1, denominator=1)
    soft.execute(config, tmp_path / "soft", state, identity)
    soft.prior.training.run_frame_pipeline(config, tmp_path / "stock", resume=path)
    a, b = [soft.prior.load(tmp_path / name / "latest.training.pt") for name in ("soft", "stock")]
    for key in ("replay", "rng", "anchor", "progress", "episodes", "phase_results"):
        soft.prior.equal(a[key], b[key])
    soft.prior.equal(
        soft.prior.load(io.BytesIO(a["model"])), soft.prior.load(io.BytesIO(b["model"]))
    )


def test_soft_resume_after_active_blends_is_exact(fork, tmp_path):
    config, path = fork
    state = soft.prior.load(path)
    before = deepcopy(state)
    identity = soft.identity_for(state, "test", path)
    soft.execute(config, tmp_path / "full", state, identity)
    soft.execute(config, tmp_path / "split", state, identity, stop_after_episodes=14)
    paused = soft.prior.load(tmp_path / "split/latest.training.pt")
    assert paused["soft_target"]["events"]
    with pytest.raises(ValueError, match="schema"):
        soft.prior.training.run_frame_pipeline(
            config, tmp_path / "unsafe", resume=tmp_path / "split/latest.training.pt"
        )
    soft.execute(config, tmp_path / "split", paused, identity, resume=True)
    a, b = [soft.prior.load(tmp_path / name / "latest.training.pt") for name in ("full", "split")]
    for key in ("replay", "rng", "anchor", "progress", "episodes", "soft_target", "soft_evidence"):
        soft.prior.equal(a[key], b[key])
    final = soft.prior.load(io.BytesIO(a["model"]))
    soft.prior.equal(final, soft.prior.load(io.BytesIO(b["model"])))
    # Independent sequential formula from the retained online snapshots.
    expected = deepcopy(a["soft_evidence"]["initial_target"])
    for current in a["soft_evidence"]["online_states"]:
        expected = {k: v * (9 / 11) + current[k] * (2 / 11) for k, v in expected.items()}
    for k, v in expected.items():
        torch.testing.assert_close(final["target_network_state_dict"][k], v, rtol=1e-6, atol=1e-7)
    assert soft.prior.target_hash(final) != identity["fork_target_sha256"]
    assert soft.prior.target_hash(final) != soft.prior.fingerprint(final["q_network_state_dict"])
    soft.prior.equal(state, before)
    assert not list((tmp_path / "split").glob(".resume-*"))


def test_corrupt_identity_interpolation_and_sidecar_rejected(fork, tmp_path):
    config, path = fork
    state = soft.prior.load(path)
    identity = soft.identity_for(state, "test", path)
    soft.execute(config, tmp_path / "run", state, identity)
    saved = soft.prior.load(tmp_path / "run/latest.training.pt")
    for key, value in (("tau_numerator", 3), ("fork_sha256", "bad"), ("variant", "slow")):
        with pytest.raises(ValueError, match="identity"):
            soft.unwrap_bundle(saved, {**identity, key: value})
    with pytest.raises(ValueError, match="schema"):
        soft.unwrap_bundle({**saved, "schema": soft.slow.SCHEMA}, identity)
    for change in ("evidence", "step", "hash", "residual"):
        broken = deepcopy(saved)
        if change == "evidence":
            tensor = next(iter(broken["soft_evidence"]["online_states"][0].values()))
            tensor.flatten()[0] += 1
        elif change == "step":
            broken["soft_target"]["events"][0]["step"] += 1
        elif change == "hash":
            broken["soft_target"]["events"][0]["after_sha256"] = "bad"
        else:
            broken["soft_target"]["events"][0]["max_abs_residual"] = 1.0
        with pytest.raises(ValueError, match="target"):
            soft.unwrap_bundle(broken, identity)
    (tmp_path / "run/intervention.json").unlink()
    with pytest.raises(ValueError, match="sidecar"):
        soft.execute(config, tmp_path / "run", saved, identity, resume=True)


def test_failure_restores_methods_and_removes_temporary(fork, tmp_path, monkeypatch):
    config, path = fork
    state = soft.prior.load(path)
    identity = soft.identity_for(state, "test", path)
    originals = (
        soft.prior.training.load_model_bytes,
        soft.prior.training._save_bundle,
        soft.prior.dqn_ai.DQNAgent.update_target_network,
    )

    def fail(*args, **kwargs):
        raise RuntimeError("forced")

    monkeypatch.setattr(soft.prior.training, "run_frame_pipeline", fail)
    with pytest.raises(RuntimeError, match="forced"):
        soft.execute(config, tmp_path / "run", state, identity)
    assert originals == (
        soft.prior.training.load_model_bytes,
        soft.prior.training._save_bundle,
        soft.prior.dqn_ai.DQNAgent.update_target_network,
    )
    assert not list((tmp_path / "run").glob(".resume-*"))


def test_prior_corruption_fails_before_training(tmp_path, monkeypatch):
    export, integrity, manifest, artifact = [
        tmp_path / name for name in ("export.json", "integrity.json", "manifest.json", "final.pth")
    ]
    soft.prior.write(export, {"complete": True})
    manifest.write_text("pinned manifest")
    artifact.write_bytes(b"pinned model")
    soft.prior.write(
        integrity,
        {
            "complete": True,
            "manifest_sha256": soft.prior.digest(manifest),
            "artifact_sha256": {str(artifact): soft.prior.digest(artifact)},
            "export_sha256": {str(export): soft.prior.digest(export)},
        },
    )
    monkeypatch.setattr(soft, "PRIOR_EXPORT", export)
    monkeypatch.setattr(soft, "PRIOR_INTEGRITY", integrity)
    monkeypatch.setattr(soft.slow, "ROOT", tmp_path)
    monkeypatch.setattr(soft.slow, "manifest_at", lambda: {})
    artifact.write_bytes(b"changed model")
    with pytest.raises(ValueError, match="changed prior artifact"):
        soft.verified_prior()
