"""Serialization normalization must retain full checkpoint and raw-file integrity."""

import hashlib
import io
from copy import deepcopy

import pytest
import torch

from docs.ai.audit_score_context_experiment import equal
from docs.ai.audit_score_context_experiment import network_hash
from docs.ai.run_target_refresh_audit import canonical_bundle


def make_state(reverse=False, optimizer_value=1.0):
    model = {
        "q_network_state_dict": {"w": torch.tensor([1.0, 2.0])},
        "target_network_state_dict": {"w": torch.tensor([3.0, 4.0])},
        "optimizer_state_dict": {"moment": torch.tensor([optimizer_value])},
        "metadata": {"version": 1},
        "training_step": 7,
    }
    if reverse:
        model = dict(reversed(list(model.items())))
    stream = io.BytesIO()
    torch.save(model, stream)
    blob = stream.getvalue()
    sha = hashlib.sha256(blob).hexdigest()
    report = {
        "model_sha256": sha,
        "network_sha256": network_hash(model),
        "checkpoint_path": "/old/candidate.pth",
    }
    return {
        "status": "complete",
        "model": blob,
        "best": {"model": blob, "report": dict(report)},
        "phase_best": {"model": blob, "report": dict(report)},
        "last_report": dict(report),
        "league": {
            "files": {"snapshot.pth": blob},
            "manifest": {"entries": [{"path": "/old/snapshot.pth", "metadata": {"sha256": sha}}]},
        },
        "replay": {"data": torch.tensor([9.0])},
        "anchor": {"bank_bytes": b"unchanged"},
    }


def test_equal_contents_with_different_serialization_and_paths_match():
    a, b = make_state(), make_state(True)
    for state in (a, b):
        state["league"]["manifest"]["entries"].append(
            {"kind": "builtin", "path": None, "metadata": {}}
        )
    assert a["model"] != b["model"]
    b["last_report"]["checkpoint_path"] = "/new/candidate.pth"
    b["league"]["manifest"]["entries"][0]["path"] = "/new/snapshot.pth"
    equal(canonical_bundle(a), canonical_bundle(b))


def test_optimizer_change_is_not_hidden_by_equal_online_weights():
    with pytest.raises(ValueError):
        equal(canonical_bundle(make_state()), canonical_bundle(make_state(True, 2.0)))


@pytest.mark.parametrize("slot", ["best", "phase_best", "last_report"])
def test_wrong_report_hash_rejected_before_normalization(slot):
    state = make_state()
    report = state[slot] if slot == "last_report" else state[slot]["report"]
    report["model_sha256"] = "bad"
    with pytest.raises(ValueError, match="hash"):
        canonical_bundle(state)


def test_corrupt_league_hash_rejected():
    state = make_state()
    state["league"]["manifest"]["entries"][0]["metadata"]["sha256"] = "bad"
    with pytest.raises(ValueError, match="hash"):
        canonical_bundle(state)


def test_replay_and_anchor_bytes_are_not_normalized_or_mutated():
    a = make_state()
    before = deepcopy(a)
    b = deepcopy(a)
    b["anchor"]["bank_bytes"] = b"changed"
    with pytest.raises(ValueError):
        equal(canonical_bundle(a), canonical_bundle(b))
    equal(a, before)
    b = deepcopy(a)
    b["replay"]["data"][0] = 8.0
    with pytest.raises(ValueError):
        equal(canonical_bundle(a), canonical_bundle(b))
