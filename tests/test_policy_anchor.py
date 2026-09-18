"""Action anchoring must preserve control RNG, phase timing and exact resume."""

import hashlib
import io
import random
from dataclasses import replace

import pytest
import torch

from docs.ai.collect_policy_anchor_bank import contact_rows
from magic_pong.ai.policy_anchor import PolicyAnchor
from magic_pong.ai.policy_anchor import action_margin_loss
from magic_pong.ai.scheduled_training import run_frame_pipeline
from tests.test_scheduled_training import assert_nested_equal
from tests.test_scheduled_training import initial_model
from tests.test_scheduled_training import payload
from tests.test_scheduled_training import small_config


def make_bank(tmp_path):
    generator = torch.Generator().manual_seed(82041)
    bank = {
        "schema": "magic_pong.policy_anchor_bank.v1",
        "states": torch.rand(32, 33, generator=generator),
        "actions": torch.arange(32) % 9,
        "strata": torch.arange(32) % 4,
        "metadata": {"reference_sha256": "a" * 64, "collection_seeds": [82041]},
    }
    path = tmp_path / "bank.pt"
    torch.save(bank, path)
    return path, hashlib.sha256(path.read_bytes()).hexdigest()


def test_margin_anchors_actions_without_anchoring_value_level():
    q = torch.tensor([[0.0, 0.04, -0.1], [0.4, 0.1, 0.0]], requires_grad=True)
    actions = torch.tensor([0, 0])
    loss = action_margin_loss(q, actions, 0.01)
    assert loss.item() == pytest.approx(0.05**2 / 2)
    assert action_margin_loss(q + 5, actions, 0.01).item() == pytest.approx(loss.item(), abs=1e-7)
    loss.backward()
    assert q.grad[0, 0] < 0 < q.grad[0, 1]
    assert torch.count_nonzero(q.grad[1]) == 0


def test_anchor_sampling_is_balanced_independent_and_resumable(tmp_path):
    path, sha = make_bank(tmp_path)
    anchor = PolicyAnchor(path.read_bytes(), sha, weight=0.1, margin=0.01, seed=17, state_size=33)
    before = random.getstate(), torch.get_rng_state().clone()
    for _ in range(3):
        indices = anchor.sample_indices(8)
        assert torch.bincount(anchor.strata[indices]).tolist() == [2, 2, 2, 2]
    assert random.getstate() == before[0]
    assert torch.equal(torch.get_rng_state(), before[1])
    restored = PolicyAnchor(path.read_bytes(), sha, weight=0.1, margin=0.01, seed=99, state_size=33)
    restored.restore(anchor.state_dict())
    assert restored.sample_indices(8) == anchor.sample_indices(8)
    with pytest.raises(ValueError, match="hash"):
        PolicyAnchor(path.read_bytes(), "0" * 64, weight=0.1, margin=0.01, seed=17, state_size=33)
    with pytest.raises(ValueError, match="width"):
        PolicyAnchor(path.read_bytes(), sha, weight=0.1, margin=0.01, seed=17, state_size=30)


@pytest.mark.parametrize("pause_episode", [3, 12, 13])
def test_anchor_final_phase_control_equivalence_and_resume(tmp_path, pause_episode):
    initial = initial_model(tmp_path)
    path, sha = make_bank(tmp_path)
    control = small_config(score_context=True, selfplay_opponent="follow_ball")
    config = replace(control, anchor_weight=0.05, anchor_bank_sha256=sha)
    run_frame_pipeline(control, tmp_path / "control", initial_checkpoint=initial)
    full = run_frame_pipeline(
        config, tmp_path / "full", initial_checkpoint=initial, anchor_bank=path
    )
    paused = run_frame_pipeline(
        config,
        tmp_path / "split",
        initial_checkpoint=initial,
        anchor_bank=path,
        stop_after_episodes=pause_episode,
    )
    assert paused["status"] == "paused"
    assert paused["anchor"]["updates"] == (4 if pause_episode == 13 else 0)
    path.unlink()  # Resuming must use the verified bank embedded in the bundle.
    resumed = run_frame_pipeline(
        config, tmp_path / "split", resume=tmp_path / "split/latest.training.pt"
    )
    assert full["status"] == resumed["status"] == "complete"
    a = payload(tmp_path / "full/latest.training.pt")
    b = payload(tmp_path / "split/latest.training.pt")
    for key in ("replay", "rng", "progress", "episodes", "anchor"):
        assert_nested_equal(a[key], b[key])
    assert_nested_equal(payload(io.BytesIO(a["model"])), payload(io.BytesIO(b["model"])))
    for phase in ("curriculum", "anneal"):
        assert_nested_equal(
            payload(tmp_path / f"control/{phase}_final.pth"),
            payload(tmp_path / f"full/{phase}_final.pth"),
        )
    assert a["anchor"]["updates"] == 16
    assert full["anchor"]["updates"] == 16
    assert any(
        not torch.equal(v, payload(tmp_path / "control/final.pth")["q_network_state_dict"][k])
        for k, v in payload(tmp_path / "full/final.pth")["q_network_state_dict"].items()
    )


def test_anchor_configuration_requires_pinned_bank_and_valid_loss():
    for changes in (
        {"anchor_weight": -1},
        {"anchor_weight": float("nan")},
        {"anchor_weight": 0.1},
        {"anchor_margin": 0},
        {"anchor_bank_sha256": "abc"},
    ):
        with pytest.raises(ValueError, match="anchor"):
            small_config(**changes)


def test_contact_bank_limits_recent_successful_window():
    pending = [{"frame": frame, "action": 1} for frame in range(50)]
    rows = contact_rows(pending, 49)
    assert len(rows) == 4
    assert rows[0]["frame"] == 20
    assert rows[-1]["frame"] == 49
    assert all(0 <= row["contact_delay"] < 30 for row in rows)
    assert contact_rows([], 50) == []
    assert contact_rows([{"frame": 1}], 50) == []
