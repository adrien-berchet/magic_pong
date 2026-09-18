"""Analytic checks for frozen-policy gradient diagnostics."""

import random
from copy import deepcopy

import numpy as np
import pytest
import torch

from docs.ai.run_anchor_decline_diagnostics import adam_delta
from docs.ai.run_anchor_decline_diagnostics import batch_indices
from docs.ai.run_anchor_decline_diagnostics import fingerprint
from docs.ai.run_anchor_decline_diagnostics import frozen_state
from docs.ai.run_anchor_decline_diagnostics import gradient_batch
from docs.ai.run_anchor_decline_diagnostics import gradient_geometry
from docs.ai.run_anchor_decline_diagnostics import td_loss
from docs.ai.run_anchor_decline_diagnostics import verify_bank
from magic_pong.ai.models.dqn_ai import DQNAgent
from magic_pong.ai.models.dqn_ai import Transition


@pytest.fixture(autouse=True)
def cpu_diagnostics(monkeypatch):
    from magic_pong.ai.models import dqn_ai

    monkeypatch.setattr(dqn_ai, "device", torch.device("cpu"))


def test_opposed_gradients_report_cancellation_and_scale():
    row = gradient_geometry(torch.tensor([2.0, 0.0]), torch.tensor([-0.5, 0.0]))
    assert row["cosine"] == pytest.approx(-1.0)
    assert row["anchor_td_norm_ratio"] == pytest.approx(0.25)
    assert row["sgd_anchor_directional_change"] == pytest.approx(0.75)
    assert row["anchor_balance_multiplier"] == pytest.approx(4.0)


def test_zero_anchor_does_not_invent_alignment():
    row = gradient_geometry(torch.ones(2), torch.zeros(2))
    assert row["cosine"] is None
    assert row["anchor_balance_multiplier"] is None
    assert row["sgd_anchor_directional_change"] == 0.0


def test_uniform_and_balanced_batches_are_private_and_reproducible():
    random.seed(5)
    before = random.getstate()
    strata = torch.arange(16).repeat_interleave(128)
    a, b = batch_indices(20000, strata, 17, batches=8)
    a2, b2 = batch_indices(20000, strata, 17, batches=8)
    assert random.getstate() == before
    assert np.array_equal(a, a2) and np.array_equal(b, b2)
    assert all(len(set(row)) == 64 for row in a)
    assert all(torch.bincount(strata[row]).tolist() == [4] * 16 for row in b)


def test_diagnostic_td_matches_real_double_dqn_loss_and_terminal_mask():
    agent = DQNAgent(state_size=3, hidden_size=8, batch_size=4, gamma=0.91)
    states = torch.randn(4, 3)
    data = {
        "states": states,
        "next_states": torch.randn(4, 3),
        "actions": torch.tensor([0, 3, 2, 8]),
        "rewards": torch.tensor([1.0, 0.0, -1.0, 0.01]),
        "dones": torch.tensor([True, False, True, False]),
    }
    expected = agent._compute_dqn_loss(
        [
            Transition(
                states[i].numpy(),
                int(data["actions"][i]),
                float(data["rewards"][i]),
                data["next_states"][i].numpy(),
                bool(data["dones"][i]),
            )
            for i in range(4)
        ],
        torch.ones(4),
    )
    actual = td_loss(agent.q_network, agent.target_network, data, agent.gamma)
    assert torch.allclose(actual, expected)
    altered = {**data, "next_states": data["next_states"].clone()}
    altered["next_states"][data["dones"]] *= 100
    assert torch.allclose(actual, td_loss(agent.q_network, agent.target_network, altered, 0.91))


@pytest.mark.parametrize("scale", [0.01, 100.0])
def test_analytic_adam_delta_matches_real_clipped_step_without_mutation(scale):
    net = torch.nn.Linear(3, 2)
    opt = torch.optim.Adam(net.parameters(), lr=0.003)
    for p in net.parameters():
        p.grad = torch.ones_like(p) * 0.7
    opt.step()
    opt.zero_grad(set_to_none=True)
    params = list(net.parameters())
    grads = tuple(torch.randn_like(p) * scale for p in params)
    before = deepcopy(opt.state_dict())
    weights = [p.detach().clone() for p in params]
    predicted = adam_delta(opt, grads)
    assert all(torch.equal(p, w) for p, w in zip(params, weights))
    for key, state in before["state"].items():
        assert all(
            torch.equal(v, opt.state_dict()["state"][key][name]) for name, v in state.items()
        )
    for p, g in zip(params, grads):
        p.grad = g.clone()
    torch.nn.utils.clip_grad_norm_(params, 1.0)
    opt.step()
    actual = torch.cat([(p.detach() - w).flatten() for p, w in zip(params, weights)])
    torch.testing.assert_close(predicted, actual, atol=4e-8, rtol=3e-4)


def test_wrong_seed_bank_is_rejected():
    bank = {"metadata": {"reference_training_seed": 22, "reference_sha256": "reference"}}
    with pytest.raises(ValueError, match="reference mismatch"):
        verify_bank(bank, 11, "reference")


def test_gradient_probe_keeps_weights_optimizer_data_and_rng_frozen():
    agent = DQNAgent(state_size=33, score_context=True, hidden_size=8)
    for p in agent.q_network.parameters():
        p.grad = torch.ones_like(p)
    agent.optimizer.step()
    agent.optimizer.zero_grad(set_to_none=True)
    data = {
        "states": torch.randn(64, 33),
        "next_states": torch.randn(64, 33),
        "actions": torch.arange(64) % 9,
        "rewards": torch.randn(64),
        "dones": torch.zeros(64, dtype=torch.bool),
    }
    bank = {"states": torch.randn(64, 33), "actions": torch.arange(64) % 9}
    before = frozen_state(agent)
    data_before = fingerprint([data, bank])
    rng = fingerprint([random.getstate(), torch.get_rng_state()])
    row = gradient_batch(agent, agent.target_network, data, bank, np.arange(64))
    assert np.isfinite(row["adam_anchor_directional_change"])
    assert frozen_state(agent) == before
    assert fingerprint([data, bank]) == data_before
    assert fingerprint([random.getstate(), torch.get_rng_state()]) == rng
    assert all(p.grad is None for p in agent.q_network.parameters())
