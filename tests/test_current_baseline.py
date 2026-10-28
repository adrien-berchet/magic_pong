"""The baseline observer must preserve the stock trainer and evaluator."""

import copy
import random

import numpy as np
import pytest
import torch

from docs.ai import run_current_baseline as baseline


@pytest.fixture(autouse=True)
def one_thread():
    before = torch.get_num_threads()
    torch.set_num_threads(1)
    yield
    torch.set_num_threads(before)


@pytest.mark.parametrize("cap", [1, 1200])
def test_observer_matches_stock_evaluation_and_preserves_state(cap):
    code = baseline.api(archived=False)
    torch.manual_seed(451)
    net = code.dqn.DQNNetwork(30, hidden_layers=[16, 8])
    net.train()
    options = {
        "opponents": ("dummy", "follow_ball"),
        "seeds": 3,
        "max_frames": cap,
        "match_target": 1,
        "seed": 719,
    }
    before = baseline.network_hash(net.state_dict())
    random_before = random.getstate()
    torch_before = torch.get_rng_state().clone()
    numpy_before = np.random.get_state()
    stock = code.stock_evaluate(net, torch.device("cpu"), **options)
    random_after = random.getstate()
    torch_after = torch.get_rng_state().clone()
    numpy_after = repr(np.random.get_state())
    random.setstate(random_before)
    torch.set_rng_state(torch_before)
    np.random.set_state(numpy_before)
    report = baseline.raw_vector_evaluation(code, net, torch.device("cpu"), **options)
    for key in stock:
        assert report[key] == stock[key], key
    assert net.training and before == baseline.network_hash(net.state_dict())
    # Match the stock evaluator's RNG effects, including GameState construction.
    assert random.getstate() == random_after
    assert torch.equal(torch_after, torch.get_rng_state())
    assert repr(np.random.get_state()) == numpy_after
    if cap == 1:
        assert report["draws"] == report["matches"]
    else:
        assert report["wins"] + report["losses"] > 0
    corrupt = copy.deepcopy(report["raw_matches"])
    corrupt[0]["goals_for"] += 1
    with pytest.raises(ValueError, match="goal totals"):
        baseline.validate_rows(report, corrupt)


def test_pins_and_network_hash_detect_real_changes(tmp_path):
    p = tmp_path / "source.py"
    p.write_text("original")
    pins = {str(p): baseline.digest(p)}
    baseline.verify_hashes(pins)
    p.write_text("changed")
    with pytest.raises(ValueError, match="pinned file changed"):
        baseline.verify_hashes(pins)
    a = {"a": torch.tensor([1.0]), "b": torch.tensor([2.0])}
    assert baseline.network_hash(a) == baseline.network_hash(dict(reversed(list(a.items()))))
    a["a"] += 1
    assert baseline.network_hash(a) != baseline.network_hash(
        {"a": torch.tensor([1.0]), "b": a["b"]}
    )


def test_panel_audit_rejects_duplicate_cases_and_wrong_randomness():
    rows = [
        {
            "opponent": "dummy",
            "player_side": s,
            "bonuses_enabled": b,
            "replicate": r,
            "panel_seed": 719,
            "evaluator": "vector",
        }
        for s in (1, 2)
        for b in (False, True)
        for r in range(20)
    ]
    manifest = {"opponents": ["dummy"]}
    baseline.validate_panel({"raw_matches": rows}, manifest, 719)
    with pytest.raises(ValueError, match="seed or evaluator"):
        baseline.validate_panel({"raw_matches": rows}, manifest, 720)
    rows[0] = rows[1].copy()
    with pytest.raises(ValueError, match="incomplete or duplicated"):
        baseline.validate_panel({"raw_matches": rows}, manifest, 719)


def test_short_training_and_selection_unchanged_by_observer(tmp_path, monkeypatch):
    code = baseline.api(archived=False)
    config = code.training.VectorTrainingConfig(
        total_frames=256,
        n_envs=4,
        action_repeat=4,
        batch_size=8,
        replay_capacity=256,
        min_replay=16,
        snapshot_interval_frames=128,
        target_update_frames=64,
        eval_interval_frames=128,
        checkpoint_interval_frames=128,
        eval_seeds=1,
        eval_max_frames=20,
        eval_opponents=("dummy",),
        opponent_mix={"dummy": 1.0},
        hidden_layers=(16, 8),
        seed=151,
        device="cpu",
    )
    stock = code.training.VectorDQNTrainer(config, tmp_path / "stock")
    stock.run(verbose=False)
    observed = code.training.VectorDQNTrainer(config, tmp_path / "observed")

    def wrapped(network, device, **kwargs):
        return baseline.raw_vector_evaluation(code, network, device, **kwargs)

    monkeypatch.setattr(code.training, "evaluate_network", wrapped)
    observed.run(verbose=False)
    assert stock.frames == observed.frames and stock.updates == observed.updates
    assert stock.best_frames == observed.best_frames and stock.best_rank == observed.best_rank
    assert stock.losses == observed.losses
    assert baseline.network_hash(stock.network.state_dict()) == baseline.network_hash(
        observed.network.state_dict()
    )
    assert baseline.network_hash(stock.target.state_dict()) == baseline.network_hash(
        observed.target.state_dict()
    )
    path = observed.save("serialization_probe.pth")
    digest = baseline.digest(path)
    observed.save("serialization_probe.pth")
    assert baseline.digest(path) == digest
