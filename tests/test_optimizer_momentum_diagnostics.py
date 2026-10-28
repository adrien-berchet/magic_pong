"""Analytic, isolation and provenance guards for disposable momentum probes."""

import copy
import hashlib
import io
import json
import os
import subprocess
import sys
from pathlib import Path
from types import SimpleNamespace

import numpy as np
import pytest
import torch

from docs.ai import run_optimizer_momentum_diagnostics as diagnostic


@pytest.fixture(autouse=True)
def single_thread():
    old = torch.get_num_threads()
    torch.set_num_threads(1)
    yield
    torch.set_num_threads(old)


def toy(moment=0.4):
    network = torch.nn.Linear(2, 1, bias=False).double()
    with torch.no_grad():
        network.weight.copy_(torch.tensor([[0.3, -0.2]], dtype=torch.float64))
    optimizer = torch.optim.Adam(network.parameters(), lr=0.03, betas=(0.8, 0.95), eps=1e-6)
    p = network.weight
    optimizer.state[p] = {
        "step": torch.tensor(12.0),
        "exp_avg": torch.full_like(p, moment),
        "exp_avg_sq": torch.tensor([[0.1, 0.2]], dtype=torch.float64),
    }
    return network, optimizer


@pytest.mark.parametrize("gradient", ([[0.03, -0.04]], [[30.0, -40.0]], [[0.0, 0.0]]))
def test_first_moment_only_intervention_matches_manual_adam_and_preserves_inputs(gradient):
    network, optimizer = toy()
    grads = [torch.tensor(gradient, dtype=torch.float64)]
    before = diagnostic.fingerprint([network.state_dict(), optimizer.state_dict(), grads])
    original, delta, _ = diagnostic.virtual_step(network, optimizer, grads)
    cleared, cleared_delta, _ = diagnostic.virtual_step(network, optimizer, grads, clear=True)
    g = grads[0] * min(1, 1 / (float(grads[0].norm()) + 1e-6))
    variance = 0.95 * torch.tensor([[0.1, 0.2]], dtype=torch.float64) + 0.05 * g.square()
    denominator = (variance / (1 - 0.95**13)).sqrt() + 1e-6
    manual = -0.03 * ((0.2 * g) / (1 - 0.8**13)) / denominator
    torch.testing.assert_close(cleared.weight, network.weight + manual, rtol=1e-13, atol=1e-15)
    torch.testing.assert_close(cleared_delta, manual.flatten(), rtol=1e-12, atol=1e-15)
    assert not torch.equal(original.weight, cleared.weight)
    assert not torch.equal(delta, cleared_delta)
    assert before == diagnostic.fingerprint([network.state_dict(), optimizer.state_dict(), grads])
    assert network.weight.grad is None


def test_already_zero_first_moment_has_no_effect_and_norm_match_is_direction_preserving():
    network, optimizer = toy(moment=0)
    grads = [torch.tensor([[0.5, -0.6]], dtype=torch.float64)]
    original, delta, _ = diagnostic.virtual_step(network, optimizer, grads)
    cleared, other, _ = diagnostic.virtual_step(network, optimizer, grads, clear=True)
    assert torch.equal(original.weight, cleared.weight)
    assert torch.equal(delta, other)
    matched, actual, scale = diagnostic.norm_matched(network, delta, 0.2)
    assert scale > 0
    assert float(actual.norm()) == pytest.approx(0.2)
    assert float(torch.dot(actual, delta) / actual.norm() / delta.norm()) == pytest.approx(1)
    assert matched is not network
    with pytest.raises(ValueError, match="zero displacement"):
        diagnostic.norm_matched(network, torch.zeros_like(delta), 1)


def test_sampler_matches_strata_and_does_not_consume_global_rng():
    strata = torch.arange(16).repeat_interleave(128)
    before = diagnostic.rng_fingerprint()
    a, b = diagnostic.batches(20000, strata, 123, n=5)
    aa, bb = diagnostic.batches(20000, strata, 123, n=5)
    assert np.array_equal(a, aa) and np.array_equal(b, bb)
    assert all(len(set(row)) == 64 for row in a)
    assert all(torch.bincount(strata[row]).tolist() == [4] * 16 for row in b)
    assert before == diagnostic.rng_fingerprint()
    with pytest.raises(ValueError, match="sixteen"):
        diagnostic.batches(20000, torch.zeros(5), 123)


def test_double_dqn_targets_use_online_selection_target_values_and_terminal_mask():
    online = torch.nn.Linear(2, 2, bias=False)
    target = torch.nn.Linear(2, 2, bias=False)
    with torch.no_grad():
        online.weight.copy_(torch.tensor([[2.0, 1.0], [1.0, 2.0]]))
        target.weight.copy_(torch.tensor([[3.0, 20.0], [30.0, 4.0]]))
    batch = {
        "next_states": torch.eye(2),
        "rewards": torch.tensor([1.0, 2.0]),
        "dones": torch.tensor([False, True]),
    }
    values = diagnostic.fixed_targets(online, target, batch, 0.9)
    torch.testing.assert_close(values, torch.tensor([3.7, 2.0]))
    assert not values.requires_grad
    with torch.no_grad():
        online.weight.zero_()
        target.weight.zero_()
    torch.testing.assert_close(values, torch.tensor([3.7, 2.0]))


def test_corrupt_hash_and_import_leakage_rejected(tmp_path, monkeypatch):
    p = tmp_path / "fork.pt"
    p.write_bytes(b"verified fork")
    pins = {str(p): diagnostic.digest(p)}
    diagnostic.verify_hashes(pins)
    p.write_bytes(b"changed fork")
    with pytest.raises(ValueError, match="hash changed"):
        diagnostic.verify_hashes(pins)
    monkeypatch.setitem(
        sys.modules, "magic_pong", SimpleNamespace(__file__=str(tmp_path / "live.py"))
    )
    with pytest.raises(ValueError, match="already imported"):
        diagnostic.historical_api()


def test_source_archive_and_script_corruption_rejected(tmp_path, monkeypatch):
    monkeypatch.chdir(tmp_path)
    root = tmp_path / "experiment"
    archive = root / "historical_source"
    own = root / "diagnostic_source"
    archive.mkdir(parents=True)
    own.mkdir()
    for p in (Path("app.py"), archive / "app.py", Path("probe.py"), own / "probe.py"):
        p.write_text("original\n")
    m = {
        "historical_source_hashes": {"app.py": diagnostic.digest(archive / "app.py")},
        "live_source_hashes": {"app.py": diagnostic.digest("app.py")},
        "input_hashes": {},
        "scripts": {"probe.py": diagnostic.digest("probe.py")},
        "runtime": {"torch": str(torch.__version__), "device": "cpu", "threads": 1},
    }
    diagnostic.write(root / "manifest.json", m)
    monkeypatch.setattr(diagnostic, "ROOT", root)
    diagnostic.check()
    for p in (archive / "app.py", own / "probe.py", Path("app.py")):
        p.write_text("changed\n")
        with pytest.raises(ValueError, match="hash changed"):
            diagnostic.check()
        p.write_text("original\n")


def test_wrong_bank_and_fork_counters_rejected():
    # The identity checks precede the heavier tensor-shape checks.
    stream = io.BytesIO()
    torch.save({"step_count": 750000, "training_step": 74901}, stream)
    bank_stream = io.BytesIO()
    torch.save({"metadata": {"reference_training_seed": 22}}, bank_stream)
    blob = bank_stream.getvalue()
    bundle = {
        "config": {
            "anchor_bank_sha256": hashlib.sha256(blob).hexdigest(),
            "hidden_layers": (256, 128),
        },
        "model": stream.getvalue(),
        "anchor": {"bank_bytes": blob},
        "progress": {"total_frames": 750000},
    }
    spec = {
        "seed": 11,
        "config": json.loads(json.dumps(bundle["config"])),
        "fork_frames": 750000,
        "fork_training_step": 74901,
    }
    with pytest.raises(ValueError, match="wrong seed bank"):
        diagnostic.validate_fork(bundle, spec)
    wrong_config = copy.deepcopy(spec)
    wrong_config["config"]["hidden_layers"] = [128, 64]
    with pytest.raises(ValueError, match="configuration differs"):
        diagnostic.validate_fork(bundle, wrong_config)
    bad = copy.deepcopy(spec)
    bad["fork_frames"] += 1
    with pytest.raises(ValueError, match="fork counters"):
        diagnostic.validate_fork(bundle, bad)


def test_unsupported_adam_variant_rejected():
    net, optimizer = toy()
    optimizer.param_groups[0]["amsgrad"] = True
    with pytest.raises(ValueError, match="unsupported"):
        diagnostic.virtual_step(net, optimizer, [torch.zeros_like(net.weight)])


def test_historical_training_path_and_probe_reproducibility_in_fresh_process(tmp_path):
    archive = diagnostic.HISTORY / "source_snapshot"
    if not archive.exists():
        pytest.skip("historical experiment artifacts unavailable")
    code = r"""
import copy,json,shutil,sys
from pathlib import Path
import numpy as np
import torch
from docs.ai import run_optimizer_momentum_diagnostics as d
torch.set_num_threads(1)
d.ROOT=Path(sys.argv[1])
shutil.copytree(d.HISTORY/'source_snapshot'/'magic_pong',d.ROOT/'historical_source'/'magic_pong')
api,loss=d.historical_api()
with torch.random.fork_rng(devices=[]):
 torch.manual_seed(123)
 online=api.DQNNetwork(33,hidden_layers=[256,128],score_context=True)
 target=copy.deepcopy(online)
 states=torch.randn(2048,33)
 actions=torch.arange(2048)%9
opt=torch.optim.Adam(online.parameters(),lr=3e-5)
for p in online.parameters():
 opt.state[p]={'step':torch.tensor(10.),'exp_avg':torch.ones_like(p)*0.002,'exp_avg_sq':torch.ones_like(p)*0.02}
bank={'states':states,'actions':actions}
data={'states':states[:64],'next_states':states[64:128], 'actions':actions[:64],
      'rewards':torch.linspace(-1,1,64),'dones':torch.arange(64)%3==0}
targets=d.fixed_targets(online,target,data,.99)
indices=np.arange(64)
before=d.fingerprint([online.state_dict(),target.state_dict(),opt.state_dict(),data,bank])
rng=d.rng_fingerprint()
for step in (10,19):
 audit=d.verify_training_path(online,target,opt,data,bank,indices,targets,api,loss,
       gamma=.99,training_step=step,target_update_frequency=20)
 assert audit['after_training_step']==step+1
 assert audit['target_refreshed']==(step==19)
gradient=d.flatten(torch.autograd.grad(.05*loss(online(states),actions,.01),tuple(online.parameters()))).double()
a=d.probe_batch(online,opt,data,targets,bank,indices,loss,gradient)
b=d.probe_batch(online,opt,data,targets,bank,indices,loss,gradient)
assert a==b
assert before==d.fingerprint([online.state_dict(),target.state_dict(),opt.state_dict(),data,bank])
assert rng==d.rng_fingerprint()
assert all(r['unsampled_bank_rows']==1984 for r in a)
assert all(Path(p).is_relative_to((d.ROOT/'historical_source').resolve()) for p in d.module_paths().values())
print(json.dumps(audit))
"""
    completed = subprocess.run(
        [sys.executable, "-c", code, str(tmp_path / "run")],
        env={**os.environ, "PYTHONDONTWRITEBYTECODE": "1"},
        text=True,
        capture_output=True,
        timeout=60,
    )
    assert completed.returncode == 0, completed.stdout + completed.stderr
    assert json.loads(completed.stdout.splitlines()[-1])["archived_training_step_exact"]


def test_summary_rejects_missing_pairs_and_counts_seeds_not_batches(monkeypatch):
    monkeypatch.setattr(diagnostic, "BATCHES", 2)
    rows = []
    for seed, improvement in ((11, 1.0), (22, -0.1), (33, 0.5)):
        for i in range(2):
            for arm in diagnostic.ARMS:
                value = 2.0 if arm == "original" else 2.0 - improvement
                rows.append(
                    {
                        "seed": seed,
                        "batch": i,
                        "arm": arm,
                        "weighted_loss_change": value,
                        "gap_change": 0,
                        "agreement_change": 0,
                        "weighted_loss_change_unsampled": value,
                        "gap_change_unsampled": 0,
                        "delta_norm": 0,
                        "bank_loss_linear_change": 0,
                        "td_loss_change_fixed_targets": 0,
                    }
                )
    _, decision = diagnostic.summarize(rows)
    assert decision["positive_seeds"] == 2 and decision["mechanistic_lead"]
    assert not decision["competitive_strength_tested"]
    with pytest.raises(ValueError, match="incomplete"):
        diagnostic.summarize(rows[:-1])
