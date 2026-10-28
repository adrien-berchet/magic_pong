"""Causal controls, loss semantics, match isolation and artifact verification."""

import copy
import importlib.util
import json
import sys
from pathlib import Path

import numpy as np
import pytest
import torch
import torch.nn.functional as F

SCRIPTS = Path(__file__).resolve().parents[1] / "docs" / "ai"
sys.path.insert(0, str(SCRIPTS))
spec = importlib.util.spec_from_file_location(
    "geometry_cost_runner", SCRIPTS / "run_planner_geometry_cost.py"
)
experiment = importlib.util.module_from_spec(spec)
spec.loader.exec_module(experiment)


class Base(torch.nn.Module):
    def __init__(self):
        super().__init__()
        self.fc_layers = torch.nn.ModuleList([torch.nn.Linear(30, 16), torch.nn.Linear(16, 8)])
        self.output_layer = torch.nn.Linear(8, 9)

    def forward(self, inputs):
        for layer in self.fc_layers:
            inputs = F.relu(layer(inputs))
        return self.output_layer(inputs)


def test_geometry_preserves_initial_predictions_and_zero_control_gradient():
    torch.manual_seed(17)
    base = Base()
    probe = experiment.ProbeNetwork(base)
    states = torch.randn(31, 30)
    features = torch.randn(31, len(experiment.FEATURE_NAMES))
    assert torch.equal(base(states), probe(torch.cat((states, features), 1)))
    zeros = torch.cat((states, torch.zeros_like(features)), 1)
    optimizer = torch.optim.Adam(probe.parameters(), lr=0.01)
    optimizer.zero_grad()
    probe(zeros).square().mean().backward()
    assert torch.count_nonzero(probe.projection.grad).item() == 0
    optimizer.step()
    assert torch.count_nonzero(probe.projection).item() == 0


def test_cost_gradient_distinguishes_expensive_wrong_actions():
    values = torch.tensor([[1000.0, 0.0, 999.0, 998.0, 997.0, 996.0, 995.0, 994.0, 993.0]])
    logits = torch.zeros_like(values, requires_grad=True)
    base = torch.autograd.grad(experiment.training_loss(logits, values), logits)[0]
    changed = torch.autograd.grad(experiment.training_loss(logits, values, True), logits)[0]
    delta = changed - base
    assert delta[0, 1] > delta[0, 2]  # Descent suppresses the catastrophic action more.
    assert delta[0, 0] < 0  # Best action is encouraged.
    assert torch.isfinite(changed).all()


def test_tied_costs_add_no_loss_or_gradient():
    logits = torch.randn(8, 9, requires_grad=True)
    values = torch.zeros_like(logits)
    base = experiment.training_loss(logits, values)
    costly = experiment.training_loss(logits, values, True)
    assert torch.equal(base, costly)
    assert torch.equal(torch.autograd.grad(base, logits)[0], torch.autograd.grad(costly, logits)[0])


def test_legacy_ce_and_probe_control_take_same_optimizer_step():
    torch.manual_seed(123)
    raw = Base()
    probe = experiment.ProbeNetwork(raw)
    values, states = torch.randn(31, 9), torch.randn(31, 30)
    target = experiment.prior.soft_targets(values)
    raw_loss = -(target * F.log_softmax(raw(states), dim=1)).sum(1).mean()
    probe_loss = experiment.training_loss(
        probe(torch.cat((states, torch.zeros(31, len(experiment.FEATURE_NAMES))), 1)), values
    )
    assert torch.equal(raw_loss, probe_loss)
    for model, loss in ((raw, raw_loss), (probe, probe_loss)):
        optimizer = torch.optim.Adam(model.parameters(), lr=0.001)
        optimizer.zero_grad()
        loss.backward()
        optimizer.step()
    for left, right in zip(raw.parameters(), probe.base.parameters()):
        assert torch.equal(left, right)


def test_old_test_rows_are_physically_excluded(tmp_path):
    directory = tmp_path / "seed_11"
    directory.mkdir()
    np.savez(
        directory / "dataset.npz",
        states=np.zeros((4, 30), np.float32),
        match=np.arange(4),
        teacher=np.array([0, 0, 0, 8]),
    )
    records = [
        {"index": i, "split": split} for i, split in enumerate(("train", "train", "dev", "test"))
    ]
    (directory / "collection.json").write_text(json.dumps({"matches": records}))
    data, masks = experiment.load_training(11, {"prior_root": str(tmp_path)})
    assert data["match"].tolist() == [0, 1, 2]
    assert set(masks) == {"train", "dev"}
    assert not np.any(masks["train"] & masks["dev"])


def test_reserved_seeds_are_disjoint():
    fresh = experiment.test_matches()
    assert len(fresh) == 32
    seeds = {r["seed"] for r in fresh}
    assert len(seeds) == len(fresh)
    assert seeds.isdisjoint(r["seed"] for r in experiment.prior.match_specs())
    assert seeds.isdisjoint(experiment.SETTINGS["gameplay_seeds"])


def test_earliest_dev_tie_keeps_epoch_zero(monkeypatch):
    monkeypatch.setitem(experiment.SETTINGS, "epochs", 2)
    network = experiment.ProbeNetwork(Base())
    initial = copy.deepcopy(network.state_dict())
    data = {
        "states": np.zeros((8, 30 + len(experiment.FEATURE_NAMES)), np.float32),
        "values": np.zeros((8, 9), np.float32),
        "teacher": np.zeros(8, int),
    }
    masks = {"train": np.arange(8) < 4, "dev": np.arange(8) >= 4}
    selected, record = experiment.fit(network, data, masks, 11, True)
    assert record["selected_epoch"] == 0
    assert record["updates"] == 2
    assert all(torch.equal(initial[k], v) for k, v in selected.state_dict().items())


def test_verifier_rejects_output_mutation(tmp_path, monkeypatch):
    monkeypatch.setattr(experiment, "read_manifest", lambda _: {})
    (tmp_path / "manifest.json").write_text("{}")
    directory = tmp_path / "seed_11"
    directory.mkdir()
    artifact = directory / "result.json"
    artifact.write_text("original")
    experiment.seal(tmp_path, directory / "completion_geometry.json", [artifact])
    assert experiment.verify(tmp_path)
    artifact.write_text("changed")
    with pytest.raises(ValueError, match="hash|Hash|changed|Changed|mismatch"):
        experiment.verify(tmp_path)


def test_feature_matrix_hash_covers_values_and_shape():
    x = np.zeros((2, 30), np.float32)
    y = x.copy()
    y[0, 0] = 1
    assert experiment.array_sha(x) != experiment.array_sha(y)
    assert experiment.array_sha(x) != experiment.array_sha(x.reshape(1, 60))


def test_manifest_rejects_changed_source_and_feature_schema(tmp_path):
    source = tmp_path / "sealed_input.npz"
    source.write_bytes(b"sealed observations")
    manifest = {
        "schema": experiment.SCHEMA,
        "settings": experiment.SETTINGS,
        "matches": experiment.test_matches(),
        "feature_names": list(experiment.FEATURE_NAMES),
        "application_archive": str(tmp_path / "archive"),
        "runtime": {
            "torch": torch.__version__,
            "numpy": np.__version__,
            "numba": experiment.importlib.metadata.version("numba"),
        },
        "input_sha256": {str(source): experiment.sha(source)},
    }
    path = tmp_path / "manifest.json"
    path.write_text(json.dumps(manifest))
    assert experiment.read_manifest(tmp_path)["schema"] == experiment.SCHEMA
    source.write_bytes(b"mutated observations")
    with pytest.raises(ValueError):
        experiment.read_manifest(tmp_path)
    source.write_bytes(b"sealed observations")
    manifest["feature_names"] = ["different feature"]
    path.write_text(json.dumps(manifest))
    with pytest.raises(ValueError, match="Protocol mismatch"):
        experiment.read_manifest(tmp_path)


@pytest.mark.parametrize("side", [1, 2])
def test_gameplay_encoder_matches_offline_canonical_features(side):
    from magic_pong.ai.agent_adapter import from_player1_dqn_action
    from magic_pong.ai.agent_adapter import to_player1_dqn_observation
    from magic_pong.ai.models.dqn_ai import ACTION_MAPPING
    from magic_pong.ai.models.dqn_ai import DQNAgent
    from magic_pong.utils.config import game_config

    config = experiment.ContactGeometryConfig(
        width=game_config.FIELD_WIDTH,
        height=game_config.FIELD_HEIGHT,
        max_ball_speed=game_config.MAX_BALL_SPEED,
        paddle_width=game_config.PADDLE_WIDTH,
        paddle_height=game_config.PADDLE_HEIGHT,
        paddle_speed=game_config.PADDLE_SPEED,
        ball_radius=game_config.BALL_RADIUS,
        fps=game_config.FPS,
    )
    original = DQNAgent()
    observation = {
        "positions_normalized": False,
        "ball_pos": [400.0, 250.0],
        "ball_vel": [-200.0, 90.0],
        "player_pos": [20.0, 150.0],
        "opponent_pos": [750.0, 250.0],
        "opponent_previous_pos": [750.0, 245.0],
        "player_paddle_size": [20.0, 100.0],
        "opponent_paddle_size": [20.0, 100.0],
    }
    if side == 2:
        observation["player_pos"], observation["opponent_pos"] = (
            observation["opponent_pos"],
            observation["player_pos"],
        )
        observation["ball_vel"][0] = 200.0
    state = original._observation_to_state(to_player1_dqn_observation(observation, side))
    expected = experiment.augment(state[None, :], config, True)

    class Capture(torch.nn.Module):
        def forward(self, inputs):
            np.testing.assert_array_equal(inputs.numpy(), expected)
            return torch.tensor([[0.0, 0.0, 0.0, 0.0, 1.0, 0.0, 0.0, 0.0, 0.0]])

    agent = experiment.make_probe_agent(Capture(), original, config, True)
    agent.bind_physics(None, side)
    actual = agent.get_action(observation)
    target = from_player1_dqn_action(ACTION_MAPPING[4], side)
    assert vars(actual) == vars(target)
