"""Replay semantics, configuration checks and an end-to-end tiny vectorised run."""

import numpy as np
import pytest
import torch

pytest.importorskip("numba")

from magic_pong.ai.competitive import CompetitiveConfig  # noqa: E402
from magic_pong.ai.competitive import evaluate_competitive_agent  # noqa: E402
from magic_pong.ai.models.dqn_ai import DQNAgent  # noqa: E402
from magic_pong.ai.vector_training import ReplayBuffer  # noqa: E402
from magic_pong.ai.vector_training import VectorDQNTrainer  # noqa: E402
from magic_pong.ai.vector_training import VectorTrainingConfig  # noqa: E402
from magic_pong.ai.vector_training import evaluate_network  # noqa: E402
from magic_pong.ai.vector_training import main  # noqa: E402


class _FixedRng:
    """Return predetermined index arrays for offsets then environments."""

    def __init__(self, offsets, envs):
        self.calls = [np.array(offsets), np.array(envs)]

    def integers(self, high, size=None):
        values = self.calls.pop(0)
        assert (values < high).all()
        return values


def _filled_buffer(gamma: float) -> ReplayBuffer:
    buffer = ReplayBuffer(capacity=8, n_envs=1, obs_dim=1)
    rewards = [1.0, 2.0, 3.0, 4.0, 5.0]
    dones = [False, False, True, False, False]
    cuts = [False, False, False, True, False]
    for t, (r, d, c) in enumerate(zip(rewards, dones, cuts)):
        buffer.add(
            np.array([[t]], dtype=np.float32),
            np.array([t % 9]),
            np.array([r], dtype=np.float32),
            np.array([[t + 0.5]], dtype=np.float32),
            np.array([d]),
            np.array([c]),
        )
    return buffer


def test_n_step_return_stops_at_a_terminal_and_marks_no_bootstrap() -> None:
    gamma = 0.9
    batch = _filled_buffer(gamma).sample(1, 3, gamma, _FixedRng([0], [0]))
    assert batch["returns"][0] == pytest.approx(1 + gamma * 2 + gamma**2 * 3)
    assert bool(batch["terminal"][0]) is True
    assert batch["discount"][0] == pytest.approx(gamma**3)
    assert batch["next_obs"][0, 0] == pytest.approx(2.5)
    assert batch["obs"][0, 0] == 0.0 and batch["actions"][0] == 0


def test_n_step_return_stops_at_a_cut_but_still_bootstraps() -> None:
    gamma = 0.9
    # A three-step window cannot start at the fourth of five decisions yet; use two steps.
    batch = _filled_buffer(gamma).sample(1, 2, gamma, _FixedRng([3], [0]))
    assert batch["returns"][0] == pytest.approx(4.0)
    assert bool(batch["terminal"][0]) is False
    assert batch["discount"][0] == pytest.approx(gamma)
    assert batch["next_obs"][0, 0] == pytest.approx(3.5)


def test_n_step_return_spans_the_full_window_when_nothing_ends() -> None:
    gamma = 0.5
    buffer = ReplayBuffer(capacity=8, n_envs=1, obs_dim=1)
    for t in range(4):
        buffer.add(
            np.array([[t]], dtype=np.float32),
            np.array([0]),
            np.array([1.0], dtype=np.float32),
            np.array([[t + 0.5]], dtype=np.float32),
            np.array([False]),
            np.array([False]),
        )
    batch = buffer.sample(1, 3, gamma, _FixedRng([0], [0]))
    assert batch["returns"][0] == pytest.approx(1 + 0.5 + 0.25)
    assert batch["discount"][0] == pytest.approx(0.125)
    assert batch["next_obs"][0, 0] == pytest.approx(2.5)
    assert len(buffer) == 4


def test_ring_buffer_wraps_and_reports_size() -> None:
    buffer = ReplayBuffer(capacity=6, n_envs=2, obs_dim=1)
    for t in range(5):
        buffer.add(
            np.full((2, 1), t, dtype=np.float32),
            np.zeros(2, dtype=np.int64),
            np.zeros(2, dtype=np.float32),
            np.zeros((2, 1), dtype=np.float32),
            np.zeros(2, dtype=np.bool_),
            np.zeros(2, dtype=np.bool_),
        )
    assert buffer.rows == 3 and len(buffer) == 6
    batch = buffer.sample(4, 1, 0.99, np.random.default_rng(0))
    assert set(batch["obs"][:, 0].tolist()) <= {2.0, 3.0, 4.0}


@pytest.mark.parametrize(
    "overrides",
    [
        {"gamma": 0.0},
        {"epsilon_start": 0.1, "epsilon_end": 0.5},
        {"min_replay": 10, "replay_capacity": 5},
        {"opponent_mix": {"nobody": 1.0}},
        {"eval_opponents": ("planner",)},
        {"n_envs": 1000, "replay_capacity": 1000, "n_step": 3},
    ],
)
def test_configuration_rejects_invalid_values(overrides) -> None:
    with pytest.raises(ValueError):
        VectorTrainingConfig(**overrides)


def _tiny_config(**overrides) -> VectorTrainingConfig:
    settings: dict = {}
    settings.update(
        total_frames=2400,
        n_envs=4,
        action_repeat=4,
        batch_size=16,
        replay_capacity=512,
        min_replay=64,
        target_update_frames=400,
        epsilon_decay_frames=1000,
        hidden_layers=(16, 8),
        opponent_mix={"dummy": 1.0, "self": 1.0, "planner": 0.5, "tracker": 1.0},
        snapshot_interval_frames=800,
        snapshot_pool=2,
        eval_interval_frames=100_000,
        eval_seeds=1,
        eval_max_frames=120,
        eval_opponents=("dummy",),
        checkpoint_interval_frames=1200,
        seed=3,
        device="cpu",
        planner_horizon=12,
        planner_action_steps=6,
        match_target=2,
        max_point_frames=300,
    )
    settings.update(overrides)
    return VectorTrainingConfig(**settings)


def test_tiny_run_trains_saves_and_loads_into_the_exact_harness(tmp_path) -> None:
    trainer = VectorDQNTrainer(_tiny_config(shaping_scale=0.1, hit_reward=0.05), tmp_path)
    initial = [p.detach().clone() for p in trainer.network.parameters()]
    summary = trainer.run(verbose=False)
    assert summary["frames"] >= 2400
    assert summary["updates"] > 0
    assert summary["snapshots"] == 3  # initial network plus two snapshots (pool of two)
    assert any(
        not torch.equal(a, b.detach()) for a, b in zip(initial, trainer.network.parameters())
    )
    for name in ("latest.pth", "final.pth", "best.pth", "training.json"):
        assert (tmp_path / name).exists()
    assert list(tmp_path.glob("evaluation_*.json"))
    agent = DQNAgent()
    agent.load_model(str(tmp_path / "final.pth"))
    assert agent.hidden_layers == [16, 8]
    result = evaluate_competitive_agent(
        agent, CompetitiveConfig(opponents=("dummy",), seeds=(101,), max_steps=60, max_score=1)
    )
    assert len(result.matches) == 4


def test_snapshot_pool_keeps_the_initial_network_and_recent_copies(tmp_path) -> None:
    trainer = VectorDQNTrainer(_tiny_config(snapshot_pool=2), tmp_path)
    first = [p.detach().clone() for p in trainer.snapshots[0].parameters()]
    for _ in range(5):
        with torch.no_grad():
            for p in trainer.network.parameters():
                p.add_(0.01)
        trainer.add_snapshot()
    assert len(trainer.snapshots) == 3
    assert all(torch.equal(a, b) for a, b in zip(first, trainer.snapshots[0].parameters()))
    assert trainer.env.snapshot_count == 3


def test_evaluate_network_reports_every_cell_and_a_rank() -> None:
    network = torch.nn.Sequential(torch.nn.Linear(30, 9))
    report = evaluate_network(
        network,
        torch.device("cpu"),
        opponents=("dummy", "tracker"),
        seeds=2,
        match_target=1,
        max_frames=200,
    )
    assert report["matches"] == 16
    assert report["wins"] + report["losses"] + report["draws"] == 16
    assert len(report["cells"]) == 8 and len(report["rank"]) == 4
    assert set(report["by_opponent"]) == {"dummy", "tracker"}


def test_cli_runs_a_tiny_configuration(tmp_path) -> None:
    code = main(
        [
            "--output",
            str(tmp_path / "run"),
            "--total_frames",
            "800",
            "--n_envs",
            "2",
            "--batch_size",
            "8",
            "--replay_capacity",
            "256",
            "--min_replay",
            "32",
            "--hidden_layers",
            "8,4",
            "--opponent_mix",
            "dummy:1",
            "--eval_seeds",
            "1",
            "--eval_max_frames",
            "60",
            "--eval_interval_frames",
            "100000",
            "--snapshot_interval_frames",
            "400",
            "--checkpoint_interval_frames",
            "400",
            "--device",
            "cpu",
            "--match_target",
            "1",
            "--log_interval_frames",
            "400",
        ]
    )
    assert code == 0
    assert (tmp_path / "run" / "final.pth").exists()
