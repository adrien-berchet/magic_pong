"""Persistent opponent coverage and frozen self-play opponents."""

import random
from pathlib import Path

import pytest

from magic_pong.ai.league import OpponentLeague
from magic_pong.ai.models.simple_ai import BUILTIN_OPPONENTS
from magic_pong.ai.models.simple_ai import TrainingDummyAI


class SnapshotAgent:
    def __init__(self, contents="weights"):
        self.contents = contents
        self.training_enabled = True
        self.epsilon = 1.0

    def save_model(self, path):
        Path(path).write_text(self.contents)

    def set_training_mode(self, enabled):
        self.training_enabled = enabled


@pytest.fixture
def factory():
    calls = []

    def build(name):
        calls.append(name)
        return SnapshotAgent()

    build.calls = calls
    return build


@pytest.fixture
def league(tmp_path, factory):
    return OpponentLeague(tmp_path / "league", seed=17, opponent_factory=factory)


def test_all_builtin_opponents_keep_coverage_and_weak_matchups_get_more_training(league):
    assert set(league.probabilities()) == set(BUILTIN_OPPONENTS)
    for _ in range(20):
        league.record_outcome("follow_ball", won=True)
        league.record_outcome("aggressive", won=False)
    probabilities = league.probabilities()
    assert probabilities["aggressive"] > probabilities["follow_ball"]
    assert min(probabilities.values()) >= league.coverage_floor / len(probabilities)
    assert sum(probabilities.values()) == pytest.approx(1.0)


def test_timeout_is_nonwin_even_when_current_score_leads(league):
    league.record_outcome("aggressive", won=True, timeout=True)
    entry = league.entries["aggressive"]
    assert (entry.games, entry.wins, entry.timeouts) == (1, 0, 1)


def test_resume_restores_rng_statistics_and_avoids_overwriting_snapshots(league, factory):
    first = league.add_snapshot(SnapshotAgent("first"), champion=True)
    first_path = Path(league.entries[first].path)
    league.record_outcome("aggressive", won=False)
    for _ in range(11):
        league.next_opponent()
    resumed = OpponentLeague(league.snapshot_dir, seed=999, opponent_factory=factory)
    expected = [league.next_opponent()[0] for _ in range(12)]
    actual = [resumed.next_opponent()[0] for _ in range(12)]
    assert actual == expected
    assert resumed.entries["aggressive"].games == 1
    second = resumed.add_snapshot(SnapshotAgent("second"))
    assert first != second
    assert first_path.read_text() == "first"


def test_pool_retains_champion_old_diversity_and_pinned_references(tmp_path, factory):
    league = OpponentLeague(
        tmp_path,
        recent_size=2,
        champion_size=1,
        historical_size=2,
        opponent_factory=factory,
    )
    oldest = league.add_snapshot(SnapshotAgent("old"))
    champion = league.add_snapshot(SnapshotAgent("champion"), champion=True)
    league.protect(champion)
    for i in range(12):
        league.add_snapshot(SnapshotAgent(str(i)))
    snapshots = [entry for entry in league.entries.values() if entry.kind == "snapshot"]
    assert len(snapshots) == 5
    assert champion in league.entries
    assert oldest in league.entries
    assert len(list(tmp_path.glob("snapshot_*.pth"))) == 5


def test_external_checkpoint_is_immutable_and_source_never_deleted(tmp_path, factory):
    source = tmp_path / "independent.pth"
    source.write_text("independent-v1")
    league = OpponentLeague(
        tmp_path / "league",
        recent_size=1,
        champion_size=0,
        historical_size=0,
        opponent_factory=factory,
    )
    imported = league.add_checkpoint(source, role="exploiter")
    source.write_text("changed outside league")
    for i in range(5):
        league.add_snapshot(SnapshotAgent(str(i)))
    assert Path(league.entries[imported].path).read_text() == "independent-v1"
    assert source.read_text() == "changed outside league"
    assert league.entries[imported].role == "exploiter"
    league.cleanup()
    assert Path(league.entries[imported].path).exists()


def test_neural_opponents_are_cached_and_frozen(league, factory):
    snapshot = league.add_snapshot(SnapshotAgent())
    _, first = league.next_opponent(only_ids=(snapshot,))
    _, second = league.next_opponent(only_ids=(snapshot,))
    assert first is second
    assert not first.training_enabled
    assert first.epsilon == 0
    assert factory.calls == [f"dqn:{league.entries[snapshot].path}"]


def test_fixed_benchmark_factories_preserve_their_opponents(league):
    initial = league.add_snapshot(SnapshotAgent("initial"), champion=True)
    factories = league.benchmark_factories((initial,))
    opponent = factories[initial]()
    for index in range(20):
        league.add_snapshot(SnapshotAgent(str(index)), champion=True)
    assert set(factories) == {initial}
    assert factories[initial]() is opponent
    assert Path(league.entries[initial].path).exists()


def test_independent_seeded_leagues_sample_same_sequence(tmp_path, factory):
    first = OpponentLeague(tmp_path / "one", rng=random.Random(30), opponent_factory=factory)
    second = OpponentLeague(tmp_path / "two", rng=random.Random(30), opponent_factory=factory)
    assert [first.next_opponent()[0] for _ in range(30)] == [
        second.next_opponent()[0] for _ in range(30)
    ]


def test_training_dummy_uses_simulated_time(monkeypatch):
    import time

    observation = {
        "player_pos": (200, 300),
        "ball_pos": (400, 300),
        "ball_velocity": (1, 1),
        "opponent_pos": (600, 300),
        "field_size": (800, 600),
        "time_elapsed": 3.0,
    }
    dummy = TrainingDummyAI()
    monkeypatch.setattr(time, "time", lambda: 0)
    first = dummy.get_action(observation)
    monkeypatch.setattr(time, "time", lambda: 100)
    assert dummy.get_action(observation) == first
    assert dummy.get_action({**observation, "time_elapsed": 8.0}) != first


def test_default_loader_disables_network_training_and_replay(tmp_path):
    import torch

    from magic_pong.ai.models.dqn_ai import DQNAgent

    trainee = DQNAgent(hidden_size=8, layer_size=1, batch_size=1, min_replay_size=1)
    league = OpponentLeague(tmp_path)
    entry_id = league.add_snapshot(trainee)
    _, frozen = league.next_opponent(only_ids=(entry_id,))
    before = [parameter.detach().clone() for parameter in frozen.q_network.parameters()]
    observation = {
        "player_pos": (200, 300),
        "ball_pos": (400, 300),
        "ball_vel": (100, 100),
        "opponent_pos": (600, 300),
        "opponent_previous_pos": (600, 300),
        "field_width": 800,
        "field_height": 600,
    }
    frozen.on_episode_start()
    for _ in range(3):
        action = frozen.get_action(observation)
        frozen.on_step(observation, action, 1.0, False, {})
    assert len(frozen.memory) == 0
    assert not frozen.q_network.training
    assert all(not parameter.requires_grad for parameter in frozen.q_network.parameters())
    assert all(torch.equal(a, b) for a, b in zip(before, frozen.q_network.parameters()))


def test_new_candidate_survives_when_references_fill_all_other_slots(tmp_path, factory):
    league = OpponentLeague(
        tmp_path,
        recent_size=3,
        champion_size=0,
        historical_size=0,
        opponent_factory=factory,
    )
    first = league.add_snapshot(SnapshotAgent("first"))
    second = league.add_snapshot(SnapshotAgent("second"))
    league.benchmark_factories((first, second))
    for index in range(5):
        candidate = league.add_snapshot(SnapshotAgent(str(index)))
        assert candidate in league.entries
    assert first in league.entries and second in league.entries


def test_modified_external_source_creates_new_immutable_opponent(tmp_path, factory):
    source = tmp_path / "model.pth"
    source.write_text("first")
    league = OpponentLeague(tmp_path / "league", opponent_factory=factory)
    first = league.add_checkpoint(source)
    assert league.add_checkpoint(source) == first
    source.write_text("second")
    second = league.add_checkpoint(source)
    assert second != first
    assert Path(league.entries[first].path).read_text() == "first"
    assert Path(league.entries[second].path).read_text() == "second"


def test_default_benchmark_reserves_room_for_new_champions_and_recent_policies(tmp_path, factory):
    league = OpponentLeague(
        tmp_path,
        recent_size=2,
        champion_size=2,
        historical_size=2,
        opponent_factory=factory,
    )
    for index in range(8):
        league.add_snapshot(SnapshotAgent(str(index)), champion=index % 2 == 0)
    references = league.snapshot_references()
    assert len(references) <= 2
    league.benchmark_factories()
    champion = league.add_snapshot(SnapshotAgent("new champion"))
    league.promote(champion)
    for index in range(6):
        league.add_snapshot(SnapshotAgent(f"recent {index}"))
    assert champion in league.entries
    assert all(reference in league.entries for reference in references)


def test_orphan_snapshot_is_never_overwritten_and_failed_save_is_removed(tmp_path, factory):
    (tmp_path / "snapshot_000001.pth").write_text("orphan")
    league = OpponentLeague(tmp_path, opponent_factory=factory)

    class FailingAgent:
        def save_model(self, path):
            Path(path).write_text("incomplete")
            raise OSError("disk failure")

    with pytest.raises(OSError, match="disk failure"):
        league.add_snapshot(FailingAgent())
    assert (tmp_path / "snapshot_000001.pth").read_text() == "orphan"
    assert not (tmp_path / "snapshot_000002.pth").exists()
    saved = league.add_snapshot(SnapshotAgent("complete"))
    assert Path(league.entries[saved].path).read_text() == "complete"


def test_missing_checkpoint_fails_resume_without_erasing_manifest(league, factory):
    snapshot = league.add_snapshot(SnapshotAgent())
    manifest = league.manifest_path.read_text()
    Path(league.entries[snapshot].path).unlink()
    with pytest.raises(FileNotFoundError, match="checkpoint missing"):
        OpponentLeague(league.snapshot_dir, opponent_factory=factory)
    assert league.manifest_path.read_text() == manifest


def test_pin_limit_is_atomic_and_factory_supports_builtins(tmp_path, factory):
    league = OpponentLeague(
        tmp_path,
        recent_size=2,
        champion_size=0,
        historical_size=0,
        opponent_factory=factory,
    )
    first = league.add_snapshot(SnapshotAgent("first"))
    second = league.add_snapshot(SnapshotAgent("second"))
    with pytest.raises(ValueError, match="leave one slot"):
        league.benchmark_factories((first, second))
    assert not any(entry.protected for entry in league.entries.values())
    benchmark = league.benchmark_factory((first,))
    assert benchmark(first, name="ignored") is benchmark(first)
    assert benchmark("aggressive") is not None


def test_pool_retains_distinct_performance_profiles(tmp_path, factory):
    league = OpponentLeague(
        tmp_path,
        recent_size=1,
        champion_size=0,
        historical_size=1,
        opponent_factory=factory,
    )
    distinct = league.add_snapshot(
        SnapshotAgent("distinct"),
        metadata={"performance_profile": {"aggressive": 1.0}},
    )
    for index in range(5):
        league.add_snapshot(
            SnapshotAgent(str(index)),
            metadata={"performance_profile": {"aggressive": 0.1}},
        )
    assert distinct in league.entries


def test_benchmark_identity_includes_contents_across_league_directories(tmp_path, factory):
    first = OpponentLeague(tmp_path / "one", opponent_factory=factory)
    second = OpponentLeague(tmp_path / "two", opponent_factory=factory)
    first_id = first.add_snapshot(SnapshotAgent("first weights"))
    second_id = second.add_snapshot(SnapshotAgent("different weights"))
    assert first_id == second_id
    assert first.benchmark_identity(first_id) != second.benchmark_identity(second_id)
    factories = first.benchmark_factories((first_id,), content_ids=True)
    assert tuple(factories) == (first.benchmark_identity(first_id),)
    assert next(iter(factories.values()))().training_enabled is False


def test_modified_checkpoint_cannot_silently_change_resumed_benchmark(league, factory):
    snapshot = league.add_snapshot(SnapshotAgent("original"))
    league.benchmark_factories((snapshot,))
    Path(league.entries[snapshot].path).write_text("replacement")
    with pytest.raises(ValueError, match="contents changed"):
        OpponentLeague(league.snapshot_dir, opponent_factory=factory)


def test_opponent_loading_and_snapshot_saving_preserve_training_rngs(tmp_path):
    import numpy as np
    import torch

    def consume_randomness():
        random.random()
        np.random.random()
        torch.rand(1)

    class NoisySnapshot(SnapshotAgent):
        def save_model(self, path):
            consume_randomness()
            super().save_model(path)

    def noisy_factory(name):
        consume_randomness()
        return SnapshotAgent()

    league = OpponentLeague(tmp_path, opponent_factory=noisy_factory)
    python_state = random.getstate()
    numpy_state = np.random.get_state()
    torch_state = torch.get_rng_state().clone()
    snapshot = league.add_snapshot(NoisySnapshot())
    league.next_opponent(only_ids=(snapshot,))
    league.next_opponent(only_ids=(snapshot,))
    assert random.getstate() == python_state
    assert np.array_equal(np.random.get_state()[1], numpy_state[1])
    assert np.random.get_state()[2:] == numpy_state[2:]
    assert torch.equal(torch.get_rng_state(), torch_state)
