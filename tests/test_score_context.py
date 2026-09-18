"""Score-context observations, loss-preserving migration and checkpoint contracts."""

import copy
import io

import numpy as np
import pytest
import torch

from magic_pong.ai.agent_adapter import to_player1_dqn_observation
from magic_pong.ai.interface import ObservationProcessor
from magic_pong.ai.models.dqn_ai import DQNAgent
from magic_pong.ai.models.dqn_checkpoint import DQNCheckpointError
from magic_pong.ai.models.dqn_checkpoint import validate_dqn_checkpoint
from magic_pong.core.physics import PhysicsEngine
from magic_pong.utils.config import ai_config_tmp
from magic_pong.utils.config import game_config_tmp


def observation(side=1, scores=(5, 7), target=11):
    with game_config_tmp(MAX_SCORE=target):
        engine = PhysicsEngine(800, 600)
        engine.score[:] = scores
        state = engine.get_game_state()
    # Processing later must use the snapshotted target, not the live global.
    with game_config_tmp(MAX_SCORE=99):
        obs = ObservationProcessor(engine.field_width, engine.field_height).process_game_state(
            state, side
        )
    return to_player1_dqn_observation(obs, side)


@pytest.mark.parametrize("side", [1, 2])
@pytest.mark.parametrize("normalized", [False, True])
def test_score_context_is_side_relative_and_snapshot_owned(side, normalized):
    with ai_config_tmp(NORMALIZE_POSITIONS=normalized):
        obs = observation(side)
    assert obs["own_score"] == (5 if side == 1 else 7)
    assert obs["opponent_score"] == (7 if side == 1 else 5)
    assert obs["match_target"] == 11
    old = DQNAgent(hidden_size=8, layer_size=1)
    new = DQNAgent(hidden_size=8, layer_size=1, score_context=True)
    assert old._observation_to_state(obs).shape == (30,)
    encoded = new._observation_to_state(obs)
    np.testing.assert_array_equal(encoded[:30], old._observation_to_state(obs))
    np.testing.assert_array_equal(
        encoded[30:],
        np.array([obs["own_score"] / 11, obs["opponent_score"] / 11, 1], dtype=np.float32),
    )


def test_zero_projection_preserves_rng_and_exact_initial_q_values():
    torch.manual_seed(456)
    old = DQNAgent(hidden_layers=[32, 16])
    rng = torch.get_rng_state()
    torch.manual_seed(456)
    new = DQNAgent(hidden_layers=[32, 16], score_context=True)
    assert torch.equal(torch.get_rng_state(), rng)
    obs = [
        observation(side, scores, target)
        for side in (1, 2)
        for scores in ((0, 0), (1, 0), (0, 1), (5, 7))
        for target in (1, 3, 5, 11)
    ]
    old_x = torch.tensor(np.array([old._observation_to_state(o) for o in obs]))
    new_x = torch.tensor(np.array([new._observation_to_state(o) for o in obs]))
    old.q_network.cpu()
    new.q_network.cpu()
    assert torch.equal(old.q_network(old_x), new.q_network(new_x))
    for a, b in zip(old_x, new_x):
        assert torch.equal(old.q_network(a), new.q_network(b))
    # The added inputs must actually influence predictions once learned.
    new.q_network.score_context_weight.data.fill_(0.01)
    assert not torch.equal(old.q_network(old_x), new.q_network(new_x))


def test_context_checkpoint_roundtrip_and_explicit_legacy_upgrade(tmp_path):
    old = DQNAgent(hidden_layers=[16, 8])
    path = tmp_path / "legacy.pth"
    old.save_model(path)
    migrated = DQNAgent()
    migrated.load_model(path)
    before = {k: v.clone() for k, v in migrated.q_network.state_dict().items()}
    migrated.enable_score_context()
    for k, v in before.items():
        assert torch.equal(v, migrated.q_network.state_dict()[k])
    assert torch.count_nonzero(migrated.q_network.score_context_weight) == 0
    # Populate Adam state, then verify all networks and optimizer survive loading.
    x = torch.tensor(
        migrated._observation_to_state(observation()),
        device=next(migrated.q_network.parameters()).device,
    )
    migrated.q_network(x).square().sum().backward()
    migrated.optimizer.step()
    migrated.save_model(tmp_path / "context.pth")
    restored = DQNAgent()
    restored.load_model(tmp_path / "context.pth")
    assert restored.score_context and restored.state_size == 33
    assert torch.equal(restored.q_network(x), migrated.q_network(x))
    assert len(restored.optimizer.state) == len(migrated.optimizer.state)
    restored.save_model(tmp_path / "roundtrip.pth")
    from tests.test_scheduled_training import assert_nested_equal
    from tests.test_scheduled_training import payload

    assert_nested_equal(payload(tmp_path / "context.pth"), payload(tmp_path / "roundtrip.pth"))
    # Old loading still rebuilds a legacy network, including after a context model.
    restored.load_model(path)
    assert not restored.score_context and restored.state_size == 30


@pytest.mark.parametrize("field", ["own_score", "opponent_score", "match_target"])
def test_missing_context_fails_loudly_only_for_context_schema(field):
    obs = observation()
    del obs[field]
    DQNAgent()._observation_to_state(obs)
    with pytest.raises(ValueError, match="score context"):
        DQNAgent(score_context=True)._observation_to_state(obs)


@pytest.mark.parametrize("change", ["schema", "size", "flag", "projection", "normalizer"])
def test_context_checkpoint_rejects_mismatched_contract(change):
    stream = io.BytesIO()
    DQNAgent(score_context=True).save_model(stream)
    stream.seek(0)
    checkpoint = torch.load(stream, weights_only=True, map_location="cpu")
    validate_dqn_checkpoint(checkpoint)
    bad = copy.deepcopy(checkpoint)
    if change == "schema":
        bad["metadata"]["state_schema"] = "magic_pong.dqn_state.v4"
    elif change == "size":
        bad["hyperparameters"]["state_size"] = 30
    elif change == "flag":
        bad["hyperparameters"]["score_context"] = False
    elif change == "projection":
        bad["q_network_state_dict"]["score_context_weight"] = torch.zeros(256, 2)
    else:
        bad["metadata"]["score_context_normalizer"] = 1
    with pytest.raises(DQNCheckpointError):
        validate_dqn_checkpoint(bad)


@pytest.mark.parametrize("side", [1, 2])
def test_context_distinguishes_aliased_scores_and_target_and_keeps_match_return(side):
    from magic_pong.ai.models.simple_ai import FollowBallAI
    from magic_pong.core.game_engine import GameEngine
    from magic_pong.utils.config import game_config

    legacy = DQNAgent()
    context = DQNAgent(score_context=True, epsilon=0.0)
    examples = [
        observation(side, (0, 0), 1),
        observation(side, (0, 0), 11),
        observation(side, (10, 10), 11),
    ]
    # Independently constructed balls have random serves: replace all geometry with
    # one shared observation so only the omitted match context differs.
    for o in examples[1:]:
        for k in examples[0]:
            if k not in ("own_score", "opponent_score", "match_target"):
                o[k] = copy.deepcopy(examples[0][k])
    old = [legacy._observation_to_state(o) for o in examples]
    new = [context._observation_to_state(o) for o in examples]
    assert all(np.array_equal(old[0], x) for x in old)
    assert len({x.tobytes() for x in new}) == 3
    with game_config_tmp(MAX_SCORE=3):
        engine = GameEngine(headless=True)
        opponent = FollowBallAI()
        engine.set_players(*((context, opponent) if side == 1 else (opponent, context)))
        engine.start_game()
        try:
            for score in (1, 2, 3):
                ball = engine.physics_engine.ball
                ball.position.x = engine.physics_engine.field_width + 100
                ball.position.y = engine.physics_engine.field_height / 2
                ball.velocity.x = game_config.BALL_SPEED
                ball.velocity.y = 0
                result = engine.update(dt=1 / 60)
                t = context.memory.buffer[-1]
                assert t.done is (score == 3)
                assert result["info"]["terminated"] is (score == 3)
                expected = [score, 0, 3] if side == 1 else [0, score, 3]
                np.testing.assert_allclose(t.next_state[30:], np.array(expected) / 11, rtol=1e-7)
        finally:
            engine.stop_game()


def test_migration_rejects_nonempty_replay():
    agent = DQNAgent()
    agent.remember(np.zeros(30), 0, 0, np.zeros(30), False)
    with pytest.raises(ValueError, match="empty replay"):
        agent.enable_score_context()


def test_new_checkpoint_loads_as_gui_candidate_and_league_opponent(tmp_path):
    from magic_pong.ai.competitive import CompetitiveConfig
    from magic_pong.ai.competitive import evaluate_competitive_checkpoint
    from magic_pong.ai.league import OpponentLeague
    from magic_pong.gui.game_app import MagicPongApp

    path = tmp_path / "context.pth"
    DQNAgent(score_context=True, hidden_layers=[8]).save_model(path)
    # The GUI uses the same validation helper before creating the player.
    app = MagicPongApp.__new__(MagicPongApp)
    info = app._load_model_info(str(path))
    assert info["valid"] is True
    app.model_info_by_path = {str(path): info}
    playable = app._create_trained_ai(str(path))
    assert playable is not None and playable.score_context and not playable.training_enabled
    league = OpponentLeague(tmp_path / "league")
    new_id = league.add_checkpoint(path)
    _, frozen = league.next_opponent(only_ids=(new_id,))
    assert frozen.score_context and not frozen.training_enabled
    old = DQNAgent(hidden_layers=[8])
    old_id = league.add_snapshot(old)
    factories = league.benchmark_factories([old_id], content_ids=True)
    result = evaluate_competitive_checkpoint(
        str(path),
        CompetitiveConfig(opponents=tuple(factories), seeds=(101,), max_steps=2, max_score=1),
        opponent_factories=factories,
    )
    from magic_pong.ai.competitive import competitive_result_to_dict

    assert competitive_result_to_dict(result)["total_matches"] == 4


def test_context_scoring_resume_and_replay_audit(tmp_path, monkeypatch):
    import magic_pong.ai.scheduled_training as scheduled
    from docs.ai.run_score_context_experiment import inspect_context
    from magic_pong.utils.config import game_config
    from tests.test_scheduled_training import assert_nested_equal
    from tests.test_scheduled_training import initial_model
    from tests.test_scheduled_training import payload
    from tests.test_scheduled_training import small_config

    class ScoringEngine(scheduled.GameEngine):
        def update(self, *args, **kwargs):
            ball = self.physics_engine.ball
            ball.position.x = self.physics_engine.field_width + 100
            ball.position.y = self.physics_engine.field_height / 2
            ball.velocity.x = game_config.BALL_SPEED
            ball.velocity.y = 0
            return super().update(*args, **kwargs)

    monkeypatch.setattr(scheduled, "GameEngine", ScoringEngine)
    config = small_config(
        score_context=True, training_scores=(1, 3, 5, 11), selfplay_opponent="follow_ball"
    )
    initial = initial_model(tmp_path)
    full = scheduled.run_frame_pipeline(config, tmp_path / "full", initial_checkpoint=initial)
    scheduled.run_frame_pipeline(
        config, tmp_path / "split", initial_checkpoint=initial, stop_after_episodes=3
    )
    resumed = scheduled.run_frame_pipeline(
        config, tmp_path / "split", resume=tmp_path / "split/latest.training.pt"
    )
    assert full["episodes"] == resumed["episodes"]
    for key in ("replay", "rng"):
        assert_nested_equal(
            payload(tmp_path / "full/latest.training.pt")[key],
            payload(tmp_path / "split/latest.training.pt")[key],
        )
    assert_nested_equal(payload(tmp_path / "full/final.pth"), payload(tmp_path / "split/final.pth"))
    audit = inspect_context(full, tmp_path / "full")
    assert audit["state_size"] == 33 and audit["score_vectors_verified"] == 16
    bundle = payload(tmp_path / "full/latest.training.pt")
    bundle["replay"]["next_states"][-1, -1] += 0.01
    torch.save(bundle, tmp_path / "full/latest.training.pt")
    with pytest.raises(ValueError, match="physical goal events"):
        inspect_context(full, tmp_path / "full")


def test_actual_warmstart_migration_matches_fixed_probe_q_values(tmp_path):
    from docs.ai.run_score_context_experiment import initial_equivalence
    from tests.test_scheduled_training import initial_model

    probes = tmp_path / "probes.npz"
    np.savez(probes, states=np.random.default_rng(123).normal(size=(3351, 30)).astype(np.float32))
    result = initial_equivalence(11, initial_model(tmp_path), probes)
    assert result["q_vectors_checked"] == 30159
    assert result["maximum_absolute_q_difference"] == 0
