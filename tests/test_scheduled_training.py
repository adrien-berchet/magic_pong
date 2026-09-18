import io
from dataclasses import replace

import pytest
import torch

from magic_pong.ai.models.dqn_ai import DQNAgent
from magic_pong.ai.scheduled_training import FramePipelineConfig
from magic_pong.ai.scheduled_training import run_frame_pipeline


def small_config(**overrides):
    defaults = {
        "phase_frames": (24, 40, 32),
        "memory_size": 8,
        "batch_size": 4,
        "min_replay_size": 4,
        "train_frequency": 2,
        "target_update_frequency": 3,
        "max_episode_steps": 8,
        "validation_interval": 24,
        "snapshot_interval": 16,
        "checkpoint_interval": 16,
        "training_scores": (1,),
        "validation_opponents": ("dummy",),
        "validation_seeds": (101,),
        "validation_max_steps": 8,
        "validation_max_score": 1,
        "regression_patience": 0,
    }
    return FramePipelineConfig(**(defaults | overrides))


def initial_model(tmp_path):
    path = tmp_path / "initial.pth"
    torch.manual_seed(7)
    DQNAgent(hidden_size=8, layer_size=1).save_model(str(path))
    return path


def payload(path):
    return torch.load(path, map_location="cpu", weights_only=True)


def assert_nested_equal(a, b):
    if isinstance(a, torch.Tensor):
        assert torch.equal(a, b)
    elif isinstance(a, dict):
        assert a.keys() == b.keys()
        for key in a:
            assert_nested_equal(a[key], b[key])
    elif isinstance(a, (tuple, list)):
        assert len(a) == len(b)
        for x, y in zip(a, b):
            assert_nested_equal(x, y)
    else:
        assert a == b


def test_frame_schedules_reach_endpoints_and_flush_shaped_replay():
    config = small_config()
    assert config.epsilon(0, 0) == pytest.approx(0.3)
    assert config.epsilon(0, 12) == pytest.approx(0.175)
    assert config.epsilon(0, 24) == pytest.approx(0.1)
    assert config.epsilon(1, 0) == pytest.approx(0.1)
    assert config.epsilon(2, 0) == pytest.approx(0.2)
    assert config.epsilon(2, 32) == pytest.approx(0.05)
    assert config.shaping(0, 0) == 1
    assert config.shaping(1, 0) == 1
    assert config.shaping(1, 16) == pytest.approx(0.5)
    assert config.shaping(1, 32) == 0
    assert config.shaping(2, 0) == 0
    assert config.zero_reward_tail >= config.memory_size


def test_retained_shaping_reaches_floor_and_requires_full_steady_replay_tail():
    config = small_config(shaping_floor=0.1)
    assert config.shaping(0, 24) == 1
    assert config.shaping(1, 0) == 1
    assert config.shaping(1, 16) == pytest.approx(0.55)
    assert config.shaping(1, 32) == 0.1
    assert config.shaping(1, 40) == 0.1
    assert config.shaping(2, 0) == config.shaping(2, 32) == 0.1
    assert config.zero_reward_tail == 0
    assert config.steady_reward_tail_frames == config.memory_size
    with pytest.raises(ValueError, match="tail"):
        small_config(shaping_floor=0.1, phase_frames=(24, 36, 32))


def test_zero_floor_preserves_previous_shaping_exactly():
    for legacy in (False, True):
        config = small_config(legacy_control=legacy, shaping_floor=0.0)
        fraction = 1.0 if legacy else config.decay_fraction
        for frame in range(41):
            assert config.shaping(1, frame) == max(0.0, 1.0 - frame / (40 * fraction))


def test_reward_floor_scales_auxiliary_rewards_but_preserves_goals():
    from magic_pong.ai.interface import RewardCalculator
    from magic_pong.utils.config import ai_config_tmp

    config = small_config(shaping_floor=0.1)
    calculator = RewardCalculator()
    with ai_config_tmp(
        MICRO_REWARD_SCALE=config.shaping(2, 0),
        REWARD_SHAPING_MODE="legacy",
        USE_PROXIMITY_REWARD=False,
        PADDLE_HIT_REWARD=0.1,
        BONUS_REWARD=0.1,
        SCORE_REWARD=1.0,
        LOSE_PENALTY=-1.0,
    ):
        for event, expected in (
            ("paddle_hits", 0.01),
            ("bonus_collected", 0.01),
            ("rotating_paddle_hits", 0.02),
            ("goals", 1.0),
        ):
            assert calculator.calculate_reward({}, {event: [{"player": 1}]}, 1) == pytest.approx(
                expected
            )
        assert calculator.calculate_reward({}, {"goals": [{"player": 2}]}, 1) == -1


@pytest.mark.parametrize(
    "changes",
    [
        {"phase_frames": (24, 36, 32)},
        {"decay_fraction": 1.0},
        {"phase_frames": (0, 40, 32)},
        {"training_scores": ()},
        {"epsilon_starts": (0.3, float("nan"), 0.2)},
        {"checkpoint_interval": 0},
        {"shaping_floor": -0.1},
        {"shaping_floor": 1.1},
        {"shaping_floor": float("nan")},
        {"shaping_floor": float("inf")},
        {"selfplay_opponent": "unknown"},
    ],
)
def test_invalid_schedule_rejected(changes):
    with pytest.raises(ValueError):
        small_config(**changes)


@pytest.mark.parametrize("pause_episode", [4, 10, 15])
@pytest.mark.parametrize(
    "shaping_floor,selfplay_opponent,training_scores,terminal_on_goal,score_context",
    [
        (0.0, "league", (1,), False, False),
        (0.1, "league", (1,), False, False),
        (0.1, "follow_ball", (1,), False, False),
        (0.1, "follow_ball", (1, 3, 5, 11), False, False),
        (0.1, "follow_ball", (1, 3, 5, 11), False, True),
        (0.1, "follow_ball", (1, 3, 5, 11), True, False),
    ],
)
def test_resume_matches_uninterrupted_training_and_keeps_completed_phase_state(
    tmp_path,
    pause_episode,
    shaping_floor,
    selfplay_opponent,
    training_scores,
    terminal_on_goal,
    score_context,
):
    initial = initial_model(tmp_path)
    config = small_config(
        shaping_floor=shaping_floor,
        selfplay_opponent=selfplay_opponent,
        training_scores=training_scores,
        terminal_on_goal=terminal_on_goal,
        score_context=score_context,
    )
    full = run_frame_pipeline(config, tmp_path / "full", initial_checkpoint=initial)
    paused = run_frame_pipeline(
        config, tmp_path / "split", initial_checkpoint=initial, stop_after_episodes=pause_episode
    )
    assert paused["status"] == "paused"
    resumed = run_frame_pipeline(
        config, tmp_path / "split", resume=tmp_path / "split/latest.training.pt"
    )
    assert resumed["status"] == full["status"] == "complete"
    a, b = (
        payload(tmp_path / "full/latest.training.pt"),
        payload(tmp_path / "split/latest.training.pt"),
    )
    assert a["progress"] == b["progress"]
    assert a["progress"]["total_frames"] == 96
    for key in ["replay", "rng"]:
        assert_nested_equal(a[key], b[key])
    assert a["league"]["manifest"]["rng_state"] == b["league"]["manifest"]["rng_state"]
    assert a["league"]["manifest"]["counter"] == b["league"]["manifest"]["counter"]
    assert a["league"]["files"].keys() == b["league"]["files"].keys()
    # PyTorch archive bytes can differ after loading identical state because
    # pickle storage/memo identities differ. Compare the actual saved state.
    for name in a["league"]["files"]:
        assert_nested_equal(
            payload(io.BytesIO(a["league"]["files"][name])),
            payload(io.BytesIO(b["league"]["files"][name])),
        )
    assert_nested_equal(payload(io.BytesIO(a["model"])), payload(io.BytesIO(b["model"])))
    assert [r["end_frames"] for r in a["phase_results"]] == [24, 64, 96]
    assert [r["agent_frames"] for r in a["phase_results"]] == [24, 64, 96]
    assert a["phase_results"][1]["zero_reward_frames"] == config.zero_reward_tail
    assert a["phase_results"][1]["steady_reward_tail_frames"] == config.memory_size
    assert a["phase_results"][1]["shaping_end"] == shaping_floor
    assert a["config"]["shaping_floor"] == shaping_floor
    assert a["config"]["selfplay_opponent"] == selfplay_opponent
    assert a["episodes"] == b["episodes"]
    assert a["config"]["training_scores"] == training_scores
    assert a["replay"]["states"].shape[1] == (33 if score_context else 30)
    assert payload(io.BytesIO(a["model"]))["hyperparameters"]["score_context"] is score_context


@pytest.mark.parametrize("terminal_on_goal", [False, True])
@pytest.mark.parametrize("episode_limit", [8, 12])
def test_mixed_score_targets_reach_engine_and_log_completion_or_frame_limit(
    tmp_path, monkeypatch, episode_limit, terminal_on_goal
):
    import magic_pong.ai.scheduled_training as scheduled
    from docs.ai.run_match_length_experiment import score_exposure
    from magic_pong.utils.config import game_config

    targets = []

    class ScoringEngine(scheduled.GameEngine):
        def __init__(self, *args, **kwargs):
            targets.append(game_config.MAX_SCORE)
            super().__init__(*args, **kwargs)

        def update(self, *args, **kwargs):
            # Exercise real scoring, game-over and truncation semantics quickly.
            ball = self.physics_engine.ball
            ball.position.x = self.physics_engine.field_width + 100
            ball.position.y = self.physics_engine.field_height / 2
            ball.velocity.x = game_config.BALL_SPEED
            ball.velocity.y = 0
            return super().update(*args, **kwargs)

    monkeypatch.setattr(scheduled, "GameEngine", ScoringEngine)
    config = small_config(
        phase_frames=(80, 80, 80),
        max_episode_steps=episode_limit,
        terminal_on_goal=terminal_on_goal,
        training_scores=(1, 3, 5, 11),
        selfplay_opponent="follow_ball",
    )
    result = run_frame_pipeline(
        config, tmp_path / "mixed", initial_checkpoint=initial_model(tmp_path)
    )
    episodes = result["episodes"]
    bundle = payload(tmp_path / "mixed/latest.training.pt")
    assert (
        payload(io.BytesIO(bundle["model"]))["hyperparameters"]["terminal_on_goal"]
        is terminal_on_goal
    )
    for e in episodes:
        assert e["episode_done"] is True
        assert e["terminated"] is bool(e["winner"])
        assert e["truncated"] is (not bool(e["winner"]))
        assert e["td_terminals"] == (e["frames"] if terminal_on_goal else int(bool(e["winner"])))
        assert len(e["goal_transitions"]) == e["frames"]
        assert sum(g["td_terminal"] for g in e["goal_transitions"]) == e["td_terminals"]
        assert e["nonterminal_goal_td_terminals"] == (
            e["frames"] - int(bool(e["winner"])) if terminal_on_goal else 0
        )
    assert [e["max_score"] for e in episodes] == targets
    assert set(targets) == {1, 3, 5, 11}
    assert any(e["winner"] and e["max_score"] == 11 for e in episodes) == (episode_limit >= 11)
    assert any(not e["winner"] for e in episodes)
    for episode in episodes:
        assert episode["frames"] <= episode["step_limit"] <= config.max_episode_steps
        if episode["winner"]:
            assert max(episode["goals_for"], episode["goals_against"]) == episode["max_score"]
        else:
            assert episode["frames"] == episode["step_limit"]
            assert max(episode["goals_for"], episode["goals_against"]) < episode["max_score"]
    from docs.ai.run_goal_bootstrap_experiment import inspect_masks

    audit = inspect_masks(result, tmp_path / "mixed", complete=True)
    assert audit["goal_transitions"] == 240
    assert audit["masked_nonterminal_goals"] == (
        audit["nonterminal_goals"] if terminal_on_goal else 0
    )
    if terminal_on_goal:
        assert audit["final_replay"]["maximum_absolute_goal_bootstrap"] == 0
    exposure = score_exposure(result)
    assert sum(group["frames"] for group in exposure.values()) == 240
    if episode_limit < 11:
        assert exposure["11"]["episode_cap"] > 0
    else:
        assert exposure["11"]["completed_by_score"] > 0
    assert sum(group["stratum_cap"] for group in exposure.values()) > 0
    episodes[0]["step_limit"] += 1
    with pytest.raises(ValueError, match="frame limit"):
        score_exposure(result)
    episodes[0]["step_limit"] -= 1
    episodes[0]["max_score"] = 99
    with pytest.raises(ValueError, match="score target"):
        score_exposure(result)


def test_fixed_final_opponent_preserves_earlier_phases_and_never_samples_league(
    tmp_path, monkeypatch
):
    from magic_pong.ai.league import OpponentLeague

    initial = initial_model(tmp_path)
    control = small_config(shaping_floor=0.1)
    run_frame_pipeline(control, tmp_path / "league", initial_checkpoint=initial)

    def unexpected_sample(self):
        pytest.fail("Fixed-opponent training must not sample league opponents")

    monkeypatch.setattr(OpponentLeague, "next_opponent", unexpected_sample)
    fixed = replace(control, selfplay_opponent="follow_ball")
    result = run_frame_pipeline(fixed, tmp_path / "fixed", initial_checkpoint=initial)
    assert result["status"] == "complete"
    final_episodes = [e for e in result["episodes"] if e["phase"] == "selfplay"]
    assert final_episodes and {e["opponent"] for e in final_episodes} == {"follow_ball"}
    assert sum(e["frames"] for e in final_episodes) == fixed.phase_frames[2]
    for name in ("initial.pth", "curriculum_final.pth", "anneal_final.pth"):
        assert_nested_equal(payload(tmp_path / "league" / name), payload(tmp_path / "fixed" / name))
    state = payload(tmp_path / "fixed/latest.training.pt")
    assert len(state["league"]["files"]) > 1  # snapshots still saved, never sampled
    assert [r["frames"] for r in result["evaluations"]] == [0, 24, 48, 64, 72, 96]


def test_resume_rejects_changed_config_and_does_not_reset_schedule(tmp_path):
    config = small_config()
    directory = tmp_path / "run"
    run_frame_pipeline(
        config, directory, initial_checkpoint=initial_model(tmp_path), stop_after_episodes=1
    )
    before = payload(directory / "latest.training.pt")
    with pytest.raises(ValueError, match="config"):
        run_frame_pipeline(
            replace(config, seed=999), directory, resume=directory / "latest.training.pt"
        )
    after = payload(directory / "latest.training.pt")
    assert after["progress"] == before["progress"]


def test_resume_rejects_changed_physics_fingerprint(tmp_path):
    config = small_config()
    directory = tmp_path / "run"
    run_frame_pipeline(
        config, directory, initial_checkpoint=initial_model(tmp_path), stop_after_episodes=1
    )
    state = payload(directory / "latest.training.pt")
    state["source_hashes"]["magic_pong/core/collision.py"] = "changed"
    tampered = tmp_path / "changed.training.pt"
    torch.save(state, tampered)
    with pytest.raises(ValueError, match="source|physics"):
        run_frame_pipeline(config, tmp_path / "restored", resume=tampered)


def test_old_config_defaults_floor_before_reporting_changed_source(tmp_path):
    config = small_config()
    directory = tmp_path / "run"
    run_frame_pipeline(
        config, directory, initial_checkpoint=initial_model(tmp_path), stop_after_episodes=1
    )
    state = payload(directory / "latest.training.pt")
    state["config"].pop("shaping_floor")
    state["config"].pop("selfplay_opponent")
    state["config"].pop("terminal_on_goal")
    state["config"].pop("score_context")
    state["progress"].pop("steady_reward_tail_frames")
    state["source_hashes"]["magic_pong/ai/scheduled_training.py"] = "old source"
    tampered = tmp_path / "old.training.pt"
    torch.save(state, tampered)
    with pytest.raises(ValueError, match="source"):
        run_frame_pipeline(config, tmp_path / "restore", resume=tampered)


def test_existing_output_requires_explicit_resume(tmp_path):
    config = small_config()
    directory = tmp_path / "run"
    run_frame_pipeline(
        config, directory, initial_checkpoint=initial_model(tmp_path), stop_after_episodes=1
    )
    with pytest.raises(FileExistsError):
        run_frame_pipeline(config, directory)


def test_bundle_epsilon_matches_frame_schedule_after_optimizer_update(tmp_path):
    config = small_config()
    directory = tmp_path / "run"
    run_frame_pipeline(
        config, directory, initial_checkpoint=initial_model(tmp_path), stop_after_episodes=1
    )
    state = payload(directory / "latest.training.pt")
    model = payload(io.BytesIO(state["model"]))
    assert model["training_step"] > 0
    assert model["hyperparameters"]["epsilon_decay"] == 1.0
    assert model["epsilon"] == config.epsilon(0, state["progress"]["phase_frames"])


def test_legacy_control_reproduces_rollback_without_changing_frame_budget(tmp_path):
    config = small_config(legacy_control=True)
    directory = tmp_path / "run"
    result = run_frame_pipeline(config, directory, initial_checkpoint=initial_model(tmp_path))
    assert result["progress"]["total_frames"] == 96
    assert result["phase_results"][1]["zero_reward_frames"] == 0
    assert result["phase_results"][2]["agent_frames"] < 96


def test_corrupt_league_snapshot_rejected_on_resume(tmp_path):
    config = small_config()
    directory = tmp_path / "run"
    run_frame_pipeline(
        config, directory, initial_checkpoint=initial_model(tmp_path), stop_after_episodes=1
    )
    state = payload(directory / "latest.training.pt")
    filename = next(iter(state["league"]["files"]))
    state["league"]["files"][filename] = b"bad checkpoint"
    checkpoint = tmp_path / "bad.training.pt"
    torch.save(state, checkpoint)
    with pytest.raises(ValueError, match="hash mismatch"):
        run_frame_pipeline(config, tmp_path / "restore", resume=checkpoint)


def test_frame_pipeline_cli_and_resume_use_saved_configuration(tmp_path):
    import train_optimized

    directory = tmp_path / "cli"
    flags = [
        "--mode",
        "frame_pipeline",
        "--output_dir",
        str(directory),
        "--load_model",
        str(initial_model(tmp_path)),
        "--pipeline_curriculum_frames",
        "24",
        "--pipeline_anneal_frames",
        "40",
        "--pipeline_selfplay_frames",
        "32",
        "--memory_size",
        "8",
        "--batch_size",
        "4",
        "--min_replay_size",
        "4",
        "--max_steps_per_episode",
        "8",
        "--validation_opponents",
        "dummy",
        "--validation_seeds",
        "101",
        "--validation_max_steps",
        "8",
        "--validation_max_score",
        "1",
        "--frame_validation_interval",
        "24",
        "--frame_regression_patience",
        "0",
        "--frame_score_context",
        "--frame_terminal_on_goal",
        "--frame_shaping_floor",
        "0.1",
        "--frame_selfplay_opponent",
        "follow_ball",
    ]
    result = train_optimized.main(flags)
    assert result["status"] == "complete"
    assert result["config"]["terminal_on_goal"] is True
    assert result["config"]["score_context"] is True
    assert result["config"]["shaping_floor"] == 0.1
    assert result["config"]["selfplay_opponent"] == "follow_ball"
    resumed = train_optimized.main(
        [
            "--mode",
            "frame_pipeline",
            "--output_dir",
            str(directory),
            "--resume_training",
            str(directory / "latest.training.pt"),
        ]
    )
    assert resumed["progress"] == result["progress"]
    inferred = train_optimized.main(
        [
            "--mode",
            "frame_pipeline",
            "--resume_training",
            str(directory / "latest.training.pt"),
        ]
    )
    assert inferred["progress"] == result["progress"]
    with pytest.raises(SystemExit):
        train_optimized.parse_arguments(
            [
                "--mode",
                "frame_pipeline",
                "--resume_training",
                str(directory / "latest.training.pt"),
                "--output_dir",
                str(tmp_path / "unrelated"),
            ]
        )
    assert not (tmp_path / "unrelated").exists()


def test_legacy_pipeline_handoff_keeps_live_learner(monkeypatch, tmp_path):
    import train_optimized

    class Agent:
        position = 0

        def load_model(self, path):
            raise AssertionError("phase handoff must not reload best playable weights")

    agent = Agent()

    def phase(agent, args):
        agent.position += 1
        return agent, {"total_episodes": 1}

    monkeypatch.setattr(train_optimized, "train_with_curriculum", phase)
    monkeypatch.setattr(train_optimized, "train_single_phase", phase)
    args = train_optimized.parse_arguments(["--mode", "pipeline", "--output_dir", str(tmp_path)])
    trained, _ = train_optimized.train_pipeline(agent, args)
    assert trained is agent and agent.position == 3


def test_regression_guard_stops_before_next_phase(monkeypatch, tmp_path):
    from magic_pong.ai import scheduled_training

    rates = iter([1.0, 0.0])
    monkeypatch.setattr(scheduled_training, "_builtin_rate", lambda *args: next(rates))
    config = small_config(regression_patience=1)
    result = run_frame_pipeline(
        config, tmp_path / "run", initial_checkpoint=initial_model(tmp_path)
    )
    assert result["status"] == "regressed"
    assert result["phase_results"] == []
    assert result["progress"]["phase"] == 0


def test_pilot_cli_matches_budgets_and_allows_resume_without_initial(monkeypatch, tmp_path):
    from magic_pong.ai import schedule_pilot

    calls = []

    def run(config, directory, **kwargs):
        calls.append((config, directory, kwargs))
        return {"status": "complete"}

    monkeypatch.setattr(schedule_pilot, "run_frame_pipeline", run)
    shared = ["--output-dir", str(tmp_path), "--seed", "11"]
    schedule_pilot.main(shared + ["--variant", "legacy", "--initial-checkpoint", "initial.pth"])
    schedule_pilot.main(shared + ["--variant", "corrected", "--resume"])
    assert calls[0][0].phase_frames == calls[1][0].phase_frames == (250000, 250000, 500000)
    assert calls[0][0].legacy_control and not calls[1][0].legacy_control
    assert calls[1][2]["initial_checkpoint"] is None
    assert calls[1][2]["resume"] == tmp_path / "seed_11/corrected/latest.training.pt"


def test_keyboard_interrupt_keeps_last_complete_boundary(monkeypatch, tmp_path):
    from magic_pong.ai.scheduled_training import GameEngine

    config = small_config()
    directory = tmp_path / "run"
    run_frame_pipeline(
        config, directory, initial_checkpoint=initial_model(tmp_path), stop_after_episodes=1
    )
    before = (directory / "latest.training.pt").read_bytes()
    update = GameEngine.update
    frames = 0

    def interrupted(self, *args, **kwargs):
        nonlocal frames
        frames += 1
        if frames == 2:
            raise KeyboardInterrupt
        return update(self, *args, **kwargs)

    monkeypatch.setattr(GameEngine, "update", interrupted)
    with pytest.raises(KeyboardInterrupt):
        run_frame_pipeline(config, directory, resume=directory / "latest.training.pt")
    assert frames == 2
    assert (directory / "latest.training.pt").read_bytes() == before


def test_resume_uses_bundled_league_even_when_live_league_changed(tmp_path):
    config = small_config()
    directory = tmp_path / "run"
    run_frame_pipeline(
        config, directory, initial_checkpoint=initial_model(tmp_path), stop_after_episodes=15
    )
    for p in (directory / "league").glob("*.pth"):
        p.unlink()
    result = run_frame_pipeline(config, directory, resume=directory / "latest.training.pt")
    assert result["status"] == "complete"


def test_goal_mask_one_point_equivalence_and_scoring_resume(tmp_path, monkeypatch):
    import magic_pong.ai.scheduled_training as scheduled
    from magic_pong.utils.config import game_config

    class ScoringEngine(scheduled.GameEngine):
        def update(self, *args, **kwargs):
            ball = self.physics_engine.ball
            ball.position.x = self.physics_engine.field_width + 100
            ball.position.y = self.physics_engine.field_height / 2
            ball.velocity.x = game_config.BALL_SPEED
            ball.velocity.y = 0
            return super().update(*args, **kwargs)

    monkeypatch.setattr(scheduled, "GameEngine", ScoringEngine)
    initial = initial_model(tmp_path)
    # A goal every frame: point-return must be numerically identical for first-to-one.
    config = small_config(selfplay_opponent="follow_ball", training_scores=(1,))
    a = run_frame_pipeline(config, tmp_path / "control", initial_checkpoint=initial)
    b = run_frame_pipeline(
        replace(config, terminal_on_goal=True), tmp_path / "treatment", initial_checkpoint=initial
    )
    assert a["episodes"] == b["episodes"]
    ca, cb = payload(tmp_path / "control/final.pth"), payload(tmp_path / "treatment/final.pth")
    ca["hyperparameters"].pop("terminal_on_goal")
    cb["hyperparameters"].pop("terminal_on_goal")
    assert_nested_equal(ca, cb)
    # Actual nonterminal goals survive pause/resume, including the replay masks.
    config = replace(config, training_scores=(3, 5, 11), terminal_on_goal=True)
    full = run_frame_pipeline(config, tmp_path / "full", initial_checkpoint=initial)
    run_frame_pipeline(
        config, tmp_path / "split", initial_checkpoint=initial, stop_after_episodes=3
    )
    resumed = run_frame_pipeline(
        config, tmp_path / "split", resume=tmp_path / "split/latest.training.pt"
    )
    assert full["episodes"] == resumed["episodes"]
    assert any(e["nonterminal_goal_td_terminals"] for e in full["episodes"])
    for key in ("replay", "rng"):
        assert_nested_equal(
            payload(tmp_path / "full/latest.training.pt")[key],
            payload(tmp_path / "split/latest.training.pt")[key],
        )
    assert_nested_equal(payload(tmp_path / "full/final.pth"), payload(tmp_path / "split/final.pth"))
