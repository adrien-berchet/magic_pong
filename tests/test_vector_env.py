"""The vectorised environment encodes exactly like the DQN pipeline and steps like the engine."""

import math
import random

import numpy as np
import pytest

pytest.importorskip("numba")

from magic_pong.ai import vector_env as ve  # noqa: E402
from magic_pong.ai.agent_adapter import to_player1_dqn_observation  # noqa: E402
from magic_pong.ai.fast_policies import NETWORK_OPPONENT  # noqa: E402
from magic_pong.ai.fast_policies import PLANNER_OPPONENT  # noqa: E402
from magic_pong.ai.fast_policies import POLICY_KINDS  # noqa: E402
from magic_pong.ai.interface import ObservationProcessor  # noqa: E402
from magic_pong.ai.models.dqn_ai import DQNAgent  # noqa: E402
from magic_pong.ai.models.simple_ai import create_ai  # noqa: E402
from magic_pong.core import fast_physics as fp  # noqa: E402
from magic_pong.core.entities import Ball  # noqa: E402
from magic_pong.core.entities import Bonus  # noqa: E402
from magic_pong.core.entities import BonusType  # noqa: E402
from magic_pong.core.entities import RotatingPaddle  # noqa: E402
from magic_pong.core.entities import Vector2D  # noqa: E402
from magic_pong.core.game_engine import GameEngine  # noqa: E402
from magic_pong.core.physics import PhysicsEngine  # noqa: E402
from magic_pong.utils.config import game_config  # noqa: E402


def _random_engine(rng: random.Random) -> PhysicsEngine:
    engine = PhysicsEngine(800, 600)
    for paddle in (engine.player1, engine.player2):
        paddle.position = Vector2D(
            rng.uniform(paddle.min_x, paddle.max_x), rng.uniform(paddle.min_y, paddle.max_y)
        )
        paddle.prev_position = paddle.position + Vector2D(rng.uniform(-8, 8), rng.uniform(-8, 8))
        if rng.random() < 0.5:
            paddle.apply_size_effect(rng.choice((0.6, 1.5)), rng.uniform(0.5, 5.0))
    speed = rng.uniform(150, 850)
    angle = rng.uniform(-math.pi, math.pi)
    engine.ball = Ball(
        rng.uniform(30, 770), rng.uniform(20, 580), speed * math.cos(angle), speed * math.sin(angle)
    )
    for _ in range(rng.randint(0, 4)):
        engine.bonuses.append(
            Bonus(rng.uniform(40, 740), rng.uniform(20, 560), rng.choice(list(BonusType)))
        )
    for _ in range(rng.randint(0, 3)):
        obstacle = RotatingPaddle(rng.uniform(100, 700), rng.uniform(60, 540), rng.choice((1, 2)))
        obstacle.angle = rng.uniform(-10, 10)
        engine.rotating_paddles.append(obstacle)
    engine.score = [rng.randint(0, 10), rng.randint(0, 10)]
    engine.game_time = rng.uniform(0, 400)
    return engine


@pytest.mark.parametrize("seed", range(15))
@pytest.mark.parametrize("score_context", [False, True])
@pytest.mark.parametrize("geometry_context", [False, True])
@pytest.mark.parametrize("normalized", [False, True])
def test_encoder_matches_the_python_dqn_pipeline(
    seed: int,
    score_context: bool,
    geometry_context: bool,
    normalized: bool,
    monkeypatch: pytest.MonkeyPatch,
) -> None:
    from magic_pong.utils.config import ai_config

    monkeypatch.setattr(ai_config, "NORMALIZE_POSITIONS", normalized)
    rng = random.Random(seed)
    engine = _random_engine(rng)
    side = rng.choice((1, 2))
    agent = DQNAgent(
        hidden_layers=[8], score_context=score_context, geometry_context=geometry_context
    )
    observation = ObservationProcessor(800, 600).process_game_state(engine.get_game_state(), side)
    expected = agent._observation_to_state(to_player1_dqn_observation(observation, side))
    state, bonuses, rotating, params = fp.snapshot(engine)
    out = np.zeros(agent.state_size, dtype=np.float32)
    ve.encode_observation(
        state, bonuses, rotating, params, side, score_context, geometry_context, out
    )
    np.testing.assert_allclose(out, expected, atol=1e-6)


def test_step_env_matches_the_engine_against_a_builtin(monkeypatch) -> None:
    monkeypatch.setattr(game_config, "BONUSES_ENABLED", False)
    random.seed(4)
    engine = GameEngine(headless=True)
    engine.set_players(create_ai("dummy"), create_ai("follow_ball"))
    engine.start_game()
    engine.physics_engine.ball = Ball(500.0, 200.0, 250.0, 120.0)
    state, bonuses, rotating, params = fp.snapshot(engine.physics_engine)
    events = np.zeros(ve.N_STEP_EVENTS, dtype=np.int64)
    rands = np.zeros((1, fp.N_RAND))
    for _ in range(40):
        engine.update()
        ve.step_env(
            state,
            bonuses,
            rotating,
            params,
            1,
            POLICY_KINDS["follow_ball"],
            0.0,
            0.0,
            0.0,
            0.0,
            1,
            False,
            rands,
            events,
            True,
        )
    physics = engine.physics_engine
    assert state[fp.BALL_X] == pytest.approx(physics.ball.position.x, abs=1e-6)
    assert state[fp.BALL_Y] == pytest.approx(physics.ball.position.y, abs=1e-6)
    assert state[fp.P2 + fp.PX] == pytest.approx(physics.player2.position.x, abs=1e-6)
    assert state[fp.P2 + fp.PY] == pytest.approx(physics.player2.position.y, abs=1e-6)


def _env(**kwargs):
    defaults: dict = {
        "seed": 1,
        "repeat": 4,
        "opponent_mix": {"dummy": 1.0},
        "sides": (1,),
        "bonus_modes": (False,),
    }
    defaults.update(kwargs)
    return ve.VectorEnv(2, **defaults)


def test_goal_against_gives_terminal_negative_reward_and_serves_again() -> None:
    env = _env()
    env.states[0, fp.BALL_X] = 30.0
    env.states[0, fp.BALL_Y] = 300.0
    env.states[0, fp.BALL_VX] = -600.0
    env.states[0, fp.BALL_VY] = 0.0
    env.states[0, fp.P1 + fp.PY] = 500.0  # paddle far away
    env.states[0, fp.P1 + fp.PPY] = 500.0
    result = env.step(np.zeros(2, dtype=np.int64))
    assert result.rewards[0] == -1.0
    assert bool(result.dones[0]) is True
    assert result.events[0, ve.EV_GOALS_AGAINST] == 1
    assert result.events[0, ve.EV_FRAMES] < 4  # the decision ended at the goal
    assert env.states[0, fp.BALL_X] == pytest.approx(400.0)  # new serve
    assert env.states[0, fp.SCORE2] == 1


def test_truncation_keeps_the_pre_reset_next_observation() -> None:
    env = _env(max_point_frames=2, repeat=1)
    env.states[0, fp.BALL_X] = 123.0
    env.states[0, fp.BALL_VX] = 0.0
    env.states[0, fp.BALL_VY] = 0.0
    env.states[0, fp.POINT_FRAMES] = 1
    result = env.step(np.zeros(2, dtype=np.int64))
    assert bool(result.truncated[0]) is True
    assert bool(result.dones[0]) is False
    assert result.next_obs[0, 0] == pytest.approx(123.0 / 800.0)
    assert env.obs[0, 0] == pytest.approx(0.5)  # restarted at the centre
    assert env.states[0, fp.POINT_FRAMES] == 0


def test_match_end_reports_result_and_reassigns_the_environment() -> None:
    env = _env(
        match_target=1,
        sides=(1, 2),
        bonus_modes=(False, True),
        opponent_mix={"dummy": 1.0, "tracker": 1.0},
    )
    env.states[0, fp.BALL_X] = 780.0
    env.states[0, fp.BALL_Y] = 50.0
    env.states[0, fp.BALL_VX] = 600.0
    env.states[0, fp.BALL_VY] = 0.0
    env.sides[0] = 1
    env.states[0, fp.P2 + fp.PY] = 500.0
    env.states[0, fp.P2 + fp.PPY] = 500.0
    started = env.matches_started
    result = env.step(np.zeros(2, dtype=np.int64))
    assert result.finished and result.finished[0]["won"] is True
    assert env.matches_started == started + 1
    assert env.states[0, fp.SCORE1] == 0 and env.states[0, fp.SCORE2] == 0  # auto reset
    assert env.kinds[0] in (POLICY_KINDS["dummy"], POLICY_KINDS["tracker"])


def test_without_auto_reset_finished_environments_stop() -> None:
    env = _env(match_target=1, auto_reset=False)
    env.states[0, fp.BALL_X] = 780.0
    env.states[0, fp.BALL_Y] = 50.0
    env.states[0, fp.BALL_VX] = 600.0
    env.states[0, fp.BALL_VY] = 0.0
    env.states[0, fp.P2 + fp.PY] = 500.0
    env.states[0, fp.P2 + fp.PPY] = 500.0
    env.step(np.zeros(2, dtype=np.int64))
    assert bool(env.active[0]) is False
    before = env.states[1].copy()
    env.active[1] = False
    env.step(np.zeros(2, dtype=np.int64))
    np.testing.assert_array_equal(env.states[1], before)


def test_opponent_mix_parsing_and_validation() -> None:
    assert ve.parse_opponent_mix("tracker:2, self, planner:0.5") == {
        "tracker": 2.0,
        "self": 1.0,
        "planner": 0.5,
    }
    with pytest.raises(ValueError):
        ve.parse_opponent_mix({"nobody": 1.0})
    with pytest.raises(ValueError):
        ve.parse_opponent_mix({"tracker": -1.0})


def test_self_play_falls_back_to_tracker_until_a_snapshot_exists() -> None:
    env = _env(opponent_mix={"self": 1.0})
    assert set(env.kinds.tolist()) == {POLICY_KINDS["tracker"]}
    env.snapshot_count = 3
    env.reset()
    assert set(env.kinds.tolist()) == {NETWORK_OPPONENT}
    assert env.snapshot_index.max() < 3


def test_planner_opponent_actions_are_planned_for_its_environments() -> None:
    env = ve.VectorEnv(
        3,
        seed=2,
        opponent_mix={"planner": 1.0},
        sides=(1,),
        bonus_modes=(False,),
        planner_horizon=30,
        planner_action_steps=6,
    )
    assert set(env.kinds.tolist()) == {PLANNER_OPPONENT}
    actions = env.planner_actions()
    assert actions.shape == (3,)
    assert all(0 <= a < 9 for a in actions)
    result = env.step(np.zeros(3, dtype=np.int64))
    assert result.events[:, ve.EV_FRAMES].min() >= 1


def test_network_opponent_observations_use_the_opponent_side() -> None:
    env = _env(opponent_mix={"self": 1.0})
    env.kinds[:] = NETWORK_OPPONENT
    own = env.obs.copy()
    other = env.opponent_observations()
    # At the symmetric serve both players see themselves at home and the other far away.
    assert other[0, 4] == pytest.approx(own[0, 4])
    assert other[0, 6] == pytest.approx(own[0, 6])
    assert other[0, 2] == pytest.approx(-own[0, 2])  # mirrored ball velocity


def test_potential_is_bounded_and_prefers_the_intercept() -> None:
    env = _env()
    env.states[0, fp.BALL_X] = 300.0
    env.states[0, fp.BALL_Y] = 100.0
    env.states[0, fp.BALL_VX] = -300.0
    env.states[0, fp.BALL_VY] = 0.0
    env.states[0, fp.P1 + fp.PY] = 60.0  # centre at 100: on the intercept line
    aligned = env.potentials()[0]
    env.states[0, fp.P1 + fp.PY] = 400.0
    away = env.potentials()[0]
    assert -1.0 <= away < aligned <= 0.0
