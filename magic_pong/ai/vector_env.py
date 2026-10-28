"""Vectorised Magic Pong on the numba physics: many games advanced per call.

Each environment holds one match (state, bonuses, rotating paddles) and one
learner side; opponents are built-in policies evaluated inside the kernel,
frozen networks whose actions the caller supplies, or the lookahead planner.
Observations use exactly the DQN checkpoint encoding (player-1 frame), so
networks trained here load into :class:`magic_pong.ai.models.dqn_ai.DQNAgent`
and evaluate through the existing competitive harness.

A decision covers ``repeat`` physics frames; it ends early at a goal so that a
goal reward is never mixed with the following serve.
"""

from __future__ import annotations

import math
from dataclasses import dataclass
from typing import Any

import numpy as np
from numba import njit

from magic_pong.ai.fast_policies import NETWORK_OPPONENT
from magic_pong.ai.fast_policies import PLANNER_OPPONENT
from magic_pong.ai.fast_policies import POLICY_KINDS
from magic_pong.ai.fast_policies import builtin_action
from magic_pong.ai.geometry import GEOMETRY_CONTEXT_HORIZON
from magic_pong.ai.geometry import get_state_layout
from magic_pong.ai.geometry import write_contact_features_scalar
from magic_pong.ai.models.dqn_checkpoint import SCORE_CONTEXT_NORMALIZER
from magic_pong.core import fast_physics as fp
from magic_pong.core.physics import PhysicsEngine
from magic_pong.utils.config import game_config

SELF_OPPONENT = "self"
PLANNER_NAME = "planner"

# Per-environment, per-decision event columns.
(
    EV_GOALS_FOR,
    EV_GOALS_AGAINST,
    EV_LEARNER_HITS,
    EV_OPP_HITS,
    EV_LEARNER_BONUSES,
    EV_MATCH_OVER,
    EV_MATCH_WON,
    EV_FRAMES,
) = range(8)
N_STEP_EVENTS = 8


@njit(cache=True)
def encode_observation(
    state: np.ndarray,
    bonuses: np.ndarray,
    rotating: np.ndarray,
    params: np.ndarray,
    side: int,
    score_context: bool,
    geometry_context: bool,
    out: np.ndarray,
) -> None:
    """Write the DQN input vector for ``side`` (player-1 frame) into ``out``."""
    width = state[fp.FIELD_W]
    height = state[fp.FIELD_H]
    max_speed = params[fp.PR_MAX_SPEED]
    paddle_w = params[fp.PR_PADDLE_WIDTH]
    paddle_h = params[fp.PR_PADDLE_HEIGHT]
    own = fp.P1 if side == 1 else fp.P2
    opp = fp.P2 if side == 1 else fp.P1
    mirror = side == 2
    ball_x = width - state[fp.BALL_X] if mirror else state[fp.BALL_X]
    own_x = width - state[own + fp.PX] - paddle_w if mirror else state[own + fp.PX]
    opp_x = width - state[opp + fp.PX] - paddle_w if mirror else state[opp + fp.PX]
    opp_prev_x = width - state[opp + fp.PPX] - paddle_w if mirror else state[opp + fp.PPX]
    vx = -state[fp.BALL_VX] if mirror else state[fp.BALL_VX]
    out[0] = ball_x / width
    out[1] = state[fp.BALL_Y] / height
    out[2] = vx / max_speed
    out[3] = state[fp.BALL_VY] / max_speed
    out[4] = own_x / width
    out[5] = state[own + fp.PY] / height
    out[6] = opp_x / width
    out[7] = state[opp + fp.PY] / height
    out[8] = opp_prev_x / width
    out[9] = state[opp + fp.PPY] / height
    out[10] = min(4.0, max(0.0, state[own + fp.PH] / paddle_h))
    out[11] = min(4.0, max(0.0, state[opp + fp.PH] / paddle_h))
    own_score = state[fp.SCORE1] if side == 1 else state[fp.SCORE2]
    opp_score = state[fp.SCORE2] if side == 1 else state[fp.SCORE1]
    out[12] = (own_score - opp_score) / 10.0
    out[13] = state[fp.TIME] / 300.0
    active = 0
    for row in range(bonuses.shape[0]):
        if bonuses[row, fp.BACTIVE] > 0:
            active += 1
    out[14] = active / 5.0
    for k in range(9):
        out[15 + k] = 0.0
    k = 0
    for row in range(bonuses.shape[0]):
        if bonuses[row, fp.BACTIVE] > 0 and k < 3:
            bx = width - bonuses[row, fp.BX] if mirror else bonuses[row, fp.BX]
            out[15 + 3 * k] = bx / width
            out[16 + 3 * k] = bonuses[row, fp.BY] / height
            out[17 + 3 * k] = bonuses[row, fp.BTYPE] / 3.0
            k += 1
    for k in range(6):
        out[24 + k] = 0.0
    k = 0
    for row in range(rotating.shape[0]):
        if rotating[row, fp.RLIFE] > 0 and k < 2:
            rx = width - rotating[row, fp.RX] if mirror else rotating[row, fp.RX]
            angle = rotating[row, fp.RANGLE]
            if mirror:
                angle = math.pi - angle
                angle = ((angle + math.pi) % (2 * math.pi)) - math.pi
            wrapped = ((angle + math.pi) % (2 * math.pi)) - math.pi
            out[24 + 3 * k] = rx / width
            out[25 + 3 * k] = rotating[row, fp.RY] / height
            out[26 + 3 * k] = (wrapped + math.pi) / (2 * math.pi)
            k += 1
    if score_context:
        out[30] = own_score / SCORE_CONTEXT_NORMALIZER
        out[31] = opp_score / SCORE_CONTEXT_NORMALIZER
        out[32] = state[fp.MATCH_TARGET] / SCORE_CONTEXT_NORMALIZER
    if geometry_context:
        geom_start = 33 if score_context else 30
        write_contact_features_scalar(
            out,
            out,
            geom_start,
            width,
            height,
            max_speed,
            paddle_w,
            paddle_h,
            params[fp.PR_PADDLE_SPEED],
            params[fp.PR_RADIUS],
            GEOMETRY_CONTEXT_HORIZON,
        )


@njit(cache=True)
def encode_batch(
    states: np.ndarray,
    bonuses: np.ndarray,
    rotating: np.ndarray,
    params: np.ndarray,
    sides: np.ndarray,
    score_context: bool,
    geometry_context: bool,
    out: np.ndarray,
) -> None:
    for i in range(states.shape[0]):
        encode_observation(
            states[i],
            bonuses[i],
            rotating[i],
            params,
            sides[i],
            score_context,
            geometry_context,
            out[i],
        )


@njit(cache=True)
def potential(state: np.ndarray, params: np.ndarray, side: int) -> float:
    """Bounded shaping potential: closeness to the intercept (incoming) or to home."""
    base = fp.P1 if side == 1 else fp.P2
    height = state[fp.FIELD_H]
    width = state[fp.FIELD_W]
    centre = state[base + fp.PY] + state[base + fp.PH] / 2
    sign = 1.0 if side == 1 else -1.0
    if state[fp.BALL_VX] * sign < 0:
        target, arrival = fp._interception(state, params, base, side)
        return -abs(target - centre) / height
    home_x = state[base + fp.PMINX] if side == 1 else state[base + fp.PMAXX]
    return -abs(centre - height / 2) / height - abs(state[base + fp.PX] - home_x) / width


@njit(cache=True)
def potential_batch(
    states: np.ndarray, params: np.ndarray, sides: np.ndarray, out: np.ndarray
) -> None:
    for i in range(states.shape[0]):
        out[i] = potential(states[i], params, sides[i])


@njit(cache=True)
def step_env(
    state: np.ndarray,
    bonuses: np.ndarray,
    rotating: np.ndarray,
    params: np.ndarray,
    side: int,
    opp_kind: int,
    ax: float,
    ay: float,
    opp_ax: float,
    opp_ay: float,
    repeat: int,
    enabled: bool,
    rand_frames: np.ndarray,
    events: np.ndarray,
    auto_reset: bool,
) -> None:
    """Advance one decision (up to ``repeat`` frames) for one environment."""
    for k in range(N_STEP_EVENTS):
        events[k] = 0
    opp_side = 3 - side
    lx = -ax if side == 2 else ax
    ly = ay
    counts = np.zeros(fp.N_EVENTS, dtype=np.int64)
    for frame in range(repeat):
        rand = rand_frames[frame]
        if opp_kind >= 0:
            ox, oy = builtin_action(opp_kind, state, bonuses, params, opp_side, rand)
        else:
            ox = -opp_ax if opp_side == 2 else opp_ax
            oy = opp_ay
        if side == 1:
            goal = fp.step(
                state, bonuses, rotating, params, lx, ly, ox, oy, counts, enabled, rand, True
            )
        else:
            goal = fp.step(
                state, bonuses, rotating, params, ox, oy, lx, ly, counts, enabled, rand, True
            )
        state[fp.POINT_FRAMES] += 1
        events[EV_FRAMES] += 1
        events[EV_LEARNER_HITS] += counts[fp.EV_HIT1 + side - 1]
        events[EV_OPP_HITS] += counts[fp.EV_HIT1 + opp_side - 1]
        events[EV_LEARNER_BONUSES] += counts[fp.EV_BONUS1 + side - 1]
        for k in range(fp.N_EVENTS):
            counts[k] = 0
        if goal:
            if goal == side:
                events[EV_GOALS_FOR] += 1
            else:
                events[EV_GOALS_AGAINST] += 1
            fp.reset_paddles(state, params)
            direction = 1.0 if goal == 2 else -1.0
            fp.serve(
                state, params, direction, -math.pi / 4 + rand[fp.RAND_SERVE_ANGLE] * math.pi / 2
            )
            target = state[fp.MATCH_TARGET]
            if state[fp.SCORE1] >= target or state[fp.SCORE2] >= target:
                events[EV_MATCH_OVER] = 1
                own_score = state[fp.SCORE1] if side == 1 else state[fp.SCORE2]
                events[EV_MATCH_WON] = 1 if own_score >= target else 0
                if auto_reset:
                    fp.reset_match(state, bonuses, rotating, params, rand)
            break


@njit(cache=True)
def step_batch(
    states: np.ndarray,
    bonuses: np.ndarray,
    rotating: np.ndarray,
    params: np.ndarray,
    sides: np.ndarray,
    kinds: np.ndarray,
    actions: np.ndarray,
    opp_actions: np.ndarray,
    repeat: int,
    enabled: np.ndarray,
    rands: np.ndarray,
    events: np.ndarray,
    auto_reset: bool,
    active: np.ndarray,
) -> None:
    for i in range(states.shape[0]):
        if not active[i]:
            for k in range(N_STEP_EVENTS):
                events[i, k] = 0
            continue
        step_env(
            states[i],
            bonuses[i],
            rotating[i],
            params,
            sides[i],
            kinds[i],
            actions[i, 0],
            actions[i, 1],
            opp_actions[i, 0],
            opp_actions[i, 1],
            repeat,
            enabled[i],
            rands[i],
            events[i],
            auto_reset,
        )


@njit(cache=True)
def restart_point(state: np.ndarray, params: np.ndarray, rand: np.ndarray) -> None:
    """Restart a rally that hit the frame cap: paddles home, random serve, scores kept."""
    fp.reset_paddles(state, params)
    fp.serve_random(state, params, rand)


@dataclass
class StepResult:
    """Outcome of one vectorised decision."""

    rewards: np.ndarray
    dones: np.ndarray
    truncated: np.ndarray
    next_obs: np.ndarray
    events: np.ndarray
    finished: list[dict[str, Any]]


def parse_opponent_mix(mix: dict[str, float] | str) -> dict[str, float]:
    """Validate an opponent mix over built-ins, ``self`` and ``planner``."""
    if isinstance(mix, str):
        parsed: dict[str, float] = {}
        for part in mix.split(","):
            part = part.strip()
            if not part:
                continue
            name, _, weight = part.partition(":")
            parsed[name.strip()] = float(weight) if weight else 1.0
        mix = parsed
    allowed = set(POLICY_KINDS) - {"continuation"} | {SELF_OPPONENT, PLANNER_NAME}
    unknown = set(mix) - allowed
    if not mix or unknown:
        raise ValueError(f"unknown opponents {sorted(unknown)}; allowed {sorted(allowed)}")
    if any(not math.isfinite(w) or w < 0 for w in mix.values()) or sum(mix.values()) <= 0:
        raise ValueError("opponent weights must be finite, nonnegative and not all zero")
    return dict(mix)


class VectorEnv:
    """``n`` independent matches with a learner on a chosen side of each."""

    def __init__(
        self,
        n: int,
        *,
        seed: int = 0,
        repeat: int = 4,
        match_target: int = 11,
        max_point_frames: int = 3600,
        terminal_on_goal: bool = True,
        score_context: bool = False,
        geometry_context: bool = False,
        opponent_mix: dict[str, float] | str | None = None,
        sides: tuple[int, ...] = (1, 2),
        bonus_modes: tuple[bool, ...] = (False, True),
        hit_reward: float = 0.0,
        bonus_reward: float = 0.0,
        auto_reset: bool = True,
        planner_horizon: int = 90,
        planner_action_steps: int = 30,
    ) -> None:
        if n < 1 or repeat < 1 or match_target < 1 or max_point_frames < 1:
            raise ValueError("n, repeat, match_target and max_point_frames must be positive")
        if any(s not in (1, 2) for s in sides) or not sides or not bonus_modes:
            raise ValueError("sides must contain 1 and/or 2; bonus_modes must not be empty")
        self.n = n
        self.repeat = repeat
        self.match_target = match_target
        self.max_point_frames = max_point_frames
        self.terminal_on_goal = terminal_on_goal
        self.score_context = score_context
        self.geometry_context = geometry_context
        self.obs_dim = get_state_layout(
            score_context=score_context, geometry_context=geometry_context
        ).state_size
        self.hit_reward = hit_reward
        self.bonus_reward = bonus_reward
        self.auto_reset = auto_reset
        self.mix = parse_opponent_mix(opponent_mix or {"follow_ball": 1.0})
        self.mix_names = list(self.mix)
        weights = np.array([self.mix[name] for name in self.mix_names], dtype=np.float64)
        self.mix_probabilities = weights / weights.sum()
        self.side_choices = tuple(sides)
        self.bonus_choices = tuple(bonus_modes)
        self.rng = np.random.default_rng(seed)
        template = PhysicsEngine(game_config.FIELD_WIDTH, game_config.FIELD_HEIGHT)
        state, bonuses, rotating, params = fp.snapshot(template)
        state[fp.MATCH_TARGET] = match_target
        self.params = params
        self.params_enabled = params.copy()
        self.params_enabled[fp.PR_BONUSES_ENABLED] = 1.0
        self.params_disabled = params.copy()
        self.params_disabled[fp.PR_BONUSES_ENABLED] = 0.0
        self.states = np.repeat(state[None, :], n, axis=0)
        self.bonuses = np.repeat(bonuses[None, :, :], n, axis=0)
        self.rotating = np.repeat(rotating[None, :, :], n, axis=0)
        self.sides = np.ones(n, dtype=np.int64)
        self.kinds = np.zeros(n, dtype=np.int64)
        self.snapshot_index = np.zeros(n, dtype=np.int64)
        self.enabled = np.zeros(n, dtype=np.bool_)
        self.active = np.ones(n, dtype=np.bool_)
        self.obs = np.zeros((n, self.obs_dim), dtype=np.float32)
        self.snapshot_count = 0
        self.planner = None
        if PLANNER_NAME in self.mix and self.mix[PLANNER_NAME] > 0:
            from magic_pong.ai.fast_teacher import FastPhysicsTeacher

            self.planner = FastPhysicsTeacher(planner_horizon, planner_action_steps)
        self.matches_started = 0
        self.reset()

    # --- assignment -----------------------------------------------------------
    def _assign(self, index: int) -> None:
        """Draw side, bonus mode and opponent for a fresh match in environment ``index``."""
        self.sides[index] = self.side_choices[self.rng.integers(len(self.side_choices))]
        self.enabled[index] = self.bonus_choices[self.rng.integers(len(self.bonus_choices))]
        name = self.mix_names[self.rng.choice(len(self.mix_names), p=self.mix_probabilities)]
        if name == SELF_OPPONENT:
            if self.snapshot_count > 0:
                self.kinds[index] = NETWORK_OPPONENT
                self.snapshot_index[index] = self.rng.integers(self.snapshot_count)
            else:
                self.kinds[index] = POLICY_KINDS["tracker"]
        elif name == PLANNER_NAME:
            self.kinds[index] = PLANNER_OPPONENT
        else:
            self.kinds[index] = POLICY_KINDS[name]
        self.matches_started += 1

    def configure(
        self,
        index: int,
        *,
        side: int,
        bonuses_enabled: bool,
        opponent: str | int,
        snapshot_index: int = 0,
    ) -> None:
        """Explicitly set one environment's match conditions (evaluation)."""
        self.sides[index] = side
        self.enabled[index] = bonuses_enabled
        if isinstance(opponent, str):
            if opponent == SELF_OPPONENT:
                self.kinds[index] = NETWORK_OPPONENT
            elif opponent == PLANNER_NAME:
                self.kinds[index] = PLANNER_OPPONENT
            else:
                self.kinds[index] = POLICY_KINDS[opponent]
        else:
            self.kinds[index] = int(opponent)
        self.snapshot_index[index] = snapshot_index

    def reset(self) -> np.ndarray:
        """Start a fresh match everywhere and return the learner observations."""
        rands = self.rng.random((self.n, fp.N_RAND))
        for i in range(self.n):
            fp.reset_match(self.states[i], self.bonuses[i], self.rotating[i], self.params, rands[i])
            self.states[i, fp.MATCH_TARGET] = self.match_target
            self._assign(i)
        self.active[:] = True
        self._encode(self.obs)
        return self.obs

    def reset_env(self, index: int) -> None:
        rand = self.rng.random(fp.N_RAND)
        fp.reset_match(
            self.states[index], self.bonuses[index], self.rotating[index], self.params, rand
        )
        self.states[index, fp.MATCH_TARGET] = self.match_target
        self._assign(index)
        self.active[index] = True
        encode_observation(
            self.states[index],
            self.bonuses[index],
            self.rotating[index],
            self.params,
            self.sides[index],
            self.score_context,
            self.geometry_context,
            self.obs[index],
        )

    # --- observations -----------------------------------------------------------
    def _encode(self, out: np.ndarray, sides: np.ndarray | None = None) -> None:
        encode_batch(
            self.states,
            self.bonuses,
            self.rotating,
            self.params,
            self.sides if sides is None else sides,
            self.score_context,
            self.geometry_context,
            out,
        )

    def opponent_observations(self) -> np.ndarray:
        """Observations from the opponent's point of view (for network opponents)."""
        out = np.zeros((self.n, self.obs_dim), dtype=np.float32)
        self._encode(out, 3 - self.sides)
        return out

    def potentials(self) -> np.ndarray:
        out = np.zeros(self.n, dtype=np.float64)
        potential_batch(self.states, self.params, self.sides, out)
        return out

    @property
    def network_mask(self) -> np.ndarray:
        return self.kinds == NETWORK_OPPONENT

    @property
    def planner_mask(self) -> np.ndarray:
        return self.kinds == PLANNER_OPPONENT

    def planner_actions(self) -> np.ndarray:
        """Action indices (player-1 frame) chosen by the planner for its environments."""
        indices = np.flatnonzero(self.planner_mask & self.active)
        out = np.zeros(self.n, dtype=np.int64)
        if len(indices) == 0:
            return out
        if self.planner is None:
            raise RuntimeError("planner opponents require the planner in the opponent mix")
        for i in indices:
            params = self.params_enabled if self.enabled[i] else self.params_disabled
            decision = self.planner.label_state(
                self.states[i], self.bonuses[i], self.rotating[i], params, int(3 - self.sides[i])
            )
            out[i] = decision.action
        return out

    # --- stepping -----------------------------------------------------------------
    def step(self, actions: np.ndarray, opponent_actions: np.ndarray | None = None) -> StepResult:
        """Advance every active environment by one decision.

        ``actions`` and ``opponent_actions`` are indices into ``fp.ACTION_TABLE`` in the
        player-1 frame; opponent actions are only read for network and planner
        opponents. Returns rewards, terminal flags, truncation flags, the learning
        next observation (pre-reset on truncation) and per-environment events. The
        current observation for the next decision is ``self.obs``.
        """
        actions = np.asarray(actions, dtype=np.int64)
        if actions.shape != (self.n,):
            raise ValueError("actions must have one entry per environment")
        if opponent_actions is None:
            opponent_actions = np.zeros(self.n, dtype=np.int64)
        opponent_actions = np.asarray(opponent_actions, dtype=np.int64)
        needs_planner = self.planner_mask & self.active
        if needs_planner.any():
            planned = self.planner_actions()
            opponent_actions = np.where(needs_planner, planned, opponent_actions)
        act = fp.ACTION_TABLE[actions]
        opp_act = fp.ACTION_TABLE[opponent_actions]
        rands = self.rng.random((self.n, self.repeat, fp.N_RAND))
        events = np.zeros((self.n, N_STEP_EVENTS), dtype=np.int64)
        step_batch(
            self.states,
            self.bonuses,
            self.rotating,
            self.params,
            self.sides,
            self.kinds,
            act,
            opp_act,
            self.repeat,
            self.enabled,
            rands,
            events,
            self.auto_reset,
            self.active,
        )
        goals_for = events[:, EV_GOALS_FOR].astype(np.float64)
        goals_against = events[:, EV_GOALS_AGAINST].astype(np.float64)
        rewards = goals_for - goals_against
        rewards += self.hit_reward * events[:, EV_LEARNER_HITS]
        rewards += self.bonus_reward * events[:, EV_LEARNER_BONUSES]
        match_over = events[:, EV_MATCH_OVER] > 0
        if self.terminal_on_goal:
            dones = (goals_for + goals_against) > 0
        else:
            dones = match_over.copy()
        next_obs = np.zeros_like(self.obs)
        self._encode(next_obs)
        truncated = (
            (self.states[:, fp.POINT_FRAMES] >= self.max_point_frames) & self.active & ~dones
        )
        finished: list[dict[str, Any]] = []
        for i in np.flatnonzero(match_over & self.active):
            finished.append(
                {
                    "env": int(i),
                    "won": bool(events[i, EV_MATCH_WON]),
                    "opponent": int(self.kinds[i]),
                    "snapshot": int(self.snapshot_index[i]),
                    "side": int(self.sides[i]),
                    "bonuses": bool(self.enabled[i]),
                }
            )
            if self.auto_reset:
                self._assign(int(i))
            else:
                self.active[i] = False
        for i in np.flatnonzero(truncated):
            restart_point(self.states[i], self.params, self.rng.random(fp.N_RAND))
        self._encode(self.obs)
        return StepResult(rewards.astype(np.float32), dones, truncated, next_obs, events, finished)
