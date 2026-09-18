"""Numba-compiled rollouts of the core physics for planning and label collection.

The kernels mirror :meth:`magic_pong.core.physics.PhysicsEngine._update_substep`
for one fixed timestep per frame: paddle effect timers, paddle moves, ball move,
rotating paddles, bonus lifetimes, then paddle, rotating-paddle, wall/goal and
bonus collisions, in that order. ``tests/core/test_fast_physics.py`` checks
frame-by-frame agreement with the Python engine.

Documented differences from the Python engine:

- no new bonuses spawn during a rollout (spawning draws from the global RNG);
- a collected rotating-paddle bonus places the obstacle at the centre of the
  collector's half instead of a random position;
- a rollout stops at the first goal instead of serving a new random ball.

The module depends only on numpy and numba so ``magic_pong.core`` stays free of
AI dependencies; the AI package wraps it in :mod:`magic_pong.ai.fast_teacher`.
"""

from __future__ import annotations

import math
from dataclasses import dataclass
from typing import TYPE_CHECKING

import numpy as np
from numba import njit

from magic_pong.utils.config import game_config

if TYPE_CHECKING:
    from magic_pong.core.physics import PhysicsEngine

# --- state vector layout -----------------------------------------------------
BALL_X, BALL_Y, BALL_VX, BALL_VY, BALL_PX, BALL_PY, BALL_LAST_HIT = range(7)
# Paddle block offsets (relative to P1 / P2 base index).
PX, PY, PPX, PPY, PW, PH, PORIG, PTIMER, PMINX, PMAXX, PMINY, PMAXY, PSPEED = range(13)
N_PADDLE = 13
P1 = 7
P2 = P1 + N_PADDLE
SCORE1 = P2 + N_PADDLE
SCORE2 = SCORE1 + 1
TIME = SCORE2 + 1
FIELD_W = TIME + 1
FIELD_H = FIELD_W + 1
SPAWN_TIMER = FIELD_H + 1  # bonus spawner clock (seconds since the last spawn attempt)
MATCH_TARGET = SPAWN_TIMER + 1  # points needed to win the match
POINT_FRAMES = MATCH_TARGET + 1  # frames since the last serve
N_STATE = POINT_FRAMES + 1

# --- parameter vector layout -------------------------------------------------
(
    PR_RADIUS,
    PR_MAX_SPEED,
    PR_INCREASE,
    PR_DT,
    PR_BONUS_DURATION,
    PR_SIZE_MULT,
    PR_SIZE_RED,
    PR_ROT_RADIUS,
    PR_ROT_SPEED,
    PR_ROT_DURATION,
    PR_BONUSES_ENABLED,
    PR_PADDLE_MARGIN,
    PR_BONUS_SIZE,
    PR_PADDLE_WIDTH,
    PR_PADDLE_HEIGHT,
    PR_PADDLE_SPEED,
    PR_FPS,
    PR_BALL_SPEED,
    PR_SPAWN_INTERVAL,
    PR_BONUS_LIFETIME,
) = range(20)
N_PARAMS = 20

# --- per-frame random draws (uniform in [0, 1)) consumed by stochastic steps ---
(
    RAND_SERVE_ANGLE,
    RAND_SERVE_DIR,
    RAND_BONUS_TYPE,
    RAND_BONUS_X,
    RAND_BONUS_Y,
    RAND_ROT_X,
    RAND_ROT_Y,
    RAND_OPP_X,
    RAND_OPP_Y,
) = range(9)
N_RAND = 9
MAX_ACTIVE_BONUSES = 4

# --- bonus and rotating paddle tables ----------------------------------------
BX, BY, BTYPE, BLIFE, BACTIVE = range(5)
N_BONUS_COLS = 5
MAX_BONUSES = 8
BONUS_CODES = {"enlarge_paddle": 1.0, "shrink_opponent": 2.0, "rotating_paddle": 3.0}

RX, RY, RANGLE, RLIFE, RPLAYER, RRADIUS = range(6)
N_ROTATING_COLS = 6
MAX_ROTATING = 8

# --- event counters -----------------------------------------------------------
EV_HIT1, EV_HIT2, EV_WALL, EV_ROTATING, EV_BONUS1, EV_BONUS2, EV_GOAL = range(7)
N_EVENTS = 7

# Same order as magic_pong.ai.models.dqn_ai.ACTION_MAPPING (player-1 frame).
ACTION_TABLE = np.array(
    [
        [0.0, 0.0],
        [0.0, -1.0],
        [0.0, 1.0],
        [-1.0, 0.0],
        [1.0, 0.0],
        [-1.0, -1.0],
        [1.0, -1.0],
        [-1.0, 1.0],
        [1.0, 1.0],
    ],
    dtype=np.float64,
)

GOAL_VALUE = 1000.0


@dataclass(frozen=True)
class SimulationResult:
    """Outcome of :func:`simulate`: frames run, event counts and final tables."""

    frames: int
    counts: np.ndarray
    state: np.ndarray
    bonuses: np.ndarray
    rotating: np.ndarray


# --- snapshot -----------------------------------------------------------------
def snapshot(
    physics: PhysicsEngine,
) -> tuple[np.ndarray, np.ndarray, np.ndarray, np.ndarray]:
    """Copy a live engine into flat arrays: (state, bonuses, rotating, params)."""
    state = np.zeros(N_STATE, dtype=np.float64)
    ball = physics.ball
    state[BALL_X], state[BALL_Y] = ball.position.x, ball.position.y
    state[BALL_VX], state[BALL_VY] = ball.velocity.x, ball.velocity.y
    state[BALL_PX], state[BALL_PY] = ball.prev_position.x, ball.prev_position.y
    state[BALL_LAST_HIT] = float(ball.last_paddle_hit or 0)
    for base, paddle in ((P1, physics.player1), (P2, physics.player2)):
        state[base + PX], state[base + PY] = paddle.position.x, paddle.position.y
        state[base + PPX], state[base + PPY] = paddle.prev_position.x, paddle.prev_position.y
        state[base + PW], state[base + PH] = paddle.width, paddle.height
        state[base + PORIG] = paddle.original_height
        state[base + PTIMER] = paddle.size_effect_timer
        state[base + PMINX], state[base + PMAXX] = paddle.min_x, paddle.max_x
        state[base + PMINY], state[base + PMAXY] = paddle.min_y, paddle.max_y
        state[base + PSPEED] = paddle.speed
    state[SCORE1], state[SCORE2] = physics.score
    state[TIME] = physics.game_time
    state[FIELD_W], state[FIELD_H] = physics.field_width, physics.field_height
    state[SPAWN_TIMER] = physics.bonus_spawner.spawn_timer
    state[MATCH_TARGET] = game_config.MAX_SCORE
    state[POINT_FRAMES] = 0.0

    bonuses = np.zeros((MAX_BONUSES, N_BONUS_COLS), dtype=np.float64)
    active = [bonus for bonus in physics.bonuses if not bonus.collected]
    for row, bonus in zip(range(MAX_BONUSES), active):
        bonuses[row, BX], bonuses[row, BY] = bonus.position.x, bonus.position.y
        bonuses[row, BTYPE] = BONUS_CODES.get(bonus.type.value, 0.0)
        bonuses[row, BLIFE] = bonus.lifetime
        bonuses[row, BACTIVE] = 1.0

    rotating = np.zeros((MAX_ROTATING, N_ROTATING_COLS), dtype=np.float64)
    for row, obstacle in zip(range(MAX_ROTATING), physics.rotating_paddles):
        rotating[row, RX], rotating[row, RY] = obstacle.center.x, obstacle.center.y
        rotating[row, RANGLE] = obstacle.angle
        rotating[row, RLIFE] = obstacle.lifetime
        rotating[row, RPLAYER] = float(obstacle.player_id)
        rotating[row, RRADIUS] = obstacle.radius

    params = np.zeros(N_PARAMS, dtype=np.float64)
    params[PR_RADIUS] = ball.radius
    params[PR_MAX_SPEED] = game_config.MAX_BALL_SPEED
    params[PR_INCREASE] = game_config.BALL_SPEED_INCREASE
    params[PR_DT] = game_config.GAME_SPEED_MULTIPLIER / game_config.FPS
    params[PR_BONUS_DURATION] = game_config.BONUS_DURATION
    params[PR_SIZE_MULT] = game_config.PADDLE_SIZE_MULTIPLIER
    params[PR_SIZE_RED] = game_config.PADDLE_SIZE_REDUCER
    params[PR_ROT_RADIUS] = game_config.ROTATING_PADDLE_RADIUS
    params[PR_ROT_SPEED] = game_config.ROTATING_PADDLE_SPEED
    params[PR_ROT_DURATION] = game_config.ROTATING_PADDLE_DURATION
    params[PR_BONUSES_ENABLED] = 1.0 if game_config.BONUSES_ENABLED else 0.0
    params[PR_PADDLE_MARGIN] = game_config.PADDLE_MARGIN
    params[PR_BONUS_SIZE] = game_config.BONUS_SIZE
    params[PR_PADDLE_WIDTH] = game_config.PADDLE_WIDTH
    params[PR_PADDLE_HEIGHT] = game_config.PADDLE_HEIGHT
    params[PR_PADDLE_SPEED] = game_config.PADDLE_SPEED
    params[PR_FPS] = game_config.FPS
    params[PR_BALL_SPEED] = game_config.BALL_SPEED
    params[PR_SPAWN_INTERVAL] = game_config.BONUS_SPAWN_INTERVAL
    params[PR_BONUS_LIFETIME] = game_config.BONUS_LIFETIME
    return state, bonuses, rotating, params


def no_random() -> np.ndarray:
    """Zero draws for deterministic kernels (rollouts never spawn or serve)."""
    return np.zeros(N_RAND, dtype=np.float64)


# --- geometry helpers (mirroring magic_pong.core.collision) ------------------
@njit(cache=True)
def _clamp(value, low, high):
    return max(low, min(high, value))


@njit(cache=True)
def _pymod(value, modulus):
    """Python's float ``%`` (numba keeps Python's sign convention for floats)."""
    return value % modulus


@njit(cache=True)
def _circle_rect(px, py, radius, rx, ry, rw, rh):
    cx = max(rx, min(px, rx + rw))
    cy = max(ry, min(py, ry + rh))
    return math.sqrt((px - cx) ** 2 + (py - cy) ** 2) <= radius


@njit(cache=True)
def _surface_normal(px, py, rx, ry, rw, rh):
    cx = max(rx, min(px, rx + rw))
    cy = max(ry, min(py, ry + rh))
    ox = px - cx
    oy = py - cy
    magnitude = math.hypot(ox, oy)
    if magnitude > 1e-12:
        return ox / magnitude, oy / magnitude
    # Embedded centre: nearest exit, first minimum wins (left, right, top, bottom).
    best = px - rx
    nx, ny = -1.0, 0.0
    if rx + rw - px < best:
        best = rx + rw - px
        nx, ny = 1.0, 0.0
    if py - ry < best:
        best = py - ry
        nx, ny = 0.0, -1.0
    if ry + rh - py < best:
        nx, ny = 0.0, 1.0
    return nx, ny


@njit(cache=True)
def _spin_normal(px, py, rx, ry, rw, rh, sx, sy):
    relative = _clamp((py - ry - rh / 2) / (rh / 2), -1.0, 1.0)
    nx = sx
    ny = sy + 0.3 * relative
    magnitude = math.hypot(nx, ny)
    if magnitude == 0.0:
        return 0.0, 0.0
    return nx / magnitude, ny / magnitude


@njit(cache=True)
def _entry_time(sx, sy, ex, ey, radius, prev_rect, cur_rect, t_start, t_end):
    epsilon = 0.001
    while t_end - t_start > epsilon:
        t_mid = (t_start + t_end) / 2
        mx = sx + (ex - sx) * t_mid
        my = sy + (ey - sy) * t_mid
        rx = prev_rect[0] + (cur_rect[0] - prev_rect[0]) * t_mid
        ry = prev_rect[1] + (cur_rect[1] - prev_rect[1]) * t_mid
        rw = prev_rect[2] + (cur_rect[2] - prev_rect[2]) * t_mid
        rh = prev_rect[3] + (cur_rect[3] - prev_rect[3]) * t_mid
        if _circle_rect(mx, my, radius, rx, ry, rw, rh):
            t_end = t_mid
        else:
            t_start = t_mid
    return t_end


@njit(cache=True)
def _trajectory_intersection(sx, sy, ex, ey, radius, prev_rect, cur_rect):
    samples = 30
    for i in range(1, samples):
        t = i / samples
        mx = sx + (ex - sx) * t
        my = sy + (ey - sy) * t
        rx = prev_rect[0] + (cur_rect[0] - prev_rect[0]) * t
        ry = prev_rect[1] + (cur_rect[1] - prev_rect[1]) * t
        rw = prev_rect[2] + (cur_rect[2] - prev_rect[2]) * t
        rh = prev_rect[3] + (cur_rect[3] - prev_rect[3]) * t
        if _circle_rect(mx, my, radius, rx, ry, rw, rh):
            return True, _entry_time(
                sx, sy, ex, ey, radius, prev_rect, cur_rect, (i - 1) / samples, t
            )
    return False, 0.0


@njit(cache=True)
def _continuous_collision(state, base, radius):
    prev_rect = (state[base + PPX], state[base + PPY], state[base + PW], state[base + PH])
    cur_rect = (state[base + PX], state[base + PY], state[base + PW], state[base + PH])
    sx, sy = state[BALL_PX], state[BALL_PY]
    ex, ey = state[BALL_X], state[BALL_Y]
    prev_in = _circle_rect(sx, sy, radius, prev_rect[0], prev_rect[1], prev_rect[2], prev_rect[3])
    curr_in = _circle_rect(ex, ey, radius, cur_rect[0], cur_rect[1], cur_rect[2], cur_rect[3])
    if prev_in:
        cx = max(prev_rect[0], min(sx, prev_rect[0] + prev_rect[2]))
        cy = max(prev_rect[1], min(sy, prev_rect[1] + prev_rect[3]))
        ox = sx - cx
        oy = sy - cy
        if abs(math.hypot(ox, oy) - radius) <= 1e-9:
            rsx = (ex - sx) - (state[base + PX] - state[base + PPX])
            rsy = (ey - sy) - (state[base + PY] - state[base + PPY])
            if rsx * ox + rsy * oy < 0:
                return True, 0.0
    if not prev_in and curr_in:
        return True, _entry_time(sx, sy, ex, ey, radius, prev_rect, cur_rect, 0.0, 1.0)
    if prev_in and not curr_in:
        return False, 0.0
    if not prev_in and not curr_in:
        return _trajectory_intersection(sx, sy, ex, ey, radius, prev_rect, cur_rect)
    return False, 0.0


@njit(cache=True)
def _separate(state, base, radius, force, has_normal, nx, ny):
    rx, ry, rw, rh = state[base + PX], state[base + PY], state[base + PW], state[base + PH]
    bx, by = state[BALL_X], state[BALL_Y]
    if not force and not _circle_rect(bx, by, radius, rx, ry, rw, rh):
        return
    if not has_normal:
        nx, ny = _surface_normal(bx, by, rx, ry, rw, rh)
    magnitude = math.hypot(nx, ny)
    if magnitude == 0.0:
        nx, ny = 0.0, 0.0
    else:
        nx, ny = nx / magnitude, ny / magnitude
    support_x = rx + (rw if nx > 0 else 0.0)
    support_y = ry + (rh if ny > 0 else 0.0)
    distance = (bx - support_x) * nx + (by - support_y) * ny
    correction = radius + 1e-7 - distance
    if correction > 0:
        state[BALL_X] = bx + nx * correction
        state[BALL_Y] = by + ny * correction


@njit(cache=True)
def _apply_bounce(state, params, nx, ny, sx, sy):
    magnitude = math.hypot(nx, ny)
    if magnitude < 1e-12:
        return
    nx, ny = nx / magnitude, ny / magnitude
    vx, vy = state[BALL_VX], state[BALL_VY]
    speed = min(math.hypot(vx, vy) * params[PR_INCREASE], params[PR_MAX_SPEED])
    dot = vx * nx + vy * ny
    rx = vx - nx * (2 * dot)
    ry = vy - ny * (2 * dot)
    if rx * sx + ry * sy < 0:
        dot = vx * sx + vy * sy
        rx = vx - sx * (2 * dot)
        ry = vy - sy * (2 * dot)
    reflected = math.hypot(rx, ry)
    if reflected == 0.0:
        state[BALL_VX], state[BALL_VY] = 0.0, 0.0
    else:
        state[BALL_VX] = (rx / reflected) * speed
        state[BALL_VY] = (ry / reflected) * speed


@njit(cache=True)
def _check_ball_paddle(state, params, base, player_id):
    radius = params[PR_RADIUS]
    dt = params[PR_DT]
    collision, time = _continuous_collision(state, base, radius)
    if not collision:
        _separate(state, base, radius, False, False, 0.0, 0.0)
        return False
    prev_x, prev_y = state[base + PPX], state[base + PPY]
    cur_x, cur_y = state[base + PX], state[base + PY]
    rw, rh = state[base + PW], state[base + PH]
    crx = prev_x + (cur_x - prev_x) * time
    cry = prev_y + (cur_y - prev_y) * time
    contact_x = state[BALL_PX] + (state[BALL_X] - state[BALL_PX]) * time
    contact_y = state[BALL_PY] + (state[BALL_Y] - state[BALL_PY]) * time
    sx, sy = _surface_normal(contact_x, contact_y, crx, cry, rw, rh)
    approach = state[BALL_VX] * sx + state[BALL_VY] * sy
    if dt > 0:
        pvx = (cur_x - prev_x) / dt
        pvy = (cur_y - prev_y) / dt
    else:
        pvx, pvy = 0.0, 0.0
    relative = approach - (pvx * sx + pvy * sy)
    bounce = approach < -1e-9 and relative < -1e-9
    if bounce:
        nx, ny = _spin_normal(contact_x, contact_y, crx, cry, rw, rh, sx, sy)
        _apply_bounce(state, params, nx, ny, sx, sy)
        state[BALL_LAST_HIT] = player_id
    _separate(state, base, radius, True, True, sx, sy)
    return bounce


@njit(cache=True)
def _rotating_hit(state, params, rotating, row):
    radius = params[PR_RADIUS]
    bx, by = state[BALL_X], state[BALL_Y]
    cx, cy = rotating[row, RX], rotating[row, RY]
    angle = rotating[row, RANGLE]
    reach = rotating[row, RRADIUS]
    for k in range(4):
        a1 = angle + k * math.pi / 2
        a2 = angle + (k + 1) * math.pi / 2
        p1x, p1y = cx + reach * math.cos(a1), cy + reach * math.sin(a1)
        p2x, p2y = cx + reach * math.cos(a2), cy + reach * math.sin(a2)
        lvx, lvy = p2x - p1x, p2y - p1y
        length = math.hypot(lvx, lvy)
        if length < 1e-6:
            hit = math.hypot(bx - p1x, by - p1y) <= radius
        else:
            ux, uy = lvx / length, lvy / length
            projection = _clamp((bx - p1x) * ux + (by - p1y) * uy, 0.0, length)
            closest_x = p1x + ux * projection
            closest_y = p1y + uy * projection
            hit = math.hypot(bx - closest_x, by - closest_y) <= radius
        if hit:
            nx, ny = -lvy, lvx
            if nx * (bx - p1x) + ny * (by - p1y) < 0:
                nx, ny = -nx, -ny
            magnitude = math.hypot(nx, ny)
            if magnitude == 0.0:
                nx, ny = 0.0, 0.0
            else:
                nx, ny = nx / magnitude, ny / magnitude
            dot = state[BALL_VX] * nx + state[BALL_VY] * ny
            state[BALL_VX] -= 2 * dot * nx
            state[BALL_VY] -= 2 * dot * ny
            return True
    return False


@njit(cache=True)
def _constrain(state, base):
    state[base + PX] = _clamp(state[base + PX], state[base + PMINX], state[base + PMAXX])
    state[base + PY] = _clamp(state[base + PY], state[base + PMINY], state[base + PMAXY])


@njit(cache=True)
def _reset_size(state, base):
    state[base + PH] = state[base + PORIG]
    state[base + PMAXY] = state[FIELD_H] - state[base + PH]
    _constrain(state, base)


@njit(cache=True)
def _apply_size_effect(state, base, multiplier, duration):
    height = state[base + PH]
    original = state[base + PORIG]
    new_size = max(min(height * multiplier, original * 4, state[FIELD_H] * 0.95), original * 0.25)
    state[base + PTIMER] = duration
    state[base + PY] -= (new_size - height) / 2
    state[base + PH] = new_size
    state[base + PMAXY] = state[FIELD_H] - new_size
    _constrain(state, base)


@njit(cache=True)
def _paddle_update(state, base, dt):
    if state[base + PTIMER] > 0:
        state[base + PTIMER] -= dt
        if state[base + PTIMER] <= 0:
            _reset_size(state, base)


@njit(cache=True)
def _paddle_move(state, base, ax, ay, dt):
    state[base + PPX] = state[base + PX]
    state[base + PPY] = state[base + PY]
    speed = state[base + PSPEED] * dt
    state[base + PX] = state[base + PX] + _clamp(ax, -1.0, 1.0) * speed
    state[base + PY] = state[base + PY] + _clamp(ay, -1.0, 1.0) * speed
    _constrain(state, base)


@njit(cache=True)
def compact_bonuses(bonuses):
    """Keep active rows first, in order, like the engine's compacted bonus list."""
    write = 0
    for row in range(bonuses.shape[0]):
        if bonuses[row, BACTIVE] > 0:
            if write != row:
                for col in range(bonuses.shape[1]):
                    bonuses[write, col] = bonuses[row, col]
            write += 1
    for row in range(write, bonuses.shape[0]):
        for col in range(bonuses.shape[1]):
            bonuses[row, col] = 0.0
    return write


@njit(cache=True)
def compact_rotating(rotating):
    """Keep live rotating paddles first, in order, like the engine's list."""
    write = 0
    for row in range(rotating.shape[0]):
        if rotating[row, RLIFE] > 0:
            if write != row:
                for col in range(rotating.shape[1]):
                    rotating[write, col] = rotating[row, col]
            write += 1
    for row in range(write, rotating.shape[0]):
        for col in range(rotating.shape[1]):
            rotating[row, col] = 0.0
    return write


@njit(cache=True)
def _add_rotating(state, params, rotating, player_id, rand, use_rand):
    row = compact_rotating(rotating)
    if row >= rotating.shape[0]:
        return
    x_margin = params[PR_PADDLE_MARGIN] + params[PR_PADDLE_WIDTH] + params[PR_ROT_RADIUS]
    y_margin = params[PR_ROT_RADIUS]
    width = state[FIELD_W]
    height = state[FIELD_H]
    if use_rand:
        span = width / 2 - 2 * x_margin
        if player_id == 1:
            x = x_margin + rand[RAND_ROT_X] * span
        else:
            x = width / 2 + x_margin + rand[RAND_ROT_X] * span
        y = y_margin + rand[RAND_ROT_Y] * (height - 2 * y_margin)
    else:
        if player_id == 1:
            x = (x_margin + (width / 2 - x_margin)) / 2
        else:
            x = (width / 2 + x_margin + (width - x_margin)) / 2
        y = height / 2
    rotating[row, RX] = x
    rotating[row, RY] = y
    rotating[row, RANGLE] = 0.0
    rotating[row, RLIFE] = params[PR_ROT_DURATION]
    rotating[row, RPLAYER] = player_id
    rotating[row, RRADIUS] = params[PR_ROT_RADIUS]


@njit(cache=True)
def spawn_bonuses(state, bonuses, params, rand):
    """Spawn a symmetric bonus pair like ``BonusSpawner`` when fewer than four are alive."""
    active = compact_bonuses(bonuses)
    if active >= MAX_ACTIVE_BONUSES or active + 2 > bonuses.shape[0]:
        return
    kind = 1.0 + min(2, int(rand[RAND_BONUS_TYPE] * 3))
    width = state[FIELD_W]
    height = state[FIELD_H]
    margin = params[PR_PADDLE_MARGIN] + params[PR_PADDLE_WIDTH] + params[PR_BONUS_SIZE]
    vertical = params[PR_BONUS_SIZE]
    left_x = margin + rand[RAND_BONUS_X] * (width / 2 - margin - margin)
    y = vertical + rand[RAND_BONUS_Y] * (height - vertical - vertical)
    right_x = width - left_x
    if not (margin <= left_x <= width - margin and vertical <= y <= height - vertical):
        return
    for row, x in ((active, left_x), (active + 1, right_x)):
        bonuses[row, BX] = x
        bonuses[row, BY] = y
        bonuses[row, BTYPE] = kind
        bonuses[row, BLIFE] = params[PR_BONUS_LIFETIME]
        bonuses[row, BACTIVE] = 1.0


@njit(cache=True)
def reset_paddles(state, params):
    """Return both paddles to their serve positions with their original size."""
    width = state[FIELD_W]
    height = state[FIELD_H]
    margin = params[PR_PADDLE_MARGIN]
    for base in (P1, P2):
        original = state[base + PORIG]
        x = margin if base == P1 else width - margin - state[base + PW]
        y = height / 2 - original / 2
        state[base + PX] = x
        state[base + PY] = y
        state[base + PPX] = x
        state[base + PPY] = y
        state[base + PH] = original
        state[base + PTIMER] = 0.0
        state[base + PMAXY] = height - original


@njit(cache=True)
def serve(state, params, direction, angle):
    """Place the ball at the centre moving at ``BALL_SPEED`` in the given direction."""
    speed = params[PR_BALL_SPEED]
    vx = direction * speed * math.cos(angle)
    vy = speed * math.sin(angle)
    state[BALL_X] = state[FIELD_W] / 2
    state[BALL_Y] = state[FIELD_H] / 2
    state[BALL_VX] = vx
    state[BALL_VY] = vy
    state[BALL_PX] = state[BALL_X] - vx / params[PR_FPS]
    state[BALL_PY] = state[BALL_Y] - vy / params[PR_FPS]
    state[BALL_LAST_HIT] = 0.0
    state[POINT_FRAMES] = 0.0


@njit(cache=True)
def serve_random(state, params, rand):
    """Serve toward a random side at a random angle within ±45 degrees."""
    direction = -1.0 if rand[RAND_SERVE_DIR] < 0.5 else 1.0
    serve(state, params, direction, -math.pi / 4 + rand[RAND_SERVE_ANGLE] * math.pi / 2)


@njit(cache=True)
def reset_match(state, bonuses, rotating, params, rand):
    """Start a fresh match: scores, clock, bonuses, obstacles, paddles and a random serve."""
    state[SCORE1] = 0.0
    state[SCORE2] = 0.0
    state[TIME] = 0.0
    state[SPAWN_TIMER] = 0.0
    for row in range(bonuses.shape[0]):
        for col in range(bonuses.shape[1]):
            bonuses[row, col] = 0.0
    for row in range(rotating.shape[0]):
        for col in range(rotating.shape[1]):
            rotating[row, col] = 0.0
    reset_paddles(state, params)
    serve_random(state, params, rand)


@njit(cache=True)
def _collect_bonuses(state, params, bonuses, rotating, player_id, counts, enabled, rand, use_rand):
    base = P1 if player_id == 1 else P2
    other = P2 if player_id == 1 else P1
    px, py, pw, ph = state[base + PX], state[base + PY], state[base + PW], state[base + PH]
    size = params[PR_BONUS_SIZE]
    for row in range(bonuses.shape[0]):
        if bonuses[row, BACTIVE] <= 0:
            continue
        bx, by = bonuses[row, BX], bonuses[row, BY]
        if px < bx + size and px + pw > bx and py < by + size and py + ph > by:
            bonuses[row, BACTIVE] = 0.0
            counts[EV_BONUS1 + player_id - 1] += 1
            if not enabled:
                continue
            kind = bonuses[row, BTYPE]
            if kind == 1.0:
                _apply_size_effect(state, base, params[PR_SIZE_MULT], params[PR_BONUS_DURATION])
            elif kind == 2.0:
                _apply_size_effect(state, other, params[PR_SIZE_RED], params[PR_BONUS_DURATION])
            elif kind == 3.0:
                _add_rotating(state, params, rotating, player_id, rand, use_rand)


@njit(cache=True)
def step(state, bonuses, rotating, params, a1x, a1y, a2x, a2y, counts, enabled, rand, use_rand):
    """Advance one frame in place. Returns the scoring player (0 if none).

    ``enabled`` applies bonus effects (and, with ``use_rand``, spawns bonuses from
    the draws in ``rand``); rollouts pass ``use_rand=False`` and stay deterministic.
    """
    dt = params[PR_DT]
    radius = params[PR_RADIUS]
    state[TIME] += dt
    _paddle_update(state, P1, dt)
    _paddle_update(state, P2, dt)
    _paddle_move(state, P1, a1x, a1y, dt)
    _paddle_move(state, P2, a2x, a2y, dt)
    state[BALL_PX], state[BALL_PY] = state[BALL_X], state[BALL_Y]
    state[BALL_X] = state[BALL_X] + state[BALL_VX] * dt
    state[BALL_Y] = state[BALL_Y] + state[BALL_VY] * dt
    for row in range(rotating.shape[0]):
        if rotating[row, RLIFE] > 0:
            rotating[row, RANGLE] += params[PR_ROT_SPEED] * dt
            rotating[row, RLIFE] -= dt
    for row in range(bonuses.shape[0]):
        if bonuses[row, BACTIVE] > 0:
            bonuses[row, BLIFE] -= dt
            if bonuses[row, BLIFE] <= 0:
                bonuses[row, BACTIVE] = 0.0
    if use_rand and enabled:
        state[SPAWN_TIMER] += dt
        if state[SPAWN_TIMER] >= params[PR_SPAWN_INTERVAL]:
            state[SPAWN_TIMER] = 0.0
            spawn_bonuses(state, bonuses, params, rand)

    if _check_ball_paddle(state, params, P1, 1):
        counts[EV_HIT1] += 1
    if _check_ball_paddle(state, params, P2, 2):
        counts[EV_HIT2] += 1
    for row in range(rotating.shape[0]):
        if rotating[row, RLIFE] > 0 and _rotating_hit(state, params, rotating, row):
            counts[EV_ROTATING] += 1

    by = state[BALL_Y]
    height = state[FIELD_H]
    if by - radius <= 0 or by + radius >= height:
        state[BALL_Y] = radius if by - radius <= 0 else height - radius
        state[BALL_VY] = -state[BALL_VY]
        speed = math.hypot(state[BALL_VX], state[BALL_VY])
        if speed > params[PR_MAX_SPEED]:
            state[BALL_VX] = state[BALL_VX] / speed * params[PR_MAX_SPEED]
            state[BALL_VY] = state[BALL_VY] / speed * params[PR_MAX_SPEED]
        state[BALL_LAST_HIT] = 0.0
        counts[EV_WALL] += 1
    elif state[BALL_X] - radius <= 0:
        state[SCORE2] += 1
        counts[EV_GOAL] = 2
        return 2
    elif state[BALL_X] + radius >= state[FIELD_W]:
        state[SCORE1] += 1
        counts[EV_GOAL] = 1
        return 1

    _collect_bonuses(state, params, bonuses, rotating, 1, counts, enabled, rand, use_rand)
    _collect_bonuses(state, params, bonuses, rotating, 2, counts, enabled, rand, use_rand)
    return 0


@njit(cache=True)
def _simulate(state, bonuses, rotating, params, actions1, actions2, counts):
    frames = 0
    enabled = params[PR_BONUSES_ENABLED] > 0
    rand = np.zeros(N_RAND, dtype=np.float64)
    for i in range(actions1.shape[0]):
        frames += 1
        goal = step(
            state,
            bonuses,
            rotating,
            params,
            actions1[i, 0],
            actions1[i, 1],
            actions2[i, 0],
            actions2[i, 1],
            counts,
            enabled,
            rand,
            False,
        )
        if goal:
            break
    return frames


def simulate(
    state: np.ndarray,
    bonuses: np.ndarray,
    rotating: np.ndarray,
    params: np.ndarray,
    actions1: np.ndarray,
    actions2: np.ndarray,
) -> SimulationResult:
    """Run explicit per-frame actions on copies of the tables until a goal or the end."""
    actions1 = np.ascontiguousarray(actions1, dtype=np.float64)
    actions2 = np.ascontiguousarray(actions2, dtype=np.float64)
    if actions1.shape != actions2.shape or actions1.ndim != 2 or actions1.shape[1] != 2:
        raise ValueError("actions must be two (frames, 2) arrays of equal length")
    state, bonuses, rotating = state.copy(), bonuses.copy(), rotating.copy()
    counts = np.zeros(N_EVENTS, dtype=np.int64)
    frames = _simulate(state, bonuses, rotating, params, actions1, actions2, counts)
    return SimulationResult(int(frames), counts, state, bonuses, rotating)


# --- teacher policies and values (mirroring magic_pong.ai.physics_teacher) ---
@njit(cache=True)
def _interception(state, params, base, player_id):
    radius = params[PR_RADIUS]
    if player_id == 1:
        contact_x = state[base + PX] + state[base + PW] + radius
    else:
        contact_x = state[base + PX] - radius
    vx = state[BALL_VX]
    height = state[FIELD_H]
    if abs(vx) < 1e-8:
        return height / 2, math.inf
    arrival = (contact_x - state[BALL_X]) / vx
    if arrival < 0:
        return height / 2, math.inf
    span = height - 2 * radius
    unfolded = _pymod(state[BALL_Y] + arrival * state[BALL_VY] - radius, 2 * span)
    return radius + min(unfolded, 2 * span - unfolded), arrival


@njit(cache=True)
def _continuation(state, params, player_id):
    base = P1 if player_id == 1 else P2
    sign = 1.0 if player_id == 1 else -1.0
    incoming = state[BALL_VX] * sign < 0
    if incoming:
        target_y, arrival = _interception(state, params, base, player_id)
        if not math.isfinite(arrival):
            target_y = state[BALL_Y]
        target_x = state[base + PX]
    else:
        target_y = state[FIELD_H] / 2
        target_x = state[base + PMINX] if player_id == 1 else state[base + PMAXX]
    step_size = max(state[base + PSPEED] * params[PR_DT], 1e-8)
    ax = (target_x - state[base + PX]) / step_size
    ay = (target_y - (state[base + PY] + state[base + PH] / 2)) / step_size
    return _clamp(ax, -1.0, 1.0), _clamp(ay, -1.0, 1.0)


@njit(cache=True)
def _position_value(state, params, player_id):
    base = P1 if player_id == 1 else P2
    height = state[FIELD_H]
    centre = state[base + PY] + state[base + PH] / 2
    sign = 1.0 if player_id == 1 else -1.0
    if state[BALL_VX] * sign < 0:
        target, arrival = _interception(state, params, base, player_id)
        gap = max(0.0, abs(target - centre) - state[base + PH] / 2 - params[PR_RADIUS])
        unreachable = max(0.0, gap - state[base + PSPEED] * arrival)
        return -4 * gap / height - 30 * unreachable / height
    home_x = state[base + PMINX] if player_id == 1 else state[base + PMAXX]
    return -2 * abs(centre - height / 2) / height - abs(state[base + PX] - home_x) / state[FIELD_W]


@njit(cache=True)
def _attack_value(state, params, player_id):
    sign = 1.0 if player_id == 1 else -1.0
    if state[BALL_VX] * sign <= 0:
        return 0.0
    other = 3 - player_id
    base = P1 if other == 1 else P2
    target, arrival = _interception(state, params, base, other)
    if not math.isfinite(arrival):
        return 0.0
    distance = abs(target - (state[base + PY] + state[base + PH] / 2))
    unreachable = max(
        0.0,
        distance - state[base + PH] / 2 - params[PR_RADIUS] - state[base + PSPEED] * arrival,
    )
    return 20 * unreachable / state[FIELD_H]


@njit(cache=True)
def _evaluate_actions(
    state,
    bonuses,
    rotating,
    params,
    player_id,
    candidates,
    horizon,
    action_steps,
    y_only,
    values,
    outcomes,
):
    counts = np.zeros(N_EVENTS, dtype=np.int64)
    hit_index = EV_HIT1 + player_id - 1
    bonus_index = EV_BONUS1 + player_id - 1
    enabled = params[PR_BONUSES_ENABLED] > 0
    rand = np.zeros(N_RAND, dtype=np.float64)
    for c in range(candidates.shape[0]):
        index = candidates[c]
        s = state.copy()
        b = bonuses.copy()
        r = rotating.copy()
        ax = ACTION_TABLE[index, 0] if player_id == 1 else -ACTION_TABLE[index, 0]
        ay = ACTION_TABLE[index, 1]
        value = 0.0
        hits = 0
        collections = 0
        goals_for = 0
        goals_against = 0
        finished = False
        for frame in range(horizon):
            if frame < action_steps:
                ox, oy = ax, ay
            else:
                ox, oy = _continuation(s, params, player_id)
            if y_only:
                ox = 0.0
            px, py = _continuation(s, params, 3 - player_id)
            counts[:] = 0
            if player_id == 1:
                goal = step(s, b, r, params, ox, oy, px, py, counts, enabled, rand, False)
            else:
                goal = step(s, b, r, params, px, py, ox, oy, counts, enabled, rand, False)
            hits += counts[hit_index]
            collections += counts[bonus_index]
            if goal:
                if goal == player_id:
                    value += GOAL_VALUE
                    goals_for = 1
                else:
                    value -= GOAL_VALUE
                    goals_against = 1
                finished = True
                break
            value += _position_value(s, params, player_id) / horizon
        if not finished:
            value += 3 * min(hits, 1) + min(collections, 2)
            value += _position_value(s, params, player_id) + _attack_value(s, params, player_id)
        values[index] = value
        outcomes[index, 0] = goals_for
        outcomes[index, 1] = goals_against
        outcomes[index, 2] = hits
        outcomes[index, 3] = collections


def evaluate_actions(
    state: np.ndarray,
    bonuses: np.ndarray,
    rotating: np.ndarray,
    params: np.ndarray,
    player_id: int,
    candidates: np.ndarray,
    horizon: int,
    action_steps: int,
    y_only: bool = False,
) -> tuple[np.ndarray, np.ndarray]:
    """Score candidate first actions with the teacher rollout.

    Returns ``(values, outcomes)`` indexed by action: ``values`` holds ``-inf``
    for actions that were not candidates and ``outcomes`` holds
    ``(goals_for, goals_against, paddle_hits, bonuses)`` per action.
    """
    if player_id not in (1, 2):
        raise ValueError("player_id must be 1 or 2")
    if horizon < 1 or action_steps < 1 or action_steps > horizon:
        raise ValueError("horizon must be >= action_steps >= 1")
    candidates = np.ascontiguousarray(candidates, dtype=np.int64)
    values = np.full(len(ACTION_TABLE), -np.inf, dtype=np.float64)
    outcomes = np.zeros((len(ACTION_TABLE), 4), dtype=np.int64)
    _evaluate_actions(
        state,
        bonuses,
        rotating,
        params,
        int(player_id),
        candidates,
        int(horizon),
        int(action_steps),
        bool(y_only),
        values,
        outcomes,
    )
    return values, outcomes
