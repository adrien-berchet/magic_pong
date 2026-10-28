"""numba ports of the rule-based opponents, acting on flat physics arrays.

Each policy reproduces the matching class in :mod:`magic_pong.ai.models.simple_ai`
(unit-coordinate arithmetic, paddle centre semantics) so that vectorised training
sees the same opponents as the Python engine. ``tests/test_fast_policies.py``
checks the actions against the Python classes on random states.
"""

from __future__ import annotations

import math

import numpy as np
from numba import njit

from magic_pong.core import fast_physics as fp

POLICY_KINDS: dict[str, int] = {
    "dummy": 0,
    "training_dummy": 1,
    "random": 2,
    "follow_ball": 3,
    "defensive": 4,
    "aggressive": 5,
    "predictive": 6,
    "tracker": 7,
    "striker": 8,
    "continuation": 9,
}
POLICY_NAMES: dict[int, str] = {code: name for name, code in POLICY_KINDS.items()}
NETWORK_OPPONENT = -1
PLANNER_OPPONENT = -2


@njit(cache=True)
def _normalized(dx: float, dy: float, scale: float) -> tuple[float, float]:
    distance = math.sqrt(dx * dx + dy * dy)
    if distance > 0:
        return dx / distance * scale, dy / distance * scale
    return 0.0, 0.0


@njit(cache=True)
def _centre(state: np.ndarray, base: int) -> tuple[float, float]:
    return (
        state[base + fp.PX] + state[base + fp.PW] / 2,
        state[base + fp.PY] + state[base + fp.PH] / 2,
    )


@njit(cache=True)
def builtin_action(
    kind: int,
    state: np.ndarray,
    bonuses: np.ndarray,
    params: np.ndarray,
    side: int,
    rand: np.ndarray,
) -> tuple[float, float]:
    """World-frame action of built-in ``kind`` for the paddle on ``side``."""
    base = fp.P1 if side == 1 else fp.P2
    width = state[fp.FIELD_W]
    height = state[fp.FIELD_H]
    max_speed = params[fp.PR_MAX_SPEED]
    cx, cy = _centre(state, base)
    cxu, cyu = cx / width, cy / height
    bxu, byu = state[fp.BALL_X] / width, state[fp.BALL_Y] / height
    vxu, vyu = state[fp.BALL_VX] / max_speed, state[fp.BALL_VY] / max_speed

    if kind == 0:  # dummy
        return 0.0, 0.0
    if kind == 1:  # training_dummy
        center_x = 0.75 if cxu > 0.5 else 0.25
        time_factor = state[fp.TIME] * 0.5
        target_x = center_x + (20.0 / width) * math.sin(time_factor)
        target_y = 0.5 + (15.0 / height) * math.cos(time_factor * 0.7)
        dx = (target_x - cxu) * 0.02
        dy = (target_y - cyu) * 0.02
        return max(-0.1, min(0.1, dx)), max(-0.1, min(0.1, dy))
    if kind == 2:  # random
        return rand[fp.RAND_OPP_X] * 2.0 - 1.0, rand[fp.RAND_OPP_Y] * 2.0 - 1.0
    if kind == 3:  # follow_ball, aggressiveness 0.8
        return _normalized(bxu - cxu, byu - cyu, 0.8)
    if kind == 4:  # defensive
        target_x = 0.1 if cxu < 0.5 else 0.9
        predicted = byu + vyu * 0.5
        target_y = max(0.1, min(0.9, predicted))
        return _normalized(target_x - cxu, target_y - cyu, 1.0)
    if kind == 5:  # aggressive: nearest bonus within 0.3, else the ball
        target_x, target_y = bxu, byu
        closest = math.inf
        for row in range(bonuses.shape[0]):
            if bonuses[row, fp.BACTIVE] <= 0:
                continue
            bx = bonuses[row, fp.BX] / width
            by = bonuses[row, fp.BY] / height
            distance = math.sqrt((bx - cxu) ** 2 + (by - cyu) ** 2)
            if distance < closest:
                closest = distance
                if distance < 0.3:
                    target_x, target_y = bx, by
        if closest >= 0.3:
            target_x, target_y = bxu, byu
        return _normalized(target_x - cxu, target_y - cyu, 1.0)
    if kind == 6:  # predictive, one second ahead
        px = bxu + vxu
        py = byu + vyu
        if py < 0:
            py = -py
        elif py > 1:
            py = 2 - py
        px = max(0.0, min(1.0, px))
        py = max(0.0, min(1.0, py))
        return _normalized(px - cxu, py - cyu, 1.0)
    if kind == 7 or kind == 8:  # goal-line tracker / edge striker
        step_size = max(state[base + fp.PSPEED] / params[fp.PR_FPS], 1e-9)
        vx = state[fp.BALL_VX]
        approaching = vx < 0 if side == 1 else vx > 0
        if kind == 7:
            target = state[fp.BALL_Y] if approaching else height / 2
        else:
            target = height / 2
            radius = params[fp.PR_RADIUS]
            half_w = params[fp.PR_PADDLE_WIDTH] / 2
            face_x = cx + half_w + radius if side == 1 else cx - half_w - radius
            if abs(vx) >= 1e-9:
                arrival = (face_x - state[fp.BALL_X]) / vx
                if arrival >= 0:
                    span = height - 2 * radius
                    if span <= 0:
                        intercept = height / 2
                    else:
                        unfolded = (state[fp.BALL_Y] + arrival * state[fp.BALL_VY] - radius) % (
                            2 * span
                        )
                        intercept = radius + min(unfolded, 2 * span - unfolded)
                    direction = 1.0 if intercept < height / 2 else -1.0
                    half_h = state[base + fp.PH] / 2
                    target = intercept - direction * 0.85 * half_h
                    target = max(half_h, min(height - half_h, target))
        move_y = max(-1.0, min(1.0, (target - cy) / step_size))
        return (-1.0 if side == 1 else 1.0), move_y
    if kind == 9:  # teacher continuation policy
        return fp._continuation(state, params, side, 0.0)
    return 0.0, 0.0


def policy_code(name: str) -> int:
    """Map an opponent name to its kernel code, or raise for unknown names."""
    if name not in POLICY_KINDS:
        raise ValueError(f"unknown fast policy {name!r}; known: {sorted(POLICY_KINDS)}")
    return POLICY_KINDS[name]


__all__ = [
    "NETWORK_OPPONENT",
    "PLANNER_OPPONENT",
    "POLICY_KINDS",
    "POLICY_NAMES",
    "builtin_action",
    "policy_code",
]

_ = np  # numpy is part of the public surface for callers building rand arrays
