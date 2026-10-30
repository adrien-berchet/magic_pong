"""Inference-time geometric barrier and continuous action shaping for Magic Pong DQNAgent."""

from __future__ import annotations

from dataclasses import dataclass

import numpy as np

from magic_pong.ai.geometry import FEATURE_NAMES
from magic_pong.ai.geometry import StateLayout
from magic_pong.core.entities import Action
from magic_pong.utils.config import game_config

_AX = np.array([0.0, 0.0, 0.0, -1.0, 1.0, -1.0, 1.0, -1.0, 1.0], dtype=np.float64)
_AY = np.array([0.0, -1.0, 1.0, 0.0, 0.0, -1.0, -1.0, 1.0, 1.0], dtype=np.float64)


@dataclass(frozen=True)
class GeometricShieldConfig:
    """Configuration for inference-time geometric barrier and soft-x shaping.

    Attributes:
        margin: Defensive margin in pixels subtracted from half_height + ball_radius.
            Clamped so that the capture half-width w >= STEP.
        soft_x_tau: Softmax temperature for continuous x-axis action interpolation.
            If None, argmax x-action is used.
    """

    margin: float = 20.0
    soft_x_tau: float | None = 0.5

    def __post_init__(self) -> None:
        if self.margin < 0.0:
            raise ValueError(f"margin must be non-negative, got {self.margin}")
        if self.soft_x_tau is not None and self.soft_x_tau <= 0.0:
            raise ValueError(f"soft_x_tau must be positive, got {self.soft_x_tau}")


def shape_action(
    q_values: np.ndarray,
    state: np.ndarray,
    layout: StateLayout,
    config: GeometricShieldConfig,
) -> Action:
    """Shape the discrete DQN policy into a continuous Action using a geometric barrier and soft-x.

    Args:
        q_values: 1D numpy array of length 9 containing network output values.
        state: Canonical state vector (e.g. 46D without score context, 49D with score context).
        layout: StateLayout instance defining slices.
        config: GeometricShieldConfig specifying margin and soft_x_tau.

    Returns:
        Action: Continuous action with move_x and move_y in [-1.0, 1.0].
    """
    if layout.geometry_slice is None:
        raise ValueError("Cannot shape action without geometry context in layout")

    argmax_idx = int(np.argmax(q_values))
    argmax_vx = float(_AX[argmax_idx])
    argmax_vy = float(_AY[argmax_idx])

    # 1. Soft-x calculation
    if config.soft_x_tau is not None:
        p = np.exp((q_values - q_values.max()) / config.soft_x_tau)
        p /= p.sum()
        vx = float((p * _AX).sum())
    else:
        vx = argmax_vx

    # 2. Geometric barrier calculation
    geom_start = layout.geometry_slice.start
    idx_incoming = geom_start + FEATURE_NAMES.index("incoming")
    idx_valid = geom_start + FEATURE_NAMES.index("own_contact_valid")
    idx_intercept = geom_start + FEATURE_NAMES.index("own_intercept_y")

    # In base state (slice 0:30):
    # state[5]: player_pos[1] / field_height (top of paddle)
    # state[10]: player_paddle_size (height multiplier, base is PADDLE_HEIGHT = 80 px)
    h_field = float(game_config.FIELD_HEIGHT)
    base_paddle_height = float(game_config.PADDLE_HEIGHT)
    ball_radius = float(game_config.BALL_RADIUS)
    step = float(game_config.PADDLE_SPEED / game_config.FPS)

    half_height = float(state[10]) * base_paddle_height / 2.0
    paddle_center_y = float(state[5]) * h_field + half_height

    is_incoming = state[idx_incoming] > 0.5
    is_valid = is_incoming and (state[idx_valid] > 0.5)

    if is_valid:
        intercept_y = float(state[idx_intercept]) * h_field
        distance_to_intercept = intercept_y - paddle_center_y
        capture_width = max(half_height + ball_radius - config.margin, step)

        if distance_to_intercept > capture_width:
            vy = 1.0
        elif distance_to_intercept < -capture_width:
            vy = -1.0
        else:
            vy = argmax_vy
    else:
        vy = argmax_vy

    return Action(move_x=vx, move_y=vy)
