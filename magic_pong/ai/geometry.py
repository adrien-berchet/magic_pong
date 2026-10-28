"""Wall-reflection geometry context and layout definitions for DQN observations."""

from __future__ import annotations

import math
from typing import NamedTuple

GEOMETRY_CONTEXT_FEATURES = (
    "predicted_intercept_y",
    "time_to_intercept",
    "own_paddle_vel_y",
    "opponent_paddle_vel_y",
)
GEOMETRY_CONTEXT_HORIZON = 1.5  # seconds (90 frames at 60 FPS)
GEOMETRY_CONTEXT_VERSION = 1
APPROACH_MIN_VX = 1e-8


class StateLayout(NamedTuple):
    base_slice: slice
    score_slice: slice | None
    geometry_slice: slice | None
    state_size: int
    input_blocks: tuple[str, ...]


def get_state_layout(score_context: bool = False, geometry_context: bool = False) -> StateLayout:
    """Return canonical slice boundaries, input_blocks, and total state_size.

    Block ordering follows append-only history:
    - Base: 0:30
    - Score context (v5): 30:33 (if enabled)
    - Geometry context (v6): appended after score if score is enabled (33:37),
      or directly after base (30:34) if score is disabled.
    """
    blocks: list[str] = ["base"]
    offset = 30
    base_slice = slice(0, 30)

    score_slice: slice | None = None
    if score_context:
        blocks.append("score_context")
        score_slice = slice(offset, offset + 3)
        offset += 3

    geometry_slice: slice | None = None
    if geometry_context:
        blocks.append("geometry_context")
        geometry_slice = slice(offset, offset + 4)
        offset += 4

    return StateLayout(
        base_slice=base_slice,
        score_slice=score_slice,
        geometry_slice=geometry_slice,
        state_size=offset,
        input_blocks=tuple(blocks),
    )


def compute_geometry_features(
    ball_x_px: float,
    ball_y_px: float,
    vx_px: float,
    vy_px: float,
    own_x_px: float,
    own_y_px: float,
    own_prev_y_px: float,
    opp_y_px: float,
    opp_prev_y_px: float,
    own_height_px: float,
    field_width: float,
    field_height: float,
    paddle_width: float,
    ball_radius: float,
    paddle_speed: float,
    frame_dt: float,
    horizon_seconds: float = GEOMETRY_CONTEXT_HORIZON,
) -> tuple[float, float, float, float]:
    """Compute the 4 canonical geometry features in pixel space using pure math.

    Operates in the canonical (player-1 / left paddle) frame.
    Returns:
        (predicted_intercept_y, time_to_intercept, own_paddle_vel_y, opponent_paddle_vel_y)
    """
    contact_x = own_x_px + paddle_width + ball_radius
    centre_px = own_y_px + own_height_px / 2.0
    sentinel_y = centre_px / field_height
    sentinel_time = 1.0

    approaching = (vx_px < -APPROACH_MIN_VX) and (ball_x_px > contact_x)

    if not approaching:
        predicted_intercept_y = sentinel_y
        time_to_intercept = sentinel_time
    else:
        arrival = (contact_x - ball_x_px) / vx_px
        span = field_height - 2.0 * ball_radius
        if span <= 0.0:
            predicted_intercept_y = sentinel_y
            time_to_intercept = sentinel_time
        else:
            raw_y = ball_y_px + arrival * vy_px - ball_radius
            modulus = 2.0 * span
            phase = raw_y - modulus * math.floor(raw_y / modulus)
            intercept_y = ball_radius + (phase if phase <= span else (modulus - phase))
            predicted_intercept_y = intercept_y / field_height
            time_to_intercept = min(arrival / horizon_seconds, 1.0)

    if frame_dt <= 0.0 or paddle_speed <= 0.0:
        own_vel = 0.0
        opp_vel = 0.0
    else:
        disp_max = paddle_speed * frame_dt
        own_vel = max(-1.0, min(1.0, (own_y_px - own_prev_y_px) / disp_max))
        opp_vel = max(-1.0, min(1.0, (opp_y_px - opp_prev_y_px) / disp_max))

    return (
        float(predicted_intercept_y),
        float(time_to_intercept),
        float(own_vel),
        float(opp_vel),
    )
