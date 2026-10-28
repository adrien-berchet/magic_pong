"""Wall-reflection geometry context and layout definitions for DQN observations."""

from __future__ import annotations

import math
from dataclasses import dataclass
from typing import Any
from typing import NamedTuple

import numpy as np
from numba import njit

FEATURE_NAMES = (
    "own_face_gap",
    "own_center_y_gap",
    "opponent_face_gap",
    "opponent_center_y_gap",
    "incoming",
    "own_contact_time",
    "own_contact_valid",
    "own_intercept_y",
    "own_signed_miss",
    "own_reachability_margin",
    "own_contact_offset",
    "outgoing",
    "opponent_contact_time",
    "opponent_contact_valid",
    "opponent_intercept_y",
    "opponent_contact_offset",
)
GEOMETRY_CONTEXT_FEATURES = FEATURE_NAMES
GEOMETRY_CONTEXT_HORIZON = 1.5  # seconds (90 frames at 60 FPS)
GEOMETRY_CONTEXT_VERSION = 2
MIN_ABS_NORMALIZED_VX = 1e-6
CONTACT_OFFSET_CLIP = 4.0


@dataclass(frozen=True)
class ContactGeometryConfig:
    """Explicit geometry constants from the pinned game configuration."""

    width: float
    height: float
    max_ball_speed: float
    paddle_width: float
    paddle_height: float
    paddle_speed: float
    ball_radius: float
    horizon_frames: int = 90
    fps: float = 60.0

    @property
    def horizon_seconds(self) -> float:
        return self.horizon_frames / self.fps

    @classmethod
    def from_game_config(cls) -> ContactGeometryConfig:
        from magic_pong.utils.config import game_config

        return cls(
            width=float(game_config.FIELD_WIDTH),
            height=float(game_config.FIELD_HEIGHT),
            max_ball_speed=float(game_config.MAX_BALL_SPEED),
            paddle_width=float(game_config.PADDLE_WIDTH),
            paddle_height=float(game_config.PADDLE_HEIGHT),
            paddle_speed=float(game_config.PADDLE_SPEED),
            ball_radius=float(game_config.BALL_RADIUS),
            horizon_frames=90,
            fps=float(game_config.FPS),
        )


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
    - Geometry context (v7): appended after score if score is enabled (33:49),
      or directly after base (30:46) if score is disabled.
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
        geometry_slice = slice(offset, offset + len(GEOMETRY_CONTEXT_FEATURES))
        offset += len(GEOMETRY_CONTEXT_FEATURES)

    return StateLayout(
        base_slice=base_slice,
        score_slice=score_slice,
        geometry_slice=geometry_slice,
        state_size=offset,
        input_blocks=tuple(blocks),
    )


@njit(cache=True)  # type: ignore[misc]
def write_contact_features_scalar(
    row30: np.ndarray,
    out: np.ndarray,
    offset: int,
    width: float,
    height: float,
    max_ball_speed: float,
    paddle_width: float,
    paddle_height: float,
    paddle_speed: float,
    ball_radius: float,
    horizon_seconds: float,
) -> None:
    """Compute the 16 contact features from a canonical 30D state row in float64 precision.

    Writes 16 float32 values into ``out[offset : offset + 16]``.
    Preserves exact bitwise parity with probe reference arithmetic.
    """
    ball_x = float(row30[0]) * width
    ball_y = float(row30[1]) * height
    vx = float(row30[2]) * max_ball_speed
    vy = float(row30[3]) * max_ball_speed
    own_x = float(row30[4]) * width
    own_y = float(row30[5]) * height
    opponent_x = float(row30[6]) * width
    opponent_y = float(row30[7]) * height
    own_height = float(row30[10]) * paddle_height
    opponent_height = float(row30[11]) * paddle_height

    own_center = own_y + own_height / 2.0
    opponent_center = opponent_y + opponent_height / 2.0

    own_gap = ball_x - (own_x + paddle_width + ball_radius)
    opponent_gap = opponent_x - ball_radius - ball_x

    incoming = 1.0 if float(row30[2]) < -MIN_ABS_NORMALIZED_VX else 0.0
    outgoing = 1.0 if float(row30[2]) > MIN_ABS_NORMALIZED_VX else 0.0

    # Own contact time
    own_on_course = (incoming > 0.0) and (own_gap >= 0.0)
    if own_on_course and (-vx) > 0.0:
        own_seconds = own_gap / (-vx)
        own_valid = 1.0 if own_seconds <= horizon_seconds else 0.0
        own_time = min(own_seconds / horizon_seconds, 1.0)
    else:
        own_seconds = 0.0
        own_valid = 0.0
        own_time = 0.0

    # Opponent contact time
    opp_on_course = (outgoing > 0.0) and (opponent_gap >= 0.0)
    if opp_on_course and vx > 0.0:
        opp_seconds = opponent_gap / vx
        opp_valid = 1.0 if opp_seconds <= horizon_seconds else 0.0
        opp_time = min(opp_seconds / horizon_seconds, 1.0)
    else:
        opp_seconds = 0.0
        opp_valid = 0.0
        opp_time = 0.0

    # Wall reflection
    span = height - 2.0 * ball_radius
    modulus = 2.0 * span

    if own_valid > 0.0:
        raw_y = ball_y + vy * own_seconds - ball_radius
        phase = raw_y - modulus * math.floor(raw_y / modulus)
        own_intercept = ball_radius + (phase if phase <= span else (modulus - phase))
        own_offset = own_intercept - own_center
        half_h = own_height / 2.0
        required_travel = max(abs(own_offset) - half_h - ball_radius, 0.0)
        sign = 1.0 if own_offset > 0.0 else (-1.0 if own_offset < 0.0 else 0.0)
        own_signed_miss = sign * required_travel
        reach_margin = paddle_speed * own_seconds - required_travel
        divisor = max(half_h, ball_radius)
        raw_offset = own_offset / divisor
        own_contact_offset = max(-CONTACT_OFFSET_CLIP, min(CONTACT_OFFSET_CLIP, raw_offset))
    else:
        own_intercept = 0.0
        own_signed_miss = 0.0
        reach_margin = 0.0
        own_contact_offset = 0.0

    if opp_valid > 0.0:
        raw_y = ball_y + vy * opp_seconds - ball_radius
        phase = raw_y - modulus * math.floor(raw_y / modulus)
        opponent_intercept = ball_radius + (phase if phase <= span else (modulus - phase))
        opponent_offset = opponent_intercept - opponent_center
        half_h = opponent_height / 2.0
        divisor = max(half_h, ball_radius)
        raw_offset = opponent_offset / divisor
        opponent_contact_offset = max(-CONTACT_OFFSET_CLIP, min(CONTACT_OFFSET_CLIP, raw_offset))
    else:
        opponent_intercept = 0.0
        opponent_contact_offset = 0.0

    out[offset + 0] = np.float32(own_gap / width)
    out[offset + 1] = np.float32((ball_y - own_center) / height)
    out[offset + 2] = np.float32(opponent_gap / width)
    out[offset + 3] = np.float32((ball_y - opponent_center) / height)
    out[offset + 4] = np.float32(incoming)
    out[offset + 5] = np.float32(own_time)
    out[offset + 6] = np.float32(own_valid)
    out[offset + 7] = np.float32(own_intercept / height)
    out[offset + 8] = np.float32(own_signed_miss / height)
    out[offset + 9] = np.float32(reach_margin / height)
    out[offset + 10] = np.float32(own_contact_offset)
    out[offset + 11] = np.float32(outgoing)
    out[offset + 12] = np.float32(opp_time)
    out[offset + 13] = np.float32(opp_valid)
    out[offset + 14] = np.float32(opponent_intercept / height)
    out[offset + 15] = np.float32(opponent_contact_offset)


@njit(cache=True)  # type: ignore[misc]
def _batch_contact_features_kernel(
    states: np.ndarray,
    out: np.ndarray,
    width: float,
    height: float,
    max_ball_speed: float,
    paddle_width: float,
    paddle_height: float,
    paddle_speed: float,
    ball_radius: float,
    horizon_seconds: float,
) -> None:
    n = states.shape[0]
    for i in range(n):
        write_contact_features_scalar(
            states[i],
            out[i],
            0,
            width,
            height,
            max_ball_speed,
            paddle_width,
            paddle_height,
            paddle_speed,
            ball_radius,
            horizon_seconds,
        )


def compute_contact_features(
    states: np.ndarray, config: ContactGeometryConfig | None = None
) -> np.ndarray:
    """Return finite float32 ``(N, 16)`` contact features from ``(N, 30)`` canonical states."""
    cfg = config if config is not None else ContactGeometryConfig.from_game_config()
    arr = np.asarray(states, dtype=np.float32)
    single = arr.ndim == 1
    if single:
        arr = arr[np.newaxis, :]
    if arr.ndim != 2 or arr.shape[1] < 30:
        raise ValueError(f"Expected canonical states with shape (N, 30), got {arr.shape}")
    if not np.isfinite(arr[:, :30]).all():
        raise ValueError("Canonical states must contain finite values")
    if np.any((arr[:, 10:12] < 0.0) | (arr[:, 10:12] > 4.0)):
        raise ValueError("Paddle height ratios must follow the DQN [0, 4] contract")

    out = np.empty((arr.shape[0], len(GEOMETRY_CONTEXT_FEATURES)), dtype=np.float32)
    _batch_contact_features_kernel(
        arr[:, :30],
        out,
        cfg.width,
        cfg.height,
        cfg.max_ball_speed,
        cfg.paddle_width,
        cfg.paddle_height,
        cfg.paddle_speed,
        cfg.ball_radius,
        cfg.horizon_seconds,
    )
    if not np.isfinite(out).all():
        raise ValueError("Geometry cannot be represented as finite float32 features")
    return out[0] if single else out


def compute_geometry_features(
    *args: Any,
    **kwargs: Any,
) -> np.ndarray | tuple[float, ...]:
    """Compute 16 contact geometry features.

    Supports canonical state inputs (single state array or batch), as well as
    legacy positional keyword arguments.
    """
    if len(args) == 1 and isinstance(args[0], np.ndarray):
        return compute_contact_features(args[0], kwargs.get("config"))
    if "state_30" in kwargs:
        return compute_contact_features(kwargs["state_30"], kwargs.get("config"))

    # Legacy scalar invocation fallback
    cfg = kwargs.get("config") or ContactGeometryConfig.from_game_config()
    fw = float(kwargs.get("field_width", cfg.width))
    fh = float(kwargs.get("field_height", cfg.height))
    ms = float(kwargs.get("max_ball_speed", cfg.max_ball_speed))
    pw = float(kwargs.get("paddle_width", cfg.paddle_width))
    ph = float(kwargs.get("paddle_height", cfg.paddle_height))
    ps = float(kwargs.get("paddle_speed", cfg.paddle_speed))
    br = float(kwargs.get("ball_radius", cfg.ball_radius))
    horizon_s = float(kwargs.get("horizon_seconds", cfg.horizon_seconds))

    row = np.zeros(30, dtype=np.float32)
    if "ball_x_px" in kwargs:
        row[0] = float(kwargs["ball_x_px"]) / fw
        row[1] = float(kwargs["ball_y_px"]) / fh
        row[2] = float(kwargs["vx_px"]) / ms
        row[3] = float(kwargs["vy_px"]) / ms
        row[4] = float(kwargs["own_x_px"]) / fw
        row[5] = float(kwargs["own_y_px"]) / fh
        if "opp_x_px" in kwargs:
            row[6] = float(kwargs["opp_x_px"]) / fw
        else:
            row[6] = (fw - pw) / fw
        row[7] = float(kwargs["opp_y_px"]) / fh
        row[10] = float(kwargs.get("own_height_px", ph)) / ph
        row[11] = float(kwargs.get("opp_height_px", ph)) / ph
    elif len(args) >= 1 and isinstance(args[0], (list, tuple)):
        row = np.asarray(args[0], dtype=np.float32)

    out = np.empty(len(GEOMETRY_CONTEXT_FEATURES), dtype=np.float32)
    write_contact_features_scalar(row, out, 0, fw, fh, ms, pw, ph, ps, br, horizon_s)
    return tuple(float(x) for x in out)


def compute_geometry_features_from_physics(
    physics: Any,
    params: Any,
    side: int,
    horizon_seconds: float = GEOMETRY_CONTEXT_HORIZON,
) -> tuple[float, ...]:
    """Derive canonical v7 geometry features from flat fast_physics and params snapshots."""
    from magic_pong.core.fast_physics import BALL_VX
    from magic_pong.core.fast_physics import BALL_VY
    from magic_pong.core.fast_physics import BALL_X
    from magic_pong.core.fast_physics import BALL_Y
    from magic_pong.core.fast_physics import FIELD_H
    from magic_pong.core.fast_physics import FIELD_W
    from magic_pong.core.fast_physics import P1
    from magic_pong.core.fast_physics import P2
    from magic_pong.core.fast_physics import PH
    from magic_pong.core.fast_physics import PR_MAX_SPEED
    from magic_pong.core.fast_physics import PR_PADDLE_HEIGHT
    from magic_pong.core.fast_physics import PR_PADDLE_SPEED
    from magic_pong.core.fast_physics import PR_PADDLE_WIDTH
    from magic_pong.core.fast_physics import PR_RADIUS
    from magic_pong.core.fast_physics import PX
    from magic_pong.core.fast_physics import PY

    fw = float(physics[FIELD_W])
    fh = float(physics[FIELD_H])
    ms = float(params[PR_MAX_SPEED])
    pw = float(params[PR_PADDLE_WIDTH])
    ph = float(params[PR_PADDLE_HEIGHT])
    ps = float(params[PR_PADDLE_SPEED])
    br = float(params[PR_RADIUS])

    mirror = side == 2
    ball_x = fw - float(physics[BALL_X]) if mirror else float(physics[BALL_X])
    ball_y = float(physics[BALL_Y])
    vx = -float(physics[BALL_VX]) if mirror else float(physics[BALL_VX])
    vy = float(physics[BALL_VY])

    own_idx = P2 if mirror else P1
    opp_idx = P1 if mirror else P2

    own_x = fw - float(physics[own_idx + PX]) - pw if mirror else float(physics[own_idx + PX])
    own_y = float(physics[own_idx + PY])
    opp_x = fw - float(physics[opp_idx + PX]) - pw if mirror else float(physics[opp_idx + PX])
    opp_y = float(physics[opp_idx + PY])

    own_h = float(physics[own_idx + PH])
    opp_h = float(physics[opp_idx + PH])

    row = np.zeros(30, dtype=np.float32)
    row[0] = ball_x / fw
    row[1] = ball_y / fh
    row[2] = vx / ms
    row[3] = vy / ms
    row[4] = own_x / fw
    row[5] = own_y / fh
    row[6] = opp_x / fw
    row[7] = opp_y / fh
    row[10] = min(4.0, max(0.0, own_h / ph))
    row[11] = min(4.0, max(0.0, opp_h / ph))

    out = np.empty(len(GEOMETRY_CONTEXT_FEATURES), dtype=np.float32)
    write_contact_features_scalar(row, out, 0, fw, fh, ms, pw, ph, ps, br, horizon_seconds)
    return tuple(float(x) for x in out)


def compute_geometry_features_from_physics_batch(
    physics: Any,
    params: Any,
    sides: Any,
    horizon_seconds: float = GEOMETRY_CONTEXT_HORIZON,
) -> np.ndarray:
    """Compute (N, 16) v7 geometry features for a batch of physics snapshots."""
    n = len(physics)
    if isinstance(sides, int):
        sides_arr = np.full(n, sides, dtype=np.int32)
    else:
        sides_arr = np.asarray(sides, dtype=np.int32)
    if len(sides_arr) != n:
        raise ValueError("physics and sides length mismatch")
    params_arr = (
        np.asarray(params)
        if np.asarray(params).ndim == 2
        else np.broadcast_to(params, (n, len(params)))
    )

    out = np.zeros((n, len(GEOMETRY_CONTEXT_FEATURES)), dtype=np.float32)
    for i in range(n):
        feats = compute_geometry_features_from_physics(
            physics[i], params_arr[i], int(sides_arr[i]), horizon_seconds=horizon_seconds
        )
        out[i] = feats
    return out
