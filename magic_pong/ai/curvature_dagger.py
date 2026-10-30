"""Curvature and bounce-targeted DAgger (Round 3) core utilities.

Provides vectorized geometric barrier evaluations, targeted tier filtering,
hybrid planner-target projection, and replay anchor weighting.
"""

from __future__ import annotations

import numpy as np
import torch
import torch.nn.functional as F

from magic_pong.ai.action_shaping import _AY
from magic_pong.utils.config import game_config

AY_TENSOR = torch.as_tensor(_AY, dtype=torch.float32)


def vectorized_barrier(
    states: np.ndarray,
    margin: float = 20.0,
    field_height: float | None = None,
    base_paddle_height: float | None = None,
    ball_radius: float | None = None,
    paddle_speed: float | None = None,
    fps: float | None = None,
) -> tuple[np.ndarray, np.ndarray]:
    """Vectorized evaluation of the geometric barrier rule matching shape_action bit-for-bit.

    Args:
        states: (N, 46) array of canonical Schema v7 states.
        margin: Protective padding margin inside paddle edge (default 20 px).
        field_height: Optional override for field height.
        base_paddle_height: Optional override for base paddle height.
        ball_radius: Optional override for ball radius.
        paddle_speed: Optional override for paddle speed.
        fps: Optional override for FPS.

    Returns:
        (active, vy):
            active: (N,) bool array indicating if the barrier requires vertical override.
            vy: (N,) float array in {-1.0, 0.0, 1.0} indicating required direction.
    """
    h_field = float(field_height if field_height is not None else game_config.FIELD_HEIGHT)
    base_h = float(
        base_paddle_height if base_paddle_height is not None else game_config.PADDLE_HEIGHT
    )
    r = float(ball_radius if ball_radius is not None else game_config.BALL_RADIUS)
    speed = float(paddle_speed if paddle_speed is not None else game_config.PADDLE_SPEED)
    f_rate = float(fps if fps is not None else game_config.FPS)
    step = float(speed / f_rate)

    inc = states[:, 34] > 0.5
    val = inc & (states[:, 36] > 0.5)

    half_h = states[:, 10] * base_h / 2.0
    cy = states[:, 5] * h_field + half_h
    dist = states[:, 37] * h_field - cy
    w = np.maximum(half_h + r - margin, step)

    active = val & (np.abs(dist) > w)
    vy = np.where(dist > w, 1.0, np.where(dist < -w, -1.0, 0.0))
    return active, vy


def filter_targeted_tiers(
    states: np.ndarray,
    values: np.ndarray,
    goal_against_flags: np.ndarray,
    ga_actions: np.ndarray,
    field_width: float = 800.0,
    field_height: float = 600.0,
    max_ball_speed: float = 900.0,
    ball_radius: float = 8.0,
    hit_against_flags: np.ndarray | None = None,
) -> tuple[np.ndarray, np.ndarray]:
    """Filter match frames into targeted trouble states (Tiers 1, 2, 3) and contrastive base states.

    Tiers:
      - Tier 1: High-urgency incoming intercept (|v| >= 500 px/s or |vx| >= 400 px/s,
                t <= 60 frames, paddle distance to intercept > 20 px).
      - Tier 2: Conceded shot trajectory (whole sequence between opponent hit and conceded goal).
      - Tier 3: Wall-flip / bounce targeting (unfolded arrival outside [r, H-r] at paddle plane).

    Inevitable conceded frames (where all 9 teacher actions concede) and NaN values are excluded.

    Returns:
        (tier_mask, base_mask):
            tier_mask: (N,) bool array of retained targeted trouble states.
            base_mask: (N,) bool array of non-incoming contrastive states (~incoming).
    """
    n_frames = len(states)
    valid_values = ~np.isnan(values).any(axis=-1)

    inc = states[:, 34] > 0.5
    val = inc & (states[:, 36] > 0.5)

    sp = np.hypot(states[:, 2], states[:, 3]) * max_ball_speed
    vx = states[:, 2] * max_ball_speed
    t_frames = states[:, 35] * 90.0

    half_h = states[:, 10] * 80.0 / 2.0
    cy = states[:, 5] * field_height + half_h
    dist = states[:, 37] * field_height - cy

    # Tier 1: High urgency and spatial miscalibration
    t1 = (
        inc
        & val
        & ((sp >= 500.0) | (np.abs(vx) >= 400.0))
        & (t_frames <= 60.0)
        & (np.abs(dist) > 20.0)
    )

    # Tier 2: Conceded shot frames
    t2 = np.zeros(n_frames, dtype=bool)
    conceded_indices = np.flatnonzero(goal_against_flags)
    for g_idx in conceded_indices:
        if hit_against_flags is not None:
            # Look backwards for the opponent's hit that initiated this conceded shot
            prior_hits = np.flatnonzero(hit_against_flags[: g_idx + 1])
            start_idx = int(prior_hits[-1]) if len(prior_hits) > 0 else max(0, g_idx - 60)
        else:
            start_idx = max(0, g_idx - 55)
        t2[start_idx : g_idx + 1] = True

    # Tier 3: Wall-flip / multi-bounce (unfolded y outside [r, H-r] at paddle plane)
    tsec = states[:, 30] * field_width / np.maximum(-states[:, 2] * max_ball_speed, 1e-6)
    unf = states[:, 1] * field_height + states[:, 3] * max_ball_speed * tsec
    t3 = inc & val & ((unf < ball_radius) | (unf > field_height - ball_radius))

    # All actions concede filter (drop frames with zero defensive solution)
    ga_all = ga_actions.all(axis=-1) if ga_actions.ndim == 2 else np.zeros(n_frames, dtype=bool)

    tier_mask = (t1 | t2 | t3) & valid_values & ~ga_all
    base_mask = ~inc & valid_values & ~ga_all

    return tier_mask, base_mask


def project_planner_target(
    values: torch.Tensor,
    states: torch.Tensor,
    frozen_logits: torch.Tensor,
    temperature: float = 0.03,
) -> tuple[torch.Tensor, int]:
    """Project teacher rollout targets onto valid barrier vertical actions when active.

    Args:
        values: (N, 9) planner action values.
        states: (N, 46) Schema v7 canonical state tensors.
        frozen_logits: (N, 9) frozen promoted baseline logits.
        temperature: Softmax temperature for teacher values.

    Returns:
        (target_distribution, fallback_count):
            target_distribution: (N, 9) probability distribution.
            fallback_count: Number of active frames where teacher mass was < 1e-3
                            and fell back to masked network logits.
    """
    soft = torch.softmax(values / temperature, dim=1)
    chosen = F.one_hot(values.argmax(dim=1), 9).to(soft.dtype)
    base_targ = 0.9 * soft + 0.1 * chosen

    active_np, vy_np = vectorized_barrier(states.detach().cpu().numpy())
    active = torch.as_tensor(active_np, device=values.device)
    vy = torch.as_tensor(vy_np, device=values.device, dtype=torch.float32)

    ay = AY_TENSOR.to(values.device)
    allow = ay[None, :] == vy[:, None]

    mass = (base_targ * allow).sum(dim=1, keepdim=True)
    proj = torch.softmax(frozen_logits.masked_fill(~allow, -1e9), dim=1)

    t_proj = torch.where(mass > 1e-3, base_targ * allow / mass.clamp_min(1e-9), proj)
    final_targ = torch.where(active[:, None], t_proj, base_targ)

    fallback_count = int((active & (mass.squeeze(1) <= 1e-3)).sum().item())
    return final_targ, fallback_count


def compute_concession_per_hit(goals_against: int, opp_paddle_hits: int) -> float:
    """Calculate concede rate per opponent paddle hit."""
    return float(goals_against) / max(1, opp_paddle_hits)
