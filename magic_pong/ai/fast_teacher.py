"""Numba-backed drop-in for :class:`magic_pong.ai.physics_teacher.PhysicsTeacher`."""

from __future__ import annotations

import numpy as np

from magic_pong.ai.models.dqn_ai import ACTION_MAPPING
from magic_pong.ai.physics_teacher import DEFAULT_AIM_OFFSETS
from magic_pong.ai.physics_teacher import RolloutOutcome
from magic_pong.ai.physics_teacher import TeacherDecision
from magic_pong.core import fast_physics
from magic_pong.core.physics import PhysicsEngine


class FastPhysicsTeacher:
    """Evaluate discrete action prefixes with compiled rollouts of the real physics.

    Produces the same values as :class:`PhysicsTeacher` (see the parity tests)
    at a small fraction of the cost, so it can run at gameplay frame rate.
    """

    def __init__(
        self,
        horizon_steps: int = 90,
        action_steps: int = 6,
        y_only: bool = False,
        aim_offsets: tuple[float, ...] = DEFAULT_AIM_OFFSETS,
    ) -> None:
        if horizon_steps < 1 or action_steps < 1 or action_steps > horizon_steps:
            raise ValueError("teacher horizon must be >= action_steps >= 1")
        if not aim_offsets or any(not -1.0 <= float(o) <= 1.0 for o in aim_offsets):
            raise ValueError("aim_offsets must be non-empty fractions in [-1, 1]")
        self.horizon_steps = horizon_steps
        self.action_steps = action_steps
        self.y_only = y_only
        self.aim_offsets = tuple(float(o) for o in aim_offsets)
        self.actions = (0, 1, 2) if y_only else tuple(ACTION_MAPPING)
        self._candidates = np.array(self.actions, dtype=np.int64)
        self._offsets = np.array(self.aim_offsets, dtype=np.float64)

    def label_state(
        self,
        state: np.ndarray,
        bonuses: np.ndarray,
        rotating: np.ndarray,
        params: np.ndarray,
        player_id: int = 1,
    ) -> TeacherDecision:
        """Label a snapshot produced by :func:`magic_pong.core.fast_physics.snapshot`."""
        values, outcomes = fast_physics.evaluate_actions(
            state,
            bonuses,
            rotating,
            params,
            player_id,
            self._candidates,
            self.horizon_steps,
            self.action_steps,
            self.y_only,
            self._offsets,
        )
        # Stable tie-breaking prefers stay, then vertical moves, as the Python teacher.
        action = max(self.actions, key=lambda index: values[index])
        rollouts = {
            index: RolloutOutcome(
                float(values[index]),
                int(outcomes[index, 0]),
                int(outcomes[index, 1]),
                int(outcomes[index, 2]),
                int(outcomes[index, 3]),
            )
            for index in self.actions
        }
        return TeacherDecision(action, values.astype(np.float32), rollouts)

    def label(self, physics: PhysicsEngine, player_id: int = 1) -> TeacherDecision:
        """Label the live engine without advancing it."""
        if player_id not in (1, 2):
            raise ValueError("player_id must be 1 or 2")
        return self.label_state(*fast_physics.snapshot(physics), player_id=player_id)
