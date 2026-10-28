"""Lookahead player that plans on compiled rollouts of the live physics engine."""

from __future__ import annotations

from typing import Any

from magic_pong.ai.agent_adapter import from_player1_dqn_action
from magic_pong.ai.interface import AIPlayer
from magic_pong.ai.models.dqn_ai import ACTION_MAPPING
from magic_pong.ai.physics_teacher import DEFAULT_AIM_OFFSETS
from magic_pong.ai.physics_teacher import PhysicsTeacher
from magic_pong.ai.physics_teacher import TeacherDecision
from magic_pong.core.entities import Action
from magic_pong.core.physics import PhysicsEngine


def make_teacher(
    horizon_steps: int = 90,
    action_steps: int = 6,
    y_only: bool = False,
    backend: str = "auto",
    aim_offsets: tuple[float, ...] = DEFAULT_AIM_OFFSETS,
) -> Any:
    """Return the compiled teacher when numba is available, else the Python one.

    ``backend`` is ``"auto"``, ``"fast"`` (require numba) or ``"python"``.
    """
    if backend not in ("auto", "fast", "python"):
        raise ValueError("backend must be 'auto', 'fast' or 'python'")
    if backend != "python":
        try:
            from magic_pong.ai.fast_teacher import FastPhysicsTeacher

            return FastPhysicsTeacher(horizon_steps, action_steps, y_only, aim_offsets)
        except ImportError:
            if backend == "fast":
                raise
    return PhysicsTeacher(horizon_steps, action_steps, y_only, aim_offsets)


class PlannerAI(AIPlayer):
    """Choose the discrete action whose short rollout scores best on the real physics.

    The engine binds the live :class:`PhysicsEngine` through ``bind_physics``
    (see :meth:`magic_pong.core.game_engine.GameEngine.set_players`); rollouts
    never advance it. ``decision_interval`` repeats each chosen action for that
    many frames, trading reaction time for planning cost.

    ``action_steps`` is how long each candidate action is held before the
    defensive continuation policy takes over inside a rollout. With a short
    commitment (6 frames) the continuation repairs any first action, all nine
    candidates tie and the planner only moves at the last moment; 30 frames
    (half a second) make the choice of direction matter, which both plays
    better and yields learnable labels for distillation.
    """

    uses_player1_dqn_frame = False

    def __init__(
        self,
        name: str = "PlannerAI",
        horizon_steps: int = 90,
        action_steps: int = 30,
        decision_interval: int = 1,
        backend: str = "auto",
        aim_offsets: tuple[float, ...] = DEFAULT_AIM_OFFSETS,
        **kwargs: Any,
    ) -> None:
        super().__init__(name=name, **kwargs)
        if decision_interval < 1:
            raise ValueError("decision_interval must be positive")
        self.teacher = make_teacher(
            horizon_steps, action_steps, backend=backend, aim_offsets=aim_offsets
        )
        self.decision_interval = decision_interval
        self.physics: PhysicsEngine | None = None
        self.player_id: int | None = None
        self.last_decision: TeacherDecision | None = None
        self._frames_since_decision = 0
        self._current_action = Action(0.0, 0.0)

    def bind_physics(self, physics: PhysicsEngine, player_id: int) -> None:
        """Attach the live engine and the side this player controls."""
        if player_id not in (1, 2):
            raise ValueError("player_id must be 1 or 2")
        self.physics = physics
        self.player_id = player_id
        self._frames_since_decision = 0

    def on_episode_start(self) -> None:
        super().on_episode_start()
        self._frames_since_decision = 0
        self._current_action = Action(0.0, 0.0)

    def get_action(self, observation: dict[str, Any] | None) -> Action:
        """Plan (or repeat the last plan) and return a world-frame action."""
        if observation is None:
            return Action(move_x=0.0, move_y=0.0)
        if self.physics is None or self.player_id is None:
            raise RuntimeError(
                "PlannerAI needs the live physics engine; play it through GameEngine or "
                "TrainingManager, which call bind_physics()."
            )
        if self._frames_since_decision % self.decision_interval == 0:
            self.last_decision = self.teacher.label(self.physics, self.player_id)
            self._current_action = from_player1_dqn_action(
                ACTION_MAPPING[self.last_decision.action], self.player_id
            )
            self._frames_since_decision = 0
        self._frames_since_decision += 1
        return self._current_action

    def on_step(
        self,
        observation: dict[str, Any],
        action: Action,
        reward: float,
        done: bool,
        info: dict[str, Any],
    ) -> None:
        """The planner does not learn."""
        self.current_episode_reward += reward
