"""Bounded simulator lookahead used only to produce pretraining action labels.

Actual collisions, temporary effects, collections and goals are simulated. A
reflection/reachability estimate evaluates unfinished trajectories at the
horizon; it is a heuristic, not an oracle or a Bellman target.
"""

import math
import random
from copy import deepcopy
from dataclasses import dataclass

import numpy as np

from magic_pong.ai.agent_adapter import from_player1_dqn_action
from magic_pong.ai.models.dqn_ai import ACTION_MAPPING
from magic_pong.core.entities import Action
from magic_pong.core.entities import Paddle
from magic_pong.core.physics import PhysicsEngine
from magic_pong.utils.config import game_config

DEFAULT_AIM_OFFSETS: tuple[float, ...] = (0.0, 0.85, -0.85)
ATTACK_PRESSURE_WEIGHT = 10.0


@dataclass(frozen=True)
class RolloutOutcome:
    value: float
    goals_for: int = 0
    goals_against: int = 0
    paddle_hits: int = 0
    bonuses: int = 0


@dataclass(frozen=True)
class TeacherDecision:
    action: int
    values: np.ndarray
    outcomes: dict[int, RolloutOutcome]


def interception(physics: PhysicsEngine, paddle: Paddle) -> tuple[float, float]:
    """Predict wall-reflected Y and arrival time at the paddle's front face."""
    ball = physics.ball
    contact_x = (
        paddle.position.x + paddle.width + ball.radius
        if paddle.player_id == 1
        else paddle.position.x - ball.radius
    )
    if abs(ball.velocity.x) < 1e-8:
        return physics.field_height / 2, math.inf
    arrival = (contact_x - ball.position.x) / ball.velocity.x
    if arrival < 0:
        return physics.field_height / 2, math.inf
    span = physics.field_height - 2 * ball.radius
    unfolded = (ball.position.y + arrival * ball.velocity.y - ball.radius) % (2 * span)
    return ball.radius + min(unfolded, 2 * span - unfolded), arrival


def continuation_action(
    physics: PhysicsEngine, player_id: int, dt: float, aim_offset: float = 0.0
) -> Action:
    """Track an approaching ball; otherwise recover to a central home guard.

    ``aim_offset`` (fraction of the half paddle height) shifts the paddle centre
    away from the intercept so the ball strikes nearer an edge; the bounce normal
    then steers the return toward the far half of the field.
    """
    paddle: Paddle = getattr(physics, f"player{player_id}")
    incoming = physics.ball.velocity.x * (1 if player_id == 1 else -1) < 0
    if incoming:
        target_y, arrival = interception(physics, paddle)
        if not math.isfinite(arrival):
            target_y = physics.ball.position.y
        elif aim_offset:
            half = paddle.height / 2
            target_y = max(half, min(physics.field_height - half, target_y - aim_offset * half))
        target_x = paddle.position.x
    else:
        target_y = physics.field_height / 2
        target_x = paddle.min_x if player_id == 1 else paddle.max_x
    step = max(paddle.speed * dt, 1e-8)
    return Action(
        (target_x - paddle.position.x) / step,
        (target_y - (paddle.position.y + paddle.height / 2)) / step,
    )


class PhysicsTeacher:
    """Compare discrete action prefixes followed by a defensive continuation.

    Candidate branches share random draws and never advance the live simulator.
    Each decision costs at most ``9 * horizon_steps`` ordinary physics updates.
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
        self.actions = (0, 1, 2) if y_only else tuple(ACTION_MAPPING)
        self.y_only = y_only
        self.aim_offsets = tuple(float(o) for o in aim_offsets)

    def _position_value(self, physics: PhysicsEngine, player_id: int) -> float:
        paddle: Paddle = getattr(physics, f"player{player_id}")
        ball = physics.ball
        height = physics.field_height
        center = paddle.position.y + paddle.height / 2
        incoming = ball.velocity.x * (1 if player_id == 1 else -1) < 0
        if incoming:
            target, arrival = interception(physics, paddle)
            gap = max(0.0, abs(target - center) - paddle.height / 2 - ball.radius)
            unreachable = max(0.0, gap - paddle.speed * arrival)
            # Close the gap early, while strongly penalizing trajectories that
            # can no longer be intercepted within the paddle's speed limits.
            return -4 * gap / height - 30 * unreachable / height
        home_x = paddle.min_x if player_id == 1 else paddle.max_x
        return (
            -2 * abs(center - height / 2) / height
            - abs(paddle.position.x - home_x) / physics.field_width
        )

    def _attack_value(self, physics: PhysicsEngine, player_id: int) -> float:
        """Reward returns that make the opponent travel, up to unreachable ones.

        ``pressure`` is the fraction of the opponent's travel capacity the return
        consumes (1 = it cannot arrive in time); the unreachable margin adds on top.
        """
        if physics.ball.velocity.x * (1 if player_id == 1 else -1) <= 0:
            return 0.0
        opponent: Paddle = getattr(physics, f"player{3 - player_id}")
        target, arrival = interception(physics, opponent)
        if not math.isfinite(arrival):
            return 0.0
        distance = abs(target - (opponent.position.y + opponent.height / 2))
        required = max(0.0, distance - opponent.height / 2 - physics.ball.radius)
        capacity = opponent.speed * arrival
        if capacity > 0:
            pressure = min(1.0, required / capacity)
        else:
            pressure = 1.0 if required > 0 else 0.0
        unreachable = max(0.0, required - capacity)
        return ATTACK_PRESSURE_WEIGHT * pressure + 20 * unreachable / physics.field_height

    def _rollout(
        self,
        physics: PhysicsEngine,
        player_id: int,
        action_index: int,
        aim_offset: float = 0.0,
    ) -> RolloutOutcome:
        dt = game_config.GAME_SPEED_MULTIPLIER / game_config.FPS
        action = from_player1_dqn_action(ACTION_MAPPING[action_index], player_id)
        value = 0.0
        hits = 0
        collections = 0
        attack: float | None = None
        for frame in range(self.horizon_steps):
            own_action = (
                action
                if frame < self.action_steps
                else continuation_action(physics, player_id, dt, aim_offset)
            )
            if self.y_only:
                own_action = Action(0.0, own_action.move_y)
            opponent_action = continuation_action(physics, 3 - player_id, dt)
            actions = (
                (own_action, opponent_action) if player_id == 1 else (opponent_action, own_action)
            )
            events = physics.update(dt, *actions)
            hits += sum(event["player"] == player_id for event in events["paddle_hits"])
            collections += sum(event["player"] == player_id for event in events["bonus_collected"])
            if hits and attack is None:
                # Judge the return where the opponent stands when the ball leaves the paddle.
                attack = self._attack_value(physics, player_id)
            for event in events["goals"]:
                won = event["player"] == player_id
                return RolloutOutcome(
                    value + (1000 if won else -1000),
                    int(won),
                    int(not won),
                    hits,
                    collections,
                )
            # A small path cost distinguishes prompt recovery from postponing
            # movement until the continuation policy catches up.
            value += self._position_value(physics, player_id) / self.horizon_steps
        value += 3 * min(hits, 1) + min(collections, 2)
        value += self._position_value(physics, player_id)
        value += attack if attack is not None else self._attack_value(physics, player_id)
        return RolloutOutcome(value, paddle_hits=hits, bonuses=collections)

    def label(self, physics: PhysicsEngine, player_id: int = 1) -> TeacherDecision:
        if player_id not in (1, 2):
            raise ValueError("player_id must be 1 or 2")
        rng_state = random.getstate()
        outcomes: dict[int, RolloutOutcome] = {}
        try:
            for action in self.actions:
                for offset in self.aim_offsets:
                    random.setstate(rng_state)
                    outcome = self._rollout(deepcopy(physics), player_id, action, offset)
                    # Keep the best aim per first action; ties prefer the first offset.
                    if action not in outcomes or outcome.value > outcomes[action].value:
                        outcomes[action] = outcome
        finally:
            random.setstate(rng_state)
        values = np.full(len(ACTION_MAPPING), -np.inf, dtype=np.float32)
        for action, outcome in outcomes.items():
            values[action] = outcome.value
        # Stable tie-breaking prefers stay, then vertical moves, at boundaries.
        action = max(self.actions, key=lambda index: outcomes[index].value)
        return TeacherDecision(action, values, outcomes)
