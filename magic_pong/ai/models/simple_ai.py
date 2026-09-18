"""
Simple AI examples for Magic Pong
"""

import math
import random
from dataclasses import dataclass
from typing import Any

from magic_pong.ai.interface import AIPlayer
from magic_pong.core.entities import Action
from magic_pong.utils.config import game_config

BUILTIN_OPPONENTS = (
    "random",
    "dummy",
    "training_dummy",
    "follow_ball",
    "defensive",
    "aggressive",
    "predictive",
    "tracker",
    "striker",
)


@dataclass(frozen=True)
class UnitObservation:
    """Observation values converted to unit-coordinate heuristic space.

    ``player_pos`` is the paddle centre, not the top-left corner reported by the
    raw observation, so heuristics can compare it directly with the ball.
    """

    ball_pos: tuple[float, float]
    player_pos: tuple[float, float]
    ball_vel: tuple[float, float]
    bonuses: tuple[tuple[Any, ...], ...]
    field_width_extent: float
    field_height_extent: float
    paddle_height: float = game_config.PADDLE_HEIGHT / game_config.FIELD_HEIGHT


@dataclass(frozen=True)
class PixelObservation:
    """Observation values in field pixels, for geometry that needs real units."""

    ball_pos: tuple[float, float]
    ball_vel: tuple[float, float]
    paddle_centre: tuple[float, float]
    paddle_height: float
    field_width: float
    field_height: float

    @property
    def side(self) -> int:
        return 1 if self.paddle_centre[0] < self.field_width / 2 else 2


class UnitObservationMixin:
    """Convert raw or normalized observation dictionaries into unit coordinates."""

    def _unit_observation(self, observation: dict[str, Any]) -> UnitObservation:
        field_width = _field_extent(observation.get("field_width"), game_config.FIELD_WIDTH)
        field_height = _field_extent(observation.get("field_height"), game_config.FIELD_HEIGHT)

        ball_pos = observation["ball_pos"]
        player_pos = observation["player_pos"]
        ball_vel = observation.get("ball_vel", [0.0, 0.0])
        normalized = observation.get("positions_normalized")
        max_ball_speed = _field_extent(
            observation.get("max_ball_speed"), game_config.MAX_BALL_SPEED
        )
        paddle_height = _paddle_height(
            observation.get("player_paddle_size"), game_config.PADDLE_HEIGHT
        )
        top_left = _unit_xy_position(player_pos, field_width, field_height, normalized)

        return UnitObservation(
            ball_pos=_unit_xy_position(ball_pos, field_width, field_height, normalized),
            player_pos=(
                top_left[0] + game_config.PADDLE_WIDTH / 2 / field_width,
                top_left[1] + paddle_height / 2 / field_height,
            ),
            ball_vel=(
                _unit_velocity(float(ball_vel[0]), normalized, max_ball_speed),
                _unit_velocity(float(ball_vel[1]), normalized, max_ball_speed),
            ),
            bonuses=_unit_bonuses(
                observation.get("bonuses", []), field_width, field_height, normalized
            ),
            field_width_extent=field_width,
            field_height_extent=field_height,
            paddle_height=paddle_height / field_height,
        )

    def _pixel_observation(self, observation: dict[str, Any]) -> PixelObservation:
        unit = self._unit_observation(observation)
        width, height = unit.field_width_extent, unit.field_height_extent
        max_ball_speed = _field_extent(
            observation.get("max_ball_speed"), game_config.MAX_BALL_SPEED
        )
        return PixelObservation(
            ball_pos=(unit.ball_pos[0] * width, unit.ball_pos[1] * height),
            ball_vel=(unit.ball_vel[0] * max_ball_speed, unit.ball_vel[1] * max_ball_speed),
            paddle_centre=(unit.player_pos[0] * width, unit.player_pos[1] * height),
            paddle_height=unit.paddle_height * height,
            field_width=width,
            field_height=height,
        )


def _paddle_height(value: Any, fallback: float) -> float:
    if isinstance(value, list | tuple) and len(value) >= 2:
        value = value[1]
    try:
        height = float(value)
    except (TypeError, ValueError):
        return fallback
    if not math.isfinite(height) or height <= 0:
        return fallback
    return height


def predict_intercept_y(
    pixels: PixelObservation, face_x: float | None = None
) -> tuple[float, float] | None:
    """Wall-reflected ball y and arrival time at the paddle face, or None if moving away."""
    radius = game_config.BALL_RADIUS
    ball_x, ball_y = pixels.ball_pos
    vel_x, vel_y = pixels.ball_vel
    if face_x is None:
        half_width = game_config.PADDLE_WIDTH / 2
        face_x = (
            pixels.paddle_centre[0] + half_width + radius
            if pixels.side == 1
            else pixels.paddle_centre[0] - half_width - radius
        )
    if abs(vel_x) < 1e-9:
        return None
    arrival = (face_x - ball_x) / vel_x
    if arrival < 0:
        return None
    span = pixels.field_height - 2 * radius
    if span <= 0:
        return pixels.field_height / 2, arrival
    unfolded = (ball_y + arrival * vel_y - radius) % (2 * span)
    return radius + min(unfolded, 2 * span - unfolded), arrival


def _field_extent(value: Any, fallback: float) -> float:
    try:
        extent = float(value)
    except (TypeError, ValueError):
        return fallback
    if extent <= 0:
        return fallback
    return extent


def _unit_xy_position(
    value: Any, field_width: float, field_height: float, normalized: bool | None = None
) -> tuple[float, float]:
    return (
        _unit_position_axis(float(value[0]), field_width, normalized),
        _unit_position_axis(float(value[1]), field_height, normalized),
    )


def _unit_position_axis(value: float, extent: float, normalized: bool | None = None) -> float:
    if normalized is not None:
        return value if normalized else value / extent
    # Retain magnitude inference only for legacy unmarked observations.
    if extent > 1.0 and abs(value) > 1.0:
        return value / extent
    return value


def _unit_velocity(
    value: float, normalized: bool | None = None, max_ball_speed: float | None = None
) -> float:
    scale = game_config.MAX_BALL_SPEED if max_ball_speed is None else max_ball_speed
    if normalized is not None:
        return value if normalized else value / scale
    if abs(value) > 1.0:
        return value / scale
    return value


def _unit_bonuses(
    bonuses: Any, field_width: float, field_height: float, normalized: bool | None = None
) -> tuple[tuple[Any, ...], ...]:
    unit_bonuses = []
    for bonus in bonuses:
        if not isinstance(bonus, list | tuple) or len(bonus) < 2:
            continue
        unit_bonuses.append(
            (
                _unit_position_axis(float(bonus[0]), field_width, normalized),
                _unit_position_axis(float(bonus[1]), field_height, normalized),
                *tuple(bonus[2:]),
            )
        )
    return tuple(unit_bonuses)


class RandomAI(AIPlayer):
    """AI that plays completely randomly"""

    def __init__(self, name: str = "RandomAI", **kwargs: Any):
        super().__init__(name=name, **kwargs)

    def get_action(self, observation: dict[str, Any] | None) -> Action:
        """Returns a random action"""
        return Action(move_x=random.uniform(-1.0, 1.0), move_y=random.uniform(-1.0, 1.0))

    def on_step(
        self,
        observation: dict[str, Any],
        action: Action,
        reward: float,
        done: bool,
        info: dict[str, Any],
    ) -> None:
        """Random AI doesn't learn"""
        self.current_episode_reward += reward


class DummyAI(AIPlayer):
    """AI that never moves - perfect for Phase 1 training (learning to hit the ball)"""

    def __init__(self, name: str = "DummyAI", **kwargs: Any):
        super().__init__(name=name, **kwargs)

    def get_action(self, observation: dict[str, Any] | None) -> Action:
        """Never moves - stays completely still"""
        return Action(move_x=0.0, move_y=0.0)

    def on_step(
        self,
        observation: dict[str, Any],
        action: Action,
        reward: float,
        done: bool,
        info: dict[str, Any],
    ) -> None:
        """Dummy AI doesn't learn"""
        self.current_episode_reward += reward


class TrainingDummyAI(UnitObservationMixin, AIPlayer):
    """
    Specialized AI for Phase 1 training - moves very minimally and predictably
    to ensure the learning agent can focus purely on ball contact
    """

    def __init__(
        self,
        name: str = "TrainingDummyAI",
        movement_factor: float = 0.02,
        **kwargs: Any,
    ):
        super().__init__(name=name, **kwargs)
        self.movement_factor = movement_factor  # Very small movement to add minimal variation
        self.center_x = 0.0  # Will be set based on player side

    def get_action(self, observation: dict[str, Any] | None) -> Action:
        """Very minimal movement around center position"""
        if observation is None:
            return Action(move_x=0.0, move_y=0.0)

        unit_observation = self._unit_observation(observation)
        player_pos = unit_observation.player_pos
        if player_pos[0] > 0.5:  # Right side
            self.center_x = 0.75
        else:  # Left side
            self.center_x = 0.25

        # Very slow, predictable movement around center position
        time_factor = float(observation.get("time_elapsed", 0.0)) * 0.5
        target_x = self.center_x + (20.0 / unit_observation.field_width_extent) * math.sin(
            time_factor
        )
        target_y = 0.5 + (15.0 / unit_observation.field_height_extent) * math.cos(time_factor * 0.7)

        # Very gentle movement towards target
        dx = (target_x - player_pos[0]) * self.movement_factor
        dy = (target_y - player_pos[1]) * self.movement_factor

        # Clamp movement to be very small
        dx = max(-0.1, min(0.1, dx))
        dy = max(-0.1, min(0.1, dy))

        return Action(move_x=dx, move_y=dy)

    def on_step(
        self,
        observation: dict[str, Any],
        action: Action,
        reward: float,
        done: bool,
        info: dict[str, Any],
    ) -> None:
        """Training dummy doesn't learn"""
        self.current_episode_reward += reward


class FollowBallAI(UnitObservationMixin, AIPlayer):
    """Simple AI that follows the ball"""

    def __init__(self, name: str = "FollowBallAI", aggressiveness: float = 0.8, **kwargs: Any):
        super().__init__(name=name, **kwargs)
        self.aggressiveness = aggressiveness

    def get_action(self, observation: dict[str, Any] | None) -> Action:
        """Follows the ball with a certain aggressiveness"""
        if observation is None:
            return Action(move_x=0.0, move_y=0.0)
        unit_observation = self._unit_observation(observation)
        ball_pos = unit_observation.ball_pos
        player_pos = unit_observation.player_pos

        # Calculate direction towards the ball
        dx = ball_pos[0] - player_pos[0]
        dy = ball_pos[1] - player_pos[1]

        # Normalize
        distance = math.sqrt(dx * dx + dy * dy)
        if distance > 0:
            move_x = (dx / distance) * self.aggressiveness
            move_y = (dy / distance) * self.aggressiveness
        else:
            move_x = 0.0
            move_y = 0.0

        return Action(move_x=move_x, move_y=move_y)

    def on_step(
        self,
        observation: dict[str, Any],
        action: Action,
        reward: float,
        done: bool,
        info: dict[str, Any],
    ) -> None:
        """Simple AI doesn't learn"""
        self.current_episode_reward += reward


class DefensiveAI(UnitObservationMixin, AIPlayer):
    """Defensive AI that stays near its goal"""

    def __init__(self, name: str = "DefensiveAI", **kwargs: Any):
        super().__init__(name=name, **kwargs)

    def get_action(self, observation: dict[str, Any] | None) -> Action:
        """Defensive strategy"""
        if observation is None:
            return Action(move_x=0.0, move_y=0.0)
        unit_observation = self._unit_observation(observation)
        ball_pos = unit_observation.ball_pos
        ball_vel = unit_observation.ball_vel
        player_pos = unit_observation.player_pos

        if player_pos[0] < 0.5:  # Left side
            target_x = 0.1  # Near left edge
        else:  # Right player
            target_x = 0.9  # Near right edge

        # Predict where the ball will go
        predicted_y = ball_pos[1] + ball_vel[1] * 0.5  # Simple prediction
        target_y = max(0.1, min(0.9, predicted_y))  # Clamp

        # Move towards target position
        dx = target_x - player_pos[0]
        dy = target_y - player_pos[1]

        # Normalize
        distance = math.sqrt(dx * dx + dy * dy)
        if distance > 0:
            move_x = dx / distance
            move_y = dy / distance
        else:
            move_x = 0.0
            move_y = 0.0

        return Action(move_x=move_x, move_y=move_y)

    def on_step(
        self,
        observation: dict[str, Any],
        action: Action,
        reward: float,
        done: bool,
        info: dict[str, Any],
    ) -> None:
        """Defensive AI doesn't learn"""
        self.current_episode_reward += reward


class AggressiveAI(UnitObservationMixin, AIPlayer):
    """Aggressive AI that seeks bonuses and attacks"""

    def __init__(self, name: str = "AggressiveAI", **kwargs: Any):
        super().__init__(name=name, **kwargs)
        self.target_bonus = None

    def get_action(self, observation: dict[str, Any] | None) -> Action:
        """Aggressive strategy"""
        if observation is None:
            return Action(move_x=0.0, move_y=0.0)
        unit_observation = self._unit_observation(observation)
        ball_pos = unit_observation.ball_pos
        player_pos = unit_observation.player_pos
        bonuses = unit_observation.bonuses

        # Look for the closest bonus
        closest_bonus = None
        closest_distance = float("inf")

        for bonus in bonuses:
            bonus_x, bonus_y, bonus_type = bonus
            distance = math.sqrt((bonus_x - player_pos[0]) ** 2 + (bonus_y - player_pos[1]) ** 2)
            if distance < closest_distance:
                closest_distance = distance
                closest_bonus = (bonus_x, bonus_y)

        # Decide target
        if closest_bonus and closest_distance < 0.3:  # Close bonus
            target_x, target_y = closest_bonus
        else:
            # Otherwise, go towards the ball
            target_x, target_y = ball_pos

        # Move towards target
        dx = target_x - player_pos[0]
        dy = target_y - player_pos[1]

        # Normalize with maximum aggressiveness
        distance = math.sqrt(dx * dx + dy * dy)
        if distance > 0:
            move_x = dx / distance
            move_y = dy / distance
        else:
            move_x = 0.0
            move_y = 0.0

        return Action(move_x=move_x, move_y=move_y)

    def on_step(
        self,
        observation: dict[str, Any],
        action: Action,
        reward: float,
        done: bool,
        info: dict[str, Any],
    ) -> None:
        """Aggressive AI doesn't learn"""
        self.current_episode_reward += reward


class PredictiveAI(UnitObservationMixin, AIPlayer):
    """AI that tries to predict the ball's trajectory"""

    def __init__(self, name: str = "PredictiveAI", prediction_time: float = 1.0, **kwargs: Any):
        super().__init__(name=name, **kwargs)
        self.prediction_time = prediction_time

    def get_action(self, observation: dict[str, Any] | None) -> Action:
        """Predicts where the ball will be and positions accordingly"""
        if observation is None:
            return Action(move_x=0.0, move_y=0.0)
        unit_observation = self._unit_observation(observation)
        ball_pos = unit_observation.ball_pos
        ball_vel = unit_observation.ball_vel
        player_pos = unit_observation.player_pos

        # Predict ball's future position
        predicted_x = ball_pos[0] + ball_vel[0] * self.prediction_time
        predicted_y = ball_pos[1] + ball_vel[1] * self.prediction_time

        # Handle wall bounces (simple approximation)
        if predicted_y < 0:
            predicted_y = -predicted_y
        elif predicted_y > 1:
            predicted_y = 2 - predicted_y

        # Clamp within field bounds
        predicted_x = max(0, min(1, predicted_x))
        predicted_y = max(0, min(1, predicted_y))

        # Move towards predicted position
        dx = predicted_x - player_pos[0]
        dy = predicted_y - player_pos[1]

        # Normalize
        distance = math.sqrt(dx * dx + dy * dy)
        if distance > 0:
            move_x = dx / distance
            move_y = dy / distance
        else:
            move_x = 0.0
            move_y = 0.0

        return Action(move_x=move_x, move_y=move_y)

    def on_step(
        self,
        observation: dict[str, Any],
        action: Action,
        reward: float,
        done: bool,
        info: dict[str, Any],
    ) -> None:
        """Predictive AI doesn't learn"""
        self.current_episode_reward += reward


class GoalLineTrackerAI(UnitObservationMixin, AIPlayer):
    """Hold the goal line and keep the paddle centre on the ball's current y.

    The paddle outruns any ball slower than ``PADDLE_SPEED`` in the vertical
    direction, so this opponent only concedes to fast, steep returns.
    """

    def __init__(self, name: str = "GoalLineTrackerAI", **kwargs: Any):
        super().__init__(name=name, **kwargs)

    def _target_y(self, pixels: PixelObservation) -> float:
        approaching = pixels.ball_vel[0] < 0 if pixels.side == 1 else pixels.ball_vel[0] > 0
        return pixels.ball_pos[1] if approaching else pixels.field_height / 2

    def get_action(self, observation: dict[str, Any] | None) -> Action:
        """Retreat to the goal line and track the target y with proportional control."""
        if observation is None:
            return Action(move_x=0.0, move_y=0.0)
        pixels = self._pixel_observation(observation)
        step = max(game_config.PADDLE_SPEED / game_config.FPS, 1e-9)
        move_y = (self._target_y(pixels) - pixels.paddle_centre[1]) / step
        move_x = -1.0 if pixels.side == 1 else 1.0
        return Action(move_x=move_x, move_y=max(-1.0, min(1.0, move_y)))

    def on_step(
        self,
        observation: dict[str, Any],
        action: Action,
        reward: float,
        done: bool,
        info: dict[str, Any],
    ) -> None:
        """Tracker doesn't learn"""
        self.current_episode_reward += reward


class EdgeStrikerAI(GoalLineTrackerAI):
    """Goal-line defender that predicts the intercept and returns with the paddle edge.

    Hitting away from the centre tilts the bounce normal, sending the ball
    across to the far half of the field where a chasing opponent is late.
    """

    def __init__(self, name: str = "EdgeStrikerAI", edge_offset: float = 0.85, **kwargs: Any):
        super().__init__(name=name, **kwargs)
        self.edge_offset = max(0.0, min(1.0, edge_offset))

    def _target_y(self, pixels: PixelObservation) -> float:
        prediction = predict_intercept_y(pixels)
        if prediction is None:
            return pixels.field_height / 2
        intercept_y, _ = prediction
        direction = 1.0 if intercept_y < pixels.field_height / 2 else -1.0
        half_height = pixels.paddle_height / 2
        target = intercept_y - direction * self.edge_offset * half_height
        return max(half_height, min(pixels.field_height - half_height, target))


class HumanPlayer:
    """Class to represent a human player (for graphical interface)"""

    def __init__(self, name: str = "Human"):
        self.name = name
        self.current_action = Action(0.0, 0.0)

    def set_action(self, move_x: float, move_y: float) -> None:
        """Sets the current action of the human player"""
        self.current_action = Action(move_x, move_y)

    def get_human_action(self) -> Action:
        """Returns the current action"""
        return self.current_action

    def get_stats(self) -> dict[str, float]:
        """Returns empty stats for compatibility"""
        return {"mean_reward": 0.0, "episodes": 0}


# Factory to easily create AIs
def create_ai(ai_type: str, **kwargs: Any) -> AIPlayer:
    """
    Factory to create AIs

    Args:
        ai_type: One of BUILTIN_OPPONENTS ('random', 'dummy', 'training_dummy', 'follow_ball',
            'defensive', 'aggressive', 'predictive', 'tracker', 'striker')
        **kwargs: Additional arguments for the AI

    Returns:
        AIPlayer: Instance of the requested AI
    """
    ai_classes: dict[str, type[AIPlayer]] = {
        "random": RandomAI,
        "dummy": DummyAI,
        "training_dummy": TrainingDummyAI,
        "follow_ball": FollowBallAI,
        "defensive": DefensiveAI,
        "aggressive": AggressiveAI,
        "predictive": PredictiveAI,
        "tracker": GoalLineTrackerAI,
        "striker": EdgeStrikerAI,
    }

    if ai_type not in ai_classes:
        raise ValueError(f"Unknown AI type: {ai_type}. Available types: {list(ai_classes.keys())}")

    ai_class = ai_classes[ai_type]
    return ai_class(**kwargs)
