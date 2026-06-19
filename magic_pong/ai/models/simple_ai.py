"""
Simple AI examples for Magic Pong
"""

import math
import random
import time
from dataclasses import dataclass
from typing import Any

from magic_pong.ai.interface import AIPlayer
from magic_pong.core.entities import Action
from magic_pong.utils.config import game_config


@dataclass(frozen=True)
class UnitObservation:
    """Observation values converted to unit-coordinate heuristic space."""

    ball_pos: tuple[float, float]
    player_pos: tuple[float, float]
    ball_vel: tuple[float, float]
    bonuses: tuple[tuple[Any, ...], ...]
    field_width_extent: float
    field_height_extent: float


class UnitObservationMixin:
    """Convert raw or normalized observation dictionaries into unit coordinates."""

    def _unit_observation(self, observation: dict[str, Any]) -> UnitObservation:
        field_width = _field_extent(observation.get("field_width"), game_config.FIELD_WIDTH)
        field_height = _field_extent(observation.get("field_height"), game_config.FIELD_HEIGHT)

        ball_pos = observation["ball_pos"]
        player_pos = observation["player_pos"]
        ball_vel = observation.get("ball_vel", [0.0, 0.0])

        return UnitObservation(
            ball_pos=_unit_xy_position(ball_pos, field_width, field_height),
            player_pos=_unit_xy_position(player_pos, field_width, field_height),
            ball_vel=(
                _unit_velocity(float(ball_vel[0])),
                _unit_velocity(float(ball_vel[1])),
            ),
            bonuses=_unit_bonuses(observation.get("bonuses", []), field_width, field_height),
            field_width_extent=field_width,
            field_height_extent=field_height,
        )


def _field_extent(value: Any, fallback: float) -> float:
    try:
        extent = float(value)
    except (TypeError, ValueError):
        return fallback
    if extent <= 0:
        return fallback
    return extent


def _unit_xy_position(value: Any, field_width: float, field_height: float) -> tuple[float, float]:
    return (
        _unit_position_axis(float(value[0]), field_width),
        _unit_position_axis(float(value[1]), field_height),
    )


def _unit_position_axis(value: float, extent: float) -> float:
    if extent > 1.0 and abs(value) > 1.0:
        return value / extent
    return value


def _unit_velocity(value: float) -> float:
    if abs(value) > 1.0:
        return value / game_config.MAX_BALL_SPEED
    return value


def _unit_bonuses(
    bonuses: Any, field_width: float, field_height: float
) -> tuple[tuple[Any, ...], ...]:
    unit_bonuses = []
    for bonus in bonuses:
        if not isinstance(bonus, list | tuple) or len(bonus) < 2:
            continue
        unit_bonuses.append(
            (
                _unit_position_axis(float(bonus[0]), field_width),
                _unit_position_axis(float(bonus[1]), field_height),
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
        time_factor = time.time() * 0.5  # Slow oscillation
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
        ai_type: AI type ('random', 'follow_ball', 'defensive', 'aggressive', 'predictive', 'dummy', 'training_dummy')
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
    }

    if ai_type not in ai_classes:
        raise ValueError(f"Unknown AI type: {ai_type}. Available types: {list(ai_classes.keys())}")

    ai_class = ai_classes[ai_type]
    return ai_class(**kwargs)
