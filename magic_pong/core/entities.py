"""
Magic Pong game entities: ball, paddles, bonuses
"""

import math
from abc import ABC
from abc import abstractmethod
from dataclasses import dataclass
from enum import Enum
from typing import Any

import numpy as np

from magic_pong.utils.config import game_config


class BonusType(Enum):
    """Available bonus types"""

    ENLARGE_PADDLE = "enlarge_paddle"
    SHRINK_OPPONENT = "shrink_opponent"
    ROTATING_PADDLE = "rotating_paddle"


@dataclass
class Vector2D:
    """Simple 2D vector for positions and velocities"""

    x: float
    y: float

    def __add__(self, other: "Vector2D") -> "Vector2D":
        return Vector2D(self.x + other.x, self.y + other.y)

    def __iadd__(self, other: "Vector2D") -> "Vector2D":
        self.x += other.x
        self.y += other.y
        return self

    def __sub__(self, other: "Vector2D") -> "Vector2D":
        return Vector2D(self.x - other.x, self.y - other.y)

    def __isub__(self, other: "Vector2D") -> "Vector2D":
        self.x -= other.x
        self.y -= other.y
        return self

    def __mul__(self, scalar: float) -> "Vector2D":
        return Vector2D(self.x * scalar, self.y * scalar)

    def __imul__(self, scalar: float) -> "Vector2D":
        self.x *= scalar
        self.y *= scalar
        return self

    def __truediv__(self, scalar: float) -> "Vector2D":
        return Vector2D(self.x / scalar, self.y / scalar)

    def __itruediv__(self, scalar: float) -> "Vector2D":
        self.x /= scalar
        self.y /= scalar
        return self

    def __neg__(self) -> "Vector2D":
        return Vector2D(-self.x, -self.y)

    def magnitude(self) -> float:
        return float(np.linalg.norm([self.x, self.y]))

    def normalize(self) -> "Vector2D":
        mag = self.magnitude()
        if mag == 0:
            return Vector2D(0, 0)
        return Vector2D(self.x / mag, self.y / mag)

    def to_tuple(self) -> tuple[float, float]:
        return (self.x, self.y)

    def copy(self) -> "Vector2D":
        return Vector2D(self.x, self.y)


class Ball:
    """Game ball"""

    def __init__(self, x: float, y: float, vx: float, vy: float):
        self.position = Vector2D(x, y)
        self.velocity = Vector2D(vx, vy)
        # Initialize prev_position slightly behind current position based on velocity
        # This ensures consistent state for first-frame collision detection
        self.prev_position = Vector2D(x - vx * (1 / 60), y - vy * (1 / 60))
        self.radius = game_config.BALL_RADIUS
        self.last_paddle_hit: int | None = None  # To avoid multiple bounces

    def update(self, dt: float) -> None:
        """Updates the ball position"""
        self.prev_position = self.position.copy()
        self.position = self.position + self.velocity * dt

    def reset_to_center(self, direction: int = 1, angle: float | None = None) -> None:
        """Resets the ball to center with a given direction and optional specific angle"""
        self.position = Vector2D(game_config.FIELD_WIDTH / 2, game_config.FIELD_HEIGHT / 2)

        # Use specific angle if provided, otherwise random within controlled range
        if angle is not None:
            # Use the provided angle directly
            ball_angle = angle
        else:
            # Random but controlled direction (±45 degrees)
            ball_angle = np.random.uniform(-math.pi / 4, math.pi / 4)

        speed = game_config.BALL_SPEED
        self.velocity = Vector2D(
            direction * speed * math.cos(ball_angle), speed * math.sin(ball_angle)
        )

    def bounce_vertical(self) -> None:
        """Vertical bounce (top/bottom walls)"""
        self.velocity.y = -self.velocity.y
        self._clamp_speed()

    def bounce_horizontal(self) -> None:
        """Horizontal bounce (paddles)"""
        self.velocity.x = -self.velocity.x
        self._clamp_speed()

    def _clamp_speed(self) -> None:
        """Clamp ball speed to MAX_BALL_SPEED to prevent runaway velocity"""
        speed = self.velocity.magnitude()
        max_speed = game_config.MAX_BALL_SPEED
        if speed > max_speed:
            self.velocity = self.velocity.normalize() * max_speed

    def get_rect(self) -> tuple[float, float, float, float]:
        """Returns the collision circle properties (x, y, width, height)"""
        return (
            self.position.x - self.radius,
            self.position.y - self.radius,
            self.radius * 2,
            self.radius * 2,
        )


class Paddle:
    """Player paddle"""

    def __init__(
        self,
        x: float,
        y: float,
        player_id: int,
        width: float | None = None,
        height: float | None = None,
    ):
        self.position = Vector2D(x, y)
        self.prev_position = Vector2D(x, y)
        self.speed = game_config.PADDLE_SPEED
        self.player_id = player_id
        self.width = width if width is not None else game_config.PADDLE_WIDTH
        self.height = height if height is not None else game_config.PADDLE_HEIGHT
        self.original_height = self.height
        self.size_effect_timer = 0.0

        # Movement limits according to player
        if player_id == 1:  # Left player
            self.min_x = game_config.PADDLE_MARGIN
            self.max_x = game_config.FIELD_WIDTH / 2 - self.width - game_config.PADDLE_MARGIN
        else:  # Right player
            self.min_x = game_config.FIELD_WIDTH / 2 + game_config.PADDLE_MARGIN
            self.max_x = game_config.FIELD_WIDTH - self.width - game_config.PADDLE_MARGIN

        self.min_y = 0.0
        self.max_y = game_config.FIELD_HEIGHT - self.original_height

    def update(self, dt: float) -> None:
        """Updates the paddle (temporary effects)"""
        if self.size_effect_timer > 0:
            self.size_effect_timer -= dt
            if self.size_effect_timer <= 0:
                self.reset_size()

    def constrain_position(self) -> None:
        """Ensures the paddle stays within its movement bounds"""
        self.position.x = max(self.min_x, min(self.max_x, self.position.x))
        self.position.y = max(self.min_y, min(self.max_y, self.position.y))

    def move(self, vx: float, vy: float, dt: float) -> None:
        """Moves the paddle with constraints"""
        self.prev_position = self.position.copy()
        speed = self.speed * dt
        self.position.x = self.position.x + vx * speed
        self.position.y = self.position.y + vy * speed

        # Movement constraints
        self.constrain_position()

    def apply_size_effect(self, multiplier: float, duration: float) -> None:
        """Applies a temporary size effect"""
        new_size = max(
            min(
                self.height * multiplier, self.original_height * 4, game_config.FIELD_HEIGHT * 0.95
            ),
            self.original_height * 0.25,
        )  # Limit min and max size
        self.size_effect_timer = duration

        # Center the paddle vertically after size change
        diff_size = new_size - self.height
        self.position.y -= diff_size / 2  # Center the paddle

        # Readjust Y limits
        self.max_y = game_config.FIELD_HEIGHT - self.height

        self.height = new_size

        # Ensure constraints
        self.constrain_position()

    def reset_size(self) -> None:
        """Resets to normal size"""
        self.height = self.original_height
        self.max_y = game_config.FIELD_HEIGHT - self.height

    def get_rect(self) -> tuple[float, float, float, float]:
        """Returns the collision rectangle properties (x, y, width, height)"""
        return (self.position.x, self.position.y, self.width, self.height)

    def get_previous_rect(self) -> tuple[float, float, float, float]:
        """Returns the previous collision rectangle properties (x, y, width, height)"""
        return (self.prev_position.x, self.prev_position.y, self.width, self.height)


@dataclass
class Action:
    """Player action"""

    move_x: float  # -1.0 to 1.0
    move_y: float  # -1.0 to 1.0

    def __post_init__(self) -> None:
        # Clamp values between -1 and 1
        self.move_x = float(max(-1, min(1, self.move_x)))
        self.move_y = float(max(-1, min(1, self.move_y)))


class Player(ABC):
    """Base interface for all players"""

    def __init__(self, name: str = "Player"):
        self.name = name

    @abstractmethod
    def get_action(self, observation: dict[str, Any] | None) -> Action:
        """
        Returns the action to perform based on the observation

        Args:
            observation: Normalized game state

        Returns:
            Action: Action to perform
        """
        pass


class RotatingPaddle:
    """Rotating paddle (bonus)"""

    def __init__(self, x: float, y: float, player_id: int):
        self.center = Vector2D(x, y)
        self.player_id = player_id
        self.radius = game_config.ROTATING_PADDLE_RADIUS
        self.thickness = game_config.ROTATING_PADDLE_THICKNESS
        self.angle = 0.0
        self.angular_speed = game_config.ROTATING_PADDLE_SPEED
        self.lifetime = game_config.ROTATING_PADDLE_DURATION

    def update(self, dt: float) -> bool:
        """Updates rotation and lifetime"""
        self.angle += self.angular_speed * dt
        self.lifetime -= dt
        return self.lifetime > 0

    def get_line_segments(self) -> list[tuple[Vector2D, Vector2D]]:
        """Returns line segments for collision"""
        # Four segments forming a rotating square
        segments = []
        for i in range(4):
            angle1 = self.angle + i * math.pi / 2
            angle2 = self.angle + (i + 1) * math.pi / 2

            p1 = Vector2D(
                self.center.x + self.radius * math.cos(angle1),
                self.center.y + self.radius * math.sin(angle1),
            )
            p2 = Vector2D(
                self.center.x + self.radius * math.cos(angle2),
                self.center.y + self.radius * math.sin(angle2),
            )
            segments.append((p1, p2))

        return segments


class Bonus:
    """Bonus that appears on the field"""

    def __init__(self, x: float, y: float, bonus_type: BonusType):
        self.position = Vector2D(x, y)
        self.type = bonus_type
        self.size = game_config.BONUS_SIZE
        self.collected = False
        self.lifetime = game_config.BONUS_LIFETIME

    def update(self, dt: float) -> bool:
        """Updates the bonus, returns False if expired"""
        self.lifetime -= dt
        return self.lifetime > 0 and not self.collected

    def collect(self) -> BonusType:
        """Collects the bonus"""
        self.collected = True
        return self.type

    def get_rect(self) -> tuple[float, float, float, float]:
        """Returns the collision rectangle"""
        return (self.position.x, self.position.y, self.size, self.size)


@dataclass
class GameState:
    """Complete game state for AI"""

    ball_position: tuple[float, float]
    ball_velocity: tuple[float, float]
    player1_position: tuple[float, float]
    player2_position: tuple[float, float]
    player1_paddle_size: float
    player2_paddle_size: float
    active_bonuses: list[tuple[float, float, str]]  # position + type
    rotating_paddles: list[tuple[float, float, float]]  # center + angle
    score: tuple[int, int]
    time_elapsed: float
    field_bounds: tuple[float, float, float, float]
