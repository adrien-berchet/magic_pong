"""
Magic Pong game entities: ball, paddles, bonuses
"""

import math
from dataclasses import dataclass
from enum import Enum

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

    def __sub__(self, other: "Vector2D") -> "Vector2D":
        return Vector2D(self.x - other.x, self.y - other.y)

    def __mul__(self, scalar: float) -> "Vector2D":
        return Vector2D(self.x * scalar, self.y * scalar)

    def magnitude(self) -> float:
        return math.sqrt(self.x**2 + self.y**2)

    def normalize(self) -> "Vector2D":
        mag = self.magnitude()
        if mag == 0:
            return Vector2D(0, 0)
        return Vector2D(self.x / mag, self.y / mag)

    def to_tuple(self) -> tuple[float, float]:
        return (self.x, self.y)


class Ball:
    """Game ball"""

    def __init__(self, x: float, y: float, vx: float, vy: float):
        self.position = Vector2D(x, y)
        self.velocity = Vector2D(vx, vy)
        self.prev_position = Vector2D(x, y) - self.velocity * (1 / 60)  # Assume initial dt=1/60
        self.radius = game_config.BALL_RADIUS
        self.last_paddle_hit: int | None = None  # To avoid multiple bounces

    def update(self, dt: float) -> None:
        """Updates the ball position"""
        self.prev_position = Vector2D(self.position.x, self.position.y)
        self.position = self.position + self.velocity * dt

    def reset_to_center(self, direction: int = 1) -> None:
        """Resets the ball to center with a given direction"""
        self.position = Vector2D(game_config.FIELD_WIDTH / 2, game_config.FIELD_HEIGHT / 2)

        # Random but controlled direction
        angle = np.random.uniform(-math.pi / 4, math.pi / 4)  # ±45 degrees
        speed = game_config.BALL_SPEED

        self.velocity = Vector2D(direction * speed * math.cos(angle), speed * math.sin(angle))

    def bounce_vertical(self) -> None:
        """Vertical bounce (top/bottom walls)"""
        self.velocity.y = -self.velocity.y

    def bounce_horizontal(self) -> None:
        """Horizontal bounce (paddles)"""
        self.velocity.x = -self.velocity.x
        # Slight acceleration after each bounce
        # speed_increase = game_config.BALL_SPEED_INCREASE
        # self.velocity = self.velocity * speed_increase


class Paddle:
    """Player paddle"""

    def __init__(self, x: float, y: float, player_id: int):
        self.position = Vector2D(x, y)
        self.prev_position = Vector2D(x, y)
        self.player_id = player_id
        self.width = game_config.PADDLE_WIDTH
        self.height = game_config.PADDLE_HEIGHT
        self.original_height = game_config.PADDLE_HEIGHT
        self.size_effect_timer = 0.0

        # Movement limits according to player
        if player_id == 1:  # Left player
            self.min_x = 0.0
            self.max_x = game_config.FIELD_WIDTH / 2 - self.width
        else:  # Right player
            self.min_x = game_config.FIELD_WIDTH / 2
            self.max_x = game_config.FIELD_WIDTH - self.width

        self.min_y = 0.0
        self.max_y = game_config.FIELD_HEIGHT - self.height

    def update(self, dt: float) -> None:
        """Updates the paddle (temporary effects)"""
        if self.size_effect_timer > 0:
            self.size_effect_timer -= dt
            if self.size_effect_timer <= 0:
                self.reset_size()

    def move(self, dx: float, dy: float, dt: float) -> None:
        """Moves the paddle with constraints"""
        self.prev_position = Vector2D(self.position.x, self.position.y)
        speed = game_config.PADDLE_SPEED * dt
        new_x = self.position.x + dx * speed
        new_y = self.position.y + dy * speed

        # Movement constraints
        self.position.x = max(self.min_x, min(self.max_x, new_x))
        self.position.y = max(self.min_y, min(self.max_y, new_y))

    def apply_size_effect(self, multiplier: float, duration: float) -> None:
        """Applies a temporary size effect"""
        self.height = self.original_height * multiplier
        self.size_effect_timer = duration
        # Readjust Y limits
        self.max_y = game_config.FIELD_HEIGHT - self.height
        if self.position.y > self.max_y:
            self.position.y = self.max_y

    def reset_size(self) -> None:
        """Resets to normal size"""
        self.height = self.original_height
        self.max_y = game_config.FIELD_HEIGHT - self.height

    def get_rect(self) -> tuple[float, float, float, float]:
        """Returns the collision rectangle (x, y, width, height)"""
        return (self.position.x, self.position.y, self.width, self.height)


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
        self.lifetime = 30.0  # Disappears after 30 seconds if not collected

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
        half_size = self.size / 2
        return (self.position.x - half_size, self.position.y - half_size, self.size, self.size)


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
    field_bounds: tuple[float, float, float, float] = (0, 800, 0, 600)


@dataclass
class Action:
    """Player action"""

    move_x: float  # -1.0 to 1.0
    move_y: float  # -1.0 to 1.0

    def __post_init__(self) -> None:
        # Clamp values between -1 and 1
        self.move_x = max(-1.0, min(1.0, self.move_x))
        self.move_y = max(-1.0, min(1.0, self.move_y))
