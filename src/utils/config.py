"""
Magic Pong game configuration
"""

from contextlib import contextmanager
from dataclasses import dataclass

import pygame


@dataclass
class KeyboardLayout:
    """Configuration for keyboard layouts"""

    name: str
    wasd_keys: dict[str, int]
    arrow_keys: dict[str, int]
    display_names: dict[str, str]


# Définition des layouts de clavier supportés
KEYBOARD_LAYOUTS = {
    "qwerty": KeyboardLayout(
        name="QWERTY",
        wasd_keys={"up": pygame.K_w, "down": pygame.K_s, "left": pygame.K_a, "right": pygame.K_d},
        arrow_keys={
            "up": pygame.K_UP,
            "down": pygame.K_DOWN,
            "left": pygame.K_LEFT,
            "right": pygame.K_RIGHT,
        },
        display_names={"up": "W", "down": "S", "left": "A", "right": "D"},
    ),
    "azerty": KeyboardLayout(
        name="AZERTY",
        wasd_keys={
            "up": pygame.K_z,  # Z à la place de W
            "down": pygame.K_s,
            "left": pygame.K_q,  # Q à la place de A
            "right": pygame.K_d,
        },
        arrow_keys={
            "up": pygame.K_UP,
            "down": pygame.K_DOWN,
            "left": pygame.K_LEFT,
            "right": pygame.K_RIGHT,
        },
        display_names={"up": "Z", "down": "S", "left": "Q", "right": "D"},
    ),
    "qwertz": KeyboardLayout(
        name="QWERTZ",
        wasd_keys={"up": pygame.K_w, "down": pygame.K_s, "left": pygame.K_a, "right": pygame.K_d},
        arrow_keys={
            "up": pygame.K_UP,
            "down": pygame.K_DOWN,
            "left": pygame.K_LEFT,
            "right": pygame.K_RIGHT,
        },
        display_names={"up": "W", "down": "S", "left": "A", "right": "D"},
    ),
}


@dataclass
class GameConfig:
    """Main game configuration"""

    # Field dimensions
    FIELD_WIDTH: int = 800
    FIELD_HEIGHT: int = 600

    # Ball physics
    BALL_RADIUS: float = 8.0
    MAX_BALL_SPEED: float = 500.0  # pixels per second
    BALL_SPEED: float = 300.0  # pixels per second
    BALL_SPEED_INCREASE: float = 1.05  # Acceleration factor after each bounce

    # Player paddles
    PADDLE_WIDTH: float = 15.0
    PADDLE_HEIGHT: float = 80.0
    PADDLE_SPEED: float = 500.0  # pixels per second
    PADDLE_MARGIN: float = 20.0  # Distance from field edge

    # Bonuses
    BONUSES_ENABLED: bool = True  # Master switch for all bonus features
    BONUS_SIZE: float = 20.0
    BONUS_SPAWN_INTERVAL: float = 15.0  # seconds
    BONUS_LIFETIME: float = 20.0  # seconds
    BONUS_DURATION: float = 10.0  # seconds
    PADDLE_SIZE_MULTIPLIER: float = 1.5  # Enlargement factor
    PADDLE_SIZE_REDUCER: float = 0.6  # Shrinking factor

    # Rotating paddle
    ROTATING_PADDLE_RADIUS: float = 40.0
    ROTATING_PADDLE_THICKNESS: float = 8.0
    ROTATING_PADDLE_SPEED: float = 2.0  # radians per second
    ROTATING_PADDLE_DURATION: float = 15.0  # seconds

    # Gameplay
    MAX_SCORE: int = 11
    GAME_SPEED_MULTIPLIER: float = 1.0  # To accelerate training

    # Keyboard layout
    KEYBOARD_LAYOUT: str = "azerty"  # Default layout for French users

    # Display
    FPS: int = 60
    BACKGROUND_COLOR: tuple[int, int, int] = (0, 0, 0)
    BALL_COLOR: tuple[int, int, int] = (255, 255, 255)
    PADDLE_COLOR: tuple[int, int, int] = (255, 255, 255)
    BONUS_COLORS: dict | None = None

    def __post_init__(self) -> None:
        if self.BONUS_COLORS is None:
            self.BONUS_COLORS = {
                "enlarge_paddle": (0, 255, 0),  # Green
                "shrink_opponent": (255, 0, 0),  # Red
                "rotating_paddle": (0, 0, 255),  # Blue
            }

    def get_keyboard_layout(self) -> KeyboardLayout:
        """Get the current keyboard layout configuration"""
        return KEYBOARD_LAYOUTS.get(self.KEYBOARD_LAYOUT, KEYBOARD_LAYOUTS["qwerty"])


@dataclass
class AIConfig:
    """Configuration for AI interface"""

    # Observation space
    NORMALIZE_POSITIONS: bool = False  # Normalize positions between -1 and 1
    INCLUDE_HISTORY: bool = False  # Include position history
    HISTORY_LENGTH: int = 3  # Number of history frames

    # Reward system
    SCORE_REWARD: float = 1.0
    LOSE_PENALTY: float = -1.0
    BONUS_REWARD: float = 0.1
    WALL_HIT_REWARD: float = 0.1  # Small bonus for hitting the ball
    USE_PROXIMITY_REWARD: bool = False  # Enable proximity reward
    PROXIMITY_REWARD_FACTOR: float = 0.001  # Reward factor for getting closer to the ball
    PROXIMITY_PENALTY_FACTOR: float = 0.001  # Penalty factor for moving away from the ball
    MAX_PROXIMITY_REWARD: float = 0.01  # Cap for proximity reward per step
    DEBUG_OPTIMAL_POINTS: bool = False  # Display optimal interception points for debugging
    SHOW_OPTIMAL_POINTS_GUI: bool = False  # Show optimal points in graphical interface

    # Training
    MAX_EPISODE_STEPS: int = 10000
    HEADLESS_MODE: bool = False
    FAST_MODE_MULTIPLIER: float = 10.0  # Acceleration in fast mode


# Global configuration instance
game_config = GameConfig()
ai_config = AIConfig()


def _change_values(obj, **kwargs):
    old_values = {}
    for name, new_value in kwargs.items():
        old_values[name] = getattr(obj, name)
        setattr(obj, name, new_value)
    return old_values


@contextmanager
def game_config_tmp(**kwargs):
    try:
        old_values = _change_values(game_config, **kwargs)
        yield
    finally:
        _change_values(game_config, **old_values)


@contextmanager
def ai_config_tmp(**kwargs):
    try:
        old_values = _change_values(ai_config, **kwargs)
        yield
    finally:
        _change_values(ai_config, **old_values)
