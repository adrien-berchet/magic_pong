"""
Magic Pong game configuration
"""

from dataclasses import dataclass


@dataclass
class GameConfig:
    """Main game configuration"""

    # Field dimensions
    FIELD_WIDTH: int = 800
    FIELD_HEIGHT: int = 600

    # Ball physics
    BALL_RADIUS: float = 8.0
    BALL_SPEED: float = 300.0  # pixels per second
    BALL_SPEED_INCREASE: float = 1.05  # Acceleration factor after each bounce

    # Player paddles
    PADDLE_WIDTH: float = 15.0
    PADDLE_HEIGHT: float = 80.0
    PADDLE_SPEED: float = 400.0  # pixels per second
    PADDLE_MARGIN: float = 50.0  # Distance from field edge

    # Bonuses
    BONUS_SIZE: float = 20.0
    BONUS_SPAWN_INTERVAL: float = 15.0  # seconds
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


@dataclass
class AIConfig:
    """Configuration for AI interface"""

    # Observation space
    NORMALIZE_POSITIONS: bool = True  # Normalize positions between -1 and 1
    INCLUDE_VELOCITY: bool = True  # Include ball velocity
    INCLUDE_HISTORY: bool = False  # Include position history
    HISTORY_LENGTH: int = 3  # Number of history frames

    # Reward system
    SCORE_REWARD: float = 1.0
    LOSE_PENALTY: float = -1.0
    BONUS_REWARD: float = 0.1
    WALL_HIT_REWARD: float = 0.01  # Small bonus for hitting the ball

    # Training
    MAX_EPISODE_STEPS: int = 10000
    HEADLESS_MODE: bool = False
    FAST_MODE_MULTIPLIER: float = 10.0  # Acceleration in fast mode


# Global configuration instance
game_config = GameConfig()
ai_config = AIConfig()
