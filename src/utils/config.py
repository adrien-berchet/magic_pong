"""
Magic Pong game configuration with Pydantic validation
"""

from contextlib import contextmanager
from dataclasses import dataclass

import pygame
from pydantic import BaseModel, Field, field_validator, model_validator


@dataclass
class KeyboardLayout:
    """Configuration for keyboard layouts"""

    name: str
    wasd_keys: dict[str, int]
    arrow_keys: dict[str, int]
    display_names: dict[str, str]


# Keyboard layouts definition
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
            "up": pygame.K_z,  # Z instead of W
            "down": pygame.K_s,
            "left": pygame.K_q,  # Q instead of A
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


class GameConfig(BaseModel):
    """Main game configuration with Pydantic validation"""

    # Allow mutation for compatibility with existing code
    model_config = {"validate_assignment": True, "arbitrary_types_allowed": True}

    # Field dimensions
    FIELD_WIDTH: int = Field(default=800, gt=0, description="Field width in pixels")
    FIELD_HEIGHT: int = Field(default=600, gt=0, description="Field height in pixels")

    # Ball physics
    BALL_RADIUS: float = Field(default=8.0, gt=0, description="Ball radius in pixels")
    MAX_BALL_SPEED: float = Field(default=500.0, gt=0, description="Maximum ball speed")
    BALL_SPEED: float = Field(default=300.0, gt=0, description="Initial ball speed")
    BALL_SPEED_INCREASE: float = Field(default=1.05, gt=0, description="Speed increase factor")

    # Player paddles
    PADDLE_WIDTH: float = Field(default=15.0, gt=0, description="Paddle width in pixels")
    PADDLE_HEIGHT: float = Field(default=80.0, gt=0, description="Paddle height in pixels")
    PADDLE_SPEED: float = Field(default=500.0, gt=0, description="Paddle speed")
    PADDLE_MARGIN: float = Field(default=20.0, ge=0, description="Paddle margin from edge")

    # Bonuses
    BONUSES_ENABLED: bool = Field(default=True, description="Enable bonus system")
    BONUS_SIZE: float = Field(default=20.0, gt=0, description="Bonus size in pixels")
    BONUS_SPAWN_INTERVAL: float = Field(default=15.0, gt=0, description="Bonus spawn interval")
    BONUS_LIFETIME: float = Field(default=20.0, gt=0, description="Bonus lifetime in seconds")
    BONUS_DURATION: float = Field(default=10.0, gt=0, description="Bonus effect duration")
    PADDLE_SIZE_MULTIPLIER: float = Field(default=1.5, gt=1.0, description="Paddle enlargement")
    PADDLE_SIZE_REDUCER: float = Field(default=0.6, gt=0, lt=1.0, description="Paddle shrinking")

    # Rotating paddle
    ROTATING_PADDLE_RADIUS: float = Field(default=40.0, gt=0, description="Rotating paddle radius")
    ROTATING_PADDLE_THICKNESS: float = Field(default=8.0, gt=0, description="Paddle thickness")
    ROTATING_PADDLE_SPEED: float = Field(default=2.0, gt=0, description="Rotation speed")
    ROTATING_PADDLE_DURATION: float = Field(default=15.0, gt=0, description="Effect duration")

    # Gameplay
    MAX_SCORE: int = Field(default=11, gt=0, description="Winning score")
    GAME_SPEED_MULTIPLIER: float = Field(default=1.0, gt=0, description="Game speed multiplier")

    # Keyboard layout
    KEYBOARD_LAYOUT: str = Field(default="azerty", description="Keyboard layout name")

    # Display
    FPS: int = Field(default=60, gt=0, description="Frames per second")
    BACKGROUND_COLOR: tuple[int, int, int] = Field(default=(0, 0, 0), description="RGB color")
    BALL_COLOR: tuple[int, int, int] = Field(default=(255, 255, 255), description="RGB color")
    PADDLE_COLOR: tuple[int, int, int] = Field(default=(255, 255, 255), description="RGB color")
    BONUS_COLORS: dict | None = Field(default=None, description="Bonus colors mapping")

    @field_validator("BALL_SPEED")
    @classmethod
    def validate_ball_speed(cls, v: float, info) -> float:
        """Validate that ball speed doesn't exceed max speed"""
        # info.data contains already validated fields
        max_speed = info.data.get("MAX_BALL_SPEED", 500.0)
        if v > max_speed:
            raise ValueError(f"BALL_SPEED ({v}) must not exceed MAX_BALL_SPEED ({max_speed})")
        return v

    @field_validator("KEYBOARD_LAYOUT")
    @classmethod
    def validate_keyboard_layout(cls, v: str) -> str:
        """Validate keyboard layout exists"""
        if v not in KEYBOARD_LAYOUTS:
            raise ValueError(
                f"Unknown keyboard layout '{v}'. Available: {list(KEYBOARD_LAYOUTS.keys())}"
            )
        return v

    @model_validator(mode="after")
    def validate_field_dimensions(self) -> "GameConfig":
        """Validate field is large enough for game elements"""
        min_width = 2 * (self.PADDLE_MARGIN + self.PADDLE_WIDTH) + 100
        if self.FIELD_WIDTH < min_width:
            raise ValueError(f"FIELD_WIDTH must be at least {min_width} pixels")

        min_height = self.PADDLE_HEIGHT + 50
        if self.FIELD_HEIGHT < min_height:
            raise ValueError(f"FIELD_HEIGHT must be at least {min_height} pixels")

        return self

    def model_post_init(self, __context) -> None:
        """Initialize bonus colors if not set"""
        if self.BONUS_COLORS is None:
            object.__setattr__(
                self,
                "BONUS_COLORS",
                {
                    "enlarge_paddle": (0, 255, 0),  # Green
                    "shrink_opponent": (255, 0, 0),  # Red
                    "rotating_paddle": (0, 0, 255),  # Blue
                },
            )

    def get_keyboard_layout(self) -> KeyboardLayout:
        """Get the current keyboard layout configuration"""
        return KEYBOARD_LAYOUTS.get(self.KEYBOARD_LAYOUT, KEYBOARD_LAYOUTS["qwerty"])


class AIConfig(BaseModel):
    """Configuration for AI interface with Pydantic validation"""

    model_config = {"validate_assignment": True}

    # Observation space
    NORMALIZE_POSITIONS: bool = Field(default=False, description="Normalize positions to [-1, 1]")
    INCLUDE_HISTORY: bool = Field(default=False, description="Include position history")
    HISTORY_LENGTH: int = Field(default=3, gt=0, description="Number of history frames")

    # Reward system
    SCORE_REWARD: float = Field(default=1.0, description="Reward for scoring")
    LOSE_PENALTY: float = Field(default=-1.0, description="Penalty for losing point")
    BONUS_REWARD: float = Field(default=0.1, description="Reward for collecting bonus")
    WALL_HIT_REWARD: float = Field(default=0.1, description="Reward for ball hitting wall")
    USE_PROXIMITY_REWARD: bool = Field(default=False, description="Enable proximity rewards")
    PROXIMITY_REWARD_FACTOR: float = Field(default=0.001, ge=0, description="Proximity reward")
    PROXIMITY_PENALTY_FACTOR: float = Field(default=0.001, ge=0, description="Proximity penalty")
    MAX_PROXIMITY_REWARD: float = Field(default=0.01, ge=0, description="Max proximity reward")
    DEBUG_OPTIMAL_POINTS: bool = Field(default=False, description="Debug optimal points")
    SHOW_OPTIMAL_POINTS_GUI: bool = Field(default=False, description="Show optimal points in GUI")

    # Training
    MAX_EPISODE_STEPS: int = Field(default=10000, gt=0, description="Max steps per episode")
    HEADLESS_MODE: bool = Field(default=False, description="Run without display")
    FAST_MODE_MULTIPLIER: float = Field(
        default=10.0, gt=0, description="Speed multiplier for fast mode"
    )

    @field_validator("FAST_MODE_MULTIPLIER")
    @classmethod
    def validate_fast_mode(cls, v: float) -> float:
        """Warn if fast mode multiplier is unreasonably high"""
        if v > 100:
            raise ValueError(
                f"FAST_MODE_MULTIPLIER ({v}) seems too high. Consider values <= 100."
            )
        return v


# Global configuration instances with validation
game_config = GameConfig()
ai_config = AIConfig()


def _change_values(obj: BaseModel, **kwargs) -> dict:
    """Helper to change config values temporarily"""
    old_values = {}
    for name, new_value in kwargs.items():
        old_values[name] = getattr(obj, name)
        setattr(obj, name, new_value)
    return old_values


@contextmanager
def game_config_tmp(**kwargs):
    """Temporarily modify game config (with validation)"""
    try:
        old_values = _change_values(game_config, **kwargs)
        yield
    finally:
        _change_values(game_config, **old_values)


@contextmanager
def ai_config_tmp(**kwargs):
    """Temporarily modify AI config (with validation)"""
    try:
        old_values = _change_values(ai_config, **kwargs)
        yield
    finally:
        _change_values(ai_config, **old_values)
