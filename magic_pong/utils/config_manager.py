"""
Configuration manager for UI settings
Defines categorized configuration options for the settings menu
"""

from dataclasses import dataclass
from enum import Enum
from typing import Any


class ConfigFieldType(Enum):
    """Types of configuration fields"""

    NUMERIC = "numeric"
    BOOLEAN = "boolean"
    SELECTION = "selection"
    COLOR = "color"


@dataclass
class ConfigOption:
    """Represents a single configuration option"""

    key: str
    label: str
    field_type: ConfigFieldType
    min_value: float | None = None
    max_value: float | None = None
    step: float = 1.0
    choices: list[str] | None = None
    description: str = ""


class ConfigCategory:
    """Configuration category with grouped options"""

    def __init__(self, name: str, options: list[ConfigOption]):
        self.name = name
        self.options = options


# Define configuration categories
GAMEPLAY_OPTIONS = ConfigCategory(
    "Gameplay",
    [
        ConfigOption(
            key="BALL_SPEED",
            label="Ball Speed",
            field_type=ConfigFieldType.NUMERIC,
            min_value=100.0,
            max_value=800.0,
            step=25.0,
            description="Initial ball speed",
        ),
        ConfigOption(
            key="PADDLE_SPEED",
            label="Paddle Speed",
            field_type=ConfigFieldType.NUMERIC,
            min_value=200.0,
            max_value=1000.0,
            step=50.0,
            description="Player paddle movement speed",
        ),
        ConfigOption(
            key="MAX_SCORE",
            label="Max Score",
            field_type=ConfigFieldType.NUMERIC,
            min_value=1,
            max_value=50,
            step=1,
            description="Points needed to win",
        ),
        ConfigOption(
            key="BONUSES_ENABLED",
            label="Bonuses",
            field_type=ConfigFieldType.BOOLEAN,
            description="Enable/disable bonus items",
        ),
    ],
)

CONTROLS_OPTIONS = ConfigCategory(
    "Controls",
    [
        ConfigOption(
            key="KEYBOARD_LAYOUT",
            label="Keyboard Layout",
            field_type=ConfigFieldType.SELECTION,
            choices=["qwerty", "azerty", "qwertz"],
            description="Keyboard layout for controls",
        ),
    ],
)

DISPLAY_OPTIONS = ConfigCategory(
    "Display",
    [
        ConfigOption(
            key="FPS",
            label="Frame Rate",
            field_type=ConfigFieldType.NUMERIC,
            min_value=30,
            max_value=144,
            step=10,
            description="Target frames per second",
        ),
    ],
)

ADVANCED_OPTIONS = ConfigCategory(
    "Advanced",
    [
        ConfigOption(
            key="FIELD_WIDTH",
            label="Field Width",
            field_type=ConfigFieldType.NUMERIC,
            min_value=600,
            max_value=1920,
            step=50,
            description="Game field width in pixels",
        ),
        ConfigOption(
            key="FIELD_HEIGHT",
            label="Field Height",
            field_type=ConfigFieldType.NUMERIC,
            min_value=400,
            max_value=1080,
            step=50,
            description="Game field height in pixels",
        ),
        ConfigOption(
            key="MAX_BALL_SPEED",
            label="Max Ball Speed",
            field_type=ConfigFieldType.NUMERIC,
            min_value=300.0,
            max_value=1500.0,
            step=50.0,
            description="Maximum ball speed (physics limit)",
        ),
        ConfigOption(
            key="BALL_SPEED_INCREASE",
            label="Ball Speed Increase",
            field_type=ConfigFieldType.NUMERIC,
            min_value=1.0,
            max_value=1.2,
            step=0.01,
            description="Speed multiplier per hit",
        ),
        ConfigOption(
            key="PADDLE_WIDTH",
            label="Paddle Width",
            field_type=ConfigFieldType.NUMERIC,
            min_value=10.0,
            max_value=30.0,
            step=1.0,
            description="Paddle width in pixels",
        ),
        ConfigOption(
            key="PADDLE_HEIGHT",
            label="Paddle Height",
            field_type=ConfigFieldType.NUMERIC,
            min_value=40.0,
            max_value=150.0,
            step=5.0,
            description="Paddle height in pixels",
        ),
        ConfigOption(
            key="BONUS_SPAWN_INTERVAL",
            label="Bonus Spawn Interval",
            field_type=ConfigFieldType.NUMERIC,
            min_value=5.0,
            max_value=60.0,
            step=5.0,
            description="Time between bonus spawns (seconds)",
        ),
        ConfigOption(
            key="BONUS_LIFETIME",
            label="Bonus Lifetime",
            field_type=ConfigFieldType.NUMERIC,
            min_value=5.0,
            max_value=60.0,
            step=5.0,
            description="How long bonuses stay on field (seconds)",
        ),
        ConfigOption(
            key="BONUS_DURATION",
            label="Bonus Effect Duration",
            field_type=ConfigFieldType.NUMERIC,
            min_value=3.0,
            max_value=30.0,
            step=1.0,
            description="How long bonus effects last (seconds)",
        ),
    ],
)

# All categories in order
CONFIG_CATEGORIES = [
    GAMEPLAY_OPTIONS,
    CONTROLS_OPTIONS,
    DISPLAY_OPTIONS,
    ADVANCED_OPTIONS,
]


def get_config_value(config_obj: Any, key: str) -> Any:
    """Get a configuration value from the config object"""
    return getattr(config_obj, key)


def set_config_value(config_obj: Any, key: str, value: Any) -> None:
    """Set a configuration value on the config object"""
    setattr(config_obj, key, value)
