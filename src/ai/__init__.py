"""
AI module for Magic Pong
"""

from magic_pong.ai.interface import (
    AIPlayer,
    GameEnvironment,
    ObservationProcessor,
    RewardCalculator,
)

__all__ = ["AIPlayer", "GameEnvironment", "ObservationProcessor", "RewardCalculator"]
