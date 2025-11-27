"""
AI module for Magic Pong
"""

from magic_pong.ai.interface import GameEnvironment
from magic_pong.ai.interface import ObservationProcessor
from magic_pong.ai.interface import RewardCalculator

__all__ = ["GameEnvironment", "ObservationProcessor", "RewardCalculator"]
