"""
AI interfaces and protocols for extensible RL training
"""

from magic_pong.ai.interfaces.observation import ObservationBuilder
from magic_pong.ai.interfaces.reward import RewardCalculator

__all__ = ["ObservationBuilder", "RewardCalculator"]
