"""
AI interfaces and protocols for extensible RL training
"""

from magic_pong.ai.interfaces.observation import HistoryObservationBuilder
from magic_pong.ai.interfaces.observation import ObservationBuilder
from magic_pong.ai.interfaces.observation import VectorObservationBuilder
from magic_pong.ai.interfaces.reward import DenseRewardCalculator
from magic_pong.ai.interfaces.reward import RewardCalculator
from magic_pong.ai.interfaces.reward import SparseRewardCalculator

__all__ = [
    # Protocols
    "ObservationBuilder",
    "RewardCalculator",
    # Concrete implementations - Rewards
    "SparseRewardCalculator",
    "DenseRewardCalculator",
    # Concrete implementations - Observations
    "VectorObservationBuilder",
    "HistoryObservationBuilder",
]
