"""
AI interfaces and protocols for extensible RL training
"""

from magic_pong.ai.interfaces.observation import (
    HistoryObservationBuilder,
    ObservationBuilder,
    VectorObservationBuilder,
)
from magic_pong.ai.interfaces.reward import (
    DenseRewardCalculator,
    RewardCalculator,
    SparseRewardCalculator,
)

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
