"""
AI models for Magic Pong
"""

from magic_pong.ai.models.simple_ai import (
    AggressiveAI,
    DefensiveAI,
    FollowBallAI,
    HumanPlayer,
    PredictiveAI,
    RandomAI,
    create_ai,
)

__all__ = [
    "RandomAI",
    "FollowBallAI",
    "DefensiveAI",
    "AggressiveAI",
    "PredictiveAI",
    "HumanPlayer",
    "create_ai",
]
