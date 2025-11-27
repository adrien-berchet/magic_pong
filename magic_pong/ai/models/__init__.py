"""
AI models for Magic Pong
"""

from magic_pong.ai.models.simple_ai import AggressiveAI
from magic_pong.ai.models.simple_ai import DefensiveAI
from magic_pong.ai.models.simple_ai import FollowBallAI
from magic_pong.ai.models.simple_ai import HumanPlayer
from magic_pong.ai.models.simple_ai import PredictiveAI
from magic_pong.ai.models.simple_ai import RandomAI
from magic_pong.ai.models.simple_ai import create_ai

__all__ = [
    "RandomAI",
    "FollowBallAI",
    "DefensiveAI",
    "AggressiveAI",
    "PredictiveAI",
    "HumanPlayer",
    "create_ai",
]
