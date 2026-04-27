"""
AI module for Magic Pong
"""

import os

os.environ.setdefault("PYGAME_HIDE_SUPPORT_PROMPT", "1")

from magic_pong.ai.interface import GameEnvironment
from magic_pong.ai.interface import ObservationProcessor
from magic_pong.ai.interface import RewardCalculator

__all__ = ["GameEnvironment", "ObservationProcessor", "RewardCalculator"]
