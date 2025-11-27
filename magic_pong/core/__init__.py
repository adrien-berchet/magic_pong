"""
Core module of Magic Pong game
"""

from magic_pong.core.entities import Action
from magic_pong.core.entities import Ball
from magic_pong.core.entities import Bonus
from magic_pong.core.entities import BonusType
from magic_pong.core.entities import GameState
from magic_pong.core.entities import Paddle
from magic_pong.core.entities import RotatingPaddle
from magic_pong.core.entities import Vector2D

__all__ = [
    "Ball",
    "Paddle",
    "RotatingPaddle",
    "Bonus",
    "BonusType",
    "GameState",
    "Action",
    "Vector2D",
]
