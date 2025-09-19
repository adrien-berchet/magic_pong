"""
GUI module for Magic Pong - PyGame interface
"""

from magic_pong.gui.game_app import GameMode, GameState, MagicPongApp, main
from magic_pong.gui.human_player import HumanPlayer, InputManager, create_human_players
from magic_pong.gui.pygame_renderer import PygameRenderer

__all__ = [
    "PygameRenderer",
    "HumanPlayer",
    "InputManager",
    "create_human_players",
    "MagicPongApp",
    "GameMode",
    "GameState",
    "main",
]
