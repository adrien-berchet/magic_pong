"""
Human player implementation for Magic Pong
"""

from typing import Any

import pygame
from magic_pong.ai.interface import AIPlayer
from magic_pong.core.entities import Action
from magic_pong.utils.config import game_config


class HumanPlayer(AIPlayer):
    """Human player that gets input from keyboard"""

    def __init__(self, player_id: int, name: str = "Human", control_scheme: str = "arrows"):
        """
        Initialize human player

        Args:
            player_id: Player ID (1 or 2)
            name: Player name
            control_scheme: "arrows" for arrow keys or "wasd" for WASD keys
        """
        super().__init__(player_id, name)
        self.control_scheme = control_scheme
        self.current_action = Action(0.0, 0.0)

        # Get the current keyboard layout configuration
        layout = game_config.get_keyboard_layout()

        # Define control mappings based on the control scheme
        if control_scheme == "arrows":
            self.key_mapping = layout.arrow_keys.copy()
            self.display_names = {"up": "↑", "down": "↓", "left": "←", "right": "→"}
        elif control_scheme == "wasd":
            self.key_mapping = layout.wasd_keys.copy()
            self.display_names = layout.display_names.copy()
        else:
            raise ValueError(f"Unknown control scheme: {control_scheme}")

    def update_from_keys(self, keys_pressed: dict[int, bool]) -> None:
        """Update action based on currently pressed keys"""
        move_x = 0.0
        move_y = 0.0

        # Check movement keys
        if keys_pressed.get(self.key_mapping["left"], False):
            move_x -= 1.0
        if keys_pressed.get(self.key_mapping["right"], False):
            move_x += 1.0
        if keys_pressed.get(self.key_mapping["up"], False):
            move_y -= 1.0
        if keys_pressed.get(self.key_mapping["down"], False):
            move_y += 1.0

        # Update current action
        self.current_action = Action(move_x, move_y)

    def get_action(self, observation: dict[str, Any]) -> Action:
        """Get the current action (required by AIPlayer interface)"""
        return self.current_action

    def get_human_action(self) -> Action:
        """Get human action (alternative interface)"""
        return self.current_action

    def on_step(
        self,
        observation: dict[str, Any],
        action: Action,
        reward: float,
        done: bool,
        info: dict[str, Any],
    ) -> None:
        """Called after each step (required by AIPlayer interface)"""
        self.current_episode_reward += reward

    def get_control_info(self) -> dict[str, str]:
        """Get information about controls for this player"""
        return self.display_names.copy()


class InputManager:
    """Manages input for multiple human players"""

    def __init__(self) -> None:
        self.players: dict[int, HumanPlayer] = {}
        self.keys_pressed: dict[int, bool] = {}
        # Keep persistent state of special keys (arrows)
        self.special_keys_state: dict[int, bool] = {}

    def add_player(self, player: HumanPlayer) -> None:
        """Add a human player to manage"""
        self.players[player.player_id] = player

    def remove_player(self, player_id: int) -> None:
        """Remove a player"""
        if player_id in self.players:
            del self.players[player_id]

    def update_keys(self) -> None:
        """Update the state of all keys using pygame key constants"""
        pygame_keys = pygame.key.get_pressed()
        self.keys_pressed = {}

        # For normal keys (indices 0-511)
        for key_code in range(len(pygame_keys)):
            if pygame_keys[key_code]:
                self.keys_pressed[key_code] = True

        # Add state of special keys
        self.keys_pressed.update(self.special_keys_state)

    def update_players(self) -> None:
        """Update all managed human players with current key states"""
        self.update_keys()
        for player in self.players.values():
            player.update_from_keys(self.keys_pressed)

    def handle_event(self, event: pygame.event.Event) -> str | None:
        """
        Handle pygame events

        Returns:
            String indicating special actions (pause, quit, etc.) or None
        """
        if event.type == pygame.KEYDOWN:
            # Update state of special keys (arrows)
            if event.key in [pygame.K_UP, pygame.K_DOWN, pygame.K_LEFT, pygame.K_RIGHT]:
                self.special_keys_state[event.key] = True

            # Global game controls
            if event.key == pygame.K_ESCAPE:
                return "menu"
            elif event.key == pygame.K_p or event.key == pygame.K_SPACE:
                return "pause"
            elif event.key == pygame.K_F1:
                return "help"
            elif event.key == pygame.K_F2:
                return "toggle_fps"
            elif event.key == pygame.K_F3:
                return "toggle_debug"
            elif event.key == pygame.K_r:
                return "restart"

        elif event.type == pygame.KEYUP:
            # Update state of special keys (arrows)
            if event.key in [pygame.K_UP, pygame.K_DOWN, pygame.K_LEFT, pygame.K_RIGHT]:
                self.special_keys_state[event.key] = False

        elif event.type == pygame.QUIT:
            return "quit"

        return None

    def get_players(self) -> dict[int, HumanPlayer]:
        """Get all managed players"""
        return self.players.copy()


def create_human_players(mode: str) -> dict[int, HumanPlayer | None]:
    """
    Create human players based on game mode

    Args:
        mode: Game mode ("1v1", "1vAI", "demo")

    Returns:
        Dict mapping player IDs to HumanPlayer objects or None
    """
    players: dict[int, HumanPlayer | None] = {1: None, 2: None}

    if mode == "1v1":
        # Two human players
        players[1] = HumanPlayer(1, "Player 1", "wasd")
        players[2] = HumanPlayer(2, "Player 2", "arrows")

    elif mode == "1vAI":
        # One human player vs AI
        players[1] = HumanPlayer(1, "Player", "wasd")
        # players[2] will be set to an AI player elsewhere

    elif mode == "demo":
        # AI vs AI demo mode - no human players
        pass

    return players
