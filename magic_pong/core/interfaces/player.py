"""
Player protocol - defines interface for all player types (human, AI, etc.)
"""

from typing import Any
from typing import Protocol

from magic_pong.core.entities import Action


class PlayerProtocol(Protocol):
    """
    Protocol that all players (human, AI, random, etc.) must implement.

    This enables polymorphic player handling - the game engine doesn't need
    to know if it's dealing with a human or AI player.
    """

    name: str

    def get_action(self, observation: dict[str, Any] | None) -> Action:
        """
        Get the next action based on the current game state.

        Args:
            observation: Game state observation (position, velocity, etc.)
                        Can be None for human players who don't need it.

        Returns:
            Action object with move_x and move_y values

        Example:
            >>> action = player.get_action(observation)
            >>> assert -1 <= action.move_x <= 1
            >>> assert -1 <= action.move_y <= 1
        """
        ...

    def on_episode_start(self) -> None:
        """
        Called when a new game episode starts.

        Optional lifecycle hook for initialization/reset logic.
        Default implementation can be empty.
        """
        ...

    def on_episode_end(self) -> None:
        """
        Called when a game episode ends.

        Optional lifecycle hook for cleanup/logging logic.
        Default implementation can be empty.
        """
        ...
