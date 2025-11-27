"""
Physics backend protocol - defines interface for physics engines
"""

from typing import Any
from typing import Protocol

from magic_pong.core.entities import Action
from magic_pong.core.entities import Ball
from magic_pong.core.entities import Paddle


class PhysicsBackend(Protocol):
    """
    Protocol for physics engine implementations.

    This allows swapping physics implementations (Box2D, custom, etc.)
    without changing game logic.
    """

    # Game objects
    ball: Ball
    player1: Paddle
    player2: Paddle
    score: list[int]
    game_time: float

    # Field dimensions
    field_width: float
    field_height: float

    def reset_game(self) -> None:
        """
        Reset the game to initial state.

        Resets score, time, ball position, paddle positions, etc.
        """
        ...

    def reset_ball(self, direction: int = 0, angle: float | None = None) -> None:
        """
        Reset ball to center with optional direction/angle.

        Args:
            direction: -1 for left, 1 for right, 0 for random
            angle: Optional specific angle in radians
        """
        ...

    def update(self, dt: float, player1_action: Action, player2_action: Action) -> dict[str, Any]:
        """
        Update physics simulation by one timestep.

        Args:
            dt: Delta time in seconds
            player1_action: Player 1's action
            player2_action: Player 2's action

        Returns:
            Dictionary with events that occurred:
            {
                "paddle_hits": [...],
                "wall_bounces": [...],
                "goals": [...],
                "bonus_collected": [...]
            }
        """
        ...

    def get_game_state(self) -> dict[str, Any]:
        """
        Get complete game state for observation.

        Returns:
            Dictionary with positions, velocities, score, etc.
        """
        ...

    def is_game_over(self) -> bool:
        """Check if game has ended (max score reached)"""
        ...

    def get_winner(self) -> int:
        """Get winning player ID (1 or 2), or 0 if no winner"""
        ...
