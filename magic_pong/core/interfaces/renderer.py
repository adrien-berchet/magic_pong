"""
Renderer protocol - defines interface for different rendering backends
"""

from typing import Any, Protocol

from magic_pong.core.entities import Ball, Bonus, Paddle, RotatingPaddle


class RendererProtocol(Protocol):
    """
    Protocol for renderer implementations.

    Enables multiple rendering backends: Pygame, headless, terminal, web, etc.
    """

    def initialize(self, width: int, height: int) -> None:
        """
        Initialize the renderer with field dimensions.

        Args:
            width: Field width in pixels
            height: Field height in pixels
        """
        ...

    def render_frame(
        self,
        ball: Ball,
        player1: Paddle,
        player2: Paddle,
        bonuses: list[Bonus],
        rotating_paddles: list[RotatingPaddle],
        score: list[int],
        additional_info: dict[str, Any] | None = None,
    ) -> None:
        """
        Render a single frame of the game.

        Args:
            ball: Ball entity
            player1: Player 1's paddle
            player2: Player 2's paddle
            bonuses: List of active bonuses
            rotating_paddles: List of rotating paddles
            score: Current score [p1_score, p2_score]
            additional_info: Optional extra data to display (FPS, debug info, etc.)
        """
        ...

    def handle_events(self) -> dict[str, Any]:
        """
        Process input events (keyboard, mouse, etc.).

        Returns:
            Dictionary with event data (quit, pause, etc.)
        """
        ...

    def cleanup(self) -> None:
        """Clean up renderer resources"""
        ...

    def is_active(self) -> bool:
        """Check if renderer is still active (window not closed)"""
        ...
