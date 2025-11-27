"""
Observation builder protocol - enables custom observation spaces for RL training
"""

from typing import Any
from typing import Protocol

import numpy as np
import numpy.typing as npt


class ObservationBuilder(Protocol):
    """
    Protocol for building observations from game state.

    Enables experimenting with different observation representations
    without modifying the environment code.

    Example implementations:
    - VectorObservationBuilder: Flat vector of positions/velocities
    - ImageObservationBuilder: Pixel-based observations
    - HistoryObservationBuilder: Includes temporal information
    - AsymmetricObservationBuilder: Different obs for each player
    """

    def build_observation(
        self, game_state: dict[str, Any], player_id: int
    ) -> npt.NDArray[np.float32]:
        """
        Build observation array from game state.

        Args:
            game_state: Complete game state dictionary
            player_id: Player to build observation for (1 or 2)

        Returns:
            Numpy array with observation data (shape depends on implementation)

        Example:
            >>> builder = VectorObservationBuilder()
            >>> obs = builder.build_observation(game_state, player_id=1)
            >>> assert obs.shape == (builder.observation_size,)
            >>> assert obs.dtype == np.float32
        """
        ...

    @property
    def observation_size(self) -> int:
        """Get the size/dimension of the observation vector"""
        ...

    def reset(self) -> None:
        """
        Reset internal state (if any).

        Called at the start of each episode. Useful for stateful
        builders that track history.
        """
        ...


class VectorObservationBuilder:
    """
    Simple vector observation with normalized positions and velocities.

    Observation format (6 values):
    [
        ball_x (normalized),
        ball_y (normalized),
        ball_vx (normalized),
        ball_vy (normalized),
        paddle_y (normalized),
        opponent_paddle_y (normalized)
    ]
    """

    def __init__(self, field_width: float = 800.0, field_height: float = 600.0):
        self.field_width = field_width
        self.field_height = field_height
        self._observation_size = 6

    def build_observation(
        self, game_state: dict[str, Any], player_id: int
    ) -> npt.NDArray[np.float32]:
        # Extract positions
        ball_pos = game_state["ball_position"]
        ball_vel = game_state["ball_velocity"]
        p1_pos = game_state["player1_position"]
        p2_pos = game_state["player2_position"]

        # Normalize to [0, 1]
        ball_x_norm = ball_pos[0] / self.field_width
        ball_y_norm = ball_pos[1] / self.field_height
        ball_vx_norm = ball_vel[0] / 1000.0  # Normalize velocity
        ball_vy_norm = ball_vel[1] / 1000.0

        # Player-relative observation
        if player_id == 1:
            paddle_y_norm = p1_pos[1] / self.field_height
            opponent_y_norm = p2_pos[1] / self.field_height
        else:
            # Mirror for player 2 (so they see from their perspective)
            ball_x_norm = 1.0 - ball_x_norm
            ball_vx_norm = -ball_vx_norm
            paddle_y_norm = p2_pos[1] / self.field_height
            opponent_y_norm = p1_pos[1] / self.field_height

        return np.array(
            [ball_x_norm, ball_y_norm, ball_vx_norm, ball_vy_norm, paddle_y_norm, opponent_y_norm],
            dtype=np.float32,
        )

    @property
    def observation_size(self) -> int:
        return self._observation_size

    def reset(self) -> None:
        pass  # No state to reset


class HistoryObservationBuilder:
    """
    Observation builder that includes temporal history.

    Stacks multiple frames to give the agent information about
    velocity and trajectory trends.

    Observation format: history_length * base_observation_size
    """

    def __init__(
        self,
        base_builder: ObservationBuilder | None = None,
        history_length: int = 3,
        field_width: float = 800.0,
        field_height: float = 600.0,
    ):
        self.base_builder = base_builder or VectorObservationBuilder(field_width, field_height)
        self.history_length = history_length
        self.history: list[npt.NDArray[np.float32]] = []
        self._observation_size = self.base_builder.observation_size * history_length

    def build_observation(
        self, game_state: dict[str, Any], player_id: int
    ) -> npt.NDArray[np.float32]:
        # Get current observation
        current_obs = self.base_builder.build_observation(game_state, player_id)

        # Add to history
        self.history.append(current_obs)

        # Keep only last N frames
        if len(self.history) > self.history_length:
            self.history.pop(0)

        # Pad if needed (at start of episode)
        while len(self.history) < self.history_length:
            self.history.insert(0, np.zeros_like(current_obs))

        # Stack history
        return np.concatenate(self.history)

    @property
    def observation_size(self) -> int:
        return self._observation_size

    def reset(self) -> None:
        self.history = []
        self.base_builder.reset()
