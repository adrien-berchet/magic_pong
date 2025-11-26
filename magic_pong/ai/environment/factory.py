"""
Environment Factory - Easy creation of training environments with custom components
"""

from typing import Any

import numpy as np

from magic_pong.ai.interfaces import (
    DenseRewardCalculator,
    ObservationBuilder,
    RewardCalculator,
    SparseRewardCalculator,
    VectorObservationBuilder,
)
from magic_pong.core.entities import Action as GameAction
from magic_pong.core.physics import PhysicsEngine
from magic_pong.utils.config import game_config


class EnvironmentFactory:
    """
    Factory for creating configured training environments.

    Makes it easy to experiment with different reward functions and
    observation spaces without modifying environment code.

    Example:
        >>> # Simple creation with defaults
        >>> env = EnvironmentFactory.create_default(physics_engine)

        >>> # Custom reward function
        >>> env = EnvironmentFactory.create(
        ...     physics_engine,
        ...     reward_calculator=DenseRewardCalculator(hit_reward=0.2)
        ... )

        >>> # Custom observation space
        >>> from magic_pong.ai.interfaces import HistoryObservationBuilder
        >>> env = EnvironmentFactory.create(
        ...     physics_engine,
        ...     observation_builder=HistoryObservationBuilder(history_length=5)
        ... )
    """

    @staticmethod
    def create_default(
        physics: PhysicsEngine, headless: bool = False, player_id: int = 1
    ) -> "GameEnvironmentWrapper":
        """
        Create environment with sensible defaults.

        Uses:
        - DenseRewardCalculator (rewards hits, wall bounces, bonuses)
        - VectorObservationBuilder (6D normalized vector)

        Args:
            physics: Physics engine instance
            headless: Run without display
            player_id: Player to train (1 or 2)

        Returns:
            Configured GameEnvironmentWrapper
        """
        return EnvironmentFactory.create(
            physics=physics,
            reward_calculator=DenseRewardCalculator(),
            observation_builder=VectorObservationBuilder(
                field_width=game_config.FIELD_WIDTH, field_height=game_config.FIELD_HEIGHT
            ),
            headless=headless,
            player_id=player_id,
        )

    @staticmethod
    def create_sparse(
        physics: PhysicsEngine, headless: bool = False, player_id: int = 1
    ) -> "GameEnvironmentWrapper":
        """
        Create environment with sparse rewards (only goals).

        Good for:
        - Testing if agent can learn from minimal signal
        - Reducing reward shaping bias
        - Curriculum learning (start sparse, add density later)

        Args:
            physics: Physics engine instance
            headless: Run without display
            player_id: Player to train (1 or 2)

        Returns:
            Configured GameEnvironmentWrapper with sparse rewards
        """
        return EnvironmentFactory.create(
            physics=physics,
            reward_calculator=SparseRewardCalculator(),
            observation_builder=VectorObservationBuilder(
                field_width=game_config.FIELD_WIDTH, field_height=game_config.FIELD_HEIGHT
            ),
            headless=headless,
            player_id=player_id,
        )

    @staticmethod
    def create(
        physics: PhysicsEngine,
        reward_calculator: RewardCalculator | None = None,
        observation_builder: ObservationBuilder | None = None,
        headless: bool = False,
        player_id: int = 1,
        **kwargs: Any,
    ) -> "GameEnvironmentWrapper":
        """
        Create environment with custom components.

        Args:
            physics: Physics engine instance
            reward_calculator: Custom reward calculator (defaults to DenseRewardCalculator)
            observation_builder: Custom observation builder (defaults to VectorObservationBuilder)
            headless: Run without display
            player_id: Player to train (1 or 2)
            **kwargs: Additional arguments for the environment

        Returns:
            Configured GameEnvironmentWrapper

        Example:
            >>> # Experiment with different reward shaping
            >>> from magic_pong.ai.interfaces import DenseRewardCalculator
            >>> env = EnvironmentFactory.create(
            ...     physics,
            ...     reward_calculator=DenseRewardCalculator(
            ...         goal_reward=10.0,  # Higher goal reward
            ...         hit_reward=0.5,    # Higher hit reward
            ...     )
            ... )
        """
        # Use defaults if not provided
        if reward_calculator is None:
            reward_calculator = DenseRewardCalculator()

        if observation_builder is None:
            observation_builder = VectorObservationBuilder(
                field_width=game_config.FIELD_WIDTH, field_height=game_config.FIELD_HEIGHT
            )

        # Create wrapper that uses the protocol-based components
        return GameEnvironmentWrapper(
            physics=physics,
            reward_calculator=reward_calculator,
            observation_builder=observation_builder,
            headless=headless,
            player_id=player_id,
            **kwargs,
        )


class GameEnvironmentWrapper:
    """
    Lightweight wrapper around PhysicsEngine that uses protocol-based components.

    This is a simplified environment that demonstrates the protocol-based approach.
    For full Gymnasium compatibility, use GameEnvironment from ai/interface.py.
    """

    def __init__(
        self,
        physics: PhysicsEngine,
        reward_calculator: RewardCalculator,
        observation_builder: ObservationBuilder,
        headless: bool = False,
        player_id: int = 1,
    ):
        self.physics = physics
        self.reward_calculator = reward_calculator
        self.observation_builder = observation_builder
        self.headless = headless
        self.player_id = player_id

        self.opponent_id = 2 if player_id == 1 else 1
        self.done = False

    def reset(self) -> Any:
        """Reset environment and return initial observation"""
        self.physics.reset_game()
        self.reward_calculator.reset()
        self.observation_builder.reset()
        self.done = False

        game_state = self.physics.get_game_state()
        return self.observation_builder.build_observation(game_state, self.player_id)

    def step(self, action: Any) -> tuple[Any, float, bool, dict]:
        """
        Take one step in the environment.

        Args:
            action: Action to take (numpy array, list, tuple, or Action object)

        Returns:
            Tuple of (observation, reward, done, info)
        """

        # Convert action to game action
        if isinstance(action, GameAction):
            game_action = action
        elif isinstance(action, np.ndarray):
            game_action = GameAction(move_x=float(action[0]), move_y=float(action[1]))
        elif isinstance(action, list | tuple):
            game_action = GameAction(move_x=float(action[0]), move_y=float(action[1]))
        else:
            raise TypeError(f"Unsupported action type: {type(action)}")

        # Opponent action (simple AI or random)
        opponent_action = GameAction(move_x=0.0, move_y=0.0)

        # Update physics
        events = self.physics.update(
            dt=1.0 / 60.0,  # 60 FPS
            player1_action=game_action if self.player_id == 1 else opponent_action,
            player2_action=opponent_action if self.player_id == 1 else game_action,
        )

        # Get new state
        game_state = self.physics.get_game_state()

        # Calculate reward using protocol
        reward = self.reward_calculator.calculate_reward(events, game_state, self.player_id)

        # Check if done
        self.done = self.physics.is_game_over()

        # Build observation using protocol
        obs = self.observation_builder.build_observation(game_state, self.player_id)

        # Info dict
        info = {"events": events, "score": game_state["score"]}

        return obs, reward, self.done, info

    @property
    def observation_space_size(self) -> int:
        """Get observation space dimension"""
        return int(self.observation_builder.observation_size)
