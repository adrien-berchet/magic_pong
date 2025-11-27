"""
Reward calculator protocol - enables custom reward functions for RL training
"""

from typing import Any
from typing import Protocol


class RewardCalculator(Protocol):
    """
    Protocol for reward calculation strategies.

    Enables experimenting with different reward functions without
    modifying the environment code.

    Example implementations:
    - SparseRewardCalculator: Only rewards on goal
    - DenseRewardCalculator: Rewards proximity, hits, etc.
    - ShapedRewardCalculator: Potential-based shaping
    - CurriculumRewardCalculator: Adapts difficulty over time
    """

    def calculate_reward(
        self, events: dict[str, list[Any]], game_state: dict[str, Any], player_id: int
    ) -> float:
        """
        Calculate reward for a player based on events and state.

        Args:
            events: Dictionary of events that occurred this step:
                {
                    "paddle_hits": [player_id, ...],
                    "wall_bounces": ["top", "bottom", ...],
                    "goals": [{"player": 1, "score": [1, 0]}, ...],
                    "bonus_collected": [(player_id, bonus_type), ...]
                }
            game_state: Current game state dictionary
            player_id: Player to calculate reward for (1 or 2)

        Returns:
            Reward value (typically float, can be positive or negative)

        Example:
            >>> calculator = SparseRewardCalculator()
            >>> events = {"goals": [{"player": 1}], "paddle_hits": []}
            >>> reward = calculator.calculate_reward(events, state, player_id=1)
            >>> assert reward == 1.0  # Scored a goal
        """
        ...

    def reset(self) -> None:
        """
        Reset internal state (if any).

        Called at the start of each episode. Useful for stateful
        reward calculators that track history.
        """
        ...


class SparseRewardCalculator:
    """
    Simple sparse reward: +1 for goal, -1 for opponent goal.

    This is the simplest reward function - only provides feedback
    when goals are scored.
    """

    def calculate_reward(
        self, events: dict[str, list[Any]], game_state: dict[str, Any], player_id: int
    ) -> float:
        reward = 0.0

        for goal_event in events.get("goals", []):
            if goal_event["player"] == player_id:
                reward += 1.0  # Scored
            else:
                reward -= 1.0  # Opponent scored

        return reward

    def reset(self) -> None:
        pass  # No state to reset


class DenseRewardCalculator:
    """
    Dense reward with intermediate signals.

    Provides rewards for:
    - Scoring goals (+1.0)
    - Hitting the ball (+0.1)
    - Ball hitting walls (+0.05)
    - Collecting bonuses (+0.1)
    - Opponent scoring (-1.0)

    This helps with exploration but can be harder to tune.
    """

    def __init__(
        self,
        goal_reward: float = 1.0,
        lose_penalty: float = -1.0,
        hit_reward: float = 0.1,
        wall_reward: float = 0.05,
        bonus_reward: float = 0.1,
    ):
        self.goal_reward = goal_reward
        self.lose_penalty = lose_penalty
        self.hit_reward = hit_reward
        self.wall_reward = wall_reward
        self.bonus_reward = bonus_reward

    def calculate_reward(
        self, events: dict[str, list[Any]], game_state: dict[str, Any], player_id: int
    ) -> float:
        reward = 0.0

        # Goal events
        for goal_event in events.get("goals", []):
            if goal_event["player"] == player_id:
                reward += self.goal_reward
            else:
                reward += self.lose_penalty

        # Paddle hits
        for hit_player in events.get("paddle_hits", []):
            if hit_player == player_id:
                reward += self.hit_reward

        # Wall bounces (small reward for keeping ball in play)
        reward += len(events.get("wall_bounces", [])) * self.wall_reward

        # Bonus collection
        for bonus_player, _ in events.get("bonus_collected", []):
            if bonus_player == player_id:
                reward += self.bonus_reward

        return reward

    def reset(self) -> None:
        pass
