"""
Agnostic AI interface for Magic Pong
"""

from abc import ABC, abstractmethod
from typing import Any

import numpy as np
from magic_pong.core.entities import Action
from magic_pong.core.physics import PhysicsEngine
from magic_pong.utils.config import ai_config


class AIPlayer(ABC):
    """Base interface for all AI players"""

    def __init__(self, player_id: int, name: str = "AI"):
        self.player_id = player_id
        self.name = name
        self.episode_rewards: list[float] = []
        self.current_episode_reward = 0.0

    @abstractmethod
    def get_action(self, observation: dict[str, Any]) -> Action:
        """
        Returns the action to perform based on the observation

        Args:
            observation: Normalized game state

        Returns:
            Action: Action to perform
        """
        pass

    @abstractmethod
    def on_step(
        self,
        observation: dict[str, Any],
        action: Action,
        reward: float,
        done: bool,
        info: dict[str, Any],
    ) -> None:
        """
        Called after each step for learning

        Args:
            observation: New observation
            action: Action performed
            reward: Reward received
            done: If the episode is finished
            info: Additional information
        """
        pass

    def on_episode_start(self) -> None:
        """Called at the start of each episode"""
        self.current_episode_reward = 0.0

    def on_episode_end(self, final_reward: float) -> None:
        """Called at the end of each episode"""
        self.episode_rewards.append(self.current_episode_reward)

    def get_stats(self) -> dict[str, float]:
        """Returns performance statistics"""
        if not self.episode_rewards:
            return {"mean_reward": 0.0, "episodes": 0}

        return {
            "mean_reward": float(
                np.mean(self.episode_rewards[-100:])
            ),  # Average over last 100 episodes
            "total_reward": sum(self.episode_rewards),
            "episodes": len(self.episode_rewards),
            "last_reward": self.episode_rewards[-1] if self.episode_rewards else 0.0,
        }


class ObservationProcessor:
    """Observation processor to normalize data"""

    def __init__(self, field_width: float, field_height: float):
        self.field_width = field_width
        self.field_height = field_height

    def process_game_state(self, game_state: dict[str, Any], player_id: int) -> dict[str, Any]:
        """
        Converts game state to normalized observation for AI

        Args:
            game_state: Raw game state
            player_id: Player ID (1 or 2)

        Returns:
            Dict: Normalized observation
        """
        observation = {}

        # Normalized positions
        if ai_config.NORMALIZE_POSITIONS:
            ball_x = game_state["ball_position"][0] / self.field_width
            ball_y = game_state["ball_position"][1] / self.field_height

            player_pos = game_state[f"player{player_id}_position"]
            opponent_pos = game_state[f"player{3-player_id}_position"]

            player_x = player_pos[0] / self.field_width
            player_y = player_pos[1] / self.field_height
            opponent_x = opponent_pos[0] / self.field_width
            opponent_y = opponent_pos[1] / self.field_height
        else:
            ball_x, ball_y = game_state["ball_position"]
            player_x, player_y = game_state[f"player{player_id}_position"]
            opponent_x, opponent_y = game_state[f"player{3-player_id}_position"]

        observation["ball_pos"] = [ball_x, ball_y]
        observation["player_pos"] = [player_x, player_y]
        observation["opponent_pos"] = [opponent_x, opponent_y]

        # Ball velocity
        if ai_config.INCLUDE_VELOCITY:
            if ai_config.NORMALIZE_POSITIONS:
                vel_x = game_state["ball_velocity"][0] / 500.0  # Normalize by approximate max speed
                vel_y = game_state["ball_velocity"][1] / 500.0
            else:
                vel_x, vel_y = game_state["ball_velocity"]
            observation["ball_vel"] = [vel_x, vel_y]

        # Paddle sizes
        observation["player_paddle_size"] = game_state[f"player{player_id}_paddle_size"]
        observation["opponent_paddle_size"] = game_state[f"player{3-player_id}_paddle_size"]

        # Active bonuses
        bonuses = []
        for bonus_x, bonus_y, bonus_type in game_state["active_bonuses"]:
            if ai_config.NORMALIZE_POSITIONS:
                bonus_x /= self.field_width
                bonus_y /= self.field_height
            bonuses.append([bonus_x, bonus_y, self._encode_bonus_type(bonus_type)])
        observation["bonuses"] = bonuses

        # Rotating paddles
        rotating_paddles = []
        for rp_x, rp_y, rp_angle in game_state["rotating_paddles"]:
            if ai_config.NORMALIZE_POSITIONS:
                rp_x /= self.field_width
                rp_y /= self.field_height
            rotating_paddles.append([rp_x, rp_y, rp_angle])
        observation["rotating_paddles"] = rotating_paddles

        # Score differential
        score = game_state["score"]
        if player_id == 1:
            observation["score_diff"] = score[0] - score[1]
        else:
            observation["score_diff"] = score[1] - score[0]

        # Elapsed time
        observation["time_elapsed"] = game_state["time_elapsed"]

        return observation

    def _encode_bonus_type(self, bonus_type: str) -> float:
        """Encodes bonus type to numeric value"""
        encoding = {"enlarge_paddle": 1.0, "shrink_opponent": 2.0, "rotating_paddle": 3.0}
        return encoding.get(bonus_type, 0.0)


class RewardCalculator:
    """Reward calculator for training"""

    def __init__(self) -> None:
        self.last_score = [0, 0]
        self.last_ball_distance = 0.0

    def calculate_reward(
        self, game_state: dict[str, Any], events: dict[str, list], player_id: int
    ) -> float:
        """
        Calculates reward for a player based on events

        Args:
            game_state: Current game state
            events: Events that occurred this step
            player_id: Player ID

        Returns:
            float: Calculated reward
        """
        reward = 0.0

        # Rewards for goals
        for goal in events.get("goals", []):
            if goal["player"] == player_id:
                reward += ai_config.SCORE_REWARD
            else:
                reward += ai_config.LOSE_PENALTY

        # Rewards for collected bonuses
        for bonus in events.get("bonus_collected", []):
            if bonus["player"] == player_id:
                reward += ai_config.BONUS_REWARD

        # Rewards for hitting the ball
        for hit in events.get("paddle_hits", []):
            if hit["player"] == player_id:
                reward += ai_config.WALL_HIT_REWARD

        # Reward for rotating paddles
        for hit in events.get("rotating_paddle_hits", []):
            if hit["player"] == player_id:
                reward += ai_config.WALL_HIT_REWARD * 2  # Bonus for using rotating paddle

        return reward

    def reset(self) -> None:
        """Resets the calculator"""
        self.last_score = [0, 0]
        self.last_ball_distance = 0.0


class GameEnvironment:
    """Game environment compatible with AI frameworks"""

    def __init__(self, physics_engine: PhysicsEngine, headless: bool = False) -> None:
        self.physics_engine = physics_engine
        self.headless = headless

        self.observation_processor = ObservationProcessor(
            physics_engine.field_width, physics_engine.field_height
        )

        self.reward_calculators = {1: RewardCalculator(), 2: RewardCalculator()}

        self.step_count = 0
        self.max_steps = ai_config.MAX_EPISODE_STEPS

    def reset(self) -> tuple[dict[str, Any], dict[str, Any]]:
        """
        Resets the environment

        Returns:
            Tuple: (observation_player1, observation_player2)
        """
        self.physics_engine.reset_game()
        self.step_count = 0

        for calc in self.reward_calculators.values():
            calc.reset()

        game_state = self.physics_engine.get_game_state()

        obs1 = self.observation_processor.process_game_state(game_state, 1)
        obs2 = self.observation_processor.process_game_state(game_state, 2)

        return obs1, obs2

    def step(
        self, action1: Action | None, action2: Action | None
    ) -> tuple[dict[str, Any], dict[str, Any], float, float, bool, dict[str, Any]]:
        """
        Performs a step in the environment

        Args:
            action1: Player 1 action
            action2: Player 2 action

        Returns:
            Tuple: (obs1, obs2, reward1, reward2, done, info)
        """
        # Handle None actions with default actions (no movement)
        if action1 is None:
            action1 = Action(move_x=0.0, move_y=0.0)
        if action2 is None:
            action2 = Action(move_x=0.0, move_y=0.0)

        # Update physics
        dt = 1.0 / 60.0  # 60 FPS
        if ai_config.HEADLESS_MODE:
            dt *= ai_config.FAST_MODE_MULTIPLIER

        events = self.physics_engine.update(dt, action1, action2)
        game_state = self.physics_engine.get_game_state()

        # Calculate rewards
        reward1 = self.reward_calculators[1].calculate_reward(game_state, events, 1)
        reward2 = self.reward_calculators[2].calculate_reward(game_state, events, 2)

        # Check if episode is finished
        done = self.physics_engine.is_game_over() or self.step_count >= self.max_steps

        # Create observations
        obs1 = self.observation_processor.process_game_state(game_state, 1)
        obs2 = self.observation_processor.process_game_state(game_state, 2)

        # Additional information
        info = {
            "events": events,
            "game_state": game_state,
            "winner": self.physics_engine.get_winner() if done else 0,
            "step_count": self.step_count,
        }

        self.step_count += 1

        return obs1, obs2, reward1, reward2, done, info

    def render(self) -> np.ndarray | None:
        """Environment rendering (to implement with graphical interface)"""
        if self.headless:
            return None
        # TODO: Implement with graphical renderer
        return None

    def close(self) -> None:
        """Closes the environment"""
        pass
