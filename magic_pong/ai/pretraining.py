"""
Pretraining module for DQN AI on the optimal point proximity task.

This module allows training the AI to approach the optimal ball interception
point before moving on to more complex training against opponents.
"""

import random
from typing import TYPE_CHECKING, Any

import numpy as np

from magic_pong.ai.interface import ObservationProcessor, RewardCalculator
from magic_pong.ai.models.dqn_ai import ACTION_MAPPING
from magic_pong.core.entities import Action, Paddle
from magic_pong.utils.config import ai_config, game_config, game_config_tmp

if TYPE_CHECKING:
    from magic_pong.ai.models.dqn_ai import DQNAgent


class OptimalPointPretrainer:
    """Pretraining class for optimal point learning"""

    def __init__(
        self,
        field_width: float | None = None,
        field_height: float | None = None,
        paddle_width: float | None = None,
        paddle_height: float | None = None,
        ball_radius: float | None = None,
        y_only: bool = False,
    ):
        """
        Args:
            field_width: Field width
            field_height: Field height
            paddle_width: Paddle width
            paddle_height: Paddle height
            ball_radius: Ball radius
            y_only: If True, only consider vertical distance for reward
        """
        self.field_width = field_width if field_width is not None else game_config.FIELD_WIDTH
        self.field_height = field_height if field_height is not None else game_config.FIELD_HEIGHT
        self.paddle_width = paddle_width if paddle_width is not None else game_config.PADDLE_WIDTH
        self.paddle_height = (
            paddle_height if paddle_height is not None else game_config.PADDLE_HEIGHT
        )
        self.ball_radius = ball_radius if ball_radius is not None else game_config.BALL_RADIUS

        # Define observation processor
        self.observation_processor = ObservationProcessor(self.field_width, self.field_height)

        # Margins to avoid edges
        self.margin = game_config.PADDLE_MARGIN

        # Reward calculator to use existing functions
        self.reward_calculator = RewardCalculator(y_only=y_only)

        self.player_config: dict[int, dict[str, int | dict[str, float]]] = {
            1: {
                "ball_spawn_zone": {
                    "x_min": self.field_width * 0.5,  # Middle of field towards right
                    "x_max": self.field_width - self.margin,
                    "y_min": self.margin,
                    "y_max": self.field_height - self.margin,
                },
                "paddle_zone": {
                    "x_min": self.margin,
                    "x_max": self.field_width * 0.5 - self.paddle_width - self.margin,
                    "y_min": self.margin,
                    "y_max": self.field_height - self.margin - self.paddle_height,
                },
                "direction": -1,  # Towards left
            },
            2: {
                "ball_spawn_zone": {
                    "x_min": self.margin,  # Middle of field towards left
                    "x_max": self.field_width * 0.5,
                    "y_min": self.margin,
                    "y_max": self.field_height - self.margin,
                },
                "paddle_zone": {
                    "x_min": self.field_width * 0.5 + self.paddle_width + self.margin,
                    "x_max": self.field_width - self.margin - self.paddle_width,
                    "y_min": self.margin,
                    "y_max": self.field_height - self.margin - self.paddle_height,
                },
                "direction": 1,  # Towards right
            },
        }

    def generate_random_ball_state(self, player_id: int = 1) -> dict[str, Any]:
        """
        Generate a random ball state heading towards the AI player's side.

        Args:
            player_id: AI player ID (1 for left, 2 for right)

        Returns:
            Dict containing ball position and velocity, and paddle position
        """
        ball_spawn_zone: dict[str, float] = self.player_config[player_id]["ball_spawn_zone"]  # type: ignore[assignment]
        paddle_zone: dict[str, float] = self.player_config[player_id]["paddle_zone"]  # type: ignore[assignment]

        # Determine direction based on player side
        ball_x = random.uniform(ball_spawn_zone["x_min"], ball_spawn_zone["x_max"])
        ball_y = random.uniform(ball_spawn_zone["y_min"], ball_spawn_zone["y_max"])

        direction = int(self.player_config[player_id]["direction"])  # type: ignore[arg-type]
        angle_rad = np.random.uniform(np.pi / 4, -np.pi / 4)
        ball_vx = direction * game_config.BALL_SPEED * np.cos(angle_rad)
        ball_vy = game_config.BALL_SPEED * np.sin(angle_rad)

        # Random paddle position within its zone
        paddle_x = np.random.uniform(paddle_zone["x_min"], paddle_zone["x_max"])
        paddle_y = np.random.uniform(paddle_zone["y_min"], paddle_zone["y_max"])

        return {
            "ball_position": (ball_x, ball_y),
            "ball_velocity": (ball_vx, ball_vy),
            "paddle_position": (paddle_x, paddle_y),
            "field_bounds": (0, self.field_width, 0, self.field_height),
        }

    def create_game_state_from_ball_state(
        self, ball_state: dict[str, Any], player_id: int = 1
    ) -> dict[str, Any]:
        """
        Create a complete game state from a simplified ball state.

        Args:
            ball_state: Ball state generated by generate_random_ball_state
            player_id: AI player ID

        Returns:
            Complete game state compatible with AI interface
        """
        # Opponent position (fixed, centered on their side)
        if player_id == 1:
            opponent_x = self.field_width - self.margin - self.paddle_width
        else:
            opponent_x = self.margin

        opponent_y = (self.field_height - self.paddle_height) / 2

        game_state = {
            "ball_position": ball_state["ball_position"],
            "ball_velocity": ball_state["ball_velocity"],
            f"player{player_id}_position": ball_state["paddle_position"],
            f"player{3-player_id}_position": (opponent_x, opponent_y),
            f"player{3-player_id}_last_position": (opponent_x, opponent_y),
            f"player{player_id}_paddle_size": self.paddle_height,
            f"player{3-player_id}_paddle_size": self.paddle_height,
            "active_bonuses": [],
            "rotating_paddles": [],
            "score": [np.random.randint(0, 10), np.random.randint(0, 10)],
            "time_elapsed": np.random.uniform(0, 300),  # Up to 5 minutes
            "field_bounds": ball_state["field_bounds"],
        }

        return game_state

    def _set_last_ball_distance(self, game_state: dict[str, Any], player_id: int) -> None:
        """
        Update the ball distance to optimal point in the reward calculator.

        Args:
            game_state: Complete game state
            player_id: Player ID
        """

        # Get positions and velocity from game state
        ball_pos = game_state.get("ball_position", (0, 0))
        ball_vel = game_state.get("ball_velocity", (0, 0))
        player_pos = game_state.get(f"player{player_id}_position", (0, 0))
        field_bounds = game_state.get("field_bounds", (0, 800, 0, 600))

        # Calculate paddle center
        paddle_center_x = player_pos[0]
        if player_id == 1:
            paddle_center_x += game_config.PADDLE_WIDTH
        paddle_center_y = player_pos[1] + game_config.PADDLE_HEIGHT / 2

        # Find optimal interception point on ball's trajectory
        optimal_point = self.reward_calculator._find_optimal_interception_point(
            ball_pos, ball_vel, (paddle_center_x, paddle_center_y), field_bounds, player_id
        )
        current_distance = float(
            np.linalg.norm(optimal_point - np.array((paddle_center_x, paddle_center_y)))
        )

        # Update in reward calculator
        self.reward_calculator.last_ball_distance[player_id] = current_distance

    def calculate_optimal_position_reward(
        self, game_state: dict[str, Any], player_id: int = 1, dt: float = 1.0 / 60.0
    ) -> tuple[float, dict[str, Any]]:
        """
        Calculate optimal point proximity reward with a system adapted for pretraining.

        Args:
            game_state: Complete game state
            player_id: Player ID
            dt: Time step

        Returns:
            Tuple (reward, detailed information)
        """
        # Temporarily enable proximity rewards
        original_use_proximity = ai_config.USE_PROXIMITY_REWARD
        ai_config.USE_PROXIMITY_REWARD = True

        try:
            # Calculate proximity reward
            proximity_reward = self.reward_calculator._calculate_proximity_reward(
                game_state, player_id
            )

            # Retrieve optimal point information
            optimal_points = self.reward_calculator.get_optimal_points()

            info = {
                "proximity_reward": proximity_reward,
                "optimal_points": optimal_points,
            }

            return proximity_reward, info

        finally:
            # Restore original configuration
            ai_config.USE_PROXIMITY_REWARD = original_use_proximity

    def simulate_paddle_movement(
        self,
        current_paddle_pos: tuple[float, float],
        action: Action,
        dt: float = 1.0 / 60.0,
        paddle_speed: float = 500.0,
    ) -> tuple[float, float]:
        """
        Simulate paddle movement based on action.

        Args:
            current_paddle_pos: Current paddle position (x, y)
            action: Action chosen by the neural network
            dt: Time step
            paddle_speed: Paddle speed

        Returns:
            New paddle position (x, y)
        """
        with game_config_tmp(
            FIELD_WIDTH=self.field_width,
            FIELD_HEIGHT=self.field_height,
            PADDLE_WIDTH=self.paddle_width,
            PADDLE_HEIGHT=self.paddle_height,
            PADDLE_MARGIN=self.margin,
        ):
            paddle_tmp = Paddle(*current_paddle_pos, player_id=1)
            paddle_tmp.move(action.move_x, action.move_y, dt)
            return paddle_tmp.position.x, paddle_tmp.position.y

    def pretraining_step(
        self, agent: "DQNAgent", player_id: int = 1, num_steps: int = 1000
    ) -> dict[str, Any]:
        """
        Perform a pretraining step on optimal point proximity.

        Args:
            agent: DQN agent to train
            player_id: Player ID
            num_steps: Number of pretraining steps

        Returns:
            Pretraining step statistics
        """
        total_reward = 0.0
        total_loss = 0.0
        loss_count = 0
        rewards_history = []

        agent.set_training_mode(True)
        dt = game_config.GAME_SPEED_MULTIPLIER / game_config.FPS

        for _ in range(num_steps):
            # Generate random ball state
            ball_state = self.generate_random_ball_state(player_id)
            game_state = self.create_game_state_from_ball_state(ball_state, player_id)
            initial_paddle_pos = game_state[f"player{player_id}_position"]

            # Initialize proximity reward before modifying system state
            self.reward_calculator._calculate_proximity_reward(game_state, player_id)

            # Convert state to observation for agent
            observation = self._game_state_to_observation(game_state, player_id)
            state = agent._observation_to_state(observation)

            # Agent chooses an action
            action_index = agent.act(state, training=True)
            action = self._index_to_action(action_index)

            # Simulate paddle movement
            new_paddle_pos = self.simulate_paddle_movement(initial_paddle_pos, action, dt=dt)

            # Update game state with new position
            game_state[f"player{player_id}_position"] = new_paddle_pos

            # Calculate proximity reward
            proximity_reward, info = self.calculate_optimal_position_reward(
                game_state, player_id, dt=dt
            )

            # Create next state (same state but with new paddle position)
            next_observation = self._game_state_to_observation(game_state, player_id)
            next_state = agent._observation_to_state(next_observation)

            # Store experience in replay memory
            agent.remember(state, action_index, proximity_reward, next_state, done=False)

            # Train agent if enough experiences
            if len(agent.memory) > agent.batch_size:
                loss = agent.replay()
                if loss is not None:
                    total_loss += loss
                    loss_count += 1

            total_reward += proximity_reward
            rewards_history.append(proximity_reward)

        # Calculate statistics
        avg_reward = total_reward / num_steps if num_steps > 0 else 0.0
        avg_loss = total_loss / loss_count if loss_count > 0 else 0.0

        return {
            "total_reward": total_reward,
            "average_reward": avg_reward,
            "average_loss": avg_loss,
            "steps": num_steps,
            "epsilon": agent.epsilon,
            "training_step": agent.training_step,
            "rewards_history": rewards_history,
        }

    def _game_state_to_observation(
        self, game_state: dict[str, Any], player_id: int
    ) -> dict[str, Any]:
        """
        Convert a game state to observation for the agent.
        Uses the same logic as ObservationProcessor.
        """
        return self.observation_processor.process_game_state(game_state, player_id)

    def _index_to_action(self, action_index: int) -> Action:
        """
        Convert an action index to Action object.
        Uses the same 3x3 grid as the DQN agent.
        """
        actions = ACTION_MAPPING

        if 0 <= action_index < len(actions):
            return actions[action_index]
        else:
            # Default action if invalid index
            print(f"Invalid action index: {action_index}, returning no movement.")
            return Action(move_x=0.0, move_y=0.0)

    def run_pretraining_phase(
        self,
        agent: "DQNAgent",
        total_steps: int = 10000,
        steps_per_batch: int = 1000,
        player_id: int = 1,
        verbose: bool = True,
    ) -> dict[str, Any]:
        """
        Execute a complete pretraining phase.

        Args:
            agent: DQN agent to pretrain
            total_steps: Total number of pretraining steps
            steps_per_batch: Number of steps per batch
            player_id: Player ID
            verbose: Display statistics

        Returns:
            Complete pretraining statistics
        """
        if verbose:
            print("🎯 Starting pretraining on optimal point")
            print(f"   Total steps: {total_steps}")
            print(f"   Steps per batch: {steps_per_batch}")

        all_rewards = []
        all_losses = []
        batch_stats = []

        remaining_steps = total_steps
        batch_num = 0

        while remaining_steps > 0:
            current_batch_steps = min(steps_per_batch, remaining_steps)
            batch_num += 1

            # Execute a pretraining batch
            batch_result = self.pretraining_step(agent, player_id, current_batch_steps)

            # Collect statistics
            all_rewards.extend(batch_result["rewards_history"])
            if batch_result["average_loss"] > 0:
                all_losses.append(batch_result["average_loss"])

            batch_stats.append(batch_result)

            if verbose:
                print(
                    f"   Batch {batch_num}: Avg reward = {batch_result['average_reward']:.3f}, "
                    f"Avg loss = {batch_result['average_loss']:.4f}, "
                    f"Epsilon = {batch_result['epsilon']:.3f}"
                )

            remaining_steps -= current_batch_steps

        # Calculate final statistics
        final_stats = {
            "total_steps": total_steps,
            "batches": batch_num,
            "final_epsilon": agent.epsilon,
            "final_training_step": agent.training_step,
            "average_reward": np.mean(all_rewards) if all_rewards else 0.0,
            "reward_std": np.std(all_rewards) if all_rewards else 0.0,
            "average_loss": np.mean(all_losses) if all_losses else 0.0,
            "batch_stats": batch_stats,
            "all_rewards": all_rewards,
        }

        if verbose:
            print("✅ Pretraining complete!")
            print(
                f"   Final avg reward: {final_stats['average_reward']:.3f} ± {final_stats['reward_std']:.3f}"
            )
            print(f"   Final avg loss: {final_stats['average_loss']:.4f}")
            print(f"   Final epsilon: {final_stats['final_epsilon']:.3f}")

        return final_stats


def create_pretrainer(**kwargs: Any) -> OptimalPointPretrainer:
    """
    Factory to create a pretrainer with default game configuration.

    Args:
        **kwargs: Additional arguments for the constructor

    Returns:
        OptimalPointPretrainer instance
    """
    defaults: dict[str, Any] = {
        "field_width": game_config.FIELD_WIDTH,
        "field_height": game_config.FIELD_HEIGHT,
        "paddle_width": game_config.PADDLE_WIDTH,
        "paddle_height": game_config.PADDLE_HEIGHT,
        "ball_radius": game_config.BALL_RADIUS,
    }

    # Merge with provided arguments
    defaults.update(kwargs)

    return OptimalPointPretrainer(**defaults)
