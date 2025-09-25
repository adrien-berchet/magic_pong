"""
Agnostic AI interface for Magic Pong
"""

from abc import ABC, abstractmethod
from typing import Any

import numpy as np
from magic_pong.core.entities import Action
from magic_pong.core.physics import PhysicsEngine
from magic_pong.utils.config import ai_config, game_config


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
        self.last_ball_distance: dict[int, float] = {}
        self.optimal_points: dict[int, dict] = {}  # Store optimal points for visualization

    def calculate_reward(
        self, game_state: dict[str, Any], events: dict[str, list], player_id: int
    ) -> float:
        """
        Calculates reward for a player based on events and proximity to ball

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

        # PROXIMITY-BASED REWARD SHAPING
        # Reward/penalize based on distance change to the ball
        if ai_config.USE_PROXIMITY_REWARD:
            proximity_reward = self._calculate_proximity_reward(game_state, player_id)
            reward += proximity_reward

        return reward

    def _calculate_proximity_reward(self, game_state: dict[str, Any], player_id: int) -> float:
        """
        Calculate reward based on proximity to the optimal interception point on ball trajectory

        Args:
            game_state: Current game state containing positions
            player_id: Player ID (1 or 2)

        Returns:
            float: Proximity reward (positive for getting closer, negative for moving away)
        """
        if not ai_config.USE_PROXIMITY_REWARD:
            return 0.0

        # Get positions and velocity from game state
        ball_pos = game_state.get("ball_position", (0, 0))
        ball_vel = game_state.get("ball_velocity", (0, 0))
        player_pos = game_state.get(f"player{player_id}_position", (0, 0))
        field_bounds = game_state.get("field_bounds", (0, 800, 0, 600))

        # Calculate paddle center
        paddle_center_x = player_pos[0]
        if player_id == 1:
            paddle_center_x += game_config.PADDLE_WIDTH / 2
        paddle_center_y = player_pos[1] + game_config.PADDLE_HEIGHT / 2

        # Find optimal interception point on ball's trajectory
        optimal_point = self._find_optimal_interception_point(
            ball_pos, ball_vel, (paddle_center_x, paddle_center_y), field_bounds, player_id
        )

        if optimal_point is None:
            # Fallback to current ball position if trajectory calculation fails
            optimal_point = (paddle_center_x, ball_pos[1])

        # Store optimal point for visualization
        self.optimal_points[player_id] = {
            "position": optimal_point,
            "ball_position": ball_pos,
            "ball_velocity": ball_vel,
            "paddle_position": (paddle_center_x, paddle_center_y),
        }

        # Debug display of optimal points
        if ai_config.DEBUG_OPTIMAL_POINTS:
            self._debug_optimal_point(
                player_id, ball_pos, ball_vel, (paddle_center_x, paddle_center_y), optimal_point
            )

        # Calculate current distance to optimal interception point
        current_distance = np.linalg.norm(
            optimal_point - np.array((paddle_center_x, paddle_center_y))
        )

        # Get previous distance for this player
        previous_distance = self.last_ball_distance.get(player_id, None)

        # Update stored distance for next calculation
        self.last_ball_distance[player_id] = float(current_distance)

        if previous_distance is None:
            return 0.0

        # Calculate distance change (negative means getting closer)
        distance_change = current_distance - previous_distance

        # Calculate proximity reward
        proximity_reward = 0.0
        if distance_change < 0:  # Getting closer to optimal interception point
            # Reward for getting closer, scaled by how much closer
            proximity_reward = ai_config.PROXIMITY_REWARD_FACTOR
            # proximity_reward = min(
            #     abs(distance_change) * ai_config.PROXIMITY_REWARD_FACTOR,
            #     ai_config.MAX_PROXIMITY_REWARD,
            # )
        elif distance_change >= 0:  # Moving away from optimal interception point
            # Small penalty for moving away
            proximity_reward = -ai_config.PROXIMITY_PENALTY_FACTOR
            # proximity_reward = -min(
            #     distance_change * ai_config.PROXIMITY_PENALTY_FACTOR + ai_config.MAX_PROXIMITY_REWARD / 4,
            #     ai_config.MAX_PROXIMITY_REWARD,
            # )

        return proximity_reward

    def _find_optimal_interception_point(
        self,
        ball_pos: tuple,
        ball_vel: tuple,
        paddle_pos: tuple,
        field_bounds: tuple,
        player_id: int,
    ) -> tuple | None:
        """
        Find the optimal interception point on the ball's trajectory considering wall bounces

        Args:
            ball_pos: Current ball position (x, y)
            ball_vel: Ball velocity (vx, vy)
            paddle_pos: Paddle center position (x, y)
            field_bounds: Field boundaries (min_x, max_x, min_y, max_y)

        Returns:
            Optimal interception point (x, y) or None if no valid trajectory
        """
        # if abs(ball_vel[0]) < 0.1 and abs(ball_vel[1]) < 0.1:
        #     # Ball is nearly stationary, return current position
        #     return ball_pos

        # For stability, we use a hybrid approach:
        # 1. If ball is moving towards paddle side, predict interception
        # 2. Otherwise, reward positioning towards current ball position

        min_x, max_x, min_y, max_y = field_bounds

        # Determine paddle side (left = player 1, right = player 2)
        is_left_paddle = player_id == 1

        # Check if ball is moving towards this paddle
        ball_moving_towards = (is_left_paddle and ball_vel[0] < 0) or (
            not is_left_paddle and ball_vel[0] > 0
        )

        if not ball_moving_towards:
            # Ball moving away or sideways, reward staying near current ball Y position
            return None

        # Ball is moving towards paddle, find best interception point
        trajectory_points = self._simulate_ball_trajectory(
            ball_pos, ball_vel, field_bounds, max_time=10.0, dt=0.05
        )

        if not trajectory_points:
            return ball_pos

        # Find closest interception points on paddle side
        # paddle_x_zone = paddle_pos[0]
        # best_point = ball_pos
        # min_distance = float('inf')

        trajectory_pts = np.array([i[0] for i in trajectory_points])
        distances = np.linalg.norm(trajectory_pts - paddle_pos, axis=1)
        best_point = trajectory_pts[np.argmin(distances)]

        # for point, time_step in trajectory_points:
        #     # Only consider points that are reasonably close to paddle's X zone
        #     x_distance_to_paddle_zone = abs(point[0] - paddle_x_zone)

        #     if x_distance_to_paddle_zone < 250:  # Within reasonable reach
        #         # Calculate reachability: distance vs time available
        #         y_distance = abs(point[1] - paddle_pos[1])

        #         # Estimate if paddle can reach this point in time
        #         paddle_speed = 500  # From config
        #         time_needed = y_distance / paddle_speed

        #         if time_needed <= time_step + 0.5:  # Add some tolerance
        #             # This point is reachable, calculate priority
        #             priority = y_distance + x_distance_to_paddle_zone * 0.1

        #             if priority < min_distance:
        #                 min_distance = priority
        #                 best_point = point

        return best_point

    def _simulate_ball_trajectory(
        self,
        ball_pos: tuple,
        ball_vel: tuple,
        field_bounds: tuple,
        max_time: float = 10.0,
        dt: float = 0.1,
    ) -> list:
        """
        Simulate ball trajectory with wall bounces

        Args:
            ball_pos: Starting ball position (x, y)
            ball_vel: Ball velocity (vx, vy)
            field_bounds: Field boundaries (min_x, max_x, min_y, max_y)
            max_time: Maximum simulation time
            dt: Time step for simulation

        Returns:
            List of (position, time) tuples along the trajectory
        """
        min_x, max_x, min_y, max_y = field_bounds
        ball_radius = game_config.BALL_RADIUS

        # Current state
        pos_x, pos_y = ball_pos
        vel_x, vel_y = ball_vel

        trajectory = [(ball_pos, 0.0)]
        current_time = 0.0

        while current_time < max_time:
            # Predict next position
            next_x = pos_x + vel_x * dt
            next_y = pos_y + vel_y * dt

            # Check for wall bounces (top and bottom walls only, like in the game)
            if next_y - ball_radius <= min_y:
                # Bottom wall bounce
                next_y = min_y + ball_radius
                vel_y = -vel_y
            elif next_y + ball_radius >= max_y:
                # Top wall bounce
                next_y = max_y - ball_radius
                vel_y = -vel_y

            # Check for left/right boundaries (goals) - stop simulation
            if next_x - ball_radius <= min_x or next_x + ball_radius >= max_x:
                # Ball would reach goal area, add final point and stop
                trajectory.append(((next_x, next_y), current_time + dt))
                break

            # Update position
            pos_x, pos_y = next_x, next_y
            current_time += dt

            # Add point to trajectory
            trajectory.append(((pos_x, pos_y), current_time))

            # Stop if ball gets too close to goals (to avoid infinite simulation)
            if pos_x < min_x + 100 or pos_x > max_x - 100:
                break

        return trajectory

    def _debug_optimal_point(
        self,
        player_id: int,
        ball_pos: tuple,
        ball_vel: tuple,
        paddle_pos: tuple,
        optimal_point: tuple,
    ) -> None:
        """
        Debug display of optimal interception point information

        Args:
            player_id: Player ID (1 or 2)
            ball_pos: Current ball position
            ball_vel: Ball velocity
            paddle_pos: Paddle center position
            optimal_point: Calculated optimal point
        """
        # Only display every few calls to avoid spam
        if not hasattr(self, "_debug_counter"):
            self._debug_counter = {1: 0, 2: 0}

        self._debug_counter[player_id] += 1

        # Display every 30 calls (roughly every 0.5 seconds at 60fps)
        if self._debug_counter[player_id] % 30 == 0:
            ball_speed = (ball_vel[0] ** 2 + ball_vel[1] ** 2) ** 0.5
            distance_to_optimal = (
                (optimal_point[0] - paddle_pos[0]) ** 2 + (optimal_point[1] - paddle_pos[1]) ** 2
            ) ** 0.5

            # Determine ball direction
            if abs(ball_vel[0]) > abs(ball_vel[1]):
                direction = "→ droite" if ball_vel[0] > 0 else "← gauche"
            else:
                direction = "↑ haut" if ball_vel[1] > 0 else "↓ bas"

            # Check if ball is moving towards this player
            is_approaching = (player_id == 1 and ball_vel[0] < 0) or (
                player_id == 2 and ball_vel[0] > 0
            )

            print(
                f"🎯 P{player_id}: Balle({ball_pos[0]:.0f},{ball_pos[1]:.0f}) {direction} v={ball_speed:.0f}"
            )
            print(
                f"   Raquette({paddle_pos[0]:.0f},{paddle_pos[1]:.0f}) → Optimal({optimal_point[0]:.0f},{optimal_point[1]:.0f}) dist={distance_to_optimal:.0f}"
            )
            print(f"   {'🎯 Se rapproche!' if is_approaching else '🛡️ Défensif'}")

    def get_optimal_points(self) -> dict:
        """Get current optimal points for visualization"""
        return self.optimal_points.copy()

    def reset(self) -> None:
        """Resets the calculator"""
        self.last_score = [0, 0]
        self.last_ball_distance = {}
        self.optimal_points = {}
        # Reset debug counter
        if hasattr(self, "_debug_counter"):
            self._debug_counter = {1: 0, 2: 0}


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
            "optimal_points": {
                1: self.reward_calculators[1].get_optimal_points().get(1),
                2: self.reward_calculators[2].get_optimal_points().get(2),
            },
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
