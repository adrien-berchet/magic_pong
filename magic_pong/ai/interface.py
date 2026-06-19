"""
Agnostic AI interface for Magic Pong
"""

from abc import abstractmethod
from typing import Any

import numpy as np

from magic_pong.core.entities import Action
from magic_pong.core.entities import Player
from magic_pong.core.physics import PhysicsEngine
from magic_pong.utils.config import ai_config
from magic_pong.utils.config import game_config


class AIPlayer(Player):
    """
    A generic AI player that can be extended for specific AI behaviors.
    """

    def __init__(self, name: str = "AIPlayer"):
        super().__init__(name)
        self.episode_rewards: list[float] = []
        self.current_episode_reward = 0.0

    @abstractmethod
    def get_action(self, observation: dict[str, Any] | None) -> Action:
        """
        Returns the action to perform based on the observation

        Args:
            observation: Normalized game state (can be None)

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

    def on_episode_end(self) -> None:
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
        observation: dict[str, Any] = {}

        ball_x, ball_y = game_state["ball_position"]
        player_x, player_y = game_state[f"player{player_id}_position"]
        opponent_x, opponent_y = game_state[f"player{3 - player_id}_position"]
        opponent_previous_x, opponent_previous_y = game_state.get(
            f"player{3 - player_id}_last_position",
            game_state[f"player{3 - player_id}_position"],
        )

        # Normalized positions
        if ai_config.NORMALIZE_POSITIONS:
            ball_x = ball_x / self.field_width
            ball_y = ball_y / self.field_height
            player_x = player_x / self.field_width
            player_y = player_y / self.field_height
            opponent_x = opponent_x / self.field_width
            opponent_y = opponent_y / self.field_height
            opponent_previous_x = opponent_previous_x / self.field_width
            opponent_previous_y = opponent_previous_y / self.field_height

        observation["ball_pos"] = [ball_x, ball_y]
        observation["player_pos"] = [player_x, player_y]
        observation["opponent_pos"] = [opponent_x, opponent_y]
        observation["opponent_previous_pos"] = [opponent_previous_x, opponent_previous_y]
        observation["field_width"] = self.field_width
        observation["field_height"] = self.field_height

        # Ball velocity
        vel_x, vel_y = game_state["ball_velocity"]
        if ai_config.NORMALIZE_POSITIONS:
            # Normalize by approximate max speed
            vel_x = vel_x / game_config.MAX_BALL_SPEED
            vel_y = vel_y / game_config.MAX_BALL_SPEED
        observation["ball_vel"] = [vel_x, vel_y]

        # Paddle sizes
        observation["player_paddle_size"] = game_state[f"player{player_id}_paddle_size"]
        observation["opponent_paddle_size"] = game_state[f"player{3 - player_id}_paddle_size"]

        # Active bonuses
        bonuses = []
        for bonus_x, bonus_y, bonus_type in game_state["active_bonuses"]:
            if ai_config.NORMALIZE_POSITIONS:
                bonus_x = bonus_x / self.field_width
                bonus_y = bonus_y / self.field_height
            bonuses.append([bonus_x, bonus_y, self._encode_bonus_type(bonus_type)])
        observation["bonuses"] = bonuses

        # Rotating paddles
        rotating_paddles = []
        for rp_x, rp_y, rp_angle in game_state["rotating_paddles"]:
            if ai_config.NORMALIZE_POSITIONS:
                rp_x = rp_x / self.field_width
                rp_y = rp_y / self.field_height
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

    def __init__(self, y_only: bool = False) -> None:
        self.last_score = [0, 0]
        self.last_ball_distance: dict[int, float] = {}
        self.optimal_points: dict[int, dict] = {}  # Store optimal points for visualization
        self._last_interception_details: dict[int, dict[str, Any]] = {}
        self.y_only = y_only

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
                reward += ai_config.PADDLE_HIT_REWARD

        # Reward for rotating paddles
        for hit in events.get("rotating_paddle_hits", []):
            if hit["player"] == player_id:
                reward += ai_config.PADDLE_HIT_REWARD * 2  # Bonus for using rotating paddle

        if events.get("goals"):
            return reward

        if ai_config.REWARD_SHAPING_MODE == "phase3":
            reward += self._calculate_phase3_reward(game_state, events, player_id)
        elif ai_config.USE_PROXIMITY_REWARD:
            proximity_reward = self._calculate_proximity_reward(game_state, player_id)
            reward += proximity_reward

        return reward

    def _calculate_phase3_reward(
        self, game_state: dict[str, Any], events: dict[str, list], player_id: int
    ) -> float:
        """Calculate opt-in Phase 3 shaping without changing terminal rewards."""
        return self._calculate_phase3_intercept_reward(
            game_state, player_id
        ) + self._calculate_phase3_successful_return_reward(game_state, events, player_id)

    def _calculate_phase3_intercept_reward(
        self, game_state: dict[str, Any], player_id: int
    ) -> float:
        ball_pos = game_state.get("ball_position", (0.0, 0.0))
        ball_vel = game_state.get("ball_velocity", (0.0, 0.0))
        if not self._ball_is_approaching_player(ball_vel, player_id):
            self.optimal_points.pop(player_id, None)
            return 0.0

        field_bounds = game_state.get(
            "field_bounds", (0.0, game_config.FIELD_WIDTH, 0.0, game_config.FIELD_HEIGHT)
        )
        paddle_center_x, paddle_center_y, previous_center_y, paddle_height = (
            self._phase3_paddle_geometry(game_state, player_id)
        )
        intercept_x = self._phase3_intercept_x(game_state, player_id)
        intercept_y = self._predict_ball_y_at_x(ball_pos, ball_vel, intercept_x, field_bounds)
        if intercept_y is None:
            self.optimal_points.pop(player_id, None)
            return 0.0

        optimal_point = (paddle_center_x, intercept_y)
        self.optimal_points[player_id] = {
            "position": optimal_point,
            "ball_position": ball_pos,
            "ball_velocity": ball_vel,
            "paddle_position": (paddle_center_x, paddle_center_y),
        }

        if ai_config.DEBUG_OPTIMAL_POINTS:
            self._debug_optimal_point(
                player_id, ball_pos, ball_vel, (paddle_center_x, paddle_center_y), optimal_point
            )

        current_distance = abs(intercept_y - paddle_center_y)
        previous_distance = abs(intercept_y - previous_center_y)
        movement_scale = max(paddle_height, 1.0)
        field_height = max(float(field_bounds[3]) - float(field_bounds[2]), 1.0)

        reward = 0.0
        distance_delta = previous_distance - current_distance
        if distance_delta > 1e-6:
            progress = min(distance_delta / movement_scale, 1.0)
            reward += ai_config.PHASE3_INTERCEPT_PROGRESS_REWARD * progress
        elif distance_delta < -1e-6:
            regress = min(abs(distance_delta) / movement_scale, 1.0)
            reward -= ai_config.PHASE3_INTERCEPT_DISTANCE_PENALTY * 0.5 * regress

        distance_deadband = paddle_height / 2
        excess_distance = max(current_distance - distance_deadband, 0.0)
        far_from_intercept = min(excess_distance / max(field_height / 2, 1.0), 1.0)
        reward -= ai_config.PHASE3_INTERCEPT_DISTANCE_PENALTY * far_from_intercept
        return reward

    def _calculate_phase3_successful_return_reward(
        self, game_state: dict[str, Any], events: dict[str, list], player_id: int
    ) -> float:
        ball_vel = game_state.get("ball_velocity", (0.0, 0.0))
        if not self._ball_is_moving_toward_opponent(ball_vel, player_id):
            return 0.0

        for hit in events.get("paddle_hits", []):
            if hit.get("player") == player_id:
                return ai_config.PHASE3_SUCCESSFUL_RETURN_REWARD
        return 0.0

    def _phase3_paddle_geometry(
        self, game_state: dict[str, Any], player_id: int
    ) -> tuple[float, float, float, float]:
        player_pos = game_state.get(f"player{player_id}_position", (0.0, 0.0))
        previous_pos = game_state.get(
            f"player{player_id}_prev_position",
            game_state.get(f"player{player_id}_last_position", player_pos),
        )
        paddle_height = float(
            game_state.get(f"player{player_id}_paddle_size", game_config.PADDLE_HEIGHT)
        )
        paddle_center_x = float(player_pos[0]) + game_config.PADDLE_WIDTH / 2
        paddle_center_y = float(player_pos[1]) + paddle_height / 2
        previous_center_y = float(previous_pos[1]) + paddle_height / 2
        return paddle_center_x, paddle_center_y, previous_center_y, paddle_height

    def _phase3_intercept_x(self, game_state: dict[str, Any], player_id: int) -> float:
        player_pos = game_state.get(f"player{player_id}_position", (0.0, 0.0))
        if player_id == 1:
            return float(player_pos[0]) + game_config.PADDLE_WIDTH + game_config.BALL_RADIUS
        return float(player_pos[0]) - game_config.BALL_RADIUS

    def _predict_ball_y_at_x(
        self,
        ball_pos: tuple[float, float],
        ball_vel: tuple[float, float],
        target_x: float,
        field_bounds: tuple[float, float, float, float],
    ) -> float | None:
        vel_x = float(ball_vel[0])
        if abs(vel_x) < 1e-9:
            return None

        time_to_target = (target_x - float(ball_pos[0])) / vel_x
        if time_to_target < 0:
            return None

        raw_y = float(ball_pos[1]) + float(ball_vel[1]) * time_to_target
        min_y = float(field_bounds[2]) + game_config.BALL_RADIUS
        max_y = float(field_bounds[3]) - game_config.BALL_RADIUS
        return self._reflect_y_within_bounds(raw_y, min_y, max_y)

    def _reflect_y_within_bounds(self, y: float, min_y: float, max_y: float) -> float:
        if max_y <= min_y:
            return min_y

        height = max_y - min_y
        offset = (y - min_y) % (height * 2)
        if offset > height:
            offset = height * 2 - offset
        return min_y + offset

    def _ball_is_approaching_player(self, ball_vel: tuple[float, float], player_id: int) -> bool:
        return (player_id == 1 and ball_vel[0] < -1e-9) or (player_id == 2 and ball_vel[0] > 1e-9)

    def _ball_is_moving_toward_opponent(
        self, ball_vel: tuple[float, float], player_id: int
    ) -> bool:
        return (player_id == 1 and ball_vel[0] > 1e-9) or (player_id == 2 and ball_vel[0] < -1e-9)

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

        paddle_height = float(
            game_state.get(f"player{player_id}_paddle_size", game_config.PADDLE_HEIGHT)
        )

        # Calculate paddle center
        paddle_center_x = player_pos[0] + game_config.PADDLE_WIDTH / 2
        paddle_center_y = player_pos[1] + paddle_height / 2

        # Find optimal interception point on ball's trajectory
        self._last_interception_details.pop(player_id, None)
        optimal_point = self._find_optimal_interception_point(
            ball_pos,
            ball_vel,
            (paddle_center_x, paddle_center_y),
            field_bounds,
            player_id,
            self.y_only,
            paddle_height,
        )

        if optimal_point is None:
            self.optimal_points.pop(player_id, None)
            self.last_ball_distance.pop(player_id, None)
            return 0.0

        # Store optimal point for visualization
        self.optimal_points[player_id] = {
            "position": optimal_point,
            "ball_position": ball_pos,
            "ball_velocity": ball_vel,
            "paddle_position": (paddle_center_x, paddle_center_y),
        }
        self.optimal_points[player_id].update(self._last_interception_details.get(player_id, {}))

        # Debug display of optimal points
        if ai_config.DEBUG_OPTIMAL_POINTS:
            self._debug_optimal_point(
                player_id, ball_pos, ball_vel, (paddle_center_x, paddle_center_y), optimal_point
            )

        # Calculate current distance to optimal interception point
        if self.y_only:
            current_distance = abs(optimal_point[1] - paddle_center_y)
        else:
            current_distance = np.linalg.norm(
                optimal_point - np.array((paddle_center_x, paddle_center_y))
            )

        # Get previous distance for this player
        previous_distance = self.last_ball_distance.get(player_id, None)

        # Update stored distance for next calculation
        self.last_ball_distance[player_id] = float(current_distance)

        if previous_distance is None:
            return 0.0

        # Reward proportional to distance improvement: positive when getting
        # closer, negative when moving away.  Magnitude reflects how much
        # better/worse the action was, so Q-learning can distinguish a large
        # stride toward the target from a tiny one.
        distance_change = current_distance - previous_distance
        reward = -distance_change * ai_config.PROXIMITY_REWARD_FACTOR
        current_paddle_covers_trajectory = bool(
            self._last_interception_details.get(player_id, {}).get(
                "current_paddle_covers_trajectory", False
            )
        )
        if (
            current_distance > self._proximity_target_deadband(paddle_height)
            and distance_change >= -1e-9
            and not current_paddle_covers_trajectory
        ):
            reward -= ai_config.PROXIMITY_PENALTY_FACTOR
        return reward

    def _proximity_target_deadband(self, paddle_height: float) -> float:
        del paddle_height
        return 1.0

    def _find_optimal_interception_point(
        self,
        ball_pos: tuple[float, float],
        ball_vel: tuple[float, float],
        paddle_pos: tuple[float, float],
        field_bounds: tuple[float, float, float, float],
        player_id: int,
        y_only: bool = False,
        paddle_height: float | None = None,
    ) -> tuple[float, float] | None:
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

        # Check if ball is moving towards this paddle
        ball_moving_towards = self._ball_is_approaching_player(ball_vel, player_id)

        if not ball_moving_towards:
            # Ball moving away or sideways, reward staying near current ball Y position
            self._last_interception_details.pop(player_id, None)
            return None

        trajectory_points = self._simulate_ball_trajectory(
            ball_pos, ball_vel, field_bounds, max_time=None
        )

        if not trajectory_points:
            return ball_pos

        paddle_height = float(paddle_height or game_config.PADDLE_HEIGHT)
        current_paddle_covers_trajectory = self._ball_path_crosses_current_paddle(
            trajectory_points, paddle_pos, player_id, paddle_height
        )
        interception = self._find_earliest_reachable_interception(
            trajectory_points, paddle_pos, field_bounds, player_id, y_only, paddle_height
        )
        if interception is None:
            self._last_interception_details.pop(player_id, None)
            return None

        self._last_interception_details[player_id] = {
            "ball_interception_position": interception["ball_position"],
            "paddle_target_position": interception["paddle_target_position"],
            "interception_time": interception["time"],
            "reachable": interception["reachable"],
            "current_paddle_covers_trajectory": current_paddle_covers_trajectory,
        }
        best_point = interception["paddle_target_position"]

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

    def _ball_path_crosses_current_paddle(
        self,
        trajectory_points: list[tuple[tuple[float, float], float]],
        paddle_pos: tuple[float, float],
        player_id: int,
        paddle_height: float,
    ) -> bool:
        """Return True when the static current paddle already covers the ball path."""
        face_x = self._paddle_face_ball_center_x(float(paddle_pos[0]), player_id)
        crossing = self._trajectory_crossing_at_x(trajectory_points, face_x)
        if crossing is None:
            return False

        ball_position, _time_step = crossing
        half_height = paddle_height / 2.0
        min_contact_y = float(paddle_pos[1]) - half_height - game_config.BALL_RADIUS
        max_contact_y = float(paddle_pos[1]) + half_height + game_config.BALL_RADIUS
        return min_contact_y <= float(ball_position[1]) <= max_contact_y

    def _find_earliest_reachable_interception(
        self,
        trajectory_points: list[tuple[tuple[float, float], float]],
        paddle_pos: tuple[float, float],
        field_bounds: tuple[float, float, float, float],
        player_id: int,
        y_only: bool,
        paddle_height: float,
    ) -> dict[str, Any] | None:
        """Find the first legal ball trajectory point the paddle can reach."""
        if y_only:
            return self._find_y_only_interception(
                trajectory_points, paddle_pos, field_bounds, player_id, paddle_height
            )

        for start, end in zip(trajectory_points, trajectory_points[1:], strict=False):
            segment_interval = self._legal_interception_time_interval(
                start, end, field_bounds, player_id, paddle_height
            )
            if segment_interval is None:
                continue

            interval_start, interval_end = segment_interval
            candidate = self._interception_candidate_at_time(
                start, end, interval_start, field_bounds, player_id, paddle_pos, paddle_height
            )
            if candidate is not None and self._can_reach_interception(candidate, paddle_pos, False):
                return candidate

            end_candidate = self._interception_candidate_at_time(
                start, end, interval_end, field_bounds, player_id, paddle_pos, paddle_height
            )
            if end_candidate is None or not self._can_reach_interception(
                end_candidate, paddle_pos, False
            ):
                continue

            low = interval_start
            high = interval_end
            for _ in range(32):
                mid = (low + high) / 2.0
                mid_candidate = self._interception_candidate_at_time(
                    start, end, mid, field_bounds, player_id, paddle_pos, paddle_height
                )
                if mid_candidate is not None and self._can_reach_interception(
                    mid_candidate, paddle_pos, False
                ):
                    high = mid
                else:
                    low = mid
            return self._interception_candidate_at_time(
                start, end, high, field_bounds, player_id, paddle_pos, paddle_height
            )

        return None

    def _find_y_only_interception(
        self,
        trajectory_points: list[tuple[tuple[float, float], float]],
        paddle_pos: tuple[float, float],
        field_bounds: tuple[float, float, float, float],
        player_id: int,
        paddle_height: float,
    ) -> dict[str, Any] | None:
        face_x = self._paddle_face_ball_center_x(paddle_pos[0], player_id)
        crossing = self._trajectory_crossing_at_x(trajectory_points, face_x)
        if crossing is None:
            return None

        ball_position, time_step = crossing
        target_y = self._nearest_legal_paddle_center_y(
            ball_position[1], paddle_pos[1], field_bounds, paddle_height
        )
        if target_y is None:
            return None

        candidate = {
            "ball_position": ball_position,
            "paddle_target_position": (float(paddle_pos[0]), target_y),
            "time": float(time_step),
            "reachable": True,
        }
        if not self._can_reach_interception(candidate, paddle_pos, True):
            return None
        return candidate

    def _legal_interception_time_interval(
        self,
        start: tuple[tuple[float, float], float],
        end: tuple[tuple[float, float], float],
        field_bounds: tuple[float, float, float, float],
        player_id: int,
        paddle_height: float,
    ) -> tuple[float, float] | None:
        start_time = float(start[1])
        end_time = float(end[1])
        if end_time < start_time:
            return None

        ball_x_min, ball_x_max = self._legal_ball_x_intercept_range(
            field_bounds, player_id, paddle_height
        )
        return self._segment_time_interval_for_value_range(
            start_time,
            end_time,
            float(start[0][0]),
            float(end[0][0]),
            ball_x_min,
            ball_x_max,
        )

    def _legal_ball_x_intercept_range(
        self,
        field_bounds: tuple[float, float, float, float],
        player_id: int,
        paddle_height: float,
    ) -> tuple[float, float]:
        center_min_x, center_max_x, _center_min_y, _center_max_y = self._legal_paddle_center_bounds(
            field_bounds, player_id, paddle_height
        )
        contact_offset = game_config.BALL_RADIUS + game_config.PADDLE_WIDTH / 2
        if player_id == 1:
            return center_min_x + contact_offset, center_max_x + contact_offset
        return center_min_x - contact_offset, center_max_x - contact_offset

    def _segment_time_interval_for_value_range(
        self,
        start_time: float,
        end_time: float,
        start_value: float,
        end_value: float,
        min_value: float,
        max_value: float,
    ) -> tuple[float, float] | None:
        if end_time == start_time:
            if min_value <= start_value <= max_value:
                return start_time, end_time
            return None

        if abs(end_value - start_value) <= 1e-12:
            if min_value <= start_value <= max_value:
                return start_time, end_time
            return None

        alpha_min = (min_value - start_value) / (end_value - start_value)
        alpha_max = (max_value - start_value) / (end_value - start_value)
        low_alpha = max(0.0, min(alpha_min, alpha_max))
        high_alpha = min(1.0, max(alpha_min, alpha_max))
        if low_alpha > high_alpha:
            return None

        duration = end_time - start_time
        return start_time + low_alpha * duration, start_time + high_alpha * duration

    def _interception_candidate_at_time(
        self,
        start: tuple[tuple[float, float], float],
        end: tuple[tuple[float, float], float],
        time_step: float,
        field_bounds: tuple[float, float, float, float],
        player_id: int,
        paddle_pos: tuple[float, float],
        paddle_height: float,
    ) -> dict[str, Any] | None:
        start_time = float(start[1])
        end_time = float(end[1])
        if end_time == start_time:
            alpha = 0.0
        else:
            alpha = np.clip((time_step - start_time) / (end_time - start_time), 0.0, 1.0)
        ball_x = float(start[0][0]) + (float(end[0][0]) - float(start[0][0])) * float(alpha)
        ball_y = float(start[0][1]) + (float(end[0][1]) - float(start[0][1])) * float(alpha)
        paddle_target = self._paddle_target_for_ball_position(
            (ball_x, ball_y), field_bounds, player_id, paddle_pos, paddle_height
        )
        if paddle_target is None:
            return None
        return {
            "ball_position": (ball_x, ball_y),
            "paddle_target_position": paddle_target,
            "time": float(time_step),
            "reachable": True,
        }

    def _paddle_target_for_ball_position(
        self,
        ball_position: tuple[float, float],
        field_bounds: tuple[float, float, float, float],
        player_id: int,
        paddle_pos: tuple[float, float],
        paddle_height: float,
    ) -> tuple[float, float] | None:
        center_min_x, center_max_x, center_min_y, center_max_y = self._legal_paddle_center_bounds(
            field_bounds, player_id, paddle_height
        )
        contact_offset = game_config.BALL_RADIUS + game_config.PADDLE_WIDTH / 2
        if player_id == 1:
            target_x = float(ball_position[0]) - contact_offset
        else:
            target_x = float(ball_position[0]) + contact_offset
        target_x = float(np.clip(target_x, center_min_x, center_max_x))
        target_y = self._nearest_legal_paddle_center_y(
            float(ball_position[1]), float(paddle_pos[1]), field_bounds, paddle_height
        )
        if target_y is None:
            return None
        return (target_x, target_y)

    def _legal_paddle_center_bounds(
        self,
        field_bounds: tuple[float, float, float, float],
        player_id: int,
        paddle_height: float,
    ) -> tuple[float, float, float, float]:
        min_x, max_x, min_y, max_y = map(float, field_bounds)
        half_width = game_config.PADDLE_WIDTH / 2
        half_height = paddle_height / 2
        if player_id == 1:
            center_min_x = min_x + game_config.PADDLE_MARGIN + half_width
            center_max_x = (max_x - min_x) / 2 + min_x - game_config.PADDLE_MARGIN - half_width
        else:
            center_min_x = (max_x - min_x) / 2 + min_x + game_config.PADDLE_MARGIN + half_width
            center_max_x = max_x - game_config.PADDLE_MARGIN - half_width
        return (
            float(center_min_x),
            float(center_max_x),
            min_y + half_height,
            max_y - half_height,
        )

    def _nearest_legal_paddle_center_y(
        self,
        ball_y: float,
        current_center_y: float,
        field_bounds: tuple[float, float, float, float],
        paddle_height: float,
    ) -> float | None:
        _center_min_x, _center_max_x, field_center_min_y, field_center_max_y = (
            self._legal_paddle_center_bounds(field_bounds, 1, paddle_height)
        )
        half_height = paddle_height / 2
        contact_min_y = max(field_center_min_y, ball_y - half_height)
        contact_max_y = min(field_center_max_y, ball_y + half_height)
        if contact_min_y > contact_max_y:
            return None
        return float(np.clip(current_center_y, contact_min_y, contact_max_y))

    def _paddle_face_ball_center_x(self, paddle_center_x: float, player_id: int) -> float:
        contact_offset = game_config.BALL_RADIUS + game_config.PADDLE_WIDTH / 2
        if player_id == 1:
            return float(paddle_center_x) + contact_offset
        return float(paddle_center_x) - contact_offset

    def _trajectory_crossing_at_x(
        self,
        trajectory_points: list[tuple[tuple[float, float], float]],
        target_x: float,
    ) -> tuple[tuple[float, float], float] | None:
        for start, end in zip(trajectory_points, trajectory_points[1:], strict=False):
            crossing_interval = self._segment_time_interval_for_value_range(
                float(start[1]),
                float(end[1]),
                float(start[0][0]),
                float(end[0][0]),
                target_x,
                target_x,
            )
            if crossing_interval is None:
                continue
            crossing_time = crossing_interval[0]
            candidate = self._interpolate_trajectory_point(start, end, crossing_time)
            return candidate, crossing_time
        return None

    def _interpolate_trajectory_point(
        self,
        start: tuple[tuple[float, float], float],
        end: tuple[tuple[float, float], float],
        time_step: float,
    ) -> tuple[float, float]:
        start_time = float(start[1])
        end_time = float(end[1])
        alpha = (
            0.0 if end_time == start_time else (time_step - start_time) / (end_time - start_time)
        )
        alpha = float(np.clip(alpha, 0.0, 1.0))
        return (
            float(start[0][0]) + (float(end[0][0]) - float(start[0][0])) * alpha,
            float(start[0][1]) + (float(end[0][1]) - float(start[0][1])) * alpha,
        )

    def _can_reach_interception(
        self, candidate: dict[str, Any], paddle_pos: tuple[float, float], y_only: bool
    ) -> bool:
        target_x, target_y = candidate["paddle_target_position"]
        if y_only:
            required_time = abs(float(target_y) - float(paddle_pos[1])) / game_config.PADDLE_SPEED
        else:
            required_time = (
                max(
                    abs(float(target_x) - float(paddle_pos[0])),
                    abs(float(target_y) - float(paddle_pos[1])),
                )
                / game_config.PADDLE_SPEED
            )
        return required_time <= float(candidate["time"]) + 1e-9

    def _simulate_ball_trajectory(
        self,
        ball_pos: tuple,
        ball_vel: tuple,
        field_bounds: tuple,
        max_time: float | None = 10.0,
        dt: float = 0.1,
    ) -> list:
        """
        Simulate ball trajectory with wall bounces

        Args:
            ball_pos: Starting ball position (x, y)
            ball_vel: Ball velocity (vx, vy)
            field_bounds: Field boundaries (min_x, max_x, min_y, max_y)
            max_time: Maximum simulation time. If None, simulate until the ball reaches a side.
            dt: Deprecated compatibility argument; the returned trajectory uses exact segment points.

        Returns:
            List of (position, time) tuples along the trajectory
        """
        del dt
        min_x, max_x, min_y, max_y = field_bounds
        ball_radius = float(game_config.BALL_RADIUS)

        pos_x, pos_y = float(ball_pos[0]), float(ball_pos[1])
        vel_x, vel_y = float(ball_vel[0]), float(ball_vel[1])
        current_time = 0.0
        trajectory = [((pos_x, pos_y), current_time)]

        if abs(vel_x) < 1e-9 and max_time is None:
            return trajectory

        while max_time is None or current_time < max_time:
            remaining_time = float("inf") if max_time is None else max_time - current_time
            side_time = self._time_to_side_boundary(pos_x, vel_x, min_x, max_x, ball_radius)
            wall_time = self._time_to_horizontal_wall(pos_y, vel_y, min_y, max_y, ball_radius)

            next_time = remaining_time
            event = "limit"
            if side_time is not None and side_time <= next_time:
                next_time = side_time
                event = "side"
            if wall_time is not None and wall_time <= next_time:
                next_time = wall_time
                event = "wall"

            if next_time == float("inf"):
                break

            pos_x += vel_x * next_time
            pos_y += vel_y * next_time
            current_time += next_time

            if event == "side":
                pos_x = min_x + ball_radius if vel_x < 0 else max_x - ball_radius
                trajectory.append(((pos_x, pos_y), current_time))
                break

            if event == "wall":
                pos_y = min_y + ball_radius if vel_y < 0 else max_y - ball_radius
                vel_y = -vel_y
                trajectory.append(((pos_x, pos_y), current_time))
                continue

            trajectory.append(((pos_x, pos_y), current_time))
            break

        return trajectory

    def _time_to_side_boundary(
        self, pos_x: float, vel_x: float, min_x: float, max_x: float, ball_radius: float
    ) -> float | None:
        if vel_x < -1e-12:
            return max(((min_x + ball_radius) - pos_x) / vel_x, 0.0)
        if vel_x > 1e-12:
            return max(((max_x - ball_radius) - pos_x) / vel_x, 0.0)
        return None

    def _time_to_horizontal_wall(
        self, pos_y: float, vel_y: float, min_y: float, max_y: float, ball_radius: float
    ) -> float | None:
        if vel_y < -1e-12:
            return max(((min_y + ball_radius) - pos_y) / vel_y, 0.0)
        if vel_y > 1e-12:
            return max(((max_y - ball_radius) - pos_y) / vel_y, 0.0)
        return None

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
        self, action1: Action | None, action2: Action | None, dt: float | None = None
    ) -> tuple[dict[str, Any], dict[str, Any], float, float, bool, dict[str, Any]]:
        """
        Performs a step in the environment

        Args:
            action1: Player 1 action
            action2: Player 2 action
            dt: Optional explicit delta time in seconds. If omitted, use the configured fixed step.

        Returns:
            Tuple: (obs1, obs2, reward1, reward2, done, info)
        """
        # Handle None actions with default actions (no movement)
        if action1 is None:
            action1 = Action(move_x=0.0, move_y=0.0)
        if action2 is None:
            action2 = Action(move_x=0.0, move_y=0.0)

        # Update physics
        if dt is None:
            dt = game_config.GAME_SPEED_MULTIPLIER / game_config.FPS
            if self.headless or ai_config.HEADLESS_MODE:
                dt *= ai_config.FAST_MODE_MULTIPLIER

        events = self.physics_engine.update(dt, action1, action2)
        game_state = self.physics_engine.get_game_state()

        # Calculate rewards
        reward1 = self.reward_calculators[1].calculate_reward(game_state, events, 1)
        reward2 = self.reward_calculators[2].calculate_reward(game_state, events, 2)

        self.step_count += 1

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
