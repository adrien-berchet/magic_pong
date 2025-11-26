"""
Physics system for Magic Pong
"""

import math
import random
from typing import Any

from magic_pong.core.collision import CollisionDetector
from magic_pong.core.entities import Action, Ball, Bonus, BonusType, Paddle, RotatingPaddle
from magic_pong.utils.config import game_config


class BonusSpawner:
    """Bonus spawning manager"""

    def __init__(self, field_width: float, field_height: float):
        self.field_width = field_width
        self.field_height = field_height
        self.spawn_timer = 0.0
        self.spawn_interval = game_config.BONUS_SPAWN_INTERVAL

    def update(self, dt: float, existing_bonuses: list[Bonus]) -> list[Bonus]:
        """Updates the spawner and returns new bonuses"""
        # Don't spawn bonuses if they are disabled
        if not game_config.BONUSES_ENABLED:
            return []

        self.spawn_timer += dt
        new_bonuses = []

        if self.spawn_timer >= self.spawn_interval:
            self.spawn_timer = 0.0

            # Don't spawn if there are already too many bonuses
            if len(existing_bonuses) < 4:
                new_bonuses = self._spawn_symmetric_bonuses()

        return new_bonuses

    def _spawn_symmetric_bonuses(self) -> list[Bonus]:
        """Spawns bonuses symmetrically"""
        bonuses = []

        # Choose a random bonus type
        bonus_type = random.choice(list(BonusType))

        # Calculate safe spawn margins based on configuration
        margin = game_config.PADDLE_MARGIN + game_config.PADDLE_WIDTH + game_config.BONUS_SIZE
        vertical_margin = game_config.BONUS_SIZE

        # Random position in the left half
        left_x = random.uniform(margin, self.field_width / 2 - margin)
        y = random.uniform(vertical_margin, self.field_height - vertical_margin)

        # Symmetric position in the right half
        right_x = self.field_width - left_x

        # Validate that positions are within field bounds
        if (
            margin <= left_x <= self.field_width - margin
            and vertical_margin <= y <= self.field_height - vertical_margin
        ):
            bonuses.append(Bonus(left_x, y, bonus_type))
            bonuses.append(Bonus(right_x, y, bonus_type))

        return bonuses


class PhysicsEngine:
    """Main physics engine"""

    def __init__(self, field_width: float, field_height: float):
        self.field_width = field_width
        self.field_height = field_height
        self.collision_detector = CollisionDetector()
        self.bonus_spawner = BonusSpawner(field_width, field_height)

        # Game state
        self.reset_paddles()

        self.bonuses: list[Bonus] = []
        self.rotating_paddles: list[RotatingPaddle] = []
        self.score: list[int] = [0, 0]
        self.game_time = 0.0

        # Initialize ball with random direction
        self.reset_ball()

    def reset_paddles(self) -> None:
        """Resets paddles to their initial position and size"""
        self.player1 = Paddle(
            game_config.PADDLE_MARGIN, self.field_height / 2 - game_config.PADDLE_HEIGHT / 2, 1
        )

        self.player2 = Paddle(
            self.field_width - game_config.PADDLE_MARGIN - game_config.PADDLE_WIDTH,
            self.field_height / 2 - game_config.PADDLE_HEIGHT / 2,
            2,
        )

    def reset_ball(self, direction: int = 0, angle: float | None = None) -> None:
        """Resets the ball to center with optional specific angle"""
        if direction == 0:
            direction = random.choice([-1, 1])
        self.ball = Ball(self.field_width / 2, self.field_height / 2, game_config.BALL_SPEED, 0)
        # self.ball.reset_to_center(direction, angle)

    def set_ball_initial_direction(self, direction: int = 1, angle_degrees: float = 0.0) -> None:
        """Sets a specific initial direction for the ball (useful for training)"""
        angle_radians = math.radians(angle_degrees)
        self.reset_ball(direction, angle_radians)

    def update(self, dt: float, player1_action: Action, player2_action: Action) -> dict:
        """Updates game physics"""
        # Apply speed multiplier
        effective_dt = dt  # / game_config.GAME_SPEED_MULTIPLIER
        self.game_time += effective_dt

        # Move players
        if player1_action:
            self.player1.move(player1_action.move_x, player1_action.move_y, effective_dt)
        if player2_action:
            self.player2.move(player2_action.move_x, player2_action.move_y, effective_dt)

        # Update entities
        self.ball.update(effective_dt)
        self.player1.update(effective_dt)
        self.player2.update(effective_dt)

        # Update rotating paddles
        self.rotating_paddles = [rp for rp in self.rotating_paddles if rp.update(effective_dt)]

        # Update bonuses
        self.bonuses = [bonus for bonus in self.bonuses if bonus.update(effective_dt)]

        # Spawn new bonuses
        new_bonuses = self.bonus_spawner.update(effective_dt, self.bonuses)
        self.bonuses.extend(new_bonuses)

        # Check collisions
        events = self._check_collisions(effective_dt)

        return events

    def _check_collisions(self, effective_dt: float) -> dict:
        """Checks all collisions and returns events"""
        events: dict[str, list] = {
            "wall_bounces": [],
            "paddle_hits": [],
            "goals": [],
            "bonus_collected": [],
            "rotating_paddle_hits": [],
        }

        # Wall collisions
        wall_collision = self.collision_detector.check_ball_walls(
            self.ball, self.field_width, self.field_height
        )

        if wall_collision == "top" or wall_collision == "bottom":
            self.ball.bounce_vertical()
            # Reset last paddle hit so ball can bounce off either paddle again
            self.ball.last_paddle_hit = None
            events["wall_bounces"].append(wall_collision)
        elif wall_collision == "left_goal":
            self.score[1] += 1  # Point for player 2
            events["goals"].append({"player": 2, "score": self.score.copy()})
            self.reset_paddles()
            self.reset_ball(1)  # Restart towards the right
        elif wall_collision == "right_goal":
            self.score[0] += 1  # Point for player 1
            events["goals"].append({"player": 1, "score": self.score.copy()})
            self.reset_paddles()
            self.reset_ball(-1)  # Restart towards the left

        # Paddle collisions with continuous detection
        if self.collision_detector.check_ball_paddle(self.ball, self.player1, effective_dt):
            events["paddle_hits"].append({"player": 1})
        if self.collision_detector.check_ball_paddle(self.ball, self.player2, effective_dt):
            events["paddle_hits"].append({"player": 2})

        # Rotating paddle collisions
        for rp in self.rotating_paddles:
            if self.collision_detector.check_ball_rotating_paddle(self.ball, rp):
                events["rotating_paddle_hits"].append({"player": rp.player_id})

        # Player-bonus collisions
        for player, paddle in [(1, self.player1), (2, self.player2)]:
            collected = self.collision_detector.check_player_bonus(paddle, self.bonuses)
            for bonus in collected:
                self._apply_bonus_effect(bonus.type, player)
                events["bonus_collected"].append({"player": player, "type": bonus.type.value})

        return events

    def _apply_bonus_effect(self, bonus_type: BonusType, player: int) -> None:
        """Applies a bonus effect"""
        # Don't apply bonus effects if bonuses are disabled
        if not game_config.BONUSES_ENABLED:
            return

        if bonus_type == BonusType.ENLARGE_PADDLE:
            # Enlarge player's paddle
            paddle = self.player1 if player == 1 else self.player2
            paddle.apply_size_effect(game_config.PADDLE_SIZE_MULTIPLIER, game_config.BONUS_DURATION)

        elif bonus_type == BonusType.SHRINK_OPPONENT:
            # Shrink opponent's paddle
            opponent_paddle = self.player2 if player == 1 else self.player1
            opponent_paddle.apply_size_effect(
                game_config.PADDLE_SIZE_REDUCER, game_config.BONUS_DURATION
            )

        elif bonus_type == BonusType.ROTATING_PADDLE:
            # Add a rotating paddle
            if player == 1:
                # Position in left half
                x = random.uniform(100, self.field_width / 2 - 100)
            else:
                # Position in right half
                x = random.uniform(self.field_width / 2 + 100, self.field_width - 100)

            y = random.uniform(100, self.field_height - 100)

            rotating_paddle = RotatingPaddle(x, y, player)
            self.rotating_paddles.append(rotating_paddle)

    def get_game_state(self) -> dict[str, Any]:
        """Returns the complete game state"""
        return {
            "ball_position": self.ball.position.to_tuple(),
            "ball_velocity": self.ball.velocity.to_tuple(),
            "player1_position": self.player1.position.to_tuple(),
            "player2_position": self.player2.position.to_tuple(),
            "player1_paddle_size": self.player1.height,
            "player2_paddle_size": self.player2.height,
            "player1_last_position": self.player1.prev_position.to_tuple(),
            "player2_last_position": self.player2.prev_position.to_tuple(),
            "active_bonuses": [
                (bonus.position.x, bonus.position.y, bonus.type.value)
                for bonus in self.bonuses
                if not bonus.collected
            ],
            "rotating_paddles": [
                (rp.center.x, rp.center.y, rp.angle) for rp in self.rotating_paddles
            ],
            "score": self.score.copy(),
            "time_elapsed": self.game_time,
            "field_bounds": (0, self.field_width, 0, self.field_height),
        }

    def is_game_over(self) -> bool:
        """Checks if the game is over"""
        return max(self.score) >= game_config.MAX_SCORE

    def get_winner(self) -> int:
        """Returns the winner (1 or 2), or 0 if no winner"""
        if self.score[0] >= game_config.MAX_SCORE:
            return 1
        elif self.score[1] >= game_config.MAX_SCORE:
            return 2
        return 0

    def reset_game(self) -> None:
        """Resets the game to zero"""
        self.score = [0, 0]
        self.game_time = 0.0
        self.bonuses.clear()
        self.rotating_paddles.clear()

        # Reset paddles to their initial position
        self.reset_paddles()

        # Reset ball to center
        self.reset_ball()
