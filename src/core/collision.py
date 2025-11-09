"""
Collision detection system for Magic Pong
"""

import math

import numpy as np
from magic_pong.core.entities import Ball, Bonus, Paddle, RotatingPaddle, Vector2D
from magic_pong.utils.config import game_config


def point_in_rect(point: Vector2D, rect: tuple[float, float, float, float]) -> bool:
    """Checks if a point is inside a rectangle"""
    x, y, width, height = rect
    return x <= point.x <= x + width and y <= point.y <= y + height  # type: ignore[no-any-return]


def circle_rect_collision(ball: Ball, rect: tuple[float, float, float, float]) -> bool:
    """Detects collision between a circle (ball) and a rectangle"""
    x, y, width, height = rect

    # Closest point on the rectangle to the ball center
    closest_x = max(x, min(ball.position.x, x + width))
    closest_y = max(y, min(ball.position.y, y + height))

    # Distance between ball center and closest point
    distance = math.sqrt((ball.position.x - closest_x) ** 2 + (ball.position.y - closest_y) ** 2)

    return distance <= ball.radius  # type: ignore[no-any-return]


def continuous_circle_paddle_collision(ball: Ball, paddle: Paddle, dt: float) -> tuple[bool, float]:
    """
    Detects collision between a moving circle and a paddle using continuous collision detection.
    Uses the ball's previous position to determine if it's entering or exiting the paddle.
    Returns (collision_occurred, collision_time) where collision_time is between 0 and 1.
    """
    # Use the ball's and paddle's stored previous positions
    prev_pos = ball.prev_position
    current_pos = ball.position
    prev_rect = paddle.get_previous_rect()
    current_rect = paddle.get_rect()

    # Check collision state at start and end
    prev_in_collision = circle_rect_collision_at_position(prev_pos, ball.radius, prev_rect)
    curr_in_collision = circle_rect_collision_at_position(current_pos, ball.radius, current_rect)

    # Case 1: Ball was outside and is now inside -> entering collision
    if not prev_in_collision and curr_in_collision:
        return True, find_entry_collision_time(
            prev_pos, current_pos, ball.radius, prev_rect, current_rect
        )

    # Case 2: Ball was inside and is now outside -> exiting (no new collision)
    if prev_in_collision and not curr_in_collision:
        return False, 0.0

    # Case 3: Ball was outside and is still outside -> check if it passed through (tunneling)
    if not prev_in_collision and not curr_in_collision:
        tunneled, time = check_trajectory_intersection(
            prev_pos, current_pos, ball.radius, prev_rect, current_rect
        )
        if tunneled:
            return True, time

    # Case 4: Ball was inside and is still inside -> already in collision (no new collision)
    return False, 0.0


def find_entry_collision_time(
    start_pos: Vector2D,
    end_pos: Vector2D,
    radius: float,
    previous_rect: tuple[float, float, float, float],
    current_rect: tuple[float, float, float, float],
) -> float:
    """Find the exact time when the ball enters the rectangle using binary search"""
    epsilon = 0.001  # Precision threshold
    t_start = 0.0
    t_end = 1.0

    # Binary search to find the entry point
    while t_end - t_start > epsilon:
        t_mid = (t_start + t_end) / 2
        mid_pos = Vector2D(
            start_pos.x + (end_pos.x - start_pos.x) * t_mid,
            start_pos.y + (end_pos.y - start_pos.y) * t_mid,
        )
        mid_rect = (
            previous_rect[0] + (current_rect[0] - previous_rect[0]) * t_mid,
            previous_rect[1] + (current_rect[1] - previous_rect[1]) * t_mid,
            previous_rect[2] + (current_rect[2] - previous_rect[2]) * t_mid,
            previous_rect[3] + (current_rect[3] - previous_rect[3]) * t_mid,
        )

        if circle_rect_collision_at_position(mid_pos, radius, mid_rect):
            # Ball is in collision at t_mid, so entry is before this point
            t_end = t_mid
        else:
            # Ball is not in collision at t_mid, so entry is after this point
            t_start = t_mid

    return t_end  # Return the first time where collision occurs


def check_trajectory_intersection(
    start_pos: Vector2D,
    end_pos: Vector2D,
    radius: float,
    prev_rect: tuple[float, float, float, float],
    current_rect: tuple[float, float, float, float],
) -> tuple[bool, float]:
    """
    Check if the ball trajectory intersects with the rectangle (high-speed tunneling case).
    This handles cases where the ball is fast enough to completely pass through in one frame.
    """

    # Sample the trajectory at regular intervals
    num_samples = 30  # More samples for high-speed detection

    for i in range(1, num_samples):
        t = i / num_samples
        sample_pos = Vector2D(
            start_pos.x + (end_pos.x - start_pos.x) * t, start_pos.y + (end_pos.y - start_pos.y) * t
        )
        sample_rect = (
            prev_rect[0] + (current_rect[0] - prev_rect[0]) * t,
            prev_rect[1] + (current_rect[1] - prev_rect[1]) * t,
            prev_rect[2] + (current_rect[2] - prev_rect[2]) * t,
            prev_rect[3] + (current_rect[3] - prev_rect[3]) * t,
        )

        if circle_rect_collision_at_position(sample_pos, radius, sample_rect):
            # Found intersection, find precise entry time
            t_prev = (i - 1) / num_samples
            return True, find_entry_collision_time_range(
                start_pos, end_pos, radius, sample_rect, t_prev, t
            )

    return False, 0.0


def find_entry_collision_time_range(
    start_pos: Vector2D,
    end_pos: Vector2D,
    radius: float,
    rect: tuple[float, float, float, float],
    t_start: float,
    t_end: float,
) -> float:
    """Find precise entry collision time in a specific range"""
    epsilon = 0.001

    while t_end - t_start > epsilon:
        t_mid = (t_start + t_end) / 2
        mid_pos = Vector2D(
            start_pos.x + (end_pos.x - start_pos.x) * t_mid,
            start_pos.y + (end_pos.y - start_pos.y) * t_mid,
        )

        if circle_rect_collision_at_position(mid_pos, radius, rect):
            t_end = t_mid
        else:
            t_start = t_mid

    return t_end


def circle_rect_collision_at_position(
    position: Vector2D, radius: float, rect: tuple[float, float, float, float]
) -> bool:
    """Helper function to check collision at a specific position"""
    x, y, width, height = rect

    # Closest point on the rectangle to the ball center
    closest_x = max(x, min(position.x, x + width))
    closest_y = max(y, min(position.y, y + height))

    # Distance between ball center and closest point
    distance = math.sqrt((position.x - closest_x) ** 2 + (position.y - closest_y) ** 2)

    return distance <= radius  # type: ignore[no-any-return]


def circle_line_collision(ball: Ball, line_start: Vector2D, line_end: Vector2D) -> bool:
    """Detects collision between a circle and a line segment"""
    # Line vector
    line_vec = line_end - line_start
    line_length = line_vec.magnitude()

    if line_length == 0:
        # Zero-length line, check distance to point
        distance = (ball.position - line_start).magnitude()
        return distance <= ball.radius

    # Normalize line vector
    line_unit = line_vec.normalize()

    # Vector from line start to ball center
    to_ball = ball.position - line_start

    # Projection of to_ball vector onto the line
    projection_length = to_ball.x * line_unit.x + to_ball.y * line_unit.y

    # Clamp projection to line length
    projection_length = max(0, min(line_length, projection_length))

    # Closest point on the line
    closest_point = line_start + line_unit * projection_length

    # Distance from ball center to closest point
    distance = (ball.position - closest_point).magnitude()

    return distance <= ball.radius  # type: ignore[no-any-return]


def get_paddle_collision_normal(ball: Ball, paddle: Paddle) -> Vector2D | None:
    """
    Calculates the collision normal for a circular ball hitting a paddle.
    The normal is based on the closest point on the paddle to the ball center,
    with added vertical spin effect based on where the ball hits the paddle.
    """
    paddle_rect = paddle.get_rect()
    paddle_x, paddle_y, paddle_width, paddle_height = paddle_rect

    ball_center_x = ball.position.x
    ball_center_y = ball.position.y

    # Find the closest point on the paddle rectangle to the ball center
    closest_x = max(paddle_x, min(ball_center_x, paddle_x + paddle_width))
    closest_y = max(paddle_y, min(ball_center_y, paddle_y + paddle_height))

    # Calculate normal from closest point to ball center
    normal = Vector2D(ball_center_x - closest_x, ball_center_y - closest_y)

    # Normalize the direction
    normal_magnitude = normal.magnitude()
    if normal_magnitude < 0.0001:  # Ball center is inside paddle (shouldn't happen)
        # Default to horizontal bounce
        return Vector2D(-1.0 if ball_center_x < paddle_x + paddle_width / 2 else 1.0, 0.0)

    normal = normal / normal_magnitude

    # Add spin effect based on where the ball hits vertically on the paddle
    paddle_center_y = paddle_y + paddle_height / 2
    relative_hit_y = (ball_center_y - paddle_center_y) / (paddle_height / 2)
    relative_hit_y = max(-1.0, min(1.0, relative_hit_y))

    # Add vertical spin component
    spin_factor = 0.3
    normal.y += relative_hit_y * spin_factor

    # Renormalize after adding spin
    return normal.normalize()


def apply_paddle_bounce(ball: Ball, paddle: Paddle, normal: Vector2D | None = None) -> None:
    """Applies the bounce effect on a paddle with spin"""
    if normal is None:
        normal = get_paddle_collision_normal(ball, paddle)
    if not normal:
        return

    # Reflect velocity off the paddle using the normal
    # Formula: v' = v - 2(v·n)n
    velocity_tuple = ball.velocity.to_tuple()
    normal_tuple = normal.to_tuple()
    dot_product = velocity_tuple[0] * normal_tuple[0] + velocity_tuple[1] * normal_tuple[1]

    ball.velocity.x = ball.velocity.x - 2 * dot_product * normal.x
    ball.velocity.y = ball.velocity.y - 2 * dot_product * normal.y

    # Optional: limit max speed to prevent runaway velocity
    speed = ball.velocity.magnitude()
    max_speed = game_config.MAX_BALL_SPEED
    if speed > max_speed:
        ball.velocity = ball.velocity.normalize() * max_speed


class CollisionDetector:
    """Main collision manager"""

    def __init__(self) -> None:
        pass

    def check_ball_walls(self, ball: Ball, field_width: float, field_height: float) -> str:
        """Checks collisions with walls. Returns the collision type."""
        # Top and bottom walls
        if ball.position.y - ball.radius <= 0:
            ball.position.y = ball.radius
            return "top"
        elif ball.position.y + ball.radius >= field_height:
            ball.position.y = field_height - ball.radius
            return "bottom"

        # Left and right walls (goals)
        if ball.position.x - ball.radius <= 0:
            return "left_goal"
        elif ball.position.x + ball.radius >= field_width:
            return "right_goal"

        return "none"

    def check_ball_paddle(self, ball: Ball, paddle: Paddle, dt: float) -> bool:
        """Checks and handles ball-paddle collision with continuous detection"""
        # Prevent multiple bounces on same paddle
        if ball.last_paddle_hit == paddle.player_id:
            return False

        # Use continuous collision detection
        collision_occurred, collision_time = continuous_circle_paddle_collision(ball, paddle, dt)

        if collision_occurred:
            # Get collision normal
            normal = get_paddle_collision_normal(ball, paddle)
            if not normal:
                return False

            # Check if ball is approaching the paddle (not moving away from it)
            ball_approach = np.dot(ball.velocity.to_tuple(), normal.to_tuple())

            if ball_approach < 0:  # Ball is moving toward the paddle surface
                # Apply bounce
                apply_paddle_bounce(ball, paddle, normal)
                ball.last_paddle_hit = paddle.player_id

                # Move ball out of collision after bounce to prevent overlap
                self.separate_ball_from_paddle(ball, paddle)

                return True

        return False

    def separate_ball_from_paddle(self, ball: Ball, paddle: Paddle) -> None:
        """Ensures the ball is positioned outside the paddle after collision"""
        rect = paddle.get_rect()
        x, y, width, height = rect

        # Calculate which side of the rectangle the ball is closest to
        ball_center_x = ball.position.x
        ball_center_y = ball.position.y

        # Calculate distances to each edge
        dist_to_left = ball_center_x - x
        dist_to_right = (x + width) - ball_center_x
        dist_to_top = ball_center_y - y
        dist_to_bottom = (y + height) - ball_center_y

        # Find the closest edge
        min_dist = min(dist_to_left, dist_to_right, dist_to_top, dist_to_bottom)

        # Push ball away from the closest edge
        safety_margin = 1.0
        required_distance = ball.radius + safety_margin

        if min_dist == dist_to_left:
            # Ball is closest to left edge, push it left
            ball.position.x = x - required_distance
        elif min_dist == dist_to_right:
            # Ball is closest to right edge, push it right
            ball.position.x = x + width + required_distance
        elif min_dist == dist_to_top:
            # Ball is closest to top edge, push it up
            ball.position.y = y - required_distance
        else:
            # Ball is closest to bottom edge, push it down
            ball.position.y = y + height + required_distance

    def check_ball_rotating_paddle(self, ball: Ball, rotating_paddle: RotatingPaddle) -> bool:
        """Checks collision with a rotating paddle"""
        segments = rotating_paddle.get_line_segments()

        for start, end in segments:
            if circle_line_collision(ball, start, end):
                # Calculate collision normal
                line_vec = end - start
                normal = Vector2D(-line_vec.y, line_vec.x).normalize()

                # Reflect velocity
                dot_product = ball.velocity.x * normal.x + ball.velocity.y * normal.y
                ball.velocity.x -= 2 * dot_product * normal.x
                ball.velocity.y -= 2 * dot_product * normal.y

                return True

        return False

    def check_player_bonus(self, paddle: Paddle, bonuses: list[Bonus]) -> list[Bonus]:
        """Checks player-bonus collisions"""
        collected = []
        paddle_rect = paddle.get_rect()

        for bonus in bonuses:
            if not bonus.collected:
                bonus_rect = bonus.get_rect()

                # Simple rectangle overlap check
                if (
                    paddle_rect[0] < bonus_rect[0] + bonus_rect[2]
                    and paddle_rect[0] + paddle_rect[2] > bonus_rect[0]
                    and paddle_rect[1] < bonus_rect[1] + bonus_rect[3]
                    and paddle_rect[1] + paddle_rect[3] > bonus_rect[1]
                ):
                    bonus.collect()
                    collected.append(bonus)

        return collected
