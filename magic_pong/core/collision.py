"""
Collision detection system for Magic Pong
"""

import math
from typing import cast

from magic_pong.core.entities import Ball
from magic_pong.core.entities import Bonus
from magic_pong.core.entities import Paddle
from magic_pong.core.entities import RotatingPaddle
from magic_pong.core.entities import Vector2D
from magic_pong.utils.config import game_config


def point_in_rect(point: Vector2D, rect: tuple[float, float, float, float]) -> bool:
    """Checks if a point is inside a rectangle"""
    x, y, width, height = rect
    return x <= point.x <= x + width and y <= point.y <= y + height


def circle_rect_collision(ball: Ball, rect: tuple[float, float, float, float]) -> bool:
    """Detects collision between a circle (ball) and a rectangle"""
    x, y, width, height = rect

    # Closest point on the rectangle to the ball center
    closest_x = max(x, min(ball.position.x, x + width))
    closest_y = max(y, min(ball.position.y, y + height))

    # Distance between ball center and closest point
    distance = math.sqrt((ball.position.x - closest_x) ** 2 + (ball.position.y - closest_y) ** 2)

    return distance <= ball.radius


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

    if prev_in_collision:
        closest = Vector2D(
            max(prev_rect[0], min(prev_pos.x, prev_rect[0] + prev_rect[2])),
            max(prev_rect[1], min(prev_pos.y, prev_rect[1] + prev_rect[3])),
        )
        offset = prev_pos - closest
        # Exact contact is not an old penetration. An incoming ball must bounce
        # in this frame, including when it crosses the whole paddle this step.
        if abs(offset.magnitude() - ball.radius) <= 1e-9:
            relative_step = (current_pos - prev_pos) - (paddle.position - paddle.prev_position)
            if relative_step.x * offset.x + relative_step.y * offset.y < 0:
                return True, 0.0

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
            start_pos.x + (end_pos.x - start_pos.x) * t,
            start_pos.y + (end_pos.y - start_pos.y) * t,
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
                start_pos, end_pos, radius, prev_rect, current_rect, t_prev, t
            )

    return False, 0.0


def find_entry_collision_time_range(
    start_pos: Vector2D,
    end_pos: Vector2D,
    radius: float,
    prev_rect: tuple[float, float, float, float],
    current_rect: tuple[float, float, float, float],
    t_start: float,
    t_end: float,
) -> float:
    """Find precise entry collision time in a specific range with interpolated rectangle"""
    epsilon = 0.001

    while t_end - t_start > epsilon:
        t_mid = (t_start + t_end) / 2
        mid_pos = Vector2D(
            start_pos.x + (end_pos.x - start_pos.x) * t_mid,
            start_pos.y + (end_pos.y - start_pos.y) * t_mid,
        )
        # Interpolate rectangle position at t_mid
        mid_rect = (
            prev_rect[0] + (current_rect[0] - prev_rect[0]) * t_mid,
            prev_rect[1] + (current_rect[1] - prev_rect[1]) * t_mid,
            prev_rect[2] + (current_rect[2] - prev_rect[2]) * t_mid,
            prev_rect[3] + (current_rect[3] - prev_rect[3]) * t_mid,
        )

        if circle_rect_collision_at_position(mid_pos, radius, mid_rect):
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

    return distance <= radius


def circle_line_collision(ball: Ball, line_start: Vector2D, line_end: Vector2D) -> bool:
    """Detects collision between a circle and a line segment"""
    # Line vector
    line_vec = line_end - line_start
    line_length = line_vec.magnitude()

    if line_length < 1e-6:  # Use epsilon for float comparison
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

    return distance <= ball.radius


def _paddle_surface_normal(position: Vector2D, rect: tuple[float, float, float, float]) -> Vector2D:
    """Outward geometric normal, without the aiming/spin adjustment."""
    x, y, width, height = rect
    closest = Vector2D(max(x, min(position.x, x + width)), max(y, min(position.y, y + height)))
    offset = position - closest
    if offset.magnitude() > 1e-12:
        return offset.normalize()
    # An embedded centre has no closest-point direction. Choose its nearest exit.
    exits = (
        (position.x - x, Vector2D(-1, 0)),
        (x + width - position.x, Vector2D(1, 0)),
        (position.y - y, Vector2D(0, -1)),
        (y + height - position.y, Vector2D(0, 1)),
    )
    return min(exits, key=lambda entry: entry[0])[1]


def _paddle_spin_normal(
    position: Vector2D, rect: tuple[float, float, float, float], surface: Vector2D
) -> Vector2D:
    relative_y = (position.y - rect[1] - rect[3] / 2) / (rect[3] / 2)
    return Vector2D(surface.x, surface.y + 0.3 * max(-1.0, min(1.0, relative_y))).normalize()


def get_paddle_collision_normal(ball: Ball, paddle: Paddle) -> Vector2D | None:
    """Return the contact-position-dependent normal used to aim a genuine bounce."""
    rect = paddle.get_rect()
    return _paddle_spin_normal(ball.position, rect, _paddle_surface_normal(ball.position, rect))


def apply_paddle_bounce(
    ball: Ball,
    paddle: Paddle,
    normal: Vector2D | None = None,
    *,
    dt: float | None = None,
    surface_normal: Vector2D | None = None,
) -> None:
    """Reflect a genuine impact and accelerate the ball by ``BALL_SPEED_INCREASE``.

    Paddle motion transfers no speed: the outgoing speed depends only on the
    incoming speed, the configured increase factor and ``MAX_BALL_SPEED``.
    ``dt`` remains accepted for existing callers; reflection is independent of it.
    Paddle catch-up is resolved separately by position-only correction.
    """
    if normal is None:
        normal = get_paddle_collision_normal(ball, paddle)
    if normal is None or normal.magnitude() < 1e-12:
        return
    normal = normal.normalize()
    speed = min(
        ball.velocity.magnitude() * game_config.BALL_SPEED_INCREASE, game_config.MAX_BALL_SPEED
    )
    dot = ball.velocity.x * normal.x + ball.velocity.y * normal.y
    reflected = ball.velocity - normal * (2 * dot)
    surface = surface_normal or _paddle_surface_normal(ball.position, paddle.get_rect())
    # A spin-adjusted normal can send a grazing impact back into the actual
    # surface. Fall back to geometric reflection instead of repeated bounces.
    if reflected.x * surface.x + reflected.y * surface.y < 0:
        dot = ball.velocity.x * surface.x + ball.velocity.y * surface.y
        reflected = ball.velocity - surface * (2 * dot)
    ball.velocity = reflected.normalize() * speed


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
        """Return True only for a bounce; a paddle pushing a ball only moves it."""
        collision, time = continuous_circle_paddle_collision(ball, paddle, dt)
        if not collision:
            # Initial/persistent overlap (including a size change) still needs
            # correction, but is not a fresh strike or a source of hit rewards.
            self.separate_ball_from_paddle(ball, paddle)
            return False

        previous = paddle.get_previous_rect()
        current = paddle.get_rect()
        contact_rect = cast(
            tuple[float, float, float, float],
            tuple(a + (b - a) * time for a, b in zip(previous, current)),
        )
        contact_position = ball.prev_position + (ball.position - ball.prev_position) * time
        surface = _paddle_surface_normal(contact_position, contact_rect)
        ball_approach = ball.velocity.x * surface.x + ball.velocity.y * surface.y
        paddle_velocity = (
            (paddle.position - paddle.prev_position) / dt if dt > 0 else Vector2D(0, 0)
        )
        relative_approach = ball_approach - (
            paddle_velocity.x * surface.x + paddle_velocity.y * surface.y
        )
        # An outgoing or tangential ball caught by a paddle must retain its
        # ENTIRE velocity vector. A faster head-on paddle still causes a bounce.
        # Swept entry and incoming relative motion identify a new contact. A
        # previous hit on this same paddle must not suppress a later real hit.
        bounce = ball_approach < -1e-9 and relative_approach < -1e-9
        if bounce:
            normal = _paddle_spin_normal(contact_position, contact_rect, surface)
            apply_paddle_bounce(ball, paddle, normal, dt=dt, surface_normal=surface)
            ball.last_paddle_hit = paddle.player_id

        # Keep the entry side even if the final positions have crossed through
        # each other. Separation never modifies velocity or bounce bookkeeping.
        self.separate_ball_from_paddle(ball, paddle, force=True, normal=surface)
        return bounce

    def separate_ball_from_paddle(
        self,
        ball: Ball,
        paddle: Paddle,
        separation_multiplier: float = 1.0,
        force: bool = False,
        *,
        normal: Vector2D | None = None,
    ) -> None:
        """Resolve penetration along the contact normal with numerical clearance.

        A supplied normal preserves the entry side of a swept collision. Without
        one, use the nearest exit and leave already separated balls untouched.
        """
        rect = paddle.get_rect()
        if not force and not circle_rect_collision(ball, rect):
            return
        if normal is None:
            normal = _paddle_surface_normal(ball.position, rect)
        normal = normal.normalize()
        x, y, width, height = rect
        support = Vector2D(x + (width if normal.x > 0 else 0), y + (height if normal.y > 0 else 0))
        distance = (ball.position.x - support.x) * normal.x + (
            ball.position.y - support.y
        ) * normal.y
        clearance = 1e-7 * max(1.0, separation_multiplier)
        correction = ball.radius + clearance - distance
        if correction > 0:
            ball.position = ball.position + normal * correction

    def check_ball_rotating_paddle(self, ball: Ball, rotating_paddle: RotatingPaddle) -> bool:
        """Checks collision with a rotating paddle"""
        segments = rotating_paddle.get_line_segments()

        for start, end in segments:
            if circle_line_collision(ball, start, end):
                # Calculate collision normal pointing from the segment toward the ball
                line_vec = end - start
                raw_normal = Vector2D(-line_vec.y, line_vec.x)
                to_ball = ball.position - start
                if raw_normal.x * to_ball.x + raw_normal.y * to_ball.y < 0:
                    raw_normal = -raw_normal
                normal = raw_normal.normalize()

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
