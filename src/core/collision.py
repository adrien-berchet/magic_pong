"""
Collision detection system for Magic Pong
"""

import math

from magic_pong.core.entities import Ball, Bonus, Paddle, RotatingPaddle, Vector2D


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

    return distance <= ball.radius


def get_paddle_collision_normal(ball: Ball, paddle: Paddle) -> Vector2D | None:
    """Calculates the collision normal with a paddle"""
    rect = paddle.get_rect()
    x, y, width, height = rect

    # Determine which side of the paddle the ball hits
    ball_center_x = ball.position.x
    ball_center_y = ball.position.y

    # Rectangle center
    rect_center_x = x + width / 2
    rect_center_y = y + height / 2

    # Differences
    dx = ball_center_x - rect_center_x
    dy = ball_center_y - rect_center_y

    # Ratios to determine the side
    width_ratio = dx / (width / 2) if width > 0 else 0
    height_ratio = dy / (height / 2) if height > 0 else 0

    # The side with the largest absolute ratio is the one hit
    if abs(width_ratio) > abs(height_ratio):
        # Horizontal collision (left or right)
        return Vector2D(1.0 if dx > 0 else -1.0, 0.0)
    else:
        # Vertical collision (top or bottom)
        return Vector2D(0.0, 1.0 if dy > 0 else -1.0)


def apply_paddle_bounce(ball: Ball, paddle: Paddle) -> None:
    """Applies the bounce effect on a paddle with spin"""
    normal = get_paddle_collision_normal(ball, paddle)
    if not normal:
        return

    # Relative position of the ball on the paddle (for spin effect)
    paddle_center_y = paddle.position.y + paddle.height / 2
    relative_hit_pos = (ball.position.y - paddle_center_y) / (paddle.height / 2)
    relative_hit_pos = max(-1.0, min(1.0, relative_hit_pos))  # Clamp between -1 and 1

    # Current speed
    speed = ball.velocity.magnitude()

    if abs(normal.x) > abs(normal.y):
        # Horizontal bounce (paddle)
        ball.velocity.x = -ball.velocity.x
        # Add vertical spin based on hit position
        ball.velocity.y += relative_hit_pos * speed * 0.3
    else:
        # Vertical bounce (wall)
        ball.velocity.y = -ball.velocity.y

    # Normalize and restore proper speed
    ball.velocity = ball.velocity.normalize() * speed

    # Slight acceleration
    ball.velocity = ball.velocity * 1.02


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

    def check_ball_paddle(self, ball: Ball, paddle: Paddle) -> bool:
        """Checks and handles ball-paddle collision"""
        rect = paddle.get_rect()
        if ball.last_paddle_hit == paddle.player_id:
            # Avoid multiple bounces on the same paddle
            # Check if ball is moving away from paddle

            if paddle.player_id == 1:  # Left paddle
                if ball.velocity.x > 0:  # Ball moving away to the right
                    ball.last_paddle_hit = None
            else:  # Right paddle
                if ball.velocity.x < 0:  # Ball moving away to the left
                    ball.last_paddle_hit = None

            return False

        if circle_rect_collision(ball, rect):
            apply_paddle_bounce(ball, paddle)
            ball.last_paddle_hit = paddle.player_id
            return True

        return False

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
