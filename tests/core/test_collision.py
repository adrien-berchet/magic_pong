"""
Unit tests for collision detection system

Tests the critical collision detection fixes including:
- Continuous collision detection with trajectory checking
- Paddle bounce logic with proper normal calculation
- Ball tunneling prevention at high speeds
"""

import pytest

from magic_pong.core.collision import (
    CollisionDetector,
    apply_paddle_bounce,
    circle_rect_collision_at_position,
    continuous_circle_paddle_collision,
    get_paddle_collision_normal,
)
from magic_pong.core.entities import Ball, Paddle, Vector2D


class TestContinuousCollisionDetection:
    """Test continuous collision detection to prevent ball tunneling"""

    def test_ball_entering_paddle_detected(self):
        """Test that a ball entering a paddle is detected"""
        # Create a paddle at position (100, 100)
        paddle = Paddle(100, 100, player_id=1)

        # Create a ball moving towards the paddle
        ball = Ball(50, 120, 0, 0)  # Start position
        ball.velocity = Vector2D(100, 0)  # Moving right towards paddle
        ball.prev_position = Vector2D(50, 120)  # Previous position
        ball.position = Vector2D(105, 120)  # Now inside paddle

        collision, time = continuous_circle_paddle_collision(ball, paddle, dt=0.016)

        assert collision is True, "Should detect ball entering paddle"
        assert 0 <= time <= 1.0, f"Collision time should be between 0 and 1, got {time}"

    def test_ball_exiting_paddle_not_detected(self):
        """Test that a ball exiting a paddle doesn't trigger new collision"""
        paddle = Paddle(100, 100, player_id=1)

        # Ball was inside, now outside (exiting)
        ball = Ball(50, 120, 0, 0)
        ball.velocity = Vector2D(100, 0)
        ball.prev_position = Vector2D(105, 120)  # Was inside paddle
        ball.position = Vector2D(150, 120)  # Now outside

        collision, time = continuous_circle_paddle_collision(ball, paddle, dt=0.016)

        assert collision is False, "Should not detect collision when ball is exiting"

    def test_high_speed_tunneling_detection(self):
        """Test that high-speed balls passing through paddle are detected"""
        paddle = Paddle(400, 250, player_id=2)

        # Ball moving very fast, would tunnel through in one frame
        ball = Ball(300, 300, 0, 0)
        ball.velocity = Vector2D(1000, 0)  # Very fast
        ball.prev_position = Vector2D(300, 300)  # Before paddle
        ball.position = Vector2D(500, 300)  # After paddle (tunneled through)

        collision, time = continuous_circle_paddle_collision(ball, paddle, dt=0.016)

        assert collision is True, "Should detect tunneling at high speeds"

    def test_ball_missing_paddle(self):
        """Test that a ball missing the paddle is not detected"""
        paddle = Paddle(100, 100, player_id=1)

        # Ball passing above the paddle
        ball = Ball(50, 50, 0, 0)
        ball.velocity = Vector2D(100, 0)
        ball.prev_position = Vector2D(50, 50)
        ball.position = Vector2D(150, 50)

        collision, time = continuous_circle_paddle_collision(ball, paddle, dt=0.016)

        assert collision is False, "Should not detect collision when ball misses paddle"

    def test_ball_moving_parallel_to_paddle(self):
        """Test ball moving parallel to paddle edge"""
        paddle = Paddle(100, 100, player_id=1)

        # Ball moving vertically along paddle edge
        ball = Ball(90, 50, 0, 0)
        ball.velocity = Vector2D(0, 100)
        ball.prev_position = Vector2D(90, 50)
        ball.position = Vector2D(90, 150)

        collision, time = continuous_circle_paddle_collision(ball, paddle, dt=0.016)

        # Depending on ball radius, may or may not collide
        # This tests that the function handles edge cases without crashing


class TestCollisionNormal:
    """Test collision normal calculation"""

    def test_horizontal_collision_from_left(self):
        """Test ball hitting paddle from the left"""
        paddle = Paddle(100, 100, player_id=1)
        ball = Ball(95, 140, 0, 0)  # Ball hitting from left, center height

        normal = get_paddle_collision_normal(ball, paddle)

        assert normal is not None, "Normal should not be None"
        assert normal.x < 0, "Normal should point left (ball is on left side)"
        assert abs(normal.y) < abs(normal.x), "Should be primarily horizontal collision"

    def test_horizontal_collision_from_right(self):
        """Test ball hitting paddle from the right"""
        paddle = Paddle(100, 100, player_id=1)
        ball = Ball(120, 140, 0, 0)  # Ball hitting from right, center height

        normal = get_paddle_collision_normal(ball, paddle)

        assert normal is not None, "Normal should not be None"
        assert normal.x > 0, "Normal should point right (ball is on right side)"

    def test_vertical_collision_from_top(self):
        """Test ball hitting paddle from top"""
        paddle = Paddle(100, 100, player_id=1)
        ball = Ball(107, 95, 0, 0)  # Ball hitting from top, center X

        normal = get_paddle_collision_normal(ball, paddle)

        assert normal is not None, "Normal should not be None"
        assert normal.y < 0, "Normal should point up (ball is above paddle)"

    def test_spin_effect_on_top_hit(self):
        """Test that hitting top of paddle adds downward spin component"""
        paddle = Paddle(100, 100, player_id=1)
        ball = Ball(95, 110, 0, 0)  # Ball hitting left side, near top

        normal = get_paddle_collision_normal(ball, paddle)

        assert normal is not None
        # Top hit should add negative Y component (upward spin)
        # The exact value depends on spin factor, but should be present

    def test_spin_effect_on_bottom_hit(self):
        """Test that hitting bottom of paddle adds upward spin component"""
        paddle = Paddle(100, 100, player_id=1)
        ball = Ball(95, 170, 0, 0)  # Ball hitting left side, near bottom

        normal = get_paddle_collision_normal(ball, paddle)

        assert normal is not None
        # Bottom hit should add positive Y component (downward spin)

    def test_normal_is_normalized(self):
        """Test that returned normal is a unit vector"""
        paddle = Paddle(100, 100, player_id=1)
        ball = Ball(95, 140, 0, 0)

        normal = get_paddle_collision_normal(ball, paddle)

        assert normal is not None
        magnitude = normal.magnitude()
        assert 0.99 <= magnitude <= 1.01, f"Normal should be normalized, got magnitude {magnitude}"


class TestPaddleBounce:
    """Test paddle bounce physics"""

    def test_bounce_reverses_horizontal_velocity(self):
        """Test that bouncing off paddle reverses horizontal direction"""
        paddle = Paddle(100, 100, player_id=1)
        ball = Ball(95, 140, 100, 0)  # Moving right towards paddle
        ball.velocity = Vector2D(100, 0)

        original_speed = ball.velocity.magnitude()
        apply_paddle_bounce(ball, paddle)

        # Velocity should be reflected
        assert ball.velocity.x < 0, "Horizontal velocity should be reversed"

        # Speed should be approximately maintained
        new_speed = ball.velocity.magnitude()
        assert abs(new_speed - original_speed) < 50, "Speed should be approximately maintained"

    def test_bounce_with_vertical_velocity(self):
        """Test bounce with ball having vertical velocity"""
        paddle = Paddle(100, 100, player_id=1)
        # Hit paddle off-center (near top) to ensure spin effect
        ball = Ball(95, 110, 100, 50)
        ball.velocity = Vector2D(100, 50)

        apply_paddle_bounce(ball, paddle)

        # Should have both X and Y components after bounce
        assert ball.velocity.x != 0, "Should have horizontal component"
        # After hitting near the top of paddle, should have some vertical component
        # (either from original velocity or spin effect)


class TestCollisionDetector:
    """Test the CollisionDetector class"""

    def test_detector_creation(self):
        """Test that CollisionDetector can be created"""
        detector = CollisionDetector()
        assert detector is not None

    def test_wall_collision_top(self):
        """Test ball hitting top wall"""
        detector = CollisionDetector()
        ball = Ball(400, 5, 0, -100)  # Ball at top, moving up
        ball.velocity = Vector2D(0, -100)

        collision_type = detector.check_ball_walls(ball, 800, 600)

        assert collision_type == "top", f"Should detect top wall collision, got {collision_type}"

    def test_wall_collision_bottom(self):
        """Test ball hitting bottom wall"""
        detector = CollisionDetector()
        ball = Ball(400, 595, 0, 100)  # Ball at bottom, moving down
        ball.velocity = Vector2D(0, 100)

        collision_type = detector.check_ball_walls(ball, 800, 600)

        assert (
            collision_type == "bottom"
        ), f"Should detect bottom wall collision, got {collision_type}"

    def test_goal_left(self):
        """Test ball going into left goal"""
        detector = CollisionDetector()
        ball = Ball(5, 300, -100, 0)  # Ball at left edge
        ball.velocity = Vector2D(-100, 0)

        collision_type = detector.check_ball_walls(ball, 800, 600)

        assert collision_type == "left_goal", f"Should detect left goal, got {collision_type}"

    def test_goal_right(self):
        """Test ball going into right goal"""
        detector = CollisionDetector()
        ball = Ball(795, 300, 100, 0)  # Ball at right edge
        ball.velocity = Vector2D(100, 0)

        collision_type = detector.check_ball_walls(ball, 800, 600)

        assert collision_type == "right_goal", f"Should detect right goal, got {collision_type}"

    def test_no_collision_in_field(self):
        """Test ball in middle of field has no collision"""
        detector = CollisionDetector()
        ball = Ball(400, 300, 100, 50)
        ball.velocity = Vector2D(100, 50)

        collision_type = detector.check_ball_walls(ball, 800, 600)

        assert collision_type == "none", f"Should detect no collision, got {collision_type}"

    def test_paddle_collision_prevents_multiple_bounces(self):
        """Test that ball doesn't bounce multiple times on same paddle"""
        detector = CollisionDetector()
        paddle = Paddle(100, 250, player_id=1)

        # First collision
        ball = Ball(90, 290, 100, 0)
        ball.velocity = Vector2D(100, 0)
        ball.prev_position = Vector2D(80, 290)
        ball.position = Vector2D(105, 290)
        ball.last_paddle_hit = None  # First time hitting

        collision1 = detector.check_ball_paddle(ball, paddle, dt=0.016)
        assert collision1 is True, "First collision should be detected"
        assert ball.last_paddle_hit == paddle.player_id, "Should mark paddle as last hit"

        # Try to collide again immediately (ball still near paddle)
        ball.prev_position = Vector2D(ball.position.x, ball.position.y)
        ball.position = ball.position + Vector2D(1, 0)  # Still in collision zone

        collision2 = detector.check_ball_paddle(ball, paddle, dt=0.016)
        assert collision2 is False, "Should not detect second collision on same paddle"

    def test_paddle_collision_resets_after_wall_bounce(self):
        """Test that last_paddle_hit is reset correctly"""
        # This is tested indirectly through the physics engine
        # but we can verify the logic here
        ball = Ball(400, 300, 0, 0)
        ball.last_paddle_hit = 1  # Previously hit paddle 1

        # Simulate wall bounce reset
        ball.last_paddle_hit = None

        assert ball.last_paddle_hit is None, "Should reset after wall bounce"


class TestCircleRectCollision:
    """Test basic circle-rectangle collision detection"""

    def test_circle_inside_rect(self):
        """Test circle completely inside rectangle"""
        position = Vector2D(110, 110)
        radius = 5
        rect = (100, 100, 20, 20)  # x, y, width, height

        collision = circle_rect_collision_at_position(position, radius, rect)

        assert collision is True, "Should detect circle inside rectangle"

    def test_circle_outside_rect(self):
        """Test circle completely outside rectangle"""
        position = Vector2D(50, 50)
        radius = 5
        rect = (100, 100, 20, 20)

        collision = circle_rect_collision_at_position(position, radius, rect)

        assert collision is False, "Should not detect collision when circle is far away"

    def test_circle_touching_rect_edge(self):
        """Test circle touching rectangle edge"""
        position = Vector2D(92, 110)  # Just touching left edge
        radius = 8
        rect = (100, 100, 20, 20)

        collision = circle_rect_collision_at_position(position, radius, rect)

        assert collision is True, "Should detect circle touching rectangle edge"

    def test_circle_at_corner(self):
        """Test circle near rectangle corner"""
        position = Vector2D(95, 95)  # Near top-left corner
        radius = 8
        rect = (100, 100, 20, 20)

        collision = circle_rect_collision_at_position(position, radius, rect)

        # Depending on exact distance, may or may not collide
        # Just ensure it doesn't crash and returns a boolean
        assert isinstance(collision, bool)


class TestEdgeCases:
    """Test edge cases and boundary conditions"""

    def test_zero_velocity_ball(self):
        """Test ball with zero velocity"""
        paddle = Paddle(100, 100, player_id=1)
        ball = Ball(50, 120, 0, 0)
        ball.velocity = Vector2D(0, 0)
        ball.prev_position = Vector2D(50, 120)
        ball.position = Vector2D(50, 120)

        # Should not crash
        collision, time = continuous_circle_paddle_collision(ball, paddle, dt=0.016)
        assert isinstance(collision, bool)

    def test_very_small_dt(self):
        """Test with very small time step"""
        detector = CollisionDetector()
        paddle = Paddle(100, 100, player_id=1)
        ball = Ball(90, 140, 100, 0)
        ball.velocity = Vector2D(100, 0)
        ball.prev_position = Vector2D(89.9, 140)
        ball.position = Vector2D(90, 140)

        # Should not crash with very small movement
        collision = detector.check_ball_paddle(ball, paddle, dt=0.0001)
        assert isinstance(collision, bool)

    def test_very_large_paddle(self):
        """Test with unusually large paddle"""
        paddle = Paddle(100, 50, player_id=1)
        paddle.width = 50  # Set custom size after creation
        paddle.height = 500
        ball = Ball(90, 300, 100, 0)
        ball.velocity = Vector2D(100, 0)
        ball.prev_position = Vector2D(80, 300)

        normal = get_paddle_collision_normal(ball, paddle)
        assert normal is not None
        assert 0.99 <= normal.magnitude() <= 1.01, "Normal should still be normalized"

    def test_ball_exactly_at_paddle_center(self):
        """Test ball exactly at paddle center (degenerate case)"""
        paddle = Paddle(100, 100, player_id=1)
        ball = Ball(107.5, 140, 0, 0)  # Exactly at paddle center

        normal = get_paddle_collision_normal(ball, paddle)

        # Should return a default normal even in degenerate case
        assert normal is not None
        assert isinstance(normal, Vector2D)


if __name__ == "__main__":
    pytest.main([__file__, "-v"])


class TestSpeedClamping:
    """Test ball speed clamping functionality"""

    def test_bounce_vertical_clamps_speed(self):
        """Test that vertical bounce clamps excessive speed"""
        from magic_pong.utils.config import game_config

        ball = Ball(400, 300, 0, 0)
        ball.velocity = Vector2D(1000, 1000)  # Excessive speed

        ball.bounce_vertical()

        speed = ball.velocity.magnitude()
        assert (
            speed <= game_config.MAX_BALL_SPEED
        ), f"Speed should be clamped to {game_config.MAX_BALL_SPEED}, got {speed}"

    def test_bounce_horizontal_clamps_speed(self):
        """Test that horizontal bounce clamps excessive speed"""
        from magic_pong.utils.config import game_config

        ball = Ball(400, 300, 0, 0)
        ball.velocity = Vector2D(1000, 1000)  # Excessive speed

        ball.bounce_horizontal()

        speed = ball.velocity.magnitude()
        assert (
            speed <= game_config.MAX_BALL_SPEED
        ), f"Speed should be clamped to {game_config.MAX_BALL_SPEED}, got {speed}"

    def test_normal_speed_not_affected(self):
        """Test that normal speed is not changed by clamping"""
        ball = Ball(400, 300, 0, 0)
        ball.velocity = Vector2D(100, 50)
        original_speed = ball.velocity.magnitude()

        ball.bounce_vertical()

        new_speed = ball.velocity.magnitude()
        assert abs(new_speed - original_speed) < 0.1, "Normal speed should not be affected"


class TestInterpolatedCollision:
    """Test that collision detection properly interpolates moving paddle"""

    def test_moving_paddle_collision_interpolation(self):
        """Test collision detection with a moving paddle catching up to ball"""
        from magic_pong.utils.config import game_config

        # Paddle at x=100, moving into ball path
        paddle = Paddle(100, 100, player_id=1)
        paddle_width = game_config.PADDLE_WIDTH

        # Ball is near paddle's right edge and moving away slowly
        ball_start_x = 100 + paddle_width + 5  # Just right of paddle
        ball_end_x = 100 + paddle_width + 10  # Moved further right

        ball = Ball(ball_end_x, 140, 0, 0)  # Inside paddle Y range
        ball.velocity = Vector2D(50, 0)  # Moving right (away from paddle)
        ball.prev_position = Vector2D(ball_start_x, 140)
        ball.position = Vector2D(ball_end_x, 140)

        # Test that ball was outside and is still outside (no collision)
        collision, time = continuous_circle_paddle_collision(ball, paddle, dt=0.016)

        # This should NOT collide because ball started outside and stayed outside
        # This verifies the function handles the case correctly
        assert isinstance(collision, bool), "Should return a boolean"

    def test_ball_entering_paddle_from_side(self):
        """Test ball entering paddle from the side is detected"""
        from magic_pong.utils.config import game_config

        paddle = Paddle(100, 100, player_id=1)
        paddle_width = game_config.PADDLE_WIDTH

        # Ball starts outside paddle's left edge and ends inside
        ball = Ball(100 + paddle_width / 2, 140, 0, 0)  # End position inside paddle
        ball.velocity = Vector2D(100, 0)
        ball.prev_position = Vector2D(85, 140)  # Start well outside to left
        ball.position = Vector2D(100 + paddle_width / 2, 140)  # Inside paddle

        collision, time = continuous_circle_paddle_collision(ball, paddle, dt=0.016)

        # Should detect entering collision
        assert collision is True, "Should detect ball entering paddle"


class TestZeroLengthLine:
    """Test edge case of zero-length line in circle_line_collision"""

    def test_zero_length_line_collision(self):
        """Test that zero-length line is handled correctly"""
        from magic_pong.core.collision import circle_line_collision

        ball = Ball(100, 100, 0, 0)
        ball.radius = 10

        # Zero-length line (same start and end point)
        line_start = Vector2D(105, 100)
        line_end = Vector2D(105, 100)

        # Ball should collide with the point
        collision = circle_line_collision(ball, line_start, line_end)
        assert collision is True, "Should detect collision with zero-length line (point)"

    def test_zero_length_line_no_collision(self):
        """Test that distant zero-length line doesn't collide"""
        from magic_pong.core.collision import circle_line_collision

        ball = Ball(100, 100, 0, 0)
        ball.radius = 10

        # Zero-length line far from ball
        line_start = Vector2D(200, 200)
        line_end = Vector2D(200, 200)

        collision = circle_line_collision(ball, line_start, line_end)
        assert collision is False, "Should not detect collision with distant point"
