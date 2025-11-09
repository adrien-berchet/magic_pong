"""
Tests for Magic Pong game entities
"""

from magic_pong.core.entities import Action, Ball, Paddle, Vector2D


class TestVector2D:
    """Tests for Vector2D class"""

    def test_creation(self) -> None:
        """Test vector creation"""
        v = Vector2D(3.0, 4.0)
        assert v.x == 3.0
        assert v.y == 4.0

    def test_addition(self) -> None:
        """Test vector addition"""
        v1 = Vector2D(1.0, 2.0)
        v2 = Vector2D(3.0, 4.0)
        result = v1 + v2
        assert result.x == 4.0
        assert result.y == 6.0

    def test_subtraction(self) -> None:
        """Test vector subtraction"""
        v1 = Vector2D(5.0, 7.0)
        v2 = Vector2D(2.0, 3.0)
        result = v1 - v2
        assert result.x == 3.0
        assert result.y == 4.0

    def test_scalar_multiplication(self) -> None:
        """Test scalar multiplication"""
        v = Vector2D(2.0, 3.0)
        result = v * 2.5
        assert result.x == 5.0
        assert result.y == 7.5

    def test_magnitude(self) -> None:
        """Test magnitude calculation"""
        v = Vector2D(3.0, 4.0)
        assert v.magnitude() == 5.0

    def test_magnitude_zero(self) -> None:
        """Test magnitude of zero vector"""
        v = Vector2D(0.0, 0.0)
        assert v.magnitude() == 0.0

    def test_normalize(self) -> None:
        """Test normalization"""
        v = Vector2D(3.0, 4.0)
        normalized = v.normalize()
        assert abs(normalized.magnitude() - 1.0) < 1e-10
        assert abs(normalized.x - 0.6) < 1e-10
        assert abs(normalized.y - 0.8) < 1e-10

    def test_normalize_zero_vector(self) -> None:
        """Test normalization of zero vector"""
        v = Vector2D(0.0, 0.0)
        normalized = v.normalize()
        assert normalized.x == 0.0
        assert normalized.y == 0.0

    def test_to_tuple(self) -> None:
        """Test tuple conversion"""
        v = Vector2D(1.5, 2.5)
        assert v.to_tuple() == (1.5, 2.5)


class TestAction:
    """Tests for Action class"""

    def test_creation_valid_values(self) -> None:
        """Test creation with valid values"""
        action = Action(0.5, -0.3)
        assert action.move_x == 0.5
        assert action.move_y == -0.3

    def test_clamp_too_large_values(self) -> None:
        """Test clamping of values that are too large"""
        action = Action(2.0, -1.5)
        assert action.move_x == 1.0
        assert action.move_y == -1.0

    def test_clamp_limit_values(self) -> None:
        """Test limit values"""
        action = Action(1.0, -1.0)
        assert action.move_x == 1.0
        assert action.move_y == -1.0


class TestBall:
    """Tests for Ball class"""

    def test_creation(self) -> None:
        """Test ball creation"""
        ball = Ball(100.0, 200.0, 50.0, -30.0)
        assert ball.position.x == 100.0
        assert ball.position.y == 200.0
        assert ball.velocity.x == 50.0
        assert ball.velocity.y == -30.0

    def test_update_position(self) -> None:
        """Test position update"""
        ball = Ball(0.0, 0.0, 100.0, 50.0)
        ball.update(0.1)  # 0.1 second
        assert ball.position.x == 10.0
        assert ball.position.y == 5.0

    def test_bounce_vertical(self) -> None:
        """Test vertical bounce"""
        ball = Ball(0.0, 0.0, 100.0, 50.0)
        ball.bounce_vertical()
        assert ball.velocity.x == 100.0
        assert ball.velocity.y == -50.0

    def test_bounce_horizontal(self) -> None:
        """Test horizontal bounce"""
        ball = Ball(0.0, 0.0, 100.0, 50.0)
        original_speed = ball.velocity.magnitude()
        ball.bounce_horizontal()
        assert ball.velocity.x == -100.0
        assert ball.velocity.y == 50.0
        # Check that speed has increased
        assert ball.velocity.magnitude() == original_speed


class TestPaddle:
    """Tests for Paddle class"""

    def test_creation_left_player(self) -> None:
        """Test creating a paddle for the left player"""
        from magic_pong.utils.config import game_config

        paddle = Paddle(50.0, 100.0, 1)
        assert paddle.position.x == 50.0
        assert paddle.position.y == 100.0
        assert paddle.player_id == 1
        assert paddle.min_x == game_config.PADDLE_MARGIN

    def test_creation_right_player(self) -> None:
        """Test creating a paddle for the right player"""
        paddle = Paddle(700.0, 100.0, 2)
        assert paddle.position.x == 700.0
        assert paddle.position.y == 100.0
        assert paddle.player_id == 2
        # min_x should be half the field width
        assert paddle.min_x > 0

    def test_get_rect(self) -> None:
        """Test getting collision rectangle"""
        paddle = Paddle(100.0, 200.0, 1)
        rect = paddle.get_rect()
        assert rect[0] == 100.0  # x
        assert rect[1] == 200.0  # y
        assert rect[2] == paddle.width
        assert rect[3] == paddle.height

    def test_apply_size_effect(self) -> None:
        """Test applying a size effect"""
        paddle = Paddle(100.0, 200.0, 1)
        original_height = paddle.height
        paddle.apply_size_effect(1.5, 5.0)
        assert paddle.height == original_height * 1.5
        assert paddle.size_effect_timer == 5.0

    def test_reset_size(self) -> None:
        """Test resetting to normal size"""
        paddle = Paddle(100.0, 200.0, 1)
        original_height = paddle.height
        paddle.apply_size_effect(2.0, 5.0)
        paddle.reset_size()
        assert paddle.height == original_height
        assert paddle.size_effect_timer == 5.0
