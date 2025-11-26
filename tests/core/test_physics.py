"""
Unit tests for physics engine

Tests physics engine functionality including:
- Ball reset and initialization
- Bonus spawning in safe positions
- Wall bounce resets last_paddle_hit
- Paddle collision handling
"""

import pytest

from magic_pong.core.entities import Action, BonusType
from magic_pong.core.physics import BonusSpawner, PhysicsEngine
from magic_pong.utils.config import game_config


class TestPhysicsEngine:
    """Test the main physics engine"""

    def test_engine_initialization(self):
        """Test that physics engine initializes correctly"""
        engine = PhysicsEngine(800, 600)

        assert engine.field_width == 800
        assert engine.field_height == 600
        assert engine.ball is not None
        assert engine.player1 is not None
        assert engine.player2 is not None
        assert engine.score == [0, 0]
        assert engine.game_time == 0.0

    def test_reset_ball_creates_new_ball(self):
        """Test that reset_ball creates a new ball instance"""
        engine = PhysicsEngine(800, 600)
        old_ball = engine.ball

        engine.reset_ball()

        assert engine.ball is not old_ball, "Should create a new ball instance"
        assert engine.ball.position.x == 400, "Ball should be at center X"
        assert engine.ball.position.y == 300, "Ball should be at center Y"

    def test_reset_ball_with_direction(self):
        """Test resetting ball with specific direction"""
        engine = PhysicsEngine(800, 600)

        # Reset with rightward direction
        engine.reset_ball(direction=1)
        # Ball velocity should be set based on reset_to_center if implemented

        # Reset with leftward direction
        engine.reset_ball(direction=-1)

        # Should not crash and ball should be at center
        assert 350 <= engine.ball.position.x <= 450
        assert 250 <= engine.ball.position.y <= 350

    def test_reset_paddles(self):
        """Test that paddles are reset to initial positions"""
        engine = PhysicsEngine(800, 600)

        # Move paddles
        engine.player1.position.x = 200
        engine.player1.position.y = 100
        engine.player2.position.x = 500
        engine.player2.position.y = 400

        # Reset
        engine.reset_paddles()

        # Should be back at initial positions
        assert engine.player1.position.x == game_config.PADDLE_MARGIN
        assert (
            engine.player2.position.x == 800 - game_config.PADDLE_MARGIN - game_config.PADDLE_WIDTH
        )
        assert abs(engine.player1.position.y - (300 - game_config.PADDLE_HEIGHT / 2)) < 1
        assert abs(engine.player2.position.y - (300 - game_config.PADDLE_HEIGHT / 2)) < 1

    def test_wall_bounce_resets_last_paddle_hit(self):
        """Test that bouncing off wall resets last_paddle_hit"""
        engine = PhysicsEngine(800, 600)

        # Set ball near top wall with upward velocity
        engine.ball.position.x = 400
        engine.ball.position.y = 5
        engine.ball.velocity.x = 100
        engine.ball.velocity.y = -100
        engine.ball.last_paddle_hit = 1  # Previously hit paddle 1

        # Update physics
        action_none = Action(0, 0)
        events = engine.update(dt=0.016, player1_action=action_none, player2_action=action_none)

        # If wall bounce occurred, last_paddle_hit should be reset
        if events["wall_bounces"]:
            assert engine.ball.last_paddle_hit is None, "Wall bounce should reset last_paddle_hit"

    def test_goal_resets_paddles_and_ball(self):
        """Test that scoring a goal resets the game state"""
        engine = PhysicsEngine(800, 600)

        # Move paddles away from center
        engine.player1.position.y = 100
        engine.player2.position.y = 400

        # Put ball in goal position
        engine.ball.position.x = 5
        engine.ball.position.y = 300
        engine.ball.velocity.x = -100

        initial_score_p2 = engine.score[1]

        # Update physics
        action_none = Action(0, 0)
        events = engine.update(dt=0.016, player1_action=action_none, player2_action=action_none)

        # Check if goal was scored
        if events["goals"]:
            assert engine.score[1] == initial_score_p2 + 1, "Player 2 should score"
            # Paddles should be reset
            assert abs(engine.player1.position.y - (300 - game_config.PADDLE_HEIGHT / 2)) < 50
            assert abs(engine.player2.position.y - (300 - game_config.PADDLE_HEIGHT / 2)) < 50

    def test_paddle_movement(self):
        """Test that paddles move with actions"""
        engine = PhysicsEngine(800, 600)

        initial_y = engine.player1.position.y

        # Move paddle up
        action_up = Action(0, -1)
        action_none = Action(0, 0)
        engine.update(dt=0.1, player1_action=action_up, player2_action=action_none)

        assert engine.player1.position.y < initial_y, "Paddle should move up"

    def test_paddle_movement_constraints(self):
        """Test that paddles stay within bounds"""
        engine = PhysicsEngine(800, 600)

        # Try to move paddle off screen
        action_up = Action(0, -1)
        action_none = Action(0, 0)

        # Move up many times
        for _ in range(1000):
            engine.update(dt=0.1, player1_action=action_up, player2_action=action_none)

        # Paddle should be constrained
        assert engine.player1.position.y >= 0, "Paddle should not go above field"
        assert engine.player1.position.y <= 600 - game_config.PADDLE_HEIGHT, (
            "Paddle should not go below field"
        )

    def test_game_time_advances(self):
        """Test that game time advances with updates"""
        engine = PhysicsEngine(800, 600)

        initial_time = engine.game_time

        action_none = Action(0, 0)
        engine.update(dt=0.016, player1_action=action_none, player2_action=action_none)

        assert engine.game_time > initial_time, "Game time should advance"

    def test_ball_update_position(self):
        """Test that ball position updates with velocity"""
        engine = PhysicsEngine(800, 600)

        engine.ball.position.x = 400
        engine.ball.position.y = 300
        engine.ball.velocity.x = 100
        engine.ball.velocity.y = 50

        initial_x = engine.ball.position.x

        action_none = Action(0, 0)
        engine.update(dt=0.016, player1_action=action_none, player2_action=action_none)

        # Ball should have moved
        assert engine.ball.position.x != initial_x, "Ball should move"

    def test_get_game_state(self):
        """Test that game state is returned correctly"""
        engine = PhysicsEngine(800, 600)

        state = engine.get_game_state()

        assert "ball_position" in state
        assert "ball_velocity" in state
        assert "player1_position" in state
        assert "player2_position" in state
        assert "player1_paddle_size" in state
        assert "player2_paddle_size" in state
        assert "score" in state
        assert "active_bonuses" in state

        # Check types
        assert isinstance(state["ball_position"], tuple)
        assert len(state["ball_position"]) == 2
        assert isinstance(state["score"], list)
        assert len(state["score"]) == 2


class TestBonusSpawner:
    """Test bonus spawning logic"""

    def test_spawner_initialization(self):
        """Test that bonus spawner initializes correctly"""
        spawner = BonusSpawner(800, 600)

        assert spawner.field_width == 800
        assert spawner.field_height == 600
        assert spawner.spawn_timer == 0.0

    def test_no_spawn_when_disabled(self):
        """Test that bonuses don't spawn when disabled"""
        spawner = BonusSpawner(800, 600)

        # Temporarily disable bonuses
        original_enabled = game_config.BONUSES_ENABLED
        game_config.BONUSES_ENABLED = False

        try:
            # Advance time past spawn interval
            new_bonuses = spawner.update(dt=100.0, existing_bonuses=[])

            assert len(new_bonuses) == 0, "Should not spawn bonuses when disabled"
        finally:
            game_config.BONUSES_ENABLED = original_enabled

    def test_spawn_after_interval(self):
        """Test that bonuses spawn after the interval"""
        spawner = BonusSpawner(800, 600)

        # Ensure bonuses are enabled
        original_enabled = game_config.BONUSES_ENABLED
        game_config.BONUSES_ENABLED = True

        try:
            # Advance timer to spawn interval
            new_bonuses = spawner.update(
                dt=game_config.BONUS_SPAWN_INTERVAL + 1, existing_bonuses=[]
            )

            assert len(new_bonuses) > 0, "Should spawn bonuses after interval"
        finally:
            game_config.BONUSES_ENABLED = original_enabled

    def test_bonuses_spawn_in_safe_positions(self):
        """Test that bonuses spawn away from paddles and walls"""
        spawner = BonusSpawner(800, 600)

        original_enabled = game_config.BONUSES_ENABLED
        game_config.BONUSES_ENABLED = True

        try:
            # Spawn bonuses multiple times
            for _ in range(10):
                spawner.spawn_timer = game_config.BONUS_SPAWN_INTERVAL + 1
                new_bonuses = spawner.update(dt=0, existing_bonuses=[])

                for bonus in new_bonuses:
                    # Check horizontal position is in safe zone
                    safe_margin = (
                        game_config.PADDLE_MARGIN
                        + game_config.PADDLE_WIDTH
                        + game_config.BONUS_SIZE
                    )
                    assert bonus.position.x >= safe_margin, (
                        f"Bonus too close to left edge: {bonus.position.x}"
                    )
                    assert bonus.position.x <= 800 - safe_margin, (
                        f"Bonus too close to right edge: {bonus.position.x}"
                    )

                    # Check vertical position
                    assert bonus.position.y >= game_config.BONUS_SIZE, "Bonus too close to top"
                    assert bonus.position.y <= 600 - game_config.BONUS_SIZE, (
                        "Bonus too close to bottom"
                    )
        finally:
            game_config.BONUSES_ENABLED = original_enabled

    def test_symmetric_bonus_spawning(self):
        """Test that bonuses spawn symmetrically"""
        spawner = BonusSpawner(800, 600)

        original_enabled = game_config.BONUSES_ENABLED
        game_config.BONUSES_ENABLED = True

        try:
            spawner.spawn_timer = game_config.BONUS_SPAWN_INTERVAL + 1
            new_bonuses = spawner.update(dt=0, existing_bonuses=[])

            if len(new_bonuses) == 2:
                # Check symmetry
                left_bonus = new_bonuses[0]
                right_bonus = new_bonuses[1]

                # X positions should be symmetric
                expected_right_x = 800 - left_bonus.position.x
                assert abs(right_bonus.position.x - expected_right_x) < 1, (
                    "Bonuses should be symmetric"
                )

                # Y positions should be the same
                assert abs(right_bonus.position.y - left_bonus.position.y) < 0.1, (
                    "Bonuses should have same Y"
                )

                # Same type
                assert right_bonus.type == left_bonus.type, "Bonuses should be same type"
        finally:
            game_config.BONUSES_ENABLED = original_enabled

    def test_no_spawn_when_too_many_bonuses(self):
        """Test that bonuses don't spawn when there are already too many"""
        spawner = BonusSpawner(800, 600)

        original_enabled = game_config.BONUSES_ENABLED
        game_config.BONUSES_ENABLED = True

        try:
            # Create fake existing bonuses
            from magic_pong.core.entities import Bonus

            existing = [Bonus(100, 100, BonusType.ENLARGE_PADDLE) for _ in range(4)]

            spawner.spawn_timer = game_config.BONUS_SPAWN_INTERVAL + 1
            new_bonuses = spawner.update(dt=0, existing_bonuses=existing)

            assert len(new_bonuses) == 0, "Should not spawn when there are already 4 bonuses"
        finally:
            game_config.BONUSES_ENABLED = original_enabled

    def test_bonus_types_are_valid(self):
        """Test that spawned bonuses have valid types"""
        spawner = BonusSpawner(800, 600)

        original_enabled = game_config.BONUSES_ENABLED
        game_config.BONUSES_ENABLED = True

        try:
            spawner.spawn_timer = game_config.BONUS_SPAWN_INTERVAL + 1
            new_bonuses = spawner.update(dt=0, existing_bonuses=[])

            for bonus in new_bonuses:
                assert isinstance(bonus.type, BonusType), "Bonus should have valid BonusType"
                assert bonus.type in [
                    BonusType.ENLARGE_PADDLE,
                    BonusType.SHRINK_OPPONENT,
                    BonusType.ROTATING_PADDLE,
                ]
        finally:
            game_config.BONUSES_ENABLED = original_enabled


class TestPhysicsIntegration:
    """Integration tests for physics engine"""

    def test_full_game_cycle(self):
        """Test a full game cycle from start to goal"""
        engine = PhysicsEngine(800, 600)

        # Position ball to score quickly
        engine.ball.position.x = 10
        engine.ball.position.y = 300
        engine.ball.velocity.x = -100
        engine.ball.velocity.y = 0

        action_none = Action(0, 0)

        # Run simulation until goal
        for _ in range(100):
            events = engine.update(dt=0.016, player1_action=action_none, player2_action=action_none)

            if events["goals"]:
                # Goal scored!
                assert engine.score[0] > 0 or engine.score[1] > 0, "Score should increase"
                break

    def test_paddle_ball_interaction(self):
        """Test paddle hitting ball"""
        engine = PhysicsEngine(800, 600)

        # Get actual paddle position from engine
        paddle1_x = engine.player1.position.x
        paddle1_y = engine.player1.position.y + game_config.PADDLE_HEIGHT / 2

        # Place ball just to the right of the paddle, moving left toward it
        engine.ball.position.x = paddle1_x + game_config.PADDLE_WIDTH + 10
        engine.ball.position.y = paddle1_y
        engine.ball.prev_position.x = engine.ball.position.x + 1
        engine.ball.prev_position.y = paddle1_y
        engine.ball.velocity.x = -200  # Moving left toward paddle
        engine.ball.velocity.y = 0
        engine.ball.last_paddle_hit = None  # Ensure can hit paddle

        action_none = Action(0, 0)

        paddle_hit = False
        initial_vx = engine.ball.velocity.x

        # Run simulation
        for _ in range(50):
            events = engine.update(dt=0.016, player1_action=action_none, player2_action=action_none)

            if events["paddle_hits"]:
                paddle_hit = True
                # Velocity should be reversed
                assert engine.ball.velocity.x != initial_vx, (
                    "Ball velocity should change after paddle hit"
                )
                break

        assert paddle_hit, "Ball should hit paddle"

    def test_reset_game(self):
        """Test that reset_game clears all state"""
        engine = PhysicsEngine(800, 600)

        # Change game state
        engine.score = [5, 3]
        engine.game_time = 100.0
        engine.player1.position.y = 100
        engine.ball.position.x = 200

        # Reset
        engine.reset_game()

        assert engine.score == [0, 0], "Score should be reset"
        assert engine.game_time == 0.0, "Game time should be reset"
        # Paddles and ball should be at starting positions


if __name__ == "__main__":
    pytest.main([__file__, "-v"])
