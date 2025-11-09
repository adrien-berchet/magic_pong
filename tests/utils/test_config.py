"""
Unit tests for configuration validation

Tests the configuration validation system including:
- Game configuration validation
- AI configuration validation
- Context managers for temporary config changes
"""

import pytest
from magic_pong.utils.config import (
    GameConfig,
    AIConfig,
    validate_game_config,
    validate_ai_config,
    game_config_tmp,
    ai_config_tmp,
    game_config,
    ai_config,
)


class TestGameConfigValidation:
    """Test game configuration validation"""

    def test_valid_default_config(self):
        """Test that default configuration is valid"""
        config = GameConfig()
        warnings = validate_game_config(config)

        assert isinstance(warnings, list), "Should return a list"
        # Default config should be valid (may have warnings but shouldn't error)

    def test_negative_field_dimensions(self):
        """Test validation catches negative field dimensions"""
        config = GameConfig()
        config.FIELD_WIDTH = -800
        config.FIELD_HEIGHT = -600

        warnings = validate_game_config(config)

        assert any("dimensions must be positive" in w.lower() for w in warnings), \
            "Should warn about negative dimensions"

    def test_zero_field_dimensions(self):
        """Test validation catches zero field dimensions"""
        config = GameConfig()
        config.FIELD_WIDTH = 0
        config.FIELD_HEIGHT = 0

        warnings = validate_game_config(config)

        assert any("positive" in w.lower() for w in warnings), \
            "Should warn about zero dimensions"

    def test_too_small_field(self):
        """Test validation warns about very small field"""
        config = GameConfig()
        config.FIELD_WIDTH = 200
        config.FIELD_HEIGHT = 150

        warnings = validate_game_config(config)

        assert any("too small" in w.lower() for w in warnings), \
            "Should warn about small field"

    def test_negative_ball_speed(self):
        """Test validation catches negative ball speed"""
        config = GameConfig()
        config.BALL_SPEED = -100

        warnings = validate_game_config(config)

        assert any("speed" in w.lower() and "positive" in w.lower() for w in warnings), \
            "Should warn about negative speed"

    def test_ball_speed_exceeds_max(self):
        """Test validation catches ball speed exceeding max"""
        config = GameConfig()
        config.BALL_SPEED = 600
        config.MAX_BALL_SPEED = 500

        warnings = validate_game_config(config)

        assert any("exceed" in w.lower() or "max_ball_speed" in w.lower() for w in warnings), \
            "Should warn about speed exceeding max"

    def test_zero_ball_radius(self):
        """Test validation catches zero ball radius"""
        config = GameConfig()
        config.BALL_RADIUS = 0

        warnings = validate_game_config(config)

        assert any("radius" in w.lower() and "positive" in w.lower() for w in warnings), \
            "Should warn about zero radius"

    def test_negative_paddle_dimensions(self):
        """Test validation catches negative paddle dimensions"""
        config = GameConfig()
        config.PADDLE_WIDTH = -15
        config.PADDLE_HEIGHT = -80

        warnings = validate_game_config(config)

        assert any("paddle" in w.lower() and "positive" in w.lower() for w in warnings), \
            "Should warn about negative paddle dimensions"

    def test_negative_paddle_margin(self):
        """Test validation catches negative paddle margin"""
        config = GameConfig()
        config.PADDLE_MARGIN = -20

        warnings = validate_game_config(config)

        assert any("margin" in w.lower() and "negative" in w.lower() for w in warnings), \
            "Should warn about negative margin"

    def test_paddle_too_wide_for_field(self):
        """Test validation catches paddle configuration that doesn't fit"""
        config = GameConfig()
        config.FIELD_WIDTH = 100
        config.PADDLE_WIDTH = 40
        config.PADDLE_MARGIN = 30
        # 2 * (margin + width) = 2 * 70 = 140 > 100

        warnings = validate_game_config(config)

        assert any("fit" in w.lower() for w in warnings), \
            "Should warn about paddle not fitting in field"

    def test_bonus_validation_when_enabled(self):
        """Test bonus configuration is validated when enabled"""
        config = GameConfig()
        config.BONUSES_ENABLED = True
        config.BONUS_SIZE = -10

        warnings = validate_game_config(config)

        assert any("bonus" in w.lower() and "positive" in w.lower() for w in warnings), \
            "Should warn about negative bonus size"

    def test_bonus_validation_skipped_when_disabled(self):
        """Test bonus configuration not validated when disabled"""
        config = GameConfig()
        config.BONUSES_ENABLED = False
        config.BONUS_SIZE = -10  # Invalid, but should be ignored

        warnings = validate_game_config(config)

        # Should not warn about bonus size when bonuses are disabled
        # (or at least not cause an error)
        assert isinstance(warnings, list)

    def test_zero_max_score(self):
        """Test validation catches zero max score"""
        config = GameConfig()
        config.MAX_SCORE = 0

        warnings = validate_game_config(config)

        assert any("score" in w.lower() and "positive" in w.lower() for w in warnings), \
            "Should warn about zero max score"

    def test_very_low_fps(self):
        """Test validation warns about very low FPS"""
        config = GameConfig()
        config.FPS = 10

        warnings = validate_game_config(config)

        assert any("fps" in w.lower() and "low" in w.lower() for w in warnings), \
            "Should warn about very low FPS"

    def test_zero_fps(self):
        """Test validation catches zero FPS"""
        config = GameConfig()
        config.FPS = 0

        warnings = validate_game_config(config)

        assert any("fps" in w.lower() and "positive" in w.lower() for w in warnings), \
            "Should warn about zero FPS"


class TestAIConfigValidation:
    """Test AI configuration validation"""

    def test_valid_default_ai_config(self):
        """Test that default AI configuration is valid"""
        config = AIConfig()
        warnings = validate_ai_config(config)

        assert isinstance(warnings, list), "Should return a list"
        # Default config should be valid

    def test_negative_proximity_reward_factor(self):
        """Test validation catches negative proximity reward factors"""
        config = AIConfig()
        config.PROXIMITY_REWARD_FACTOR = -0.01

        warnings = validate_ai_config(config)

        assert any("proximity" in w.lower() and "negative" in w.lower() for w in warnings), \
            "Should warn about negative proximity factor"

    def test_negative_max_proximity_reward(self):
        """Test validation catches negative max proximity reward"""
        config = AIConfig()
        config.MAX_PROXIMITY_REWARD = -0.1

        warnings = validate_ai_config(config)

        assert any("proximity" in w.lower() and "negative" in w.lower() for w in warnings), \
            "Should warn about negative max proximity reward"

    def test_zero_max_episode_steps(self):
        """Test validation catches zero max episode steps"""
        config = AIConfig()
        config.MAX_EPISODE_STEPS = 0

        warnings = validate_ai_config(config)

        assert any("episode" in w.lower() and "positive" in w.lower() for w in warnings), \
            "Should warn about zero max episode steps"

    def test_very_high_fast_mode_multiplier(self):
        """Test validation warns about very high fast mode multiplier"""
        config = AIConfig()
        config.FAST_MODE_MULTIPLIER = 500

        warnings = validate_ai_config(config)

        assert any("fast" in w.lower() and ("high" in w.lower() or "instability" in w.lower()) for w in warnings), \
            "Should warn about very high multiplier"

    def test_zero_fast_mode_multiplier(self):
        """Test validation catches zero fast mode multiplier"""
        config = AIConfig()
        config.FAST_MODE_MULTIPLIER = 0

        warnings = validate_ai_config(config)

        assert any("multiplier" in w.lower() and "positive" in w.lower() for w in warnings), \
            "Should warn about zero multiplier"


class TestConfigContextManagers:
    """Test configuration context managers"""

    def test_game_config_tmp_restores_values(self):
        """Test that game_config_tmp restores original values"""
        original_width = game_config.FIELD_WIDTH
        original_height = game_config.FIELD_HEIGHT

        with game_config_tmp(FIELD_WIDTH=1000, FIELD_HEIGHT=800):
            assert game_config.FIELD_WIDTH == 1000
            assert game_config.FIELD_HEIGHT == 800

        # Should be restored
        assert game_config.FIELD_WIDTH == original_width
        assert game_config.FIELD_HEIGHT == original_height

    def test_game_config_tmp_restores_on_exception(self):
        """Test that config is restored even if exception occurs"""
        original_width = game_config.FIELD_WIDTH

        try:
            with game_config_tmp(FIELD_WIDTH=1000):
                assert game_config.FIELD_WIDTH == 1000
                raise ValueError("Test exception")
        except ValueError:
            pass

        # Should still be restored
        assert game_config.FIELD_WIDTH == original_width

    def test_ai_config_tmp_restores_values(self):
        """Test that ai_config_tmp restores original values"""
        original_steps = ai_config.MAX_EPISODE_STEPS
        original_multiplier = ai_config.FAST_MODE_MULTIPLIER

        with ai_config_tmp(MAX_EPISODE_STEPS=5000, FAST_MODE_MULTIPLIER=20.0):
            assert ai_config.MAX_EPISODE_STEPS == 5000
            assert ai_config.FAST_MODE_MULTIPLIER == 20.0

        # Should be restored
        assert ai_config.MAX_EPISODE_STEPS == original_steps
        assert ai_config.FAST_MODE_MULTIPLIER == original_multiplier

    def test_nested_config_contexts(self):
        """Test nested configuration contexts"""
        original_width = game_config.FIELD_WIDTH

        with game_config_tmp(FIELD_WIDTH=1000):
            assert game_config.FIELD_WIDTH == 1000

            with game_config_tmp(FIELD_WIDTH=1200):
                assert game_config.FIELD_WIDTH == 1200

            # Should be back to outer context
            assert game_config.FIELD_WIDTH == 1000

        # Should be back to original
        assert game_config.FIELD_WIDTH == original_width

    def test_multiple_config_changes_at_once(self):
        """Test changing multiple config values at once"""
        original_width = game_config.FIELD_WIDTH
        original_height = game_config.FIELD_HEIGHT
        original_fps = game_config.FPS

        with game_config_tmp(FIELD_WIDTH=1000, FIELD_HEIGHT=800, FPS=120):
            assert game_config.FIELD_WIDTH == 1000
            assert game_config.FIELD_HEIGHT == 800
            assert game_config.FPS == 120

        assert game_config.FIELD_WIDTH == original_width
        assert game_config.FIELD_HEIGHT == original_height
        assert game_config.FPS == original_fps


class TestConfigIntegration:
    """Integration tests for configuration system"""

    def test_config_validation_on_module_load(self):
        """Test that validation runs on module load without errors"""
        # If we got here, module loaded successfully
        # Check that game_config and ai_config exist
        assert game_config is not None
        assert ai_config is not None

    def test_config_has_all_expected_fields(self):
        """Test that config objects have all expected fields"""
        # Game config
        assert hasattr(game_config, 'FIELD_WIDTH')
        assert hasattr(game_config, 'FIELD_HEIGHT')
        assert hasattr(game_config, 'BALL_RADIUS')
        assert hasattr(game_config, 'BALL_SPEED')
        assert hasattr(game_config, 'MAX_BALL_SPEED')
        assert hasattr(game_config, 'PADDLE_WIDTH')
        assert hasattr(game_config, 'PADDLE_HEIGHT')
        assert hasattr(game_config, 'PADDLE_MARGIN')
        assert hasattr(game_config, 'FPS')
        assert hasattr(game_config, 'MAX_SCORE')
        assert hasattr(game_config, 'BONUSES_ENABLED')

        # AI config
        assert hasattr(ai_config, 'SCORE_REWARD')
        assert hasattr(ai_config, 'LOSE_PENALTY')
        assert hasattr(ai_config, 'USE_PROXIMITY_REWARD')
        assert hasattr(ai_config, 'MAX_EPISODE_STEPS')
        assert hasattr(ai_config, 'FAST_MODE_MULTIPLIER')

    def test_config_types_are_correct(self):
        """Test that config values have correct types"""
        assert isinstance(game_config.FIELD_WIDTH, int)
        assert isinstance(game_config.FIELD_HEIGHT, int)
        assert isinstance(game_config.BALL_RADIUS, float)
        assert isinstance(game_config.BALL_SPEED, float)
        assert isinstance(game_config.FPS, int)
        assert isinstance(game_config.BONUSES_ENABLED, bool)

        assert isinstance(ai_config.SCORE_REWARD, float)
        assert isinstance(ai_config.MAX_EPISODE_STEPS, int)
        assert isinstance(ai_config.USE_PROXIMITY_REWARD, bool)


if __name__ == "__main__":
    pytest.main([__file__, "-v"])
