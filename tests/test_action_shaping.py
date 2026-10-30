"""Unit tests for inference-time geometric shield and continuous action shaping."""

import numpy as np
import pytest

from magic_pong.ai.action_shaping import GeometricShieldConfig
from magic_pong.ai.action_shaping import shape_action
from magic_pong.ai.agent_adapter import from_player1_dqn_action
from magic_pong.ai.geometry import FEATURE_NAMES
from magic_pong.ai.geometry import get_state_layout
from magic_pong.ai.models.dqn_ai import DQNAgent
from magic_pong.core.entities import Action


def test_geometric_shield_config_validation() -> None:
    """Test validation of GeometricShieldConfig parameters."""
    cfg = GeometricShieldConfig(margin=15.0, soft_x_tau=0.8)
    assert cfg.margin == 15.0
    assert cfg.soft_x_tau == 0.8

    with pytest.raises(ValueError, match="margin must be non-negative"):
        GeometricShieldConfig(margin=-1.0)

    with pytest.raises(ValueError, match="soft_x_tau must be positive"):
        GeometricShieldConfig(soft_x_tau=0.0)

    with pytest.raises(ValueError, match="soft_x_tau must be positive"):
        GeometricShieldConfig(soft_x_tau=-0.5)


def test_shape_action_requires_geometry() -> None:
    """Test that shape_action raises if layout lacks geometry context."""
    layout = get_state_layout(score_context=False, geometry_context=False)
    q = np.zeros(9, dtype=np.float32)
    s = np.zeros(30, dtype=np.float32)
    cfg = GeometricShieldConfig()
    with pytest.raises(ValueError, match="Cannot shape action without geometry context"):
        shape_action(q, s, layout, cfg)


def test_shape_action_barrier_override_and_in_window() -> None:
    """Test that barrier overrides vy when out of window, but preserves argmax vy inside window."""
    layout = get_state_layout(score_context=False, geometry_context=True)
    state = np.zeros(46, dtype=np.float32)

    # Base state:
    # state[5] = paddle top y normalized = 260 / 600
    # state[10] = paddle size multiplier = 1.0 (height = 80)
    # paddle_center_y = 260 + 40 = 300
    state[5] = 260.0 / 600.0
    state[10] = 1.0

    geom_start = layout.geometry_slice.start
    idx_incoming = geom_start + FEATURE_NAMES.index("incoming")
    idx_valid = geom_start + FEATURE_NAMES.index("own_contact_valid")
    idx_intercept = geom_start + FEATURE_NAMES.index("own_intercept_y")

    state[idx_incoming] = 1.0
    state[idx_valid] = 1.0

    # Half height = 40, R = 8, margin = 20 -> capture_width = max(40 + 8 - 20, 8.33) = 28 px
    cfg = GeometricShieldConfig(margin=20.0, soft_x_tau=None)

    # Case 1: Intercept far below paddle center (y = 400, d = +100 > 28)
    state[idx_intercept] = 400.0 / 600.0
    # Argmax chooses action 1: UP (move_y = -1.0)
    q_up = np.zeros(9, dtype=np.float32)
    q_up[1] = 10.0
    action = shape_action(q_up, state, layout, cfg)
    assert action.move_y == 1.0, "Barrier must override to +1.0 (DOWN) when intercept is below"

    # Case 2: Intercept far above paddle center (y = 200, d = -100 < -28)
    state[idx_intercept] = 200.0 / 600.0
    # Argmax chooses action 2: DOWN (move_y = +1.0)
    q_down = np.zeros(9, dtype=np.float32)
    q_down[2] = 10.0
    action = shape_action(q_down, state, layout, cfg)
    assert action.move_y == -1.0, "Barrier must override to -1.0 (UP) when intercept is above"

    # Case 3: Intercept inside capture window (y = 310, d = +10 < 28)
    state[idx_intercept] = 310.0 / 600.0
    # Argmax chooses action 1: UP (move_y = -1.0)
    action = shape_action(q_up, state, layout, cfg)
    assert action.move_y == -1.0, "Inside window, barrier must preserve argmax vy (-1.0)"


def test_shape_action_score_context_offset() -> None:
    """Test that feature indexing properly adapts when score_context=True (49D)."""
    layout = get_state_layout(score_context=True, geometry_context=True)
    assert layout.geometry_slice.start == 33
    state = np.zeros(49, dtype=np.float32)
    state[5] = 260.0 / 600.0
    state[10] = 1.0

    geom_start = layout.geometry_slice.start
    idx_incoming = geom_start + FEATURE_NAMES.index("incoming")
    idx_valid = geom_start + FEATURE_NAMES.index("own_contact_valid")
    idx_intercept = geom_start + FEATURE_NAMES.index("own_intercept_y")

    state[idx_incoming] = 1.0
    state[idx_valid] = 1.0
    state[idx_intercept] = 400.0 / 600.0  # Far below

    q_up = np.zeros(9, dtype=np.float32)
    q_up[1] = 10.0
    cfg = GeometricShieldConfig(margin=20.0, soft_x_tau=None)
    action = shape_action(q_up, state, layout, cfg)
    assert action.move_y == 1.0


def test_shape_action_outgoing_or_invalid_leaves_vy_argmax() -> None:
    """Test that when ball is outgoing or invalid, vy is purely argmax while vx is still blended."""
    layout = get_state_layout(score_context=False, geometry_context=True)
    state = np.zeros(46, dtype=np.float32)
    state[5] = 260.0 / 600.0
    state[10] = 1.0

    geom_start = layout.geometry_slice.start
    idx_incoming = geom_start + FEATURE_NAMES.index("incoming")
    idx_valid = geom_start + FEATURE_NAMES.index("own_contact_valid")
    idx_intercept = geom_start + FEATURE_NAMES.index("own_intercept_y")

    # Ball is outgoing / not valid
    state[idx_incoming] = 0.0
    state[idx_valid] = 0.0
    state[idx_intercept] = 400.0 / 600.0

    q = np.zeros(9, dtype=np.float32)
    q[1] = 5.0  # Action 1: (0, -1) [UP]
    q[4] = 4.0  # Action 4: (1, 0) [RIGHT]

    cfg = GeometricShieldConfig(margin=20.0, soft_x_tau=0.5)
    action = shape_action(q, state, layout, cfg)
    assert action.move_y == -1.0, "vy must remain argmax (-1.0) on outgoing/invalid ball"
    assert action.move_x > 0.0, "vx must be blended positively towards action 4"


def test_shape_action_paddle_shrink_clamp() -> None:
    """Test that small paddle size bonus (e.g. height=20) maintains capture width >= STEP."""
    layout = get_state_layout(score_context=False, geometry_context=True)
    state = np.zeros(46, dtype=np.float32)
    state[5] = 260.0 / 600.0
    # Paddle shrunk: state[10] = 20 / 80 = 0.25 (height = 20, half = 10)
    state[10] = 0.25

    geom_start = layout.geometry_slice.start
    state[geom_start + FEATURE_NAMES.index("incoming")] = 1.0
    state[geom_start + FEATURE_NAMES.index("own_contact_valid")] = 1.0

    # half = 10, R = 8, margin = 20 -> half + R - margin = -2 px
    # Clamping must ensure capture width is at least STEP (8.333 px)
    cfg = GeometricShieldConfig(margin=20.0, soft_x_tau=None)

    # Intercept at 5 px distance (within STEP)
    paddle_center = 260.0 + 10.0
    state[geom_start + FEATURE_NAMES.index("own_intercept_y")] = (paddle_center + 5.0) / 600.0

    q_still = np.zeros(9, dtype=np.float32)
    q_still[0] = 10.0  # Action 0: (0, 0)
    action = shape_action(q_still, state, layout, cfg)
    assert action.move_y == 0.0, (
        "Distance 5px is within STEP (8.33px), so argmax (0.0) is preserved"
    )


def test_player2_mirroring_fractional_x() -> None:
    """Test that player 2 frame mirroring preserves fractional continuous x velocity."""
    action_p1 = Action(move_x=0.42, move_y=0.75)
    action_world_p1 = from_player1_dqn_action(action_p1, player_id=1)
    assert action_world_p1.move_x == 0.42
    assert action_world_p1.move_y == 0.75

    action_world_p2 = from_player1_dqn_action(action_p1, player_id=2)
    assert action_world_p2.move_x == -0.42
    assert action_world_p2.move_y == 0.75


def test_dqn_agent_action_shaping_lifecycle_and_guards() -> None:
    """Test configuration guards, training guards, and memory safety on DQNAgent."""
    agent_no_geom = DQNAgent(state_size=30, geometry_context=False)
    with pytest.raises(ValueError, match="Action shaping requires geometry_context=True"):
        agent_no_geom.configure_action_shaper(GeometricShieldConfig())

    agent = DQNAgent(state_size=46, hidden_layers=[256, 128], geometry_context=True)

    # Agent starts with training_enabled=True, exploration_enabled=True
    with pytest.raises(ValueError, match="cannot be configured while training"):
        agent.configure_action_shaper(GeometricShieldConfig())

    agent.set_training_mode(False)
    # exploration_enabled was also set to False by set_training_mode(False)
    agent.configure_action_shaper(GeometricShieldConfig(margin=20.0, soft_x_tau=0.5))
    assert agent.action_shaper_config is not None

    # Cannot enable training or exploration while shaper is active
    with pytest.raises(ValueError, match="Cannot enable training mode while action shaping"):
        agent.set_training_mode(True)

    with pytest.raises(ValueError, match="Cannot enable exploration mode while action shaping"):
        agent.set_exploration_mode(True)

    # get_action raises if explore=True is passed
    obs = {
        "ball_pos": (400.0, 300.0),
        "ball_vel": (-500.0, 0.0),
        "player_pos": (50.0, 260.0),
        "opponent_pos": (750.0, 260.0),
        "opponent_previous_pos": (750.0, 260.0),
    }
    with pytest.raises(ValueError, match="cannot be used while training or exploring"):
        agent.get_action(obs, explore=True)

    # get_action works when eval mode is properly set
    act = agent.get_action(obs)
    assert isinstance(act, Action)
    assert -1.0 <= act.move_x <= 1.0
    assert -1.0 <= act.move_y <= 1.0
    assert agent._last_chosen_action is not None

    # on_step does not add transitions
    agent.on_step(obs, act, reward=1.0, done=False, info={})
    assert len(agent.memory) == 0, "No transitions recorded while shaped"

    # Resetting shaper restores unshaped behavior
    agent.configure_action_shaper(None)
    assert agent.action_shaper_config is None
    agent.set_training_mode(True)
    assert agent.training_enabled is True
