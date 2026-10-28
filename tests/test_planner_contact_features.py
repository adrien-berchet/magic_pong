"""Analytic tests for pure observation-derived contact geometry."""

from dataclasses import replace

import numpy as np
import pytest

from docs.ai.planner_contact_features import FEATURE_NAMES
from docs.ai.planner_contact_features import ContactGeometryConfig
from docs.ai.planner_contact_features import contact_features


@pytest.fixture
def config():
    return ContactGeometryConfig(
        width=100,
        height=100,
        max_ball_speed=50,
        paddle_width=2,
        paddle_height=20,
        paddle_speed=10,
        ball_radius=1,
    )


def sample():
    state = np.zeros((1, 30), dtype=np.float32)
    state[0, :8] = [0.28, 0.60, -0.4, 0, 0.05, 0.40, 0.93, 0.40]
    state[0, 10:12] = 1
    return state


def named(state, config):
    return dict(zip(FEATURE_NAMES, contact_features(state, config)[0]))


def test_analytic_incoming_contact_and_normalizations(config):
    state = sample()
    original = state.copy()
    expected = [0.2, 0.1, 0.64, 0.1, 1, 2 / 3, 1, 0.6, 0, 0.1, 1, 0, 0, 0, 0, 0]
    features = contact_features(state, config)
    assert features.dtype == np.float32 and features.shape == (1, 16)
    np.testing.assert_allclose(features[0], expected, atol=1e-6)
    np.testing.assert_array_equal(state, original)


def test_miss_sign_reachability_and_observed_size_ratio(config):
    state = sample()
    state[0, 1] = 0.8
    normal = named(state, config)
    assert normal["own_signed_miss"] == pytest.approx(0.19)
    assert normal["own_reachability_margin"] == pytest.approx(-0.09)
    assert normal["own_contact_offset"] == pytest.approx(3)
    # Doubling the observed height, keeping its center at y=50, makes the save reachable.
    state[0, 10], state[0, 5] = 2, 0.3
    enlarged = named(state, config)
    assert enlarged["own_signed_miss"] == pytest.approx(0.09)
    assert enlarged["own_reachability_margin"] == pytest.approx(0.01)
    assert enlarged["own_contact_offset"] == pytest.approx(1.5)
    state[0, 1] = 0.2
    above = named(state, config)
    assert above["own_signed_miss"] == pytest.approx(-0.09)
    assert above["own_contact_offset"] == pytest.approx(-1.5)


def test_wall_reflection_accounts_for_ball_radius_and_multiple_bounces(config):
    state = sample()
    state[0, 1], state[0, 3] = 0.95, 0.4
    # y=95 +20*1 =>115; bottom ball-center wall is99, so reflection gives83.
    assert named(state, config)["own_intercept_y"] == pytest.approx(0.83, abs=1e-6)
    fast = replace(config, max_ball_speed=500)
    state[0, 2], state[0, 3] = -0.04, 0.4
    # y=95 +200*1 =>295; two complete wall-span traversals return to the bottom wall.
    assert named(state, fast)["own_intercept_y"] == pytest.approx(0.99, abs=1e-6)


@pytest.mark.parametrize("vx", [0, 1e-9, -1e-9, 1e-6, -1e-6])
def test_zero_and_nearzero_horizontal_velocity_are_finite_and_masked(config, vx):
    state = sample()
    state[0, 2] = vx
    result = named(state, config)
    assert all(np.isfinite(value) for value in result.values())
    for feature in (
        "incoming",
        "outgoing",
        "own_contact_time",
        "own_contact_valid",
        "own_intercept_y",
        "own_signed_miss",
        "own_reachability_margin",
        "own_contact_offset",
        "opponent_contact_time",
        "opponent_contact_valid",
        "opponent_intercept_y",
        "opponent_contact_offset",
    ):
        assert result[feature] == 0


def test_far_contact_caps_time_but_does_not_present_a_capped_intercept(config):
    state = sample()
    state[0, 0] = 0.48  # gap40 at20 pixels/s =>2s, beyond the1.5s horizon.
    result = named(state, config)
    assert result["incoming"] == 1 and result["own_contact_time"] == 1
    assert result["own_contact_valid"] == 0
    assert result["own_intercept_y"] == result["own_signed_miss"] == 0
    assert result["own_reachability_margin"] == result["own_contact_offset"] == 0
    state[0, 0] = 0.06  # Already behind the inward contact plane at x=8.
    result = named(state, config)
    assert result["own_face_gap"] == pytest.approx(-0.02)
    assert result["own_contact_valid"] == result["own_contact_time"] == 0


def test_outgoing_opponent_contact_and_canonical_side_symmetry(config):
    state = sample()
    state[0, 0], state[0, 1], state[0, 2] = 0.72, 0.8, 0.4
    result = named(state, config)
    assert result["incoming"] == result["own_contact_valid"] == 0
    assert result["outgoing"] == result["opponent_contact_valid"] == 1
    assert result["opponent_contact_time"] == pytest.approx(2 / 3)
    assert result["opponent_intercept_y"] == pytest.approx(0.8)
    assert result["opponent_contact_offset"] == pytest.approx(3)

    # A mirrored world/player role produces exactly these same canonical inputs.
    def mirror_world_positions(canonical):
        mirrored = canonical.copy()
        mirrored[:, 0] = 1 - mirrored[:, 0]
        mirrored[:, 2] *= -1
        for index in (4, 6, 8):
            mirrored[:, index] = 1 - mirrored[:, index] - config.paddle_width / config.width
        return mirrored

    np.testing.assert_allclose(
        contact_features(mirror_world_positions(mirror_world_positions(state)), config),
        contact_features(state, config),
        atol=1e-6,
    )


def test_vectorization_empty_inputs_and_unused_context(config):
    states = np.repeat(sample(), 3, axis=0)
    states[1, 1] = 0.8
    states[2, 2] *= -1
    batched = contact_features(states, config)
    individual = np.concatenate([contact_features(row[None], config) for row in states])
    np.testing.assert_array_equal(batched, individual)
    changed = states.copy()
    changed[:, 8:10] = 0.123
    changed[:, 12:] = 0.456
    np.testing.assert_array_equal(batched, contact_features(changed, config))
    assert contact_features(np.zeros((0, 30)), config).shape == (0, 16)
    zero_height = sample()
    zero_height[:, 10:12] = 0
    assert np.isfinite(contact_features(zero_height, config)).all()


def test_invalid_state_and_config_fail_explicitly(config):
    with pytest.raises(ValueError, match="shape"):
        contact_features(np.zeros((2, 33)), config)
    state = sample()
    state[0, 0] = np.nan
    with pytest.raises(ValueError, match="finite"):
        contact_features(state, config)
    state = sample()
    state[0, 10] = 5
    with pytest.raises(ValueError, match="height ratios"):
        contact_features(state, config)
    with pytest.raises(ValueError, match="diameter"):
        replace(config, height=2)
    with pytest.raises(ValueError, match="finite and positive"):
        replace(config, fps=0)
    with pytest.raises(ValueError, match="positive integer"):
        replace(config, horizon_frames=1.5)
