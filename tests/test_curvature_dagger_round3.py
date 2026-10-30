"""Unit tests for Curvature & Bounce-Targeted DAgger (Round 3) utilities."""

import numpy as np
import pytest
import torch

from magic_pong.ai.action_shaping import GeometricShieldConfig
from magic_pong.ai.action_shaping import shape_action
from magic_pong.ai.curvature_dagger import compute_concession_per_hit
from magic_pong.ai.curvature_dagger import filter_targeted_tiers
from magic_pong.ai.curvature_dagger import project_planner_target
from magic_pong.ai.curvature_dagger import vectorized_barrier
from magic_pong.ai.geometry import get_state_layout
from magic_pong.ai.models.dqn_ai import DQNAgent


def test_vectorized_barrier_matches_shape_action():
    """Verify bit-for-bit parity between vectorized_barrier and shape_action."""
    layout = get_state_layout(score_context=False, geometry_context=True)
    cfg = GeometricShieldConfig(margin=20.0)

    rng = np.random.default_rng(123)
    n = 300
    states = rng.uniform(0.1, 0.9, size=(n, 46)).astype(np.float32)

    # Force some to be incoming and valid
    states[:150, 34] = 1.0  # incoming
    states[:150, 36] = 1.0  # valid

    # Diverse intercept points
    states[:75, 37] = 0.05  # near top edge
    states[75:150, 37] = 0.95  # near bottom edge

    active, vy = vectorized_barrier(states, margin=cfg.margin)

    for i in range(n):
        s = states[i]
        q = rng.normal(size=9).astype(np.float32)
        act = shape_action(q, s, layout, cfg)
        if active[i]:
            assert act.move_y == vy[i], (
                f"Mismatch at index {i}: shape_action={act.move_y}, barrier={vy[i]}"
            )
        else:
            # When barrier is not active, vy should be 0.0 or match nothing
            pass


def test_filter_targeted_tiers():
    """Verify targeted tier filtering logic and inevitable concede exclusion."""
    n = 100
    states = np.zeros((n, 46), dtype=np.float32)
    values = np.zeros((n, 9), dtype=np.float32)
    ga_flags = np.zeros(n, dtype=bool)
    ga_actions = np.zeros((n, 9), dtype=bool)

    # State 0: Normal center play (Tier 1 false, Tier 2 false, Tier 3 false)
    states[0, 34] = 1.0  # incoming
    states[0, 36] = 1.0  # valid
    states[0, 1] = 0.5  # ball in center y
    states[0, 2] = -0.2  # vx slow
    states[0, 3] = 0.0  # vy 0
    states[0, 30] = 0.5  # own_face_gap
    states[0, 35] = 0.8  # time far
    states[0, 5] = 0.5  # paddle center
    states[0, 10] = 1.0  # paddle height
    states[0, 37] = 0.5  # intercept center (dist ~ 0)

    # State 1: Tier 1 candidate (fast incoming, short time, large distance)
    states[1, 34] = 1.0  # incoming
    states[1, 36] = 1.0  # valid
    states[1, 2] = -0.8  # vx fast (-720 px/s)
    states[1, 3] = 0.4
    states[1, 35] = 0.3  # t <= 60 frames
    states[1, 5] = 0.2  # paddle at top
    states[1, 10] = 1.0  # paddle height multiplier 1.0
    states[1, 37] = 0.8  # intercept at bottom (dist > 20px)

    # State 2: Tier 2 conceded shot frame (e.g. goal conceded at index 70, hit at 60)
    hit_flags = np.zeros(n, dtype=bool)
    hit_flags[60] = True
    ga_flags[70] = True

    # State 3: Tier 3 wall-bounce arrival candidate
    states[3, 34] = 1.0
    states[3, 36] = 1.0
    states[3, 1] = 0.8  # ball y
    states[3, 3] = 0.9  # vy downward
    states[3, 2] = -0.5  # vx
    states[3, 30] = 0.7  # own_face_gap

    # State 4: Candidate where all actions concede
    states[4, 34] = 1.0
    states[4, 36] = 1.0
    states[4, 2] = -0.9
    states[4, 35] = 0.2
    states[4, 37] = 0.9
    ga_actions[4, :] = True  # all actions concede

    tier_mask, base_mask = filter_targeted_tiers(
        states, values, ga_flags, ga_actions, hit_against_flags=hit_flags
    )

    assert not tier_mask[0]
    assert tier_mask[1], "Tier 1 should be selected"
    assert tier_mask[65], "Tier 2 shot frame should be selected"
    assert tier_mask[3], "Tier 3 should be selected"
    assert not tier_mask[4], "State 4 should be dropped because all actions concede"


def test_project_planner_target_barrier_enforcement():
    """Verify planner target projection restricts probability mass to valid vertical moves."""
    states = np.zeros((2, 46), dtype=np.float32)
    # State 0: Inactive barrier (incoming=0)
    states[0, 34] = 0.0

    # State 1: Active barrier (intercept low, paddle high -> vy must be +1.0)
    states[1, 34] = 1.0
    states[1, 36] = 1.0
    states[1, 5] = 0.1  # paddle top
    states[1, 10] = 1.0
    states[1, 37] = 0.9  # intercept bottom

    values = torch.zeros((2, 9), dtype=torch.float32)
    values[1, 1] = 10.0  # planner erroneously preferred up (action 1)
    values[1, 2] = 5.0  # planner secondary choice down (action 2)

    frozen_logits = torch.randn((2, 9), dtype=torch.float32)

    targ, fallbacks = project_planner_target(
        values, torch.as_tensor(states), frozen_logits, temperature=0.03
    )

    # State 1: Required vy is +1.0. Valid action indices are 2, 7, 8.
    valid_indices = [2, 7, 8]
    invalid_indices = [0, 1, 3, 4, 5, 6]

    for inv in invalid_indices:
        assert targ[1, inv].item() < 1e-6, f"Action {inv} should have 0 mass"
    assert targ[1, valid_indices].sum().item() == pytest.approx(1.0, abs=1e-5)


def test_compute_concession_per_hit():
    """Test concession per hit calculation."""
    assert compute_concession_per_hit(0, 100) == 0.0
    assert compute_concession_per_hit(5, 200) == 0.025
    assert compute_concession_per_hit(10, 0) == 10.0  # safe division with max(1, 0)


def test_checkpoint_save_and_load_roundtrip(tmp_path):
    """Verify DQNAgent checkpoint roundtrip with geometry context and target network sync."""
    agent = DQNAgent()
    agent.load_model("models/dqn_v7_promoted.pth")
    assert agent.geometry_context is True
    assert agent.q_network.geometry_context_weight is not None

    agent.update_target_network()
    save_path = tmp_path / "test_roundtrip.pth"
    agent.save_model(str(save_path))

    # Reload into fresh agent
    loaded = DQNAgent()
    loaded.load_model(str(save_path))
    assert loaded.geometry_context is True
    assert loaded.q_network.geometry_context_weight is not None

    # Check parameter equality
    for p1, p2 in zip(agent.q_network.parameters(), loaded.q_network.parameters()):
        assert torch.allclose(p1, p2)


def test_optimizer_contains_geometry_context_weight():
    """Verify fresh Adam optimizer includes geometry_context_weight."""
    agent = DQNAgent()
    agent.load_model("models/dqn_v7_promoted.pth")
    q_net = agent.q_network

    optimizer = torch.optim.Adam(q_net.parameters(), lr=2e-4)
    opt_params = {p for g in optimizer.param_groups for p in g["params"]}
    assert {p for p in q_net.parameters() if p.requires_grad} == opt_params
    assert q_net.geometry_context_weight in opt_params
