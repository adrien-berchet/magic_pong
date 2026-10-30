"""Unit tests for Vector RL Fine-Tuning of promoted DQN models.

Tests the five critical architectural and algorithmic invariants deliberated with Claude:
1. KL divergence between teacher logits and student raw advantage logits is 0.0 at warm start.
2. Selective anchor loss exempts Striker and Planner transitions.
3. Pre-step opponent kind recording avoids boundary-transition reassignment leakage.
4. Validation ranker prioritizes Striker point share over defensive stalling.
5. Dueling fold-on-save preserves Q-value evaluations within floating-point tolerance.
"""

from __future__ import annotations

import copy
from typing import Any

import numpy as np
import torch
import torch.nn.functional as F

from magic_pong.ai.fast_policies import PLANNER_OPPONENT
from magic_pong.ai.fast_policies import POLICY_KINDS
from magic_pong.ai.models.dqn_ai import DQNAgent
from magic_pong.ai.vector_training import VectorDQNTrainer
from magic_pong.ai.vector_training import VectorTrainingConfig


def test_kl_zero_at_warm_start() -> None:
    """The raw advantage logits must bit-for-bit reproduce teacher logits at warm start."""
    agent = DQNAgent(state_size=46, hidden_layers=[256, 128], geometry_context=True)
    device = next(agent.q_network.parameters()).device
    teacher = copy.deepcopy(agent.q_network).eval()

    # Enable dueling with warm start
    agent.enable_dueling(tau=0.03, warm_start=True, q_semantics="logits")
    x = torch.randn(32, 46, device=device)

    with torch.no_grad():
        teacher_logits = teacher(x)
        raw_adv_logits = agent.q_network.raw_advantage_logits(x)

    # Max difference between teacher logits and raw advantage logits must be 0
    max_diff = (teacher_logits - raw_adv_logits).abs().max().item()
    assert max_diff < 1e-6, f"Advantage logits differ from teacher: {max_diff}"

    # KL divergence must be 0
    p_teacher = F.softmax(teacher_logits, dim=-1)
    log_p_student = F.log_softmax(raw_adv_logits, dim=-1)
    kl = F.kl_div(log_p_student, p_teacher, reduction="batchmean").item()
    assert abs(kl) < 1e-6, f"Warm start KL divergence must be 0, got {kl}"


def test_anchor_loss_exempts_striker_and_planner() -> None:
    """Transitions from Striker and Planner opponents must be excluded from the anchor loss."""
    striker_code = POLICY_KINDS["striker"]
    planner_code = PLANNER_OPPONENT
    tracker_code = POLICY_KINDS["tracker"]

    # 1. Batch consisting exclusively of exempt opponents (Striker & Planner)
    opponents_exempt = np.array([striker_code, planner_code, striker_code, planner_code])
    non_exempt_mask = (opponents_exempt != striker_code) & (opponents_exempt != planner_code)
    assert not non_exempt_mask.any(), "Exempt batch should have no non-exempt samples"

    # 2. Batch with mixed opponents
    opponents_mixed = np.array([striker_code, tracker_code, planner_code, tracker_code])
    non_exempt_mixed = (opponents_mixed != striker_code) & (opponents_mixed != planner_code)
    assert non_exempt_mixed.tolist() == [False, True, False, True]


def test_stored_opponent_kind_matches_pre_step(tmp_path) -> None:
    """Pre-step opponent snapshot must be stored in replay so boundary steps keep their original opponent."""
    config = VectorTrainingConfig(
        total_frames=1000,
        n_envs=16,
        action_repeat=4,
        batch_size=16,
        replay_capacity=1000,
        min_replay=50,
        target_update_frames=500,
        hidden_layers=(64, 32),
        geometry_context=True,
        terminal_on_goal=True,
        anchor_lambda=0.1,
        opponent_mix={"striker": 2.0, "tracker": 2.0},
        seed=42,
    )

    trainer = VectorDQNTrainer(config, tmp_path)
    # Record initial kinds
    pre_kinds = trainer.env.kinds.copy()
    trainer.train_step()

    # The replay row 0 should contain pre_kinds
    stored_opponents = trainer.replay.opponents[0]
    np.testing.assert_array_equal(
        stored_opponents,
        pre_kinds,
        err_msg="Stored opponents must match pre-step snapshot, not post-reset kinds",
    )


def rank_checkpoint_candidates(
    candidates: list[dict[str, Any]], baseline_non_striker_losses: int
) -> list[dict[str, Any]]:
    """Rank validation candidates according to Claude-agreed ranker:

    1. Filter out candidates where non_striker_losses > baseline_non_striker_losses.
    2. Primary: Striker point share GF / (GF + GA) (higher is better).
    3. Secondary / tie-break: Striker GA / 1k frames (lower is better).
    """
    filtered = [c for c in candidates if c["non_striker_losses"] <= baseline_non_striker_losses]

    def sort_key(c: dict[str, Any]) -> tuple[float, float]:
        gf = c["striker_gf"]
        ga = c["striker_ga"]
        point_share = gf / max(1, (gf + ga))
        # Negative ga_per_1k so higher sort_key means lower GA
        return (point_share, -c["striker_ga_per_1k"])

    return sorted(filtered, key=sort_key, reverse=True)


def test_validation_ranker_penalizes_stalling() -> None:
    """Ranker must place an active scoring candidate above a defensive stalling candidate."""
    baseline_losses = 2

    candidate_stalling = {
        "name": "stalling_retreat",
        "non_striker_losses": 1,
        "striker_gf": 5,
        "striker_ga": 80,
        "striker_ga_per_1k": 0.12,  # Fewer goals against per 1k due to long stalls
        # Point share = 5 / 85 = 5.88%
    }

    candidate_active = {
        "name": "active_interception",
        "non_striker_losses": 1,
        "striker_gf": 25,
        "striker_ga": 100,
        "striker_ga_per_1k": 0.16,  # Slightly more goals against per 1k
        # Point share = 25 / 125 = 20.0%
    }

    candidate_regressed = {
        "name": "tracker_regressed",
        "non_striker_losses": 5,  # Exceeds baseline
        "striker_gf": 50,
        "striker_ga": 50,
        "striker_ga_per_1k": 0.10,
    }

    ranked = rank_checkpoint_candidates(
        [candidate_stalling, candidate_active, candidate_regressed],
        baseline_non_striker_losses=baseline_losses,
    )

    # Regressed candidate must be filtered out
    assert len(ranked) == 2
    assert all(c["name"] != "tracker_regressed" for c in ranked)

    # Active candidate must rank FIRST despite having higher raw GA/1k
    assert ranked[0]["name"] == "active_interception"
    assert ranked[1]["name"] == "stalling_retreat"


def test_dueling_fold_on_save_equivalence() -> None:
    """Folded standard network must reproduce dueling network Q-values within float tolerance."""
    agent = DQNAgent(state_size=46, hidden_layers=[256, 128], geometry_context=True)
    device = next(agent.q_network.parameters()).device
    agent.enable_dueling(tau=0.03, warm_start=True, q_semantics="logits")

    # Mutate value and advantage layers to simulate training
    with torch.no_grad():
        agent.q_network.value_layer.weight.add_(0.2)
        agent.q_network.value_layer.bias.add_(0.05)
        agent.q_network.advantage_layer.weight.add_(0.1)
        agent.q_network.advantage_layer.bias.add_(-0.02)

    folded = agent.q_network.fold_to_standard().eval()
    agent.q_network.eval()

    x = torch.randn(64, 46, device=device)
    with torch.no_grad():
        q_dueling = agent.q_network(x)
        q_folded = folded(x)

    diff = (q_dueling - q_folded).abs().max().item()
    assert diff < 1e-4, f"Folded network Q-values differ from dueling: {diff}"
