"""Unit tests for Dueling DQN architecture, folding, step-0 invariance, and serialization."""

import numpy as np
import torch
import torch.nn as nn

from magic_pong.ai.models.dqn_ai import DQNAgent
from magic_pong.ai.models.dqn_ai import DQNNetwork
from magic_pong.ai.models.dqn_checkpoint import safe_torch_load
from magic_pong.ai.models.dqn_checkpoint import validate_dqn_checkpoint
from magic_pong.ai.vector_training import VectorDQNTrainer
from magic_pong.ai.vector_training import VectorTrainingConfig


def test_dueling_network_initialization_and_forward():
    """Verify that a dueling DQN network forward pass produces valid shapes."""
    net = DQNNetwork(
        input_size=46,
        hidden_layers=[256, 128],
        output_size=9,
        geometry_context=True,
        dueling=True,
    )
    assert net.dueling is True
    assert hasattr(net, "advantage_layer")
    assert hasattr(net, "value_layer")
    assert hasattr(net, "log_tau")

    x = torch.randn(8, 46)
    q = net(x)
    assert q.shape == (8, 9)


def test_dueling_fold_exactness():
    """Verify that fold_dueling_weights produces identical outputs to the dueling forward pass."""
    net = DQNNetwork(
        input_size=46,
        hidden_layers=[256, 128],
        output_size=9,
        geometry_context=True,
        dueling=True,
    )
    # Give non-trivial weights to value, advantage, and log_tau
    nn.init.normal_(net.advantage_layer.weight, std=0.5)
    nn.init.normal_(net.advantage_layer.bias, std=0.5)
    nn.init.normal_(net.value_layer.weight, std=0.5)
    nn.init.normal_(net.value_layer.bias, std=0.5)
    net.log_tau.data.fill_(float(np.log(0.042)))

    folded = net.fold_to_standard()
    assert folded.dueling is False
    assert hasattr(folded, "output_layer")

    x = torch.randn(32, 46)
    with torch.no_grad():
        q_dueling = net(x)
        q_folded = folded(x)

    # In float32, numerical precision difference should be <= 1e-6
    diff = torch.abs(q_dueling - q_folded).max().item()
    assert diff < 1e-5, f"Folded Q values differ by {diff} > 1e-5"


def test_conversion_from_promoted_preserves_greedy_policy():
    """Converting dqn_v7_promoted.pth to dueling must preserve exact greedy action selection."""
    agent = DQNAgent(
        state_size=46,
        hidden_layers=[256, 128],
        geometry_context=True,
    )
    agent.load_model("models/dqn_v7_promoted.pth")

    device = next(agent.q_network.parameters()).device
    x = torch.randn(100, 46, device=device)
    with torch.no_grad():
        orig_q = agent.q_network(x)
        orig_actions = orig_q.argmax(dim=-1)

    agent.enable_dueling(tau=0.03, q_semantics="logits")
    with torch.no_grad():
        duel_q = agent.q_network(x)
        duel_actions = duel_q.argmax(dim=-1)

    # Argmax actions must be 100% identical on all states
    np.testing.assert_array_equal(
        orig_actions.cpu().numpy(),
        duel_actions.cpu().numpy(),
        err_msg="Greedy action selection changed after dueling conversion!",
    )


def test_phase1_freeze_parameter_isolation():
    """In Phase 1, only value_layer and log_tau receive gradients; trunk and advantage remain frozen."""
    agent = DQNAgent(
        state_size=46,
        hidden_layers=[256, 128],
        geometry_context=True,
    )
    agent.load_model("models/dqn_v7_promoted.pth")
    agent.enable_dueling(tau=0.03, q_semantics="logits")

    # Freeze for phase 1
    agent.freeze_for_phase1()

    # Verify requires_grad
    for name, p in agent.q_network.named_parameters():
        if "value_layer" in name or name == "log_tau":
            assert p.requires_grad is True, f"{name} should be trainable in Phase 1"
        else:
            assert p.requires_grad is False, f"{name} should be frozen in Phase 1"

    # Count trainable parameters: 128 weights + 1 bias + 1 log_tau = 130
    trainable = [p for p in agent.q_network.parameters() if p.requires_grad]
    trainable_count = sum(p.numel() for p in trainable)
    assert trainable_count == 130, f"Expected 130 trainable params, got {trainable_count}"


def test_save_and_validate_folded_checkpoint(tmp_path):
    """Saving a dueling agent folds weights and passes strict validate_dqn_checkpoint."""
    agent = DQNAgent(
        state_size=46,
        hidden_layers=[256, 128],
        geometry_context=True,
    )
    agent.load_model("models/dqn_v7_promoted.pth")
    agent.enable_dueling(tau=0.03, q_semantics="logits")

    save_path = tmp_path / "folded_test.pth"
    agent.save_model(str(save_path))

    # Must load and validate cleanly with standard validator
    checkpoint = safe_torch_load(str(save_path))
    val = validate_dqn_checkpoint(checkpoint)
    assert val.valid is True
    assert "output_layer.weight" in checkpoint["q_network_state_dict"]
    assert "advantage_layer.weight" not in checkpoint["q_network_state_dict"]
    assert checkpoint["metadata"].get("dueling_trained") is True
    assert checkpoint["metadata"].get("q_semantics") == "returns"


def test_vector_trainer_dueling_two_phase_warm_start(tmp_path):
    """Vector trainer executes two-phase warm start with dueling DQN and outputs valid checkpoints."""
    config = VectorTrainingConfig(
        total_frames=1200,
        n_envs=16,
        action_repeat=4,
        batch_size=32,
        replay_capacity=2000,
        min_replay=64,
        target_update_frames=200,
        snapshot_interval_frames=400,
        eval_interval_frames=2000,
        checkpoint_interval_frames=600,
        dueling=True,
        warm_start_tau=0.03,
        phase1_frames=400,
        phase1_lr=2e-4,
        phase2_lr=3e-5,
        warmup_updates=10,
        initial_checkpoint="models/dqn_v7_promoted.pth",
        geometry_context=True,
        device="cpu",
    )
    trainer = VectorDQNTrainer(config, tmp_path / "run")
    assert trainer.in_phase1 is True
    assert trainer.agent.dueling is True

    summary = trainer.run(verbose=False)
    assert summary["frames"] >= 1200
    assert trainer.in_phase1 is False  # Transitioned to Phase 2

    saved = trainer.save("final.pth")
    assert saved.is_file()
    cp = safe_torch_load(str(saved))
    val = validate_dqn_checkpoint(cp)
    assert val.valid is True
    assert cp["metadata"].get("dueling_trained") is True
