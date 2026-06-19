"""Focused tests for DQN parity and replay stability."""

from typing import Any

import numpy as np
import pytest

from magic_pong.ai.models.dqn_checkpoint import EXPECTED_STATE_SIZE


def _transition(index: int) -> tuple[np.ndarray, int, float, np.ndarray, bool]:
    # States in the replay buffer are raw 30-dim observation vectors.
    state = np.full(EXPECTED_STATE_SIZE, index / 10.0, dtype=np.float32)
    next_state = state + 0.05
    return state, index % 9, float(index), next_state, index % 3 == 0


def _dqn_observation() -> dict[str, Any]:
    return {
        "ball_pos": [0.4, 0.5],
        "ball_vel": [-0.2, 0.1],
        "player_pos": [0.05, 0.45],
        "opponent_pos": [0.95, 0.55],
        "opponent_previous_pos": [0.95, 0.56],
        "field_width": 1.0,
        "field_height": 1.0,
        "player_paddle_size": 1.0,
        "opponent_paddle_size": 1.0,
        "score_diff": 0.0,
        "time_elapsed": 0.0,
        "bonuses": [],
        "rotating_paddles": [],
    }


def test_dqn_network_single_and_batched_forward_match() -> None:
    torch = pytest.importorskip("torch")
    from magic_pong.ai.models.dqn_ai import DQNNetwork

    torch.manual_seed(123)
    network = DQNNetwork(input_size=32, hidden_size=16, output_size=3, layer_size=2)
    network.eval()
    sample = torch.linspace(-1.0, 1.0, steps=32).unsqueeze(0)
    batch = torch.cat([sample, sample], dim=0)

    with torch.no_grad():
        single_output = network(sample)
        batched_output = network(batch)[:1]

    assert torch.allclose(single_output, batched_output, atol=1e-6)


def test_dqn_replay_loss_is_deterministic() -> None:
    torch = pytest.importorskip("torch")
    from magic_pong.ai.models.dqn_ai import DQNAgent
    from magic_pong.ai.models.dqn_ai import Transition
    from magic_pong.ai.models.dqn_ai import device

    torch.manual_seed(123)
    agent = DQNAgent(batch_size=4, min_replay_size=4)
    agent.q_network.train()
    agent.target_network.train()
    experiences = [Transition(*_transition(index)) for index in range(4)]
    weights = torch.ones(len(experiences), device=device)

    first_loss = agent._compute_dqn_loss(experiences, weights).detach()
    second_loss = agent._compute_dqn_loss(experiences, weights).detach()

    assert first_loss.item() == pytest.approx(second_loss.item())
    assert agent.q_network.training is True
    assert agent.target_network.training is True


def test_standard_replay_clips_gradients(monkeypatch: Any) -> None:
    torch = pytest.importorskip("torch")
    from magic_pong.ai.models.dqn_ai import DQNAgent

    agent = DQNAgent(batch_size=4, min_replay_size=4)
    for transition in (_transition(index) for index in range(4)):
        agent.remember(*transition)

    calls: list[float] = []

    def fake_clip_grad_norm_(parameters: Any, max_norm: float) -> torch.Tensor:
        list(parameters)
        calls.append(max_norm)
        return torch.tensor(0.0)

    monkeypatch.setattr(torch.nn.utils, "clip_grad_norm_", fake_clip_grad_norm_)

    loss = agent.replay()

    assert loss is not None
    assert calls == [1.0]


def test_eval_action_selection_does_not_mutate_replay_or_training_state() -> None:
    pytest.importorskip("torch")
    from magic_pong.ai.models.dqn_ai import DQNAgent

    agent = DQNAgent(epsilon=0.0, batch_size=4, min_replay_size=4)
    agent.remember(*_transition(1))
    agent.set_training_mode(False)
    observation = _dqn_observation()
    memory_size = len(agent.memory)

    action = agent.get_action(observation)

    assert agent.last_state is None
    assert agent.last_action is None

    agent.on_step(observation, action, reward=1.0, done=False, info={})

    assert len(agent.memory) == memory_size
    assert agent.training_step == 0
    assert agent.step_count == 0
    assert agent.loss_history == []
    assert agent.reward_history == []
    assert agent.last_state is None
    assert agent.last_action is None
