import numpy as np
import pytest
import torch

from docs.ai.run_late_decline_diagnostics import decompose_td
from docs.ai.run_late_decline_diagnostics import paired_matches
from docs.ai.run_late_decline_diagnostics import replay_arrays
from docs.ai.run_late_decline_diagnostics import reward_components


@pytest.mark.parametrize("scalar_tensors", [False, True])
def test_replay_accepts_actual_mixed_tensor_list_bundle(scalar_tensors):
    replay = {
        "states": torch.zeros(2, 33),
        "next_states": torch.ones(2, 33),
        "actions": [1, 3],
        "rewards": [1.01, -1.0],
        "dones": [False, True],
    }
    if scalar_tensors:
        for name in ("actions", "rewards", "dones"):
            replay[name] = torch.tensor(replay[name])
    result = replay_arrays(replay)
    assert result["actions"].dtype == np.int64
    assert result["dones"].dtype == np.bool_
    assert result["states"].shape == (2, 33)
    np.testing.assert_allclose(result["rewards"], [1.01, -1.0])
    replay["actions"] = [1]
    with pytest.raises(ValueError, match="shapes"):
        replay_arrays(replay)


def test_td_uses_online_selection_target_evaluation_and_physical_terminal_mask():
    online = np.array([[3, 1], [1, 4], [4, 0]], dtype=float)
    target = np.array([[5, 8], [7, 2], [6, 9]], dtype=float)
    q = np.array([[1, 2], [2, 3], [0, 1]], dtype=float)
    out = decompose_td(
        q,
        online,
        target,
        np.array([0, 1, 0]),
        np.array([1.0, -1.0, 0.01]),
        np.array([True, False, False]),
        0.5,
    )
    np.testing.assert_allclose(out["bootstrap"], [0, 1, 3])
    np.testing.assert_allclose(out["target"], [1, 0, 3.01])
    np.testing.assert_allclose(out["td"], [0, -3, 3.01])


def test_goal_reward_comes_from_events_even_when_auxiliary_coincides():
    episodes = [
        {
            "side": 2,
            "goal_transitions": [{"frame": 8, "scorers": [2]}, {"frame": 10, "scorers": [1]}],
        }
    ]
    goal, aux = reward_components(episodes, 10, np.array([1.01, 0.02, -1.0]))
    np.testing.assert_array_equal(goal, [1, 0, -1])
    np.testing.assert_allclose(aux, [0.01, 0.02, 0])


def test_pairing_reports_lost_gained_and_goal_changes_without_external_ids():
    base = {
        "opponent": "random",
        "player_side": 2,
        "bonuses_enabled": False,
        "seed": 101,
        "agent_won": True,
        "goals_for": 11,
        "goals_against": 4,
        "timeout": False,
        "steps": 300,
    }
    last = {**base, "agent_won": False, "goals_for": 2, "goals_against": 11}
    result = paired_matches([base], [last])
    assert result["lost_wins"] == 1
    assert result["gained_wins"] == 0
    assert result["pairs"][0]["margin_change"] == -16
    with pytest.raises(ValueError, match="match keys"):
        paired_matches([base], [{**last, "seed": 102}])


def test_gradient_components_sum_to_actual_squared_td_gradient():
    from docs.ai.run_late_decline_diagnostics import gradient_parts

    net = torch.nn.Linear(2, 2, bias=False)
    x = torch.tensor([[1.0, 0.0], [0.0, 1.0], [1.0, 1.0]])
    actions = torch.tensor([0, 1, 0])
    goal = torch.tensor([1.0, -1.0, 0.0])
    aux = torch.tensor([0.01, 0.0, 0.02])
    bootstrap = torch.tensor([0.0, 0.4, 0.3])
    actual = net(x).gather(1, actions[:, None]).squeeze(1)
    loss = ((goal + aux + bootstrap - actual) ** 2).mean()
    expected = torch.autograd.grad(loss, tuple(net.parameters()))[0].flatten()
    parts = gradient_parts(net, x, actions, goal, aux, bootstrap)
    torch.testing.assert_close(sum(parts.values()), expected)


def test_real_trace_records_cutoff_action_and_preserves_frozen_weights(monkeypatch):
    from docs.ai.run_late_decline_diagnostics import trace_one
    from magic_pong.ai.models import dqn_ai
    from magic_pong.ai.pilot import PilotConfig
    from magic_pong.ai.pilot import _isolated_training
    from magic_pong.utils.config import ai_config

    monkeypatch.setattr(dqn_ai, "device", torch.device("cpu"))
    agent = dqn_ai.DQNAgent(score_context=True)
    agent.set_training_mode(False)
    agent.set_exploration_mode(False)
    before = {k: v.clone() for k, v in agent.q_network.state_dict().items()}
    source = {"label": "test", "seed": 33, "variant": "score_context", "stage": 2}
    with _isolated_training(PilotConfig(threads=1)):
        ai_config.REWARD_SHAPING_MODE = "legacy"
        ai_config.USE_PROXIMITY_REWARD = False
        ai_config.MICRO_REWARD_SCALE = 0.1
        row, probes, _ = trace_one(source, agent, agent, "follow_ball", 2, False, 12, 8)
    assert row["steps"] == 8
    assert row["timeout"]
    assert len(probes) == 8
    assert sorted(p["frame"] for p in probes) == list(range(8))
    assert all(0 <= p["action"] < 9 for p in probes)
    assert all(p["state"].shape == (33,) and p["next_state"].shape == (33,) for p in probes)
    assert not len(agent.memory)
    assert not agent.training_step
    for key, value in before.items():
        torch.testing.assert_close(value, agent.q_network.state_dict()[key], rtol=0, atol=0)
