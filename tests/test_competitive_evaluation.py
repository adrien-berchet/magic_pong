"""Competitive validation must measure weaknesses without changing training."""

import json
import random
from dataclasses import replace

import numpy as np
import pytest
import torch

from magic_pong.ai.competitive import CompetitiveConfig
from magic_pong.ai.competitive import competitive_rank
from magic_pong.ai.competitive import competitive_result_to_dict
from magic_pong.ai.competitive import evaluate_competitive_agent
from magic_pong.ai.competitive import is_better
from magic_pong.ai.competitive import main
from magic_pong.ai.competitive import wilson_lower_bound
from magic_pong.ai.models.simple_ai import create_ai
from magic_pong.utils.config import ai_config
from magic_pong.utils.config import game_config


class StatefulAgent:
    def __init__(self, name="candidate"):
        self.name = name
        self.training_enabled = True
        self.exploration_enabled = True
        self.epsilon = 0.8
        self.current_episode_reward = 3.0
        self.last_state = np.array([1.0, 2.0])
        self._last_chosen_action = 2
        self.nested_state = {"history": [7]}
        self.q_network = torch.nn.Sequential(torch.nn.Linear(2, 2), torch.nn.Dropout())
        self.q_network.train()
        self.q_network[1].eval()


def tiny_config(**kwargs):
    return CompetitiveConfig(
        opponents=("random",),
        seeds=(42, 43),
        held_out_seeds=(10042,),
        max_steps=20,
        **kwargs,
    )


def test_match_grid_uses_same_seeds_both_sides_and_bonus_modes():
    agent = StatefulAgent()
    reused_opponent = StatefulAgent("opponent")
    seen = []

    class Manager:
        def __init__(self, **kwargs):
            assert kwargs["headless"] is True

        def train_episode(self, player1, player2, max_steps):
            for player in (player1, player2):
                assert player.training_enabled is False
                assert player.exploration_enabled is False
                assert player.epsilon == 0
                assert not any(module.training for module in player.q_network.modules())
                assert player.nested_state == {"history": [7]}
                player.nested_state["history"].append(99)
                player._last_chosen_action = 8
            assert game_config.MAX_SCORE == 11
            assert game_config.GAME_SPEED_MULTIPLIER == 1
            assert ai_config.FAST_MODE_MULTIPLIER == 1
            side = 1 if player1.name == "candidate" else 2
            seen.append((side, game_config.BONUSES_ENABLED, random.random()))
            # Player two wins, irrespective of which side is the candidate.
            return {"winner": 2, "steps": 8, "events": [{"player": 2}] * 11}

    result = evaluate_competitive_agent(
        agent,
        tiny_config(),
        opponent_factories={"random": lambda: reused_opponent},
        training_manager_factory=Manager,
    )
    assert len(result.matches) == 8
    assert len(result.matchups) == 4
    assert {(m.player_side, m.bonuses_enabled) for m in result.matchups} == {
        (1, False),
        (1, True),
        (2, False),
        (2, True),
    }
    for matchup in result.matchups:
        assert matchup.matches == 2
        assert matchup.wins == (2 if matchup.player_side == 2 else 0)
        assert matchup.avg_goal_diff == (11 if matchup.player_side == 2 else -11)
    assert len({entry[2] for entry in seen}) == 2
    for player in (agent, reused_opponent):
        assert player.nested_state == {"history": [7]}
        assert player._last_chosen_action == 2
        assert player.training_enabled and player.exploration_enabled
        assert player.epsilon == 0.8
        assert player.q_network.training
        assert not player.q_network[1].training
    payload = competitive_result_to_dict(result)
    assert json.loads(json.dumps(payload))["config"]["seeds"] == [42, 43]
    assert payload["resolved_game_config"]["MAX_SCORE"] == 11


def test_rng_configs_and_players_are_restored_even_on_failure():
    agent = StatefulAgent()
    opponent = StatefulAgent("opponent")
    random.seed(17)
    np.random.seed(18)
    torch.manual_seed(19)
    python_state, numpy_state, torch_state = (
        random.getstate(),
        np.random.get_state(),
        torch.get_rng_state(),
    )
    original_game, original_ai = game_config.model_dump(), ai_config.model_dump()

    class RaisingManager:
        def __init__(self, **kwargs):
            pass

        def train_episode(self, p1, p2, max_steps):
            p1.nested_state["history"].append(4)
            p2.current_episode_reward = 100
            random.random()
            np.random.random()
            torch.rand(1)
            game_config.BONUS_COLORS["extra"] = (1, 2, 3)
            ai_config.USE_PROXIMITY_REWARD = True
            raise RuntimeError("failure")

    with pytest.raises(RuntimeError, match="failure"):
        evaluate_competitive_agent(
            agent,
            tiny_config(),
            opponent_factories={"random": lambda: opponent},
            training_manager_factory=RaisingManager,
        )
    assert random.getstate() == python_state
    np.testing.assert_array_equal(np.random.get_state()[1], numpy_state[1])
    assert torch.equal(torch.get_rng_state(), torch_state)
    assert game_config.model_dump() == original_game
    assert ai_config.model_dump() == original_ai
    assert agent.nested_state == {"history": [7]}
    assert opponent.current_episode_reward == 3


def test_worst_matchup_controls_ranking_and_protocols_must_match():
    class WinningManager:
        def __init__(self, **kwargs):
            pass

        def train_episode(self, p1, p2, max_steps):
            return {"winner": 1, "steps": 1}

    result = evaluate_competitive_agent(
        StatefulAgent(),
        tiny_config(),
        opponent_factories={"random": StatefulAgent},
        training_manager_factory=WinningManager,
    )
    # An all-round 50% player beats a 100%/0% player despite the same mean win rate.
    balanced = replace(
        result,
        matchups=tuple(
            replace(
                m,
                wins=1,
                losses=1,
                win_rate=0.5,
                wilson_lower_bound=wilson_lower_bound(1, 2),
            )
            for m in result.matchups
        ),
    )
    assert competitive_rank(balanced) > competitive_rank(result)
    assert is_better(balanced, result)
    assert is_better(result, None)
    assert not is_better(result, result)
    with pytest.raises(ValueError, match="protocol"):
        is_better(result, replace(result, config=replace(result.config, max_score=1)))


def test_timeouts_are_nonwins_and_held_out_seeds_are_separate():
    class TimeoutManager:
        def __init__(self, **kwargs):
            pass

        def train_episode(self, p1, p2, max_steps):
            return {"winner": 0, "steps": max_steps, "events": [{"player": 1}]}

    result = evaluate_competitive_agent(
        StatefulAgent(),
        tiny_config(),
        split="held_out",
        opponent_factories={"random": StatefulAgent},
        training_manager_factory=TimeoutManager,
    )
    assert {match.seed for match in result.matches} == {10042}
    assert all(m.timeouts == 1 and m.wins == 0 for m in result.matchups)
    assert competitive_rank(result)[0] == 0
    with pytest.raises(ValueError, match="disjoint"):
        CompetitiveConfig(seeds=(1,), held_out_seeds=(1,))
    with pytest.raises(ValueError, match="non-empty"):
        CompetitiveConfig(seeds=())


def test_small_real_physics_run_is_repeatable_and_restores_builtin_history():
    agent = create_ai("random")
    config = CompetitiveConfig(
        opponents=("training_dummy",),
        seeds=(9,),
        held_out_seeds=(10,),
        max_steps=5,
    )
    result1 = evaluate_competitive_agent(agent, config)
    result2 = evaluate_competitive_agent(agent, config)
    assert result1 == result2
    assert len(result1.matches) == 4
    assert all(match.steps == 5 and match.timeout for match in result1.matches)


def test_real_dqn_evaluation_preserves_pending_learning_and_networks():
    from magic_pong.ai.models.dqn_ai import DQNAgent

    agent = DQNAgent(hidden_size=8, batch_size=2, min_replay_size=1)
    opponent = DQNAgent(hidden_size=8, batch_size=2, min_replay_size=1)
    for player in (agent, opponent):
        player.last_state = np.zeros(30, dtype=np.float32)
        player.last_action = 4
        player._last_chosen_action = 4
        player.remember(np.zeros(30), 0, 1.0, np.ones(30), False)
    parameters = [parameter.detach().clone() for parameter in agent.q_network.parameters()]
    config = CompetitiveConfig(
        opponents=("frozen:checkpoint-1",),
        seeds=(8,),
        held_out_seeds=(9,),
        max_steps=3,
    )
    result = evaluate_competitive_agent(
        agent,
        config,
        opponent_factories={"frozen:checkpoint-1": lambda: opponent},
    )
    assert len(result.matches) == 4
    for player in (agent, opponent):
        assert len(player.memory) == 1
        assert player.last_action == 4
        assert player._last_chosen_action == 4
        np.testing.assert_array_equal(player.last_state, np.zeros(30))
        assert player.training_step == 0
        assert player.training_enabled
    assert all(
        torch.equal(before, after)
        for before, after in zip(parameters, agent.q_network.parameters())
    )


def test_cli_writes_serializable_heldout_report(tmp_path, monkeypatch):
    from magic_pong.ai import competitive

    class Manager:
        def __init__(self, **kwargs):
            pass

        def train_episode(self, p1, p2, max_steps):
            return {"winner": 0, "steps": max_steps}

    seen = []

    def fake_checkpoint(path, config, **kwargs):
        seen.append((path, kwargs["split"]))
        return evaluate_competitive_agent(
            StatefulAgent(),
            config,
            opponent_factories={"random": StatefulAgent},
            training_manager_factory=Manager,
            **kwargs,
        )

    monkeypatch.setattr(competitive, "evaluate_competitive_checkpoint", fake_checkpoint)
    output = tmp_path / "report.json"
    assert (
        main(
            [
                "agent.pth",
                "--output",
                str(output),
                "--opponents",
                "random",
                "--seeds",
                "1",
                "--held-out-seeds",
                "2",
                "--max-steps",
                "5",
            ]
        )
        == 0
    )
    assert seen == [("agent.pth", "held_out")]
    payload = json.loads(output.read_text())
    assert payload["total_matches"] == 4
    assert payload["matches_per_matchup"] == 1
    assert payload["timing"]["physics_dt_seconds"] == pytest.approx(1 / 60)


@pytest.mark.parametrize(
    "kwargs",
    [
        {"player_sides": (3,)},
        {"bonus_modes": (0,)},
        {"seeds": (-1,)},
        {"max_steps": 0},
        {"confidence_z": float("nan")},
        {"game_overrides": {"FIELD_HIGHT": 600}},
        {"game_overrides": {"MAX_SCORE": 1}},
    ],
)
def test_invalid_protocol_is_rejected(kwargs):
    with pytest.raises(ValueError):
        CompetitiveConfig(**kwargs)
