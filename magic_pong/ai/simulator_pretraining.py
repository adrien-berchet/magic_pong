"""Imitate bounded physics lookahead on mixed teacher/learner trajectories.

The label buffer aggregates earlier learner-visited states (DAgger style).
Cross-entropy trains action ordering, not estimates of discounted return. No
synthetic transitions enter the DQN replay buffer or its exploration schedule.
"""

import random
from collections import Counter
from collections import deque
from typing import Any

import numpy as np
import torch
import torch.nn.functional as F

from magic_pong.ai.agent_adapter import from_player1_dqn_action
from magic_pong.ai.agent_adapter import to_player1_dqn_observation
from magic_pong.ai.interface import ObservationProcessor
from magic_pong.ai.models.dqn_ai import ACTION_MAPPING
from magic_pong.ai.models.dqn_ai import DQNAgent
from magic_pong.ai.physics_teacher import PhysicsTeacher
from magic_pong.ai.physics_teacher import continuation_action
from magic_pong.core.entities import Action
from magic_pong.core.entities import Ball
from magic_pong.core.entities import Bonus
from magic_pong.core.entities import BonusType
from magic_pong.core.entities import Vector2D
from magic_pong.core.physics import PhysicsEngine
from magic_pong.utils.config import game_config
from magic_pong.utils.config import game_config_tmp


class SimulatorPretrainer:
    """Supervised warm start with interception, recovery and attacking cases."""

    scenario_modes = (
        "intercept",
        "recovery",
        "attack",
        "corner",
        "small_paddle",
        "bonus",
    )

    def __init__(
        self,
        field_width: float | None = None,
        field_height: float | None = None,
        paddle_width: float | None = None,
        paddle_height: float | None = None,
        ball_radius: float | None = None,
        y_only: bool = False,
        teacher_horizon: int = 90,
        teacher_action_steps: int = 6,
        rollout_steps: int = 120,
        learner_probability: float = 0.5,
        supervised_batch_size: int = 64,
        seed: int = 0,
    ) -> None:
        if rollout_steps < 1 or supervised_batch_size < 1:
            raise ValueError("rollout_steps and supervised_batch_size must be positive")
        if not 0 <= learner_probability <= 1:
            raise ValueError("learner_probability must be between 0 and 1")
        self.teacher = PhysicsTeacher(teacher_horizon, teacher_action_steps, y_only)
        self.field_width = field_width if field_width is not None else game_config.FIELD_WIDTH
        self.field_height = field_height if field_height is not None else game_config.FIELD_HEIGHT
        self.paddle_width = paddle_width if paddle_width is not None else game_config.PADDLE_WIDTH
        self.paddle_height = (
            paddle_height if paddle_height is not None else game_config.PADDLE_HEIGHT
        )
        self.ball_radius = ball_radius if ball_radius is not None else game_config.BALL_RADIUS
        if (
            min(
                self.field_width,
                self.field_height,
                self.paddle_width,
                self.paddle_height,
                self.ball_radius,
            )
            <= 0
        ):
            raise ValueError("scenario geometry must be positive")
        if self.field_height <= max(self.paddle_height, 2 * self.ball_radius):
            raise ValueError("scenario height must accommodate paddle and ball")
        self.observation_processor = ObservationProcessor(self.field_width, self.field_height)
        self.rollout_steps = rollout_steps
        self.learner_probability = learner_probability
        self.supervised_batch_size = supervised_batch_size
        self.y_only = y_only
        self.rng = random.Random(seed)
        self.physics_rng = random.Random(seed + 1)
        self.samples: deque[tuple[np.ndarray, int]] = deque(maxlen=20000)
        self.physics: PhysicsEngine | None = None
        self.scenario_index = 0
        self.current_mode = "intercept"
        self.current_player: int | None = None
        self.rollout_age = 0
        self.optimizer: torch.optim.Optimizer | None = None
        self.agent: DQNAgent | None = None

    def _new_scenario(self, player_id: int) -> None:
        self.current_mode = self.scenario_modes[self.scenario_index % len(self.scenario_modes)]
        self.scenario_index += 1
        self.current_player = player_id
        self.rollout_age = 0
        physics = PhysicsEngine(self.field_width, self.field_height)
        self.physics = physics
        paddle = getattr(physics, f"player{player_id}")
        opponent = getattr(physics, f"player{3 - player_id}")
        rng = self.rng
        direction = 1 if player_id == 1 else -1
        paddle.position.y = rng.uniform(0, paddle.max_y)
        opponent.position.y = rng.uniform(0, opponent.max_y)
        # Include forward opponents as well as defensive ones; the distance to
        # their paddle materially changes whether an attacking return is reachable.
        if rng.random() < 0.5:
            opponent.position.x = rng.uniform(opponent.min_x, opponent.max_x)
        if self.current_mode == "small_paddle":
            paddle.apply_size_effect(0.5, 10.0)
            opponent.apply_size_effect(0.5, 10.0)
        if self.current_mode in ("recovery", "bonus") and not self.y_only:
            paddle.position.x = rng.uniform(paddle.min_x, paddle.max_x)
        front = paddle.position.x + (paddle.width if player_id == 1 else 0)
        distance = rng.uniform(30, 100) if self.current_mode == "attack" else rng.uniform(100, 300)
        ball_x = np.clip(
            front + direction * distance,
            self.ball_radius,
            self.field_width - self.ball_radius,
        )
        ball_y = rng.uniform(self.ball_radius, self.field_height - self.ball_radius)
        speed = game_config.BALL_SPEED
        ball_vx = -direction * speed
        ball_vy = rng.uniform(-speed * 0.7, speed * 0.7)
        if self.current_mode == "attack":
            ball_y = paddle.position.y + paddle.height * rng.uniform(0.1, 0.9)
            ball_vy = 0.0
        elif self.current_mode in ("recovery", "bonus"):
            ball_vx = direction * speed
        elif self.current_mode == "corner":
            ball_y = rng.choice((self.ball_radius + 2, self.field_height - self.ball_radius - 2))
            paddle.position.y = 0 if ball_y < self.field_height / 2 else paddle.max_y
        # These components describe direction; adding vertical motion must not
        # raise total speed above the speed used by normal gameplay serves.
        velocity = Vector2D(ball_vx, ball_vy).normalize() * speed
        physics.ball = Ball(float(ball_x), float(ball_y), velocity.x, velocity.y)
        physics.ball.radius = self.ball_radius
        if self.current_mode == "bonus":
            bonus_x = paddle.position.x + direction * 45
            bonus_y = paddle.position.y + paddle.height / 2 + rng.choice((-40, 40))
            physics.bonuses = [
                Bonus(
                    max(0, min(self.field_width - game_config.BONUS_SIZE, bonus_x)),
                    max(0, min(self.field_height - game_config.BONUS_SIZE, bonus_y)),
                    rng.choice(list(BonusType)),
                )
            ]
        for current in (paddle, opponent):
            current.constrain_position()
            current.prev_position = current.position.copy()

    def _state(self, agent: DQNAgent, player_id: int) -> np.ndarray:
        assert self.physics is not None
        observation = self.observation_processor.process_game_state(
            self.physics.get_game_state(), player_id
        )
        return agent._observation_to_state(to_player1_dqn_observation(observation, player_id))

    def _fit(self, agent: DQNAgent) -> float:
        assert self.optimizer is not None
        batch = self.rng.sample(
            list(self.samples), min(len(self.samples), self.supervised_batch_size)
        )
        device = next(agent.q_network.parameters()).device
        states = torch.as_tensor(np.stack([sample[0] for sample in batch]), device=device)
        labels = torch.tensor([sample[1] for sample in batch], dtype=torch.long, device=device)
        logits = agent.q_network(states)
        # y_only labels retain the canonical three vertical action indices.
        loss = F.cross_entropy(logits, labels)
        self.optimizer.zero_grad()
        loss.backward()
        torch.nn.utils.clip_grad_norm_(agent.q_network.parameters(), 10.0)
        self.optimizer.step()
        return float(loss.item())

    def pretraining_step(
        self, agent: DQNAgent, player_id: int = 1, num_steps: int = 1000
    ) -> dict[str, Any]:
        if player_id not in (1, 2) or num_steps < 1:
            raise ValueError("player_id must be 1 or 2 and num_steps must be positive")
        if self.agent is not None and self.agent is not agent:
            raise ValueError("a SimulatorPretrainer instance belongs to one agent")
        self.agent = agent
        if self.optimizer is None:
            self.optimizer = torch.optim.Adam(agent.q_network.parameters(), lr=agent.lr)
        agent.set_training_mode(True)
        losses = []
        values = []
        counts: Counter[str] = Counter()
        learner_decisions = 0
        physics_steps = 0
        goals_for = 0
        goals_against = 0
        rng_before = random.getstate()
        random.setstate(self.physics_rng.getstate())
        try:
            with game_config_tmp(
                FIELD_WIDTH=self.field_width,
                FIELD_HEIGHT=self.field_height,
                PADDLE_WIDTH=self.paddle_width,
                PADDLE_HEIGHT=self.paddle_height,
                BALL_RADIUS=self.ball_radius,
            ):
                for step in range(num_steps):
                    if (
                        self.physics is None
                        or self.current_player != player_id
                        or self.rollout_age >= self.rollout_steps
                    ):
                        self._new_scenario(player_id)
                    assert self.physics is not None
                    counts[self.current_mode] += 1
                    with game_config_tmp(BONUSES_ENABLED=self.current_mode == "bonus"):
                        state = self._state(agent, player_id)
                        decision = self.teacher.label(self.physics, player_id)
                        self.samples.append((state.copy(), decision.action))
                        values.append(float(decision.values[decision.action]))
                        action_index = decision.action
                        if self.rng.random() < self.learner_probability:
                            learner_decisions += 1
                            device = next(agent.q_network.parameters()).device
                            with torch.no_grad():
                                logits = agent.q_network(torch.as_tensor(state, device=device))
                                action_index = max(
                                    self.teacher.actions,
                                    key=lambda index: float(logits[index]),
                                )
                        action = from_player1_dqn_action(ACTION_MAPPING[action_index], player_id)
                        if self.y_only:
                            action = Action(0.0, action.move_y)
                        dt = game_config.GAME_SPEED_MULTIPLIER / game_config.FPS
                        frames = min(
                            self.teacher.action_steps,
                            self.rollout_steps - self.rollout_age,
                        )
                        for _ in range(frames):
                            other_action = continuation_action(self.physics, 3 - player_id, dt)
                            actions = (
                                (action, other_action) if player_id == 1 else (other_action, action)
                            )
                            events = self.physics.update(dt, *actions)
                            self.rollout_age += 1
                            physics_steps += 1
                            if events["goals"]:
                                goals_for += sum(g["player"] == player_id for g in events["goals"])
                                goals_against += sum(
                                    g["player"] != player_id for g in events["goals"]
                                )
                                self.physics = None
                                break
                    if (step + 1) % self.supervised_batch_size == 0 or step == num_steps - 1:
                        losses.append(self._fit(agent))
        finally:
            self.physics_rng.setstate(random.getstate())
            random.setstate(rng_before)
            agent.update_target_network()
        return {
            "steps": num_steps,
            "average_reward": float(np.mean(values)),
            "total_reward": sum(values),
            "average_loss": float(np.mean(losses)),
            "rewards_history": values,
            "epsilon": agent.epsilon,
            "training_step": agent.training_step,
            "scenario_counts": dict(counts),
            "learner_decisions": learner_decisions,
            "physics_steps": physics_steps,
            "supervised_updates": len(losses),
            "goals_for": goals_for,
            "goals_against": goals_against,
        }

    def run_pretraining_phase(
        self,
        agent: DQNAgent,
        total_steps: int = 10000,
        steps_per_batch: int = 1000,
        player_id: int = 1,
        verbose: bool = True,
    ) -> dict[str, Any]:
        if total_steps < 1 or steps_per_batch < 1:
            raise ValueError("total_steps and steps_per_batch must be positive")
        batches = []
        rewards = []
        counts: Counter[str] = Counter()
        remaining = total_steps
        while remaining:
            result = self.pretraining_step(agent, player_id, min(remaining, steps_per_batch))
            batches.append(result)
            rewards.extend(result["rewards_history"])
            counts.update(result["scenario_counts"])
            remaining -= result["steps"]
            if verbose:
                print(
                    f"   Simulator pretraining {total_steps - remaining}/{total_steps}: "
                    f"imitation loss={result['average_loss']:.4f}, scenarios={result['scenario_counts']}"
                )
        return {
            "mode": "simulator",
            "total_steps": total_steps,
            "batches": len(batches),
            "batch_stats": batches,
            "all_rewards": rewards,
            "average_reward": float(np.mean(rewards)),
            "reward_std": float(np.std(rewards)),
            "average_loss": float(np.mean([batch["average_loss"] for batch in batches])),
            "final_epsilon": agent.epsilon,
            "final_training_step": agent.training_step,
            "scenario_counts": dict(counts),
            **{
                key: sum(batch[key] for batch in batches)
                for key in (
                    "learner_decisions",
                    "physics_steps",
                    "supervised_updates",
                    "goals_for",
                    "goals_against",
                )
            },
        }
