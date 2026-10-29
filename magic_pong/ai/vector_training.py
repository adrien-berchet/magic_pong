"""Vectorised Double-DQN training on the numba physics.

Rebuild of the training loop: ``n_envs`` matches advance in one compiled call
per decision, each decision holds an action for ``action_repeat`` frames, the
replay is a lock-step ring buffer of one million transitions with n-step
returns computed at sampling time, the loss is Huber on Double-DQN targets,
opponents mix built-ins, frozen self-play snapshots and the lookahead planner,
and evaluation runs 20+ seeds per cell on the same vectorised engine. Weights
are saved as ordinary :class:`DQNAgent` checkpoints.

Run ``python -m magic_pong.ai.vector_training --help``.
"""

from __future__ import annotations

import argparse
import copy
import json
import math
import random
import time
from dataclasses import asdict
from dataclasses import dataclass
from dataclasses import field
from pathlib import Path
from typing import Any

import numpy as np
import torch
import torch.nn.functional as F

from magic_pong.ai.competitive import wilson_lower_bound
from magic_pong.ai.models.dqn_ai import DQNAgent
from magic_pong.ai.models.simple_ai import BUILTIN_OPPONENTS
from magic_pong.ai.vector_env import EV_GOALS_AGAINST
from magic_pong.ai.vector_env import EV_GOALS_FOR
from magic_pong.ai.vector_env import EV_LEARNER_HITS
from magic_pong.ai.vector_env import VectorEnv
from magic_pong.ai.vector_env import parse_opponent_mix

DEFAULT_MIX: dict[str, float] = {
    "tracker": 1.0,
    "striker": 1.0,
    "follow_ball": 1.0,
    "aggressive": 1.0,
    "defensive": 1.0,
    "predictive": 1.0,
    "random": 0.5,
    "dummy": 0.25,
    "training_dummy": 0.25,
    "self": 2.0,
    "planner": 0.5,
}


@dataclass
class VectorTrainingConfig:
    total_frames: int = 20_000_000
    n_envs: int = 128
    action_repeat: int = 4
    gamma: float = 0.99
    n_step: int = 3
    learning_rate: float = 1e-4
    batch_size: int = 256
    replay_capacity: int = 1_000_000
    min_replay: int = 50_000
    replay_ratio: float = 8.0
    target_update_frames: int = 200_000
    epsilon_start: float = 1.0
    epsilon_end: float = 0.05
    epsilon_decay_frames: int = 4_000_000
    huber_delta: float = 1.0
    grad_clip: float = 10.0
    hidden_layers: tuple[int, ...] = (256, 128)
    score_context: bool = False
    geometry_context: bool = False
    terminal_on_goal: bool = True
    match_target: int = 11
    max_point_frames: int = 3600
    hit_reward: float = 0.0
    shaping_scale: float = 0.0
    opponent_mix: dict[str, float] = field(default_factory=lambda: dict(DEFAULT_MIX))
    snapshot_interval_frames: int = 500_000
    snapshot_pool: int = 8
    eval_interval_frames: int = 1_000_000
    eval_seeds: int = 20
    eval_max_frames: int = 18_000
    eval_opponents: tuple[str, ...] = BUILTIN_OPPONENTS
    checkpoint_interval_frames: int = 1_000_000
    seed: int = 0
    device: str = "auto"
    planner_horizon: int = 90
    planner_action_steps: int = 30
    initial_checkpoint: str | None = None
    dueling: bool = False
    warm_start_tau: float = 0.03
    phase1_frames: int = 0
    phase1_lr: float = 2e-4
    phase2_lr: float = 3e-5
    warmup_updates: int = 2_000
    soft_target_tau: float = 0.0

    def __post_init__(self) -> None:
        self.opponent_mix = parse_opponent_mix(self.opponent_mix)
        positives = (
            self.total_frames,
            self.n_envs,
            self.action_repeat,
            self.n_step,
            self.batch_size,
            self.replay_capacity,
            self.min_replay,
            self.target_update_frames,
            self.epsilon_decay_frames,
            self.match_target,
            self.max_point_frames,
            self.snapshot_interval_frames,
            self.eval_interval_frames,
            self.eval_seeds,
            self.eval_max_frames,
            self.checkpoint_interval_frames,
        )
        if any(value < 1 for value in positives):
            raise ValueError("counts, sizes and intervals must be positive")
        if not 0 < self.gamma <= 1 or self.learning_rate <= 0 or self.replay_ratio <= 0:
            raise ValueError("gamma must be in (0, 1]; learning_rate and replay_ratio positive")
        if not 0 <= self.epsilon_end <= self.epsilon_start <= 1:
            raise ValueError("epsilon must satisfy 0 <= end <= start <= 1")
        if self.batch_size > self.replay_capacity or self.min_replay > self.replay_capacity:
            raise ValueError("batch_size and min_replay must fit the replay capacity")
        if self.replay_capacity // self.n_envs < self.n_step + 1:
            raise ValueError("replay_capacity must hold at least n_step+1 decisions per env")
        if self.snapshot_pool < 1 or self.huber_delta <= 0 or self.grad_clip <= 0:
            raise ValueError("snapshot_pool, huber_delta and grad_clip must be positive")
        if self.shaping_scale < 0 or self.hit_reward < 0:
            raise ValueError("shaping_scale and hit_reward must be nonnegative")
        if self.phase1_frames < 0:
            raise ValueError("phase1_frames must be nonnegative")
        if self.warm_start_tau <= 0:
            raise ValueError("warm_start_tau must be positive")
        if self.phase1_lr <= 0 or self.phase2_lr <= 0:
            raise ValueError("phase1_lr and phase2_lr must be positive")
        if not 0.0 <= self.soft_target_tau <= 1.0:
            raise ValueError("soft_target_tau must be in [0, 1]")
        unknown = set(self.eval_opponents) - set(BUILTIN_OPPONENTS)
        if unknown or not self.eval_opponents:
            raise ValueError(f"eval_opponents must be built-ins; unknown {sorted(unknown)}")
        self.hidden_layers = tuple(int(w) for w in self.hidden_layers)
        self.eval_opponents = tuple(self.eval_opponents)


class ReplayBuffer:
    """Lock-step ring buffer: every environment writes one transition per decision."""

    def __init__(self, capacity: int, n_envs: int, obs_dim: int) -> None:
        self.rows = max(1, capacity // n_envs)
        self.n_envs = n_envs
        self.obs = np.zeros((self.rows, n_envs, obs_dim), dtype=np.float32)
        self.next_obs = np.zeros((self.rows, n_envs, obs_dim), dtype=np.float32)
        self.actions = np.zeros((self.rows, n_envs), dtype=np.int64)
        self.rewards = np.zeros((self.rows, n_envs), dtype=np.float32)
        self.dones = np.zeros((self.rows, n_envs), dtype=np.bool_)
        self.cuts = np.zeros((self.rows, n_envs), dtype=np.bool_)
        self.written = 0  # total decisions written (monotonic)

    def __len__(self) -> int:
        return min(self.written, self.rows) * self.n_envs

    def add(
        self,
        obs: np.ndarray,
        actions: np.ndarray,
        rewards: np.ndarray,
        next_obs: np.ndarray,
        dones: np.ndarray,
        cuts: np.ndarray,
    ) -> None:
        row = self.written % self.rows
        self.obs[row] = obs
        self.next_obs[row] = next_obs
        self.actions[row] = actions
        self.rewards[row] = rewards
        self.dones[row] = dones
        self.cuts[row] = cuts
        self.written += 1

    def sample(
        self, batch_size: int, n_step: int, gamma: float, rng: np.random.Generator
    ) -> dict[str, np.ndarray]:
        """Sample transitions with n-step returns truncated at terminals and cuts."""
        available = min(self.written, self.rows)
        usable = available - n_step + 1
        if usable < 1:
            raise ValueError("not enough decisions stored for the requested n-step")
        oldest = self.written - available
        offsets = rng.integers(usable, size=batch_size)
        envs = rng.integers(self.n_envs, size=batch_size)
        start = oldest + offsets
        returns = np.zeros(batch_size, dtype=np.float64)
        discount = np.ones(batch_size, dtype=np.float64)
        alive = np.ones(batch_size, dtype=np.bool_)
        terminal = np.zeros(batch_size, dtype=np.bool_)
        length = np.zeros(batch_size, dtype=np.int64)
        for k in range(n_step):
            rows = (start + k) % self.rows
            reward = self.rewards[rows, envs]
            returns += np.where(alive, discount * reward, 0.0)
            length += alive
            done = self.dones[rows, envs]
            cut = self.cuts[rows, envs]
            terminal |= alive & done
            alive &= ~(done | cut)
            discount *= gamma
        last = (start + length - 1) % self.rows
        first = start % self.rows
        return {
            "obs": self.obs[first, envs],
            "actions": self.actions[first, envs],
            "returns": returns.astype(np.float32),
            "next_obs": self.next_obs[last, envs],
            "terminal": terminal,
            "discount": (gamma**length).astype(np.float32),
        }


def _select_device(name: str) -> torch.device:
    if name == "auto":
        return torch.device("cuda" if torch.cuda.is_available() else "cpu")
    return torch.device(name)


def greedy_actions(network: torch.nn.Module, obs: np.ndarray, device: torch.device) -> np.ndarray:
    expected_size: int = obs.shape[1]
    layout = getattr(network, "layout", None)
    if layout is not None and getattr(layout, "state_size", None) is not None:
        expected_size = int(layout.state_size)
    elif hasattr(network, "fc_layers"):
        fc_layers = network.fc_layers
        if len(fc_layers) > 0:
            in_f = getattr(fc_layers[0], "in_features", None)
            if in_f is not None:
                expected_size = int(in_f)
    input_obs = obs[:, :expected_size] if obs.shape[1] > expected_size else obs
    with torch.no_grad():
        q = network(torch.as_tensor(input_obs, device=device))
    return q.argmax(dim=1).cpu().numpy()


@dataclass
class EvaluationCell:
    opponent: str
    side: int
    bonuses: bool
    wins: int
    losses: int
    draws: int

    @property
    def matches(self) -> int:
        return self.wins + self.losses + self.draws


def evaluate_network(
    network: torch.nn.Module,
    device: torch.device,
    *,
    opponents: tuple[str, ...] = BUILTIN_OPPONENTS,
    seeds: int = 20,
    sides: tuple[int, ...] = (1, 2),
    bonus_modes: tuple[bool, ...] = (False, True),
    match_target: int = 11,
    max_frames: int = 18_000,
    score_context: bool = False,
    geometry_context: bool = False,
    seed: int = 12345,
) -> dict[str, Any]:
    """Play one match per (opponent, side, bonus mode, seed) with the greedy policy.

    Draws are matches that reach ``max_frames`` without a winner and count as
    non-wins. The ``rank`` tuple follows the competitive harness: weakest cell's
    Wilson lower bound first, then the mean bound, mean goal difference and mean
    win rate.
    """
    cells = [(o, s, b) for o in opponents for s in sides for b in bonus_modes]
    n = len(cells) * seeds
    env = VectorEnv(
        n,
        seed=seed,
        repeat=1,
        match_target=match_target,
        max_point_frames=max_frames + 1,
        terminal_on_goal=False,
        score_context=score_context,
        geometry_context=geometry_context,
        opponent_mix={"dummy": 1.0},
        auto_reset=False,
    )
    for index, (opponent, side, bonuses) in enumerate(cells * seeds):
        env.configure(index, side=side, bonuses_enabled=bonuses, opponent=opponent)
    env.reset()
    for index, (opponent, side, bonuses) in enumerate(cells * seeds):
        env.configure(index, side=side, bonuses_enabled=bonuses, opponent=opponent)
    env.active[:] = True
    was_training = network.training
    network.eval()
    won = np.zeros(n, dtype=np.bool_)
    lost = np.zeros(n, dtype=np.bool_)
    goals_for = np.zeros(n, dtype=np.int64)
    goals_against = np.zeros(n, dtype=np.int64)
    frames = np.zeros(n, dtype=np.int64)
    try:
        while env.active.any():
            actions = greedy_actions(network, env.obs, device)
            result = env.step(actions)
            goals_for += result.events[:, EV_GOALS_FOR]
            goals_against += result.events[:, EV_GOALS_AGAINST]
            frames += result.events[:, 7]
            for record in result.finished:
                won[record["env"]] = record["won"]
                lost[record["env"]] = not record["won"]
            env.active &= frames < max_frames
    finally:
        network.train(was_training)
    summary: list[EvaluationCell] = []
    for c, (opponent, side, bonuses) in enumerate(cells):
        indices = [c + k * len(cells) for k in range(seeds)]
        w = int(won[indices].sum())
        lo = int(lost[indices].sum())
        summary.append(EvaluationCell(opponent, side, bonuses, w, lo, seeds - w - lo))
    bounds = [wilson_lower_bound(cell.wins, cell.matches) for cell in summary]
    goal_diff = float((goals_for - goals_against).mean())
    win_rate = float(won.mean())
    return {
        "cells": [asdict(cell) | {"matches": cell.matches} for cell in summary],
        "wins": int(won.sum()),
        "losses": int(lost.sum()),
        "draws": int(n - won.sum() - lost.sum()),
        "matches": n,
        "goal_diff": goal_diff,
        "win_rate": win_rate,
        "rank": (min(bounds), float(np.mean(bounds)), goal_diff, win_rate),
        "by_opponent": {
            opponent: {
                "wins": int(sum(cell.wins for cell in summary if cell.opponent == opponent)),
                "losses": int(sum(cell.losses for cell in summary if cell.opponent == opponent)),
                "draws": int(sum(cell.draws for cell in summary if cell.opponent == opponent)),
            }
            for opponent in opponents
        },
    }


class VectorDQNTrainer:
    """Own the learner, replay, opponents, evaluation and checkpoints of one run."""

    def __init__(self, config: VectorTrainingConfig, output_dir: str | Path) -> None:
        self.config = config
        self.output = Path(output_dir)
        self.output.mkdir(parents=True, exist_ok=True)
        self.device = _select_device(config.device)
        random.seed(config.seed)
        np.random.seed(config.seed % (2**32))
        torch.manual_seed(config.seed)
        self.rng = np.random.default_rng(config.seed + 1)
        from magic_pong.ai.models import dqn_ai

        previous_device = dqn_ai.device
        dqn_ai.device = self.device
        try:
            self.agent = DQNAgent(
                hidden_layers=list(config.hidden_layers),
                gamma=config.gamma,
                lr=config.learning_rate,
                score_context=config.score_context,
                geometry_context=config.geometry_context,
                terminal_on_goal=config.terminal_on_goal,
                dueling=config.dueling,
            )
            if config.initial_checkpoint:
                self.agent.load_model(config.initial_checkpoint)
                # Safeguard 2: restore correct learning rate and terminal_on_goal from config
                self.agent.lr = config.learning_rate
                self.agent.terminal_on_goal = config.terminal_on_goal
                self.agent.q_network.to(self.device)
                self.agent.target_network.to(self.device)
            if config.score_context and not self.agent.score_context:
                self.agent.enable_score_context()
            if config.geometry_context and not self.agent.geometry_context:
                self.agent.enable_geometry_context()
            # Safeguard 1: Convert to dueling before building optimizer and target
            if config.dueling and not self.agent.q_network.dueling:
                self.agent.enable_dueling(
                    tau=config.warm_start_tau, warm_start=True, q_semantics="logits"
                )
                self.agent.q_network.to(self.device)
                self.agent.target_network.to(self.device)
            if config.phase1_frames > 0:
                if config.dueling:
                    self.agent.freeze_for_phase1()
                self.agent.lr = config.phase1_lr
        finally:
            dqn_ai.device = previous_device
        self.network = self.agent.q_network
        self.target = self.agent.target_network
        self.target.load_state_dict(self.network.state_dict())
        self.target.eval()
        if config.phase1_frames > 0:
            self.in_phase1 = True
            self.optimizer = self.agent.optimizer
            for param_group in self.optimizer.param_groups:
                param_group["lr"] = config.phase1_lr
        else:
            self.in_phase1 = False
            self.optimizer = torch.optim.Adam(
                [p for p in self.network.parameters() if p.requires_grad],
                lr=config.learning_rate,
            )
            self.agent.optimizer = self.optimizer
        self.env = VectorEnv(
            config.n_envs,
            seed=config.seed + 2,
            repeat=config.action_repeat,
            match_target=config.match_target,
            max_point_frames=config.max_point_frames,
            terminal_on_goal=config.terminal_on_goal,
            score_context=config.score_context,
            geometry_context=config.geometry_context,
            opponent_mix=config.opponent_mix,
            hit_reward=config.hit_reward,
            planner_horizon=config.planner_horizon,
            planner_action_steps=config.planner_action_steps,
        )
        # Safeguard 3: Observation dimension assertion
        expected_obs_dim = 46 if config.geometry_context else 30
        assert self.env.obs_dim == expected_obs_dim, (
            f"Observation dimension mismatch: env has {self.env.obs_dim}, expected {expected_obs_dim}"
        )
        self.replay = ReplayBuffer(config.replay_capacity, config.n_envs, self.env.obs_dim)
        self.snapshots: list[torch.nn.Module] = []
        self.frames = 0
        self.updates = 0
        self.decisions = 0
        self.history: list[dict[str, Any]] = []
        self.best_rank: tuple[float, ...] | None = None
        self.best_frames = 0
        self.losses: list[float] = []
        self.match_log: list[dict[str, Any]] = []
        self._update_budget = 0.0
        self._next_target = config.target_update_frames
        self._next_snapshot = config.snapshot_interval_frames
        self._next_eval = config.eval_interval_frames
        self._next_checkpoint = config.checkpoint_interval_frames
        self.add_snapshot()

    # --- schedules ----------------------------------------------------------------
    def epsilon(self) -> float:
        c = self.config
        progress = min(1.0, self.frames / c.epsilon_decay_frames)
        return c.epsilon_start + (c.epsilon_end - c.epsilon_start) * progress

    def add_snapshot(self) -> None:
        frozen = copy.deepcopy(self.network).eval()
        frozen.requires_grad_(False)
        self.snapshots.append(frozen)
        if len(self.snapshots) > self.config.snapshot_pool + 1:
            # Keep the initial network as a permanent historical opponent.
            del self.snapshots[1]
        self.env.snapshot_count = len(self.snapshots)

    # --- acting -------------------------------------------------------------------
    def _opponent_actions(self) -> np.ndarray:
        actions = np.zeros(self.env.n, dtype=np.int64)
        mask = self.env.network_mask & self.env.active
        if not mask.any():
            return actions
        obs = self.env.opponent_observations()
        for index, snapshot in enumerate(self.snapshots):
            chosen = mask & (self.env.snapshot_index == index)
            if chosen.any():
                actions[chosen] = greedy_actions(snapshot, obs[chosen], self.device)
        return actions

    def _learner_actions(self, epsilon: float) -> np.ndarray:
        actions = greedy_actions(self.network, self.env.obs, self.device)
        explore = self.rng.random(self.env.n) < epsilon
        actions[explore] = self.rng.integers(9, size=int(explore.sum()))
        return actions

    # --- learning -----------------------------------------------------------------
    def _update(self) -> float:
        c = self.config
        # Linear warmup in Phase 2
        if (
            hasattr(self, "_warmup_start_updates")
            and c.warmup_updates > 0
            and not getattr(self, "in_phase1", False)
        ):
            phase2_elapsed = self.updates - self._warmup_start_updates
            if phase2_elapsed < c.warmup_updates:
                factor = 0.1 + 0.9 * (phase2_elapsed / c.warmup_updates)
                current_lr = c.phase2_lr * factor
                for param_group in self.optimizer.param_groups:
                    param_group["lr"] = current_lr

        batch = self.replay.sample(c.batch_size, c.n_step, c.gamma, self.rng)
        obs = torch.as_tensor(batch["obs"], device=self.device)
        next_obs = torch.as_tensor(batch["next_obs"], device=self.device)
        actions = torch.as_tensor(batch["actions"], device=self.device)
        returns = torch.as_tensor(batch["returns"], device=self.device)
        terminal = torch.as_tensor(batch["terminal"], device=self.device)
        discount = torch.as_tensor(batch["discount"], device=self.device)
        with torch.no_grad():
            next_actions = self.network(next_obs).argmax(dim=1, keepdim=True)
            next_q = self.target(next_obs).gather(1, next_actions).squeeze(1)
            target = returns + discount * next_q * (~terminal)
        q = self.network(obs).gather(1, actions.unsqueeze(1)).squeeze(1)
        loss = F.smooth_l1_loss(q, target, beta=c.huber_delta)
        self.optimizer.zero_grad()
        loss.backward()
        torch.nn.utils.clip_grad_norm_(self.network.parameters(), c.grad_clip)
        self.optimizer.step()
        self.updates += 1

        if c.soft_target_tau > 0.0:
            with torch.no_grad():
                for p_target, p_online in zip(self.target.parameters(), self.network.parameters()):
                    p_target.data.mul_(1.0 - c.soft_target_tau)
                    p_target.data.add_(p_online.data * c.soft_target_tau)

        return float(loss.item())

    def train_step(self) -> dict[str, float]:
        """One decision for every environment plus the due gradient updates."""
        c = self.config
        # Check if Phase 1 calibration has finished and transition to Phase 2 fine-tuning
        if getattr(self, "in_phase1", False) and self.frames >= c.phase1_frames:
            self.in_phase1 = False
            if c.dueling:
                self.agent.unfreeze_for_phase2(lr=c.phase2_lr)
                self.optimizer = self.agent.optimizer
            else:
                self.optimizer = torch.optim.Adam(self.network.parameters(), lr=c.phase2_lr)
                self.agent.optimizer = self.optimizer
            self.target.load_state_dict(self.network.state_dict())
            self._warmup_start_updates = self.updates

        epsilon = self.epsilon()
        obs = self.env.obs.copy()
        actions = self._learner_actions(epsilon)
        opponent_actions = self._opponent_actions()
        shaping_before = self.env.potentials() if c.shaping_scale > 0 else None
        result = self.env.step(actions, opponent_actions)
        rewards = result.rewards.astype(np.float64)
        if shaping_before is not None:
            after = np.zeros(self.env.n, dtype=np.float64)
            from magic_pong.ai.vector_env import potential_batch

            # Potential of the learning next state; a terminal state has potential 0.
            potential_batch(self.env.states, self.env.params, self.env.sides, after)
            after = np.where(result.dones | result.truncated, 0.0, after)
            rewards += c.shaping_scale * (c.gamma * after - shaping_before)
        self.replay.add(
            obs,
            actions,
            rewards.astype(np.float32),
            result.next_obs,
            result.dones,
            result.truncated,
        )
        self.frames += int(result.events[:, 7].sum())
        self.decisions += 1
        for record in result.finished:
            record["frames"] = self.frames
            self.match_log.append(record)
        stats = {
            "epsilon": epsilon,
            "goals_for": float(result.events[:, EV_GOALS_FOR].sum()),
            "goals_against": float(result.events[:, EV_GOALS_AGAINST].sum()),
            "hits": float(result.events[:, EV_LEARNER_HITS].sum()),
            "loss": float("nan"),
        }
        if len(self.replay) >= c.min_replay:
            self._update_budget += c.replay_ratio * self.env.n / c.batch_size
            losses = []
            while self._update_budget >= 1.0:
                losses.append(self._update())
                self._update_budget -= 1.0
            if losses:
                stats["loss"] = float(np.mean(losses))
                self.losses.extend(losses)
        if c.soft_target_tau <= 0.0 and self.frames >= self._next_target:
            self.target.load_state_dict(self.network.state_dict())
            self._next_target += c.target_update_frames
        if self.frames >= self._next_snapshot:
            self.add_snapshot()
            self._next_snapshot += c.snapshot_interval_frames
        return stats

    # --- evaluation and checkpoints -----------------------------------------------
    def evaluate(self) -> dict[str, Any]:
        c = self.config
        report = evaluate_network(
            self.network,
            self.device,
            opponents=c.eval_opponents,
            seeds=c.eval_seeds,
            match_target=c.match_target,
            max_frames=c.eval_max_frames,
            score_context=c.score_context,
            geometry_context=c.geometry_context,
            seed=c.seed + 9_999,
        )
        report["frames"] = self.frames
        report["updates"] = self.updates
        rank = tuple(report["rank"])
        selected = self.best_rank is None or rank > self.best_rank
        if selected:
            self.best_rank = rank
            self.best_frames = self.frames
            self.save("best.pth")
        report["selected"] = selected
        self.history.append(report)
        (self.output / f"evaluation_{self.frames:010d}.json").write_text(
            json.dumps(report, indent=2) + "\n"
        )
        return report

    def save(self, name: str) -> Path:
        self.agent.epsilon = self.epsilon()
        self.agent.training_step = self.updates
        self.agent.step_count = self.frames
        self.agent.loss_history = self.losses[-1000:]
        path = self.output / name
        self.agent.save_model(str(path))
        return path

    def summary(self, seconds: float) -> dict[str, Any]:
        recent = self.match_log[-500:]
        return {
            "config": asdict(self.config),
            "frames": self.frames,
            "decisions": self.decisions,
            "updates": self.updates,
            "replay_size": len(self.replay),
            "snapshots": len(self.snapshots),
            "seconds": seconds,
            "frames_per_second": self.frames / seconds if seconds > 0 else None,
            "recent_match_win_rate": (
                float(np.mean([r["won"] for r in recent])) if recent else None
            ),
            "matches_played": len(self.match_log),
            "best_frames": self.best_frames,
            "best_rank": self.best_rank,
            "evaluations": [
                {k: v for k, v in report.items() if k != "cells"} for report in self.history
            ],
        }

    def run(self, *, verbose: bool = True, log_interval_frames: int = 250_000) -> dict[str, Any]:
        c = self.config
        start = time.perf_counter()
        next_log = log_interval_frames
        window: list[dict[str, float]] = []
        while self.frames < c.total_frames:
            window.append(self.train_step())
            if verbose and self.frames >= next_log:
                losses = [w["loss"] for w in window if not math.isnan(w["loss"])]
                goals_for = sum(w["goals_for"] for w in window)
                goals_against = sum(w["goals_against"] for w in window)
                elapsed = time.perf_counter() - start
                recent = self.match_log[-200:]
                print(
                    f"frames={self.frames:,} updates={self.updates:,} eps={self.epsilon():.3f} "
                    f"loss={np.mean(losses) if losses else float('nan'):.4f} "
                    f"goals={goals_for:.0f}/{goals_against:.0f} "
                    f"match_win={np.mean([r['won'] for r in recent]) if recent else float('nan'):.2f} "
                    f"fps={self.frames / elapsed:,.0f}",
                    flush=True,
                )
                window = []
                while next_log <= self.frames:
                    next_log += log_interval_frames
            if self.frames >= self._next_eval:
                report = self.evaluate()
                if verbose:
                    print(
                        f"evaluation frames={self.frames:,} wins={report['wins']}/{report['matches']} "
                        f"losses={report['losses']} draws={report['draws']} "
                        f"goal_diff={report['goal_diff']:+.2f} selected={report['selected']}",
                        flush=True,
                    )
                while self._next_eval <= self.frames:
                    self._next_eval += c.eval_interval_frames
            if self.frames >= self._next_checkpoint:
                self.save("latest.pth")
                (self.output / "training.json").write_text(
                    json.dumps(self.summary(time.perf_counter() - start), indent=2) + "\n"
                )
                while self._next_checkpoint <= self.frames:
                    self._next_checkpoint += c.checkpoint_interval_frames
        report = self.evaluate()
        self.save("latest.pth")
        self.save("final.pth")
        summary = self.summary(time.perf_counter() - start)
        (self.output / "training.json").write_text(json.dumps(summary, indent=2) + "\n")
        if verbose:
            print(
                f"final evaluation wins={report['wins']}/{report['matches']} losses={report['losses']} "
                f"draws={report['draws']}; best at {self.best_frames:,} frames",
                flush=True,
            )
        return summary


def _parse_layers(value: str) -> tuple[int, ...]:
    layers = tuple(int(part) for part in value.split(",") if part.strip())
    if not layers or any(width <= 0 for width in layers):
        raise argparse.ArgumentTypeError("hidden layers must be positive integers")
    return layers


def build_parser() -> argparse.ArgumentParser:
    parser = argparse.ArgumentParser(
        description="Vectorised Double-DQN training on the numba physics"
    )
    defaults = VectorTrainingConfig()
    parser.add_argument("--output", required=True)
    parser.add_argument("--total_frames", type=int, default=defaults.total_frames)
    parser.add_argument("--n_envs", type=int, default=defaults.n_envs)
    parser.add_argument("--action_repeat", type=int, default=defaults.action_repeat)
    parser.add_argument("--gamma", type=float, default=defaults.gamma)
    parser.add_argument("--n_step", type=int, default=defaults.n_step)
    parser.add_argument("--learning_rate", type=float, default=defaults.learning_rate)
    parser.add_argument("--batch_size", type=int, default=defaults.batch_size)
    parser.add_argument("--replay_capacity", type=int, default=defaults.replay_capacity)
    parser.add_argument("--min_replay", type=int, default=defaults.min_replay)
    parser.add_argument("--replay_ratio", type=float, default=defaults.replay_ratio)
    parser.add_argument("--target_update_frames", type=int, default=defaults.target_update_frames)
    parser.add_argument("--epsilon_start", type=float, default=defaults.epsilon_start)
    parser.add_argument("--epsilon_end", type=float, default=defaults.epsilon_end)
    parser.add_argument("--epsilon_decay_frames", type=int, default=defaults.epsilon_decay_frames)
    parser.add_argument("--hidden_layers", type=_parse_layers, default=defaults.hidden_layers)
    parser.add_argument("--score_context", action="store_true")
    parser.add_argument("--geometry_context", action="store_true")
    parser.add_argument("--full_match_returns", action="store_true", help="bootstrap through goals")
    parser.add_argument("--match_target", type=int, default=defaults.match_target)
    parser.add_argument("--max_point_frames", type=int, default=defaults.max_point_frames)
    parser.add_argument("--hit_reward", type=float, default=defaults.hit_reward)
    parser.add_argument("--shaping_scale", type=float, default=defaults.shaping_scale)
    parser.add_argument(
        "--opponent_mix",
        default=",".join(f"{k}:{v:g}" for k, v in DEFAULT_MIX.items()),
        help="name:weight list over built-ins, 'self' and 'planner'",
    )
    parser.add_argument(
        "--snapshot_interval_frames", type=int, default=defaults.snapshot_interval_frames
    )
    parser.add_argument("--snapshot_pool", type=int, default=defaults.snapshot_pool)
    parser.add_argument("--eval_interval_frames", type=int, default=defaults.eval_interval_frames)
    parser.add_argument("--eval_seeds", type=int, default=defaults.eval_seeds)
    parser.add_argument("--eval_max_frames", type=int, default=defaults.eval_max_frames)
    parser.add_argument(
        "--checkpoint_interval_frames", type=int, default=defaults.checkpoint_interval_frames
    )
    parser.add_argument("--seed", type=int, default=defaults.seed)
    parser.add_argument("--device", default=defaults.device)
    parser.add_argument("--planner_horizon", type=int, default=defaults.planner_horizon)
    parser.add_argument("--planner_action_steps", type=int, default=defaults.planner_action_steps)
    parser.add_argument("--initial_checkpoint")
    parser.add_argument("--dueling", action="store_true")
    parser.add_argument("--warm_start_tau", type=float, default=defaults.warm_start_tau)
    parser.add_argument("--phase1_frames", type=int, default=defaults.phase1_frames)
    parser.add_argument("--phase1_lr", type=float, default=defaults.phase1_lr)
    parser.add_argument("--phase2_lr", type=float, default=defaults.phase2_lr)
    parser.add_argument("--warmup_updates", type=int, default=defaults.warmup_updates)
    parser.add_argument("--soft_target_tau", type=float, default=defaults.soft_target_tau)
    parser.add_argument("--log_interval_frames", type=int, default=250_000)
    parser.add_argument(
        "--final_panel", action="store_true", help="run the exact competitive panel on best.pth"
    )
    return parser


def config_from_args(args: argparse.Namespace) -> VectorTrainingConfig:
    return VectorTrainingConfig(
        total_frames=args.total_frames,
        n_envs=args.n_envs,
        action_repeat=args.action_repeat,
        gamma=args.gamma,
        n_step=args.n_step,
        learning_rate=args.learning_rate,
        batch_size=args.batch_size,
        replay_capacity=args.replay_capacity,
        min_replay=args.min_replay,
        replay_ratio=args.replay_ratio,
        target_update_frames=args.target_update_frames,
        epsilon_start=args.epsilon_start,
        epsilon_end=args.epsilon_end,
        epsilon_decay_frames=args.epsilon_decay_frames,
        hidden_layers=args.hidden_layers,
        score_context=args.score_context,
        geometry_context=args.geometry_context,
        terminal_on_goal=not args.full_match_returns,
        match_target=args.match_target,
        max_point_frames=args.max_point_frames,
        hit_reward=args.hit_reward,
        shaping_scale=args.shaping_scale,
        opponent_mix=parse_opponent_mix(args.opponent_mix),
        snapshot_interval_frames=args.snapshot_interval_frames,
        snapshot_pool=args.snapshot_pool,
        eval_interval_frames=args.eval_interval_frames,
        eval_seeds=args.eval_seeds,
        eval_max_frames=args.eval_max_frames,
        checkpoint_interval_frames=args.checkpoint_interval_frames,
        seed=args.seed,
        device=args.device,
        planner_horizon=args.planner_horizon,
        planner_action_steps=args.planner_action_steps,
        initial_checkpoint=args.initial_checkpoint,
        dueling=args.dueling,
        warm_start_tau=args.warm_start_tau,
        phase1_frames=args.phase1_frames,
        phase1_lr=args.phase1_lr,
        phase2_lr=args.phase2_lr,
        warmup_updates=args.warmup_updates,
        soft_target_tau=args.soft_target_tau,
    )


def main(argv: list[str] | None = None) -> int:
    args = build_parser().parse_args(argv)
    config = config_from_args(args)
    trainer = VectorDQNTrainer(config, args.output)
    summary = trainer.run(log_interval_frames=args.log_interval_frames)
    print(
        f"trained {summary['frames']:,} frames in {summary['seconds']:.0f}s; output {args.output}"
    )
    if args.final_panel:
        from magic_pong.ai.competitive import CompetitiveConfig
        from magic_pong.ai.competitive import competitive_result_to_dict
        from magic_pong.ai.competitive import evaluate_competitive_checkpoint

        result = evaluate_competitive_checkpoint(trainer.output / "best.pth", CompetitiveConfig())
        wins = sum(m.wins for m in result.matchups)
        total = sum(m.matches for m in result.matchups)
        (trainer.output / "best.panel.json").write_text(
            json.dumps(competitive_result_to_dict(result), indent=2) + "\n"
        )
        print(f"exact competitive panel for best.pth: {wins}/{total} wins")
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
