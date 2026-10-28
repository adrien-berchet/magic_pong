"""Distil the lookahead planner into the DQN network with DAgger on real matches.

The learner plays complete matches against built-in opponents (and optionally the
planner itself). Every visited state is labelled by the planner's rollout
evaluation; the executed action follows the planner with probability ``beta``
and the network otherwise, so labels cover the states the network actually
reaches. The network outputs are trained with a soft cross-entropy toward
``softmax(rollout values / temperature)`` mixed with a small one-hot share of the
planner's actual choice: near-tied actions leave the network almost free (the
one-hot share preserves the planner's tie-break), decisive states such as a save
or a scoring return become sharp targets. No synthetic transitions enter the
replay buffer.
"""

from __future__ import annotations

import argparse
import json
import random
import time
from collections import Counter
from collections import deque
from pathlib import Path
from typing import Any

import numpy as np
import torch
import torch.nn.functional as F

from magic_pong.ai.agent_adapter import from_player1_dqn_action
from magic_pong.ai.agent_adapter import to_player1_dqn_observation
from magic_pong.ai.evaluation import preserve_global_rng_state
from magic_pong.ai.interface import AIPlayer
from magic_pong.ai.models.dqn_ai import ACTION_MAPPING
from magic_pong.ai.models.dqn_ai import DQNAgent
from magic_pong.ai.models.planner_ai import PlannerAI
from magic_pong.ai.models.planner_ai import make_teacher
from magic_pong.ai.models.simple_ai import BUILTIN_OPPONENTS
from magic_pong.ai.models.simple_ai import create_ai
from magic_pong.core.entities import Action
from magic_pong.core.game_engine import TrainingManager
from magic_pong.core.physics import PhysicsEngine
from magic_pong.utils.config import game_config_tmp

DEFAULT_DISTILL_OPPONENTS = (
    "tracker",
    "striker",
    "follow_ball",
    "aggressive",
    "defensive",
    "predictive",
    "planner",
)


class _DaggerPlayer(AIPlayer):
    """Mixture of planner and network actions that records planner labels."""

    uses_player1_dqn_frame = False

    def __init__(self, distiller: PlannerDistiller, beta: float) -> None:
        super().__init__(name="DAgger learner")
        self.distiller = distiller
        self.beta = beta
        self.physics: PhysicsEngine | None = None
        self.player_id = 1
        self.agreements = 0
        self.decisions = 0

    def bind_physics(self, physics: PhysicsEngine, player_id: int) -> None:
        self.physics = physics
        self.player_id = player_id

    def get_action(self, observation: dict[str, Any] | None) -> Action:
        if observation is None or self.physics is None:
            return Action(0.0, 0.0)
        distiller = self.distiller
        agent = distiller.agent
        assert agent is not None
        state = agent._observation_to_state(to_player1_dqn_observation(observation, self.player_id))
        decision = distiller.teacher.label(self.physics, self.player_id)
        distiller.samples.append((state.copy(), np.asarray(decision.values, dtype=np.float32)))
        network_action = distiller.network_action(state)
        self.decisions += 1
        self.agreements += network_action == decision.action
        index = decision.action if distiller.rng.random() < self.beta else network_action
        return from_player1_dqn_action(ACTION_MAPPING[index], self.player_id)

    def on_step(
        self,
        observation: dict[str, Any],
        action: Action,
        reward: float,
        done: bool,
        info: dict[str, Any],
    ) -> None:
        self.current_episode_reward += reward


class PlannerDistiller:
    """Supervised imitation of the planner on learner-visited match states."""

    def __init__(
        self,
        *,
        opponents: tuple[str, ...] = DEFAULT_DISTILL_OPPONENTS,
        teacher_horizon: int = 90,
        teacher_action_steps: int = 30,
        backend: str = "auto",
        beta_start: float = 1.0,
        beta_end: float = 0.0,
        match_steps: int = 3000,
        match_score: int = 3,
        buffer_size: int = 500_000,
        batch_size: int = 256,
        learning_rate: float = 1e-3,
        temperature: float = 0.03,
        tie_break_weight: float = 0.1,
        epochs_per_match: float = 0.25,
        final_epochs: int = 10,
        seed: int = 0,
        field_width: float | None = None,
        field_height: float | None = None,
        paddle_width: float | None = None,
        paddle_height: float | None = None,
        ball_radius: float | None = None,
    ) -> None:
        unknown = set(opponents) - set(BUILTIN_OPPONENTS) - {"planner"}
        if not opponents or unknown:
            raise ValueError(f"unknown distillation opponents: {sorted(unknown) or 'none given'}")
        if not 0 <= beta_end <= beta_start <= 1:
            raise ValueError("beta must satisfy 0 <= beta_end <= beta_start <= 1")
        if min(match_steps, match_score, buffer_size, batch_size, final_epochs) < 1:
            raise ValueError("match_steps, match_score, buffer_size, batch_size and epochs > 0")
        if learning_rate <= 0 or temperature <= 0 or epochs_per_match < 0:
            raise ValueError("learning_rate and temperature must be positive, epochs >= 0")
        if not 0 <= tie_break_weight <= 1:
            raise ValueError("tie_break_weight must be between 0 and 1")
        self.opponents = tuple(opponents)
        self.teacher = make_teacher(teacher_horizon, teacher_action_steps, backend=backend)
        self.teacher_horizon = teacher_horizon
        self.teacher_action_steps = teacher_action_steps
        self.backend = backend
        self.beta_start = beta_start
        self.beta_end = beta_end
        self.match_steps = match_steps
        self.match_score = match_score
        self.batch_size = batch_size
        self.learning_rate = learning_rate
        self.temperature = temperature
        self.tie_break_weight = tie_break_weight
        self.epochs_per_match = epochs_per_match
        self.final_epochs = final_epochs
        self.rng = random.Random(seed)
        self.seed = seed
        # Each sample keeps the full rollout value vector so the fit can use soft targets.
        self.samples: deque[tuple[np.ndarray, np.ndarray]] = deque(maxlen=buffer_size)
        self.geometry = {
            "FIELD_WIDTH": field_width,
            "FIELD_HEIGHT": field_height,
            "PADDLE_WIDTH": paddle_width,
            "PADDLE_HEIGHT": paddle_height,
            "BALL_RADIUS": ball_radius,
        }
        self.agent: DQNAgent | None = None
        self.optimizer: torch.optim.Optimizer | None = None
        self.matches: list[dict[str, Any]] = []
        self.losses: list[float] = []

    # --- helpers --------------------------------------------------------------
    def _make_opponent(self, name: str) -> Any:
        if name == "planner":
            return PlannerAI(
                horizon_steps=self.teacher_horizon,
                action_steps=self.teacher_action_steps,
                backend=self.backend,
            )
        return create_ai(name)

    def network_action(self, state: np.ndarray) -> int:
        assert self.agent is not None
        device = next(self.agent.q_network.parameters()).device
        with torch.no_grad():
            logits = self.agent.q_network(torch.as_tensor(state, device=device))
        return int(max(self.teacher.actions, key=lambda index: float(logits[index])))

    def beta(self, progress: float) -> float:
        progress = min(1.0, max(0.0, progress))
        return self.beta_start + (self.beta_end - self.beta_start) * progress

    def _targets(self, values: torch.Tensor) -> torch.Tensor:
        """Soft targets: sharp where one action clearly wins, flat where actions tie.

        A ``tie_break_weight`` share of one-hot mass on the planner's own choice keeps
        its stable tie-break (stay, then vertical moves) where the values are level.
        """
        soft = torch.softmax(values / self.temperature, dim=1)
        if self.tie_break_weight == 0:
            return soft
        chosen = F.one_hot(values.argmax(dim=1), values.shape[1]).to(soft.dtype)
        return (1 - self.tie_break_weight) * soft + self.tie_break_weight * chosen

    def _fit(self, updates: int) -> float:
        assert self.agent is not None and self.optimizer is not None
        agent_params = {p for p in self.agent.q_network.parameters() if p.requires_grad}
        opt_params = {p for group in self.optimizer.param_groups for p in group["params"]}
        if opt_params != agent_params:
            raise AssertionError(
                "PlannerDistiller optimizer parameters do not match agent network parameters"
            )
        if not self.samples or updates < 1:
            return float("nan")
        device = next(self.agent.q_network.parameters()).device
        population = list(self.samples)
        losses = []
        was_training = self.agent.q_network.training
        self.agent.q_network.train(True)
        for _ in range(updates):
            batch = self.rng.sample(population, min(len(population), self.batch_size))
            states = torch.as_tensor(np.stack([s for s, _ in batch]), device=device)
            values = torch.as_tensor(np.stack([v for _, v in batch]), device=device)
            log_probs = F.log_softmax(self.agent.q_network(states), dim=1)
            loss = -(self._targets(values) * log_probs).sum(dim=1).mean()
            self.optimizer.zero_grad()
            loss.backward()
            torch.nn.utils.clip_grad_norm_(self.agent.q_network.parameters(), 10.0)
            self.optimizer.step()
            losses.append(float(loss.item()))
        self.agent.q_network.train(was_training)
        self.losses.extend(losses)
        return float(np.mean(losses))

    def agreement(self, recent: int = 5000, decisive_gap: float | None = None) -> float:
        """Fraction of the most recent labels the network currently reproduces.

        With ``decisive_gap`` only states whose best rollout value beats the runner-up
        by at least that margin count; the rest are ties the planner breaks arbitrarily.
        """
        if self.agent is None or not self.samples:
            return float("nan")
        batch = list(self.samples)[-recent:]
        values = np.stack([v for _, v in batch])
        if decisive_gap is not None:
            finite = np.where(np.isfinite(values), values, -np.inf)
            top_two = -np.sort(-finite, axis=1)[:, :2]
            keep = (top_two[:, 0] - top_two[:, 1]) >= decisive_gap
            if not keep.any():
                return float("nan")
            batch = [sample for sample, flag in zip(batch, keep) if flag]
            values = values[keep]
        device = next(self.agent.q_network.parameters()).device
        with torch.no_grad():
            logits = self.agent.q_network(
                torch.as_tensor(np.stack([s for s, _ in batch]), device=device)
            )
        mask = torch.full((logits.shape[1],), -np.inf, device=device)
        mask[list(self.teacher.actions)] = 0.0
        predicted = (logits + mask).argmax(dim=1).cpu().numpy()
        return float((predicted == values.argmax(axis=1)).mean())

    # --- collection -------------------------------------------------------------
    def play_match(
        self, agent: DQNAgent, beta: float, opponent_name: str, side: int, bonuses: bool
    ) -> dict[str, Any]:
        """Play one labelled match; the learner controls ``side``."""
        self._attach(agent)
        learner = _DaggerPlayer(self, beta)
        opponent = self._make_opponent(opponent_name)
        before = len(self.samples)
        match_seed = self.rng.randrange(2**31)
        overrides = {k: v for k, v in self.geometry.items() if v is not None}
        with preserve_global_rng_state():
            random.seed(match_seed)
            np.random.seed(match_seed % (2**32))
            with game_config_tmp(BONUSES_ENABLED=bonuses, MAX_SCORE=self.match_score, **overrides):
                manager = TrainingManager(headless=True)
                try:
                    first, second = (learner, opponent) if side == 1 else (opponent, learner)
                    stats = manager.train_episode(first, second, max_steps=self.match_steps)
                finally:
                    manager.cleanup()
        goals_for = sum(goal["player"] == side for goal in stats["events"])
        goals_against = sum(goal["player"] != side for goal in stats["events"])
        record = {
            "opponent": opponent_name,
            "side": side,
            "bonuses": bonuses,
            "beta": beta,
            "seed": match_seed,
            "steps": int(stats["steps"]),
            "labels": len(self.samples) - before,
            "winner": int(stats["winner"]),
            "won": stats["winner"] == side,
            "goals_for": goals_for,
            "goals_against": goals_against,
            "label_agreement": learner.agreements / max(1, learner.decisions),
        }
        self.matches.append(record)
        return record

    def _attach(self, agent: DQNAgent) -> None:
        if self.agent is not None and self.agent is not agent:
            raise ValueError("a PlannerDistiller instance belongs to one agent")
        self.agent = agent
        agent_params = {p for p in agent.q_network.parameters() if p.requires_grad}
        opt_params = (
            {p for group in self.optimizer.param_groups for p in group["params"]}
            if self.optimizer is not None
            else set()
        )
        if self.optimizer is None or opt_params != agent_params:
            self.optimizer = torch.optim.Adam(agent.q_network.parameters(), lr=self.learning_rate)

    def run_pretraining_phase(
        self,
        agent: DQNAgent,
        total_steps: int = 100_000,
        steps_per_batch: int = 1000,
        player_id: int = 1,
        verbose: bool = True,
    ) -> dict[str, Any]:
        """Collect ``total_steps`` planner labels over matches, fitting after each match.

        ``player_id`` is accepted for interface compatibility; sides alternate so the
        network learns both. ``steps_per_batch`` is the progress-report cadence.
        """
        if total_steps < 1 or steps_per_batch < 1:
            raise ValueError("total_steps and steps_per_batch must be positive")
        self._attach(agent)
        agent.set_training_mode(True)
        start = time.perf_counter()
        collected = 0
        match_index = 0
        next_report = steps_per_batch
        counts: Counter[str] = Counter()
        while collected < total_steps:
            beta = self.beta(collected / total_steps)
            opponent_name = self.opponents[match_index % len(self.opponents)]
            side = 1 + (match_index // len(self.opponents)) % 2
            bonuses = bool((match_index // (2 * len(self.opponents))) % 2)
            record = self.play_match(agent, beta, opponent_name, side, bonuses)
            collected += record["labels"]
            counts[opponent_name] += 1
            # Re-fit on the aggregated buffer after every match, as DAgger prescribes.
            updates = max(1, int(self.epochs_per_match * len(self.samples) / self.batch_size))
            record["loss"] = self._fit(updates)
            match_index += 1
            if verbose and collected >= next_report:
                print(
                    f"   Distillation {min(collected, total_steps)}/{total_steps} labels, "
                    f"{match_index} matches, beta={beta:.2f}, loss={record['loss']:.3f}, "
                    f"agreement={self.agreement():.1%} "
                    f"(decisive {self.agreement(decisive_gap=0.1):.1%})",
                    flush=True,
                )
                while next_report <= collected:
                    next_report += steps_per_batch
        final_updates = self.final_epochs * max(1, len(self.samples) // self.batch_size)
        final_loss = self._fit(final_updates)
        agent.update_target_network()
        wins = sum(record["won"] for record in self.matches)
        return {
            "mode": "distill",
            "total_steps": collected,
            "matches": match_index,
            "match_wins": wins,
            "opponent_counts": dict(counts),
            "final_loss": final_loss,
            "average_loss": float(np.mean(self.losses)) if self.losses else float("nan"),
            "label_agreement": self.agreement(),
            "decisive_agreement": self.agreement(decisive_gap=0.1),
            "buffer_size": len(self.samples),
            "beta_start": self.beta_start,
            "beta_end": self.beta_end,
            "temperature": self.temperature,
            "tie_break_weight": self.tie_break_weight,
            "epochs_per_match": self.epochs_per_match,
            "final_epochs": self.final_epochs,
            "teacher_horizon": self.teacher_horizon,
            "teacher_action_steps": self.teacher_action_steps,
            "teacher": type(self.teacher).__name__,
            "seconds": time.perf_counter() - start,
            "match_log": self.matches,
            # Keys shared with the other pretrainers for the training script.
            "average_reward": float("nan"),
            "final_epsilon": agent.epsilon,
            "final_training_step": agent.training_step,
        }


def _parse_layers(value: str) -> list[int]:
    layers = [int(part) for part in value.split(",") if part.strip()]
    if not layers or any(width <= 0 for width in layers):
        raise argparse.ArgumentTypeError("hidden layers must be positive integers")
    return layers


def main(argv: list[str] | None = None) -> int:
    parser = argparse.ArgumentParser(description="Distil the lookahead planner into a DQN network")
    parser.add_argument("--labels", type=int, default=100_000, help="planner labels to collect")
    parser.add_argument("--output", required=True, help="checkpoint path for the distilled network")
    parser.add_argument("--hidden_layers", type=_parse_layers, default=[256, 128])
    parser.add_argument("--opponents", default=",".join(DEFAULT_DISTILL_OPPONENTS))
    parser.add_argument("--teacher_horizon", type=int, default=90)
    parser.add_argument("--teacher_action_steps", type=int, default=30)
    parser.add_argument("--beta_end", type=float, default=0.0)
    parser.add_argument("--match_steps", type=int, default=3000)
    parser.add_argument("--match_score", type=int, default=3)
    parser.add_argument("--batch_size", type=int, default=256)
    parser.add_argument("--learning_rate", type=float, default=1e-3)
    parser.add_argument("--temperature", type=float, default=0.03)
    parser.add_argument("--tie_break_weight", type=float, default=0.1)
    parser.add_argument("--epochs_per_match", type=float, default=0.25)
    parser.add_argument("--final_epochs", type=int, default=10)
    parser.add_argument("--seed", type=int, default=0)
    parser.add_argument("--load_model", help="start from an existing checkpoint")
    parser.add_argument(
        "--geometry_context",
        action="store_true",
        help="enable wall-reflection geometry features",
    )
    parser.add_argument("--evaluate", action="store_true", help="run the competitive panel")
    args = parser.parse_args(argv)

    torch.set_num_threads(max(1, torch.get_num_threads()))
    random.seed(args.seed)
    np.random.seed(args.seed)
    torch.manual_seed(args.seed)
    agent = DQNAgent(hidden_layers=args.hidden_layers, geometry_context=args.geometry_context)
    if args.load_model:
        agent.load_model(args.load_model)
        if args.geometry_context and not agent.geometry_context:
            agent.enable_geometry_context()
    distiller = PlannerDistiller(
        opponents=tuple(name.strip() for name in args.opponents.split(",") if name.strip()),
        teacher_horizon=args.teacher_horizon,
        teacher_action_steps=args.teacher_action_steps,
        beta_end=args.beta_end,
        match_steps=args.match_steps,
        match_score=args.match_score,
        batch_size=args.batch_size,
        learning_rate=args.learning_rate,
        temperature=args.temperature,
        tie_break_weight=args.tie_break_weight,
        epochs_per_match=args.epochs_per_match,
        final_epochs=args.final_epochs,
        seed=args.seed,
    )
    stats = distiller.run_pretraining_phase(agent, total_steps=args.labels, steps_per_batch=10_000)
    output = Path(args.output)
    output.parent.mkdir(parents=True, exist_ok=True)
    agent.epsilon = agent.epsilon_min
    agent.save_model(str(output))
    print(
        f"Distilled {stats['total_steps']} labels over {stats['matches']} matches; "
        f"agreement {stats['label_agreement']:.1%}; saved {output}"
    )
    if args.evaluate:
        from magic_pong.ai.competitive import CompetitiveConfig
        from magic_pong.ai.competitive import competitive_result_to_dict
        from magic_pong.ai.competitive import evaluate_competitive_agent

        result = evaluate_competitive_agent(agent, CompetitiveConfig())
        stats["evaluation"] = competitive_result_to_dict(result)
        wins = sum(matchup.wins for matchup in result.matchups)
        total = sum(matchup.matches for matchup in result.matchups)
        scoring_rates = {}
        for opponent in sorted({m.opponent for m in result.matches}):
            opp_m = [m for m in result.matches if m.opponent == opponent]
            gf = sum(m.goals_for for m in opp_m)
            ga = sum(m.goals_against for m in opp_m)
            steps = sum(m.steps for m in opp_m)
            gf_per_1k = 1000.0 * gf / max(1, steps)
            ga_per_1k = 1000.0 * ga / max(1, steps)
            frames_per_goal = steps / max(1, gf)
            scoring_rates[opponent] = {
                "goals_for": gf,
                "goals_against": ga,
                "steps": steps,
                "goals_for_per_1000_frames": gf_per_1k,
                "goals_against_per_1000_frames": ga_per_1k,
                "frames_per_goal_for": frames_per_goal,
            }
        stats["scoring_rates"] = scoring_rates
        print(f"Competitive panel: {wins}/{total} wins")
        for matchup in result.matchups:
            opp_m = [
                m
                for m in result.matches
                if m.opponent == matchup.opponent
                and m.player_side == matchup.player_side
                and m.bonuses_enabled == matchup.bonuses_enabled
            ]
            cell_gf = sum(m.goals_for for m in opp_m)
            cell_steps = sum(m.steps for m in opp_m)
            cell_gf_1k = 1000.0 * cell_gf / max(1, cell_steps)
            print(
                f"  {matchup.opponent:15s} side={matchup.player_side} bonuses={matchup.bonuses_enabled!s:5s} "
                f"wins={matchup.wins}/{matchup.matches} goal_diff={matchup.avg_goal_diff:+.1f} "
                f"gf_per_1k={cell_gf_1k:.2f}"
            )
        for opponent, rates in scoring_rates.items():
            print(
                f"  Summary {opponent:15s}: {rates['goals_for']} goals / {rates['steps']} frames "
                f"({rates['goals_for_per_1000_frames']:.3f} GF/1k frames)"
            )
    output.with_suffix(".distillation.json").write_text(json.dumps(stats, indent=2) + "\n")
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
