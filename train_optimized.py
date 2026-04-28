#!/usr/bin/env python3
"""
Advanced DQN Training Script with Comprehensive Configuration Options

This script provides extensive customization for DQN training including:
- Checkpoint saving and resuming
- Model loading and continuation
- DQN hyperparameter customization
- Flexible output paths
- Curriculum learning phases
- Comprehensive evaluation
"""

import argparse
import json
import math
import os
import random
import time
from collections.abc import Sequence
from contextlib import contextmanager
from dataclasses import dataclass
from datetime import datetime
from pathlib import Path
from typing import Any

import numpy as np

from magic_pong.ai import evaluation as ai_evaluation
from magic_pong.ai.models.dqn_ai import DQNAgent
from magic_pong.ai.models.simple_ai import create_ai
from magic_pong.core.game_engine import TrainingManager
from magic_pong.utils.config import ai_config
from magic_pong.utils.config import game_config

AI_TYPES = [
    "follow_ball",
    "defensive",
    "aggressive",
    "random",
    "predictive",
    "dummy",
    "training_dummy",
]

DEFAULT_OPPONENT_MIX = {
    "defensive": 0.35,
    "follow_ball": 0.20,
    "predictive": 0.20,
    "random": 0.15,
    "dummy": 0.10,
}
DEFAULT_OPPONENT_MIX_CLI = ",".join(
    f"{name}:{weight:g}" for name, weight in DEFAULT_OPPONENT_MIX.items()
)
DEFAULT_TRAINING_SCORE_MIX = (3, 5, 11)
DEFAULT_TRAINING_SCORE_MIX_CLI = ",".join(str(score) for score in DEFAULT_TRAINING_SCORE_MIX)
DEFAULT_CHECKPOINT_EVAL_OPPONENTS = tuple(DEFAULT_OPPONENT_MIX)
DEFAULT_CHECKPOINT_EVAL_EPISODES = 10
MIXED_FINE_TUNING_MODES = frozenset({"continue", "single", "dual_scale"})


@dataclass(frozen=True)
class WeightedOpponent:
    """Normalized opponent weight used for deterministic mixed fine-tuning."""

    name: str
    weight: float


class OpponentSampler:
    """Provide a fresh sampled opponent for each training episode."""

    def __init__(self, opponent_mix: Sequence[WeightedOpponent], rng: random.Random):
        self.opponent_mix = tuple(opponent_mix)
        self.rng = rng

    def next_opponent(self):
        selected_opponent = sample_weighted_opponent(self.opponent_mix, self.rng)
        return selected_opponent.name, get_opponent(selected_opponent.name)


def get_pyplot():
    """Import matplotlib only when a plotting path is used."""
    import matplotlib.pyplot as plt

    return plt


def create_agent_from_args(args):
    """Create a DQN agent with parameters from command line arguments"""
    # Determine training mode and dual-scale configuration
    training_mode = getattr(args, "training_mode", "step_by_step")
    enable_dual_scale = getattr(args, "enable_dual_scale_training", False)

    return DQNAgent(
        state_size=args.state_size,
        action_size=args.action_size,
        lr=args.learning_rate,
        gamma=args.gamma,
        epsilon=args.epsilon_start,
        epsilon_decay=args.epsilon_decay,
        epsilon_min=args.epsilon_min,
        batch_size=args.batch_size,
        use_prioritized_replay=args.use_prioritized_replay,
        tau=args.tau,
        memory_size=args.memory_size,
        # Advanced training configuration
        training_mode=training_mode,
        enable_dual_scale_training=enable_dual_scale,
        tactical_train_frequency=getattr(args, "tactical_train_frequency", 10),
        tactical_learning_rate=getattr(args, "tactical_learning_rate", None),
        strategic_learning_rate=getattr(args, "strategic_learning_rate", None),
        train_frequency=getattr(args, "train_frequency", 10),
        min_replay_size=getattr(args, "min_replay_size", 1000),
        reward_normalization=getattr(args, "reward_normalization", True),
    )


def get_ball_config_from_args(args):
    """Get ball direction and angle configuration from command line arguments"""
    verbose = args.verbose

    # If specific angle is provided, use it
    if args.ball_angle is not None:
        angle_rad = math.radians(args.ball_angle)
        direction = 1

    # Convert direction name to angle and direction
    direction_configs = {
        "right": (1, math.radians(0)),
        "up_right": (1, math.radians(-45)),
        "down_right": (1, math.radians(45)),
        "left": (1, math.radians(180)),
        "up_left": (1, math.radians(-135)),
        "down_left": (1, math.radians(135)),
    }
    cone_direction_configs = {
        "cone_right": (1, math.pi / 4, -math.pi / 4),  # -45° to 45°
        "cone_left": (-1, math.pi / 4, -math.pi / 4),  # 135° to 225°
    }

    if isinstance(args.ball_direction, list | tuple):
        if len(args.ball_direction) == 2:
            direction, angle_rad = args.ball_direction
        elif len(args.ball_direction) == 3:
            angle_rad = np.random.uniform(args.ball_direction[0], args.ball_direction[1])
            direction = args.ball_direction[2]
        else:
            raise ValueError("ball_direction as tuple/list must have exactly 2 or 3 elements")
    elif args.ball_direction in direction_configs:
        direction, angle_rad = direction_configs[args.ball_direction]
    elif args.ball_direction in cone_direction_configs:
        direction, angle_min, angle_max = cone_direction_configs[args.ball_direction]
        angle_rad = np.random.uniform(angle_min, angle_max)
    else:
        direction, angle_rad = 0, None  # fallback to random

    if verbose:
        print(
            f"Ball initial direction: {args.ball_direction} -> direction={direction}, angle={angle_rad}"
        )
    return direction, angle_rad


def save_checkpoint(
    agent, episode, phase, total_episodes, rewards, wins, checkpoint_dir, prefix="checkpoint"
):
    """Save a training checkpoint"""
    checkpoint_dir = Path(checkpoint_dir)
    checkpoint_dir.mkdir(parents=True, exist_ok=True)

    timestamp = datetime.now().strftime("%Y%m%d_%H%M%S")
    checkpoint_name = f"{prefix}_ep{episode}_phase{phase}_{timestamp}.pth"
    checkpoint_path = checkpoint_dir / checkpoint_name

    # Save model
    agent.save_model(str(checkpoint_path))

    # Save training metadata
    metadata = {
        "episode": episode,
        "phase": phase,
        "total_episodes": total_episodes,
        "rewards": rewards,
        "wins": wins,
        "epsilon": agent.epsilon,
        "timestamp": timestamp,
        "model_path": str(checkpoint_path),
    }

    metadata_path = checkpoint_dir / f"{prefix}_ep{episode}_phase{phase}_{timestamp}.json"
    with open(metadata_path, "w") as f:
        json.dump(metadata, f, indent=2)

    print(f"✅ Checkpoint saved: {checkpoint_path}")
    return str(checkpoint_path), str(metadata_path)


def load_checkpoint(agent, checkpoint_path):
    """Load a training checkpoint"""
    checkpoint_path = Path(checkpoint_path)

    if checkpoint_path.suffix == ".json":
        # Load from metadata file
        with open(checkpoint_path) as f:
            metadata = json.load(f)
        model_path = metadata["model_path"]
    else:
        # Direct model path
        model_path = str(checkpoint_path)
        metadata = None

    # Load model
    agent.load_model(model_path)
    print(f"✅ Model loaded from: {model_path}")

    return metadata


def load_existing_model(agent, model_path):
    """Load an existing trained model"""
    if not os.path.exists(model_path):
        raise FileNotFoundError(f"Model file not found: {model_path}")

    agent.load_model(model_path)
    print(f"✅ Existing model loaded from: {model_path}")


def get_opponent(opponent_type, player_id=2):
    """Create opponent based on type"""
    if opponent_type not in AI_TYPES:
        print(f"Warning: Unknown opponent type '{opponent_type}', using 'follow_ball'")
        opponent_type = "follow_ball"

    return create_ai(opponent_type)


def parse_opponent_mix(
    mix_value: str | Sequence[tuple[str, float]],
) -> tuple[WeightedOpponent, ...]:
    """Parse and normalize a weighted opponent mix."""
    if isinstance(mix_value, str):
        raw_items: list[tuple[str, str | float]] = []
        for item in mix_value.split(","):
            item = item.strip()
            if not item:
                continue
            if ":" not in item:
                raise ValueError(
                    f"Invalid opponent mix entry {item!r}; expected '<opponent>:<weight>'"
                )
            name, weight = item.split(":", 1)
            raw_items.append((name.strip(), weight.strip()))
    else:
        raw_items = [(name, weight) for name, weight in mix_value]

    if not raw_items:
        raise ValueError("Opponent mix must contain at least one '<opponent>:<weight>' entry")

    seen = set()
    parsed: list[tuple[str, float]] = []
    for name, raw_weight in raw_items:
        if name not in AI_TYPES:
            valid_names = ", ".join(AI_TYPES)
            raise ValueError(f"Unknown opponent type {name!r}. Valid opponents: {valid_names}")
        if name in seen:
            raise ValueError(f"Duplicate opponent type {name!r} in opponent mix")
        seen.add(name)

        try:
            weight = float(raw_weight)
        except (TypeError, ValueError) as exc:
            raise ValueError(f"Invalid weight for opponent {name!r}: {raw_weight!r}") from exc
        if not math.isfinite(weight) or weight <= 0:
            raise ValueError(f"Weight for opponent {name!r} must be a positive finite number")
        parsed.append((name, weight))

    total_weight = sum(weight for _, weight in parsed)
    return tuple(
        WeightedOpponent(name=name, weight=weight / total_weight) for name, weight in parsed
    )


def parse_training_score_targets(score_value: str | Sequence[int]) -> tuple[int, ...]:
    """Parse positive MAX_SCORE targets for mixed-opponent fine-tuning."""
    if isinstance(score_value, str):
        raw_scores: Sequence[Any] = [
            item.strip() for item in score_value.split(",") if item.strip()
        ]
    else:
        raw_scores = score_value

    if not raw_scores:
        raise ValueError("Training score mix must contain at least one positive integer")

    scores: list[int] = []
    for raw_score in raw_scores:
        try:
            score = int(raw_score)
        except (TypeError, ValueError) as exc:
            raise ValueError(f"Invalid training score target {raw_score!r}") from exc
        if score <= 0:
            raise ValueError("Training score targets must be positive integers")
        scores.append(score)
    return tuple(scores)


def sample_weighted_opponent(
    opponent_mix: Sequence[WeightedOpponent], rng: random.Random
) -> WeightedOpponent:
    """Sample one normalized weighted opponent using the provided deterministic RNG."""
    return rng.choices(
        list(opponent_mix),
        weights=[opponent.weight for opponent in opponent_mix],
        k=1,
    )[0]


def sample_training_score_target(score_targets: Sequence[int], rng: random.Random) -> int:
    """Sample a training MAX_SCORE target using the provided deterministic RNG."""
    return rng.choice(list(score_targets))


def mixed_fine_tuning_enabled(args) -> bool:
    """Return whether this phase should use mixed-opponent fine-tuning."""
    return bool(getattr(args, "mixed_opponents", False)) and (
        getattr(args, "mode", None) in MIXED_FINE_TUNING_MODES
    )


def validate_mixed_fine_tuning_args(args) -> None:
    """Keep mixed-opponent training scoped to explicit fine-tuning runs."""
    if not getattr(args, "mixed_opponents", False):
        return

    if args.mode not in MIXED_FINE_TUNING_MODES:
        modes = ", ".join(sorted(MIXED_FINE_TUNING_MODES))
        raise ValueError(f"--mixed_opponents can only be used with --mode in: {modes}")
    if not (args.load_model or args.load_checkpoint):
        raise ValueError(
            "--mixed_opponents is a fine-tuning mode and requires --load_model or --load_checkpoint"
        )


def _opponent_mix_from_args(args) -> tuple[WeightedOpponent, ...]:
    parsed_mix = getattr(args, "parsed_opponent_mix", None)
    if parsed_mix is not None:
        return tuple(parsed_mix)
    return parse_opponent_mix(getattr(args, "opponent_mix", DEFAULT_OPPONENT_MIX_CLI))


def _training_score_targets_from_args(args) -> tuple[int, ...]:
    parsed_scores = getattr(args, "parsed_training_score_targets", None)
    if parsed_scores is not None:
        return tuple(parsed_scores)
    return parse_training_score_targets(
        getattr(args, "training_score_mix", DEFAULT_TRAINING_SCORE_MIX_CLI)
    )


@contextmanager
def temporary_game_config_value(name: str, value: Any):
    """Temporarily set a game_config value and always restore it."""
    original_value = getattr(game_config, name)
    setattr(game_config, name, value)
    try:
        yield
    finally:
        setattr(game_config, name, original_value)


@contextmanager
def preserve_global_rng_state():
    """Preserve global RNG state around deterministic checkpoint evaluation."""
    python_state = random.getstate()
    numpy_state = np.random.get_state()
    torch_module = None
    torch_state = None
    cuda_states = None

    try:
        import torch

        torch_module = torch
        torch_state = torch.get_rng_state()
        if torch.cuda.is_available():
            cuda_states = torch.cuda.get_rng_state_all()
    except ImportError:
        pass

    try:
        yield
    finally:
        random.setstate(python_state)
        np.random.set_state(numpy_state)
        if torch_module is not None and torch_state is not None:
            torch_module.set_rng_state(torch_state)
            if cuda_states is not None:
                torch_module.cuda.set_rng_state_all(cuda_states)


def _checkpoint_eval_opponents(args) -> tuple[str, ...]:
    repeated_opponents = tuple(getattr(args, "checkpoint_eval_opponents", None) or ())
    csv_opponents: tuple[str, ...] = ()
    csv_value = getattr(args, "checkpoint_eval_opponents_csv", None)
    if csv_value:
        csv_opponents = tuple(item.strip() for item in csv_value.split(",") if item.strip())

    opponents = repeated_opponents + csv_opponents
    if not opponents:
        return DEFAULT_CHECKPOINT_EVAL_OPPONENTS

    invalid = sorted(set(opponents) - set(AI_TYPES))
    if invalid:
        raise ValueError(f"Unknown checkpoint evaluation opponent(s): {', '.join(invalid)}")
    if "training_dummy" in opponents:
        raise ValueError("training_dummy cannot be used for deterministic checkpoint evaluation")
    return opponents


def checkpoint_evaluation_enabled(args) -> bool:
    return bool(getattr(args, "evaluate_checkpoints", False)) or (
        getattr(args, "checkpoint_eval_episodes", 0) > 0
    )


def checkpoint_evaluation_config(args) -> ai_evaluation.EvaluationConfig:
    """Build the reusable evaluation harness config for saved training checkpoints."""
    bonuses_enabled = getattr(args, "checkpoint_eval_bonuses_enabled", None)
    if bonuses_enabled is None and mixed_fine_tuning_enabled(args):
        bonuses_enabled = bool(getattr(args, "fine_tune_bonuses", False))

    checkpoint_eval_max_steps = getattr(args, "checkpoint_eval_max_steps", None)
    if checkpoint_eval_max_steps is None:
        checkpoint_eval_max_steps = getattr(args, "max_steps_per_episode", 1000)

    return ai_evaluation.EvaluationConfig(
        opponents=_checkpoint_eval_opponents(args),
        episodes=(getattr(args, "checkpoint_eval_episodes", 0) or DEFAULT_CHECKPOINT_EVAL_EPISODES),
        max_steps=checkpoint_eval_max_steps,
        seed=getattr(args, "checkpoint_eval_seed", 0),
        eval_epsilon=getattr(args, "checkpoint_eval_epsilon", 0.0),
        max_score=getattr(args, "checkpoint_eval_max_score", None),
        bonuses_enabled=bonuses_enabled,
        include_episodes=getattr(args, "checkpoint_eval_include_episodes", False),
    )


def checkpoint_evaluation_path(checkpoint_path: str | Path) -> Path:
    """Return the JSON path written next to a checkpoint candidate."""
    checkpoint_path = Path(checkpoint_path)
    return checkpoint_path.with_name(f"{checkpoint_path.stem}.eval.json")


def maybe_save_checkpoint_evaluation(
    checkpoint_path: str | Path, args, training_context: dict[str, Any] | None = None
) -> str | None:
    """Evaluate a saved checkpoint and write JSON next to it when requested."""
    if not checkpoint_evaluation_enabled(args):
        return None

    config = checkpoint_evaluation_config(args)
    with preserve_global_rng_state():
        result = ai_evaluation.evaluate_checkpoint(checkpoint_path, config)
    output_path = checkpoint_evaluation_path(checkpoint_path)
    payload = ai_evaluation.result_to_dict(result)
    if training_context is not None:
        payload["training_context"] = training_context
    output_path.write_text(json.dumps(payload, indent=2) + "\n", encoding="utf-8")
    print(f"Checkpoint evaluation saved: {output_path}")
    return str(output_path)


def checkpoint_training_context(
    *,
    episode: int,
    phase_num: int,
    phase_name: str,
    total_episodes: int,
    use_mixed_fine_tuning: bool,
    opponent,
    opponent_mix: Sequence[WeightedOpponent],
    score_targets: Sequence[int],
    episode_max_score: int,
    episode_opponent_name: str,
) -> dict[str, Any]:
    """Metadata that ties checkpoint evaluation output back to training curriculum state."""
    context: dict[str, Any] = {
        "episode": episode,
        "phase": phase_num,
        "phase_name": phase_name,
        "phase_total_episodes": total_episodes,
        "mixed_opponents": use_mixed_fine_tuning,
        "episode_opponent": episode_opponent_name,
        "episode_max_score": episode_max_score,
        "bonuses_enabled": game_config.BONUSES_ENABLED,
    }
    if use_mixed_fine_tuning:
        context["opponent_mix"] = [
            {"opponent": entry.name, "weight": entry.weight} for entry in opponent_mix
        ]
        context["training_max_scores"] = list(score_targets)
    else:
        context["fixed_opponent"] = getattr(opponent, "name", str(opponent))
        context["training_max_score"] = episode_max_score
    return context


def final_model_training_context(
    args,
    *,
    label: str,
    total_episodes: int,
    rewards: Sequence[float],
    wins: int,
) -> dict[str, Any]:
    """Metadata used when evaluating a final saved model candidate."""
    recent_rewards = rewards[-100:] if len(rewards) >= 100 else rewards
    context: dict[str, Any] = {
        "candidate_type": "final_model",
        "mode": getattr(args, "mode", None),
        "label": label,
        "total_episodes": total_episodes,
        "wins": wins,
        "win_rate": wins / total_episodes * 100 if total_episodes > 0 else 0,
        "avg_reward": float(np.mean(recent_rewards)) if recent_rewards else 0.0,
    }

    if mixed_fine_tuning_enabled(args):
        context["mixed_opponents"] = True
        context["opponent_mix"] = [
            {"opponent": entry.name, "weight": entry.weight}
            for entry in _opponent_mix_from_args(args)
        ]
        context["training_max_scores"] = list(_training_score_targets_from_args(args))
        context["bonuses_enabled"] = bool(getattr(args, "fine_tune_bonuses", False))
    else:
        context["mixed_opponents"] = False
        context["training_max_score"] = getattr(args, "training_max_score", None)

    return context


def train_phase(
    agent,
    opponent,
    phase_name,
    episodes,
    training_manager,
    args,
    phase_num=1,
    total_phases=2,
    start_episode=0,
):
    """Train a single phase"""
    print(f"\n📚 PHASE {phase_num}: {phase_name} ({episodes} episodes)")

    original_max_score = game_config.MAX_SCORE
    use_mixed_fine_tuning = mixed_fine_tuning_enabled(args)
    opponent_source = opponent
    opponent_mix: tuple[WeightedOpponent, ...] = ()
    score_targets: tuple[int, ...] = ()
    mix_rng: random.Random | None = None

    if use_mixed_fine_tuning:
        opponent_mix = _opponent_mix_from_args(args)
        score_targets = _training_score_targets_from_args(args)
        mix_rng = getattr(args, "opponent_mix_rng", None)
        if mix_rng is None:
            mix_rng = random.Random(getattr(args, "opponent_mix_seed", 0))
            args.opponent_mix_rng = mix_rng
        opponent_source = OpponentSampler(opponent_mix, mix_rng)

        mix_summary = ", ".join(f"{item.name}:{item.weight:.2f}" for item in opponent_mix)
        print(f"Mixed fine-tuning opponents: {mix_summary}")
        print(f"Mixed fine-tuning MAX_SCORE targets: {', '.join(map(str, score_targets))}")
        if getattr(args, "fine_tune_bonuses", False):
            print("Mixed fine-tuning bonuses: ENABLED")
        else:
            print("Mixed fine-tuning bonuses: DISABLED")
    elif args.training_max_score != original_max_score:
        print(
            f"Configuration temporaire: MAX_SCORE = {args.training_max_score} (au lieu de {original_max_score})"
        )

    original_bonuses_enabled = game_config.BONUSES_ENABLED
    if use_mixed_fine_tuning:
        game_config.BONUSES_ENABLED = bool(getattr(args, "fine_tune_bonuses", False))

    try:
        rewards = []
        wins = 0
        wins_history = []  # Track wins per episode for recent win rate calculation
        best_avg_reward = float("-inf")
        episodes_since_improvement = 0

        for episode in range(start_episode, start_episode + episodes):
            if hasattr(opponent_source, "next_opponent"):
                episode_opponent_name, episode_opponent = opponent_source.next_opponent()
                if score_targets:
                    assert mix_rng is not None
                    episode_max_score = sample_training_score_target(score_targets, mix_rng)
                else:
                    episode_max_score = args.training_max_score
                if args.verbose:
                    print(
                        f"    Episode {episode}: opponent={episode_opponent_name}, MAX_SCORE={episode_max_score}"
                    )
            else:
                episode_opponent = opponent_source
                episode_opponent_name = getattr(opponent_source, "name", str(opponent_source))
                episode_max_score = args.training_max_score

            ball_direction, ball_angle = get_ball_config_from_args(args)
            training_manager.set_ball_initial_direction(ball_direction, ball_angle)

            agent.on_episode_start()

            with temporary_game_config_value("MAX_SCORE", episode_max_score):
                episode_stats = training_manager.train_episode(
                    agent, episode_opponent, max_steps=args.max_steps_per_episode
                )
            episode_reward = episode_stats["total_reward_p1"]
            rewards.append(episode_reward)

            winner = episode_stats.get("winner", 0)
            episode_won = 1 if winner == 1 else 0
            wins_history.append(episode_won)
            if winner == 1:
                wins += 1

            # Debug info for troubleshooting
            if args.verbose:
                print(
                    f"    Episode {episode}: reward={episode_reward:.2f}, winner={winner}, agent_wins={wins}"
                )

            # Progress reporting
            if (episode + 1) % args.log_interval == 0 and episode > start_episode:
                recent_rewards = rewards[-args.log_interval :]
                avg_reward = np.mean(recent_rewards)

                # Calculate recent win rate (same window as recent rewards)
                recent_wins = wins_history[-args.log_interval :]
                recent_win_rate = np.mean(recent_wins) * 100  # Convert to percentage

                # Global win rate for comparison
                global_win_rate = wins / (episode - start_episode + 1) * 100

                # Get detailed training statistics
                training_stats = agent.get_training_stats()

                base_info = (
                    f"  Episode {episode}: avg reward = {avg_reward:.2f}, "
                    f"win rate = {recent_win_rate:.1f}% (global: {global_win_rate:.1f}%), epsilon = {agent.epsilon:.3f}"
                )

                # Add dual-scale specific information if enabled
                if training_stats.get("dual_scale_training", False):
                    dual_scale_info = (
                        f", tactical steps = {training_stats['tactical_step_count']}, "
                        f"memory = {training_stats['memory_size']}"
                    )
                    print(base_info + dual_scale_info)

                    # Verbose dual-scale statistics
                    if args.verbose:
                        print(f"    🎯 Tactical LR: {training_stats['tactical_optimizer_lr']:.6f}")
                        print(
                            f"    🧠 Strategic LR: {training_stats['strategic_optimizer_lr']:.6f}"
                        )
                        print(f"    📚 Training mode: {training_stats['training_mode']}")
                        print(
                            f"    📊 Episode buffer: {training_stats['episode_buffer_size']} experiences"
                        )
                else:
                    print(base_info)
                    if args.verbose:
                        print(f"    📚 Training mode: {training_stats['training_mode']}")
                        print(f"    📊 Memory: {training_stats['memory_size']} experiences")

                # Check for improvement
                if avg_reward > best_avg_reward:
                    best_avg_reward = avg_reward
                    episodes_since_improvement = 0
                else:
                    episodes_since_improvement += args.log_interval

            # Save checkpoint
            if (
                args.checkpoint_interval > 0
                and episode % args.checkpoint_interval == 0
                and episode > start_episode
            ):
                checkpoint_path, _metadata_path = save_checkpoint(
                    agent,
                    episode,
                    phase_num,
                    start_episode + episodes,
                    rewards,
                    wins,
                    args.checkpoint_dir,
                )
                maybe_save_checkpoint_evaluation(
                    checkpoint_path,
                    args,
                    checkpoint_training_context(
                        episode=episode,
                        phase_num=phase_num,
                        phase_name=phase_name,
                        total_episodes=start_episode + episodes,
                        use_mixed_fine_tuning=use_mixed_fine_tuning,
                        opponent=opponent,
                        opponent_mix=opponent_mix,
                        score_targets=score_targets,
                        episode_max_score=episode_max_score,
                        episode_opponent_name=episode_opponent_name,
                    ),
                )

            # Early stopping
            if args.early_stopping > 0 and episodes_since_improvement >= args.early_stopping:
                print(
                    f"Early stopping triggered after {episodes_since_improvement} episodes without improvement"
                )
                break

        final_episodes = len(rewards)
        avg_reward = np.mean(rewards[-100:]) if len(rewards) >= 100 else np.mean(rewards)
        win_rate = wins / final_episodes * 100 if final_episodes > 0 else 0

        print(f"✅ Phase {phase_num} completed:")
        print(f"   Episodes: {final_episodes}")
        print(f"   Average reward: {avg_reward:.2f}")
        print(f"   Win rate: {win_rate:.1f}% ({wins}/{final_episodes} wins)")

        # Debug: show some reward statistics
        if rewards:
            print(
                f"   Reward stats: min={min(rewards):.2f}, max={max(rewards):.2f}, std={np.std(rewards):.2f}"
            )

        return rewards, wins, final_episodes

    finally:
        # Restaurer la configuration originale
        game_config.MAX_SCORE = original_max_score
        if use_mixed_fine_tuning:
            game_config.BONUSES_ENABLED = original_bonuses_enabled


def create_optimized_agent():
    """Create a DQN agent with optimized hyperparameters and advanced dual-scale training"""
    return DQNAgent(
        state_size=32,
        action_size=9,
        lr=0.001,  # Base learning rate
        gamma=0.99,
        epsilon=1.0,
        epsilon_decay=0.998,  # Slower decay for more exploration
        epsilon_min=0.05,  # Slightly higher min epsilon
        batch_size=64,  # Larger batch size for stability
        use_prioritized_replay=True,
        tau=0.003,  # Slower soft update
        memory_size=20000,  # Larger buffer
        # Advanced dual-scale training configuration
        training_mode="step_by_step",
        enable_dual_scale_training=True,
        tactical_train_frequency=10,  # Tactical training every 10 steps
        tactical_learning_rate=0.0003,  # Conservative for tactical stability
        strategic_learning_rate=0.0015,  # More aggressive for strategic adaptation
        train_frequency=1,  # Allow frequent updates due to dual-scale optimization
        min_replay_size=1000,  # Start training after sufficient experience
        reward_normalization=True,
    )


def train_with_curriculum(agent, args):
    """
    Training with curriculum learning
    Phase 1: Basic training (simple game)
    Phase 2: Advanced training (with bonuses and complexity)
    """
    print("=== CURRICULUM LEARNING TRAINING ===")

    # Get ball configuration
    ball_direction, ball_angle = get_ball_config_from_args(args)
    training_manager = TrainingManager(
        headless=args.headless, initial_ball_direction=ball_direction, initial_ball_angle=ball_angle
    )

    # Phase 1: Basic training
    phase1_opponent = get_opponent(args.phase1_opponent)
    phase1_rewards, phase1_wins, phase1_episodes = train_phase(
        agent,
        phase1_opponent,
        f"Basic learning vs {args.phase1_opponent}",
        args.episodes_per_phase,
        training_manager,
        args,
        phase_num=1,
        total_phases=2,
    )

    # Save phase 1 model
    if args.save_phase_models:
        phase1_path = Path(args.output_dir) / f"{args.model_prefix}_phase1.pth"
        agent.save_model(str(phase1_path))
        print(f"Phase 1 model saved: {phase1_path}")

    # Phase 2: Advanced training (reduce exploration)
    if args.phase2_epsilon_reduction:
        agent.epsilon = max(agent.epsilon * 0.3, agent.epsilon_min)
        agent.epsilon_decay = 0.999

    phase2_opponent = get_opponent(args.phase2_opponent)
    phase2_rewards, phase2_wins, phase2_episodes = train_phase(
        agent,
        phase2_opponent,
        f"Advanced learning vs {args.phase2_opponent}",
        args.episodes_per_phase,
        training_manager,
        args,
        phase_num=2,
        total_phases=2,
        start_episode=phase1_episodes,
    )

    # Save final model
    final_model_path = Path(args.output_dir) / f"{args.model_prefix}_final.pth"
    agent.save_model(str(final_model_path))
    print(f"Final model saved: {final_model_path}")

    # Create training plots
    if args.save_plots:
        create_training_plots(
            phase1_rewards,
            phase2_rewards,
            phase1_wins / phase1_episodes * 100 if phase1_episodes > 0 else 0,
            phase2_wins / phase2_episodes * 100 if phase2_episodes > 0 else 0,
            args,
        )

    # Cleanup training manager
    training_manager.cleanup()

    maybe_save_checkpoint_evaluation(
        final_model_path,
        args,
        final_model_training_context(
            args,
            label="curriculum_final",
            total_episodes=phase1_episodes + phase2_episodes,
            rewards=phase1_rewards + phase2_rewards,
            wins=phase1_wins + phase2_wins,
        ),
    )

    return agent, {
        "phase1_avg": np.mean(phase1_rewards[-100:])
        if len(phase1_rewards) >= 100
        else np.mean(phase1_rewards)
        if phase1_rewards
        else 0,
        "phase1_winrate": phase1_wins / phase1_episodes * 100 if phase1_episodes > 0 else 0,
        "phase2_avg": np.mean(phase2_rewards[-100:])
        if len(phase2_rewards) >= 100
        else np.mean(phase2_rewards)
        if phase2_rewards
        else 0,
        "phase2_winrate": phase2_wins / phase2_episodes * 100 if phase2_episodes > 0 else 0,
        "total_episodes": phase1_episodes + phase2_episodes,
        "final_model_path": str(final_model_path),
    }


def train_single_phase(agent, args):
    """Training with a single phase (no curriculum)"""
    print("=== SINGLE PHASE TRAINING ===")

    # Get ball configuration
    ball_direction, ball_angle = get_ball_config_from_args(args)
    training_manager = TrainingManager(
        headless=args.headless, initial_ball_direction=ball_direction, initial_ball_angle=ball_angle
    )
    opponent = get_opponent(args.opponent)

    rewards, wins, episodes = train_phase(
        agent,
        opponent,
        f"Training vs {args.opponent}",
        args.total_episodes,
        training_manager,
        args,
        phase_num=1,
        total_phases=1,
    )

    # Save final model
    final_model_path = Path(args.output_dir) / f"{args.model_prefix}_final.pth"
    agent.save_model(str(final_model_path))
    print(f"Final model saved: {final_model_path}")

    # Create training plots
    if args.save_plots:
        create_single_phase_plots(rewards, wins / episodes * 100 if episodes > 0 else 0, args)

    # Cleanup training manager
    training_manager.cleanup()

    maybe_save_checkpoint_evaluation(
        final_model_path,
        args,
        final_model_training_context(
            args,
            label="single_phase_final",
            total_episodes=episodes,
            rewards=rewards,
            wins=wins,
        ),
    )

    return agent, {
        "avg_reward": np.mean(rewards[-100:])
        if len(rewards) >= 100
        else np.mean(rewards)
        if rewards
        else 0,
        "win_rate": wins / episodes * 100 if episodes > 0 else 0,
        "total_episodes": episodes,
        "final_model_path": str(final_model_path),
    }


def train_progressive_curriculum(agent, args):
    """
    Progressive 4-phase curriculum learning:
    Phase 1: Learn to hit the ball (vs stationary dummy opponent - no interference)
    Phase 2: Learn to return the ball to opponent side (vs defensive AI)
    Phase 3: Learn to play against smart opponent (vs predictive AI)
    Phase 4: Master game with bonuses (vs predictive AI with bonuses enabled)
    """
    print("=== PROGRESSIVE 4-PHASE CURRICULUM LEARNING ===")
    print("Phase 1: Learning to hit the ball (vs stationary opponent)")
    print("Phase 2: Learning to return the ball")
    print("Phase 3: Playing vs intelligent AI")
    print("Phase 4: Mastering with bonuses")

    # Disable bonuses for Phase 1, 2 and 3
    original_bonuses_enabled = game_config.BONUSES_ENABLED
    game_config.BONUSES_ENABLED = False

    # Get ball configuration
    training_manager = TrainingManager(headless=args.headless, fast_gui=args.fast_gui)

    all_phase_data = []
    total_episodes_count = 0

    initial_ball_direction = args.ball_direction
    args.ball_direction = [-math.pi / 12, math.pi / 12, -1]  # Cone towards left side

    # Phase 1: Learn to hit the ball - vs dummy or training_dummy
    print("\n🎯 PHASE 1: Learning to Hit the Ball")
    print("Objective: Basic ball contact and paddle control")
    if args.phase1_training_opponent == "dummy":
        print("Opponent: Dummy AI (completely stationary - never interferes)")
        print("Focus: Pure ball contact learning without any opponent interference")
    else:
        print("Opponent: Training Dummy AI (minimal predictable movement)")
        print("Focus: Ball contact with minimal, non-threatening opponent movement")

    initial_score_reward = ai_config.SCORE_REWARD
    initial_lose_penalty = ai_config.LOSE_PENALTY
    initial_wall_hit_reward = ai_config.WALL_HIT_REWARD
    initial_use_proximity_reward = ai_config.USE_PROXIMITY_REWARD
    initial_proximity_reward_factor = ai_config.PROXIMITY_REWARD_FACTOR
    initial_proximity_penalty_factor = ai_config.PROXIMITY_PENALTY_FACTOR
    initial_max_proximity_reward = ai_config.MAX_PROXIMITY_REWARD
    initial_game_speed_multiplier = game_config.GAME_SPEED_MULTIPLIER
    initial_fps = game_config.FPS
    ai_config.SCORE_REWARD = 0
    ai_config.LOSE_PENALTY = 0
    ai_config.WALL_HIT_REWARD = 1
    ai_config.USE_PROXIMITY_REWARD = True
    ai_config.PROXIMITY_REWARD_FACTOR = 0.1
    ai_config.PROXIMITY_PENALTY_FACTOR = 0.1
    ai_config.MAX_PROXIMITY_REWARD = 0.5
    ai_config.SHOW_OPTIMAL_POINTS_GUI = True
    # game_config.GAME_SPEED_MULTIPLIER = 5.0
    game_config.FPS = 300.0

    phase1_opponent = get_opponent(args.phase1_training_opponent)
    phase1_rewards, phase1_wins, phase1_episodes = train_phase(
        agent,
        phase1_opponent,
        f"Ball contact training vs {args.phase1_training_opponent.replace('_', ' ').title()}",
        args.progressive_episodes_per_phase,
        training_manager,
        args,
        phase_num=1,
        total_phases=4,
        start_episode=0,
    )
    ai_config.SCORE_REWARD = initial_score_reward
    ai_config.LOSE_PENALTY = initial_lose_penalty
    ai_config.WALL_HIT_REWARD = initial_wall_hit_reward
    ai_config.USE_PROXIMITY_REWARD = initial_use_proximity_reward
    ai_config.PROXIMITY_REWARD_FACTOR = initial_proximity_reward_factor
    ai_config.PROXIMITY_PENALTY_FACTOR = initial_proximity_penalty_factor
    ai_config.MAX_PROXIMITY_REWARD = initial_max_proximity_reward
    ai_config.SHOW_OPTIMAL_POINTS_GUI = False
    game_config.GAME_SPEED_MULTIPLIER = initial_game_speed_multiplier
    game_config.FPS = initial_fps

    total_episodes_count += phase1_episodes
    all_phase_data.append(
        {
            "name": "Hit Ball",
            "rewards": phase1_rewards,
            "wins": phase1_wins,
            "episodes": phase1_episodes,
            "objective": "Ball contact",
        }
    )

    # Save Phase 1 model
    if args.save_phase_models:
        phase1_path = Path(args.output_dir) / f"{args.model_prefix}_phase1_hit.pth"
        agent.save_model(str(phase1_path))
        print(f"Phase 1 model saved: {phase1_path}")

    # Phase 2: Learn to return the ball - vs defensive (stays in position)
    print("\n🏓 PHASE 2: Learning to Return the Ball")
    print("Objective: Return ball to opponent's side")
    print("Opponent: Defensive AI (predictable positioning)")

    initial_score_reward = ai_config.SCORE_REWARD
    initial_lose_penalty = ai_config.LOSE_PENALTY
    initial_wall_hit_reward = ai_config.WALL_HIT_REWARD
    ai_config.SCORE_REWARD = 0.5
    ai_config.LOSE_PENALTY = -0.5
    ai_config.WALL_HIT_REWARD = 1  # Increase reward for hitting wall to encourage contact

    # Reduce exploration slightly for phase 2
    if args.progressive_epsilon_reduction > 0:
        agent.epsilon = max(agent.epsilon * args.progressive_epsilon_reduction, agent.epsilon_min)
        print(f"Epsilon reduced to: {agent.epsilon:.3f}")

    phase2_opponent = get_opponent("defensive")
    phase2_rewards, phase2_wins, phase2_episodes = train_phase(
        agent,
        phase2_opponent,
        "Ball return training vs Defensive AI",
        args.progressive_episodes_per_phase,
        training_manager,
        args,
        phase_num=2,
        total_phases=4,
        start_episode=total_episodes_count,
    )
    ai_config.SCORE_REWARD = initial_score_reward
    ai_config.LOSE_PENALTY = initial_lose_penalty
    ai_config.WALL_HIT_REWARD = initial_wall_hit_reward

    total_episodes_count += phase2_episodes
    all_phase_data.append(
        {
            "name": "Return Ball",
            "rewards": phase2_rewards,
            "wins": phase2_wins,
            "episodes": phase2_episodes,
            "objective": "Ball return",
        }
    )

    # Save Phase 2 model
    if args.save_phase_models:
        phase2_path = Path(args.output_dir) / f"{args.model_prefix}_phase2_return.pth"
        agent.save_model(str(phase2_path))
        print(f"Phase 2 model saved: {phase2_path}")

    # Phase 3: Play against intelligent opponent - vs predictive (no bonuses)
    print("\n🧠 PHASE 3: Playing vs Intelligent AI")
    print("Objective: Compete against smart opponent")
    print("Opponent: Defensive AI (anticipates ball trajectory)")
    print("Bonuses: DISABLED for focused strategic learning")

    args.ball_direction = initial_ball_direction

    # Further reduce exploration for phase 3
    if args.progressive_epsilon_reduction > 0:
        agent.epsilon = max(agent.epsilon * args.progressive_epsilon_reduction, agent.epsilon_min)
        print(f"Epsilon reduced to: {agent.epsilon:.3f}")

    phase3_opponent = get_opponent("defensive")
    phase3_rewards, phase3_wins, phase3_episodes = train_phase(
        agent,
        phase3_opponent,
        "Strategic play vs Defensive AI (no bonuses)",
        args.progressive_episodes_per_phase,
        training_manager,
        args,
        phase_num=3,
        total_phases=4,
        start_episode=total_episodes_count,
    )

    total_episodes_count += phase3_episodes
    all_phase_data.append(
        {
            "name": "Strategic Play",
            "rewards": phase3_rewards,
            "wins": phase3_wins,
            "episodes": phase3_episodes,
            "objective": "Smart competition",
        }
    )

    # Save Phase 3 model
    if args.save_phase_models:
        phase3_path = Path(args.output_dir) / f"{args.model_prefix}_phase3_strategic.pth"
        agent.save_model(str(phase3_path))
        print(f"Phase 3 model saved: {phase3_path}")

    # Phase 4: Master the game with bonuses - vs predictive with bonuses
    print("\n🏆 PHASE 4: Mastering with Bonuses")
    print("Objective: Master complete game with all features")
    print("Opponent: Aggressive AI with bonuses enabled")
    print("Features: Rotating paddles, bonuses, full complexity")
    print("Bonuses: ENABLED for complete game mastery")

    # Enable bonuses for Phase 4
    game_config.BONUSES_ENABLED = True

    # Final exploration reduction for phase 4
    if args.progressive_epsilon_reduction > 0:
        agent.epsilon = max(agent.epsilon * args.progressive_epsilon_reduction, agent.epsilon_min)
        print(f"Final epsilon: {agent.epsilon:.3f}")

    phase4_opponent = get_opponent("aggressive")
    phase4_rewards, phase4_wins, phase4_episodes = train_phase(
        agent,
        phase4_opponent,
        "Master training vs Aggressive AI + Bonuses",
        args.progressive_episodes_per_phase,
        training_manager,
        args,
        phase_num=4,
        total_phases=4,
        start_episode=total_episodes_count,
    )

    total_episodes_count += phase4_episodes
    all_phase_data.append(
        {
            "name": "Master Play",
            "rewards": phase4_rewards,
            "wins": phase4_wins,
            "episodes": phase4_episodes,
            "objective": "Full mastery",
        }
    )

    # Restore original bonuses configuration
    game_config.BONUSES_ENABLED = original_bonuses_enabled
    print(f"Bonuses configuration restored to: {original_bonuses_enabled}")

    # Save final model
    final_model_path = Path(args.output_dir) / f"{args.model_prefix}_progressive_final.pth"
    agent.save_model(str(final_model_path))
    print(f"Final progressive model saved: {final_model_path}")

    # Create comprehensive training plots
    if args.save_plots:
        create_progressive_training_plots(all_phase_data, args)

    # Cleanup training manager
    training_manager.cleanup()

    maybe_save_checkpoint_evaluation(
        final_model_path,
        args,
        final_model_training_context(
            args,
            label="progressive_final",
            total_episodes=total_episodes_count,
            rewards=phase1_rewards + phase2_rewards + phase3_rewards + phase4_rewards,
            wins=phase1_wins + phase2_wins + phase3_wins + phase4_wins,
        ),
    )

    # Calculate results
    final_avg = (
        np.mean(phase4_rewards[-100:])
        if len(phase4_rewards) >= 100
        else np.mean(phase4_rewards)
        if phase4_rewards
        else 0
    )
    final_winrate = phase4_wins / phase4_episodes * 100 if phase4_episodes > 0 else 0

    print("\n✨ PROGRESSIVE TRAINING COMPLETED ✨")
    print(f"Total episodes across all phases: {total_episodes_count}")
    print(f"Final performance: {final_avg:.2f} avg reward, {final_winrate:.1f}% win rate")

    return agent, {
        "phase1_avg": np.mean(phase1_rewards[-50:])
        if len(phase1_rewards) >= 50
        else np.mean(phase1_rewards)
        if phase1_rewards
        else 0,
        "phase1_winrate": phase1_wins / phase1_episodes * 100 if phase1_episodes > 0 else 0,
        "phase2_avg": np.mean(phase2_rewards[-50:])
        if len(phase2_rewards) >= 50
        else np.mean(phase2_rewards)
        if phase2_rewards
        else 0,
        "phase2_winrate": phase2_wins / phase2_episodes * 100 if phase2_episodes > 0 else 0,
        "phase3_avg": np.mean(phase3_rewards[-50:])
        if len(phase3_rewards) >= 50
        else np.mean(phase3_rewards)
        if phase3_rewards
        else 0,
        "phase3_winrate": phase3_wins / phase3_episodes * 100 if phase3_episodes > 0 else 0,
        "phase4_avg": final_avg,
        "phase4_winrate": final_winrate,
        "total_episodes": total_episodes_count,
        "final_model_path": str(final_model_path),
        "all_phase_data": all_phase_data,
    }


def train_dual_scale(agent, args):
    """
    Advanced dual-scale training: tactical + strategic learning optimized for Pong.

    Features:
    - Tactical learning: Frequent updates with conservative LR for immediate feedback
    - Strategic learning: Episode-end updates with aggressive LR for long-term planning
    - Hybrid reward calculation for both time scales
    - Enhanced performance monitoring
    """
    print("=== ADVANCED DUAL-SCALE TRAINING ===")
    print("🎯 Tactical Learning: Immediate feedback, ball tracking, positioning")
    print("🧠 Strategic Learning: Long-term planning, match outcomes, sequences")

    # Get ball configuration
    ball_direction, ball_angle = get_ball_config_from_args(args)
    training_manager = TrainingManager(
        headless=args.headless,
        fast_gui=args.fast_gui,
        initial_ball_direction=ball_direction,
        initial_ball_angle=ball_angle,
    )

    # Create opponent based on args
    opponent = get_opponent(args.opponent if hasattr(args, "opponent") else "follow_ball")
    print(f"Training opponent: {opponent.name}")

    # Ensure dual-scale training is enabled
    if not agent.enable_dual_scale_training:
        print("⚠️  Warning: Agent was not configured for dual-scale training!")
        print("    Enabling dual-scale training now...")
        agent.enable_dual_scale_training = True
        # Initialize dual-scale components if needed
        agent._initialize_dual_scale_training()

    # Display training configuration
    stats = agent.get_training_stats()
    print("\n📊 Training Configuration:")
    print(f"   Dual-scale training: {stats['dual_scale_training']}")
    print(f"   Tactical frequency: every {stats['tactical_train_frequency']} steps")
    print(f"   Tactical LR: {stats['tactical_optimizer_lr']:.6f}")
    print(f"   Strategic LR: {stats['strategic_optimizer_lr']:.6f}")
    print(f"   Training mode: {stats['training_mode']}")
    print(f"   Memory size: {stats['memory_size']}")
    print(f"   Min replay size: {stats['min_replay_size']}")

    # Training phase
    episodes = args.total_episodes if hasattr(args, "total_episodes") else 1000
    rewards, wins, final_episodes = train_phase(
        agent,
        opponent,
        f"Dual-scale training vs {opponent.name}",
        episodes,
        training_manager,
        args,
        phase_num=1,
        total_phases=1,
    )

    # Save final model
    final_model_path = Path(args.output_dir) / f"{args.model_prefix}_dual_scale_final.pth"
    agent.save_model(str(final_model_path))
    print(f"Final dual-scale model saved: {final_model_path}")

    # Create specialized dual-scale plots
    if args.save_plots:
        create_dual_scale_plots(
            rewards, wins / final_episodes * 100 if final_episodes > 0 else 0, agent, args
        )

    # Cleanup training manager
    training_manager.cleanup()

    maybe_save_checkpoint_evaluation(
        final_model_path,
        args,
        final_model_training_context(
            args,
            label="dual_scale_final",
            total_episodes=final_episodes,
            rewards=rewards,
            wins=wins,
        ),
    )

    # Final statistics
    final_stats = agent.get_training_stats()
    print("\n✨ DUAL-SCALE TRAINING COMPLETED ✨")
    print("📈 Final Performance:")
    print(f"   Episodes: {final_episodes}")
    print(
        f"   Average reward: {np.mean(rewards[-100:]) if len(rewards) >= 100 else np.mean(rewards) if rewards else 0:.2f}"
    )
    print(f"   Win rate: {wins / final_episodes * 100 if final_episodes > 0 else 0:.1f}%")
    print(f"   Tactical steps: {final_stats['tactical_step_count']}")
    print(f"   Memory utilization: {final_stats['memory_size']}")

    return agent, {
        "avg_reward": np.mean(rewards[-100:])
        if len(rewards) >= 100
        else np.mean(rewards)
        if rewards
        else 0,
        "win_rate": wins / final_episodes * 100 if final_episodes > 0 else 0,
        "total_episodes": final_episodes,
        "final_model_path": str(final_model_path),
        "tactical_steps": final_stats["tactical_step_count"],
        "dual_scale_stats": final_stats,
    }


def create_dual_scale_plots(rewards, win_rate, agent, args):
    """Create specialized plots for dual-scale training analysis"""
    plt = get_pyplot()
    fig, ((ax1, ax2), (ax3, ax4)) = plt.subplots(2, 2, figsize=(16, 12))

    # Plot 1: Reward progression with tactical annotations
    episodes = range(len(rewards))
    ax1.plot(episodes, rewards, "b-", alpha=0.3, linewidth=0.8, label="Episode rewards")

    # Moving average
    window = min(50, len(rewards) // 4) if rewards else 1
    if len(rewards) >= window and window > 1:
        rolling = np.convolve(rewards, np.ones(window) / window, mode="valid")
        ax1.plot(
            range(window - 1, len(rewards)),
            rolling,
            "b-",
            linewidth=2,
            label=f"Moving avg ({window})",
        )

    # Add tactical training frequency markers
    stats = agent.get_training_stats()
    tactical_freq = stats.get("tactical_train_frequency", 10)

    # Mark every tactical training point
    tactical_episodes = list(range(0, len(rewards), tactical_freq))
    if tactical_episodes:
        tactical_rewards = [rewards[i] for i in tactical_episodes if i < len(rewards)]
        ax1.scatter(
            tactical_episodes[: len(tactical_rewards)],
            tactical_rewards,
            color="red",
            alpha=0.6,
            s=10,
            label=f"Tactical updates (every {tactical_freq})",
        )

    ax1.set_title("Dual-Scale Training: Reward Progression")
    ax1.set_xlabel("Episode")
    ax1.set_ylabel("Reward")
    ax1.legend()
    ax1.grid(True, alpha=0.3)

    # Plot 2: Training mode comparison
    if rewards:
        recent_performance = np.mean(rewards[-50:]) if len(rewards) >= 50 else np.mean(rewards)
        early_performance = np.mean(rewards[:50]) if len(rewards) >= 50 else np.mean(rewards)

        performance_data = {
            "Early Training\n(First 50 ep)": early_performance,
            "Recent Training\n(Last 50 ep)": recent_performance,
            "Overall Average": np.mean(rewards),
        }

        bars = ax2.bar(
            performance_data.keys(),
            performance_data.values(),
            color=["lightblue", "lightgreen", "orange"],
            alpha=0.7,
        )
        ax2.set_title("Performance Evolution")
        ax2.set_ylabel("Average Reward")

        # Add values on bars
        for bar, value in zip(bars, performance_data.values(), strict=False):
            height = bar.get_height()
            ax2.text(
                bar.get_x() + bar.get_width() / 2.0,
                height + height * 0.01,
                f"{value:.2f}",
                ha="center",
                va="bottom",
            )
        ax2.grid(True, alpha=0.3)

    # Plot 3: Dual-scale statistics
    dual_stats = {
        "Tactical Steps": stats.get("tactical_step_count", 0),
        "Total Episodes": len(rewards),
        "Memory Size": stats.get("memory_size", 0),
        "Episode Buffer": stats.get("episode_buffer_size", 0),
    }

    colors = ["skyblue", "lightcoral", "lightgreen", "wheat"]
    bars = ax3.bar(dual_stats.keys(), dual_stats.values(), color=colors, alpha=0.7)
    ax3.set_title("Dual-Scale Training Statistics")
    ax3.set_ylabel("Count")
    ax3.tick_params(axis="x", rotation=45)

    for bar, value in zip(bars, dual_stats.values(), strict=False):
        height = bar.get_height()
        ax3.text(
            bar.get_x() + bar.get_width() / 2.0,
            height + height * 0.01,
            f"{int(value)}",
            ha="center",
            va="bottom",
        )
    ax3.grid(True, alpha=0.3)

    # Plot 4: Learning rate comparison
    lr_data = {
        "Tactical LR": stats.get("tactical_optimizer_lr", 0),
        "Strategic LR": stats.get("strategic_optimizer_lr", 0),
        "Base LR": stats.get("current_lr", 0),
    }

    bars = ax4.bar(lr_data.keys(), lr_data.values(), color=["red", "blue", "green"], alpha=0.7)
    ax4.set_title("Learning Rate Configuration")
    ax4.set_ylabel("Learning Rate")
    ax4.set_yscale("log")  # Log scale for better visualization

    for bar, value in zip(bars, lr_data.values(), strict=False):
        height = bar.get_height()
        ax4.text(
            bar.get_x() + bar.get_width() / 2.0,
            height * 1.1,
            f"{value:.6f}",
            ha="center",
            va="bottom",
            rotation=45,
        )
    ax4.grid(True, alpha=0.3)

    plt.tight_layout()

    plots_path = Path(args.output_dir) / f"{args.model_prefix}_dual_scale_training.png"
    plt.savefig(plots_path, dpi=150, bbox_inches="tight")
    plt.close()
    print(f"📊 Dual-scale training plots saved: {plots_path}")


def create_training_plots(phase1_rewards, phase2_rewards, phase1_winrate, phase2_winrate, args):
    """Create training progress plots"""
    plt = get_pyplot()

    fig, ((ax1, ax2), (ax3, ax4)) = plt.subplots(2, 2, figsize=(15, 10))

    # Rewards per episode
    all_rewards = phase1_rewards + phase2_rewards
    episodes = range(len(all_rewards))

    ax1.plot(
        episodes[: len(phase1_rewards)], phase1_rewards, "b-", alpha=0.3, label="Phase 1 (basic)"
    )
    ax1.plot(
        range(len(phase1_rewards), len(all_rewards)),
        phase2_rewards,
        "r-",
        alpha=0.3,
        label="Phase 2 (advanced)",
    )

    # Moving averages
    window = min(50, len(phase1_rewards) // 4) if phase1_rewards else 1
    if len(phase1_rewards) >= window and window > 1:
        rolling1 = np.convolve(phase1_rewards, np.ones(window) / window, mode="valid")
        ax1.plot(range(window - 1, len(phase1_rewards)), rolling1, "b-", linewidth=2)

    if len(phase2_rewards) >= window and window > 1:
        rolling2 = np.convolve(phase2_rewards, np.ones(window) / window, mode="valid")
        ax1.plot(
            range(len(phase1_rewards) + window - 1, len(all_rewards)), rolling2, "r-", linewidth=2
        )

    ax1.axvline(
        x=len(phase1_rewards), color="k", linestyle="--", alpha=0.5, label="Phase transition"
    )
    ax1.set_title("Reward Progress")
    ax1.set_xlabel("Episode")
    ax1.set_ylabel("Reward")
    ax1.legend()
    ax1.grid(True, alpha=0.3)

    # Reward distribution by phase
    if phase1_rewards:
        ax2.hist(
            phase1_rewards,
            bins=30,
            alpha=0.7,
            label=f"Phase 1 (μ={np.mean(phase1_rewards):.2f})",
            color="blue",
        )
    if phase2_rewards:
        ax2.hist(
            phase2_rewards,
            bins=30,
            alpha=0.7,
            label=f"Phase 2 (μ={np.mean(phase2_rewards):.2f})",
            color="red",
        )
    ax2.set_title("Reward Distribution")
    ax2.set_xlabel("Reward")
    ax2.set_ylabel("Frequency")
    ax2.legend()
    ax2.grid(True, alpha=0.3)

    # Win rates
    phases = ["Phase 1\n(basic)", "Phase 2\n(advanced)"]
    winrates = [phase1_winrate, phase2_winrate]
    colors = ["blue", "red"]

    bars = ax3.bar(phases, winrates, color=colors, alpha=0.7)
    ax3.set_title("Win Rate by Phase")
    ax3.set_ylabel("Win Rate (%)")
    ax3.set_ylim(0, 100)

    # Add values on bars
    for bar, rate in zip(bars, winrates, strict=False):
        height = bar.get_height()
        ax3.text(
            bar.get_x() + bar.get_width() / 2.0,
            height + 1,
            f"{rate:.1f}%",
            ha="center",
            va="bottom",
        )
    ax3.grid(True, alpha=0.3)

    # Statistics comparison
    stats_data = {}
    if phase1_rewards:
        stats_data["Avg Reward"] = [
            np.mean(phase1_rewards),
            np.mean(phase2_rewards) if phase2_rewards else 0,
        ]
        stats_data["Max Reward"] = [
            np.max(phase1_rewards),
            np.max(phase2_rewards) if phase2_rewards else 0,
        ]
        stats_data["Std Dev"] = [
            np.std(phase1_rewards),
            np.std(phase2_rewards) if phase2_rewards else 0,
        ]

    if stats_data:
        x = np.arange(len(phases))
        width = 0.25

        for i, (stat, values) in enumerate(stats_data.items()):
            ax4.bar(x + i * width, values, width, label=stat, alpha=0.7)

        ax4.set_title("Statistics Comparison")
        ax4.set_xlabel("Phase")
        ax4.set_ylabel("Value")
        ax4.set_xticks(x + width)
        ax4.set_xticklabels(phases)
        ax4.legend()
        ax4.grid(True, alpha=0.3)

    plt.tight_layout()

    plots_path = Path(args.output_dir) / f"{args.model_prefix}_curriculum_training.png"
    plt.savefig(plots_path, dpi=150, bbox_inches="tight")
    plt.close()
    print(f"📊 Training plots saved: {plots_path}")


def create_progressive_training_plots(all_phase_data, args):
    """Create comprehensive training plots for 4-phase progressive learning"""
    plt = get_pyplot()

    fig, ((ax1, ax2), (ax3, ax4)) = plt.subplots(2, 2, figsize=(16, 12))

    # Combine all rewards for continuous plot
    all_rewards = []
    phase_boundaries = [0]
    phase_colors = ["blue", "green", "orange", "red"]
    phase_labels = []

    for i, phase_data in enumerate(all_phase_data):
        all_rewards.extend(phase_data["rewards"])
        phase_boundaries.append(len(all_rewards))
        phase_labels.append(f"Phase {i + 1}: {phase_data['name']}")

    episodes = range(len(all_rewards))

    # Plot 1: Rewards progression across all phases
    ax1.plot(episodes, all_rewards, "k-", alpha=0.3, linewidth=0.5, label="Episode rewards")

    # Add phase boundaries and colors
    for i, (start, end) in enumerate(
        zip(phase_boundaries[:-1], phase_boundaries[1:], strict=False)
    ):
        phase_rewards = all_rewards[start:end]
        # phase_episodes = range(start, end)

        if phase_rewards:
            # Moving average for each phase
            window = min(20, len(phase_rewards) // 3) if len(phase_rewards) > 3 else 1
            if len(phase_rewards) >= window and window > 1:
                rolling = np.convolve(phase_rewards, np.ones(window) / window, mode="valid")
                ax1.plot(
                    range(start + window - 1, end),
                    rolling,
                    color=phase_colors[i],
                    linewidth=2,
                    label=phase_labels[i],
                )

            # Phase boundaries
            if i > 0:
                ax1.axvline(x=start, color="gray", linestyle="--", alpha=0.6)

    ax1.set_title("Progressive Learning: Reward Evolution Across 4 Phases")
    ax1.set_xlabel("Episode")
    ax1.set_ylabel("Reward")
    ax1.legend(bbox_to_anchor=(1.05, 1), loc="upper left")
    ax1.grid(True, alpha=0.3)

    # Plot 2: Win rates by phase
    phase_names = [data["name"] for data in all_phase_data]
    win_rates = []
    avg_rewards = []

    for data in all_phase_data:
        win_rate = data["wins"] / data["episodes"] * 100 if data["episodes"] > 0 else 0
        win_rates.append(win_rate)
        avg_reward = np.mean(data["rewards"]) if data["rewards"] else 0
        avg_rewards.append(avg_reward)

    bars = ax2.bar(phase_names, win_rates, color=phase_colors, alpha=0.7)
    ax2.set_title("Win Rate by Training Phase")
    ax2.set_ylabel("Win Rate (%)")
    ax2.set_ylim(0, 100)
    ax2.tick_params(axis="x", rotation=45)

    # Add values on bars
    for bar, rate in zip(bars, win_rates, strict=False):
        height = bar.get_height()
        ax2.text(
            bar.get_x() + bar.get_width() / 2.0,
            height + 1,
            f"{rate:.1f}%",
            ha="center",
            va="bottom",
            fontsize=10,
        )
    ax2.grid(True, alpha=0.3)

    # Plot 3: Average rewards by phase
    bars = ax3.bar(phase_names, avg_rewards, color=phase_colors, alpha=0.7)
    ax3.set_title("Average Reward by Training Phase")
    ax3.set_ylabel("Average Reward")
    ax3.tick_params(axis="x", rotation=45)

    # Add values on bars
    for bar, reward in zip(bars, avg_rewards, strict=False):
        height = bar.get_height()
        ax3.text(
            bar.get_x() + bar.get_width() / 2.0,
            height + height * 0.01,
            f"{reward:.2f}",
            ha="center",
            va="bottom",
            fontsize=10,
        )
    ax3.grid(True, alpha=0.3)

    # Plot 4: Training progression metrics
    # episodes_per_phase = [data["episodes"] for data in all_phase_data]
    # std_rewards = [np.std(data["rewards"]) if data["rewards"] else 0 for data in all_phase_data]

    x = np.arange(len(phase_names))
    # width = 0.35

    ax4_twin = ax4.twinx()

    # bars1 = ax4.bar(
    #     x - width / 2, episodes_per_phase, width, label="Episodes", color="lightblue", alpha=0.7
    # )
    # bars2 = ax4_twin.bar(
    #     x + width / 2, std_rewards, width, label="Reward Std Dev", color="lightcoral", alpha=0.7
    # )

    ax4.set_title("Training Metrics by Phase")
    ax4.set_xlabel("Training Phase")
    ax4.set_ylabel("Episodes", color="blue")
    ax4_twin.set_ylabel("Reward Standard Deviation", color="red")
    ax4.set_xticks(x)
    ax4.set_xticklabels(phase_names, rotation=45)

    # Add legends
    ax4.legend(loc="upper left")
    ax4_twin.legend(loc="upper right")
    ax4.grid(True, alpha=0.3)

    plt.tight_layout()

    plots_path = Path(args.output_dir) / f"{args.model_prefix}_progressive_training.png"
    plt.savefig(plots_path, dpi=150, bbox_inches="tight")
    plt.close()
    print(f"📊 Progressive training plots saved: {plots_path}")

    # Create detailed phase breakdown plot


def create_phase_breakdown_plot(all_phase_data, args):
    """Create detailed breakdown plot for each phase"""
    plt = get_pyplot()

    fig, axes = plt.subplots(2, 2, figsize=(16, 12))
    axes = axes.flatten()

    phase_colors = ["blue", "green", "orange", "red"]

    for i, (phase_data, ax) in enumerate(zip(all_phase_data, axes, strict=False)):
        rewards = phase_data["rewards"]
        if not rewards:
            ax.text(0.5, 0.5, "No data available", ha="center", va="center", transform=ax.transAxes)
            ax.set_title(f"Phase {i + 1}: {phase_data['name']}")
            continue

        episodes = range(len(rewards))

        # Plot rewards
        ax.plot(episodes, rewards, color=phase_colors[i], alpha=0.5, linewidth=0.8)

        # Moving average
        window = min(20, len(rewards) // 3) if len(rewards) > 3 else 1
        if len(rewards) >= window and window > 1:
            rolling = np.convolve(rewards, np.ones(window) / window, mode="valid")
            ax.plot(
                range(window - 1, len(rewards)),
                rolling,
                color=phase_colors[i],
                linewidth=2,
                label=f"Avg ({window} ep)",
            )

        # Statistics
        mean_reward = np.mean(rewards)
        ax.axhline(
            y=mean_reward, color="red", linestyle="--", alpha=0.7, label=f"Mean: {mean_reward:.2f}"
        )

        win_rate = phase_data["wins"] / phase_data["episodes"] * 100

        ax.set_title(
            f"Phase {i + 1}: {phase_data['name']}\n"
            f"Win Rate: {win_rate:.1f}% | Objective: {phase_data['objective']}"
        )
        ax.set_xlabel("Episode")
        ax.set_ylabel("Reward")
        ax.legend()
        ax.grid(True, alpha=0.3)

    plt.tight_layout()

    breakdown_path = Path(args.output_dir) / f"{args.model_prefix}_phase_breakdown.png"
    plt.savefig(breakdown_path, dpi=150, bbox_inches="tight")
    plt.close()
    print(f"📊 Phase breakdown plots saved: {breakdown_path}")


def create_single_phase_plots(rewards, win_rate, args):
    """Create plots for single phase training"""
    plt = get_pyplot()

    fig, ((ax1, ax2), (ax3, ax4)) = plt.subplots(2, 2, figsize=(15, 10))

    # Rewards per episode
    episodes = range(len(rewards))
    ax1.plot(episodes, rewards, "b-", alpha=0.3, label="Episode rewards")

    # Moving average
    window = min(50, len(rewards) // 4) if rewards else 1
    if len(rewards) >= window and window > 1:
        rolling = np.convolve(rewards, np.ones(window) / window, mode="valid")
        ax1.plot(
            range(window - 1, len(rewards)),
            rolling,
            "b-",
            linewidth=2,
            label=f"Moving avg ({window})",
        )

    ax1.set_title("Reward Progress")
    ax1.set_xlabel("Episode")
    ax1.set_ylabel("Reward")
    ax1.legend()
    ax1.grid(True, alpha=0.3)

    # Reward distribution
    if rewards:
        ax2.hist(rewards, bins=30, alpha=0.7, color="blue")
        ax2.axvline(
            np.mean(rewards), color="red", linestyle="--", label=f"Mean: {np.mean(rewards):.2f}"
        )
        ax2.set_title("Reward Distribution")
        ax2.set_xlabel("Reward")
        ax2.set_ylabel("Frequency")
        ax2.legend()
        ax2.grid(True, alpha=0.3)

    # Win rate display
    ax3.bar(["Training"], [win_rate], color="blue", alpha=0.7)
    ax3.set_title("Overall Win Rate")
    ax3.set_ylabel("Win Rate (%)")
    ax3.set_ylim(0, 100)
    ax3.text(0, win_rate + 2, f"{win_rate:.1f}%", ha="center", va="bottom")
    ax3.grid(True, alpha=0.3)

    # Training statistics
    if rewards:
        stats = {
            "Mean": np.mean(rewards),
            "Std Dev": np.std(rewards),
            "Min": np.min(rewards),
            "Max": np.max(rewards),
        }

        bars = ax4.bar(stats.keys(), stats.values(), alpha=0.7)
        ax4.set_title("Training Statistics")
        ax4.set_ylabel("Value")

        # Add values on bars
        for bar, value in zip(bars, stats.values(), strict=False):
            height = bar.get_height()
            ax4.text(
                bar.get_x() + bar.get_width() / 2.0,
                height + height * 0.01,
                f"{value:.2f}",
                ha="center",
                va="bottom",
            )
        ax4.grid(True, alpha=0.3)

    plt.tight_layout()

    plots_path = Path(args.output_dir) / f"{args.model_prefix}_training.png"
    plt.savefig(plots_path, dpi=150, bbox_inches="tight")
    plt.close()
    print(f"📊 Training plots saved: {plots_path}")
    """Create plots for single phase training"""

    fig, ((ax1, ax2), (ax3, ax4)) = plt.subplots(2, 2, figsize=(15, 10))

    # Rewards per episode
    episodes = range(len(rewards))
    ax1.plot(episodes, rewards, "b-", alpha=0.3, label="Episode rewards")

    # Moving average
    window = min(50, len(rewards) // 4) if rewards else 1
    if len(rewards) >= window and window > 1:
        rolling = np.convolve(rewards, np.ones(window) / window, mode="valid")
        ax1.plot(
            range(window - 1, len(rewards)),
            rolling,
            "b-",
            linewidth=2,
            label=f"Moving avg ({window})",
        )

    ax1.set_title("Reward Progress")
    ax1.set_xlabel("Episode")
    ax1.set_ylabel("Reward")
    ax1.legend()
    ax1.grid(True, alpha=0.3)

    # Reward distribution
    if rewards:
        ax2.hist(rewards, bins=30, alpha=0.7, color="blue")
        ax2.axvline(
            np.mean(rewards), color="red", linestyle="--", label=f"Mean: {np.mean(rewards):.2f}"
        )
        ax2.set_title("Reward Distribution")
        ax2.set_xlabel("Reward")
        ax2.set_ylabel("Frequency")
        ax2.legend()
        ax2.grid(True, alpha=0.3)

    # Win rate display
    ax3.bar(["Training"], [win_rate], color="blue", alpha=0.7)
    ax3.set_title("Overall Win Rate")
    ax3.set_ylabel("Win Rate (%)")
    ax3.set_ylim(0, 100)
    ax3.text(0, win_rate + 2, f"{win_rate:.1f}%", ha="center", va="bottom")
    ax3.grid(True, alpha=0.3)

    # Training statistics
    if rewards:
        stats = {
            "Mean": np.mean(rewards),
            "Std Dev": np.std(rewards),
            "Min": np.min(rewards),
            "Max": np.max(rewards),
        }

        bars = ax4.bar(stats.keys(), stats.values(), alpha=0.7)
        ax4.set_title("Training Statistics")
        ax4.set_ylabel("Value")

        # Add values on bars
        for bar, value in zip(bars, stats.values(), strict=False):
            height = bar.get_height()
            ax4.text(
                bar.get_x() + bar.get_width() / 2.0,
                height + height * 0.01,
                f"{value:.2f}",
                ha="center",
                va="bottom",
            )
        ax4.grid(True, alpha=0.3)

    plt.tight_layout()

    plots_path = Path(args.output_dir) / f"{args.model_prefix}_training.png"
    plt.savefig(plots_path, dpi=150, bbox_inches="tight")
    plt.close()
    print(f"📊 Training plots saved: {plots_path}")


def evaluate_final_performance(agent, args):
    """Final performance evaluation of the agent"""
    print(f"\n🎯 FINAL EVALUATION ({args.eval_episodes} episodes)")

    original_max_score = game_config.MAX_SCORE
    training_manager = None

    # Disable learning for evaluation while honoring explicit evaluation exploration.
    was_training = agent.training_enabled
    was_exploring = agent.exploration_enabled
    original_epsilon = agent.epsilon

    try:
        game_config.MAX_SCORE = args.eval_max_score

        # Test against different opponents
        opponents = {
            "follow_ball": get_opponent("follow_ball"),
            "defensive": get_opponent("defensive"),
            "aggressive": get_opponent("aggressive"),
            "random": get_opponent("random"),
            "predictive": get_opponent("predictive"),
        }

        # Get ball configuration
        ball_direction, ball_angle = get_ball_config_from_args(args)
        training_manager = TrainingManager(
            headless=args.headless,
            initial_ball_direction=ball_direction,
            initial_ball_angle=ball_angle,
        )

        agent.set_training_mode(False)
        agent.epsilon = args.eval_epsilon
        agent.set_exploration_mode(args.eval_epsilon > 0)

        results = {}

        for opponent_name, opponent in opponents.items():
            print(f"\nAgainst {opponent_name}:")
            wins = 0
            rewards = []

            for _ in range(args.eval_episodes):
                agent.on_episode_start()
                episode_stats = training_manager.train_episode(
                    agent, opponent, max_steps=args.max_steps_per_episode
                )

                if episode_stats.get("winner") == 1:
                    wins += 1
                rewards.append(episode_stats["total_reward_p1"])

            win_rate = wins / args.eval_episodes * 100
            avg_reward = np.mean(rewards)

            results[opponent_name] = {
                "win_rate": win_rate,
                "avg_reward": avg_reward,
                "std_reward": np.std(rewards),
            }

            print(f"  Win rate: {win_rate:.1f}%")
            print(f"  Average reward: {avg_reward:.2f} ± {np.std(rewards):.2f}")

        # Save evaluation results
        if args.save_eval_results:
            eval_path = Path(args.output_dir) / f"{args.model_prefix}_evaluation.json"
            with open(eval_path, "w") as f:
                json.dump(results, f, indent=2)
            print(f"Evaluation results saved: {eval_path}")

        return results

    finally:
        agent.epsilon = original_epsilon
        agent.set_training_mode(was_training)
        agent.set_exploration_mode(was_exploring)

        # Cleanup training manager
        if training_manager is not None:
            training_manager.cleanup()

        game_config.MAX_SCORE = original_max_score


def parse_arguments(argv: list[str] | None = None):
    """Parse command line arguments"""
    parser = argparse.ArgumentParser(
        description="Advanced DQN Training Script with Comprehensive Configuration",
        formatter_class=argparse.ArgumentDefaultsHelpFormatter,
    )

    # Training mode
    parser.add_argument(
        "--mode",
        choices=["curriculum", "single", "continue", "progressive", "dual_scale"],
        default="curriculum",
        help="Training mode: curriculum learning (2 phases), single phase, continue from checkpoint, progressive (4 phases), or dual_scale (advanced tactical+strategic learning)",
    )

    # Model and checkpoint management
    parser.add_argument(
        "--output_dir", type=str, default="models", help="Directory to save models and outputs"
    )
    parser.add_argument(
        "--model_prefix", type=str, default="dqn_optimized", help="Prefix for saved model files"
    )
    parser.add_argument(
        "--load_model",
        type=str,
        default=None,
        help="Path to existing model to load and continue training",
    )
    parser.add_argument(
        "--load_checkpoint",
        type=str,
        default=None,
        help="Path to checkpoint file to resume training",
    )

    # Checkpoint settings
    parser.add_argument(
        "--checkpoint_interval",
        type=int,
        default=500,
        help="Save checkpoint every N episodes (0 to disable)",
    )
    parser.add_argument(
        "--checkpoint_dir",
        type=str,
        default=None,
        help="Directory to save training checkpoints",
    )
    parser.add_argument(
        "--save_phase_models", action="store_true", help="Save models after each training phase"
    )

    # DQN hyperparameters
    parser.add_argument("--state_size", type=int, default=32, help="Size of the state space")
    parser.add_argument("--action_size", type=int, default=9, help="Size of the action space")
    parser.add_argument(
        "--learning_rate", type=float, default=0.0003, help="Learning rate for the neural network"
    )
    parser.add_argument(
        "--gamma", type=float, default=0.99, help="Discount factor for future rewards"
    )
    parser.add_argument("--epsilon_start", type=float, default=1.0, help="Initial exploration rate")
    parser.add_argument("--epsilon_decay", type=float, default=0.998, help="Exploration decay rate")
    parser.add_argument("--epsilon_min", type=float, default=0.05, help="Minimum exploration rate")
    parser.add_argument("--batch_size", type=int, default=64, help="Batch size for training")
    parser.add_argument(
        "--memory_size", type=int, default=20000, help="Size of the replay memory buffer"
    )
    parser.add_argument(
        "--tau", type=float, default=0.003, help="Soft update parameter for target network"
    )
    parser.add_argument(
        "--use_prioritized_replay", action="store_true", help="Use prioritized experience replay"
    )

    # Training configuration
    parser.add_argument(
        "--episodes_per_phase",
        type=int,
        default=200,
        help="Number of episodes per training phase (curriculum mode)",
    )
    parser.add_argument(
        "--total_episodes", type=int, default=1000, help="Total number of episodes (single mode)"
    )
    parser.add_argument(
        "--max_steps_per_episode", type=int, default=1000, help="Maximum steps per episode"
    )
    parser.add_argument(
        "--training_max_score",
        type=int,
        default=1,
        help="Score needed to win during training when mixed fine-tuning is disabled",
    )
    parser.add_argument(
        "--mixed_opponents",
        "--mixed_opponent_finetune",
        dest="mixed_opponents",
        action="store_true",
        help="Enable mixed-opponent fine-tuning with weighted opponent and MAX_SCORE sampling",
    )
    parser.add_argument(
        "--opponent_mix",
        type=str,
        default=DEFAULT_OPPONENT_MIX_CLI,
        help="Comma-separated weighted opponent mix as '<name>:<weight>' entries",
    )
    parser.add_argument(
        "--opponent_mix_seed",
        type=int,
        default=0,
        help="Seed for deterministic mixed-opponent and training target sampling",
    )
    parser.add_argument(
        "--training_score_mix",
        "--training_max_scores",
        dest="training_score_mix",
        type=str,
        default=DEFAULT_TRAINING_SCORE_MIX_CLI,
        help="Comma-separated positive MAX_SCORE values sampled during mixed fine-tuning",
    )
    parser.add_argument(
        "--fine_tune_bonuses",
        "--enable_training_bonuses",
        dest="fine_tune_bonuses",
        action="store_true",
        help="Enable bonuses during mixed-opponent fine-tuning; disabled by default for base-game fine-tuning",
    )
    parser.add_argument(
        "--early_stopping",
        type=int,
        default=0,
        help="Stop training after N episodes without improvement (0 to disable)",
    )

    # Curriculum learning settings
    parser.add_argument(
        "--phase1_opponent",
        type=str,
        default="follow_ball",
        choices=AI_TYPES,
        help="Opponent type for phase 1 training",
    )
    parser.add_argument(
        "--phase2_opponent",
        type=str,
        default="follow_ball",
        choices=AI_TYPES,
        help="Opponent type for phase 2 training",
    )
    parser.add_argument(
        "--phase2_epsilon_reduction",
        action="store_true",
        help="Reduce epsilon for phase 2 training",
    )

    # Progressive learning settings (4 phases)
    parser.add_argument(
        "--progressive_episodes_per_phase",
        type=int,
        default=150,
        help="Number of episodes per phase in progressive mode (4 phases total)",
    )
    parser.add_argument(
        "--progressive_epsilon_reduction",
        type=float,
        default=0.7,
        help="Epsilon reduction factor between progressive phases (multiplied each phase)",
    )
    parser.add_argument(
        "--progressive_early_stopping",
        type=int,
        default=100,
        help="Early stopping threshold for progressive phases (0 to disable)",
    )
    parser.add_argument(
        "--phase1_training_opponent",
        type=str,
        default="dummy",
        choices=["dummy", "training_dummy"],
        help="Opponent type for Phase 1 (ball contact training): 'dummy' (completely still) or 'training_dummy' (minimal movement)",
    )

    # Single phase settings
    parser.add_argument(
        "--opponent",
        type=str,
        default="follow_ball",
        choices=AI_TYPES,
        help="Opponent type for single phase training",
    )

    # Evaluation settings
    parser.add_argument(
        "--evaluate",
        action="store_true",
        help="Perform final evaluation against multiple opponents",
    )
    parser.add_argument(
        "--eval_episodes", type=int, default=100, help="Number of episodes for evaluation"
    )
    parser.add_argument(
        "--eval_epsilon", type=float, default=0.01, help="Exploration rate during evaluation"
    )
    parser.add_argument(
        "--save_eval_results", action="store_true", help="Save evaluation results to JSON file"
    )
    parser.add_argument(
        "--eval_max_score",
        type=int,
        default=1,
        help="Score needed to win during evaluation (default: 1, game default: 11)",
    )
    parser.add_argument(
        "--evaluate_checkpoints",
        action="store_true",
        help="Evaluate each saved training checkpoint and final model, writing JSON next to each .pth",
    )
    parser.add_argument(
        "--checkpoint_eval_opponent",
        dest="checkpoint_eval_opponents",
        action="append",
        choices=[opponent for opponent in AI_TYPES if opponent != "training_dummy"],
        help="Opponent for checkpoint evaluation. May be repeated.",
    )
    parser.add_argument(
        "--checkpoint_eval_opponents",
        dest="checkpoint_eval_opponents_csv",
        default=None,
        help="Comma-separated checkpoint evaluation opponents",
    )
    parser.add_argument(
        "--checkpoint_eval_episodes",
        type=int,
        default=0,
        help="Episodes per opponent for checkpoint evaluation (0 disables unless --evaluate_checkpoints is set)",
    )
    parser.add_argument(
        "--checkpoint_eval_max_steps",
        type=int,
        default=None,
        help="Max steps per checkpoint evaluation episode (defaults to --max_steps_per_episode)",
    )
    parser.add_argument(
        "--checkpoint_eval_seed",
        type=int,
        default=0,
        help="Seed for deterministic checkpoint evaluation",
    )
    parser.add_argument(
        "--checkpoint_eval_epsilon",
        type=float,
        default=0.0,
        help="Exploration epsilon used only during checkpoint evaluation",
    )
    parser.add_argument(
        "--checkpoint_eval_max_score",
        type=int,
        default=None,
        help="Optional MAX_SCORE override used only during checkpoint evaluation",
    )
    parser.add_argument(
        "--checkpoint_eval_include_episodes",
        action="store_true",
        help="Include per-episode rows in checkpoint evaluation JSON",
    )
    checkpoint_bonus_group = parser.add_mutually_exclusive_group()
    checkpoint_bonus_group.add_argument(
        "--checkpoint_eval_bonuses",
        dest="checkpoint_eval_bonuses_enabled",
        action="store_true",
        default=None,
        help="Enable bonuses during checkpoint evaluation",
    )
    checkpoint_bonus_group.add_argument(
        "--checkpoint_eval_no_bonuses",
        dest="checkpoint_eval_bonuses_enabled",
        action="store_false",
        help="Disable bonuses during checkpoint evaluation",
    )

    # Output and logging
    parser.add_argument(
        "--save_plots", action="store_true", default=True, help="Save training progress plots"
    )
    parser.add_argument(
        "--log_interval", type=int, default=50, help="Print progress every N episodes"
    )
    parser.add_argument("--verbose", action="store_true", help="Enable verbose logging")

    # Ball direction control
    parser.add_argument(
        "--ball_direction",
        type=str,
        default="random",
        choices=[
            "random",
            "left",
            "right",
            "up_left",
            "up_right",
            "down_left",
            "down_right",
            "cone_left",
            "cone_right",
        ],
        help="Initial ball direction: random, left, right, up_left, up_right, down_left, down_right, cone_left, cone_right",
    )
    parser.add_argument(
        "--ball_angle",
        type=float,
        default=None,
        help="Specific ball angle in degrees (0-360). Overrides --ball_direction if specified",
    )

    # Advanced dual-scale training settings
    parser.add_argument(
        "--training_mode",
        type=str,
        choices=["episode_end", "step_by_step"],
        default="step_by_step",
        help="Training mode: 'episode_end' for traditional RL, 'step_by_step' for immediate feedback",
    )
    parser.add_argument(
        "--enable_dual_scale_training",
        action="store_true",
        help="Enable advanced dual-scale training (tactical + strategic learning)",
    )
    parser.add_argument(
        "--tactical_train_frequency",
        type=int,
        default=10,
        help="Frequency of tactical training updates (every N steps)",
    )
    parser.add_argument(
        "--tactical_learning_rate",
        type=float,
        default=None,
        help="Learning rate for tactical training (default: 0.3 * main LR)",
    )
    parser.add_argument(
        "--strategic_learning_rate",
        type=float,
        default=None,
        help="Learning rate for strategic training (default: 1.5 * main LR)",
    )
    parser.add_argument(
        "--train_frequency",
        type=int,
        default=10,
        help="Base training frequency (steps between training updates)",
    )
    parser.add_argument(
        "--min_replay_size",
        type=int,
        default=1000,
        help="Minimum replay buffer size before training starts",
    )
    parser.add_argument(
        "--reward_normalization",
        action="store_true",
        default=True,
        help="Enable reward normalization for stable training",
    )
    parser.add_argument(
        "--no_reward_normalization",
        dest="reward_normalization",
        action="store_false",
        help="Disable reward normalization",
    )

    # Display settings
    parser.add_argument(
        "--headless",
        action="store_true",
        default=True,
        help="Run in headless mode (no GUI). Use --no-headless to enable GUI display",
    )
    parser.add_argument(
        "--no-headless",
        dest="headless",
        action="store_false",
        help="Enable GUI display during training/evaluation",
    )
    parser.add_argument(
        "--fast_gui",
        action="store_true",
        default=True,
        help="Run in fast GUI mode (high FPS). Use --no-fast_gui for normal speed",
    )
    parser.add_argument(
        "--no-fast_gui",
        dest="fast_gui",
        action="store_false",
        help="Run in normal speed GUI mode (lower FPS for better visualization)",
    )

    args = parser.parse_args(argv)
    try:
        validate_mixed_fine_tuning_args(args)
        args.parsed_opponent_mix = parse_opponent_mix(args.opponent_mix)
        args.parsed_training_score_targets = parse_training_score_targets(args.training_score_mix)
        if args.checkpoint_eval_episodes < 0:
            raise ValueError("--checkpoint_eval_episodes must be non-negative")
        if args.checkpoint_eval_opponents_csv:
            _checkpoint_eval_opponents(args)
        if checkpoint_evaluation_enabled(args):
            checkpoint_evaluation_config(args)
    except ValueError as exc:
        parser.error(str(exc))

    return args


def main(argv: list[str] | None = None):
    args = parse_arguments(argv)

    print("🚀 ADVANCED DQN TRAINING STARTED")
    print(f"Mode: {args.mode}")
    print(f"Display: {'Headless (no GUI)' if args.headless else 'GUI enabled'}")

    # Training configuration info
    if args.mode == "progressive":
        print(f"Configuration: {args.progressive_episodes_per_phase} episodes per phase (4 phases)")
    else:
        print(
            f"Configuration: {args.episodes_per_phase if args.mode == 'curriculum' else args.total_episodes} episodes"
        )

    # Advanced training features info
    print(f"Training Mode: {args.training_mode}")
    if args.enable_dual_scale_training:
        print("🎯 DUAL-SCALE TRAINING ENABLED")
        print(f"   Tactical frequency: every {args.tactical_train_frequency} steps")
        print(f"   Tactical LR: {args.tactical_learning_rate or f'{args.learning_rate * 0.3:.6f}'}")
        print(
            f"   Strategic LR: {args.strategic_learning_rate or f'{args.learning_rate * 1.5:.6f}'}"
        )
        print(f"   Base LR: {args.learning_rate}")
    else:
        print(f"Standard training (LR: {args.learning_rate})")

    if args.reward_normalization:
        print("📊 Reward normalization: ENABLED")
    else:
        print("📊 Reward normalization: DISABLED")

    if args.mixed_opponents:
        print(f"Mixed-opponent fine-tuning: ENABLED (seed: {args.opponent_mix_seed})")
        print(f"   Opponent mix: {args.opponent_mix}")
        print(f"   Training MAX_SCORE mix: {args.training_score_mix}")

    # Create output directories
    Path(args.output_dir).mkdir(parents=True, exist_ok=True)
    if args.checkpoint_dir is None:
        args.checkpoint_dir = str(Path(args.output_dir) / "checkpoints")
    if args.checkpoint_interval > 0:
        Path(args.checkpoint_dir).mkdir(parents=True, exist_ok=True)

    start_time = time.time()

    # Create or load agent
    if args.load_model:
        print(f"Loading existing model: {args.load_model}")
        agent = create_agent_from_args(args)
        load_existing_model(agent, args.load_model)
    elif args.load_checkpoint:
        print(f"Loading from checkpoint: {args.load_checkpoint}")
        agent = create_agent_from_args(args)
        checkpoint_metadata = load_checkpoint(agent, args.load_checkpoint)
        if checkpoint_metadata:
            print(
                f"Resuming from episode {checkpoint_metadata['episode']}, phase {checkpoint_metadata['phase']}"
            )
    else:
        print("Creating new agent with optimized parameters")
        agent = create_agent_from_args(args)

    # Training
    if args.mode == "curriculum":
        agent, training_results = train_with_curriculum(agent, args)
    elif args.mode == "single":
        agent, training_results = train_single_phase(agent, args)
    elif args.mode == "progressive":
        agent, training_results = train_progressive_curriculum(agent, args)
    elif args.mode == "dual_scale":
        agent, training_results = train_dual_scale(agent, args)
    else:  # continue mode
        if not (args.load_model or args.load_checkpoint):
            print("Error: Continue mode requires --load_model or --load_checkpoint")
            return
        # For continue mode, use single phase training with loaded model
        agent, training_results = train_single_phase(agent, args)

    training_time = time.time() - start_time

    # Print training summary
    print(f"\n{'=' * 60}")
    print("TRAINING SUMMARY")
    print(f"{'=' * 60}")
    print(f"Training time: {training_time / 60:.1f} minutes")
    print(f"Total episodes: {training_results['total_episodes']}")

    if args.mode == "curriculum":
        print(
            f"Phase 1 - Avg reward: {training_results['phase1_avg']:.2f}, Win rate: {training_results['phase1_winrate']:.1f}%"
        )
        print(
            f"Phase 2 - Avg reward: {training_results['phase2_avg']:.2f}, Win rate: {training_results['phase2_winrate']:.1f}%"
        )
        improvement = training_results["phase2_avg"] - training_results["phase1_avg"]
        if improvement > 0:
            print(f"✅ Improvement: +{improvement:.2f} reward points")
        else:
            print(f"❌ Performance change: {improvement:.2f} reward points")
    elif args.mode == "progressive":
        print(
            f"Phase 1 (Hit Ball) - Avg reward: {training_results['phase1_avg']:.2f}, Win rate: {training_results['phase1_winrate']:.1f}%"
        )
        print(
            f"Phase 2 (Return Ball) - Avg reward: {training_results['phase2_avg']:.2f}, Win rate: {training_results['phase2_winrate']:.1f}%"
        )
        print(
            f"Phase 3 (Strategic Play) - Avg reward: {training_results['phase3_avg']:.2f}, Win rate: {training_results['phase3_winrate']:.1f}%"
        )
        print(
            f"Phase 4 (Master Play) - Avg reward: {training_results['phase4_avg']:.2f}, Win rate: {training_results['phase4_winrate']:.1f}%"
        )

        total_improvement = training_results["phase4_avg"] - training_results["phase1_avg"]
        if total_improvement > 0:
            print(f"✅ Total improvement: +{total_improvement:.2f} reward points across all phases")
        else:
            print(f"❌ Total change: {total_improvement:.2f} reward points")
    elif args.mode == "dual_scale":
        print(f"Average reward: {training_results['avg_reward']:.2f}")
        print(f"Win rate: {training_results['win_rate']:.1f}%")
        print(f"Tactical training steps: {training_results['tactical_steps']}")
        print("🎯 Dual-scale training features:")
        print(
            f"   ✓ Tactical learning: {training_results['dual_scale_stats']['tactical_optimizer_lr']:.6f} LR"
        )
        print(
            f"   ✓ Strategic learning: {training_results['dual_scale_stats']['strategic_optimizer_lr']:.6f} LR"
        )
        print(f"   ✓ Training mode: {training_results['dual_scale_stats']['training_mode']}")
    else:
        print(f"Average reward: {training_results['avg_reward']:.2f}")
        print(f"Win rate: {training_results['win_rate']:.1f}%")

    print(f"Final model saved: {training_results['final_model_path']}")

    # Final evaluation
    if args.evaluate:
        eval_results = evaluate_final_performance(agent, args)

        print(f"\n{'=' * 60}")
        print("FINAL EVALUATION")
        print(f"{'=' * 60}")
        for opponent, results in eval_results.items():
            print(
                f"{opponent:12}: {results['win_rate']:5.1f}% wins, {results['avg_reward']:6.2f} avg reward"
            )

    print(f"\n🎉 Training completed! Models saved in {args.output_dir}/")


if __name__ == "__main__":
    main()
