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
from collections.abc import Mapping
from collections.abc import Sequence
from contextlib import contextmanager
from dataclasses import dataclass
from datetime import datetime
from pathlib import Path
from typing import Any

import numpy as np

from magic_pong.ai import evaluation as ai_evaluation
from magic_pong.ai.models.dqn_ai import DQNAgent
from magic_pong.ai.models.dqn_checkpoint import EXPECTED_ACTION_SIZE
from magic_pong.ai.models.dqn_checkpoint import EXPECTED_STATE_SIZE
from magic_pong.ai.models.simple_ai import create_ai
from magic_pong.ai.pretraining import create_pretrainer
from magic_pong.core.game_engine import TrainingManager
from magic_pong.utils.config import ai_config
from magic_pong.utils.config import ai_config_tmp
from magic_pong.utils.config import game_config

AI_TYPES = [
    "follow_ball",
    "defensive",
    "aggressive",
    "random",
    "predictive",
    "dummy",
    "training_dummy",
    "tracker",
    "striker",
]

DEFAULT_OPPONENT_MIX = {
    "defensive": 0.16,
    "follow_ball": 0.16,
    "aggressive": 0.20,
    "predictive": 0.12,
    "random": 0.08,
    "dummy": 0.04,
    "training_dummy": 0.04,
    "tracker": 0.10,
    "striker": 0.10,
}
DEFAULT_OPPONENT_MIX_CLI = ",".join(
    f"{name}:{weight:g}" for name, weight in DEFAULT_OPPONENT_MIX.items()
)
DEFAULT_TRAINING_SCORE_MIX = (3, 5, 11)
DEFAULT_TRAINING_SCORE_MIX_CLI = ",".join(str(score) for score in DEFAULT_TRAINING_SCORE_MIX)
DEFAULT_CHECKPOINT_EVAL_OPPONENTS = tuple(DEFAULT_OPPONENT_MIX)
DEFAULT_CHECKPOINT_EVAL_EPISODES = 10
MIXED_FINE_TUNING_MODES = frozenset({"continue", "single"})
REWARD_SHAPING_MODES = ("legacy", "phase3")
EVAL_GATE_METRICS = frozenset(
    {
        "win_rate",
        "avg_goal_diff",
    }
)


@dataclass(frozen=True)
class WeightedOpponent:
    """Normalized opponent weight used for deterministic mixed fine-tuning."""

    name: str
    weight: float


@dataclass(frozen=True)
class EvaluationGate:
    """Minimum metric requirement for a checkpoint evaluation opponent."""

    opponent: str
    metric: str
    threshold: float


@dataclass
class PhaseTrainingResult:
    """Training phase result that still unpacks like the legacy five-tuple."""

    rewards: list[float]
    goals_for: int
    goals: int
    episodes: int
    metric_snapshots: list[dict]
    best_model_path: Path | None = None
    best_validation_report_path: Path | None = None
    selected_best_this_run: bool = False

    def __iter__(self):
        yield self.rewards
        yield self.goals_for
        yield self.goals
        yield self.episodes
        yield self.metric_snapshots


def _phase_training_result(value) -> PhaseTrainingResult:
    if isinstance(value, PhaseTrainingResult):
        return value
    rewards, goals_for, goals, episodes, metric_snapshots = value
    return PhaseTrainingResult(rewards, goals_for, goals, episodes, metric_snapshots)


class OpponentSampler:
    """Provide a fresh sampled opponent for each training episode."""

    def __init__(self, opponent_mix: Sequence[WeightedOpponent], rng: random.Random):
        self.opponent_mix = tuple(opponent_mix)
        self.rng = rng

    def next_opponent(self):
        selected_opponent = sample_weighted_opponent(self.opponent_mix, self.rng)
        return selected_opponent.name, get_opponent(selected_opponent.name)


class SelfPlaySampler:
    """Training adapter for the persistent, adaptive opponent league."""

    def __init__(
        self,
        base_mix: tuple[WeightedOpponent, ...],
        rng: random.Random,
        snapshot_dir: Path,
        pool_size: int,
        snapshot_weight: float,
        *,
        champion_size: int = 5,
        historical_size: int = 5,
        coverage_floor: float = 0.2,
        checkpoints: Sequence[str] = (),
        exploiter_target: str | None = None,
    ) -> None:
        from magic_pong.ai.league import OpponentLeague

        self.base_mix = base_mix
        self.rng = rng
        self.snapshot_dir = snapshot_dir
        self.pool_size = pool_size
        self.snapshot_weight = snapshot_weight
        self.league = OpponentLeague(
            snapshot_dir,
            rng=rng,
            base_weights={item.name: item.weight for item in base_mix},
            recent_size=pool_size,
            champion_size=champion_size,
            historical_size=historical_size,
            snapshot_weight=snapshot_weight,
            coverage_floor=coverage_floor,
            opponent_factory=get_opponent,
        )
        for path in checkpoints:
            self.league.add_checkpoint(path, role="independent")
        self.exploiter_target_id = (
            self.league.add_checkpoint(exploiter_target, role="target")
            if exploiter_target
            else None
        )

    def add_snapshot(
        self,
        agent: object,
        *,
        champion: bool = False,
        role: str = "snapshot",
        validation=None,
    ):
        metadata = {}
        if validation is not None:
            metadata["performance_profile"] = {
                f"{cell.opponent}:{cell.player_side}:{cell.bonuses_enabled}": cell.win_rate
                for cell in validation.matchups
            }
        if self.exploiter_target_id:
            metadata["exploiter_target"] = self.exploiter_target_id
        return self.league.add_snapshot(agent, champion=champion, role=role, metadata=metadata)

    def next_opponent(self):
        if self.exploiter_target_id:
            return self.league.next_opponent(only_ids=(self.exploiter_target_id,))
        return self.league.next_opponent()

    def record_outcome(self, opponent_id: str, *, won: bool, timeout: bool = False):
        self.league.record_outcome(opponent_id, won=won, timeout=timeout)

    def validation_factories(self):
        return self.league.benchmark_factories(content_ids=True)

    def cleanup(self) -> None:
        """Persist the league so future runs can reuse its opponents."""
        self.league.cleanup()


def get_pyplot():
    """Import matplotlib only when a plotting path is used."""
    import matplotlib.pyplot as plt

    return plt


def parse_hidden_layers(value: str) -> list[int]:
    """Parse a comma-separated '--hidden_layers' argument into a list of positive ints."""
    try:
        widths = [int(part.strip()) for part in value.split(",") if part.strip()]
    except ValueError as exc:
        raise argparse.ArgumentTypeError(f"Invalid --hidden_layers value: {value!r}") from exc
    if not widths or any(w <= 0 for w in widths):
        raise argparse.ArgumentTypeError(
            f"--hidden_layers must be a comma-separated list of positive integers: {value!r}"
        )
    return widths


def create_agent_from_args(args):
    """Create a DQN agent with parameters from command line arguments"""
    ai_evaluation._seed_everything(getattr(args, "seed", 0))
    return DQNAgent(
        state_size=EXPECTED_STATE_SIZE,
        action_size=EXPECTED_ACTION_SIZE,
        lr=args.learning_rate,
        gamma=args.gamma,
        epsilon=args.epsilon_start,
        epsilon_decay=args.epsilon_decay,
        epsilon_min=args.epsilon_min,
        batch_size=args.batch_size,
        memory_size=args.memory_size,
        train_frequency=getattr(args, "train_frequency", 10),
        min_replay_size=getattr(args, "min_replay_size", 1000),
        hidden_size=getattr(args, "hidden_size", 256),
        layer_size=getattr(args, "layer_size", 2),
        hidden_layers=getattr(args, "hidden_layers", None),
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
    agent,
    episode,
    phase,
    total_episodes,
    rewards,
    goals_for,
    goals,
    checkpoint_dir,
    prefix="checkpoint",
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
        "goals_for": goals_for,
        "goals": goals,
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


def _copy_overlapping_tensor(new_tensor, old_tensor):
    """Return a copy of new_tensor with the overlapping old_tensor slice copied in."""
    copied = new_tensor.clone()
    if copied.ndim != old_tensor.ndim:
        return copied, ()

    overlap = tuple(
        min(new_dim, old_dim) for new_dim, old_dim in zip(copied.shape, old_tensor.shape)
    )
    if any(size <= 0 for size in overlap):
        return copied, overlap

    slices = tuple(slice(0, size) for size in overlap)
    old_slice = old_tensor[slices].to(device=copied.device, dtype=copied.dtype)
    copied[slices] = old_slice
    return copied, overlap


def _copy_first_layer_for_warm_start(
    *, new_weight, old_weight, current_state_size: int, legacy_state_size: int | None
):
    """Copy legacy observation columns without leaking them into prev-action columns."""
    copied = new_weight.clone()
    if copied.ndim != 2 or old_weight.ndim != 2:
        return copied, ()

    rows = min(copied.shape[0], old_weight.shape[0])
    if rows <= 0:
        return copied, (0, 0)

    if legacy_state_size == 32 and current_state_size == 30:
        # Legacy schema-v1 stored field_width/field_height at columns 10-11.
        # Current schema-v3 removed those constants, so the tail shifts left by two.
        first_block = min(10, copied.shape[1], old_weight.shape[1])
        second_block = min(20, copied.shape[1] - 10, old_weight.shape[1] - 12)
        if first_block > 0:
            copied[:rows, :first_block] = old_weight[:rows, :first_block].to(
                device=copied.device, dtype=copied.dtype
            )
        if second_block > 0:
            copied[:rows, 10 : 10 + second_block] = old_weight[:rows, 12 : 12 + second_block].to(
                device=copied.device, dtype=copied.dtype
            )
        return copied, (rows, first_block + second_block)

    old_observation_cols = (
        legacy_state_size if legacy_state_size is not None else old_weight.shape[1]
    )
    cols = min(copied.shape[1], old_weight.shape[1], current_state_size, old_observation_cols)
    if cols > 0:
        copied[:rows, :cols] = old_weight[:rows, :cols].to(device=copied.device, dtype=copied.dtype)
    return copied, (rows, cols)


def warm_start_from_v1_checkpoint(agent: DQNAgent, checkpoint_path: str) -> None:
    """Transfer compatible weights from a legacy checkpoint into the current DQN agent.

    Legacy checkpoints in this project may have a different observation width and
    hidden-layer width from the current schema.  The optimizer is intentionally not
    restored because the parameter space can change across schemas.
    """
    import torch

    from magic_pong.ai.models.dqn_checkpoint import safe_torch_load

    if not os.path.exists(checkpoint_path):
        raise FileNotFoundError(f"Warm-start checkpoint not found: {checkpoint_path}")

    checkpoint = safe_torch_load(checkpoint_path)
    for state_dict_key in ("q_network_state_dict", "target_network_state_dict"):
        if state_dict_key not in checkpoint or not isinstance(checkpoint[state_dict_key], Mapping):
            raise ValueError(f"Warm-start checkpoint missing {state_dict_key}")

    metadata = checkpoint.get("metadata", {})
    hyperparameters = checkpoint.get("hyperparameters", {})
    legacy_state_size = metadata.get("state_size", hyperparameters.get("state_size"))
    if legacy_state_size is not None:
        legacy_state_size = int(legacy_state_size)

    transfer_summary: list[str] = []

    for net, old_sd_key in [
        (agent.q_network, "q_network_state_dict"),
        (agent.target_network, "target_network_state_dict"),
    ]:
        old_sd = checkpoint[old_sd_key]
        new_sd = net.state_dict()

        copied_shapes: dict[str, tuple[int, ...]] = {}

        first_layer_key = "fc_layers.0.weight"
        if first_layer_key in old_sd and first_layer_key in new_sd:
            new_w0, overlap = _copy_first_layer_for_warm_start(
                new_weight=new_sd[first_layer_key],
                old_weight=old_sd[first_layer_key],
                current_state_size=agent.state_size,
                legacy_state_size=legacy_state_size,
            )
            new_sd[first_layer_key] = new_w0
            copied_shapes[first_layer_key] = overlap

        for key, old_value in old_sd.items():
            if key == first_layer_key or key not in new_sd:
                continue
            new_value = new_sd[key]
            if not torch.is_tensor(old_value) or not torch.is_tensor(new_value):
                continue
            if key.startswith("output_layer."):
                # Output Q-values depend on all preceding representations; never transfer
                # output layer weights across architectures to avoid corrupting policy.
                continue
            copied_value, overlap = _copy_overlapping_tensor(new_value, old_value)
            new_sd[key] = copied_value
            copied_shapes[key] = overlap

        net.load_state_dict(new_sd)

        if old_sd_key == "q_network_state_dict":
            partial = [
                f"{key}{tuple(old_sd[key].shape)}->{tuple(new_sd[key].shape)} overlap={overlap}"
                for key, overlap in copied_shapes.items()
                if key in old_sd and tuple(old_sd[key].shape) != tuple(new_sd[key].shape)
            ]
            transfer_summary.extend(partial)

    agent.epsilon = float(checkpoint.get("epsilon", agent.epsilon))
    agent.training_step = int(checkpoint.get("training_step", 0))
    agent.step_count = int(checkpoint.get("step_count", 0))
    agent.loss_history = list(checkpoint.get("loss_history", []))
    agent.reward_history = list(checkpoint.get("reward_history", []))

    summary = "; ".join(transfer_summary) if transfer_summary else "all tensors copied exactly"
    print(
        f"Warm-started from {checkpoint_path}: "
        f"compatible tensor slices transferred ({summary}); optimizer reset"
    )


def get_opponent(opponent_type, player_id=2):
    """Create opponent based on type.

    Accepts either a name from AI_TYPES or the special form ``dqn:<path>`` to
    load a frozen DQN checkpoint as a non-learning opponent.
    """
    if opponent_type.startswith("dqn:"):
        model_path = opponent_type[len("dqn:") :]
        from magic_pong.ai.models.dqn_checkpoint import safe_torch_load

        ckpt = safe_torch_load(model_path)
        hp = ckpt.get("hyperparameters", {})
        raw_hidden_layers = hp.get("hidden_layers")
        opponent = DQNAgent(
            state_size=EXPECTED_STATE_SIZE,
            action_size=EXPECTED_ACTION_SIZE,
            hidden_size=int(hp.get("hidden_size", 256)),
            layer_size=int(hp.get("layer_size", 2)),
            hidden_layers=([int(w) for w in raw_hidden_layers] if raw_hidden_layers else None),
        )
        opponent.load_model(model_path)
        opponent.set_training_mode(False)
        opponent.name = f"dqn:{model_path}"
        return opponent

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


def parse_eval_gate(gate_value: str) -> EvaluationGate:
    """Parse '<opponent>:<metric>:<minimum>' checkpoint evaluation gate syntax."""
    parts = [part.strip() for part in gate_value.split(":")]
    if len(parts) != 3 or not all(parts):
        raise ValueError(
            f"Invalid evaluation gate {gate_value!r}; expected '<opponent>:<metric>:<minimum>'"
        )

    opponent, metric, raw_threshold = parts
    if opponent not in AI_TYPES or opponent == "training_dummy":
        valid_opponents = ", ".join(
            opponent for opponent in AI_TYPES if opponent != "training_dummy"
        )
        raise ValueError(
            f"Unknown evaluation gate opponent {opponent!r}. Valid opponents: {valid_opponents}"
        )
    if metric not in EVAL_GATE_METRICS:
        valid_metrics = ", ".join(sorted(EVAL_GATE_METRICS))
        raise ValueError(
            f"Unknown evaluation gate metric {metric!r}. Valid metrics: {valid_metrics}"
        )

    try:
        threshold = float(raw_threshold)
    except ValueError as exc:
        raise ValueError(
            f"Invalid threshold for evaluation gate {gate_value!r}: {raw_threshold!r}"
        ) from exc
    if not math.isfinite(threshold):
        raise ValueError("Evaluation gate thresholds must be finite")
    if metric == "win_rate" and not 0.0 <= threshold <= 1.0:
        raise ValueError("win_rate evaluation gates use a 0.0 to 1.0 fraction threshold")

    return EvaluationGate(opponent=opponent, metric=metric, threshold=threshold)


def parse_eval_gates(gate_values: Sequence[str] | None) -> tuple[EvaluationGate, ...]:
    """Parse and deduplicate checkpoint evaluation gates."""
    gates = tuple(parse_eval_gate(value) for value in (gate_values or ()))
    seen: set[tuple[str, str]] = set()
    for gate in gates:
        key = (gate.opponent, gate.metric)
        if key in seen:
            raise ValueError(
                f"Duplicate evaluation gate for opponent {gate.opponent!r} metric {gate.metric!r}"
            )
        seen.add(key)
    return gates


def validate_reward_shaping_args(args) -> None:
    """Validate reward shaping CLI options before mutating global config."""
    mode = getattr(args, "reward_shaping", "legacy")
    if mode not in REWARD_SHAPING_MODES:
        valid_modes = ", ".join(REWARD_SHAPING_MODES)
        raise ValueError(f"--reward_shaping must be one of: {valid_modes}")

    reward_weight_args = (
        "phase3_intercept_progress_reward",
        "phase3_intercept_distance_penalty",
        "phase3_successful_return_reward",
    )
    for attr_name in reward_weight_args:
        value = getattr(args, attr_name, None)
        if value is None:
            continue
        if not math.isfinite(value) or value < 0:
            cli_name = "--" + attr_name.replace("_", "-")
            raise ValueError(f"{cli_name} must be a non-negative finite number")


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
    if not (args.load_model or args.load_checkpoint or args.warm_start_checkpoint):
        raise ValueError(
            "--mixed_opponents is a fine-tuning mode and requires --load_model, "
            "--load_checkpoint, or --warm_start_checkpoint"
        )


def validate_pretrain_args(args) -> None:
    """Block pretraining when an existing model is loaded.

    Pretraining replaces existing Q-value estimates with proximity-reward
    signal.  A loaded checkpoint has already learned game-play behaviour;
    overwriting it with pretraining may degrade that knowledge.  Use
    pretraining only when training a fresh agent from scratch (no
    --load_model / --load_checkpoint / --warm_start_checkpoint).
    """
    if not getattr(args, "pretrain", False):
        return
    if args.load_model or args.load_checkpoint or args.warm_start_checkpoint:
        raise ValueError(
            "--pretrain cannot be combined with --load_model, --load_checkpoint, or "
            "--warm_start_checkpoint.  Pretraining is designed for fresh agents "
            "(epsilon=1.0) and will corrupt loaded weights when epsilon is already low."
        )


def _validate_opponent_arg(args) -> None:
    """Validate the --opponent argument for single-phase training."""
    opponent = getattr(args, "opponent", None)
    if opponent is None:
        return
    if opponent.startswith("dqn:"):
        path = opponent[len("dqn:") :]
        if not os.path.exists(path):
            raise ValueError(f"DQN opponent model not found: {path!r}")
        return
    if opponent not in AI_TYPES:
        valid = ", ".join(AI_TYPES)
        raise ValueError(
            f"Unknown opponent {opponent!r}. Use a named type ({valid}) "
            "or 'dqn:<path>' to load a trained DQN checkpoint."
        )


def _validate_self_play_args(args) -> None:
    """Validate self-play configuration flags."""
    interval = getattr(args, "self_play_interval", 0)
    if interval < 0:
        raise ValueError("--self_play_interval must be >= 0")
    if interval == 0:
        return
    if getattr(args, "mode", None) not in ("single", "continue", "pipeline"):
        raise ValueError("--self_play_interval requires --mode single, continue, or pipeline")
    pool_size = getattr(args, "self_play_pool_size", 5)
    if pool_size < 1:
        raise ValueError("--self_play_pool_size must be >= 1")
    weight = getattr(args, "self_play_weight", 0.5)
    if not (0.0 <= weight < 1.0):
        raise ValueError("--self_play_weight must be in [0.0, 1.0)")


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


def _eval_gates_from_args(args) -> tuple[EvaluationGate, ...]:
    parsed_gates = getattr(args, "parsed_eval_gates", None)
    if parsed_gates is not None:
        return tuple(parsed_gates)
    return parse_eval_gates(getattr(args, "eval_gates", None))


def reward_shaping_config_from_args(args) -> dict[str, Any]:
    """Return temporary ai_config values requested by CLI args."""
    mode = getattr(args, "reward_shaping", None)
    if mode is None:
        return {}

    config_values: dict[str, Any] = {"REWARD_SHAPING_MODE": mode}
    if mode == "phase3":
        arg_to_config = {
            "phase3_intercept_progress_reward": "PHASE3_INTERCEPT_PROGRESS_REWARD",
            "phase3_intercept_distance_penalty": "PHASE3_INTERCEPT_DISTANCE_PENALTY",
            "phase3_successful_return_reward": "PHASE3_SUCCESSFUL_RETURN_REWARD",
        }
        for arg_name, config_name in arg_to_config.items():
            if hasattr(args, arg_name):
                config_values[config_name] = getattr(args, arg_name)
    return config_values


@contextmanager
def temporary_reward_shaping_config(args):
    """Temporarily apply CLI reward shaping config and always restore it."""
    config_values = reward_shaping_config_from_args(args)
    if not config_values:
        yield
        return

    with ai_config_tmp(**config_values):
        yield


@contextmanager
def temporary_legacy_reward_evaluation():
    """Run evaluation with deployment-style reward metrics, not training shaping."""
    with ai_config_tmp(REWARD_SHAPING_MODE="legacy", USE_PROXIMITY_REWARD=False):
        yield


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

    gate_opponents = tuple(gate.opponent for gate in _eval_gates_from_args(args))
    opponents = repeated_opponents + csv_opponents
    if not opponents:
        opponents = DEFAULT_CHECKPOINT_EVAL_OPPONENTS

    for gate_opponent in gate_opponents:
        if gate_opponent not in opponents:
            opponents = opponents + (gate_opponent,)

    invalid = sorted(set(opponents) - set(AI_TYPES))
    if invalid:
        raise ValueError(f"Unknown checkpoint evaluation opponent(s): {', '.join(invalid)}")
    return opponents


_DOMAIN_RAND_PARAMS = ("BALL_SPEED", "PADDLE_SPEED", "FIELD_HEIGHT")


@contextmanager
def _domain_rand_context(args, training_manager=None):
    """Per-episode game parameter randomization (domain randomization)."""
    noise = getattr(args, "domain_rand_noise", 0.0)
    originals = {name: getattr(game_config, name) for name in _DOMAIN_RAND_PARAMS}
    engine = getattr(training_manager, "game_engine", None)
    environment = getattr(engine, "ai_environment", None)

    def sync_geometry():
        if environment is not None:
            environment.configure_geometry(game_config.FIELD_WIDTH, game_config.FIELD_HEIGHT)

    try:
        if noise > 0.0:
            sampled_values = {}
            for name, base in originals.items():
                factor = 1.0 + random.uniform(-noise, noise)
                value = max(float(base) * factor, 1e-6)
                if name == "BALL_SPEED":
                    value = min(value, float(game_config.MAX_BALL_SPEED))
                elif name == "FIELD_HEIGHT":
                    value = max(value, float(game_config.PADDLE_HEIGHT + 50))
                sampled_values[name] = type(base)(value)
            type(game_config)(**{**game_config.model_dump(), **sampled_values})
            for name in ("FIELD_HEIGHT", "BALL_SPEED", "PADDLE_SPEED"):
                if name in sampled_values:
                    setattr(game_config, name, sampled_values[name])
        sync_geometry()
        yield
    finally:
        for name in ("FIELD_HEIGHT", "BALL_SPEED", "PADDLE_SPEED"):
            val = originals[name]
            setattr(game_config, name, val)
        sync_geometry()


def checkpoint_evaluation_enabled(args) -> bool:
    return (
        bool(getattr(args, "evaluate_checkpoints", False))
        or (getattr(args, "checkpoint_eval_episodes", 0) > 0)
        or bool(_eval_gates_from_args(args))
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


def evaluation_gate_results(
    evaluation_payload: dict[str, Any], gates: Sequence[EvaluationGate]
) -> dict[str, Any]:
    """Evaluate gate requirements against an existing evaluation JSON payload."""
    opponents = {
        opponent.get("opponent"): opponent
        for opponent in evaluation_payload.get("opponents", [])
        if isinstance(opponent, dict)
    }
    gate_results = []

    for gate in gates:
        opponent_result = opponents.get(gate.opponent)
        actual_value = None
        error = None
        if opponent_result is None:
            error = f"Opponent {gate.opponent!r} was not evaluated"
        elif gate.metric not in opponent_result:
            error = f"Metric {gate.metric!r} is missing for opponent {gate.opponent!r}"
        else:
            try:
                actual_value = float(opponent_result[gate.metric])
            except (TypeError, ValueError):
                error = f"Metric {gate.metric!r} is not numeric for opponent {gate.opponent!r}"

        passed = actual_value is not None and actual_value >= gate.threshold
        gate_results.append(
            {
                "opponent": gate.opponent,
                "metric": gate.metric,
                "comparison": ">=",
                "threshold": gate.threshold,
                "actual": actual_value,
                "passed": passed,
                "error": error,
            }
        )

    return {
        "passed": all(result["passed"] for result in gate_results),
        "gates": gate_results,
    }


def checkpoint_evaluation_gates_passed(eval_json_path: str | Path) -> bool:
    """Return True when a written checkpoint evaluation payload passed all gates."""
    payload = json.loads(Path(eval_json_path).read_text(encoding="utf-8"))
    gate_results = payload.get("eval_gates")
    return bool(gate_results and gate_results.get("passed"))


def maybe_save_checkpoint_evaluation(
    checkpoint_path: str | Path, args, training_context: dict[str, Any] | None = None
) -> str | None:
    """Evaluate a saved checkpoint and write JSON next to it when requested."""
    if not checkpoint_evaluation_enabled(args):
        return None

    config = checkpoint_evaluation_config(args)
    with temporary_legacy_reward_evaluation(), preserve_global_rng_state():
        result = ai_evaluation.evaluate_checkpoint(checkpoint_path, config)
    output_path = checkpoint_evaluation_path(checkpoint_path)
    payload = ai_evaluation.result_to_dict(result)
    payload["reward_shaping"] = {
        "evaluation_mode": "legacy",
        "training_mode": getattr(args, "reward_shaping", "legacy"),
    }
    if training_context is not None:
        payload["training_context"] = training_context
    gates = _eval_gates_from_args(args)
    if gates:
        payload["eval_gates"] = evaluation_gate_results(payload, gates)
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
    goals_for: int,
    goals: int,
) -> dict[str, Any]:
    """Metadata used when evaluating a final saved model candidate."""
    recent_rewards = rewards[-100:] if len(rewards) >= 100 else rewards
    context: dict[str, Any] = {
        "candidate_type": "final_model",
        "mode": getattr(args, "mode", None),
        "label": label,
        "total_episodes": total_episodes,
        "goals_for": goals_for,
        "goals": goals,
        "point_win_rate": goals_for / goals * 100 if goals > 0 else 50.0,
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
    original_micro_reward_scale = ai_config.MICRO_REWARD_SCALE
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
        if not isinstance(opponent_source, SelfPlaySampler):
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

    anneal_enabled = False
    fixed_micro_scale = None

    try:
        rewards = []
        point_win_rates_history: list[float] = []
        dqn_snap_win_rates: list[float] = []
        rule_based_win_rates: list[float] = []
        total_goals_for = 0
        total_goals = 0
        best_avg_reward = float("-inf")
        best_model_path: Path | None = None
        episodes_since_improvement = 0

        # Extended metric tracking
        episode_lengths: list[int] = []
        draw_flags: list[bool] = []
        match_wins: list[bool] = []
        goals_for_per_ep: list[int] = []
        goals_against_per_ep: list[int] = []
        _metric_snapshots: list[dict] = []
        _probe_states = None  # torch.Tensor once populated
        _track_q = getattr(args, "track_q_values", True)

        best_model_enabled = getattr(args, "save_best_model", True)
        _best_model_file = (
            Path(getattr(args, "output_dir", "."))
            / f"{getattr(args, 'model_prefix', 'dqn')}_best.pth"
        )
        selector = None
        validation_interval = getattr(args, "competitive_eval_interval", 0)
        if best_model_enabled and validation_interval > 0:
            from magic_pong.ai.training_validation import CheckpointSelector
            from magic_pong.ai.training_validation import validation_config_from_args

            factories = (
                opponent_source.validation_factories()
                if isinstance(opponent_source, SelfPlaySampler)
                else {}
            )
            selector = CheckpointSelector(
                _best_model_file,
                validation_config_from_args(args, tuple(factories)),
                validation_interval,
                factories,
                patience_baseline_episode=start_episode,
            )
            selector.consider(agent, start_episode, force=True)
            best_model_path = selector.best_path_this_run

        _anneal_episodes_arg = getattr(args, "micro_reward_anneal_episodes", 0)
        anneal_episodes = _anneal_episodes_arg if _anneal_episodes_arg > 0 else episodes
        anneal_enabled = getattr(args, "anneal_micro_rewards", False)
        fixed_micro_scale = getattr(args, "micro_reward_scale", None)
        if fixed_micro_scale is not None:
            ai_config.MICRO_REWARD_SCALE = float(fixed_micro_scale)

        with temporary_reward_shaping_config(args):
            for episode in range(start_episode, start_episode + episodes):
                if anneal_enabled:
                    elapsed = episode - start_episode
                    ai_config.MICRO_REWARD_SCALE = float(max(0.0, 1.0 - elapsed / anneal_episodes))

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

                with (
                    temporary_game_config_value("MAX_SCORE", episode_max_score),
                    _domain_rand_context(args, training_manager),
                ):
                    episode_stats = training_manager.train_episode(
                        agent, episode_opponent, max_steps=args.max_steps_per_episode
                    )
                episode_reward = episode_stats["total_reward_p1"]
                if isinstance(opponent_source, SelfPlaySampler):
                    winner = episode_stats.get("winner", 0)
                    opponent_source.record_outcome(
                        episode_opponent_name, won=winner == 1, timeout=winner == 0
                    )

                # Draw penalty: apply when episode timed out with no goals scored
                _draw_penalty = getattr(args, "draw_penalty", 0.0)
                if _draw_penalty != 0.0 and episode_stats.get("winner", 0) == 0:
                    episode_goals_check = episode_stats.get("events", [])
                    if not episode_goals_check:
                        agent.add_terminal_reward(_draw_penalty)
                        episode_reward += _draw_penalty

                rewards.append(episode_reward)

                episode_goals = episode_stats.get("events", [])
                ep_goals_for = sum(1 for g in episode_goals if g.get("player") == 1)
                ep_goals_against = sum(1 for g in episode_goals if g.get("player") == 2)
                ep_total = ep_goals_for + ep_goals_against
                episode_point_win_rate = ep_goals_for / ep_total if ep_total > 0 else 0.5
                point_win_rates_history.append(episode_point_win_rate)
                if episode_opponent_name.startswith("dqn:") or (
                    isinstance(opponent_source, SelfPlaySampler)
                    and episode_opponent_name.removeprefix("builtin:") not in AI_TYPES
                ):
                    dqn_snap_win_rates.append(episode_point_win_rate)
                else:
                    rule_based_win_rates.append(episode_point_win_rate)
                total_goals_for += ep_goals_for
                total_goals += ep_total

                # Extended per-episode tracking
                episode_lengths.append(episode_stats.get("steps", 0))
                draw_flags.append(episode_stats.get("winner", 0) == 0 and ep_total == 0)
                match_wins.append(episode_stats.get("winner", 0) == 1)
                goals_for_per_ep.append(ep_goals_for)
                goals_against_per_ep.append(ep_goals_against)

                # Q-value probe: capture batch once the buffer is warm
                if (
                    _track_q
                    and _probe_states is None
                    and len(agent.memory) >= agent.min_replay_size
                ):
                    import torch

                    from magic_pong.ai.models.dqn_ai import device as _dqn_device

                    probe_samples = agent.memory.sample(min(256, len(agent.memory)))
                    _probe_states = torch.FloatTensor(
                        np.array([t.state for t in probe_samples])
                    ).to(_dqn_device)

                # Debug info for troubleshooting
                if args.verbose:
                    winner = episode_stats.get("winner", 0)
                    print(
                        f"    Episode {episode}: reward={episode_reward:.2f}, "
                        f"points={ep_goals_for}/{ep_total} ({episode_point_win_rate * 100:.0f}%), winner={winner}"
                    )

                # Progress reporting
                if (episode + 1) % args.log_interval == 0 and episode > start_episode:
                    recent_rewards = rewards[-args.log_interval :]
                    avg_reward = np.mean(recent_rewards)

                    # Calculate recent point win rate (same window as recent rewards)
                    recent_rates = point_win_rates_history[-args.log_interval :]
                    recent_win_rate = np.mean(recent_rates) * 100

                    # Global point win rate for comparison
                    global_win_rate = (
                        (total_goals_for / total_goals * 100) if total_goals > 0 else 50.0
                    )

                    # Get detailed training statistics
                    training_stats = agent.get_training_stats()

                    recent_rb = rule_based_win_rates[-args.log_interval :]
                    recent_dqn = dqn_snap_win_rates[-args.log_interval :]
                    breakdown = ""
                    if recent_rb and recent_dqn:
                        breakdown = (
                            f" [rule:{np.mean(recent_rb) * 100:.0f}%"
                            f" / snap:{np.mean(recent_dqn) * 100:.0f}%]"
                        )
                    elif recent_rb:
                        breakdown = f" [rule:{np.mean(recent_rb) * 100:.0f}%]"
                    elif recent_dqn:
                        breakdown = f" [snap:{np.mean(recent_dqn) * 100:.0f}%]"
                    base_info = (
                        f"  Episode {episode}: avg reward = {avg_reward:.2f}, "
                        f"point win rate = {recent_win_rate:.1f}%{breakdown}"
                        f" (global: {global_win_rate:.1f}%), epsilon = {agent.epsilon:.3f}"
                    )
                    print(base_info)
                    if args.verbose:
                        print(f"    Memory: {training_stats['memory_size']} experiences")

                    # Legacy reward patience is only used when validation is disabled.
                    if selector is None and avg_reward > best_avg_reward:
                        best_avg_reward = avg_reward
                        episodes_since_improvement = 0
                    elif selector is None:
                        episodes_since_improvement += args.log_interval

                    # Collect interval snapshot for metrics JSON / detailed plot
                    _ws = slice(-args.log_interval, None)
                    mean_q_val = None
                    if _track_q and _probe_states is not None:
                        import torch

                        with torch.no_grad():
                            agent.q_network.eval()
                            mean_q_val = float(
                                agent.q_network(_probe_states).max(dim=1).values.mean().item()
                            )
                            agent.q_network.train()
                    _metric_snapshots.append(
                        {
                            "episode": episode,
                            "avg_reward": float(avg_reward),
                            "point_win_rate": float(recent_win_rate / 100),
                            "match_win_rate": (
                                float(np.mean(match_wins[_ws])) if match_wins else None
                            ),
                            "rule_based_win_rate": (
                                float(np.mean(recent_rb)) if recent_rb else None
                            ),
                            "snap_win_rate": (float(np.mean(recent_dqn)) if recent_dqn else None),
                            "draw_rate": (float(np.mean(draw_flags[_ws])) if draw_flags else 0.0),
                            "avg_goals_for": (
                                float(np.mean(goals_for_per_ep[_ws])) if goals_for_per_ep else 0.0
                            ),
                            "avg_goals_against": (
                                float(np.mean(goals_against_per_ep[_ws]))
                                if goals_against_per_ep
                                else 0.0
                            ),
                            "avg_episode_length": (
                                float(np.mean(episode_lengths[_ws])) if episode_lengths else 0.0
                            ),
                            "epsilon": float(agent.epsilon),
                            "td_loss": (
                                float(np.mean(agent.loss_history[-100:]))
                                if agent.loss_history
                                else None
                            ),
                            "grad_norm": (
                                float(np.mean(agent.grad_norm_history[-100:]))
                                if agent.grad_norm_history
                                else None
                            ),
                            "mean_q": mean_q_val,
                        }
                    )

                completed_episode = episode + 1
                if selector is not None:
                    improved = selector.consider(agent, completed_episode)
                    episodes_since_improvement = completed_episode - selector.last_improved_episode
                    if improved and isinstance(opponent_source, SelfPlaySampler):
                        opponent_source.add_snapshot(
                            agent, champion=True, validation=selector.best_result
                        )

                # Self-play snapshot update
                _self_play_interval = getattr(args, "self_play_interval", 0)
                if (
                    _self_play_interval > 0
                    and hasattr(opponent_source, "add_snapshot")
                    and (episode - start_episode) > 0
                    and (episode - start_episode) % _self_play_interval == 0
                ):
                    snap_min_wr = getattr(args, "self_play_snapshot_min_win_rate", 0.0)
                    window = point_win_rates_history[-_self_play_interval:]
                    recent_wr = np.mean(window) * 100 if window else 50.0
                    if recent_wr >= snap_min_wr:
                        opponent_source.add_snapshot(agent)
                    else:
                        print(
                            f"  [self-play] snapshot skipped: "
                            f"win rate {recent_wr:.1f}% < {snap_min_wr:.1f}% threshold"
                        )
                    snap_epsilon_boost = getattr(args, "self_play_epsilon_boost", 0.0)
                    if snap_epsilon_boost > 0:
                        agent.epsilon = max(agent.epsilon, snap_epsilon_boost)

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
                        total_goals_for,
                        total_goals,
                        args.checkpoint_dir,
                    )
                    eval_path = maybe_save_checkpoint_evaluation(
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
                    if (
                        eval_path
                        and getattr(args, "stop_when_eval_gates_pass", False)
                        and checkpoint_evaluation_gates_passed(eval_path)
                    ):
                        print(f"Evaluation gates passed at checkpoint {checkpoint_path}")
                        break

                # Early stopping
                if (
                    args.early_stopping > 0
                    and episodes_since_improvement >= args.early_stopping
                    and (selector is None or selector.last_evaluated_episode == completed_episode)
                ):
                    print(
                        f"Early stopping triggered after {episodes_since_improvement} episodes without improvement"
                    )
                    break

        final_episodes = len(rewards)
        if selector is not None:
            improved = selector.consider(agent, start_episode + final_episodes, force=True)
            if improved and isinstance(opponent_source, SelfPlaySampler):
                opponent_source.add_snapshot(agent, champion=True, validation=selector.best_result)
            best_model_path = selector.best_path_this_run
        best_validation_report_path = (
            selector.best_report_path_this_run if selector is not None else None
        )
        selected_best_this_run = bool(selector is not None and selector.selected_best_this_run)
        avg_reward = np.mean(rewards[-100:]) if len(rewards) >= 100 else np.mean(rewards)
        point_win_rate = (total_goals_for / total_goals * 100) if total_goals > 0 else 50.0

        print(f"✅ Phase {phase_num} completed:")
        print(f"   Episodes: {final_episodes}")
        print(f"   Average reward: {avg_reward:.2f}")
        print(f"   Point win rate: {point_win_rate:.1f}% ({total_goals_for}/{total_goals} points)")
        if best_model_path is not None:
            print(f"   Best model (competitive validation): {best_model_path}")

        # Debug: show some reward statistics
        if rewards:
            print(
                f"   Reward stats: min={min(rewards):.2f}, max={max(rewards):.2f}, std={np.std(rewards):.2f}"
            )

        return PhaseTrainingResult(
            rewards,
            total_goals_for,
            total_goals,
            final_episodes,
            _metric_snapshots,
            best_model_path=best_model_path,
            best_validation_report_path=best_validation_report_path,
            selected_best_this_run=selected_best_this_run,
        )

    finally:
        # Restaurer la configuration originale
        game_config.MAX_SCORE = original_max_score
        if use_mixed_fine_tuning:
            game_config.BONUSES_ENABLED = original_bonuses_enabled
        if anneal_enabled or fixed_micro_scale is not None:
            ai_config.MICRO_REWARD_SCALE = original_micro_reward_scale


def create_optimized_agent():
    """Create a DQN agent with optimized hyperparameters."""
    return DQNAgent(
        state_size=EXPECTED_STATE_SIZE,
        action_size=EXPECTED_ACTION_SIZE,
        lr=0.001,
        gamma=0.99,
        epsilon=1.0,
        epsilon_decay=0.998,
        epsilon_min=0.05,
        batch_size=64,
        memory_size=20000,
        train_frequency=1,
        min_replay_size=1000,
    )


def _pipeline_args(args, **overrides):
    """Return a shallow copy of args with the given overrides applied."""
    import copy

    p = copy.copy(args)
    for k, v in overrides.items():
        setattr(p, k, v)
    return p


def _pipeline_load_best(agent, phase_results: Mapping[str, Any], label: str) -> Path | None:
    """Load a best checkpoint reported by the current phase validation run."""
    path_value = phase_results.get("best_model_path")
    if not path_value:
        print(f"[pipeline] No current-run validated {label} best model, keeping current weights.")
        return None
    path = Path(path_value)
    if not path.exists():
        print(
            f"[pipeline] Validated {label} best model missing at {path}, keeping current weights."
        )
        return None
    agent.load_model(str(path))
    print(f"[pipeline] Loaded {label} best model: {path}")
    return path


def train_pipeline(agent, args):
    """Full training pipeline: pre-train → curriculum → anneal → self-play.

    Continue the live learner across phases. Best playable checkpoints are
    retained separately and must not rewind a completed reward curriculum.
    """

    output_dir = Path(args.output_dir)
    prefix = args.model_prefix

    def _ep(attr, fallback):
        v = getattr(args, attr, 0)
        return v if v > 0 else fallback

    curriculum_ep = _ep("pipeline_curriculum_episodes", args.episodes_per_phase)
    anneal_ep = _ep("pipeline_anneal_episodes", args.total_episodes)
    selfplay_ep = _ep("pipeline_selfplay_episodes", args.total_episodes)

    total_ep = 0

    # ── Phase 1: Curriculum ────────────────────────────────────────────────
    print(f"\n{'=' * 60}")
    print("PIPELINE — Phase 1: Curriculum training")
    print(f"{'=' * 60}")
    p1_prefix = f"{prefix}_p1_curriculum"
    p1_args = _pipeline_args(
        args,
        mode="curriculum",
        episodes_per_phase=curriculum_ep,
        model_prefix=p1_prefix,
        anneal_micro_rewards=False,
        micro_reward_scale=None,
        self_play_interval=0,
        # disable pretrain here (already handled by caller if needed)
        pretrain=False,
    )
    agent, p1_results = train_with_curriculum(agent, p1_args)
    total_ep += p1_results.get("total_episodes", 0)

    # ── Phase 2: Reward annealing ──────────────────────────────────────────
    print(f"\n{'=' * 60}")
    print("PIPELINE — Phase 2: Reward annealing (dense → sparse)")
    print(f"{'=' * 60}")
    p2_prefix = f"{prefix}_p2_anneal"
    anneal_opponent = (
        getattr(args, "pipeline_anneal_opponent", None)
        or getattr(args, "phase2_opponent", None)
        or getattr(args, "opponent", None)
        or "follow_ball"
    )
    p2_args = _pipeline_args(
        args,
        mode="single",
        total_episodes=anneal_ep,
        model_prefix=p2_prefix,
        opponent=anneal_opponent,
        anneal_micro_rewards=True,
        micro_reward_anneal_episodes=anneal_ep,
        micro_reward_scale=None,
        self_play_interval=0,
        pretrain=False,
        load_model=None,
        load_checkpoint=None,
    )
    agent, p2_results = train_single_phase(agent, p2_args)
    total_ep += p2_results.get("total_episodes", 0)

    # ── Phase 3: Self-play ─────────────────────────────────────────────────
    print(f"\n{'=' * 60}")
    print("PIPELINE — Phase 3: Self-play (pure goal rewards)")
    print(f"{'=' * 60}")
    p3_prefix = f"{prefix}_p3_selfplay"
    sp_interval = getattr(args, "self_play_interval", 0) or 500
    p3_args = _pipeline_args(
        args,
        mode="single",
        total_episodes=selfplay_ep,
        model_prefix=p3_prefix,
        anneal_micro_rewards=False,
        micro_reward_scale=0.0,
        self_play_interval=sp_interval,
        pretrain=False,
        load_model=None,
        load_checkpoint=None,
    )
    agent, p3_results = train_single_phase(agent, p3_args)
    total_ep += p3_results.get("total_episodes", 0)

    # Copy final best to top-level name for convenience
    final_best_value = p3_results.get("best_model_path")
    top_best = output_dir / f"{prefix}_pipeline_best.pth"
    pipeline_best_path = None
    if final_best_value:
        import shutil

        final_best = Path(final_best_value)
        shutil.copy2(str(final_best), str(top_best))
        pipeline_best_path = top_best
        print(f"\n[pipeline] Best self-play model → {top_best}")
        report_value = p3_results.get("best_validation_report_path")
        if report_value:
            report_path = Path(report_value)
            if report_path.exists():
                top_report = top_best.with_suffix(".validation.json")
                shutil.copy2(str(report_path), str(top_report))
                print(f"[pipeline] Best self-play validation report → {top_report}")
    else:
        print("\n[pipeline] No current-run validated self-play best model to publish.")

    return agent, {
        "total_episodes": total_ep,
        "final_model_path": p3_results.get("final_model_path", ""),
        "best_model_path": str(pipeline_best_path) if pipeline_best_path else None,
        "curriculum_win_rate": p1_results.get("phase2_winrate", p1_results.get("win_rate", 0)),
        "anneal_win_rate": p2_results.get("win_rate", 0),
        "selfplay_win_rate": p3_results.get("win_rate", 0),
    }


def train_with_curriculum(agent, args):
    """
    Training with curriculum learning
    Phase 1: Basic training (simple game)
    Phase 2: Advanced training (with bonuses and complexity)
    """
    print("=== CURRICULUM LEARNING TRAINING ===")
    _t_start = time.time()

    # Get ball configuration
    ball_direction, ball_angle = get_ball_config_from_args(args)
    training_manager = TrainingManager(
        headless=args.headless,
        initial_ball_direction=ball_direction,
        initial_ball_angle=ball_angle,
    )

    try:
        # Phase 1: Basic training
        phase1_opponent = get_opponent(args.phase1_opponent)
        phase1_result = _phase_training_result(
            train_phase(
                agent,
                phase1_opponent,
                f"Basic learning vs {args.phase1_opponent}",
                args.episodes_per_phase,
                training_manager,
                args,
                phase_num=1,
                total_phases=2,
            )
        )
        (
            phase1_rewards,
            phase1_goals_for,
            phase1_goals,
            phase1_episodes,
            phase1_snapshots,
        ) = phase1_result

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
        phase2_result = _phase_training_result(
            train_phase(
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
        )
        (
            phase2_rewards,
            phase2_goals_for,
            phase2_goals,
            phase2_episodes,
            phase2_snapshots,
        ) = phase2_result

        # Save final model
        final_model_path = Path(args.output_dir) / f"{args.model_prefix}_final.pth"
        agent.save_model(str(final_model_path))
        print(f"Final model saved: {final_model_path}")

        # Create training plots
        if args.save_plots:
            create_training_plots(
                phase1_rewards,
                phase2_rewards,
                phase1_goals_for / phase1_goals * 100 if phase1_goals > 0 else 50.0,
                phase2_goals_for / phase2_goals * 100 if phase2_goals > 0 else 50.0,
                args,
            )
            all_snapshots = phase1_snapshots + phase2_snapshots
            create_detailed_metrics_plot(Path(args.output_dir), args.model_prefix, all_snapshots)

        if getattr(args, "save_metrics", True):
            all_snapshots = phase1_snapshots + phase2_snapshots
            save_metrics_json(
                Path(args.output_dir),
                args.model_prefix,
                agent,
                args,
                all_snapshots,
                time.time() - _t_start,
                phase1_episodes + phase2_episodes,
            )

        all_goals_for = phase1_goals_for + phase2_goals_for
        all_goals = phase1_goals + phase2_goals
        maybe_save_checkpoint_evaluation(
            final_model_path,
            args,
            final_model_training_context(
                args,
                label="curriculum_final",
                total_episodes=phase1_episodes + phase2_episodes,
                rewards=phase1_rewards + phase2_rewards,
                goals_for=all_goals_for,
                goals=all_goals,
            ),
        )

        best_result = phase2_result.best_model_path or phase1_result.best_model_path
        best_report = (
            phase2_result.best_validation_report_path or phase1_result.best_validation_report_path
        )
        return agent, {
            "phase1_avg": (
                np.mean(phase1_rewards[-100:])
                if len(phase1_rewards) >= 100
                else np.mean(phase1_rewards)
                if phase1_rewards
                else 0
            ),
            "phase1_winrate": (phase1_goals_for / phase1_goals * 100 if phase1_goals > 0 else 50.0),
            "phase2_avg": (
                np.mean(phase2_rewards[-100:])
                if len(phase2_rewards) >= 100
                else np.mean(phase2_rewards)
                if phase2_rewards
                else 0
            ),
            "phase2_winrate": (phase2_goals_for / phase2_goals * 100 if phase2_goals > 0 else 50.0),
            "total_episodes": phase1_episodes + phase2_episodes,
            "final_model_path": str(final_model_path),
            "best_model_path": str(best_result) if best_result else None,
            "best_validation_report_path": str(best_report) if best_report else None,
        }
    finally:
        training_manager.cleanup()


def train_single_phase(agent, args):
    """Training with a single phase (no curriculum)"""
    print("=== SINGLE PHASE TRAINING ===")
    _t_start = time.time()

    # Get ball configuration
    ball_direction, ball_angle = get_ball_config_from_args(args)
    training_manager = TrainingManager(
        headless=args.headless,
        initial_ball_direction=ball_direction,
        initial_ball_angle=ball_angle,
    )

    opponent = None
    try:
        self_play_interval = getattr(args, "self_play_interval", 0)
        if self_play_interval > 0:
            base_mix = parse_opponent_mix(
                getattr(args, "self_play_base_mix", DEFAULT_OPPONENT_MIX_CLI)
            )
            mix_rng = random.Random(getattr(args, "opponent_mix_seed", 0))
            snap_dir = Path(getattr(args, "league_dir", None) or Path(args.output_dir) / "league")
            opponent = SelfPlaySampler(
                base_mix=base_mix,
                rng=mix_rng,
                snapshot_dir=snap_dir,
                pool_size=args.self_play_pool_size,
                snapshot_weight=args.self_play_weight,
                champion_size=getattr(args, "league_champion_size", 5),
                historical_size=getattr(args, "league_historical_size", 5),
                coverage_floor=getattr(args, "league_coverage_floor", 0.2),
                checkpoints=getattr(args, "league_opponent", ()),
                exploiter_target=getattr(args, "league_exploiter_target", None),
            )
            if not opponent.league.snapshot_references():
                opponent.add_snapshot(agent)
            phase_label = "Self-play training"
            print(
                f"[self-play] pool size={args.self_play_pool_size}, "
                f"snapshot weight={args.self_play_weight:.0%}, "
                f"update interval={self_play_interval} episodes"
            )
            base_summary = ", ".join(f"{w.name}:{w.weight:.2f}" for w in base_mix)
            print(f"[self-play] base mix: {base_summary}")
        else:
            opponent = get_opponent(args.opponent)
            phase_label = f"Training vs {args.opponent}"

        phase_result = _phase_training_result(
            train_phase(
                agent,
                opponent,
                phase_label,
                args.total_episodes,
                training_manager,
                args,
                phase_num=1,
                total_phases=1,
            )
        )
        rewards, goals_for, goals, episodes, metric_snapshots = phase_result

        # Save final model
        final_model_path = Path(args.output_dir) / f"{args.model_prefix}_final.pth"
        agent.save_model(str(final_model_path))
        print(f"Final model saved: {final_model_path}")

        if isinstance(opponent, SelfPlaySampler):
            opponent.add_snapshot(agent, role=getattr(args, "league_role", "snapshot"))

        # Create training plots
        if args.save_plots:
            create_single_phase_plots(rewards, goals_for / goals * 100 if goals > 0 else 50.0, args)
            create_detailed_metrics_plot(Path(args.output_dir), args.model_prefix, metric_snapshots)

        if getattr(args, "save_metrics", True):
            save_metrics_json(
                Path(args.output_dir),
                args.model_prefix,
                agent,
                args,
                metric_snapshots,
                time.time() - _t_start,
                episodes,
            )

        maybe_save_checkpoint_evaluation(
            final_model_path,
            args,
            final_model_training_context(
                args,
                label="single_phase_final",
                total_episodes=episodes,
                rewards=rewards,
                goals_for=goals_for,
                goals=goals,
            ),
        )

        return agent, {
            "avg_reward": (
                np.mean(rewards[-100:])
                if len(rewards) >= 100
                else np.mean(rewards)
                if rewards
                else 0
            ),
            "win_rate": goals_for / goals * 100 if goals > 0 else 50.0,
            "total_episodes": episodes,
            "final_model_path": str(final_model_path),
            "best_model_path": (
                str(phase_result.best_model_path) if phase_result.best_model_path else None
            ),
            "best_validation_report_path": (
                str(phase_result.best_validation_report_path)
                if phase_result.best_validation_report_path
                else None
            ),
        }
    finally:
        training_manager.cleanup()
        if isinstance(opponent, SelfPlaySampler):
            opponent.cleanup()


def train_progressive_curriculum(agent, args):
    """Progressive 4-phase curriculum learning."""
    print("=== PROGRESSIVE 4-PHASE CURRICULUM LEARNING ===")
    _t_start = time.time()
    print("Phase 1: Learning to hit the ball (vs stationary opponent)")
    print("Phase 2: Learning to return the ball")
    print("Phase 3: Playing vs intelligent AI")
    print("Phase 4: Mastering with bonuses")

    original_game_values = {
        "BONUSES_ENABLED": game_config.BONUSES_ENABLED,
        "GAME_SPEED_MULTIPLIER": game_config.GAME_SPEED_MULTIPLIER,
        "FPS": game_config.FPS,
    }
    original_ai_values = {
        "SCORE_REWARD": ai_config.SCORE_REWARD,
        "LOSE_PENALTY": ai_config.LOSE_PENALTY,
        "PADDLE_HIT_REWARD": ai_config.PADDLE_HIT_REWARD,
        "USE_PROXIMITY_REWARD": ai_config.USE_PROXIMITY_REWARD,
        "PROXIMITY_REWARD_FACTOR": ai_config.PROXIMITY_REWARD_FACTOR,
        "PROXIMITY_PENALTY_FACTOR": ai_config.PROXIMITY_PENALTY_FACTOR,
        "MAX_PROXIMITY_REWARD": ai_config.MAX_PROXIMITY_REWARD,
        "SHOW_OPTIMAL_POINTS_GUI": ai_config.SHOW_OPTIMAL_POINTS_GUI,
    }
    initial_ball_direction = args.ball_direction
    training_manager = TrainingManager(headless=args.headless, fast_gui=args.fast_gui)
    all_phase_data = []
    total_episodes_count = 0
    phase_results: dict[str, PhaseTrainingResult] = {}

    def restore_training_globals() -> None:
        for name, value in original_ai_values.items():
            setattr(ai_config, name, value)
        for name, value in original_game_values.items():
            setattr(game_config, name, value)
        args.ball_direction = initial_ball_direction

    def avg_tail(values: Sequence[float], window: int) -> float:
        if len(values) >= window:
            return float(np.mean(values[-window:]))
        return float(np.mean(values)) if values else 0.0

    def win_rate(result: PhaseTrainingResult) -> float:
        return result.goals_for / result.goals * 100 if result.goals > 0 else 50.0

    def run_progressive_phase(
        *,
        key: str,
        display_name: str,
        objective: str,
        opponent_type: str,
        label: str,
        phase_num: int,
        bonuses_enabled: bool,
        reward_overrides: Mapping[str, Any] | None = None,
        game_overrides: Mapping[str, Any] | None = None,
        ball_direction: Any | None = None,
        reduce_exploration: bool = False,
        save_suffix: str | None = None,
    ) -> PhaseTrainingResult:
        nonlocal total_episodes_count

        if reduce_exploration and args.progressive_epsilon_reduction > 0:
            agent.epsilon = max(
                agent.epsilon * args.progressive_epsilon_reduction,
                agent.epsilon_min,
            )
            print(f"Epsilon reduced to: {agent.epsilon:.3f}")

        if ball_direction is not None:
            args.ball_direction = ball_direction
        game_config.BONUSES_ENABLED = bonuses_enabled
        for name, value in (reward_overrides or {}).items():
            setattr(ai_config, name, value)
        for name, value in (game_overrides or {}).items():
            setattr(game_config, name, value)

        opponent = get_opponent(opponent_type)
        try:
            result = _phase_training_result(
                train_phase(
                    agent,
                    opponent,
                    label,
                    args.progressive_episodes_per_phase,
                    training_manager,
                    args,
                    phase_num=phase_num,
                    total_phases=4,
                    start_episode=total_episodes_count,
                )
            )
        finally:
            restore_training_globals()

        total_episodes_count += result.episodes
        all_phase_data.append(
            {
                "name": display_name,
                "rewards": result.rewards,
                "goals_for": result.goals_for,
                "goals": result.goals,
                "episodes": result.episodes,
                "objective": objective,
            }
        )
        phase_results[key] = result

        if args.save_phase_models and save_suffix:
            phase_path = Path(args.output_dir) / f"{args.model_prefix}_{save_suffix}.pth"
            agent.save_model(str(phase_path))
            print(f"Phase {phase_num} model saved: {phase_path}")
        return result

    try:
        print("\n🎯 PHASE 1: Learning to Hit the Ball")
        print("Objective: Basic ball contact and paddle control")
        if args.phase1_training_opponent == "dummy":
            print("Opponent: Dummy AI (completely stationary - never interferes)")
            print("Focus: Pure ball contact learning without any opponent interference")
        else:
            print("Opponent: Training Dummy AI (minimal predictable movement)")
            print("Focus: Ball contact with minimal, non-threatening opponent movement")
        phase1 = run_progressive_phase(
            key="phase1",
            display_name="Hit Ball",
            objective="Ball contact",
            opponent_type=args.phase1_training_opponent,
            label=(
                "Ball contact training vs "
                f"{args.phase1_training_opponent.replace('_', ' ').title()}"
            ),
            phase_num=1,
            bonuses_enabled=False,
            reward_overrides={
                "SCORE_REWARD": 0,
                "LOSE_PENALTY": 0,
                "PADDLE_HIT_REWARD": 1,
                "USE_PROXIMITY_REWARD": True,
                "PROXIMITY_REWARD_FACTOR": 0.1,
                "PROXIMITY_PENALTY_FACTOR": 0.1,
                "MAX_PROXIMITY_REWARD": 0.5,
                "SHOW_OPTIMAL_POINTS_GUI": True,
            },
            game_overrides={"FPS": 300.0},
            ball_direction=[-math.pi / 12, math.pi / 12, -1],
            save_suffix="phase1_hit",
        )

        print("\n🏓 PHASE 2: Learning to Return the Ball")
        print("Objective: Return ball to opponent's side")
        print("Opponent: Defensive AI (predictable positioning)")
        phase2 = run_progressive_phase(
            key="phase2",
            display_name="Return Ball",
            objective="Ball return",
            opponent_type="defensive",
            label="Ball return training vs Defensive AI",
            phase_num=2,
            bonuses_enabled=False,
            reward_overrides={
                "SCORE_REWARD": 0.5,
                "LOSE_PENALTY": -0.5,
                "PADDLE_HIT_REWARD": 1,
            },
            reduce_exploration=True,
            save_suffix="phase2_return",
        )

        print("\n🧠 PHASE 3: Playing vs Intelligent AI")
        print("Objective: Compete against smart opponent")
        print("Opponent: Defensive AI (anticipates ball trajectory)")
        print("Bonuses: DISABLED for focused strategic learning")
        phase3 = run_progressive_phase(
            key="phase3",
            display_name="Strategic Play",
            objective="Smart competition",
            opponent_type="defensive",
            label="Strategic play vs Defensive AI (no bonuses)",
            phase_num=3,
            bonuses_enabled=False,
            ball_direction=initial_ball_direction,
            reduce_exploration=True,
            save_suffix="phase3_strategic",
        )

        print("\n🏆 PHASE 4: Mastering with Bonuses")
        print("Objective: Master complete game with all features")
        print("Opponent: Aggressive AI with bonuses enabled")
        print("Features: Rotating paddles, bonuses, full complexity")
        print("Bonuses: ENABLED for complete game mastery")
        phase4 = run_progressive_phase(
            key="phase4",
            display_name="Master Play",
            objective="Full mastery",
            opponent_type="aggressive",
            label="Master training vs Aggressive AI + Bonuses",
            phase_num=4,
            bonuses_enabled=True,
            reduce_exploration=True,
        )

        game_config.BONUSES_ENABLED = original_game_values["BONUSES_ENABLED"]
        print(f"Bonuses configuration restored to: {original_game_values['BONUSES_ENABLED']}")

        final_model_path = Path(args.output_dir) / f"{args.model_prefix}_progressive_final.pth"
        agent.save_model(str(final_model_path))
        print(f"Final progressive model saved: {final_model_path}")

        all_snapshots = [
            snapshot for result in phase_results.values() for snapshot in result.metric_snapshots
        ]
        if args.save_plots:
            create_progressive_training_plots(all_phase_data, args)
            create_detailed_metrics_plot(Path(args.output_dir), args.model_prefix, all_snapshots)

        if getattr(args, "save_metrics", True):
            save_metrics_json(
                Path(args.output_dir),
                args.model_prefix,
                agent,
                args,
                all_snapshots,
                time.time() - _t_start,
                total_episodes_count,
            )

        all_rewards = [reward for result in phase_results.values() for reward in result.rewards]
        all_goals_for = sum(result.goals_for for result in phase_results.values())
        all_goals = sum(result.goals for result in phase_results.values())
        maybe_save_checkpoint_evaluation(
            final_model_path,
            args,
            final_model_training_context(
                args,
                label="progressive_final",
                total_episodes=total_episodes_count,
                rewards=all_rewards,
                goals_for=all_goals_for,
                goals=all_goals,
            ),
        )

        final_avg = avg_tail(phase4.rewards, 100)
        final_winrate = win_rate(phase4)
        best_phase = next(
            (
                result
                for result in reversed(tuple(phase_results.values()))
                if result.best_model_path
            ),
            None,
        )

        print("\n✨ PROGRESSIVE TRAINING COMPLETED ✨")
        print(f"Total episodes across all phases: {total_episodes_count}")
        print(f"Final performance: {final_avg:.2f} avg reward, {final_winrate:.1f}% point win rate")

        return agent, {
            "phase1_avg": avg_tail(phase1.rewards, 50),
            "phase1_winrate": win_rate(phase1),
            "phase2_avg": avg_tail(phase2.rewards, 50),
            "phase2_winrate": win_rate(phase2),
            "phase3_avg": avg_tail(phase3.rewards, 50),
            "phase3_winrate": win_rate(phase3),
            "phase4_avg": final_avg,
            "phase4_winrate": final_winrate,
            "total_episodes": total_episodes_count,
            "final_model_path": str(final_model_path),
            "best_model_path": (
                str(best_phase.best_model_path)
                if best_phase and best_phase.best_model_path
                else None
            ),
            "best_validation_report_path": (
                str(best_phase.best_validation_report_path)
                if best_phase and best_phase.best_validation_report_path
                else None
            ),
            "all_phase_data": all_phase_data,
        }
    finally:
        restore_training_globals()
        training_manager.cleanup()


def create_training_plots(phase1_rewards, phase2_rewards, phase1_winrate, phase2_winrate, args):
    """Create training progress plots"""
    plt = get_pyplot()

    fig, ((ax1, ax2), (ax3, ax4)) = plt.subplots(2, 2, figsize=(15, 10))

    # Rewards per episode
    all_rewards = phase1_rewards + phase2_rewards
    episodes = range(len(all_rewards))

    ax1.plot(
        episodes[: len(phase1_rewards)],
        phase1_rewards,
        "b-",
        alpha=0.3,
        label="Phase 1 (basic)",
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
            range(len(phase1_rewards) + window - 1, len(all_rewards)),
            rolling2,
            "r-",
            linewidth=2,
        )

    ax1.axvline(
        x=len(phase1_rewards),
        color="k",
        linestyle="--",
        alpha=0.5,
        label="Phase transition",
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
    ax3.set_title("Point Win Rate by Phase")
    ax3.set_ylabel("Point Win Rate (%)")
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
        win_rate = data["goals_for"] / data["goals"] * 100 if data.get("goals", 0) > 0 else 50.0
        win_rates.append(win_rate)
        avg_reward = np.mean(data["rewards"]) if data["rewards"] else 0
        avg_rewards.append(avg_reward)

    bars = ax2.bar(phase_names, win_rates, color=phase_colors, alpha=0.7)
    ax2.set_title("Point Win Rate by Training Phase")
    ax2.set_ylabel("Point Win Rate (%)")
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
            ax.text(
                0.5,
                0.5,
                "No data available",
                ha="center",
                va="center",
                transform=ax.transAxes,
            )
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
            y=mean_reward,
            color="red",
            linestyle="--",
            alpha=0.7,
            label=f"Mean: {mean_reward:.2f}",
        )

        win_rate = (
            phase_data["goals_for"] / phase_data["goals"] * 100
            if phase_data.get("goals", 0) > 0
            else 50.0
        )

        ax.set_title(
            f"Phase {i + 1}: {phase_data['name']}\n"
            f"Point Win Rate: {win_rate:.1f}% | Objective: {phase_data['objective']}"
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
            np.mean(rewards),
            color="red",
            linestyle="--",
            label=f"Mean: {np.mean(rewards):.2f}",
        )
        ax2.set_title("Reward Distribution")
        ax2.set_xlabel("Reward")
        ax2.set_ylabel("Frequency")
        ax2.legend()
        ax2.grid(True, alpha=0.3)

    # Win rate display
    ax3.bar(["Training"], [win_rate], color="blue", alpha=0.7)
    ax3.set_title("Overall Point Win Rate")
    ax3.set_ylabel("Point Win Rate (%)")
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
            np.mean(rewards),
            color="red",
            linestyle="--",
            label=f"Mean: {np.mean(rewards):.2f}",
        )
        ax2.set_title("Reward Distribution")
        ax2.set_xlabel("Reward")
        ax2.set_ylabel("Frequency")
        ax2.legend()
        ax2.grid(True, alpha=0.3)

    # Win rate display
    ax3.bar(["Training"], [win_rate], color="blue", alpha=0.7)
    ax3.set_title("Overall Point Win Rate")
    ax3.set_ylabel("Point Win Rate (%)")
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


def save_metrics_json(
    output_dir: Path,
    model_prefix: str,
    agent: DQNAgent,
    args: Any,
    snapshots: list[dict],
    training_time_seconds: float,
    total_episodes: int,
) -> Path:
    """Save per-interval training metrics to a JSON file for later analysis."""
    path = output_dir / f"{model_prefix}_metrics.json"
    keys = list(snapshots[0].keys()) if snapshots else []
    per_interval = {k: [s[k] for s in snapshots] for k in keys}
    payload = {
        "metadata": {
            "model_prefix": model_prefix,
            "mode": getattr(args, "mode", "single"),
            "architecture": {
                "hidden_size": agent.hidden_size,
                "layer_size": agent.layer_size,
                "hidden_layers": agent.hidden_layers,
                "param_count": agent.param_count,
            },
            "hyperparameters": {
                "lr": getattr(args, "lr", 0.001),
                "gamma": getattr(args, "gamma", 0.99),
                "batch_size": getattr(args, "batch_size", 64),
                "epsilon_min": getattr(args, "epsilon_min", 0.01),
                "memory_size": getattr(args, "memory_size", 10000),
            },
            "training_time_seconds": round(training_time_seconds, 1),
            "total_episodes": total_episodes,
            "inference_us": round(agent.benchmark_inference(), 2),
        },
        "per_interval": per_interval,
    }
    with open(path, "w") as f:
        json.dump(payload, f, indent=2)
    print(f"📊 Metrics saved: {path}")
    return path


def create_detailed_metrics_plot(
    output_dir: Path,
    model_prefix: str,
    snapshots: list[dict],
) -> None:
    """Save a 4×2 detailed metrics plot covering all tracked training signals."""
    if not snapshots:
        return

    plt = get_pyplot()
    episodes = [s["episode"] for s in snapshots]

    def _series(key: str) -> list:
        return [s.get(key) for s in snapshots]

    def _smooth(values: list, w: int = 5) -> list:
        result = []
        for i, v in enumerate(values):
            if v is None:
                result.append(None)
                continue
            window = [x for x in values[max(0, i - w + 1) : i + 1] if x is not None]
            result.append(float(np.mean(window)) if window else None)
        return result

    def _plot_series(ax, x, y, label, color=None, secondary_ax=None, secondary_label=None):
        valid = [(xi, yi) for xi, yi in zip(x, y) if yi is not None]
        if not valid:
            ax.text(
                0.5,
                0.5,
                "N/A",
                ha="center",
                va="center",
                transform=ax.transAxes,
                fontsize=14,
            )
            return
        xs, ys = zip(*valid)
        ax.plot(xs, ys, alpha=0.35, linewidth=0.8, color=color)
        ax.plot(xs, _smooth(list(ys)), linewidth=1.5, label=label, color=color)

    fig, axes = plt.subplots(4, 2, figsize=(14, 16))
    fig.suptitle(f"Training Metrics — {model_prefix}", fontsize=14, fontweight="bold")

    # [0,0] Point win rate
    ax = axes[0, 0]
    _plot_series(ax, episodes, _series("point_win_rate"), "Point win rate", color="steelblue")
    rb = _series("rule_based_win_rate")
    if any(v is not None for v in rb):
        _plot_series(ax, episodes, rb, "vs rule-based", color="green")
    sn = _series("snap_win_rate")
    if any(v is not None for v in sn):
        _plot_series(ax, episodes, sn, "vs snapshot", color="orange")
    ax.axhline(0.5, color="gray", linestyle="--", linewidth=0.8, alpha=0.6)
    ax.set_ylim(0, 1)
    ax.set_title("Point Win Rate")
    ax.set_xlabel("Episode")
    ax.set_ylabel("Win rate")
    ax.legend(fontsize=8)

    # [0,1] Match win rate
    ax = axes[0, 1]
    _plot_series(ax, episodes, _series("match_win_rate"), "Match win rate", color="mediumorchid")
    ax.axhline(0.5, color="gray", linestyle="--", linewidth=0.8, alpha=0.6)
    ax.set_ylim(0, 1)
    ax.set_title("Match Win Rate")
    ax.set_xlabel("Episode")
    ax.set_ylabel("Win rate")

    # [1,0] TD loss
    ax = axes[1, 0]
    _plot_series(ax, episodes, _series("td_loss"), "TD loss", color="crimson")
    ax.set_title("TD Loss (Bellman error)")
    ax.set_xlabel("Episode")
    ax.set_ylabel("Loss")

    # [1,1] Mean Q-value
    ax = axes[1, 1]
    _plot_series(ax, episodes, _series("mean_q"), "Mean Q", color="teal")
    ax.set_title("Mean Q-Value (probe batch)")
    ax.set_xlabel("Episode")
    ax.set_ylabel("Q-value")

    # [2,0] Gradient norm
    ax = axes[2, 0]
    _plot_series(ax, episodes, _series("grad_norm"), "Grad norm", color="darkorange")
    ax.set_title("Gradient Norm (pre-clip)")
    ax.set_xlabel("Episode")
    ax.set_ylabel("L2 norm")

    # [2,1] Epsilon
    ax = axes[2, 1]
    _plot_series(ax, episodes, _series("epsilon"), "Epsilon", color="slategray")
    ax.set_ylim(0, 1)
    ax.set_title("Epsilon (exploration rate)")
    ax.set_xlabel("Episode")
    ax.set_ylabel("Epsilon")

    # [3,0] Draw rate + episode length (dual y-axis)
    ax = axes[3, 0]
    draw_valid = [
        (e, s["draw_rate"]) for e, s in zip(episodes, snapshots) if s.get("draw_rate") is not None
    ]
    if draw_valid:
        dx, dy = zip(*draw_valid)
        ax.plot(dx, dy, color="salmon", alpha=0.5, linewidth=0.8)
        ax.plot(dx, _smooth(list(dy)), color="salmon", linewidth=1.5, label="Draw rate")
    ax.set_ylabel("Draw rate", color="salmon")
    ax2 = ax.twinx()
    len_valid = [
        (e, s["avg_episode_length"])
        for e, s in zip(episodes, snapshots)
        if s.get("avg_episode_length") is not None
    ]
    if len_valid:
        lx, ly = zip(*len_valid)
        ax2.plot(lx, ly, color="royalblue", alpha=0.5, linewidth=0.8)
        ax2.plot(lx, _smooth(list(ly)), color="royalblue", linewidth=1.5, label="Ep. length")
        ax2.set_ylabel("Avg episode length", color="royalblue")
    ax.set_title("Draw Rate & Episode Length")
    ax.set_xlabel("Episode")
    lines1, labels1 = ax.get_legend_handles_labels()
    lines2, labels2 = ax2.get_legend_handles_labels()
    ax.legend(lines1 + lines2, labels1 + labels2, fontsize=8)

    # [3,1] Goals for / against
    ax = axes[3, 1]
    _plot_series(ax, episodes, _series("avg_goals_for"), "Goals for", color="mediumseagreen")
    _plot_series(ax, episodes, _series("avg_goals_against"), "Goals against", color="tomato")
    ax.set_title("Avg Goals For / Against")
    ax.set_xlabel("Episode")
    ax.set_ylabel("Goals per episode")
    ax.legend(fontsize=8)

    plt.tight_layout()
    out_path = output_dir / f"{model_prefix}_detail.png"
    plt.savefig(out_path, dpi=150, bbox_inches="tight")
    plt.close()
    print(f"📊 Detailed metrics plot saved: {out_path}")


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
        with temporary_legacy_reward_evaluation():
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

    parser.add_argument(
        "--seed",
        type=int,
        default=0,
        help="Seed for network initialization and training randomness",
    )

    # Training mode
    parser.add_argument(
        "--mode",
        choices=["curriculum", "single", "continue", "progressive", "pipeline", "frame_pipeline"],
        default="curriculum",
        help=(
            "Training mode: "
            "pipeline = full sequence (curriculum → anneal → self-play); "
            "frame_pipeline = scheduled frame budgets with complete training-state resume; "
            "curriculum = 2-phase rule-based curriculum; "
            "single = one training phase; "
            "continue = resume from checkpoint; "
            "progressive = 4-phase progressive curriculum."
        ),
    )

    # Pipeline mode settings
    parser.add_argument("--pipeline_curriculum_frames", type=int, default=1000000)
    parser.add_argument("--pipeline_anneal_frames", type=int, default=1000000)
    parser.add_argument("--pipeline_selfplay_frames", type=int, default=2000000)
    parser.add_argument("--frame_epsilon_starts", default="0.30,0.10,0.20")
    parser.add_argument("--frame_epsilon_ends", default="0.10,0.05,0.05")
    parser.add_argument("--frame_decay_fraction", type=float, default=0.8)
    parser.add_argument(
        "--frame_shaping_floor",
        type=float,
        default=0.0,
        help="Auxiliary reward scale retained after annealing in frame_pipeline (goals stay unscaled)",
    )
    parser.add_argument(
        "--frame_score_context",
        action="store_true",
        help="Experimental frame_pipeline observation: add absolute own/opponent scores and match target",
    )
    parser.add_argument(
        "--frame_terminal_on_goal",
        action="store_true",
        help="Experimental frame_pipeline objective: end the learning return at every goal while matches continue",
    )
    parser.add_argument("--frame_validation_interval", type=int, default=100000)
    parser.add_argument(
        "--frame_selfplay_opponent",
        choices=("league", "follow_ball"),
        default="league",
        help="Final frame_pipeline phase opponent: league, or fixed Follow Ball for comparison",
    )
    parser.add_argument("--frame_snapshot_interval", type=int, default=100000)
    parser.add_argument("--frame_checkpoint_interval", type=int, default=50000)
    parser.add_argument("--frame_regression_patience", type=int, default=3)
    parser.add_argument(
        "--resume_training",
        default=None,
        help="Resume a frame_pipeline .training.pt bundle with its saved configuration",
    )
    parser.add_argument(
        "--pipeline_curriculum_episodes",
        type=int,
        default=0,
        help="Episodes for the curriculum phase in pipeline mode (0 = use --episodes_per_phase).",
    )
    parser.add_argument(
        "--pipeline_anneal_episodes",
        type=int,
        default=0,
        help="Episodes for the reward-annealing phase in pipeline mode (0 = use --total_episodes).",
    )
    parser.add_argument(
        "--pipeline_selfplay_episodes",
        type=int,
        default=0,
        help="Episodes for the self-play phase in pipeline mode (0 = use --total_episodes).",
    )
    parser.add_argument(
        "--pipeline_anneal_opponent",
        type=str,
        default=None,
        help="Opponent for the annealing phase in pipeline mode (default: --phase2_opponent).",
    )

    # Model and checkpoint management
    parser.add_argument(
        "--output_dir",
        type=str,
        default=None,
        help="Directory to save outputs (default: models, or the resumed training bundle's directory)",
    )
    parser.add_argument(
        "--model_prefix",
        type=str,
        default="dqn_optimized",
        help="Prefix for saved model files",
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
    parser.add_argument(
        "--warm_start_checkpoint",
        type=str,
        default=None,
        help="Path to an older checkpoint to warm-start from (transfers compatible layer weights).",
    )

    # Pretraining using simulator teaching, or the legacy interception task.
    parser.add_argument(
        "--pretrain",
        action="store_true",
        help=(
            "Teach interception, recovery, and scoring using simulator rollouts before RL. "
            "Requires a fresh model; use --pretrain_mode legacy for the old proximity task."
        ),
    )
    parser.add_argument(
        "--pretrain_steps",
        type=int,
        default=5000,
        help="Number of teacher-labelled learner decisions (legacy: synthetic samples)",
    )

    # Checkpoint settings
    parser.add_argument(
        "--checkpoint_interval",
        type=int,
        default=50000,
        help="Save checkpoint every N episodes (0 to disable)",
    )
    parser.add_argument(
        "--checkpoint_dir",
        type=str,
        default=None,
        help="Directory to save training checkpoints",
    )
    parser.add_argument(
        "--save_phase_models",
        action="store_true",
        help="Save models after each training phase",
    )

    # DQN hyperparameters
    parser.add_argument(
        "--learning_rate",
        type=float,
        default=0.0003,
        help="Learning rate for the neural network",
    )
    parser.add_argument(
        "--gamma", type=float, default=0.99, help="Discount factor for future rewards"
    )
    parser.add_argument("--epsilon_start", type=float, default=1.0, help="Initial exploration rate")
    parser.add_argument("--epsilon_decay", type=float, default=0.998, help="Exploration decay rate")
    parser.add_argument("--epsilon_min", type=float, default=0.05, help="Minimum exploration rate")
    parser.add_argument("--batch_size", type=int, default=64, help="Batch size for training")
    parser.add_argument(
        "--memory_size",
        type=int,
        default=20000,
        help="Size of the replay memory buffer",
    )
    parser.add_argument(
        "--hidden_size",
        type=int,
        default=256,
        help="Hidden layer width of the Q-network (first layer; subsequent layers halve geometrically)",
    )
    parser.add_argument(
        "--layer_size",
        type=int,
        default=2,
        help="Number of hidden layers in the Q-network",
    )
    parser.add_argument(
        "--hidden_layers",
        type=parse_hidden_layers,
        default=None,
        help=(
            "Comma-separated explicit hidden layer widths, e.g. '256,128,64'. "
            "Overrides --hidden_size/--layer_size with a fully manual architecture."
        ),
    )
    # Training configuration
    parser.add_argument(
        "--episodes_per_phase",
        type=int,
        default=200,
        help="Number of episodes per training phase (curriculum mode)",
    )
    parser.add_argument(
        "--total_episodes",
        type=int,
        default=1000,
        help="Total number of episodes (single mode)",
    )
    parser.add_argument(
        "--max_steps_per_episode",
        type=int,
        default=10000,
        help="Maximum steps per episode (10000 ≈ 166.7s at 60 Hz)",
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
    reward_shaping_group = parser.add_mutually_exclusive_group()
    reward_shaping_group.add_argument(
        "--reward_shaping",
        choices=REWARD_SHAPING_MODES,
        default="legacy",
        help="Reward shaping mode used temporarily for training and evaluation",
    )
    reward_shaping_group.add_argument(
        "--phase3_reward_shaping",
        dest="reward_shaping",
        action="store_const",
        const="phase3",
        help="Enable Phase 3 reward shaping with conservative default weights",
    )
    parser.add_argument(
        "--phase3_intercept_progress_reward",
        type=float,
        default=ai_config.PHASE3_INTERCEPT_PROGRESS_REWARD,
        help="Phase 3 reward weight for moving toward the predicted intercept",
    )
    parser.add_argument(
        "--phase3_intercept_distance_penalty",
        type=float,
        default=ai_config.PHASE3_INTERCEPT_DISTANCE_PENALTY,
        help="Phase 3 penalty weight for being far from the predicted intercept",
    )
    parser.add_argument(
        "--phase3_successful_return_reward",
        type=float,
        default=ai_config.PHASE3_SUCCESSFUL_RETURN_REWARD,
        help="Phase 3 reward for a paddle hit that returns the ball toward the opponent",
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
        help=(
            "Opponent for single phase training. Either a named type "
            f"({', '.join(AI_TYPES)}) or 'dqn:<path>' to use a frozen DQN checkpoint."
        ),
    )

    # Self-play settings
    parser.add_argument(
        "--self_play_interval",
        type=int,
        default=0,
        help=(
            "Take a DQN snapshot every N episodes and add it to the opponent pool. "
            "0 = disabled (default). Only applies to --mode single."
        ),
    )
    parser.add_argument(
        "--self_play_pool_size",
        type=int,
        default=5,
        help="Number of recent snapshots retained, in addition to champions and history.",
    )
    parser.add_argument(
        "--self_play_weight",
        type=float,
        default=0.5,
        help=(
            "Fraction of opponent sampling weight given to DQN snapshots (0.0–1.0). "
            "The remainder goes to rule-based AIs from --self_play_base_mix."
        ),
    )
    parser.add_argument(
        "--self_play_base_mix",
        type=str,
        default=DEFAULT_OPPONENT_MIX_CLI,
        help=(
            "Rule-based AI mix used alongside DQN snapshots, in the same "
            "'<name>:<weight>' format as --opponent_mix."
        ),
    )
    parser.add_argument(
        "--self_play_snapshot_min_win_rate",
        type=float,
        default=0.0,
        help=(
            "Minimum recent blended win rate (%%) required to add a self-play snapshot. "
            "Prevents adding snapshots when the agent is regressing. "
            "E.g. 68.0 means skip the snapshot unless recent win rate >= 68%%. "
            "0 = always add (default, original behavior)."
        ),
    )
    parser.add_argument(
        "--self_play_epsilon_boost",
        type=float,
        default=0.0,
        help=(
            "After adding a self-play snapshot, set epsilon to at least this value "
            "so the agent can re-explore against the new, harder opponent. "
            "E.g. 0.15 gives a short burst of exploration. 0 = disabled (default)."
        ),
    )

    # Best model tracking
    parser.add_argument(
        "--no_save_best_model",
        dest="save_best_model",
        action="store_false",
        default=True,
        help="Disable competitive validation and saving a separate validated _best.pth.",
    )

    # Domain randomization
    parser.add_argument(
        "--domain_rand_noise",
        type=float,
        default=0.0,
        help=(
            "Per-episode game parameter randomization: each of BALL_SPEED, PADDLE_SPEED, "
            "FIELD_HEIGHT is sampled uniformly in [base*(1-noise), base*(1+noise)]. "
            "0.0 = disabled (default). 0.2 = ±20%%."
        ),
    )

    # Draw penalty
    parser.add_argument(
        "--draw_penalty",
        type=float,
        default=0.0,
        help=(
            "Reward applied when an episode ends by timeout with no goals scored (a draw). "
            "Should be negative to discourage passive play, e.g. -0.3. "
            "0.0 = disabled (default)."
        ),
    )

    # Reward annealing
    parser.add_argument(
        "--anneal_micro_rewards",
        action="store_true",
        help=(
            "Linearly decay all auxiliary rewards (hits, bonuses, proximity and Phase 3) "
            "to zero over training. "
            "Only +1/-1 goal rewards remain at the end."
        ),
    )
    parser.add_argument(
        "--micro_reward_anneal_episodes",
        type=int,
        default=0,
        help=(
            "Number of episodes over which micro-rewards decay to zero. "
            "Defaults to --total_episodes when --anneal_micro_rewards is set."
        ),
    )
    parser.add_argument(
        "--micro_reward_scale",
        type=float,
        default=None,
        help=(
            "Fixed scale for all auxiliary rewards (hits, bonuses, proximity and Phase 3). "
            "0.0 = pure goal rewards only, 1.0 = full (default). "
            "Overridden episode-by-episode when --anneal_micro_rewards is set."
        ),
    )

    # Evaluation settings
    parser.add_argument(
        "--evaluate",
        action="store_true",
        help="Perform final evaluation against multiple opponents",
    )
    parser.add_argument(
        "--eval_episodes",
        type=int,
        default=100,
        help="Number of episodes for evaluation",
    )
    parser.add_argument(
        "--eval_epsilon",
        type=float,
        default=0.01,
        help="Exploration rate during evaluation",
    )
    parser.add_argument(
        "--save_eval_results",
        action="store_true",
        help="Save evaluation results to JSON file",
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
        choices=AI_TYPES,
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
    parser.add_argument(
        "--eval_gate",
        dest="eval_gates",
        action="append",
        default=[],
        help=(
            "Checkpoint evaluation gate as '<opponent>:<metric>:<minimum>', "
            "for example defensive:win_rate:0.50. May be repeated."
        ),
    )
    parser.add_argument(
        "--stop_when_eval_gates_pass",
        action="store_true",
        help="Stop the current training phase when a saved checkpoint passes all eval gates",
    )

    # Output and logging
    parser.add_argument(
        "--save_plots",
        action=argparse.BooleanOptionalAction,
        default=True,
        help="Save training progress plots",
    )
    parser.add_argument(
        "--save_metrics",
        action=argparse.BooleanOptionalAction,
        default=True,
        help="Save per-interval metrics to {prefix}_metrics.json (default: on)",
    )
    parser.add_argument(
        "--track_q_values",
        action=argparse.BooleanOptionalAction,
        default=True,
        help="Track mean Q-value on a probe batch during training (default: on)",
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

    parser.add_argument(
        "--pretrain_mode",
        choices=("simulator", "legacy", "distill"),
        default="simulator",
        help="simulator: synthetic teacher scenarios; distill: DAgger imitation of the "
        "lookahead planner on real matches (see magic_pong.ai.distillation)",
    )
    parser.add_argument(
        "--distill_opponents",
        default="tracker,striker,follow_ball,aggressive,defensive,predictive,planner",
        help="Comma-separated opponents for --pretrain_mode distill",
    )
    parser.add_argument("--distill_beta_end", type=float, default=0.0)
    parser.add_argument(
        "--distill_action_steps",
        type=int,
        default=30,
        help="Frames each candidate action is held in the planner rollouts used as labels",
    )
    parser.add_argument("--distill_match_steps", type=int, default=3000)
    parser.add_argument("--distill_match_score", type=int, default=3)
    parser.add_argument(
        "--teacher_horizon",
        type=int,
        default=90,
        help="Physics frames simulated per candidate teaching action",
    )
    parser.add_argument("--teacher_action_steps", type=int, default=6)
    parser.add_argument("--teacher_rollout_steps", type=int, default=120)
    parser.add_argument("--teacher_learner_probability", type=float, default=0.5)
    parser.add_argument("--teacher_seed", type=int, default=0)
    parser.add_argument(
        "--competitive_eval_interval",
        type=int,
        default=5000,
        help="Validate every N episodes, plus phase boundaries; 0 disables best selection",
    )
    parser.add_argument("--validation_seeds", default="101,102,103")
    parser.add_argument(
        "--test_seeds",
        default="10001,10002,10003",
        help="Disjoint held-out seeds, never used to select checkpoints",
    )
    parser.add_argument("--validation_opponents", default=",".join(AI_TYPES))
    parser.add_argument("--validation_max_steps", type=int, default=18000)
    parser.add_argument("--validation_max_score", type=int, default=11)
    parser.add_argument(
        "--league_dir",
        default=None,
        help="Persistent league directory; default OUTPUT_DIR/league",
    )
    parser.add_argument(
        "--league_opponent",
        action="append",
        default=[],
        help="Import an independent neural checkpoint; repeat for several opponents",
    )
    parser.add_argument(
        "--league_exploiter_target",
        default=None,
        help="Train a specialist against this frozen checkpoint (single mode)",
    )
    parser.add_argument(
        "--league_role",
        choices=("snapshot", "independent", "exploiter"),
        default="snapshot",
        help="Role assigned to the final league snapshot",
    )
    parser.add_argument("--league_champion_size", type=int, default=5)
    parser.add_argument("--league_historical_size", type=int, default=5)
    parser.add_argument("--league_coverage_floor", type=float, default=0.2)

    args = parser.parse_args(argv)
    try:
        if args.output_dir is None:
            args.output_dir = (
                str(Path(args.resume_training).resolve().parent)
                if args.resume_training
                else "models"
            )
        if (
            args.resume_training
            and Path(args.output_dir).resolve() != Path(args.resume_training).resolve().parent
        ):
            raise ValueError("--output_dir must match the resumed training bundle's directory")
        if args.resume_training and args.mode != "frame_pipeline":
            raise ValueError("--resume_training requires --mode frame_pipeline")
        if args.mode == "frame_pipeline":
            if (
                args.pipeline_curriculum_episodes
                or args.pipeline_anneal_episodes
                or args.pipeline_selfplay_episodes
            ):
                raise ValueError("frame_pipeline uses --pipeline_*_frames, not episode budgets")
            if args.load_checkpoint or args.warm_start_checkpoint:
                raise ValueError(
                    "frame_pipeline uses --load_model for adaptation or --resume_training for exact resume"
                )
            if args.resume_training and (args.load_model or args.pretrain):
                raise ValueError("resume cannot be combined with warm start or pretraining")
            if (
                args.pretrain
                or args.domain_rand_noise
                or args.league_dir
                or args.league_opponent
                or args.league_exploiter_target
            ):
                raise ValueError(
                    "frame_pipeline currently requires pretraining as a separate stage and owns its fixed-physics league"
                )
            from magic_pong.ai.scheduled_training import config_from_args

            if not args.resume_training:
                config_from_args(args)
        _validate_training_extensions(args)
        validate_mixed_fine_tuning_args(args)
        validate_pretrain_args(args)
        validate_reward_shaping_args(args)
        _validate_opponent_arg(args)
        _validate_self_play_args(args)
        args.parsed_opponent_mix = parse_opponent_mix(args.opponent_mix)
        args.parsed_training_score_targets = parse_training_score_targets(args.training_score_mix)
        args.parsed_eval_gates = parse_eval_gates(args.eval_gates)
        if args.stop_when_eval_gates_pass and not args.parsed_eval_gates:
            raise ValueError("--stop_when_eval_gates_pass requires at least one --eval_gate")
        if args.checkpoint_eval_episodes < 0:
            raise ValueError("--checkpoint_eval_episodes must be non-negative")
        if args.checkpoint_eval_opponents_csv:
            _checkpoint_eval_opponents(args)
        if checkpoint_evaluation_enabled(args):
            checkpoint_evaluation_config(args)
    except ValueError as exc:
        parser.error(str(exc))

    return args


def _validate_training_extensions(args):
    from magic_pong.ai.training_validation import validation_config_from_args

    if not 0 <= args.seed < 2**32:
        raise ValueError("--seed must be a non-negative 32-bit integer")
    if args.competitive_eval_interval < 0:
        raise ValueError("--competitive_eval_interval must be non-negative")
    validation_config_from_args(args)
    if not 0 <= args.domain_rand_noise < 1:
        raise ValueError("--domain_rand_noise must be in [0, 1)")
    if args.pretrain and args.pretrain_steps <= 0:
        raise ValueError("--pretrain_steps must be positive")
    if min(args.teacher_horizon, args.teacher_action_steps, args.teacher_rollout_steps) <= 0:
        raise ValueError("Teacher horizon, action steps, and rollout steps must be positive")
    if args.teacher_action_steps > args.teacher_horizon:
        raise ValueError("--teacher_action_steps cannot exceed --teacher_horizon")
    if not 0 <= args.teacher_learner_probability <= 1:
        raise ValueError("--teacher_learner_probability must be in [0, 1]")
    if min(args.league_champion_size, args.league_historical_size) < 0:
        raise ValueError("League retention sizes must be non-negative")
    if not 0 < args.league_coverage_floor <= 1:
        raise ValueError("--league_coverage_floor must be in (0, 1]")
    if args.league_exploiter_target:
        if args.mode != "single":
            raise ValueError("--league_exploiter_target requires --mode single")
        args.self_play_interval = args.self_play_interval or 500
        args.league_role = "exploiter"
    if args.league_opponent and args.mode not in ("single", "pipeline", "continue"):
        raise ValueError("--league_opponent requires single, continue, or pipeline mode")
    if args.league_opponent and not args.self_play_interval and args.mode != "pipeline":
        args.self_play_interval = 500
    for path in [
        *args.league_opponent,
        *([args.league_exploiter_target] if args.league_exploiter_target else []),
    ]:
        if not Path(path).is_file():
            raise ValueError(f"League checkpoint not found: {path}")
    if args.mixed_opponents and (args.self_play_interval > 0 or args.mode == "pipeline"):
        raise ValueError(
            "Use --self_play_base_mix for a league; --mixed_opponents uses a separate sampler"
        )


def main(argv: list[str] | None = None):
    args = parse_arguments(argv)

    if args.mode == "frame_pipeline":
        from magic_pong.ai.scheduled_training import config_from_args
        from magic_pong.ai.scheduled_training import run_frame_pipeline

        if args.resume_training:
            import torch

            from magic_pong.ai.scheduled_training import FramePipelineConfig

            saved = torch.load(args.resume_training, map_location="cpu", weights_only=True)
            config = FramePipelineConfig(**saved["config"])
            print(
                "Resuming saved frame budgets and schedules; saved configuration takes precedence."
            )
        else:
            config = config_from_args(args)
        result = run_frame_pipeline(
            config, args.output_dir, initial_checkpoint=args.load_model, resume=args.resume_training
        )
        print(
            f"Frame pipeline {result['status']}: {result['progress']['total_frames']} frames. "
            f"Playable model: {Path(args.output_dir) / 'best.pth'}; resume: latest.training.pt"
        )
        return result

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

    print(f"Learning rate: {args.learning_rate}")

    if args.reward_shaping == "phase3":
        print("Reward shaping: Phase 3 ENABLED")
        print(
            "   Weights: "
            f"progress={args.phase3_intercept_progress_reward:g}, "
            f"distance_penalty={args.phase3_intercept_distance_penalty:g}, "
            f"successful_return={args.phase3_successful_return_reward:g}"
        )

    if args.mixed_opponents:
        print(f"Mixed-opponent fine-tuning: ENABLED (seed: {args.opponent_mix_seed})")
        print(f"   Opponent mix: {args.opponent_mix}")
        print(f"   Training MAX_SCORE mix: {args.training_score_mix}")

    if args.parsed_eval_gates:
        gate_summary = ", ".join(
            f"{gate.opponent}:{gate.metric}>={gate.threshold:g}" for gate in args.parsed_eval_gates
        )
        print(f"Checkpoint evaluation gates: {gate_summary}")

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
    elif args.warm_start_checkpoint:
        print(f"Warm-starting v2 agent from v1 checkpoint: {args.warm_start_checkpoint}")
        agent = create_agent_from_args(args)
        warm_start_from_v1_checkpoint(agent, args.warm_start_checkpoint)
    else:
        print("Creating new agent with optimized parameters")
        agent = create_agent_from_args(args)

    # Optional simulator teaching; runtime inference remains purely neural.
    if getattr(args, "pretrain", False):
        pretrain_steps = args.pretrain_steps
        print(f"\nPretraining phase: {pretrain_steps} {args.pretrain_mode} decisions")
        saved_epsilon = agent.epsilon
        saved_epsilon_min = agent.epsilon_min
        try:
            if args.pretrain_mode == "legacy":
                agent.epsilon = agent.epsilon_min = 1.0
                pretrainer = create_pretrainer(y_only=True)
                with ai_config_tmp(PROXIMITY_REWARD_FACTOR=0.1):
                    pretraining_stats = pretrainer.run_pretraining_phase(
                        agent,
                        total_steps=pretrain_steps,
                        steps_per_batch=min(1000, pretrain_steps),
                        player_id=1,
                        verbose=True,
                    )
            elif args.pretrain_mode == "distill":
                pretrainer = create_pretrainer(
                    mode="distill",
                    opponents=tuple(
                        name.strip() for name in args.distill_opponents.split(",") if name.strip()
                    ),
                    teacher_horizon=args.teacher_horizon,
                    teacher_action_steps=args.distill_action_steps,
                    beta_end=args.distill_beta_end,
                    match_steps=args.distill_match_steps,
                    match_score=args.distill_match_score,
                    seed=args.teacher_seed,
                )
                pretraining_stats = pretrainer.run_pretraining_phase(
                    agent,
                    total_steps=pretrain_steps,
                    steps_per_batch=min(10000, pretrain_steps),
                    player_id=1,
                    verbose=True,
                )
            else:
                pretrainer = create_pretrainer(
                    mode="simulator",
                    y_only=False,
                    teacher_horizon=args.teacher_horizon,
                    teacher_action_steps=args.teacher_action_steps,
                    rollout_steps=args.teacher_rollout_steps,
                    learner_probability=args.teacher_learner_probability,
                    seed=args.teacher_seed,
                )
                pretraining_stats = pretrainer.run_pretraining_phase(
                    agent,
                    total_steps=pretrain_steps,
                    steps_per_batch=min(1000, pretrain_steps),
                    player_id=1,
                    verbose=True,
                )
        finally:
            agent.epsilon, agent.epsilon_min = saved_epsilon, saved_epsilon_min
        pretrain_path = Path(args.output_dir) / f"{args.model_prefix}_pretrained.pth"
        agent.save_model(str(pretrain_path))
        stats_path = pretrain_path.with_suffix(".pretraining.json")
        stats_path.write_text(json.dumps(pretraining_stats, indent=2) + "\n")
        print(f"Pretrained checkpoint saved: {pretrain_path}")

    # Training
    if args.mode == "pipeline":
        agent, training_results = train_pipeline(agent, args)
    elif args.mode == "curriculum":
        agent, training_results = train_with_curriculum(agent, args)
    elif args.mode == "single":
        agent, training_results = train_single_phase(agent, args)
    elif args.mode == "progressive":
        agent, training_results = train_progressive_curriculum(agent, args)
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

    if args.mode == "pipeline":
        print(f"Curriculum win rate:  {training_results.get('curriculum_win_rate', 0):.1f}%")
        print(f"Annealing win rate:   {training_results.get('anneal_win_rate', 0):.1f}%")
        print(f"Self-play win rate:   {training_results.get('selfplay_win_rate', 0):.1f}%")
    elif args.mode == "curriculum":
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
