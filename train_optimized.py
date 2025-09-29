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
import time
from datetime import datetime
from pathlib import Path

# Import pour configuration des jeux
import matplotlib.pyplot as plt
import numpy as np
from magic_pong.ai.models.dqn_ai import DQNAgent
from magic_pong.ai.models.simple_ai import create_ai
from magic_pong.core.game_engine import TrainingManager
from magic_pong.utils.config import ai_config, game_config

AI_TYPES = [
    "follow_ball",
    "defensive",
    "aggressive",
    "random",
    "predictive",
    "dummy",
    "training_dummy",
]


def create_agent_from_args(args):
    """Create a DQN agent with parameters from command line arguments"""
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

    if args.ball_direction in direction_configs:
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

    return create_ai(opponent_type, player_id=player_id)


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

    # Patch temporaire du MAX_SCORE pour l'entraînement
    original_max_score = game_config.MAX_SCORE
    if args.training_max_score != original_max_score:
        print(
            f"Configuration temporaire: MAX_SCORE = {args.training_max_score} (au lieu de {original_max_score})"
        )
        game_config.MAX_SCORE = args.training_max_score

    try:
        rewards = []
        wins = 0
        best_avg_reward = float("-inf")
        episodes_since_improvement = 0

        for episode in range(start_episode, start_episode + episodes):
            ball_direction, ball_angle = get_ball_config_from_args(args)
            training_manager.set_ball_initial_direction(ball_direction, ball_angle)

            agent.on_episode_start()

            episode_stats = training_manager.train_episode(
                agent, opponent, max_steps=args.max_steps_per_episode
            )
            episode_reward = episode_stats["total_reward_p1"]
            rewards.append(episode_reward)

            winner = episode_stats.get("winner", 0)
            if winner == 1:
                wins += 1

            # Debug info for troubleshooting
            if args.verbose:
                print(
                    f"    Episode {episode}: reward={episode_reward:.2f}, winner={winner}, agent_wins={wins}"
                )

            # Progress reporting
            if episode % args.log_interval == 0 and episode > start_episode:
                recent_rewards = rewards[-args.log_interval :]
                avg_reward = np.mean(recent_rewards)
                win_rate = wins / (episode - start_episode + 1) * 100

                print(
                    f"  Episode {episode}: avg reward = {avg_reward:.2f}, "
                    f"win rate = {win_rate:.1f}%, epsilon = {agent.epsilon:.3f}"
                )

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
                save_checkpoint(
                    agent,
                    episode,
                    phase_num,
                    start_episode + episodes,
                    rewards,
                    wins,
                    args.checkpoint_dir,
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


def create_optimized_agent():
    """Create a DQN agent with optimized hyperparameters for convergence"""
    return DQNAgent(
        state_size=28,
        action_size=9,
        lr=0.0003,  # Reduced learning rate for stability
        gamma=0.99,
        epsilon=1.0,
        epsilon_decay=0.998,  # Slower decay for more exploration
        epsilon_min=0.05,  # Slightly higher min epsilon
        batch_size=64,  # Larger batch size for stability
        use_prioritized_replay=True,
        tau=0.003,  # Slower soft update
        memory_size=20000,  # Larger buffer
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
    training_manager = TrainingManager(headless=args.headless)

    all_phase_data = []
    total_episodes_count = 0

    initial_ball_direction = args.ball_direction
    args.ball_direction = "cone_left"

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
    ai_config.SCORE_REWARD = 0
    ai_config.LOSE_PENALTY = 0
    ai_config.WALL_HIT_REWARD = 1
    ai_config.USE_PROXIMITY_REWARD = True
    ai_config.PROXIMITY_REWARD_FACTOR = 0.1
    ai_config.PROXIMITY_PENALTY_FACTOR = 0.1
    ai_config.MAX_PROXIMITY_REWARD = 0.5
    ai_config.SHOW_OPTIMAL_POINTS_GUI = True

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
    print("Opponent: Aggressive AI (anticipates ball trajectory)")
    print("Bonuses: DISABLED for focused strategic learning")

    args.ball_direction = initial_ball_direction

    # Further reduce exploration for phase 3
    if args.progressive_epsilon_reduction > 0:
        agent.epsilon = max(agent.epsilon * args.progressive_epsilon_reduction, agent.epsilon_min)
        print(f"Epsilon reduced to: {agent.epsilon:.3f}")

    phase3_opponent = get_opponent("aggressive")
    phase3_rewards, phase3_wins, phase3_episodes = train_phase(
        agent,
        phase3_opponent,
        "Strategic play vs Aggressive AI (no bonuses)",
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


def create_training_plots(phase1_rewards, phase2_rewards, phase1_winrate, phase2_winrate, args):
    """Create training progress plots"""

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

    fig, ((ax1, ax2), (ax3, ax4)) = plt.subplots(2, 2, figsize=(16, 12))

    # Combine all rewards for continuous plot
    all_rewards = []
    phase_boundaries = [0]
    phase_colors = ["blue", "green", "orange", "red"]
    phase_labels = []

    for i, phase_data in enumerate(all_phase_data):
        all_rewards.extend(phase_data["rewards"])
        phase_boundaries.append(len(all_rewards))
        phase_labels.append(f"Phase {i+1}: {phase_data['name']}")

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

    fig, axes = plt.subplots(2, 2, figsize=(16, 12))
    axes = axes.flatten()

    phase_colors = ["blue", "green", "orange", "red"]

    for i, (phase_data, ax) in enumerate(zip(all_phase_data, axes, strict=False)):
        rewards = phase_data["rewards"]
        if not rewards:
            ax.text(0.5, 0.5, "No data available", ha="center", va="center", transform=ax.transAxes)
            ax.set_title(f"Phase {i+1}: {phase_data['name']}")
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
            f"Phase {i+1}: {phase_data['name']}\n"
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
        headless=args.headless, initial_ball_direction=ball_direction, initial_ball_angle=ball_angle
    )

    # Disable exploration for evaluation
    original_epsilon = agent.epsilon
    agent.epsilon = args.eval_epsilon

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

    # Restore epsilon
    agent.epsilon = original_epsilon

    # Save evaluation results
    if args.save_eval_results:
        eval_path = Path(args.output_dir) / f"{args.model_prefix}_evaluation.json"
        with open(eval_path, "w") as f:
            json.dump(results, f, indent=2)
        print(f"Evaluation results saved: {eval_path}")

    # Cleanup training manager
    training_manager.cleanup()

    game_config.MAX_SCORE = original_max_score

    return results


def parse_arguments():
    """Parse command line arguments"""
    parser = argparse.ArgumentParser(
        description="Advanced DQN Training Script with Comprehensive Configuration",
        formatter_class=argparse.ArgumentDefaultsHelpFormatter,
    )

    # Training mode
    parser.add_argument(
        "--mode",
        choices=["curriculum", "single", "continue", "progressive"],
        default="curriculum",
        help="Training mode: curriculum learning (2 phases), single phase, continue from checkpoint, or progressive (4 phases)",
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
    parser.add_argument("--state_size", type=int, default=28, help="Size of the state space")
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
        help="Score needed to win during training (default: 3, game default: 11)",
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

    return parser.parse_args()


def main():
    args = parse_arguments()

    print("🚀 ADVANCED DQN TRAINING STARTED")
    print(f"Mode: {args.mode}")
    print(f"Display: {'Headless (no GUI)' if args.headless else 'GUI enabled'}")
    if args.mode == "progressive":
        print(f"Configuration: {args.progressive_episodes_per_phase} episodes per phase (4 phases)")
    else:
        print(
            f"Configuration: {args.episodes_per_phase if args.mode == 'curriculum' else args.total_episodes} episodes"
        )

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
    else:  # continue mode
        if not (args.load_model or args.load_checkpoint):
            print("Error: Continue mode requires --load_model or --load_checkpoint")
            return
        # For continue mode, use single phase training with loaded model
        agent, training_results = train_single_phase(agent, args)

    training_time = time.time() - start_time

    # Print training summary
    print(f"\n{'='*60}")
    print("TRAINING SUMMARY")
    print(f"{'='*60}")
    print(f"Training time: {training_time/60:.1f} minutes")
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
    else:
        print(f"Average reward: {training_results['avg_reward']:.2f}")
        print(f"Win rate: {training_results['win_rate']:.1f}%")

    print(f"Final model saved: {training_results['final_model_path']}")

    # Final evaluation
    if args.evaluate:
        eval_results = evaluate_final_performance(agent, args)

        print(f"\n{'='*60}")
        print("FINAL EVALUATION")
        print(f"{'='*60}")
        for opponent, results in eval_results.items():
            print(
                f"{opponent:12}: {results['win_rate']:5.1f}% wins, {results['avg_reward']:6.2f} avg reward"
            )

    print(f"\n🎉 Training completed! Models saved in {args.output_dir}/")


if __name__ == "__main__":
    main()
