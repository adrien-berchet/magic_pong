"""
DQN training script with pretraining on the optimal point
"""

import argparse
import os
import time
from typing import Any

import matplotlib.pyplot as plt
import numpy as np

from magic_pong.ai.models.dqn_ai import DQNAgent
from magic_pong.ai.models.simple_ai import create_ai
from magic_pong.ai.pretraining import create_pretrainer
from magic_pong.core.game_engine import TrainingManager
from magic_pong.utils.config import ai_config, game_config


class DQNPretrainer:
    """DQN training manager with pretraining on the optimal point"""

    def __init__(
        self,
        episodes: int = 1000,
        pretraining_steps: int = 10000,
        save_interval: int = 100,
        eval_interval: int = 50,
        eval_episodes: int = 10,
        model_dir: str = "models",
    ):
        """
        Args:
            episodes: Number of main training episodes
            pretraining_steps: Number of pretraining steps
            save_interval: Model save interval
            eval_interval: Evaluation interval
            eval_episodes: Number of evaluation episodes
            model_dir: Model save directory
        """
        self.episodes = episodes
        self.pretraining_steps = pretraining_steps
        self.save_interval = save_interval
        self.eval_interval = eval_interval
        self.eval_episodes = eval_episodes
        self.model_dir = model_dir

        # Create models directory
        os.makedirs(model_dir, exist_ok=True)

        # Training metrics
        self.training_rewards = []
        self.pretraining_rewards = []
        self.win_rates = []

        # For training resumption
        self.start_episode = 0
        self.best_avg_reward = float("-inf")
        self.pretraining_completed = False

    def run_pretraining_phase(
        self,
        agent: DQNAgent,
        player_id: int = 1,
        steps_per_batch: int = 1000,
        save_pretrained_model: bool = True,
        y_only: bool = True,
    ) -> dict[str, Any]:
        """
        Execute the pretraining phase on optimal point proximity.

        Args:
            agent: DQN agent to pretrain
            player_id: Player ID (1 for left, 2 for right)
            steps_per_batch: Number of steps per batch
            save_pretrained_model: Save model after pretraining
            y_only: If True, only consider vertical distance for reward

        Returns:
            Pretraining statistics
        """
        print("🎯 === PRETRAINING PHASE ===")
        print("Objective: Learn to approach the optimal interception point")
        print(f"Pretraining steps: {self.pretraining_steps}")
        print()

        # Create pretrainer
        pretrainer = create_pretrainer(y_only=y_only)

        # Enable headless mode for speed
        original_headless = ai_config.HEADLESS_MODE
        original_fast_mode = ai_config.FAST_MODE_MULTIPLIER
        initial_game_speed_multiplier = game_config.GAME_SPEED_MULTIPLIER
        initial_fps = game_config.FPS
        ai_config.USE_PROXIMITY_REWARD = True
        ai_config.PROXIMITY_REWARD_FACTOR = 1
        ai_config.PROXIMITY_PENALTY_FACTOR = 1
        ai_config.MAX_PROXIMITY_REWARD = 1000
        ai_config.HEADLESS_MODE = True
        ai_config.FAST_MODE_MULTIPLIER = 1.0  # No need for high speed during pretraining
        game_config.GAME_SPEED_MULTIPLIER = 5.0
        game_config.FPS = 300.0

        start_time = time.time()

        try:
            # Execute pretraining
            pretraining_stats = pretrainer.run_pretraining_phase(
                agent=agent,
                total_steps=self.pretraining_steps,
                steps_per_batch=steps_per_batch,
                player_id=player_id,
                verbose=True,
            )

            self.pretraining_rewards = pretraining_stats["all_rewards"]
            self.pretraining_completed = True

            # Save pretrained model
            if save_pretrained_model:
                pretrained_model_path = os.path.join(self.model_dir, "pretrained_optimal_point.pth")
                agent.save_model(pretrained_model_path)
                print(f"📁 Pretrained model saved: {pretrained_model_path}")

            elapsed_time = time.time() - start_time
            print(f"\n✅ Pretraining completed in {elapsed_time:.1f}s")
            print(f"   Proximity reward improvement: {pretraining_stats['average_reward']:.3f}")
            print("   Agent ready for main training!")

            return pretraining_stats

        finally:
            # Restore original configuration
            ai_config.HEADLESS_MODE = original_headless
            ai_config.FAST_MODE_MULTIPLIER = original_fast_mode
            game_config.GAME_SPEED_MULTIPLIER = initial_game_speed_multiplier
            game_config.FPS = initial_fps

    def train_with_pretraining(
        self,
        opponent_type: str = "follow_ball",
        agent_kwargs: dict[str, Any] | None = None,
        skip_pretraining: bool = False,
        pretraining_only: bool = False,
        resume_training: bool = False,
    ) -> DQNAgent:
        """
        Train the agent with pretraining then main training.

        Args:
            opponent_type: Opponent type for main training
            agent_kwargs: Arguments for DQN agent creation
            skip_pretraining: Skip pretraining phase
            pretraining_only: Do only pretraining
            resume_training: Resume existing training

        Returns:
            Trained DQN agent
        """
        if agent_kwargs is None:
            agent_kwargs = {}

        print("🚀 === DQN TRAINING WITH PRETRAINING ===")
        print(f"Phase 1: Pretraining ({self.pretraining_steps} steps)")
        print(f"Phase 2: Main training ({self.episodes} episodes vs {opponent_type})")
        print()

        # Add correct state size if not specified
        if "state_size" not in agent_kwargs:
            agent_kwargs["state_size"] = 32  # Correct size for extended state

        # Create DQN agent
        dqn_agent = DQNAgent(name="DQN_Pretrained", **agent_kwargs)

        # Phase 1: Pretraining (unless skipped)
        if not skip_pretraining:
            pretraining_stats = self.run_pretraining_phase(dqn_agent, y_only=True)

            # Plot pretraining results
            self.plot_pretraining_results(pretraining_stats)

            if pretraining_only:
                print("🎯 Pretraining only completed!")
                return dqn_agent

        # Phase 2: Main training
        print("\n🥊 === MAIN TRAINING PHASE ===")
        print(f"Opponent: {opponent_type}")
        print(f"Episodes: {self.episodes}")

        # Configuration for fast training
        ai_config.HEADLESS_MODE = True
        ai_config.FAST_MODE_MULTIPLIER = 10.0

        # Create opponent
        opponent = create_ai(opponent_type, name=f"Opponent_{opponent_type}")

        # Create training manager
        training_manager = TrainingManager(headless=True)

        # Tracking variables
        episode_rewards = []
        recent_rewards = []
        best_avg_reward = self.best_avg_reward

        start_time = time.time()

        for episode in range(self.episodes):
            # Play a complete episode
            episode_stats = training_manager.train_episode(dqn_agent, opponent, max_steps=1000)
            episode_reward = episode_stats["total_reward_p1"]

            episode_rewards.append(episode_reward)
            recent_rewards.append(episode_reward)

            # Keep only last 100 episodes
            if len(recent_rewards) > 100:
                recent_rewards.pop(0)

            # Periodic logging
            if (episode + 1) % 50 == 0:
                avg_reward = np.mean(recent_rewards)
                elapsed_time = time.time() - start_time
                print(f"Episode {episode + 1}/{self.episodes}")
                print(f"  Average reward (last 100): {avg_reward:.2f}")
                print(f"  Epsilon: {dqn_agent.epsilon:.3f}")
                print(f"  Elapsed time: {elapsed_time:.1f}s")
                print(f"  Training steps: {dqn_agent.training_step}")

                # Save best model
                if avg_reward > best_avg_reward:
                    best_avg_reward = avg_reward
                    self.best_avg_reward = best_avg_reward
                    model_path = os.path.join(
                        self.model_dir, f"best_pretrained_vs_{opponent_type}.pth"
                    )
                    dqn_agent.save_model(model_path)
                    print(f"  🏆 New best model saved! Reward: {avg_reward:.2f}")

            # Periodic save
            if (episode + 1) % self.save_interval == 0:
                model_path = os.path.join(
                    self.model_dir, f"checkpoint_pretrained_ep{episode + 1}_vs_{opponent_type}.pth"
                )
                dqn_agent.save_model(model_path)

            # Periodic evaluation
            if (episode + 1) % self.eval_interval == 0:
                win_rate = self.evaluate_agent(dqn_agent, opponent_type)
                self.win_rates.append(win_rate)
                print(f"  📊 Win rate: {win_rate:.1%}")

        # Save final model
        final_model_path = os.path.join(self.model_dir, f"final_pretrained_vs_{opponent_type}.pth")
        dqn_agent.save_model(final_model_path)

        # Store metrics
        self.training_rewards = episode_rewards

        print("\n✅ Main training completed!")
        print(f"Total time: {time.time() - start_time:.1f}s")
        print(f"Final average reward: {np.mean(recent_rewards):.2f}")

        return dqn_agent

    def evaluate_agent(self, agent: DQNAgent, opponent_type: str) -> float:
        """Evaluate the agent over multiple games"""
        # Set agent to evaluation mode
        agent.set_training_mode(False)

        # Create opponent
        opponent = create_ai(opponent_type)

        # Create evaluation manager
        eval_manager = TrainingManager(headless=True)

        wins = 0

        for _ in range(self.eval_episodes):
            # Play a complete game
            episode_stats = eval_manager.train_episode(agent, opponent, max_steps=1000)

            # Check who won (agent is always player 1)
            if episode_stats["winner"] == 1:
                wins += 1

        # Set agent back to training mode
        agent.set_training_mode(True)

        return wins / self.eval_episodes

    def plot_pretraining_results(self, pretraining_stats: dict[str, Any]) -> None:
        """Display pretraining results"""
        rewards = pretraining_stats["all_rewards"]
        if not rewards:
            return

        plt.figure(figsize=(12, 6))

        # Pretraining rewards graph
        plt.subplot(1, 2, 1)
        plt.plot(rewards, alpha=0.7, color="blue", linewidth=0.8)

        # Moving average
        window_size = min(100, len(rewards) // 10)
        if window_size > 1:
            moving_avg = np.convolve(rewards, np.ones(window_size) / window_size, mode="valid")
            plt.plot(
                range(window_size - 1, len(rewards)),
                moving_avg,
                color="red",
                linewidth=2,
                label=f"Moving average ({window_size})",
            )

        plt.xlabel("Pretraining step")
        plt.ylabel("Proximity reward")
        plt.title("Evolution during pretraining")
        plt.legend()
        plt.grid(True, alpha=0.3)

        # Rewards histogram
        plt.subplot(1, 2, 2)
        plt.hist(rewards, bins=50, alpha=0.7, color="green", edgecolor="black")
        plt.xlabel("Proximity reward")
        plt.ylabel("Frequency")
        plt.title("Reward distribution")
        plt.grid(True, alpha=0.3)

        # Statistics
        stats_text = f"""Pretraining statistics:
Average reward: {np.mean(rewards):.3f}
Std deviation: {np.std(rewards):.3f}
Min: {np.min(rewards):.3f}
Max: {np.max(rewards):.3f}
Steps: {len(rewards)}"""

        plt.figtext(0.02, 0.02, stats_text, fontsize=10, verticalalignment="bottom")

        plt.tight_layout()
        plt.subplots_adjust(bottom=0.15)

        # Save plot
        plot_path = os.path.join(self.model_dir, "pretraining_results.png")
        plt.savefig(plot_path, dpi=300, bbox_inches="tight")
        print(f"📊 Pretraining plots saved: {plot_path}")

        plt.show()

    def plot_full_training_results(self) -> None:
        """Display complete results (pretraining + main training)"""
        if not self.training_rewards and not self.pretraining_rewards:
            print("No training data to display")
            return

        plt.figure(figsize=(15, 10))

        # Combined rewards graph
        plt.subplot(2, 2, 1)

        # Pretraining
        if self.pretraining_rewards:
            pretraining_x = np.arange(len(self.pretraining_rewards)) - len(self.pretraining_rewards)
            plt.plot(
                pretraining_x,
                self.pretraining_rewards,
                alpha=0.5,
                color="blue",
                label="Pretraining",
            )

        # Main training
        if self.training_rewards:
            training_x = np.arange(len(self.training_rewards))
            plt.plot(training_x, self.training_rewards, alpha=0.7, color="red", label="Training")

            # Moving average for training
            window_size = min(50, len(self.training_rewards) // 5)
            if window_size > 1:
                moving_avg = np.convolve(
                    self.training_rewards, np.ones(window_size) / window_size, mode="valid"
                )
                plt.plot(
                    training_x[window_size - 1 :],
                    moving_avg,
                    color="darkred",
                    linewidth=2,
                    label=f"Moving average ({window_size})",
                )

        plt.axvline(x=0, color="black", linestyle="--", alpha=0.5, label="Training start")
        plt.xlabel("Step / Episode")
        plt.ylabel("Reward")
        plt.title("Complete reward evolution")
        plt.legend()
        plt.grid(True, alpha=0.3)

        # Win rate
        if self.win_rates:
            plt.subplot(2, 2, 2)
            episodes_eval = np.arange(
                self.eval_interval, len(self.win_rates) * self.eval_interval + 1, self.eval_interval
            )
            plt.plot(episodes_eval, self.win_rates, "o-", color="green", linewidth=2)
            plt.xlabel("Episode")
            plt.ylabel("Win rate")
            plt.title("Win rate evolution")
            plt.grid(True, alpha=0.3)

        # Histogram comparison
        plt.subplot(2, 2, 3)
        if self.pretraining_rewards:
            plt.hist(
                self.pretraining_rewards,
                bins=30,
                alpha=0.5,
                color="blue",
                label="Pretraining",
                density=True,
            )
        if self.training_rewards:
            plt.hist(
                self.training_rewards,
                bins=30,
                alpha=0.5,
                color="red",
                label="Training",
                density=True,
            )
        plt.xlabel("Reward")
        plt.ylabel("Density")
        plt.title("Reward distribution")
        plt.legend()
        plt.grid(True, alpha=0.3)

        # Global statistics
        plt.subplot(2, 2, 4)
        stats_lines = ["Global statistics:\n"]

        if self.pretraining_rewards:
            stats_lines.extend(
                [
                    "Pretraining:",
                    f"  Steps: {len(self.pretraining_rewards)}",
                    f"  Avg reward: {np.mean(self.pretraining_rewards):.3f}",
                    f"  Final reward: {np.mean(self.pretraining_rewards[-100:]):.3f}",
                    "",
                ]
            )

        if self.training_rewards:
            recent_rewards = self.training_rewards[-100:]
            stats_lines.extend(
                [
                    "Main training:",
                    f"  Episodes: {len(self.training_rewards)}",
                    f"  Avg reward: {np.mean(self.training_rewards):.2f}",
                    f"  Final reward: {np.mean(recent_rewards):.2f}",
                    "",
                ]
            )

        if self.win_rates:
            stats_lines.extend(
                [
                    "Performance:",
                    f"  Final win rate: {self.win_rates[-1]:.1%}",
                    f"  Best win rate: {max(self.win_rates):.1%}",
                ]
            )

        plt.text(0.1, 0.5, "\n".join(stats_lines), fontsize=11, verticalalignment="center")
        plt.axis("off")

        plt.tight_layout()

        # Save
        plot_path = os.path.join(self.model_dir, "full_training_results.png")
        plt.savefig(plot_path, dpi=300, bbox_inches="tight")
        print(f"📊 Complete plots saved: {plot_path}")

        plt.show()


def main():
    """Main function with pretraining"""
    parser = argparse.ArgumentParser(
        description="DQN training with pretraining on the optimal point"
    )

    # Training arguments
    parser.add_argument(
        "--episodes", type=int, default=1000, help="Number of main training episodes"
    )
    parser.add_argument(
        "--pretraining_steps",
        type=int,
        default=10000,
        help="Number of pretraining steps on the optimal point",
    )
    parser.add_argument(
        "--opponent",
        type=str,
        default="follow_ball",
        choices=["random", "follow_ball", "defensive", "aggressive", "predictive"],
        help="Opponent type for main training",
    )

    # Network arguments
    parser.add_argument("--lr", type=float, default=0.001, help="Learning rate")
    parser.add_argument(
        "--tau",
        type=float,
        default=0.005,
        help="Coefficient for target network soft updates",
    )
    parser.add_argument("--gamma", type=float, default=0.99, help="Discount factor")
    parser.add_argument(
        "--epsilon", type=float, default=1.0, help="Initial epsilon for exploration"
    )
    parser.add_argument("--epsilon_decay", type=float, default=0.995, help="Epsilon decay factor")
    parser.add_argument(
        "--epsilon_min", type=float, default=0.01, help="Minimum epsilon for exploration"
    )
    parser.add_argument("--memory_size", type=int, default=20000, help="Replay buffer size")
    parser.add_argument("--batch_size", type=int, default=32, help="Training batch size")

    # Control arguments
    parser.add_argument(
        "--skip_pretraining",
        action="store_true",
        help="Skip the pretraining phase",
    )
    parser.add_argument(
        "--pretraining_only",
        action="store_true",
        help="Do only the pretraining phase",
    )
    parser.add_argument("--model_dir", type=str, default="models", help="Model save directory")
    parser.add_argument("--plot", action="store_true", help="Display training plots")

    args = parser.parse_args()

    # Create trainer with pretraining
    trainer = DQNPretrainer(
        episodes=args.episodes,
        pretraining_steps=args.pretraining_steps,
        model_dir=args.model_dir,
    )

    # Agent configuration
    agent_kwargs = {
        "tau": args.tau,
        "lr": args.lr,
        "gamma": args.gamma,
        "epsilon": args.epsilon,
        "epsilon_decay": args.epsilon_decay,
        "epsilon_min": args.epsilon_min,
        "memory_size": args.memory_size,
        "batch_size": args.batch_size,
    }

    print("🎯 Configuration:")
    print(f"   Pretraining: {args.pretraining_steps} steps")
    print(f"   Training: {args.episodes} episodes vs {args.opponent}")
    print(f"   Save directory: {args.model_dir}")
    print()

    # Train the agent
    trainer.train_with_pretraining(
        opponent_type=args.opponent,
        agent_kwargs=agent_kwargs,
        skip_pretraining=args.skip_pretraining,
        pretraining_only=args.pretraining_only,
    )

    # Display plots if requested
    if args.plot:
        trainer.plot_full_training_results()

    print("\n🎉 Training with pretraining completed successfully!")


if __name__ == "__main__":
    main()
