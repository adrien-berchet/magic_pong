"""
Training script for Magic Pong DQN AI
"""

import argparse
import json
import os
import time
from typing import Any

import matplotlib.pyplot as plt
import numpy as np

from magic_pong.ai.models.dqn_ai import DQNAgent
from magic_pong.ai.models.simple_ai import create_ai
from magic_pong.core.game_engine import TrainingManager
from magic_pong.utils.config import ai_config


class DQNTrainer:
    """Training manager for the DQN agent"""

    def __init__(
        self,
        episodes: int = 1000,
        save_interval: int = 100,
        eval_interval: int = 50,
        eval_episodes: int = 10,
        model_dir: str = "models",
    ):
        """
        Args:
            episodes: Number of training episodes
            save_interval: Model save interval
            eval_interval: Evaluation interval
            eval_episodes: Number of evaluation episodes
            model_dir: Directory for saving models
        """
        self.episodes = episodes
        self.save_interval = save_interval
        self.eval_interval = eval_interval
        self.eval_episodes = eval_episodes
        self.model_dir = model_dir

        # Create models directory
        os.makedirs(model_dir, exist_ok=True)

        # Training metrics
        self.training_rewards: list[float] = []
        self.evaluation_scores: list[float] = []
        self.win_rates: list[float] = []

        # For training resume
        self.start_episode = 0
        self.best_avg_reward = float("-inf")
        self.training_history: dict = {
            "episode_rewards": [],
            "win_rates": [],
            "eval_episodes": [],
            "best_model_path": None,
            "training_params": {},
        }

    def save_training_state(
        self, filepath: str, agent: DQNAgent, episode: int, opponent_type: str
    ) -> None:
        """Save the complete training state"""
        training_state = {
            "episode": episode,
            "opponent_type": opponent_type,
            "start_episode": self.start_episode,
            "best_avg_reward": self.best_avg_reward,
            "training_history": self.training_history,
            "trainer_config": {
                "episodes": self.episodes,
                "save_interval": self.save_interval,
                "eval_interval": self.eval_interval,
                "eval_episodes": self.eval_episodes,
                "model_dir": self.model_dir,
            },
        }

        # Save as JSON
        with open(filepath, "w") as f:
            json.dump(training_state, f, indent=2)

        print(f"Training state saved to {filepath}")

    def load_training_state(self, filepath: str) -> dict[str, Any]:
        """Load training state from a file"""
        if not os.path.exists(filepath):
            print(f"No training state found in {filepath}")
            return {}

        with open(filepath) as f:
            training_state: dict[str, Any] = json.load(f)

        # Restore state
        self.start_episode = training_state.get("episode", 0) + 1  # Resume at next episode
        self.best_avg_reward = training_state.get("best_avg_reward", float("-inf"))
        self.training_history = training_state.get(
            "training_history",
            {
                "episode_rewards": [],
                "win_rates": [],
                "eval_episodes": [],
                "best_model_path": None,
                "training_params": {},
            },
        )

        print(f"Training state loaded from {filepath}")
        print(f"Resuming at episode {self.start_episode}")
        print(f"Previous best reward: {self.best_avg_reward:.2f}")

        return training_state

    def find_latest_checkpoint(self, opponent_type: str) -> tuple[str, str] | None:
        """Find the latest available checkpoint"""
        # Search for checkpoint files
        training_state_pattern = f"training_state_vs_{opponent_type}.json"

        checkpoint_files = []
        for file in os.listdir(self.model_dir):
            if file.startswith("checkpoint_ep") and file.endswith(f"_vs_{opponent_type}.pth"):
                # Extract episode number
                try:
                    episode_num = int(file.split("_ep")[1].split("_vs_")[0])
                    checkpoint_files.append((episode_num, file))
                except (ValueError, IndexError):
                    continue

        if not checkpoint_files:
            return None

        # Find the most recent checkpoint
        latest_episode, latest_checkpoint = max(checkpoint_files, key=lambda x: x[0])
        model_path = os.path.join(self.model_dir, latest_checkpoint)
        state_path = os.path.join(self.model_dir, training_state_pattern)

        if os.path.exists(model_path):
            return model_path, state_path
        return None

    def train_against_ai(
        self,
        opponent_type: str = "follow_ball",
        agent_kwargs: dict[str, Any] | None = None,
        resume_training: bool = False,
        checkpoint_path: str | None = None,
    ) -> DQNAgent:
        """
        Train the DQN agent against a simple AI

        Args:
            opponent_type: Type of opponent AI ('random', 'follow_ball', etc.)
            agent_kwargs: Arguments for creating the DQN agent
            resume_training: If True, attempt to resume existing training
            checkpoint_path: Path to a specific checkpoint to load

        Returns:
            DQNAgent: Trained agent
        """
        if agent_kwargs is None:
            agent_kwargs = {}

        print(f"Starting training against {opponent_type}")

        # Configuration for fast training
        ai_config.HEADLESS_MODE = True
        ai_config.FAST_MODE_MULTIPLIER = 10.0

        # Create the DQN agent
        dqn_agent = DQNAgent(name="DQN_Trainee", **agent_kwargs)

        # Attempt to resume training
        if resume_training or checkpoint_path:
            model_path = checkpoint_path
            state_path = None

            if not model_path:
                # Automatically search for the latest checkpoint
                checkpoint_info = self.find_latest_checkpoint(opponent_type)
                if checkpoint_info:
                    model_path, state_path = checkpoint_info

            if model_path and os.path.exists(model_path):
                print(f"Loading model from {model_path}")
                try:
                    dqn_agent.load_model(model_path)

                    # Load training state if available
                    if state_path and os.path.exists(state_path):
                        self.load_training_state(state_path)
                    else:
                        # Search for default state file
                        default_state_path = os.path.join(
                            self.model_dir, f"training_state_vs_{opponent_type}.json"
                        )
                        if os.path.exists(default_state_path):
                            self.load_training_state(default_state_path)

                    print("Training resume successful!")

                except Exception as e:
                    print(f"Error loading: {e}")
                    print("Starting new training...")
                    self.start_episode = 0
            else:
                if resume_training:
                    print("No checkpoint found, starting new training...")
                self.start_episode = 0

        # Adjust total number of episodes
        episodes_remaining = self.episodes - self.start_episode
        if episodes_remaining <= 0:
            print("Training already complete!")
            return dqn_agent

        print(f"Episodes remaining: {episodes_remaining} (total: {self.episodes})")
        print(f"Configuration: {agent_kwargs}")

        # Create the opponent
        opponent = create_ai(opponent_type, name=f"Opponent_{opponent_type}")

        # Create the training manager
        training_manager = TrainingManager(headless=True)

        # Variables for tracking
        episode_rewards = self.training_history.get("episode_rewards", [])
        recent_rewards = episode_rewards[-100:] if episode_rewards else []
        best_avg_reward = self.best_avg_reward

        start_time = time.time()

        for episode in range(self.start_episode, self.episodes):
            # Play a complete episode
            episode_stats = training_manager.train_episode(dqn_agent, opponent, max_steps=1000)
            episode_reward = episode_stats["total_reward_p1"]

            episode_rewards.append(episode_reward)
            recent_rewards.append(episode_reward)

            # Keep only the last 100 episodes
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
                    model_path = os.path.join(self.model_dir, f"best_model_vs_{opponent_type}.pth")
                    dqn_agent.save_model(model_path)
                    self.training_history["best_model_path"] = model_path
                    print(f"  New best model saved! Reward: {avg_reward:.2f}")

            # Periodic save
            if (episode + 1) % self.save_interval == 0:
                model_path = os.path.join(
                    self.model_dir, f"checkpoint_ep{episode+1}_vs_{opponent_type}.pth"
                )
                dqn_agent.save_model(model_path)

                # Save training state
                state_path = os.path.join(self.model_dir, f"training_state_vs_{opponent_type}.json")
                self.training_history["episode_rewards"] = episode_rewards
                self.training_history["training_params"] = agent_kwargs
                self.save_training_state(state_path, dqn_agent, episode, opponent_type)

            # Periodic evaluation
            if (episode + 1) % self.eval_interval == 0:
                win_rate = self.evaluate_agent(dqn_agent, opponent_type)
                self.win_rates.append(win_rate)
                self.training_history["win_rates"].append(win_rate)
                self.training_history["eval_episodes"].append(episode + 1)
                print(f"  Win rate: {win_rate:.1%}")

        # Final save
        final_model_path = os.path.join(self.model_dir, f"final_model_vs_{opponent_type}.pth")
        dqn_agent.save_model(final_model_path)

        # Store metrics
        self.training_rewards = episode_rewards

        print("\nTraining complete!")
        print(f"Total time: {time.time() - start_time:.1f}s")
        print(f"Final average reward: {np.mean(recent_rewards):.2f}")

        return dqn_agent

    def evaluate_agent(self, agent: DQNAgent, opponent_type: str) -> float:
        """
        Evaluate the agent over multiple games

        Args:
            agent: Agent to evaluate
            opponent_type: Opponent type

        Returns:
            float: Win rate
        """
        # Put agent in evaluation mode
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

        # Put agent back in training mode
        agent.set_training_mode(True)

        return wins / self.eval_episodes

    def plot_training_metrics(self, save_path: str | None = None) -> None:
        """Display training metrics"""
        if not self.training_rewards:
            print("No training data to display")
            return

        # Calculate moving average
        window_size = 50
        if len(self.training_rewards) >= window_size:
            moving_avg = np.convolve(
                self.training_rewards, np.ones(window_size) / window_size, mode="valid"
            ).tolist()
        else:
            moving_avg = self.training_rewards

        plt.figure(figsize=(12, 8))

        # Rewards graph
        plt.subplot(2, 2, 1)
        plt.plot(self.training_rewards, alpha=0.3, color="blue", label="Rewards")
        if len(moving_avg) > 0:
            plt.plot(
                range(window_size - 1, len(self.training_rewards)),
                moving_avg,
                color="red",
                label=f"Moving average ({window_size})",
            )
        plt.xlabel("Episode")
        plt.ylabel("Reward")
        plt.title("Reward evolution")
        plt.legend()
        plt.grid(True)

        # Win rate graph
        if self.win_rates:
            plt.subplot(2, 2, 2)
            episodes_eval = np.arange(
                self.eval_interval, len(self.win_rates) * self.eval_interval + 1, self.eval_interval
            )
            plt.plot(episodes_eval, self.win_rates, "o-", color="green")
            plt.xlabel("Episode")
            plt.ylabel("Win rate")
            plt.title("Win rate evolution")
            plt.grid(True)

        # Rewards histogram
        plt.subplot(2, 2, 3)
        plt.hist(self.training_rewards, bins=50, alpha=0.7, color="purple")
        plt.xlabel("Reward")
        plt.ylabel("Frequency")
        plt.title("Reward distribution")
        plt.grid(True)

        # Recent statistics
        plt.subplot(2, 2, 4)
        recent_episodes = min(100, len(self.training_rewards))
        recent_rewards = self.training_rewards[-recent_episodes:]

        stats_text = (
            f"""Statistics (last {recent_episodes} episodes):

Average reward: {np.mean(recent_rewards):.2f}
Median reward: {np.median(recent_rewards):.2f}
Standard deviation: {np.std(recent_rewards):.2f}
Minimum: {np.min(recent_rewards):.2f}
Maximum: {np.max(recent_rewards):.2f}

Final win rate: {self.win_rates[-1]:.1%}"""
            if self.win_rates
            else ""
        )

        plt.text(0.1, 0.5, stats_text, fontsize=12, verticalalignment="center")
        plt.axis("off")
        plt.title("Training statistics")

        plt.tight_layout()

        if save_path:
            plt.savefig(save_path, dpi=300, bbox_inches="tight")
            print(f"Graphs saved to {save_path}")

        plt.show()


def main() -> None:
    """Main training function"""
    parser = argparse.ArgumentParser(description="DQN AI training for Magic Pong")
    parser.add_argument(
        "--episodes", type=int, default=1000, help="Number of training episodes"
    )
    parser.add_argument(
        "--opponent",
        type=str,
        default="follow_ball",
        choices=["random", "follow_ball", "defensive", "aggressive", "predictive"],
        help="Opponent type",
    )
    parser.add_argument("--lr", type=float, default=0.001, help="Learning rate")
    parser.add_argument("--gamma", type=float, default=0.99, help="Discount factor")
    parser.add_argument(
        "--epsilon", type=float, default=1.0, help="Initial epsilon for exploration"
    )
    parser.add_argument(
        "--epsilon_decay", type=float, default=0.995, help="Epsilon decay factor"
    )
    parser.add_argument("--memory_size", type=int, default=10000, help="Replay buffer size")
    parser.add_argument(
        "--batch_size", type=int, default=32, help="Training batch size"
    )
    parser.add_argument(
        "--use_prioritized_replay",
        action="store_true",
        help="Use prioritized experience replay",
    )
    parser.add_argument(
        "--tau", type=float, default=0.005, help="Target network soft update coefficient"
    )
    parser.add_argument(
        "--model_dir", type=str, default="models", help="Directory for saving models"
    )
    parser.add_argument(
        "--plot", action="store_true", help="Display training graphs"
    )
    parser.add_argument(
        "--resume",
        action="store_true",
        help="Resume training from last checkpoint",
    )
    parser.add_argument(
        "--checkpoint",
        type=str,
        default=None,
        help="Path to a specific checkpoint to load",
    )
    parser.add_argument(
        "--list_checkpoints",
        action="store_true",
        help="List available checkpoints and exit",
    )

    args = parser.parse_args()

    # Créer le trainer
    trainer = DQNTrainer(episodes=args.episodes, model_dir=args.model_dir)

    # List checkpoints if requested
    if args.list_checkpoints:
        print(f"Available checkpoints in {args.model_dir}:")
        if not os.path.exists(args.model_dir):
            print("  No models directory found.")
            return

        checkpoint_files = []
        for file in os.listdir(args.model_dir):
            if file.startswith("checkpoint_ep") and file.endswith(".pth"):
                checkpoint_files.append(file)

        if not checkpoint_files:
            print("  No checkpoints found.")
        else:
            checkpoint_files.sort()
            for file in checkpoint_files:
                filepath = os.path.join(args.model_dir, file)
                size_mb = os.path.getsize(filepath) / (1024 * 1024)
                print(f"  {file} ({size_mb:.1f} MB)")

        # Find latest checkpoint for specified opponent
        latest = trainer.find_latest_checkpoint(args.opponent)
        if latest:
            model_path, state_path = latest
            print(f"\nLatest checkpoint for {args.opponent}: {os.path.basename(model_path)}")
        return

    # Agent configuration
    agent_kwargs = {
        "lr": args.lr,
        "gamma": args.gamma,
        "epsilon": args.epsilon,
        "epsilon_decay": args.epsilon_decay,
        "memory_size": args.memory_size,
        "batch_size": args.batch_size,
        "use_prioritized_replay": args.use_prioritized_replay,
        "tau": args.tau,
    }

    # Entraîner l'agent
    trainer.train_against_ai(
        opponent_type=args.opponent,
        agent_kwargs=agent_kwargs,
        resume_training=args.resume,
        checkpoint_path=args.checkpoint,
    )

    # Display graphs if requested
    if args.plot:
        plot_path = os.path.join(args.model_dir, f"training_metrics_vs_{args.opponent}.png")
        trainer.plot_training_metrics(save_path=plot_path)

    print("Training completed successfully!")


if __name__ == "__main__":
    main()
