"""
AI vs AI training example for Magic Pong
"""

from typing import Any

from magic_pong.ai.models.simple_ai import create_ai
from magic_pong.core.game_engine import TrainingManager
from magic_pong.utils.config import ai_config
from magic_pong.utils.config import game_config


def training() -> None:
    """AI vs AI training example"""
    print("=== Magic Pong - AI vs AI Training ===")

    # Configuration for fast training
    ai_config.HEADLESS_MODE = True
    ai_config.FAST_MODE_MULTIPLIER = 10.0
    game_config.GAME_SPEED_MULTIPLIER = 5.0

    # Create training manager
    trainer = TrainingManager(headless=True)

    # Create AIs
    player1 = create_ai("aggressive", name="AggressiveAI_1")
    player2 = create_ai("defensive", name="DefensiveAI_1")

    print(f"Player 1: {player1.name}")
    print(f"Player 2: {player2.name}")
    print(f"Headless mode: {ai_config.HEADLESS_MODE}")
    print(f"Speed multiplier: {game_config.GAME_SPEED_MULTIPLIER}")
    print()

    # Train multiple episodes
    num_episodes = 100
    print(f"Training {num_episodes} episodes...")

    for episode in range(num_episodes):
        trainer.train_episode(player1, player2, max_steps=5000)

        if (episode + 1) % 10 == 0:
            training_stats = trainer.get_training_stats()
            print(f"Episode {episode + 1}/{num_episodes}")
            print(f"  P1 wins: {training_stats['player1_wins']}")
            print(f"  P2 wins: {training_stats['player2_wins']}")
            print(f"  Average length: {training_stats['average_episode_length']:.1f} steps")
            print(f"  Average reward P1: {training_stats['average_rewards']['player1']:.3f}")
            print(f"  Average reward P2: {training_stats['average_rewards']['player2']:.3f}")
            print()

    # Final statistics
    final_stats = trainer.get_training_stats()
    print("=== Final Statistics ===")
    print(f"Total episodes: {final_stats['episodes']}")
    print(f"Total steps: {final_stats['total_steps']}")
    print(
        f"{player1.name} wins: {final_stats['player1_wins']} ({final_stats['player1_wins'] / final_stats['episodes'] * 100:.1f}%)"
    )
    print(
        f"{player2.name} wins: {final_stats['player2_wins']} ({final_stats['player2_wins'] / final_stats['episodes'] * 100:.1f}%)"
    )
    print(f"Average episode length: {final_stats['average_episode_length']:.1f} steps")
    print(f"Average reward {player1.name}: {final_stats['average_rewards']['player1']:.3f}")
    print(f"Average reward {player2.name}: {final_stats['average_rewards']['player2']:.3f}")

    # AI statistics
    p1_stats = player1.get_stats()
    p2_stats = player2.get_stats()
    print(f"\nAI stats {player1.name}: {p1_stats}")
    print(f"AI stats {player2.name}: {p2_stats}")


def tournament() -> None:
    """Tournament between different AI types"""

    print("=== Magic Pong - AI Tournament ===")

    # Fast configuration
    ai_config.HEADLESS_MODE = True
    game_config.GAME_SPEED_MULTIPLIER = 10.0

    # AI types to test
    ai_types = ["random", "follow_ball", "defensive", "aggressive", "predictive"]
    results: dict[str, dict[str, Any]] = {}

    trainer = TrainingManager(headless=True)

    # Round-robin tournament
    for i, ai1_type in enumerate(ai_types):
        for j, ai2_type in enumerate(ai_types):
            if i > j:  # Avoid duplicates and self-matches
                continue

            print(f"\nMatch: {ai1_type} vs {ai2_type}")

            # Create AIs
            player1 = create_ai(ai1_type)
            player2 = create_ai(ai2_type)

            # Play multiple games
            wins_p1 = 0
            wins_p2 = 0
            total_games = 100

            for _ in range(total_games):
                episode_stats = trainer.train_episode(player1, player2, max_steps=3000)
                if episode_stats["winner"] == 1:
                    wins_p1 += 1
                elif episode_stats["winner"] == 2:
                    wins_p2 += 1

            # Record results
            match_key = f"{ai1_type}_vs_{ai2_type}"
            results[match_key] = {
                "player1": ai1_type,
                "player2": ai2_type,
                "wins_p1": wins_p1,
                "wins_p2": wins_p2,
                "total_games": total_games,
            }

            print(f"  {ai1_type}: {wins_p1}/{total_games} wins")
            print(f"  {ai2_type}: {wins_p2}/{total_games} wins")

    # Display rankings
    print("\n=== Tournament Results ===")
    ai_scores: dict[str, int] = dict.fromkeys(ai_types, 0)

    for result in results.values():
        ai_scores[result["player1"]] += result["wins_p1"]
        ai_scores[result["player2"]] += result["wins_p2"]

    # Sort by score
    sorted_ais = sorted(ai_scores.items(), key=lambda x: x[1], reverse=True)

    print("Rankings:")
    for rank, (ai_type, score) in enumerate(sorted_ais, 1):
        print(f"{rank}. {ai_type}: {score} wins")


if __name__ == "__main__":
    import argparse

    parser = argparse.ArgumentParser(description="Magic Pong AI Training")
    parser.add_argument(
        "--mode", choices=["training", "tournament"], default="training", help="Execution mode"
    )

    args = parser.parse_args()

    if args.mode == "training":
        training()
    elif args.mode == "tournament":
        tournament()
