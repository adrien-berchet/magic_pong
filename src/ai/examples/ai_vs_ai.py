"""
Exemple d'entraînement IA vs IA pour Magic Pong
"""

from typing import Any

from magic_pong.ai.examples.simple_ai import create_ai
from magic_pong.core.game_engine import TrainingManager
from magic_pong.utils.config import ai_config, game_config


def main() -> None:
    """Exemple d'entraînement IA vs IA"""

    print("=== Magic Pong - Entraînement IA vs IA ===")

    # Configuration pour l'entraînement rapide
    ai_config.HEADLESS_MODE = True
    ai_config.FAST_MODE_MULTIPLIER = 10.0
    game_config.GAME_SPEED_MULTIPLIER = 5.0

    # Créer le gestionnaire d'entraînement
    trainer = TrainingManager(headless=True)

    # Créer les IA
    player1 = create_ai("aggressive", 1, name="AggressiveAI_1")
    player2 = create_ai("defensive", 2, name="DefensiveAI_1")

    print(f"Joueur 1: {player1.name}")
    print(f"Joueur 2: {player2.name}")
    print(f"Mode headless: {ai_config.HEADLESS_MODE}")
    print(f"Multiplicateur de vitesse: {game_config.GAME_SPEED_MULTIPLIER}")
    print()

    # Entraîner plusieurs épisodes
    num_episodes = 100
    print(f"Entraînement de {num_episodes} épisodes...")

    for episode in range(num_episodes):
        episode_stats = trainer.train_episode(player1, player2, max_steps=5000)

        if (episode + 1) % 10 == 0:
            training_stats = trainer.get_training_stats()
            print(f"Épisode {episode + 1}/{num_episodes}")
            print(f"  Victoires P1: {training_stats['player1_wins']}")
            print(f"  Victoires P2: {training_stats['player2_wins']}")
            print(f"  Longueur moyenne: {training_stats['average_episode_length']:.1f} steps")
            print(f"  Récompense moyenne P1: {training_stats['average_rewards']['player1']:.3f}")
            print(f"  Récompense moyenne P2: {training_stats['average_rewards']['player2']:.3f}")
            print()

    # Statistiques finales
    final_stats = trainer.get_training_stats()
    print("=== Statistiques finales ===")
    print(f"Total épisodes: {final_stats['episodes']}")
    print(f"Total steps: {final_stats['total_steps']}")
    print(
        f"Victoires {player1.name}: {final_stats['player1_wins']} ({final_stats['player1_wins']/final_stats['episodes']*100:.1f}%)"
    )
    print(
        f"Victoires {player2.name}: {final_stats['player2_wins']} ({final_stats['player2_wins']/final_stats['episodes']*100:.1f}%)"
    )
    print(f"Longueur moyenne des épisodes: {final_stats['average_episode_length']:.1f} steps")
    print(f"Récompense moyenne {player1.name}: {final_stats['average_rewards']['player1']:.3f}")
    print(f"Récompense moyenne {player2.name}: {final_stats['average_rewards']['player2']:.3f}")

    # Statistiques des IA
    p1_stats = player1.get_stats()
    p2_stats = player2.get_stats()
    print(f"\nStats IA {player1.name}: {p1_stats}")
    print(f"Stats IA {player2.name}: {p2_stats}")


def tournament() -> None:
    """Tournoi entre différents types d'IA"""

    print("=== Magic Pong - Tournoi d'IA ===")

    # Configuration rapide
    ai_config.HEADLESS_MODE = True
    game_config.GAME_SPEED_MULTIPLIER = 10.0

    # Types d'IA à tester
    ai_types = ["random", "follow_ball", "defensive", "aggressive", "predictive"]
    results: dict[str, dict[str, Any]] = {}

    trainer = TrainingManager(headless=True)

    # Tournoi round-robin
    for i, ai1_type in enumerate(ai_types):
        for j, ai2_type in enumerate(ai_types):
            if i >= j:  # Éviter les doublons et les matchs contre soi-même
                continue

            print(f"\nMatch: {ai1_type} vs {ai2_type}")

            # Créer les IA
            player1 = create_ai(ai1_type, 1)
            player2 = create_ai(ai2_type, 2)

            # Jouer plusieurs parties
            wins_p1 = 0
            wins_p2 = 0
            total_games = 20

            for game in range(total_games):
                episode_stats = trainer.train_episode(player1, player2, max_steps=3000)
                if episode_stats["winner"] == 1:
                    wins_p1 += 1
                elif episode_stats["winner"] == 2:
                    wins_p2 += 1

            # Enregistrer les résultats
            match_key = f"{ai1_type}_vs_{ai2_type}"
            results[match_key] = {
                "player1": ai1_type,
                "player2": ai2_type,
                "wins_p1": wins_p1,
                "wins_p2": wins_p2,
                "total_games": total_games,
            }

            print(f"  {ai1_type}: {wins_p1}/{total_games} victoires")
            print(f"  {ai2_type}: {wins_p2}/{total_games} victoires")

    # Afficher le classement
    print("\n=== Résultats du tournoi ===")
    ai_scores: dict[str, int] = dict.fromkeys(ai_types, 0)

    for match_key, result in results.items():
        ai_scores[result["player1"]] += result["wins_p1"]
        ai_scores[result["player2"]] += result["wins_p2"]

    # Trier par score
    sorted_ais = sorted(ai_scores.items(), key=lambda x: x[1], reverse=True)

    print("Classement:")
    for rank, (ai_type, score) in enumerate(sorted_ais, 1):
        print(f"{rank}. {ai_type}: {score} victoires")


if __name__ == "__main__":
    import argparse

    parser = argparse.ArgumentParser(description="Entraînement IA Magic Pong")
    parser.add_argument(
        "--mode", choices=["training", "tournament"], default="training", help="Mode d'exécution"
    )

    args = parser.parse_args()

    if args.mode == "training":
        main()
    elif args.mode == "tournament":
        tournament()
