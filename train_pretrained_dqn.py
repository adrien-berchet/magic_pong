"""
Script d'entraînement DQN avec pré-entraînement sur le point optimal
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
from magic_pong.utils.config import ai_config
from magic_pong.utils.config import game_config


class DQNPretrainer:
    """Gestionnaire d'entraînement DQN avec pré-entraînement sur le point optimal"""

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
            episodes: Nombre d'épisodes d'entraînement principal
            pretraining_steps: Nombre d'étapes de pré-entraînement
            save_interval: Intervalle de sauvegarde du modèle
            eval_interval: Intervalle d'évaluation
            eval_episodes: Nombre d'épisodes d'évaluation
            model_dir: Répertoire de sauvegarde des modèles
        """
        self.episodes = episodes
        self.pretraining_steps = pretraining_steps
        self.save_interval = save_interval
        self.eval_interval = eval_interval
        self.eval_episodes = eval_episodes
        self.model_dir = model_dir

        # Créer le répertoire des modèles
        os.makedirs(model_dir, exist_ok=True)

        # Métriques d'entraînement
        self.training_rewards = []
        self.pretraining_rewards = []
        self.win_rates = []

        # Pour la reprise d'entraînement
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
        Exécute la phase de pré-entraînement sur la proximité au point optimal.

        Args:
            agent: Agent DQN à pré-entraîner
            player_id: ID du joueur (1 pour gauche, 2 pour droite)
            steps_per_batch: Nombre d'étapes par batch
            save_pretrained_model: Sauvegarder le modèle après pré-entraînement
            y_only: Si True, ne considère que la distance verticale pour la récompense

        Returns:
            Statistiques du pré-entraînement
        """
        print("🎯 === PHASE DE PRÉ-ENTRAÎNEMENT ===")
        print("Objectif: Apprendre à s'approcher du point optimal d'interception")
        print(f"Étapes de pré-entraînement: {self.pretraining_steps}")
        print()

        # Créer le pré-entraîneur
        pretrainer = create_pretrainer(y_only=y_only)

        # Activer le mode headless pour la vitesse
        original_headless = ai_config.HEADLESS_MODE
        original_fast_mode = ai_config.FAST_MODE_MULTIPLIER
        initial_game_speed_multiplier = game_config.GAME_SPEED_MULTIPLIER
        initial_fps = game_config.FPS
        ai_config.USE_PROXIMITY_REWARD = True
        ai_config.PROXIMITY_REWARD_FACTOR = 1
        ai_config.PROXIMITY_PENALTY_FACTOR = 1
        ai_config.MAX_PROXIMITY_REWARD = 1000
        ai_config.HEADLESS_MODE = True
        ai_config.FAST_MODE_MULTIPLIER = (
            1.0  # Pas besoin de vitesse élevée pour le pré-entraînement
        )
        game_config.GAME_SPEED_MULTIPLIER = 5.0
        game_config.FPS = 300.0

        start_time = time.time()

        try:
            # Exécuter le pré-entraînement
            pretraining_stats = pretrainer.run_pretraining_phase(
                agent=agent,
                total_steps=self.pretraining_steps,
                steps_per_batch=steps_per_batch,
                player_id=player_id,
                verbose=True,
            )

            self.pretraining_rewards = pretraining_stats["all_rewards"]
            self.pretraining_completed = True

            # Sauvegarder le modèle pré-entraîné
            if save_pretrained_model:
                pretrained_model_path = os.path.join(self.model_dir, "pretrained_optimal_point.pth")
                agent.save_model(pretrained_model_path)
                print(f"📁 Modèle pré-entraîné sauvegardé: {pretrained_model_path}")

            elapsed_time = time.time() - start_time
            print(f"\n✅ Pré-entraînement terminé en {elapsed_time:.1f}s")
            print(
                f"   Amélioration de la récompense de proximité: {pretraining_stats['average_reward']:.3f}"
            )
            print("   Agent prêt pour l'entraînement principal!")

            return pretraining_stats

        finally:
            # Restaurer la configuration originale
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
        Entraîne l'agent avec pré-entraînement puis entraînement principal.

        Args:
            opponent_type: Type d'adversaire pour l'entraînement principal
            agent_kwargs: Arguments pour la création de l'agent DQN
            skip_pretraining: Ignorer la phase de pré-entraînement
            pretraining_only: Faire seulement le pré-entraînement
            resume_training: Reprendre un entraînement existant

        Returns:
            Agent DQN entraîné
        """
        if agent_kwargs is None:
            agent_kwargs = {}

        print("🚀 === ENTRAÎNEMENT DQN AVEC PRÉ-ENTRAÎNEMENT ===")
        print(f"Phase 1: Pré-entraînement ({self.pretraining_steps} étapes)")
        print(f"Phase 2: Entraînement principal ({self.episodes} épisodes vs {opponent_type})")
        print()

        # Ajouter la taille d'état correcte si non spécifiée
        if "state_size" not in agent_kwargs:
            agent_kwargs["state_size"] = 28  # Taille correcte pour l'état étendu

        # Créer l'agent DQN
        dqn_agent = DQNAgent(player_id=1, name="DQN_Pretrained", **agent_kwargs)

        # Phase 1: Pré-entraînement (sauf si demandé de l'ignorer)
        if not skip_pretraining:
            pretraining_stats = self.run_pretraining_phase(dqn_agent, y_only=True)

            # Tracer les résultats du pré-entraînement
            self.plot_pretraining_results(pretraining_stats)

            if pretraining_only:
                print("🎯 Pré-entraînement seul terminé!")
                return dqn_agent

        # Phase 2: Entraînement principal
        print("\n🥊 === PHASE D'ENTRAÎNEMENT PRINCIPAL ===")
        print(f"Adversaire: {opponent_type}")
        print(f"Épisodes: {self.episodes}")

        # Configuration pour l'entraînement rapide
        ai_config.HEADLESS_MODE = True
        ai_config.FAST_MODE_MULTIPLIER = 10.0

        # Créer l'adversaire
        opponent = create_ai(opponent_type, player_id=2, name=f"Opponent_{opponent_type}")

        # Créer le gestionnaire d'entraînement
        training_manager = TrainingManager(headless=True)

        # Variables pour le suivi
        episode_rewards = []
        recent_rewards = []
        best_avg_reward = self.best_avg_reward

        start_time = time.time()

        for episode in range(self.episodes):
            # Jouer un épisode complet
            episode_stats = training_manager.train_episode(dqn_agent, opponent, max_steps=1000)
            episode_reward = episode_stats["total_reward_p1"]

            episode_rewards.append(episode_reward)
            recent_rewards.append(episode_reward)

            # Garder seulement les 100 derniers épisodes
            if len(recent_rewards) > 100:
                recent_rewards.pop(0)

            # Logging périodique
            if (episode + 1) % 50 == 0:
                avg_reward = np.mean(recent_rewards)
                elapsed_time = time.time() - start_time
                print(f"Épisode {episode + 1}/{self.episodes}")
                print(f"  Récompense moyenne (100 derniers): {avg_reward:.2f}")
                print(f"  Epsilon: {dqn_agent.epsilon:.3f}")
                print(f"  Temps écoulé: {elapsed_time:.1f}s")
                print(f"  Étapes d'entraînement: {dqn_agent.training_step}")

                # Sauvegarder le meilleur modèle
                if avg_reward > best_avg_reward:
                    best_avg_reward = avg_reward
                    self.best_avg_reward = best_avg_reward
                    model_path = os.path.join(
                        self.model_dir, f"best_pretrained_vs_{opponent_type}.pth"
                    )
                    dqn_agent.save_model(model_path)
                    print(f"  🏆 Nouveau meilleur modèle sauvegardé! Récompense: {avg_reward:.2f}")

            # Sauvegarde périodique
            if (episode + 1) % self.save_interval == 0:
                model_path = os.path.join(
                    self.model_dir, f"checkpoint_pretrained_ep{episode+1}_vs_{opponent_type}.pth"
                )
                dqn_agent.save_model(model_path)

            # Évaluation périodique
            if (episode + 1) % self.eval_interval == 0:
                win_rate = self.evaluate_agent(dqn_agent, opponent_type)
                self.win_rates.append(win_rate)
                print(f"  📊 Taux de victoire: {win_rate:.1%}")

        # Sauvegarder le modèle final
        final_model_path = os.path.join(self.model_dir, f"final_pretrained_vs_{opponent_type}.pth")
        dqn_agent.save_model(final_model_path)

        # Stocker les métriques
        self.training_rewards = episode_rewards

        print("\n✅ Entraînement principal terminé!")
        print(f"Temps total: {time.time() - start_time:.1f}s")
        print(f"Récompense moyenne finale: {np.mean(recent_rewards):.2f}")

        return dqn_agent

    def evaluate_agent(self, agent: DQNAgent, opponent_type: str) -> float:
        """Évalue l'agent sur plusieurs parties"""
        # Mettre l'agent en mode évaluation
        agent.set_training_mode(False)

        # Créer l'adversaire
        opponent = create_ai(opponent_type, player_id=2)

        # Créer le gestionnaire d'évaluation
        eval_manager = TrainingManager(headless=True)

        wins = 0

        for _ in range(self.eval_episodes):
            # Jouer une partie complète
            episode_stats = eval_manager.train_episode(agent, opponent, max_steps=1000)

            # Vérifier qui a gagné
            if episode_stats["winner"] == agent.player_id:
                wins += 1

        # Remettre l'agent en mode entraînement
        agent.set_training_mode(True)

        return wins / self.eval_episodes

    def plot_pretraining_results(self, pretraining_stats: dict[str, Any]) -> None:
        """Affiche les résultats du pré-entraînement"""
        rewards = pretraining_stats["all_rewards"]
        if not rewards:
            return

        plt.figure(figsize=(12, 6))

        # Graphique des récompenses de pré-entraînement
        plt.subplot(1, 2, 1)
        plt.plot(rewards, alpha=0.7, color="blue", linewidth=0.8)

        # Moyenne mobile
        window_size = min(100, len(rewards) // 10)
        if window_size > 1:
            moving_avg = np.convolve(rewards, np.ones(window_size) / window_size, mode="valid")
            plt.plot(
                range(window_size - 1, len(rewards)),
                moving_avg,
                color="red",
                linewidth=2,
                label=f"Moyenne mobile ({window_size})",
            )

        plt.xlabel("Étape de pré-entraînement")
        plt.ylabel("Récompense de proximité")
        plt.title("Évolution pendant le pré-entraînement")
        plt.legend()
        plt.grid(True, alpha=0.3)

        # Histogramme des récompenses
        plt.subplot(1, 2, 2)
        plt.hist(rewards, bins=50, alpha=0.7, color="green", edgecolor="black")
        plt.xlabel("Récompense de proximité")
        plt.ylabel("Fréquence")
        plt.title("Distribution des récompenses")
        plt.grid(True, alpha=0.3)

        # Statistiques
        stats_text = f"""Statistiques du pré-entraînement:
Récompense moyenne: {np.mean(rewards):.3f}
Écart-type: {np.std(rewards):.3f}
Min: {np.min(rewards):.3f}
Max: {np.max(rewards):.3f}
Étapes: {len(rewards)}"""

        plt.figtext(0.02, 0.02, stats_text, fontsize=10, verticalalignment="bottom")

        plt.tight_layout()
        plt.subplots_adjust(bottom=0.15)

        # Sauvegarder le graphique
        plot_path = os.path.join(self.model_dir, "pretraining_results.png")
        plt.savefig(plot_path, dpi=300, bbox_inches="tight")
        print(f"📊 Graphiques du pré-entraînement sauvegardés: {plot_path}")

        plt.show()

    def plot_full_training_results(self) -> None:
        """Affiche les résultats complets (pré-entraînement + entraînement principal)"""
        if not self.training_rewards and not self.pretraining_rewards:
            print("Aucune donnée d'entraînement à afficher")
            return

        plt.figure(figsize=(15, 10))

        # Graphique combiné des récompenses
        plt.subplot(2, 2, 1)

        # Pré-entraînement
        if self.pretraining_rewards:
            pretraining_x = np.arange(len(self.pretraining_rewards)) - len(self.pretraining_rewards)
            plt.plot(
                pretraining_x,
                self.pretraining_rewards,
                alpha=0.5,
                color="blue",
                label="Pré-entraînement",
            )

        # Entraînement principal
        if self.training_rewards:
            training_x = np.arange(len(self.training_rewards))
            plt.plot(
                training_x, self.training_rewards, alpha=0.7, color="red", label="Entraînement"
            )

            # Moyenne mobile pour l'entraînement
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
                    label=f"Moyenne mobile ({window_size})",
                )

        plt.axvline(x=0, color="black", linestyle="--", alpha=0.5, label="Début entraînement")
        plt.xlabel("Étape / Épisode")
        plt.ylabel("Récompense")
        plt.title("Évolution complète des récompenses")
        plt.legend()
        plt.grid(True, alpha=0.3)

        # Taux de victoire
        if self.win_rates:
            plt.subplot(2, 2, 2)
            episodes_eval = np.arange(
                self.eval_interval, len(self.win_rates) * self.eval_interval + 1, self.eval_interval
            )
            plt.plot(episodes_eval, self.win_rates, "o-", color="green", linewidth=2)
            plt.xlabel("Épisode")
            plt.ylabel("Taux de victoire")
            plt.title("Évolution du taux de victoire")
            plt.grid(True, alpha=0.3)

        # Comparaison des histogrammes
        plt.subplot(2, 2, 3)
        if self.pretraining_rewards:
            plt.hist(
                self.pretraining_rewards,
                bins=30,
                alpha=0.5,
                color="blue",
                label="Pré-entraînement",
                density=True,
            )
        if self.training_rewards:
            plt.hist(
                self.training_rewards,
                bins=30,
                alpha=0.5,
                color="red",
                label="Entraînement",
                density=True,
            )
        plt.xlabel("Récompense")
        plt.ylabel("Densité")
        plt.title("Distribution des récompenses")
        plt.legend()
        plt.grid(True, alpha=0.3)

        # Statistiques globales
        plt.subplot(2, 2, 4)
        stats_lines = ["Statistiques globales:\n"]

        if self.pretraining_rewards:
            stats_lines.extend(
                [
                    "Pré-entraînement:",
                    f"  Étapes: {len(self.pretraining_rewards)}",
                    f"  Récompense moy.: {np.mean(self.pretraining_rewards):.3f}",
                    f"  Récompense fin: {np.mean(self.pretraining_rewards[-100:]):.3f}",
                    "",
                ]
            )

        if self.training_rewards:
            recent_rewards = self.training_rewards[-100:]
            stats_lines.extend(
                [
                    "Entraînement principal:",
                    f"  Épisodes: {len(self.training_rewards)}",
                    f"  Récompense moy.: {np.mean(self.training_rewards):.2f}",
                    f"  Récompense finale: {np.mean(recent_rewards):.2f}",
                    "",
                ]
            )

        if self.win_rates:
            stats_lines.extend(
                [
                    "Performance:",
                    f"  Taux de victoire final: {self.win_rates[-1]:.1%}",
                    f"  Meilleur taux: {max(self.win_rates):.1%}",
                ]
            )

        plt.text(0.1, 0.5, "\n".join(stats_lines), fontsize=11, verticalalignment="center")
        plt.axis("off")

        plt.tight_layout()

        # Sauvegarder
        plot_path = os.path.join(self.model_dir, "full_training_results.png")
        plt.savefig(plot_path, dpi=300, bbox_inches="tight")
        print(f"📊 Graphiques complets sauvegardés: {plot_path}")

        plt.show()


def main():
    """Fonction principale avec pré-entraînement"""
    parser = argparse.ArgumentParser(
        description="Entraînement DQN avec pré-entraînement sur le point optimal"
    )

    # Arguments d'entraînement
    parser.add_argument(
        "--episodes", type=int, default=1000, help="Nombre d'épisodes d'entraînement principal"
    )
    parser.add_argument(
        "--pretraining_steps",
        type=int,
        default=10000,
        help="Nombre d'étapes de pré-entraînement sur le point optimal",
    )
    parser.add_argument(
        "--opponent",
        type=str,
        default="follow_ball",
        choices=["random", "follow_ball", "defensive", "aggressive", "predictive"],
        help="Type d'adversaire pour l'entraînement principal",
    )

    # Arguments du réseau
    parser.add_argument("--lr", type=float, default=0.001, help="Taux d'apprentissage")
    parser.add_argument("--tau", type=float, default=0.005, help="Coefficient pour les soft updates du target network")
    parser.add_argument("--gamma", type=float, default=0.99, help="Facteur de discount")
    parser.add_argument(
        "--epsilon", type=float, default=1.0, help="Epsilon initial pour l'exploration"
    )
    parser.add_argument(
        "--epsilon_decay", type=float, default=0.995, help="Facteur de décroissance d'epsilon"
    )
    parser.add_argument(
        "--epsilon_min", type=float, default=0.01, help="Epsilon minimum pour l'exploration"
    )
    parser.add_argument("--memory_size", type=int, default=20000, help="Taille du replay buffer")
    parser.add_argument(
        "--batch_size", type=int, default=32, help="Taille des batches d'entraînement"
    )

    # Arguments de contrôle
    parser.add_argument(
        "--skip_pretraining",
        action="store_true",
        help="Ignorer la phase de pré-entraînement",
    )
    parser.add_argument(
        "--pretraining_only",
        action="store_true",
        help="Faire seulement la phase de pré-entraînement",
    )
    parser.add_argument(
        "--model_dir", type=str, default="models", help="Répertoire de sauvegarde des modèles"
    )
    parser.add_argument(
        "--plot", action="store_true", help="Afficher les graphiques d'entraînement"
    )

    args = parser.parse_args()

    # Créer le trainer avec pré-entraînement
    trainer = DQNPretrainer(
        episodes=args.episodes,
        pretraining_steps=args.pretraining_steps,
        model_dir=args.model_dir,
    )

    # Configuration de l'agent
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
    print(f"   Pré-entraînement: {args.pretraining_steps} étapes")
    print(f"   Entraînement: {args.episodes} épisodes vs {args.opponent}")
    print(f"   Sauvegarde: {args.model_dir}")
    print()

    # Entraîner l'agent
    trainer.train_with_pretraining(
        opponent_type=args.opponent,
        agent_kwargs=agent_kwargs,
        skip_pretraining=args.skip_pretraining,
        pretraining_only=args.pretraining_only,
    )

    # Afficher les graphiques si demandé
    if args.plot:
        trainer.plot_full_training_results()

    print("\n🎉 Entraînement avec pré-entraînement terminé avec succès!")


if __name__ == "__main__":
    main()
