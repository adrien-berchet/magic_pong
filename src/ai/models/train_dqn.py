"""
Script d'entraînement pour l'IA DQN de Magic Pong
"""

import argparse
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
    """Gestionnaire d'entraînement pour l'agent DQN"""

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
            episodes: Nombre d'épisodes d'entraînement
            save_interval: Intervalle de sauvegarde du modèle
            eval_interval: Intervalle d'évaluation
            eval_episodes: Nombre d'épisodes d'évaluation
            model_dir: Répertoire de sauvegarde des modèles
        """
        self.episodes = episodes
        self.save_interval = save_interval
        self.eval_interval = eval_interval
        self.eval_episodes = eval_episodes
        self.model_dir = model_dir

        # Créer le répertoire des modèles
        os.makedirs(model_dir, exist_ok=True)

        # Métriques d'entraînement
        self.training_rewards: list[float] = []
        self.evaluation_scores: list[float] = []
        self.win_rates: list[float] = []

        # Pour la reprise d'entraînement
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
        """Sauvegarde l'état complet de l'entraînement"""
        import json

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

        # Sauvegarder en JSON
        with open(filepath, "w") as f:
            json.dump(training_state, f, indent=2)

        print(f"État d'entraînement sauvegardé dans {filepath}")

    def load_training_state(self, filepath: str) -> dict:
        """Charge l'état de l'entraînement depuis un fichier"""
        import json

        if not os.path.exists(filepath):
            print(f"Aucun état d'entraînement trouvé dans {filepath}")
            return {}

        with open(filepath) as f:
            training_state = json.load(f)

        # Restaurer l'état
        self.start_episode = training_state.get("episode", 0) + 1  # Reprendre à l'épisode suivant
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

        print(f"État d'entraînement chargé depuis {filepath}")
        print(f"Reprise à l'épisode {self.start_episode}")
        print(f"Meilleure récompense précédente: {self.best_avg_reward:.2f}")

        return training_state

    def find_latest_checkpoint(self, opponent_type: str) -> tuple[str, str] | None:
        """Trouve le dernier checkpoint disponible"""
        # Chercher les fichiers de checkpoint
        training_state_pattern = f"training_state_vs_{opponent_type}.json"

        checkpoint_files = []
        for file in os.listdir(self.model_dir):
            if file.startswith("checkpoint_ep") and file.endswith(f"_vs_{opponent_type}.pth"):
                # Extraire le numéro d'épisode
                try:
                    episode_num = int(file.split("_ep")[1].split("_vs_")[0])
                    checkpoint_files.append((episode_num, file))
                except (ValueError, IndexError):
                    continue

        if not checkpoint_files:
            return None

        # Trouver le checkpoint le plus récent
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
        Entraîne l'agent DQN contre une IA simple

        Args:
            opponent_type: Type d'IA adversaire ('random', 'follow_ball', etc.)
            agent_kwargs: Arguments pour la création de l'agent DQN
            resume_training: Si True, tente de reprendre un entraînement existant
            checkpoint_path: Chemin vers un checkpoint spécifique à charger

        Returns:
            DQNAgent: Agent entraîné
        """
        if agent_kwargs is None:
            agent_kwargs = {}

        print(f"Début de l'entraînement contre {opponent_type}")

        # Configuration pour l'entraînement rapide
        ai_config.HEADLESS_MODE = True
        ai_config.FAST_MODE_MULTIPLIER = 10.0

        # Créer l'agent DQN
        dqn_agent = DQNAgent(player_id=1, name="DQN_Trainee", **agent_kwargs)

        # Tentative de reprise d'entraînement
        if resume_training or checkpoint_path:
            model_path = checkpoint_path
            state_path = None

            if not model_path:
                # Chercher automatiquement le dernier checkpoint
                checkpoint_info = self.find_latest_checkpoint(opponent_type)
                if checkpoint_info:
                    model_path, state_path = checkpoint_info

            if model_path and os.path.exists(model_path):
                print(f"Chargement du modèle depuis {model_path}")
                try:
                    dqn_agent.load_model(model_path)

                    # Charger l'état d'entraînement si disponible
                    if state_path and os.path.exists(state_path):
                        self.load_training_state(state_path)
                    else:
                        # Chercher le fichier d'état par défaut
                        default_state_path = os.path.join(
                            self.model_dir, f"training_state_vs_{opponent_type}.json"
                        )
                        if os.path.exists(default_state_path):
                            self.load_training_state(default_state_path)

                    print("Reprise d'entraînement réussie!")

                except Exception as e:
                    print(f"Erreur lors du chargement: {e}")
                    print("Démarrage d'un nouvel entraînement...")
                    self.start_episode = 0
            else:
                if resume_training:
                    print("Aucun checkpoint trouvé, démarrage d'un nouvel entraînement...")
                self.start_episode = 0

        # Ajuster le nombre total d'épisodes
        episodes_remaining = self.episodes - self.start_episode
        if episodes_remaining <= 0:
            print("Entraînement déjà terminé!")
            return dqn_agent

        print(f"Épisodes restants: {episodes_remaining} (total: {self.episodes})")
        print(f"Configuration: {agent_kwargs}")

        # Créer l'adversaire
        opponent = create_ai(opponent_type, player_id=2, name=f"Opponent_{opponent_type}")

        # Créer le gestionnaire d'entraînement
        training_manager = TrainingManager(headless=True)

        # Variables pour le suivi
        episode_rewards = self.training_history.get("episode_rewards", [])
        recent_rewards = episode_rewards[-100:] if episode_rewards else []
        best_avg_reward = self.best_avg_reward

        start_time = time.time()

        for episode in range(self.start_episode, self.episodes):
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
                    model_path = os.path.join(self.model_dir, f"best_model_vs_{opponent_type}.pth")
                    dqn_agent.save_model(model_path)
                    self.training_history["best_model_path"] = model_path
                    print(f"  Nouveau meilleur modèle sauvegardé! Récompense: {avg_reward:.2f}")

            # Sauvegarde périodique
            if (episode + 1) % self.save_interval == 0:
                model_path = os.path.join(
                    self.model_dir, f"checkpoint_ep{episode+1}_vs_{opponent_type}.pth"
                )
                dqn_agent.save_model(model_path)

                # Sauvegarder l'état d'entraînement
                state_path = os.path.join(self.model_dir, f"training_state_vs_{opponent_type}.json")
                self.training_history["episode_rewards"] = episode_rewards
                self.training_history["training_params"] = agent_kwargs
                self.save_training_state(state_path, dqn_agent, episode, opponent_type)

            # Évaluation périodique
            if (episode + 1) % self.eval_interval == 0:
                win_rate = self.evaluate_agent(dqn_agent, opponent_type)
                self.win_rates.append(win_rate)
                self.training_history["win_rates"].append(win_rate)
                self.training_history["eval_episodes"].append(episode + 1)
                print(f"  Taux de victoire: {win_rate:.1%}")

        # Sauvegarde finale
        final_model_path = os.path.join(self.model_dir, f"final_model_vs_{opponent_type}.pth")
        dqn_agent.save_model(final_model_path)

        # Stocker les métriques
        self.training_rewards = episode_rewards

        print("\nEntraînement terminé!")
        print(f"Temps total: {time.time() - start_time:.1f}s")
        print(f"Récompense moyenne finale: {np.mean(recent_rewards):.2f}")

        return dqn_agent

    def evaluate_agent(self, agent: DQNAgent, opponent_type: str) -> float:
        """
        Évalue l'agent sur plusieurs parties

        Args:
            agent: Agent à évaluer
            opponent_type: Type d'adversaire

        Returns:
            float: Taux de victoire
        """
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

    def plot_training_metrics(self, save_path: str | None = None) -> None:
        """Affiche les métriques d'entraînement"""
        if not self.training_rewards:
            print("Aucune donnée d'entraînement à afficher")
            return

        # Calculer la moyenne mobile
        window_size = 50
        if len(self.training_rewards) >= window_size:
            moving_avg = np.convolve(
                self.training_rewards, np.ones(window_size) / window_size, mode="valid"
            ).tolist()
        else:
            moving_avg = self.training_rewards

        plt.figure(figsize=(12, 8))

        # Graphique des récompenses
        plt.subplot(2, 2, 1)
        plt.plot(self.training_rewards, alpha=0.3, color="blue", label="Récompenses")
        if len(moving_avg) > 0:
            plt.plot(
                range(window_size - 1, len(self.training_rewards)),
                moving_avg,
                color="red",
                label=f"Moyenne mobile ({window_size})",
            )
        plt.xlabel("Épisode")
        plt.ylabel("Récompense")
        plt.title("Évolution des récompenses")
        plt.legend()
        plt.grid(True)

        # Graphique des taux de victoire
        if self.win_rates:
            plt.subplot(2, 2, 2)
            episodes_eval = np.arange(
                self.eval_interval, len(self.win_rates) * self.eval_interval + 1, self.eval_interval
            )
            plt.plot(episodes_eval, self.win_rates, "o-", color="green")
            plt.xlabel("Épisode")
            plt.ylabel("Taux de victoire")
            plt.title("Évolution du taux de victoire")
            plt.grid(True)

        # Histogramme des récompenses
        plt.subplot(2, 2, 3)
        plt.hist(self.training_rewards, bins=50, alpha=0.7, color="purple")
        plt.xlabel("Récompense")
        plt.ylabel("Fréquence")
        plt.title("Distribution des récompenses")
        plt.grid(True)

        # Statistiques récentes
        plt.subplot(2, 2, 4)
        recent_episodes = min(100, len(self.training_rewards))
        recent_rewards = self.training_rewards[-recent_episodes:]

        stats_text = (
            f"""Statistiques (derniers {recent_episodes} épisodes):

Récompense moyenne: {np.mean(recent_rewards):.2f}
Récompense médiane: {np.median(recent_rewards):.2f}
Écart-type: {np.std(recent_rewards):.2f}
Minimum: {np.min(recent_rewards):.2f}
Maximum: {np.max(recent_rewards):.2f}

Taux de victoire final: {self.win_rates[-1]:.1%}"""
            if self.win_rates
            else ""
        )

        plt.text(0.1, 0.5, stats_text, fontsize=12, verticalalignment="center")
        plt.axis("off")
        plt.title("Statistiques d'entraînement")

        plt.tight_layout()

        if save_path:
            plt.savefig(save_path, dpi=300, bbox_inches="tight")
            print(f"Graphiques sauvegardés dans {save_path}")

        plt.show()


def main() -> None:
    """Fonction principale d'entraînement"""
    parser = argparse.ArgumentParser(description="Entraînement de l'IA DQN pour Magic Pong")
    parser.add_argument(
        "--episodes", type=int, default=1000, help="Nombre d'épisodes d'entraînement"
    )
    parser.add_argument(
        "--opponent",
        type=str,
        default="follow_ball",
        choices=["random", "follow_ball", "defensive", "aggressive", "predictive"],
        help="Type d'adversaire",
    )
    parser.add_argument("--lr", type=float, default=0.001, help="Taux d'apprentissage")
    parser.add_argument("--gamma", type=float, default=0.99, help="Facteur de discount")
    parser.add_argument(
        "--epsilon", type=float, default=1.0, help="Epsilon initial pour l'exploration"
    )
    parser.add_argument(
        "--epsilon_decay", type=float, default=0.995, help="Facteur de décroissance d'epsilon"
    )
    parser.add_argument("--memory_size", type=int, default=10000, help="Taille du replay buffer")
    parser.add_argument(
        "--batch_size", type=int, default=32, help="Taille des batches d'entraînement"
    )
    parser.add_argument(
        "--use_prioritized_replay",
        action="store_true",
        help="Utiliser le prioritized experience replay",
    )
    parser.add_argument(
        "--tau", type=float, default=0.005, help="Coefficient de soft update du target network"
    )
    parser.add_argument(
        "--model_dir", type=str, default="models", help="Répertoire de sauvegarde des modèles"
    )
    parser.add_argument(
        "--plot", action="store_true", help="Afficher les graphiques d'entraînement"
    )
    parser.add_argument(
        "--resume",
        action="store_true",
        help="Reprendre l'entraînement depuis le dernier checkpoint",
    )
    parser.add_argument(
        "--checkpoint",
        type=str,
        default=None,
        help="Chemin vers un checkpoint spécifique à charger",
    )
    parser.add_argument(
        "--list_checkpoints",
        action="store_true",
        help="Lister les checkpoints disponibles et quitter",
    )

    args = parser.parse_args()

    # Créer le trainer
    trainer = DQNTrainer(episodes=args.episodes, model_dir=args.model_dir)

    # Lister les checkpoints si demandé
    if args.list_checkpoints:
        print(f"Checkpoints disponibles dans {args.model_dir}:")
        if not os.path.exists(args.model_dir):
            print("  Aucun répertoire de modèles trouvé.")
            return

        checkpoint_files = []
        for file in os.listdir(args.model_dir):
            if file.startswith("checkpoint_ep") and file.endswith(".pth"):
                checkpoint_files.append(file)

        if not checkpoint_files:
            print("  Aucun checkpoint trouvé.")
        else:
            checkpoint_files.sort()
            for file in checkpoint_files:
                filepath = os.path.join(args.model_dir, file)
                size_mb = os.path.getsize(filepath) / (1024 * 1024)
                print(f"  {file} ({size_mb:.1f} MB)")

        # Chercher le dernier checkpoint pour l'adversaire spécifié
        latest = trainer.find_latest_checkpoint(args.opponent)
        if latest:
            model_path, state_path = latest
            print(f"\nDernier checkpoint pour {args.opponent}: {os.path.basename(model_path)}")
        return

    # Configuration de l'agent
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

    # Afficher les graphiques si demandé
    if args.plot:
        plot_path = os.path.join(args.model_dir, f"training_metrics_vs_{args.opponent}.png")
        trainer.plot_training_metrics(save_path=plot_path)

    print("Entraînement terminé avec succès!")


if __name__ == "__main__":
    main()
