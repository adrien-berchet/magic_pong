"""
Moteur principal du jeu Magic Pong
"""

import time
from typing import Any

from magic_pong.ai.interface import AIPlayer, GameEnvironment
from magic_pong.core.entities import Action
from magic_pong.core.physics import PhysicsEngine
from magic_pong.utils.config import game_config


class GameEngine:
    """Moteur principal qui orchestre le jeu"""

    def __init__(self, headless: bool = False):
        self.headless = headless
        self.physics_engine = PhysicsEngine(game_config.FIELD_WIDTH, game_config.FIELD_HEIGHT)

        # Environnement pour l'IA
        self.ai_environment = GameEnvironment(self.physics_engine, headless)

        # État du jeu
        self.running = False
        self.paused = False
        self.last_update_time = 0.0

        # Joueurs (peuvent être humains ou IA)
        self.player1: AIPlayer | None = None
        self.player2: AIPlayer | None = None

        # Statistiques
        self.total_games = 0
        self.game_stats = {
            "player1_wins": 0,
            "player2_wins": 0,
            "total_steps": 0,
            "average_game_length": 0.0,
        }

    def set_players(self, player1: AIPlayer | None, player2: AIPlayer | None) -> None:
        """Définit les joueurs (humains ou IA)"""
        self.player1 = player1
        self.player2 = player2

        # Notifier les joueurs IA du début d'épisode
        if player1 and hasattr(player1, "on_episode_start"):
            player1.on_episode_start()
        if player2 and hasattr(player2, "on_episode_start"):
            player2.on_episode_start()

    def start_game(self) -> None:
        """Démarre une nouvelle partie"""
        self.running = True
        self.paused = False
        self.physics_engine.reset_game()
        self.last_update_time = time.time()

        # Reset de l'environnement IA
        self.ai_environment.reset()

    def stop_game(self) -> None:
        """Arrête la partie en cours"""
        self.running = False

        # Notifier les joueurs IA de la fin d'épisode
        if self.player1 and hasattr(self.player1, "on_episode_end"):
            self.player1.on_episode_end(0.0)
        if self.player2 and hasattr(self.player2, "on_episode_end"):
            self.player2.on_episode_end(0.0)

    def pause_game(self) -> None:
        """Met en pause / reprend la partie"""
        self.paused = not self.paused
        if not self.paused:
            self.last_update_time = time.time()

    def update(self, dt: float | None = None) -> dict[str, Any]:
        """
        Met à jour le jeu d'un frame

        Args:
            dt: Delta time en secondes. Si None, calculé automatiquement

        Returns:
            Dict contenant les événements et l'état du jeu
        """
        if not self.running or self.paused:
            return {"events": {}, "game_state": self.physics_engine.get_game_state()}

        # Calculer le delta time
        if dt is None:
            current_time = time.time()
            dt = current_time - self.last_update_time
            self.last_update_time = current_time

        # Limiter le delta time pour éviter les gros sauts
        dt = min(dt, 1.0 / 30.0)  # Max 30 FPS minimum

        # Obtenir les actions des joueurs
        action1 = self._get_player_action(self.player1, 1)
        action2 = self._get_player_action(self.player2, 2)

        # Mettre à jour la physique via l'environnement IA
        obs1, obs2, reward1, reward2, done, info = self.ai_environment.step(action1, action2)

        # Notifier les joueurs IA
        if self.player1 and hasattr(self.player1, "on_step"):
            # Créer une action par défaut si None
            safe_action1 = action1 if action1 is not None else Action(move_x=0.0, move_y=0.0)
            self.player1.on_step(obs1, safe_action1, reward1, done, info)
        if self.player2 and hasattr(self.player2, "on_step"):
            # Créer une action par défaut si None
            safe_action2 = action2 if action2 is not None else Action(move_x=0.0, move_y=0.0)
            self.player2.on_step(obs2, safe_action2, reward2, done, info)

        # Vérifier la fin de partie
        if done:
            self._handle_game_end(info)

        return {
            "events": info["events"],
            "game_state": info["game_state"],
            "observations": {"player1": obs1, "player2": obs2},
            "rewards": {"player1": reward1, "player2": reward2},
            "done": done,
            "info": info,
        }

    def _get_player_action(self, player: AIPlayer | None, player_id: int) -> Action | None:
        """Obtient l'action d'un joueur"""
        if player is None:
            return None

        if hasattr(player, "get_action"):
            # Joueur IA
            game_state = self.physics_engine.get_game_state()
            observation = self.ai_environment.observation_processor.process_game_state(
                game_state, player_id
            )
            action: Action = player.get_action(observation)
            return action
        elif hasattr(player, "get_human_action"):
            # Joueur humain (à implémenter avec l'interface graphique)
            human_action: Action | None = player.get_human_action()
            return human_action
        else:
            return None

    def _handle_game_end(self, info: dict[str, Any]) -> None:
        """Gère la fin d'une partie"""
        winner = info.get("winner", 0)

        # Mettre à jour les statistiques
        self.total_games += 1
        if winner == 1:
            self.game_stats["player1_wins"] += 1
        elif winner == 2:
            self.game_stats["player2_wins"] += 1

        self.game_stats["total_steps"] += info.get("step_count", 0)
        self.game_stats["average_game_length"] = self.game_stats["total_steps"] / self.total_games

        # Notifier les joueurs IA
        if self.player1 and hasattr(self.player1, "on_episode_end"):
            final_reward = 1.0 if winner == 1 else -1.0 if winner == 2 else 0.0
            self.player1.on_episode_end(final_reward)
        if self.player2 and hasattr(self.player2, "on_episode_end"):
            final_reward = 1.0 if winner == 2 else -1.0 if winner == 1 else 0.0
            self.player2.on_episode_end(final_reward)

        # Arrêter le jeu
        self.running = False

    def get_game_state(self) -> dict[str, Any]:
        """Retourne l'état complet du jeu"""
        return self.physics_engine.get_game_state()

    def get_stats(self) -> dict[str, Any]:
        """Retourne les statistiques du jeu"""
        stats: dict[str, Any] = self.game_stats.copy()

        # Ajouter les stats des joueurs IA
        if self.player1 and hasattr(self.player1, "get_stats"):
            stats["player1_ai_stats"] = self.player1.get_stats()
        if self.player2 and hasattr(self.player2, "get_stats"):
            stats["player2_ai_stats"] = self.player2.get_stats()

        return stats

    def reset_stats(self) -> None:
        """Remet les statistiques à zéro"""
        self.total_games = 0
        self.game_stats = {
            "player1_wins": 0,
            "player2_wins": 0,
            "total_steps": 0,
            "average_game_length": 0.0,
        }

    def set_speed_multiplier(self, multiplier: float) -> None:
        """Définit le multiplicateur de vitesse pour l'entraînement"""
        game_config.GAME_SPEED_MULTIPLIER = multiplier

    def is_running(self) -> bool:
        """Vérifie si le jeu est en cours"""
        return self.running

    def is_paused(self) -> bool:
        """Vérifie si le jeu est en pause"""
        return self.paused

    def is_game_over(self) -> bool:
        """Vérifie si la partie est terminée"""
        return self.physics_engine.is_game_over()

    def get_winner(self) -> int:
        """Retourne le gagnant de la partie"""
        return self.physics_engine.get_winner()


class TrainingManager:
    """Gestionnaire pour l'entraînement d'IA"""

    def __init__(self, headless: bool = True):
        self.game_engine = GameEngine(headless=headless)
        self.training_stats: dict[str, Any] = {
            "episodes": 0,
            "total_steps": 0,
            "player1_wins": 0,
            "player2_wins": 0,
            "average_episode_length": 0.0,
            "average_rewards": {"player1": 0.0, "player2": 0.0},
        }

    def train_episode(
        self, player1: AIPlayer, player2: AIPlayer, max_steps: int = 10000
    ) -> dict[str, Any]:
        """
        Entraîne une épisode complet

        Args:
            player1: Premier joueur (IA)
            player2: Deuxième joueur (IA)
            max_steps: Nombre maximum de steps par épisode

        Returns:
            Dict: Statistiques de l'épisode
        """
        self.game_engine.set_players(player1, player2)
        self.game_engine.start_game()

        episode_stats: dict[str, Any] = {
            "steps": 0,
            "winner": 0,
            "total_reward_p1": 0.0,
            "total_reward_p2": 0.0,
            "events": [],
        }

        while self.game_engine.is_running() and episode_stats["steps"] < max_steps:
            result = self.game_engine.update()

            episode_stats["steps"] += 1
            episode_stats["total_reward_p1"] += result["rewards"]["player1"]
            episode_stats["total_reward_p2"] += result["rewards"]["player2"]
            episode_stats["events"].extend(result["events"].get("goals", []))

            if result["done"]:
                episode_stats["winner"] = result["info"].get("winner", 0)
                break

        # Mettre à jour les statistiques d'entraînement
        self._update_training_stats(episode_stats)

        return episode_stats

    def _update_training_stats(self, episode_stats: dict[str, Any]) -> None:
        """Met à jour les statistiques d'entraînement"""
        self.training_stats["episodes"] += 1
        self.training_stats["total_steps"] += episode_stats["steps"]

        if episode_stats["winner"] == 1:
            self.training_stats["player1_wins"] += 1
        elif episode_stats["winner"] == 2:
            self.training_stats["player2_wins"] += 1

        # Moyenne mobile des récompenses
        alpha = 0.01  # Facteur de lissage
        self.training_stats["average_rewards"]["player1"] = (1 - alpha) * self.training_stats[
            "average_rewards"
        ]["player1"] + alpha * episode_stats["total_reward_p1"]
        self.training_stats["average_rewards"]["player2"] = (1 - alpha) * self.training_stats[
            "average_rewards"
        ]["player2"] + alpha * episode_stats["total_reward_p2"]

        self.training_stats["average_episode_length"] = (
            self.training_stats["total_steps"] / self.training_stats["episodes"]
        )

    def get_training_stats(self) -> dict[str, Any]:
        """Retourne les statistiques d'entraînement"""
        return self.training_stats.copy()

    def reset_training_stats(self) -> None:
        """Remet les statistiques d'entraînement à zéro"""
        self.training_stats = {
            "episodes": 0,
            "total_steps": 0,
            "player1_wins": 0,
            "player2_wins": 0,
            "average_episode_length": 0.0,
            "average_rewards": {"player1": 0.0, "player2": 0.0},
        }
