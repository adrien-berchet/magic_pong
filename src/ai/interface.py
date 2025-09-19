"""
Interface IA agnostique pour Magic Pong
"""

from abc import ABC, abstractmethod
from typing import Any

import numpy as np
from magic_pong.core.entities import Action
from magic_pong.core.physics import PhysicsEngine
from magic_pong.utils.config import ai_config


class AIPlayer(ABC):
    """Interface de base pour tous les joueurs IA"""

    def __init__(self, player_id: int, name: str = "AI"):
        self.player_id = player_id
        self.name = name
        self.episode_rewards: list[float] = []
        self.current_episode_reward = 0.0

    @abstractmethod
    def get_action(self, observation: dict[str, Any]) -> Action:
        """
        Retourne l'action à effectuer basée sur l'observation

        Args:
            observation: État du jeu normalisé

        Returns:
            Action: Action à effectuer
        """
        pass

    @abstractmethod
    def on_step(
        self,
        observation: dict[str, Any],
        action: Action,
        reward: float,
        done: bool,
        info: dict[str, Any],
    ) -> None:
        """
        Appelé après chaque step pour l'apprentissage

        Args:
            observation: Nouvelle observation
            action: Action effectuée
            reward: Récompense reçue
            done: Si l'épisode est terminé
            info: Informations supplémentaires
        """
        pass

    def on_episode_start(self) -> None:
        """Appelé au début de chaque épisode"""
        self.current_episode_reward = 0.0

    def on_episode_end(self, final_reward: float) -> None:
        """Appelé à la fin de chaque épisode"""
        self.episode_rewards.append(self.current_episode_reward)

    def get_stats(self) -> dict[str, float]:
        """Retourne les statistiques de performance"""
        if not self.episode_rewards:
            return {"mean_reward": 0.0, "episodes": 0}

        return {
            "mean_reward": float(
                np.mean(self.episode_rewards[-100:])
            ),  # Moyenne sur 100 derniers épisodes
            "total_reward": sum(self.episode_rewards),
            "episodes": len(self.episode_rewards),
            "last_reward": self.episode_rewards[-1] if self.episode_rewards else 0.0,
        }


class ObservationProcessor:
    """Processeur d'observations pour normaliser les données"""

    def __init__(self, field_width: float, field_height: float):
        self.field_width = field_width
        self.field_height = field_height

    def process_game_state(self, game_state: dict[str, Any], player_id: int) -> dict[str, Any]:
        """
        Convertit l'état du jeu en observation normalisée pour l'IA

        Args:
            game_state: État brut du jeu
            player_id: ID du joueur (1 ou 2)

        Returns:
            Dict: Observation normalisée
        """
        observation = {}

        # Positions normalisées
        if ai_config.NORMALIZE_POSITIONS:
            ball_x = game_state["ball_position"][0] / self.field_width
            ball_y = game_state["ball_position"][1] / self.field_height

            player_pos = game_state[f"player{player_id}_position"]
            opponent_pos = game_state[f"player{3-player_id}_position"]

            player_x = player_pos[0] / self.field_width
            player_y = player_pos[1] / self.field_height
            opponent_x = opponent_pos[0] / self.field_width
            opponent_y = opponent_pos[1] / self.field_height
        else:
            ball_x, ball_y = game_state["ball_position"]
            player_x, player_y = game_state[f"player{player_id}_position"]
            opponent_x, opponent_y = game_state[f"player{3-player_id}_position"]

        observation["ball_pos"] = [ball_x, ball_y]
        observation["player_pos"] = [player_x, player_y]
        observation["opponent_pos"] = [opponent_x, opponent_y]

        # Vélocité de la balle
        if ai_config.INCLUDE_VELOCITY:
            if ai_config.NORMALIZE_POSITIONS:
                vel_x = (
                    game_state["ball_velocity"][0] / 500.0
                )  # Normaliser par vitesse max approximative
                vel_y = game_state["ball_velocity"][1] / 500.0
            else:
                vel_x, vel_y = game_state["ball_velocity"]
            observation["ball_vel"] = [vel_x, vel_y]

        # Tailles des raquettes
        observation["player_paddle_size"] = game_state[f"player{player_id}_paddle_size"]
        observation["opponent_paddle_size"] = game_state[f"player{3-player_id}_paddle_size"]

        # Bonus actifs
        bonuses = []
        for bonus_x, bonus_y, bonus_type in game_state["active_bonuses"]:
            if ai_config.NORMALIZE_POSITIONS:
                bonus_x /= self.field_width
                bonus_y /= self.field_height
            bonuses.append([bonus_x, bonus_y, self._encode_bonus_type(bonus_type)])
        observation["bonuses"] = bonuses

        # Raquettes tournantes
        rotating_paddles = []
        for rp_x, rp_y, rp_angle in game_state["rotating_paddles"]:
            if ai_config.NORMALIZE_POSITIONS:
                rp_x /= self.field_width
                rp_y /= self.field_height
            rotating_paddles.append([rp_x, rp_y, rp_angle])
        observation["rotating_paddles"] = rotating_paddles

        # Score différentiel
        score = game_state["score"]
        if player_id == 1:
            observation["score_diff"] = score[0] - score[1]
        else:
            observation["score_diff"] = score[1] - score[0]

        # Temps écoulé
        observation["time_elapsed"] = game_state["time_elapsed"]

        return observation

    def _encode_bonus_type(self, bonus_type: str) -> float:
        """Encode le type de bonus en valeur numérique"""
        encoding = {"enlarge_paddle": 1.0, "shrink_opponent": 2.0, "rotating_paddle": 3.0}
        return encoding.get(bonus_type, 0.0)


class RewardCalculator:
    """Calculateur de récompenses pour l'entraînement"""

    def __init__(self) -> None:
        self.last_score = [0, 0]
        self.last_ball_distance = 0.0

    def calculate_reward(
        self, game_state: dict[str, Any], events: dict[str, list], player_id: int
    ) -> float:
        """
        Calcule la récompense pour un joueur basée sur les événements

        Args:
            game_state: État actuel du jeu
            events: Événements survenus ce step
            player_id: ID du joueur

        Returns:
            float: Récompense calculée
        """
        reward = 0.0

        # Récompenses pour les buts
        for goal in events.get("goals", []):
            if goal["player"] == player_id:
                reward += ai_config.SCORE_REWARD
            else:
                reward += ai_config.LOSE_PENALTY

        # Récompenses pour les bonus collectés
        for bonus in events.get("bonus_collected", []):
            if bonus["player"] == player_id:
                reward += ai_config.BONUS_REWARD

        # Récompenses pour toucher la balle
        for hit in events.get("paddle_hits", []):
            if hit["player"] == player_id:
                reward += ai_config.WALL_HIT_REWARD

        # Récompense pour les raquettes tournantes
        for hit in events.get("rotating_paddle_hits", []):
            if hit["player"] == player_id:
                reward += ai_config.WALL_HIT_REWARD * 2  # Bonus pour utiliser la raquette tournante

        return reward

    def reset(self) -> None:
        """Remet le calculateur à zéro"""
        self.last_score = [0, 0]
        self.last_ball_distance = 0.0


class GameEnvironment:
    """Environnement de jeu compatible avec les frameworks d'IA"""

    def __init__(self, physics_engine: PhysicsEngine, headless: bool = False) -> None:
        self.physics_engine = physics_engine
        self.headless = headless

        self.observation_processor = ObservationProcessor(
            physics_engine.field_width, physics_engine.field_height
        )

        self.reward_calculators = {1: RewardCalculator(), 2: RewardCalculator()}

        self.step_count = 0
        self.max_steps = ai_config.MAX_EPISODE_STEPS

    def reset(self) -> tuple[dict[str, Any], dict[str, Any]]:
        """
        Remet l'environnement à zéro

        Returns:
            Tuple: (observation_player1, observation_player2)
        """
        self.physics_engine.reset_game()
        self.step_count = 0

        for calc in self.reward_calculators.values():
            calc.reset()

        game_state = self.physics_engine.get_game_state()

        obs1 = self.observation_processor.process_game_state(game_state, 1)
        obs2 = self.observation_processor.process_game_state(game_state, 2)

        return obs1, obs2

    def step(
        self, action1: Action | None, action2: Action | None
    ) -> tuple[dict[str, Any], dict[str, Any], float, float, bool, dict[str, Any]]:
        """
        Effectue un step dans l'environnement

        Args:
            action1: Action du joueur 1
            action2: Action du joueur 2

        Returns:
            Tuple: (obs1, obs2, reward1, reward2, done, info)
        """
        # Gérer les actions None avec des actions par défaut (aucun mouvement)
        if action1 is None:
            action1 = Action(move_x=0.0, move_y=0.0)
        if action2 is None:
            action2 = Action(move_x=0.0, move_y=0.0)

        # Mettre à jour la physique
        dt = 1.0 / 60.0  # 60 FPS
        if ai_config.HEADLESS_MODE:
            dt *= ai_config.FAST_MODE_MULTIPLIER

        events = self.physics_engine.update(dt, action1, action2)
        game_state = self.physics_engine.get_game_state()

        # Calculer les récompenses
        reward1 = self.reward_calculators[1].calculate_reward(game_state, events, 1)
        reward2 = self.reward_calculators[2].calculate_reward(game_state, events, 2)

        # Vérifier si l'épisode est terminé
        done = self.physics_engine.is_game_over() or self.step_count >= self.max_steps

        # Créer les observations
        obs1 = self.observation_processor.process_game_state(game_state, 1)
        obs2 = self.observation_processor.process_game_state(game_state, 2)

        # Informations supplémentaires
        info = {
            "events": events,
            "game_state": game_state,
            "winner": self.physics_engine.get_winner() if done else 0,
            "step_count": self.step_count,
        }

        self.step_count += 1

        return obs1, obs2, reward1, reward2, done, info

    def render(self) -> np.ndarray | None:
        """Rendu de l'environnement (à implémenter avec l'interface graphique)"""
        if self.headless:
            return None
        # TODO: Implémenter avec le renderer graphique
        return None

    def close(self) -> None:
        """Ferme l'environnement"""
        pass
