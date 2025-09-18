"""
Exemples d'IA simples pour Magic Pong
"""

import random
import math
from typing import Dict, Any

from magic_pong.ai.interface import AIPlayer
from magic_pong.core.entities import Action


class RandomAI(AIPlayer):
    """IA qui joue de manière complètement aléatoire"""

    def __init__(self, player_id: int, name: str = "RandomAI"):
        super().__init__(player_id, name)

    def get_action(self, observation: Dict[str, Any]) -> Action:
        """Retourne une action aléatoire"""
        return Action(
            move_x=random.uniform(-1.0, 1.0),
            move_y=random.uniform(-1.0, 1.0)
        )

    def on_step(self, observation: Dict[str, Any], action: Action,
                reward: float, done: bool, info: Dict[str, Any]) -> None:
        """L'IA aléatoire n'apprend pas"""
        self.current_episode_reward += reward


class FollowBallAI(AIPlayer):
    """IA simple qui suit la balle"""

    def __init__(self, player_id: int, name: str = "FollowBallAI", aggressiveness: float = 0.8):
        super().__init__(player_id, name)
        self.aggressiveness = aggressiveness  # Entre 0 et 1

    def get_action(self, observation: Dict[str, Any]) -> Action:
        """Suit la balle avec une certaine agressivité"""
        ball_pos = observation['ball_pos']
        player_pos = observation['player_pos']

        # Calculer la direction vers la balle
        dx = ball_pos[0] - player_pos[0]
        dy = ball_pos[1] - player_pos[1]

        # Normaliser et appliquer l'agressivité
        distance = math.sqrt(dx*dx + dy*dy)
        if distance > 0:
            move_x = (dx / distance) * self.aggressiveness
            move_y = (dy / distance) * self.aggressiveness
        else:
            move_x = 0.0
            move_y = 0.0

        return Action(move_x=move_x, move_y=move_y)

    def on_step(self, observation: Dict[str, Any], action: Action,
                reward: float, done: bool, info: Dict[str, Any]) -> None:
        """L'IA simple n'apprend pas"""
        self.current_episode_reward += reward


class DefensiveAI(AIPlayer):
    """IA défensive qui reste près de son but"""

    def __init__(self, player_id: int, name: str = "DefensiveAI"):
        super().__init__(player_id, name)

    def get_action(self, observation: Dict[str, Any]) -> Action:
        """Stratégie défensive"""
        ball_pos = observation['ball_pos']
        ball_vel = observation.get('ball_vel', [0, 0])
        player_pos = observation['player_pos']

        # Position défensive (près du but)
        if self.player_id == 1:  # Joueur gauche
            target_x = 0.1  # Près du bord gauche
        else:  # Joueur droite
            target_x = 0.9  # Près du bord droit

        # Prédire où la balle va aller
        predicted_y = ball_pos[1] + ball_vel[1] * 0.5  # Prédiction simple
        target_y = max(0.1, min(0.9, predicted_y))  # Clamp

        # Se diriger vers la position cible
        dx = target_x - player_pos[0]
        dy = target_y - player_pos[1]

        # Normaliser
        distance = math.sqrt(dx*dx + dy*dy)
        if distance > 0:
            move_x = dx / distance
            move_y = dy / distance
        else:
            move_x = 0.0
            move_y = 0.0

        return Action(move_x=move_x, move_y=move_y)

    def on_step(self, observation: Dict[str, Any], action: Action,
                reward: float, done: bool, info: Dict[str, Any]) -> None:
        """L'IA défensive n'apprend pas"""
        self.current_episode_reward += reward


class AggressiveAI(AIPlayer):
    """IA agressive qui cherche les bonus et attaque"""

    def __init__(self, player_id: int, name: str = "AggressiveAI"):
        super().__init__(player_id, name)
        self.target_bonus = None

    def get_action(self, observation: Dict[str, Any]) -> Action:
        """Stratégie agressive"""
        ball_pos = observation['ball_pos']
        player_pos = observation['player_pos']
        bonuses = observation.get('bonuses', [])

        # Chercher le bonus le plus proche
        closest_bonus = None
        closest_distance = float('inf')

        for bonus in bonuses:
            bonus_x, bonus_y, bonus_type = bonus
            distance = math.sqrt((bonus_x - player_pos[0])**2 + (bonus_y - player_pos[1])**2)
            if distance < closest_distance:
                closest_distance = distance
                closest_bonus = (bonus_x, bonus_y)

        # Décider de la cible
        if closest_bonus and closest_distance < 0.3:  # Bonus proche
            target_x, target_y = closest_bonus
        else:
            # Sinon, aller vers la balle
            target_x, target_y = ball_pos

        # Se diriger vers la cible
        dx = target_x - player_pos[0]
        dy = target_y - player_pos[1]

        # Normaliser avec agressivité maximale
        distance = math.sqrt(dx*dx + dy*dy)
        if distance > 0:
            move_x = dx / distance
            move_y = dy / distance
        else:
            move_x = 0.0
            move_y = 0.0

        return Action(move_x=move_x, move_y=move_y)

    def on_step(self, observation: Dict[str, Any], action: Action,
                reward: float, done: bool, info: Dict[str, Any]) -> None:
        """L'IA agressive n'apprend pas"""
        self.current_episode_reward += reward


class PredictiveAI(AIPlayer):
    """IA qui essaie de prédire la trajectoire de la balle"""

    def __init__(self, player_id: int, name: str = "PredictiveAI", prediction_time: float = 1.0):
        super().__init__(player_id, name)
        self.prediction_time = prediction_time

    def get_action(self, observation: Dict[str, Any]) -> Action:
        """Prédit où sera la balle et s'y positionne"""
        ball_pos = observation['ball_pos']
        ball_vel = observation.get('ball_vel', [0, 0])
        player_pos = observation['player_pos']

        # Prédire la position future de la balle
        predicted_x = ball_pos[0] + ball_vel[0] * self.prediction_time
        predicted_y = ball_pos[1] + ball_vel[1] * self.prediction_time

        # Gérer les rebonds sur les murs (approximation simple)
        if predicted_y < 0 or predicted_y > 1:
            predicted_y = ball_pos[1] - ball_vel[1] * self.prediction_time

        # Clamp dans les limites du terrain
        predicted_x = max(0, min(1, predicted_x))
        predicted_y = max(0, min(1, predicted_y))

        # Se diriger vers la position prédite
        dx = predicted_x - player_pos[0]
        dy = predicted_y - player_pos[1]

        # Normaliser
        distance = math.sqrt(dx*dx + dy*dy)
        if distance > 0:
            move_x = dx / distance
            move_y = dy / distance
        else:
            move_x = 0.0
            move_y = 0.0

        return Action(move_x=move_x, move_y=move_y)

    def on_step(self, observation: Dict[str, Any], action: Action,
                reward: float, done: bool, info: Dict[str, Any]) -> None:
        """L'IA prédictive n'apprend pas"""
        self.current_episode_reward += reward


class HumanPlayer:
    """Classe pour représenter un joueur humain (pour l'interface graphique)"""

    def __init__(self, player_id: int, name: str = "Human"):
        self.player_id = player_id
        self.name = name
        self.current_action = Action(0.0, 0.0)

    def set_action(self, move_x: float, move_y: float) -> None:
        """Définit l'action actuelle du joueur humain"""
        self.current_action = Action(move_x, move_y)

    def get_human_action(self) -> Action:
        """Retourne l'action actuelle"""
        return self.current_action

    def get_stats(self) -> Dict[str, float]:
        """Retourne des stats vides pour compatibilité"""
        return {'mean_reward': 0.0, 'episodes': 0}


# Factory pour créer facilement des IA
def create_ai(ai_type: str, player_id: int, **kwargs) -> AIPlayer:
    """
    Factory pour créer des IA

    Args:
        ai_type: Type d'IA ('random', 'follow_ball', 'defensive', 'aggressive', 'predictive')
        player_id: ID du joueur (1 ou 2)
        **kwargs: Arguments supplémentaires pour l'IA

    Returns:
        AIPlayer: Instance de l'IA demandée
    """
    ai_classes = {
        'random': RandomAI,
        'follow_ball': FollowBallAI,
        'defensive': DefensiveAI,
        'aggressive': AggressiveAI,
        'predictive': PredictiveAI
    }

    if ai_type not in ai_classes:
        raise ValueError(f"Type d'IA inconnu: {ai_type}. Types disponibles: {list(ai_classes.keys())}")

    ai_class = ai_classes[ai_type]
    return ai_class(player_id, **kwargs)
