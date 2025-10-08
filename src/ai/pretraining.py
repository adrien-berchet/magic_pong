"""
Module de pré-entraînement pour l'IA DQN sur la tâche de proximité au point optimal.

Ce module permet d'entraîner l'IA à s'approcher du point optimal d'interception
de la balle avant de passer à un entraînement plus complexe contre des adversaires.
"""

import random
from typing import Any

import numpy as np
from magic_pong.ai.interface import ObservationProcessor, RewardCalculator
from magic_pong.ai.models.dqn_ai import ACTION_MAPPING
from magic_pong.core.entities import Action, Paddle, Player
from magic_pong.utils.config import ai_config, game_config, game_config_tmp


class OptimalPointPretrainer:
    """Classe de pré-entraînement pour l'apprentissage du point optimal"""

    def __init__(
        self,
        field_width: float | None = None,
        field_height: float | None = None,
        paddle_width: float | None = None,
        paddle_height: float | None = None,
        ball_radius: float | None = None,
        y_only: bool = False,
    ):
        """
        Args:
            field_width: Largeur du terrain
            field_height: Hauteur du terrain
            paddle_width: Largeur de la raquette
            paddle_height: Hauteur de la raquette
            ball_radius: Rayon de la balle
            y_only: Si True, ne considère que la distance verticale pour la récompense
        """
        self.field_width = field_width if field_width is not None else game_config.FIELD_WIDTH
        self.field_height = field_height if field_height is not None else game_config.FIELD_HEIGHT
        self.paddle_width = paddle_width if paddle_width is not None else game_config.PADDLE_WIDTH
        self.paddle_height = (
            paddle_height if paddle_height is not None else game_config.PADDLE_HEIGHT
        )
        self.ball_radius = ball_radius if ball_radius is not None else game_config.BALL_RADIUS

        # Define observation processor
        self.observation_processor = ObservationProcessor(field_width, field_height)

        # Marges pour éviter les bords
        self.margin = game_config.PADDLE_MARGIN

        # Calculateur de récompenses pour utiliser les fonctions existantes
        self.reward_calculator = RewardCalculator(y_only=y_only)

        self.player_config: dict[int, dict[str, int | dict[str, float]]] = {
            1: {
                "ball_spawn_zone": {
                    "x_min": self.field_width * 0.5,  # Milieu du terrain vers la droite
                    "x_max": self.field_width - self.margin,
                    "y_min": self.margin,
                    "y_max": self.field_height - self.margin,
                },
                "paddle_zone": {
                    "x_min": self.margin,
                    "x_max": self.field_width * 0.5 - self.paddle_width - self.margin,
                    "y_min": self.margin,
                    "y_max": self.field_height - self.margin - self.paddle_height,
                },
                "direction": -1,  # Vers la gauche
            },
            2: {
                "ball_spawn_zone": {
                    "x_min": self.margin,  # Milieu du terrain vers la droite
                    "x_max": self.field_width * 0.5,
                    "y_min": self.margin,
                    "y_max": self.field_height - self.margin,
                },
                "paddle_zone": {
                    "x_min": self.field_width * 0.5 + self.paddle_width + self.margin,
                    "x_max": self.field_width - self.margin - self.paddle_width,
                    "y_min": self.margin,
                    "y_max": self.field_height - self.margin - self.paddle_height,
                },
                "direction": 1,  # Vers la droite
            },
        }

    def generate_random_ball_state(self, player_id: int = 1) -> dict[str, Any]:
        """
        Génère un état aléatoire de balle qui se dirige vers le côté du joueur IA.

        Args:
            player_id: ID du joueur IA (1 pour gauche, 2 pour droite)

        Returns:
            Dict contenant la position et vélocité de la balle, et la position de la raquette
        """
        ball_spawn_zone: dict[str, float] = self.player_config[player_id]["ball_spawn_zone"]  # type: ignore[assignment]
        paddle_zone: dict[str, float] = self.player_config[player_id]["paddle_zone"]  # type: ignore[assignment]

        # Déterminer la direction selon le côté du joueur
        ball_x = random.uniform(ball_spawn_zone["x_min"], ball_spawn_zone["x_max"])
        ball_y = random.uniform(ball_spawn_zone["y_min"], ball_spawn_zone["y_max"])

        angle_rad = np.random.uniform(np.pi / 4, -np.pi / 4)
        ball_vx = (
            self.player_config[player_id]["direction"] * game_config.BALL_SPEED * np.cos(angle_rad)
        )
        ball_vy = game_config.BALL_SPEED * np.sin(angle_rad)

        # Position aléatoire de la raquette dans sa zone
        paddle_x = np.random.uniform(paddle_zone["x_min"], paddle_zone["x_max"])
        paddle_y = np.random.uniform(paddle_zone["y_min"], paddle_zone["y_max"])

        return {
            "ball_position": (ball_x, ball_y),
            "ball_velocity": (ball_vx, ball_vy),
            "paddle_position": (paddle_x, paddle_y),
            "field_bounds": (0, self.field_width, 0, self.field_height),
        }

    def create_game_state_from_ball_state(
        self, ball_state: dict[str, Any], player_id: int = 1
    ) -> dict[str, Any]:
        """
        Crée un état de jeu complet à partir d'un état de balle simplifié.

        Args:
            ball_state: État de balle généré par generate_random_ball_state
            player_id: ID du joueur IA

        Returns:
            État de jeu complet compatible avec l'interface IA
        """
        # Position de l'adversaire (fixe, au centre de son côté)
        if player_id == 1:
            opponent_x = self.field_width - self.margin - self.paddle_width
        else:
            opponent_x = self.margin

        opponent_y = (self.field_height - self.paddle_height) / 2

        game_state = {
            "ball_position": ball_state["ball_position"],
            "ball_velocity": ball_state["ball_velocity"],
            f"player{player_id}_position": ball_state["paddle_position"],
            f"player{3-player_id}_position": (opponent_x, opponent_y),
            f"player{player_id}_paddle_size": self.paddle_height,
            f"player{3-player_id}_paddle_size": self.paddle_height,
            "active_bonuses": [],
            "rotating_paddles": [],
            "score": [np.random.randint(0, 10), np.random.randint(0, 10)],
            "time_elapsed": np.random.uniform(0, 300),  # Jusqu'à 5 minutes
            "field_bounds": ball_state["field_bounds"],
        }

        return game_state

    def _set_last_ball_distance(self, game_state: dict[str, Any], player_id: int) -> None:
        """
        Met à jour la distance de la balle au point optimal dans le calculateur de récompenses.

        Args:
            game_state: État de jeu complet
            player_id: ID du joueur
        """

        # Get positions and velocity from game state
        ball_pos = game_state.get("ball_position", (0, 0))
        ball_vel = game_state.get("ball_velocity", (0, 0))
        player_pos = game_state.get(f"player{player_id}_position", (0, 0))
        field_bounds = game_state.get("field_bounds", (0, 800, 0, 600))

        # Calculate paddle center
        paddle_center_x = player_pos[0]
        if player_id == 1:
            paddle_center_x += game_config.PADDLE_WIDTH
        paddle_center_y = player_pos[1] + game_config.PADDLE_HEIGHT / 2

        # Find optimal interception point on ball's trajectory
        optimal_point = self.reward_calculator._find_optimal_interception_point(
            ball_pos, ball_vel, (paddle_center_x, paddle_center_y), field_bounds, player_id
        )
        current_distance = np.linalg.norm(
            optimal_point - np.array((paddle_center_x, paddle_center_y))
        )

        # Mettre à jour dans le calculateur de récompenses
        self.reward_calculator.last_ball_distance[player_id] = current_distance

    def calculate_optimal_position_reward(
        self, game_state: dict[str, Any], player_id: int = 1, dt: float = 1.0 / 60.0
    ) -> tuple[float, dict[str, Any]]:
        """
        Calcule la récompense de proximité au point optimal avec un système adapté au pré-entraînement.

        Args:
            game_state: État de jeu complet
            player_id: ID du joueur
            dt: Pas de temps

        Returns:
            Tuple (récompense, informations détaillées)
        """
        # Activer temporairement les récompenses de proximité
        original_use_proximity = ai_config.USE_PROXIMITY_REWARD
        ai_config.USE_PROXIMITY_REWARD = True

        try:
            # Calculer la récompense de proximité
            proximity_reward = self.reward_calculator._calculate_proximity_reward(
                game_state, player_id
            )

            # Récupérer les informations sur le point optimal
            optimal_points = self.reward_calculator.get_optimal_points()

            info = {
                "proximity_reward": proximity_reward,
                "optimal_points": optimal_points,
            }

            return proximity_reward, info

        finally:
            # Restaurer la configuration originale
            ai_config.USE_PROXIMITY_REWARD = original_use_proximity

    def simulate_paddle_movement(
        self,
        current_paddle_pos: tuple[float, float],
        action: Action,
        dt: float = 1.0 / 60.0,
        paddle_speed: float = 500.0,
    ) -> tuple[float, float]:
        """
        Simule le mouvement de la raquette en fonction de l'action.

        Args:
            current_paddle_pos: Position actuelle de la raquette (x, y)
            action: Action choisie par le réseau de neurones
            dt: Pas de temps
            paddle_speed: Vitesse de la raquette

        Returns:
            Nouvelle position de la raquette (x, y)
        """
        with game_config_tmp(
            FIELD_WIDTH=self.field_width,
            FIELD_HEIGHT=self.field_height,
            PADDLE_WIDTH=self.paddle_width,
            PADDLE_HEIGHT=self.paddle_height,
            PADDLE_MARGIN=self.margin,
        ):
            paddle_tmp = Paddle(*current_paddle_pos, player_id=1)
            paddle_tmp.move(action.move_x, action.move_y, dt)
            return paddle_tmp.position.x, paddle_tmp.position.y

        # paddle_x, paddle_y = current_paddle_pos

        # # Appliquer le mouvement
        # new_x = paddle_x + action.move_x * paddle_speed * dt
        # new_y = paddle_y + action.move_y * paddle_speed * dt

        # # Contraindre la position dans les limites du terrain
        # new_x = max(self.margin, min(new_x, self.field_width - self.margin - self.paddle_width))
        # new_y = max(0, min(new_y, self.field_height - self.paddle_height))

        # return new_x, new_y

    def pretraining_step(
        self, agent: Player, player_id: int = 1, num_steps: int = 1000
    ) -> dict[str, Any]:
        """
        Effectue une étape de pré-entraînement sur la proximité au point optimal.

        Args:
            agent: Agent DQN à entraîner
            player_id: ID du joueur
            num_steps: Nombre d'étapes de pré-entraînement

        Returns:
            Statistiques de l'étape de pré-entraînement
        """
        total_reward = 0.0
        total_loss = 0.0
        loss_count = 0
        rewards_history = []

        agent.set_training_mode(True)
        dt = game_config.GAME_SPEED_MULTIPLIER / game_config.FPS

        for _ in range(num_steps):
            # Générer un état aléatoire de balle
            ball_state = self.generate_random_ball_state(player_id)
            game_state = self.create_game_state_from_ball_state(ball_state, player_id)
            initial_paddle_pos = game_state[f"player{player_id}_position"]

            # Initialise la récompense de proximité avant de modifier l'état du système
            self.reward_calculator._calculate_proximity_reward(game_state, player_id)

            # Convertir l'état en observation pour l'agent
            observation = self._game_state_to_observation(game_state, player_id)
            state = agent._observation_to_state(observation)

            # L'agent choisit une action
            action_index = agent.act(state, training=True)
            action = self._index_to_action(action_index)

            # Simuler le mouvement de la raquette
            new_paddle_pos = self.simulate_paddle_movement(initial_paddle_pos, action, dt=dt)

            # Mettre à jour l'état du jeu avec la nouvelle position
            game_state[f"player{player_id}_position"] = new_paddle_pos

            # Calculer la récompense de proximité
            proximity_reward, info = self.calculate_optimal_position_reward(
                game_state, player_id, dt=dt
            )

            # ############################################################### #
            # optimal_point = tuple(info['optimal_points'][player_id]['position'].tolist())
            # initial_distance = np.linalg.norm(
            #     np.array(optimal_point) - np.array((
            #         initial_paddle_pos[0] + game_config.PADDLE_WIDTH / 2,
            #         initial_paddle_pos[1] + game_config.PADDLE_HEIGHT / 2,
            #     ))
            # )
            # new_distance = np.linalg.norm(
            #     np.array(optimal_point) - np.array((
            #         new_paddle_pos[0] + game_config.PADDLE_WIDTH / 2,
            #         new_paddle_pos[1] + game_config.PADDLE_HEIGHT / 2,
            #     ))
            # )
            # print(
            #     f"Step {step+1}/{num_steps}:"
            #     f"\n\t- Optimal point: {optimal_point}"
            #     f"\n\t- Initial Paddle Position: {initial_paddle_pos}"
            #     f"\n\t- Initial distance: {initial_distance}"
            #     f"\n\t- Action: {action_index}"
            #     f"\n\t- New Paddle Position: {new_paddle_pos}"
            #     f"\n\t- New distance: {new_distance}"
            #     f"\n\t- Delta distance (negative means closer): {new_distance - initial_distance}"
            #     f"\n\t- Reward: {proximity_reward:.4f}"
            #     "\n"
            # )
            # ############################################################### #

            # Créer l'état suivant (même état mais avec nouvelle position de raquette)
            next_observation = self._game_state_to_observation(game_state, player_id)
            next_state = agent._observation_to_state(next_observation)

            # Stocker l'expérience dans la mémoire de replay
            agent.remember(state, action_index, proximity_reward, next_state, done=False)

            # Entraîner l'agent si assez d'expériences
            if len(agent.memory) > agent.batch_size:
                loss = agent.replay()
                if loss is not None:
                    total_loss += loss
                    loss_count += 1

            total_reward += proximity_reward
            rewards_history.append(proximity_reward)

        # Calcul des statistiques
        avg_reward = total_reward / num_steps if num_steps > 0 else 0.0
        avg_loss = total_loss / loss_count if loss_count > 0 else 0.0

        return {
            "total_reward": total_reward,
            "average_reward": avg_reward,
            "average_loss": avg_loss,
            "steps": num_steps,
            "epsilon": agent.epsilon,
            "training_step": agent.training_step,
            "rewards_history": rewards_history,
        }

    def _game_state_to_observation(
        self, game_state: dict[str, Any], player_id: int
    ) -> dict[str, Any]:
        """
        Convertit un état de jeu en observation pour l'agent.
        Utilise la même logique que ObservationProcessor.
        """
        return self.observation_processor.process_game_state(game_state, player_id)

    def _index_to_action(self, action_index: int) -> Action:
        """
        Convertit un index d'action en objet Action.
        Utilise la même grille 3x3 que l'agent DQN.
        """
        actions = ACTION_MAPPING

        if 0 <= action_index < len(actions):
            return actions[action_index]
        else:
            # Action par défaut si index invalide
            print(f"Invalid action index: {action_index}, returning no movement.")
            return Action(move_x=0.0, move_y=0.0)

    def run_pretraining_phase(
        self,
        agent: Player,
        total_steps: int = 10000,
        steps_per_batch: int = 1000,
        player_id: int = 1,
        verbose: bool = True,
    ) -> dict[str, Any]:
        """
        Exécute une phase complète de pré-entraînement.

        Args:
            agent: Agent DQN à pré-entraîner
            total_steps: Nombre total d'étapes de pré-entraînement
            steps_per_batch: Nombre d'étapes par batch
            player_id: ID du joueur
            verbose: Affichage des statistiques

        Returns:
            Statistiques complètes du pré-entraînement
        """
        if verbose:
            print("🎯 Début du pré-entraînement sur le point optimal")
            print(f"   Total d'étapes: {total_steps}")
            print(f"   Étapes par batch: {steps_per_batch}")

        all_rewards = []
        all_losses = []
        batch_stats = []

        remaining_steps = total_steps
        batch_num = 0

        while remaining_steps > 0:
            current_batch_steps = min(steps_per_batch, remaining_steps)
            batch_num += 1

            # Exécuter un batch de pré-entraînement
            batch_result = self.pretraining_step(agent, player_id, current_batch_steps)

            # Collecter les statistiques
            all_rewards.extend(batch_result["rewards_history"])
            if batch_result["average_loss"] > 0:
                all_losses.append(batch_result["average_loss"])

            batch_stats.append(batch_result)

            if verbose:
                print(
                    f"   Batch {batch_num}: Récompense moy. = {batch_result['average_reward']:.3f}, "
                    f"Loss moy. = {batch_result['average_loss']:.4f}, "
                    f"Epsilon = {batch_result['epsilon']:.3f}"
                )

            remaining_steps -= current_batch_steps

        # Calcul des statistiques finales
        final_stats = {
            "total_steps": total_steps,
            "batches": batch_num,
            "final_epsilon": agent.epsilon,
            "final_training_step": agent.training_step,
            "average_reward": np.mean(all_rewards) if all_rewards else 0.0,
            "reward_std": np.std(all_rewards) if all_rewards else 0.0,
            "average_loss": np.mean(all_losses) if all_losses else 0.0,
            "batch_stats": batch_stats,
            "all_rewards": all_rewards,
        }

        if verbose:
            print("✅ Pré-entraînement terminé!")
            print(
                f"   Récompense finale moyenne: {final_stats['average_reward']:.3f} ± {final_stats['reward_std']:.3f}"
            )
            print(f"   Loss finale moyenne: {final_stats['average_loss']:.4f}")
            print(f"   Epsilon final: {final_stats['final_epsilon']:.3f}")

        return final_stats


def create_pretrainer(**kwargs: dict[str, Any]) -> OptimalPointPretrainer:
    """
    Factory pour créer un pré-entraîneur avec la configuration par défaut du jeu.

    Args:
        **kwargs: Arguments supplémentaires pour le constructeur

    Returns:
        Instance de OptimalPointPretrainer
    """
    defaults = {
        "field_width": game_config.FIELD_WIDTH,
        "field_height": game_config.FIELD_HEIGHT,
        "paddle_width": game_config.PADDLE_WIDTH,
        "paddle_height": game_config.PADDLE_HEIGHT,
        "ball_radius": game_config.BALL_RADIUS,
    }

    # Fusionner avec les arguments fournis
    defaults.update(kwargs)

    return OptimalPointPretrainer(**defaults)
