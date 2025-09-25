"""
Deep Q-Network (DQN) AI implementation for Magic Pong using PyTorch
Version améliorée avec techniques de stabilisation
"""

import random
from collections import deque, namedtuple
from typing import Any

import numpy as np
import torch
import torch.nn as nn
import torch.nn.functional as F
import torch.optim as optim
from magic_pong.ai.interface import AIPlayer
from magic_pong.core.entities import Action

# Transition tuple for replay buffer
Transition = namedtuple("Transition", ("state", "action", "reward", "next_state", "done"))

device = torch.device("cuda" if torch.cuda.is_available() else "cpu")


class DQNNetwork(nn.Module):
    """Deep Q-Network avec améliorations pour la stabilité"""

    def __init__(self, input_size: int, hidden_size: int = 256, output_size: int = 9):
        """
        Args:
            input_size: Taille de l'état d'entrée
            hidden_size: Taille des couches cachées
            output_size: Nombre d'actions possibles
        """
        super().__init__()

        # Architecture plus profonde et stable
        self.fc1 = nn.Linear(input_size, hidden_size)
        self.fc2 = nn.Linear(hidden_size, hidden_size)
        self.fc3 = nn.Linear(hidden_size, hidden_size // 2)
        self.fc4 = nn.Linear(hidden_size // 2, output_size)

        # Normalisation par batch pour améliorer la stabilité
        self.bn1 = nn.BatchNorm1d(hidden_size)
        self.bn2 = nn.BatchNorm1d(hidden_size)
        self.bn3 = nn.BatchNorm1d(hidden_size // 2)

        # Dropout pour éviter le surapprentissage
        self.dropout1 = nn.Dropout(0.2)
        self.dropout2 = nn.Dropout(0.2)
        self.dropout3 = nn.Dropout(0.1)

        # Initialisation Xavier pour la stabilité
        self._initialize_weights()

    def _initialize_weights(self) -> None:
        """Initialisation Xavier des poids"""
        for m in self.modules():
            if isinstance(m, nn.Linear):
                nn.init.xavier_uniform_(m.weight)
                nn.init.constant_(m.bias, 0)

    def forward(self, x: torch.Tensor) -> torch.Tensor:
        """Forward pass avec normalisation et dropout"""
        x = x.to(device)

        # Première couche
        x = self.fc1(x)
        if x.size(0) > 1:  # BatchNorm nécessite plus d'un échantillon
            x = self.bn1(x)
        x = F.relu(x)
        x = self.dropout1(x)

        # Deuxième couche
        x = self.fc2(x)
        if x.size(0) > 1:
            x = self.bn2(x)
        x = F.relu(x)
        x = self.dropout2(x)

        # Troisième couche
        x = self.fc3(x)
        if x.size(0) > 1:
            x = self.bn3(x)
        x = F.relu(x)
        x = self.dropout3(x)

        # Couche de sortie
        x = self.fc4(x)
        return x


class PrioritizedReplayBuffer:
    """Replay buffer avec échantillonnage prioritaire basé sur l'erreur TD"""

    def __init__(
        self, capacity: int, alpha: float = 0.6, beta: float = 0.4, beta_increment: float = 0.001
    ):
        self.capacity = capacity
        self.alpha = alpha  # priorité
        self.beta = beta  # correction d'importance
        self.beta_increment = beta_increment

        self.buffer = deque(maxlen=capacity)
        self.priorities = deque(maxlen=capacity)
        self.epsilon = 1e-6  # pour éviter les priorités nulles

    def add(
        self,
        state: np.ndarray,
        action: int,
        reward: float,
        next_state: np.ndarray,
        done: bool,
        td_error: float | None = None,
    ) -> None:
        """Ajoute une transition avec priorité basée sur l'erreur TD"""
        if td_error is None:
            # Si pas d'erreur TD fournie, utiliser la priorité maximale
            priority = max(self.priorities) if self.priorities else 1.0
        else:
            priority = abs(td_error) + self.epsilon

        transition = Transition(state, action, reward, next_state, done)
        self.buffer.append(transition)
        self.priorities.append(priority**self.alpha)

    def sample(self, batch_size: int) -> tuple[list[Transition], np.ndarray, np.ndarray]:
        """Échantillonnage prioritaire"""
        if len(self.buffer) < batch_size:
            batch_size = len(self.buffer)

        # Calcul des probabilités
        priorities = np.array(self.priorities)
        probabilities = priorities / priorities.sum()

        # Échantillonnage
        indices = np.random.choice(len(self.buffer), batch_size, p=probabilities)
        samples = [self.buffer[i] for i in indices]

        # Poids d'importance
        weights = (len(self.buffer) * probabilities[indices]) ** (-self.beta)
        weights = weights / weights.max()  # normalisation

        # Increment beta
        self.beta = min(1.0, self.beta + self.beta_increment)

        return samples, weights, indices

    def update_priorities(self, indices: np.ndarray, td_errors: np.ndarray) -> None:
        """Met à jour les priorités avec les nouvelles erreurs TD"""
        for idx, td_error in zip(indices, td_errors, strict=False):
            priority = (abs(td_error) + self.epsilon) ** self.alpha
            self.priorities[idx] = priority

    def __len__(self) -> int:
        return len(self.buffer)


class ReplayBuffer:
    """Buffer de replay simple pour comparaison"""

    def __init__(self, capacity: int):
        self.buffer = deque(maxlen=capacity)

    def add(
        self,
        state: np.ndarray,
        action: int,
        reward: float,
        next_state: np.ndarray,
        done: bool,
        td_error: float | None = None,
    ) -> None:
        """Ajoute une transition"""
        transition = Transition(state, action, reward, next_state, done)
        self.buffer.append(transition)

    def sample(self, batch_size: int) -> list[Transition]:
        """Échantillonnage aléatoire"""
        return random.sample(self.buffer, min(batch_size, len(self.buffer)))

    def __len__(self) -> int:
        return len(self.buffer)


class DQNAgent(AIPlayer):
    """Agent DQN avec améliorations de stabilité"""

    def __init__(
        self,
        state_size: int = 28,
        action_size: int = 9,
        lr: float = 0.001,
        gamma: float = 0.99,
        epsilon: float = 1.0,
        epsilon_min: float = 0.01,
        epsilon_decay: float = 0.9995,
        batch_size: int = 64,
        memory_size: int = 10000,
        use_prioritized_replay: bool = False,
        tau: float = 0.005,
        player_id: int = 1,
        name: str = "DQN AI",
    ):
        """
        Args:
            state_size: Taille de l'espace d'état
            action_size: Nombre d'actions possibles
            lr: Taux d'apprentissage
            gamma: Facteur de discount
            epsilon: Probabilité d'exploration initiale
            epsilon_min: Probabilité d'exploration minimale
            epsilon_decay: Décroissance d'epsilon
            batch_size: Taille du batch pour l'entraînement
            memory_size: Taille du buffer de replay
            use_prioritized_replay: Utiliser le replay prioritaire
            tau: Coefficient pour les soft updates du target network
            player_id: ID du joueur
            name: Nom de l'agent
        """
        super().__init__(player_id, name)

        self.state_size = state_size
        self.action_size = action_size
        self.lr = lr
        self.gamma = gamma
        self.epsilon = epsilon
        self.epsilon_min = epsilon_min
        self.epsilon_decay = epsilon_decay
        self.batch_size = batch_size
        self.tau = tau

        # État précédent pour l'apprentissage
        self.last_state = None
        self.last_action = None

        # Réseaux de neurones (taille d'état étendue pour inclure bonus et autres infos)
        self.q_network = DQNNetwork(state_size, 256, action_size).to(device)
        self.target_network = DQNNetwork(state_size, 256, action_size).to(device)
        self.optimizer = optim.Adam(self.q_network.parameters(), lr=lr, weight_decay=1e-4)

        # Scheduler pour réduire le learning rate
        self.scheduler = optim.lr_scheduler.ReduceLROnPlateau(
            self.optimizer, mode="max", factor=0.8, patience=100
        )

        # Buffer de replay
        self.memory: ReplayBuffer | PrioritizedReplayBuffer
        if use_prioritized_replay:
            self.memory = PrioritizedReplayBuffer(memory_size)
        else:
            self.memory = ReplayBuffer(memory_size)

        self.use_prioritized_replay = use_prioritized_replay

        # Copier les poids initiaux
        self.update_target_network()

        # Statistiques d'entraînement
        self.training_step = 0
        self.loss_history = []
        self.reward_history = []

    def get_action(self, observation: dict[str, Any]) -> Action:
        """
        Interface requise : convertit l'observation en action

        Args:
            observation: Observation du jeu formatée

        Returns:
            Action: Action à effectuer
        """
        # Convertir l'observation en vecteur d'état
        state = self._observation_to_state(observation)

        # Obtenir l'action numérique
        action_idx = self.act(state, training=True)

        # Convertir en Action du jeu
        return ACTION_MAPPING[action_idx]

    def on_step(
        self,
        observation: dict[str, Any],
        action: Action,
        reward: float,
        done: bool,
        info: dict[str, Any],
    ) -> None:
        """
        Interface requise : appelée après chaque étape pour l'apprentissage

        Args:
            observation: Nouvelle observation
            action: Action effectuée
            reward: Récompense reçue
            done: Si l'épisode est terminé
            info: Informations additionnelles
        """
        # Convertir l'observation en vecteur d'état
        current_state = self._observation_to_state(observation)

        # Si on a un état précédent, stocker l'expérience
        if self.last_state is not None and self.last_action is not None:
            self.remember(self.last_state, self.last_action, reward, current_state, done)

            # Entraîner le réseau si on a assez d'expériences
            loss = self.replay()
            if loss is not None:
                self.update_learning_rate(reward)

        # Mettre à jour la récompense de l'épisode
        self.current_episode_reward += reward

        # Stocker l'état actuel pour le prochain step
        if not done:
            self.last_state = current_state
            # Convertir l'action en index numérique
            self.last_action = self._action_to_index(action)
        else:
            self.last_state = None
            self.last_action = None

    def _observation_to_state(self, observation: dict[str, Any]) -> np.ndarray:
        """
        Convertit une observation du jeu en vecteur d'état pour le réseau
        Version étendue incluant bonus, tailles de paddles, score, etc.

        Args:
            observation: Observation formatée du jeu

        Returns:
            Vecteur d'état normalisé de dimension variable selon les bonus actifs
        """
        # État de base (8 dimensions)
        base_state = [
            observation["ball_pos"][0],  # Position X de la balle
            observation["ball_pos"][1],  # Position Y de la balle
            observation["ball_vel"][0] if "ball_vel" in observation else 0.0,  # Vitesse X balle
            observation["ball_vel"][1] if "ball_vel" in observation else 0.0,  # Vitesse Y balle
            observation["player_pos"][0],  # Position X du joueur
            observation["player_pos"][1],  # Position Y du joueur
            observation["opponent_pos"][0],  # Position X de l'adversaire
            observation["opponent_pos"][1],  # Position Y de l'adversaire
        ]

        # Informations supplémentaires importantes (5 dimensions)
        extra_state = [
            self._get_paddle_size(
                observation.get("player_paddle_size", 1.0)
            ),  # Hauteur paddle joueur
            self._get_paddle_size(
                observation.get("opponent_paddle_size", 1.0)
            ),  # Hauteur paddle adversaire
            observation.get("score_diff", 0.0) / 10.0,  # Différence de score (normalisée)
            observation.get("time_elapsed", 0.0) / 300.0,  # Temps écoulé (normalisé sur 5 min)
            len(observation.get("bonuses", [])) / 5.0,  # Nombre de bonus actifs (normalisé)
        ]

        # Informations sur les bonus actifs (jusqu'à 3 bonus x 3 infos = 9 dimensions max)
        bonus_state = []
        bonuses = observation.get("bonuses", [])
        max_bonuses = 3  # Limiter le nombre de bonus considérés

        for i in range(max_bonuses):
            if i < len(bonuses):
                bonus = bonuses[i]
                if len(bonus) >= 3:  # [x, y, type]
                    bonus_state.extend(
                        [
                            bonus[0],  # Position X du bonus (déjà normalisée)
                            bonus[1],  # Position Y du bonus (déjà normalisée)
                            bonus[2] / 3.0,  # Type de bonus (normalisé : 1,2,3 -> 0.33,0.67,1.0)
                        ]
                    )
                else:
                    bonus_state.extend([0.0, 0.0, 0.0])  # Bonus vide
            else:
                bonus_state.extend([0.0, 0.0, 0.0])  # Pas de bonus

        # Informations sur les paddles rotatifs (jusqu'à 2 paddles x 3 infos = 6 dimensions max)
        rotating_paddle_state = []
        rotating_paddles = observation.get("rotating_paddles", [])
        max_rotating_paddles = 2

        for i in range(max_rotating_paddles):
            if i < len(rotating_paddles):
                rp = rotating_paddles[i]
                if len(rp) >= 3:  # [x, y, angle]
                    # Normaliser l'angle : convertir en radians si nécessaire puis normaliser
                    angle_rad = np.radians(rp[2]) if abs(rp[2]) > 2 * np.pi else rp[2]
                    normalized_angle = (angle_rad + np.pi) / (
                        2 * np.pi
                    )  # Normaliser [-π, π] -> [0, 1]
                    rotating_paddle_state.extend(
                        [
                            rp[0],  # Position X (déjà normalisée)
                            rp[1],  # Position Y (déjà normalisée)
                            normalized_angle,  # Angle normalisé ([-π, π] -> [0, 1])
                        ]
                    )
                else:
                    rotating_paddle_state.extend([0.0, 0.0, 0.0])
            else:
                rotating_paddle_state.extend([0.0, 0.0, 0.0])

        # Combiner tous les états
        # Total: 8 (base) + 5 (extra) + 9 (bonus) + 6 (rotating) = 28 dimensions
        full_state = base_state + extra_state + bonus_state + rotating_paddle_state

        return np.array(full_state, dtype=np.float32)

    def _action_to_index(self, action: Action) -> int:
        """
        Convertit une Action en index numérique

        Args:
            action: Action du jeu

        Returns:
            Index de l'action (0-8)
        """
        # Chercher l'action correspondante dans le mapping
        for idx, mapped_action in ACTION_MAPPING.items():
            if (
                abs(mapped_action.move_x - action.move_x) < 0.1
                and abs(mapped_action.move_y - action.move_y) < 0.1
            ):
                return idx

        # Action par défaut si non trouvée
        return 0

    def _get_paddle_size(self, paddle_size_data) -> float:
        """
        Extrait la taille du paddle depuis différents formats possibles

        Args:
            paddle_size_data: Peut être float, list, ou tuple

        Returns:
            Taille normalisée du paddle (0.0 à 1.0)
        """
        if isinstance(paddle_size_data, list | tuple) and len(paddle_size_data) >= 2:
            size = paddle_size_data[1]  # Hauteur (index 1)
        elif isinstance(paddle_size_data, int | float):
            size = float(paddle_size_data)
        else:
            size = 1.0  # Valeur par défaut

        # Normaliser la taille (assumer taille min=0.5, max=2.0)
        normalized_size = np.clip((size - 0.5) / (2.0 - 0.5), 0.0, 1.0)
        return normalized_size

    def update_target_network(self):
        """Copie les poids du réseau principal vers le réseau cible"""
        self.target_network.load_state_dict(self.q_network.state_dict())

    def set_training_mode(self, training: bool):
        """Active ou désactive le mode entraînement"""
        self.q_network.train(training)
        self.target_network.train(training)
        """Copie les poids du réseau principal vers le réseau cible"""
        self.target_network.load_state_dict(self.q_network.state_dict())

    def soft_update_target_network(self):
        """Soft update du target network"""
        for target_param, main_param in zip(
            self.target_network.parameters(), self.q_network.parameters(), strict=False
        ):
            target_param.data.copy_(
                self.tau * main_param.data + (1.0 - self.tau) * target_param.data
            )

    def remember(
        self, state: np.ndarray, action: int, reward: float, next_state: np.ndarray, done: bool
    ):
        """Stocke une expérience dans la mémoire de replay"""
        self.memory.add(state, action, reward, next_state, done)

    def act(self, state: np.ndarray, training: bool = True) -> int:
        """Choisit une action avec politique epsilon-greedy"""
        if training and random.random() < self.epsilon:
            return random.randrange(self.action_size)

        # Conversion vers tensor PyTorch
        state_tensor = torch.FloatTensor(state).unsqueeze(0).to(device)

        # Prédiction avec le réseau
        self.q_network.eval()
        with torch.no_grad():
            q_values = self.q_network(state_tensor)
        self.q_network.train()

        return q_values.cpu().data.numpy().argmax()

    def replay(self) -> float | None:
        """Entraînement du réseau avec experience replay"""
        if len(self.memory) < self.batch_size:
            return None

        # Échantillonnage
        if self.use_prioritized_replay:
            experiences, weights, indices = self.memory.sample(self.batch_size)
            weights = torch.FloatTensor(weights).to(device)
        else:
            experiences, weights, indices = self.memory.sample(self.batch_size)
            weights = torch.ones(len(experiences)).to(device)
            indices = None

        # Extraction des données (optimisé pour éviter les warnings PyTorch)
        states = torch.FloatTensor(np.array([e.state for e in experiences])).to(device)
        actions = torch.LongTensor([e.action for e in experiences]).to(device)
        rewards = torch.FloatTensor([e.reward for e in experiences]).to(device)
        next_states = torch.FloatTensor(np.array([e.next_state for e in experiences])).to(device)
        dones = torch.BoolTensor([e.done for e in experiences]).to(device)

        # Q-values actuelles
        current_q_values = self.q_network(states).gather(1, actions.unsqueeze(1))

        # Double DQN: action selection avec main network, évaluation avec target network
        with torch.no_grad():
            next_actions = self.q_network(next_states).argmax(1).unsqueeze(1)
            next_q_values = self.target_network(next_states).gather(1, next_actions)
            target_q_values = rewards.unsqueeze(1) + (
                self.gamma * next_q_values * (~dones).unsqueeze(1)
            )

        # Calcul de la loss avec pondération d'importance
        td_errors = target_q_values - current_q_values
        loss = (weights.unsqueeze(1) * td_errors.pow(2)).mean()

        # Optimisation avec gradient clipping
        self.optimizer.zero_grad()
        loss.backward()
        torch.nn.utils.clip_grad_norm_(self.q_network.parameters(), max_norm=1.0)
        self.optimizer.step()

        # Mise à jour des priorités si replay prioritaire
        if self.use_prioritized_replay and indices is not None:
            td_errors_np = td_errors.detach().cpu().numpy().flatten()
            self.memory.update_priorities(indices, td_errors_np)

        # Soft update du target network
        self.soft_update_target_network()

        # Décroissance d'epsilon avec planification adaptive
        if self.epsilon > self.epsilon_min:
            # Décroissance plus lente en début d'entraînement
            decay_factor = self.epsilon_decay
            self.epsilon = max(self.epsilon_min, self.epsilon * decay_factor)

        self.training_step += 1
        self.loss_history.append(loss.item())

        return loss.item()

    def save_model(self, filepath: str):
        """Sauvegarde le modèle"""
        torch.save(
            {
                "q_network_state_dict": self.q_network.state_dict(),
                "target_network_state_dict": self.target_network.state_dict(),
                "optimizer_state_dict": self.optimizer.state_dict(),
                "epsilon": self.epsilon,
                "training_step": self.training_step,
                "loss_history": self.loss_history,
                "reward_history": self.reward_history,
                "hyperparameters": {
                    "state_size": self.state_size,
                    "action_size": self.action_size,
                    "lr": self.lr,
                    "gamma": self.gamma,
                    "epsilon_min": self.epsilon_min,
                    "epsilon_decay": self.epsilon_decay,
                    "batch_size": self.batch_size,
                    "tau": self.tau,
                    "use_prioritized_replay": self.use_prioritized_replay,
                },
            },
            filepath,
        )

    def load_model(self, filepath: str):
        """Charge le modèle"""
        checkpoint = torch.load(filepath, map_location=device)

        self.q_network.load_state_dict(checkpoint["q_network_state_dict"])
        self.target_network.load_state_dict(checkpoint["target_network_state_dict"])
        self.optimizer.load_state_dict(checkpoint["optimizer_state_dict"])
        self.epsilon = checkpoint["epsilon"]
        self.training_step = checkpoint["training_step"]
        self.loss_history = checkpoint["loss_history"]
        self.reward_history = checkpoint["reward_history"]

        # Charger les hyperparamètres si disponibles
        if "hyperparameters" in checkpoint:
            hyperparams = checkpoint["hyperparameters"]
            self.state_size = hyperparams["state_size"]
            self.action_size = hyperparams["action_size"]
            self.lr = hyperparams["lr"]
            self.gamma = hyperparams["gamma"]
            self.epsilon_min = hyperparams["epsilon_min"]
            self.epsilon_decay = hyperparams["epsilon_decay"]
            self.batch_size = hyperparams["batch_size"]
            self.tau = hyperparams["tau"]
            self.use_prioritized_replay = hyperparams["use_prioritized_replay"]

    def update_learning_rate(self, reward: float):
        """Met à jour le learning rate basé sur la performance"""
        self.scheduler.step(reward)

    def get_exploration_rate(self) -> float:
        """Retourne le taux d'exploration actuel"""
        return self.epsilon

    def get_training_stats(self) -> dict:
        """Retourne les statistiques d'entraînement"""
        return {
            "training_step": self.training_step,
            "epsilon": self.epsilon,
            "avg_loss": np.mean(self.loss_history[-100:]) if self.loss_history else 0,
            "avg_reward": np.mean(self.reward_history[-100:]) if self.reward_history else 0,
            "current_lr": self.optimizer.param_groups[0]["lr"],
        }


# Actions mappées selon l'interface du jeu
ACTION_MAPPING = {
    0: Action(0.0, 0.0),  # Rester immobile
    1: Action(0.0, -1.0),  # Monter
    2: Action(0.0, 1.0),  # Descendre
    3: Action(-1.0, 0.0),  # Gauche
    4: Action(1.0, 0.0),  # Droite
    5: Action(-1.0, -1.0),  # Haut-gauche
    6: Action(1.0, -1.0),  # Haut-droite
    7: Action(-1.0, 1.0),  # Bas-gauche
    8: Action(1.0, 1.0),  # Bas-droite
}


def create_state_vector(
    ball, paddle, opponent_paddle, screen_width: int = 800, screen_height: int = 600
) -> np.ndarray:
    """
    Crée un vecteur d'état normalisé pour le réseau de neurones (version legacy)

    ATTENTION: Cette fonction est maintenant obsolète pour l'usage avec DQNAgent.
    DQNAgent utilise maintenant _observation_to_state() qui inclut les bonus et autres informations.
    Cette fonction est conservée pour compatibilité avec d'anciens scripts.

    Args:
        ball: Objet Ball du jeu
        paddle: Paddle du joueur IA
        opponent_paddle: Paddle de l'adversaire
        screen_width: Largeur de l'écran
        screen_height: Hauteur de l'écran

    Returns:
        Vecteur d'état normalisé de dimension 8 (version simple, sans bonus)
    """
    state = np.array(
        [
            ball.x / screen_width,  # Position X de la balle (normalisée)
            ball.y / screen_height,  # Position Y de la balle (normalisée)
            ball.velocity_x / 10.0,  # Vitesse X de la balle (normalisée)
            ball.velocity_y / 10.0,  # Vitesse Y de la balle (normalisée)
            paddle.y / screen_height,  # Position Y du paddle IA (normalisée)
            (paddle.y + paddle.height / 2) / screen_height,  # Centre du paddle IA
            opponent_paddle.y / screen_height,  # Position Y du paddle adversaire (normalisée)
            (opponent_paddle.y + opponent_paddle.height / 2)
            / screen_height,  # Centre du paddle adversaire
        ],
        dtype=np.float32,
    )

    return state
