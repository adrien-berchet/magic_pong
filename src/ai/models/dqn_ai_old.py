# ruff: noqa
# type: ignore
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

    def _initialize_weights(self):
        """Initialisation des poids pour une meilleure stabilité"""
        for module in self.modules():
            if isinstance(module, nn.Linear):
                nn.init.xavier_uniform_(module.weight)
                if module.bias is not None:
                    nn.init.constant_(module.bias, 0)

    def forward(self, x: torch.Tensor) -> torch.Tensor:
        """Forward pass du réseau"""
        # Gérer le cas d'un seul échantillon pour BatchNorm
        if x.dim() == 1:
            x = x.unsqueeze(0)

        x = F.relu(self.fc1(x))
        if x.size(0) > 1:  # BatchNorm seulement si batch_size > 1
            x = self.bn1(x)
        x = self.dropout1(x)

        x = F.relu(self.fc2(x))
        if x.size(0) > 1:
            x = self.bn2(x)
        x = self.dropout2(x)

        x = F.relu(self.fc3(x))
        if x.size(0) > 1:
            x = self.bn3(x)
        x = self.dropout3(x)

        x = self.fc4(x)
        return x


class PrioritizedReplayBuffer:
    """Buffer de mémoire avec priorité pour améliorer l'apprentissage"""

    def __init__(self, capacity: int, alpha: float = 0.6):
        self.capacity = capacity
        self.alpha = alpha
        self.buffer = []
        self.priorities = deque(maxlen=capacity)
        self.position = 0

    def push(
        self,
        state: np.ndarray,
        action: int,
        reward: float,
        next_state: np.ndarray,
        done: bool,
        td_error: float = None,
    ) -> None:
        """Ajoute une transition avec priorité"""
        # Priorité basée sur l'erreur TD
        priority = (abs(td_error) + 1e-6) ** self.alpha if td_error is not None else 1.0

        transition = Transition(state, action, reward, next_state, done)

        if len(self.buffer) < self.capacity:
            self.buffer.append(transition)
            self.priorities.append(priority)
        else:
            self.buffer[self.position] = transition
            self.priorities[self.position] = priority

        self.position = (self.position + 1) % self.capacity

    def sample(self, batch_size: int, beta: float = 0.4):
        """Échantillonnage avec priorité"""
        if len(self.buffer) < batch_size:
            return None

        # Calcul des probabilités
        priorities = np.array(list(self.priorities)[: len(self.buffer)])
        probabilities = priorities / priorities.sum()

        # Échantillonnage
        indices = np.random.choice(len(self.buffer), batch_size, p=probabilities)
        transitions = [self.buffer[idx] for idx in indices]

        # Calcul des poids d'importance
        weights = (len(self.buffer) * probabilities[indices]) ** (-beta)
        weights = weights / weights.max()

        return transitions, indices, weights

    def update_priorities(self, indices, td_errors):
        """Met à jour les priorités basées sur les erreurs TD"""
        for idx, td_error in zip(indices, td_errors):
            priority = (abs(td_error) + 1e-6) ** self.alpha
            self.priorities[idx] = priority

    def __len__(self):
        return len(self.buffer)


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


class DQNNetwork(nn.Module):
    """Deep Q-Network pour l'approximation de fonction Q"""

    def __init__(self, input_size: int, hidden_size: int = 256, output_size: int = 9):
        """
        Args:
            input_size: Taille de l'état d'entrée
            hidden_size: Taille des couches cachées
            output_size: Nombre d'actions possibles (9 pour un contrôle directionnel 3x3)
        """
        super().__init__()

        self.fc1 = nn.Linear(input_size, hidden_size)
        self.fc2 = nn.Linear(hidden_size, hidden_size)
        self.fc3 = nn.Linear(hidden_size, hidden_size)
        self.fc4 = nn.Linear(hidden_size, output_size)

        # Normalisation par batch pour améliorer la stabilité
        self.bn1 = nn.BatchNorm1d(hidden_size)
        self.bn2 = nn.BatchNorm1d(hidden_size)
        self.bn3 = nn.BatchNorm1d(hidden_size)

        # Dropout pour éviter le surapprentissage
        self.dropout = nn.Dropout(0.1)

    def forward(self, x: torch.Tensor) -> torch.Tensor:
        """Forward pass du réseau"""
        x = F.relu(self.fc1(x))
        if x.size(0) > 1:  # BatchNorm seulement si batch_size > 1
            x = self.bn1(x)
        x = self.dropout(x)

        x = F.relu(self.fc2(x))
        if x.size(0) > 1:
            x = self.bn2(x)
        x = self.dropout(x)

        x = F.relu(self.fc3(x))
        if x.size(0) > 1:
            x = self.bn3(x)

        x = self.fc4(x)
        return x


class ReplayBuffer:
    """Buffer de mémoire pour l'apprentissage par batch"""

    def __init__(self, capacity: int):
        self.buffer = deque(maxlen=capacity)

    def push(
        self, state: np.ndarray, action: int, reward: float, next_state: np.ndarray, done: bool
    ) -> None:
        """Ajoute une transition au buffer"""
        self.buffer.append(Transition(state, action, reward, next_state, done))

    def sample(self, batch_size: int) -> list[Transition]:
        """Échantillonne un batch de transitions"""
        return random.sample(self.buffer, batch_size)

    def __len__(self) -> int:
        return len(self.buffer)


class DQNAgent(AIPlayer):
    """Agent DQN qui utilise l'apprentissage par renforcement profond"""

    def __init__(
        self,
        player_id: int,
        name: str = "DQN_AI",
        state_size: int = 20,
        hidden_size: int = 256,
        lr: float = 0.001,
        gamma: float = 0.99,
        epsilon: float = 1.0,
        epsilon_min: float = 0.01,
        epsilon_decay: float = 0.995,
        memory_size: int = 10000,
        batch_size: int = 32,
        target_update: int = 1000,
    ):
        """
        Args:
            player_id: ID du joueur (1 ou 2)
            name: Nom de l'agent
            state_size: Taille de l'état d'observation
            hidden_size: Taille des couches cachées
            lr: Taux d'apprentissage
            gamma: Facteur de discount
            epsilon: Probabilité d'exploration initiale
            epsilon_min: Probabilité d'exploration minimale
            epsilon_decay: Facteur de décroissance d'epsilon
            memory_size: Taille du replay buffer
            batch_size: Taille des batches d'entraînement
            target_update: Fréquence de mise à jour du réseau target
        """
        super().__init__(player_id, name)

        # Hyperparamètres
        self.state_size = state_size
        self.action_size = 9  # 3x3 grille d'actions directionnelles
        self.lr = lr
        self.gamma = gamma
        self.epsilon = epsilon
        self.epsilon_min = epsilon_min
        self.epsilon_decay = epsilon_decay
        self.batch_size = batch_size
        self.target_update = target_update

        # Dispositif (CPU/GPU)
        self.device = torch.device("cuda" if torch.cuda.is_available() else "cpu")
        print(f"DQN Agent utilise: {self.device}")

        # Réseaux de neurones
        self.q_network = DQNNetwork(state_size, hidden_size, self.action_size).to(self.device)
        self.target_network = DQNNetwork(state_size, hidden_size, self.action_size).to(self.device)
        self.optimizer = optim.Adam(self.q_network.parameters(), lr=lr)

        # Initialise le réseau target avec les poids du réseau principal
        self.target_network.load_state_dict(self.q_network.state_dict())

        # Replay buffer
        self.memory = ReplayBuffer(memory_size)

        # Compteurs
        self.steps = 0
        self.training_step = 0

        # Actions possibles (grille 3x3 normalisée)
        self.actions = [
            (-1, -1),
            (-1, 0),
            (-1, 1),  # Haut-gauche, haut, haut-droite
            (0, -1),
            (0, 0),
            (0, 1),  # Gauche, statique, droite
            (1, -1),
            (1, 0),
            (1, 1),  # Bas-gauche, bas, bas-droite
        ]

    def _preprocess_observation(self, observation: dict[str, Any]) -> np.ndarray:
        """Convertit l'observation en vecteur d'état pour le réseau"""
        state = []

        # Position de la balle
        state.extend(observation["ball_pos"])

        # Vitesse de la balle (si disponible)
        if "ball_vel" in observation:
            state.extend(observation["ball_vel"])
        else:
            state.extend([0.0, 0.0])

        # Position du joueur
        state.extend(observation["player_pos"])

        # Position de l'adversaire
        state.extend(observation["opponent_pos"])

        # Tailles des raquettes
        state.append(observation["player_paddle_size"])
        state.append(observation["opponent_paddle_size"])

        # Différence de score
        state.append(observation["score_diff"])

        # Temps écoulé (normalisé)
        state.append(observation["time_elapsed"] / 60.0)  # Assume max 60s par partie

        # Bonus actifs (jusqu'à 3 bonus, padding avec des zéros)
        bonuses = observation.get("bonuses", [])
        for i in range(3):
            if i < len(bonuses):
                state.extend(bonuses[i])  # [x, y, type]
            else:
                state.extend([0.0, 0.0, 0.0])

        # S'assurer que la taille correspond à state_size
        state = np.array(state, dtype=np.float32)
        if len(state) < self.state_size:
            # Padding avec des zéros si nécessaire
            state = np.pad(state, (0, self.state_size - len(state)))
        elif len(state) > self.state_size:
            # Tronquer si trop long
            state = state[: self.state_size]

        return state

    def get_action(self, observation: dict[str, Any]) -> Action:
        """Sélectionne une action basée sur la politique epsilon-greedy"""
        state = self._preprocess_observation(observation)

        # Exploration vs exploitation
        if random.random() < self.epsilon:
            # Action aléatoire
            action_idx = random.randint(0, self.action_size - 1)
        else:
            # Action optimale selon le Q-network
            with torch.no_grad():
                state_tensor = torch.FloatTensor(state).unsqueeze(0).to(self.device)
                q_values = self.q_network(state_tensor)
                action_idx = q_values.argmax().item()

        # Convertir l'index d'action en mouvement
        move_x, move_y = self.actions[action_idx]

        # Stocker pour l'apprentissage
        self.last_state = state
        self.last_action = action_idx

        return Action(move_x=float(move_x), move_y=float(move_y))

    def on_step(
        self,
        observation: dict[str, Any],
        action: Action,
        reward: float,
        done: bool,
        info: dict[str, Any],
    ) -> None:
        """Appelé après chaque étape pour l'apprentissage"""
        self.current_episode_reward += reward

        # Stocker la transition dans le replay buffer
        if hasattr(self, "last_state") and hasattr(self, "last_action"):
            next_state = self._preprocess_observation(observation)
            self.memory.push(self.last_state, self.last_action, reward, next_state, done)

        # Entraîner le réseau si suffisamment de données
        if len(self.memory) > self.batch_size:
            self._train()

        # Mettre à jour le réseau target périodiquement
        if self.steps % self.target_update == 0:
            self.target_network.load_state_dict(self.q_network.state_dict())

        # Décroissance d'epsilon
        if self.epsilon > self.epsilon_min:
            self.epsilon *= self.epsilon_decay

        self.steps += 1

    def _train(self) -> None:
        """Entraîne le réseau Q avec un batch du replay buffer"""
        if len(self.memory) < self.batch_size:
            return

        # Échantillonner un batch
        transitions = self.memory.sample(self.batch_size)
        batch = Transition(*zip(*transitions))

        # Convertir en tensors
        state_batch = torch.FloatTensor(np.array(batch.state)).to(self.device)
        action_batch = torch.LongTensor(batch.action).to(self.device)
        reward_batch = torch.FloatTensor(batch.reward).to(self.device)
        next_state_batch = torch.FloatTensor(np.array(batch.next_state)).to(self.device)
        done_batch = torch.BoolTensor(batch.done).to(self.device)

        # Q-values actuelles
        current_q_values = self.q_network(state_batch).gather(1, action_batch.unsqueeze(1))

        # Q-values suivantes (Double DQN)
        with torch.no_grad():
            next_actions = self.q_network(next_state_batch).argmax(1, keepdim=True)
            next_q_values = self.target_network(next_state_batch).gather(1, next_actions)
            target_q_values = reward_batch.unsqueeze(1) + (
                self.gamma * next_q_values * ~done_batch.unsqueeze(1)
            )

        # Calculer la perte
        loss = F.mse_loss(current_q_values, target_q_values)

        # Rétropropagation
        self.optimizer.zero_grad()
        loss.backward()
        # Gradient clipping pour la stabilité
        torch.nn.utils.clip_grad_norm_(self.q_network.parameters(), 1.0)
        self.optimizer.step()

        self.training_step += 1

    def save_model(self, filepath: str) -> None:
        """Sauvegarde le modèle"""
        torch.save(
            {
                "q_network_state_dict": self.q_network.state_dict(),
                "target_network_state_dict": self.target_network.state_dict(),
                "optimizer_state_dict": self.optimizer.state_dict(),
                "epsilon": self.epsilon,
                "steps": self.steps,
                "training_step": self.training_step,
            },
            filepath,
        )
        print(f"Modèle sauvegardé dans {filepath}")

    def load_model(self, filepath: str) -> None:
        """Charge un modèle"""
        checkpoint = torch.load(filepath, map_location=self.device)
        self.q_network.load_state_dict(checkpoint["q_network_state_dict"])
        self.target_network.load_state_dict(checkpoint["target_network_state_dict"])
        self.optimizer.load_state_dict(checkpoint["optimizer_state_dict"])
        self.epsilon = checkpoint["epsilon"]
        self.steps = checkpoint["steps"]
        self.training_step = checkpoint["training_step"]
        print(f"Modèle chargé depuis {filepath}")

    def set_training_mode(self, training: bool = True) -> None:
        """Active/désactive le mode entraînement"""
        if training:
            self.q_network.train()
        else:
            self.q_network.eval()
            self.epsilon = 0.0  # Pas d'exploration en mode évaluation


def create_dqn_agent(player_id: int, **kwargs: Any) -> DQNAgent:
    """Factory pour créer un agent DQN"""
    return DQNAgent(player_id, **kwargs)
