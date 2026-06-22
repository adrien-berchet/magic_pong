"""Deep Q-Network (DQN) AI implementation for Magic Pong using PyTorch.

Vanilla DQN with Double DQN action selection (van Hasselt et al., 2015).
"""

import random
from collections import deque
from collections import namedtuple
from typing import Any
from typing import cast

import numpy as np
import torch
import torch.nn as nn
import torch.nn.functional as F
import torch.optim as optim

from magic_pong.ai.interface import AIPlayer
from magic_pong.ai.models.dqn_checkpoint import build_dqn_checkpoint_metadata
from magic_pong.ai.models.dqn_checkpoint import safe_torch_load
from magic_pong.ai.models.dqn_checkpoint import validate_dqn_checkpoint
from magic_pong.core.entities import Action
from magic_pong.utils.config import game_config

# Transition tuple for replay buffer
Transition = namedtuple("Transition", ("state", "action", "reward", "next_state", "done"))

device = torch.device("cuda" if torch.cuda.is_available() else "cpu")


class DQNNetwork(nn.Module):
    """Deep Q-Network: stack of Linear→ReLU layers."""

    def __init__(
        self,
        input_size: int,
        hidden_size: int = 256,
        output_size: int = 9,
        layer_size: int = 2,
    ):
        super().__init__()
        layer_sizes = [hidden_size // (2**i) for i in range(layer_size)]

        self.fc_layers = nn.ModuleList()
        for i in range(layer_size):
            in_features = input_size if i == 0 else layer_sizes[i - 1]
            self.fc_layers.append(nn.Linear(in_features, layer_sizes[i]))

        self.output_layer = nn.Linear(layer_sizes[-1], output_size)
        self._initialize_weights()

    def _initialize_weights(self) -> None:
        for m in self.modules():
            if isinstance(m, nn.Linear):
                nn.init.kaiming_normal_(m.weight, mode="fan_in", nonlinearity="relu")
                nn.init.constant_(m.bias, 0.01)

    def forward(self, x: torch.Tensor) -> torch.Tensor:
        for fc in self.fc_layers:
            x = F.relu(fc(x))
        return self.output_layer(x)


class ReplayBuffer:
    """Uniform experience replay buffer."""

    def __init__(self, capacity: int):
        self.buffer: deque[Transition] = deque(maxlen=capacity)

    def add(
        self,
        state: np.ndarray,
        action: int,
        reward: float,
        next_state: np.ndarray,
        done: bool,
        td_error: float | None = None,
    ) -> None:
        self.buffer.append(Transition(state, action, reward, next_state, done))

    def sample(self, batch_size: int) -> list[Transition]:
        return random.sample(self.buffer, min(batch_size, len(self.buffer)))

    def __len__(self) -> int:
        return len(self.buffer)


class DQNAgent(AIPlayer):
    """Vanilla DQN agent with Double DQN action selection."""

    uses_player1_dqn_frame = True

    def __init__(
        self,
        state_size: int = 30,
        action_size: int = 9,
        lr: float = 0.001,
        gamma: float = 0.99,
        epsilon: float = 1.0,
        epsilon_min: float = 0.01,
        epsilon_decay: float = 0.9995,
        batch_size: int = 64,
        memory_size: int = 10000,
        train_frequency: int = 10,
        min_replay_size: int = 1000,
        target_update_frequency: int = 1000,
        name: str = "DQN AI",
    ):
        """
        Vanilla DQN agent.

        Args:
            state_size: Observation vector dimension (30 by default)
            action_size: Number of discrete actions
            lr: Learning rate
            gamma: Discount factor
            epsilon: Initial exploration probability
            epsilon_min: Minimum exploration probability
            epsilon_decay: Per-training-step epsilon multiplier
            batch_size: Minibatch size for replay
            memory_size: Replay buffer capacity
            train_frequency: Steps between training updates
            min_replay_size: Buffer must hold this many transitions before training begins
            target_update_frequency: Training steps between hard target-network copies
            name: Agent display name
        """
        super().__init__(name)

        self.state_size = state_size
        self.action_size = action_size
        self.lr = lr
        self.gamma = gamma
        self.epsilon = epsilon
        self.epsilon_min = epsilon_min
        self.epsilon_decay = epsilon_decay
        self.batch_size = batch_size
        self.train_frequency = train_frequency
        self.min_replay_size = min(memory_size, min_replay_size)
        self.target_update_frequency = target_update_frequency

        # Decision-time state/action waiting for the environment step result
        self.last_state: np.ndarray | None = None
        self.last_action: int | None = None
        self.training_enabled = True
        self.exploration_enabled = True
        # Kept for pretraining compatibility (pretraining.py reads/writes this)
        self._last_chosen_action: int | None = None

        self.step_count = 0

        self.q_network = DQNNetwork(state_size, output_size=action_size).to(device)
        self.target_network = DQNNetwork(state_size, output_size=action_size).to(device)

        self.optimizer = optim.Adam(self.q_network.parameters(), lr=lr)

        self.memory = ReplayBuffer(memory_size)

        self.update_target_network()

        self.training_step = 0
        self.loss_history: list[float] = []
        self.reward_history: list[float] = []

    def get_action(self, observation: dict[str, Any] | None, explore: bool | None = None) -> Action:
        """Convert observation to action via the Q-network (or epsilon-greedy)."""
        if observation is None:
            self._clear_pending_transition()
            self._last_chosen_action = None
            return Action(move_x=0.0, move_y=0.0)
        state = self._observation_to_state(observation)
        action_idx = self.act(state, explore=explore)
        if self.training_enabled:
            self.last_state = state
            self.last_action = action_idx
        return ACTION_MAPPING[action_idx]

    def on_step(
        self,
        observation: dict[str, Any],
        action: Action,
        reward: float,
        done: bool,
        info: dict[str, Any],
    ) -> None:
        """Store transition and train when conditions are met."""
        current_state = self._observation_to_state(observation)

        if self.training_enabled and self.last_state is not None and self.last_action is not None:
            self.remember(self.last_state, self.last_action, reward, current_state, done)
            self.step_count += 1

            if (
                len(self.memory) >= self.min_replay_size
                and self.step_count % self.train_frequency == 0
            ):
                self.replay()

        self.current_episode_reward += reward
        self._clear_pending_transition()

    def _observation_to_state(self, observation: dict[str, Any]) -> np.ndarray:
        """
        Convert a game observation dict to a 30-dimensional state vector.

        Dimensions: 10 base + 5 extra + 9 bonus (3×3) + 6 rotating paddle (2×3) = 30
        """
        _fw = float(game_config.FIELD_WIDTH)
        _fh = float(game_config.FIELD_HEIGHT)
        _ms = float(game_config.MAX_BALL_SPEED)

        # Base state: 10 dimensions
        base_state = [
            observation["ball_pos"][0] / _fw,
            observation["ball_pos"][1] / _fh,
            observation["ball_vel"][0] / _ms,
            observation["ball_vel"][1] / _ms,
            observation["player_pos"][0] / _fw,
            observation["player_pos"][1] / _fh,
            observation["opponent_pos"][0] / _fw,
            observation["opponent_pos"][1] / _fh,
            observation["opponent_previous_pos"][0] / _fw,
            observation["opponent_previous_pos"][1] / _fh,
        ]

        # Extra state: 5 dimensions
        extra_state = [
            self._get_paddle_size(observation.get("player_paddle_size", 1.0)),
            self._get_paddle_size(observation.get("opponent_paddle_size", 1.0)),
            observation.get("score_diff", 0.0) / 10.0,
            observation.get("time_elapsed", 0.0) / 300.0,
            len(observation.get("bonuses", [])) / 5.0,
        ]

        # Bonus state: up to 3 bonuses × 3 info = 9 dimensions
        bonus_state = []
        bonuses = observation.get("bonuses", [])
        for i in range(3):
            if i < len(bonuses):
                bonus = bonuses[i]
                if len(bonus) >= 3:
                    bonus_state.extend([bonus[0] / _fw, bonus[1] / _fh, bonus[2] / 3.0])
                else:
                    bonus_state.extend([0.0, 0.0, 0.0])
            else:
                bonus_state.extend([0.0, 0.0, 0.0])

        # Rotating paddle state: up to 2 paddles × 3 info = 6 dimensions
        rotating_paddle_state = []
        rotating_paddles = observation.get("rotating_paddles", [])
        for i in range(2):
            if i < len(rotating_paddles):
                rp = rotating_paddles[i]
                if len(rp) >= 3:
                    # Physics stores radians unwrapped; wrap to [-π, π] before normalizing
                    angle_rad = ((float(rp[2]) + np.pi) % (2 * np.pi)) - np.pi
                    normalized_angle = (angle_rad + np.pi) / (2 * np.pi)
                    rotating_paddle_state.extend([rp[0] / _fw, rp[1] / _fh, normalized_angle])
                else:
                    rotating_paddle_state.extend([0.0, 0.0, 0.0])
            else:
                rotating_paddle_state.extend([0.0, 0.0, 0.0])

        full_state = base_state + extra_state + bonus_state + rotating_paddle_state
        return np.array(full_state, dtype=np.float32)

    def _action_to_index(self, action: Action) -> int:
        for idx, mapped_action in ACTION_MAPPING.items():
            if (
                abs(mapped_action.move_x - action.move_x) < 0.1
                and abs(mapped_action.move_y - action.move_y) < 0.1
            ):
                return idx
        return 0

    def _get_paddle_size(self, paddle_size_data: float | list | tuple) -> float:
        if isinstance(paddle_size_data, list | tuple) and len(paddle_size_data) >= 2:
            size = paddle_size_data[1]
        elif isinstance(paddle_size_data, int | float):
            size = float(paddle_size_data)
        else:
            size = 1.0
        return float(np.clip(size / game_config.PADDLE_HEIGHT, 0.0, 4.0))

    def _extend_state(self, state: np.ndarray, prev_action: int | None) -> np.ndarray:
        """Identity — kept for pretraining compatibility."""
        return state

    def _clear_pending_transition(self) -> None:
        """Discard last_state/last_action after they are consumed."""
        self.last_state = None
        self.last_action = None

    def on_episode_start(self) -> None:
        super().on_episode_start()
        self._clear_pending_transition()
        self._last_chosen_action = None

    def on_episode_end(self) -> None:
        self._clear_pending_transition()
        self._last_chosen_action = None
        super().on_episode_end()

    def update_target_network(self) -> None:
        """Hard copy main network weights into the target network."""
        self.target_network.load_state_dict(self.q_network.state_dict())

    def set_training_mode(self, training: bool) -> None:
        """Toggle training behavior and module train/eval modes."""
        self.training_enabled = training
        self.exploration_enabled = training
        self.q_network.train(training)
        self.target_network.eval()
        self._clear_pending_transition()

    def set_exploration_mode(self, explore: bool) -> None:
        """Enable or disable epsilon-greedy exploration independently."""
        self.exploration_enabled = explore

    def remember(
        self,
        state: np.ndarray,
        action: int,
        reward: float,
        next_state: np.ndarray,
        done: bool,
    ) -> None:
        """Add a transition to the replay buffer."""
        self.memory.add(state, action, reward, next_state, done)

    def add_terminal_reward(self, extra_reward: float) -> None:
        """Add extra reward to the most recently stored transition.

        Used to apply post-episode outcomes (e.g. draw penalty) that are only
        known after the episode ends, without changing the environment interface.
        """
        if not self.training_enabled or not self.memory.buffer:
            return
        last = self.memory.buffer[-1]
        self.memory.buffer[-1] = Transition(
            last.state, last.action, last.reward + extra_reward, last.next_state, last.done
        )

    def act(
        self,
        state: np.ndarray,
        explore: bool | None = None,
        *,
        training: bool | None = None,
    ) -> int:
        """Choose an action via epsilon-greedy policy."""
        if training is not None:
            if explore is not None and explore != training:
                raise ValueError("act() received conflicting explore and training flags")
            explore = training

        if explore is None:
            explore = self.exploration_enabled

        if explore and random.random() < self.epsilon:
            action_idx = random.randrange(self.action_size)
            self._last_chosen_action = action_idx
            return action_idx

        state_tensor = torch.FloatTensor(state).unsqueeze(0).to(device)
        was_training = self.q_network.training
        self.q_network.eval()
        with torch.no_grad():
            q_values = self.q_network(state_tensor)
        self.q_network.train(was_training)

        action_idx = int(q_values.cpu().data.numpy().flatten().argmax())
        self._last_chosen_action = action_idx
        return action_idx

    def _compute_dqn_loss(
        self, experiences: list[Transition], weights: torch.Tensor
    ) -> torch.Tensor:
        """
        Compute Double DQN loss.

        Uses uniform weights=1 for vanilla DQN. The weights parameter is
        retained so this method can be called directly in tests.
        """
        states = torch.FloatTensor(np.array([e.state for e in experiences])).to(device)
        actions = torch.LongTensor([e.action for e in experiences]).to(device)
        rewards = torch.FloatTensor([e.reward for e in experiences]).to(device)
        next_states = torch.FloatTensor(np.array([e.next_state for e in experiences])).to(device)
        dones = torch.BoolTensor([e.done for e in experiences]).to(device)

        with torch.no_grad():
            # Double DQN: main network selects action, target network evaluates value
            next_actions = self.q_network(next_states).argmax(1).unsqueeze(1)
            next_q_values = self.target_network(next_states).gather(1, next_actions)
            target_q_values = rewards.unsqueeze(1) + (
                self.gamma * next_q_values * (~dones).unsqueeze(1)
            )

        current_q_values = self.q_network(states).gather(1, actions.unsqueeze(1))
        td_errors = target_q_values - current_q_values
        return cast(torch.Tensor, (weights.unsqueeze(1) * td_errors.pow(2)).mean())

    def replay(self) -> float | None:
        """Sample a minibatch and perform one gradient update."""
        if not self.training_enabled:
            return None
        if len(self.memory) < self.batch_size:
            return None

        experiences = self.memory.sample(self.batch_size)
        weights = torch.ones(len(experiences)).to(device)

        loss = self._compute_dqn_loss(experiences, weights)

        self.optimizer.zero_grad()
        loss.backward()
        torch.nn.utils.clip_grad_norm_(self.q_network.parameters(), max_norm=1.0)
        self.optimizer.step()

        if self.epsilon > self.epsilon_min:
            self.epsilon = max(self.epsilon_min, self.epsilon * self.epsilon_decay)

        self.training_step += 1
        self.loss_history.append(loss.item())

        if self.training_step % self.target_update_frequency == 0:
            self.update_target_network()

        return float(loss.item())

    def save_model(self, filepath: str) -> None:
        """Save checkpoint to disk."""
        torch.save(
            {
                "q_network_state_dict": self.q_network.state_dict(),
                "target_network_state_dict": self.target_network.state_dict(),
                "optimizer_state_dict": self.optimizer.state_dict(),
                "epsilon": self.epsilon,
                "training_step": self.training_step,
                "step_count": self.step_count,
                "loss_history": self.loss_history[-1000:],
                "reward_history": self.reward_history[-1000:],
                "metadata": build_dqn_checkpoint_metadata(
                    state_size=self.state_size,
                    action_size=self.action_size,
                ),
                "hyperparameters": {
                    "state_size": self.state_size,
                    "action_size": self.action_size,
                    "lr": self.lr,
                    "gamma": self.gamma,
                    "epsilon_min": self.epsilon_min,
                    "epsilon_decay": self.epsilon_decay,
                    "batch_size": self.batch_size,
                },
            },
            filepath,
        )

    def load_model(self, filepath: str) -> None:
        """Load checkpoint from disk."""
        checkpoint = safe_torch_load(filepath, map_location=device)
        validate_dqn_checkpoint(checkpoint)

        self.q_network.load_state_dict(checkpoint["q_network_state_dict"])
        self.target_network.load_state_dict(checkpoint["target_network_state_dict"])
        self.optimizer.load_state_dict(checkpoint["optimizer_state_dict"])
        self.epsilon = checkpoint["epsilon"]
        self.training_step = checkpoint["training_step"]
        self.step_count = checkpoint.get("step_count", 0)
        self.loss_history = checkpoint["loss_history"]
        self.reward_history = checkpoint["reward_history"]

        if "hyperparameters" in checkpoint:
            hyperparams = checkpoint["hyperparameters"]
            self.state_size = hyperparams["state_size"]
            self.action_size = hyperparams["action_size"]
            self.lr = hyperparams["lr"]
            self.gamma = hyperparams["gamma"]
            self.epsilon_min = hyperparams["epsilon_min"]
            self.epsilon_decay = hyperparams["epsilon_decay"]
            self.batch_size = hyperparams["batch_size"]

        self._clear_pending_transition()

    def get_training_stats(self) -> dict:
        """Return training statistics."""
        return {
            "training_step": self.training_step,
            "step_count": self.step_count,
            "train_frequency": self.train_frequency,
            "min_replay_size": self.min_replay_size,
            "memory_size": len(self.memory),
            "epsilon": self.epsilon,
            "avg_loss": (float(np.mean(self.loss_history[-100:])) if self.loss_history else 0.0),
            "avg_reward": (
                float(np.mean(self.reward_history[-100:])) if self.reward_history else 0.0
            ),
            "current_lr": self.optimizer.param_groups[0]["lr"],
        }


# Actions mapped according to the game interface
ACTION_MAPPING = {
    0: Action(0.0, 0.0),  # Stay still
    1: Action(0.0, -1.0),  # Up
    2: Action(0.0, 1.0),  # Down
    3: Action(-1.0, 0.0),  # Left
    4: Action(1.0, 0.0),  # Right
    5: Action(-1.0, -1.0),  # Up-left
    6: Action(1.0, -1.0),  # Up-right
    7: Action(-1.0, 1.0),  # Down-left
    8: Action(1.0, 1.0),  # Down-right
}
