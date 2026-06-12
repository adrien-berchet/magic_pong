"""
Deep Q-Network (DQN) AI implementation for Magic Pong using PyTorch
Improved version with stabilization techniques
"""

import random
from collections import deque
from collections import namedtuple
from contextlib import contextmanager
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


@contextmanager
def _temporary_eval_mode(*modules: nn.Module) -> Any:
    """Temporarily disable training-only module behavior while preserving gradients."""
    was_training = [module.training for module in modules]
    try:
        for module in modules:
            module.eval()
        yield
    finally:
        for module, training in zip(modules, was_training, strict=False):
            module.train(training)


class HybridRewardCalculator:
    """
    Professional reward calculator for dual-scale training.

    Implements two complementary reward systems:
    - Tactical rewards: Immediate feedback for ball tracking and positioning
    - Strategic rewards: Long-term credit assignment for match outcomes and sequences
    """

    def __init__(self, gamma: float = 0.99):
        """
        Initialize the hybrid reward calculator.

        Args:
            gamma: Discount factor for strategic reward calculation
        """
        self.gamma = gamma

    def calculate_tactical_reward(
        self, observation: dict[str, Any], action: Action, base_reward: float
    ) -> float:
        """
        Calculate immediate tactical rewards for step-by-step learning.

        Focus areas:
        - Ball tracking and prediction
        - Immediate positioning optimization
        - Movement efficiency
        - Reaction speed

        Args:
            observation: Current game observation
            action: Action taken by the agent
            base_reward: Original reward from the environment

        Returns:
            Enhanced tactical reward for immediate learning
        """
        tactical_reward = 0.0

        # Extract key information
        raw_ball_pos = observation.get("ball_pos", [0.5, 0.5])
        raw_player_pos = observation.get("player_pos", [0.0, 0.5])
        raw_ball_vel = observation.get("ball_vel", [0.0, 0.0])
        field_width = float(observation.get("field_width", 1.0))
        field_height = float(observation.get("field_height", 1.0))

        ball_pos = [
            self._normalize_position_axis(float(raw_ball_pos[0]), field_width),
            self._normalize_position_axis(float(raw_ball_pos[1]), field_height),
        ]
        player_pos = [
            self._normalize_position_axis(float(raw_player_pos[0]), field_width),
            self._normalize_position_axis(float(raw_player_pos[1]), field_height),
        ]
        ball_vel = [
            self._normalize_velocity_axis(float(raw_ball_vel[0])),
            self._normalize_velocity_axis(float(raw_ball_vel[1])),
        ]

        # 1. Ball tracking reward - core tactical skill
        if abs(ball_vel[0]) > 0.01:  # Ball is moving horizontally
            # Predict ball position in next few frames (simple linear prediction)
            prediction_time = 0.1  # Look ahead 100ms
            predicted_ball_y = ball_pos[1] + ball_vel[1] * prediction_time
            predicted_ball_y = np.clip(predicted_ball_y, 0.0, 1.0)  # Clamp to valid range

            # Reward positioning when the ball approaches the left-side canonical player.
            if ball_vel[0] < 0 and ball_pos[0] < 0.7:  # Ball coming towards player
                distance_to_predicted = abs(player_pos[1] - predicted_ball_y)
                tracking_reward = max(0, 1.0 - distance_to_predicted * 3.0) * 0.15
                tactical_reward += tracking_reward

        # 2. Optimal positioning reward for interception
        if ball_vel[0] < 0 and ball_pos[0] < 0.5:  # Ball approaching player side
            distance_to_ball_y = abs(ball_pos[1] - player_pos[1])
            if distance_to_ball_y < 0.15:  # Well positioned for interception
                positioning_reward = (0.15 - distance_to_ball_y) / 0.15 * 0.1
                tactical_reward += positioning_reward

        # 3. Movement efficiency - avoid unnecessary actions
        movement_magnitude = abs(action.move_x) + abs(action.move_y)

        # Encourage stillness when well-positioned
        if ball_vel[0] > 0 or abs(ball_pos[1] - player_pos[1]) < 0.08:  # Ball away or aligned
            if movement_magnitude < 0.2:  # Minimal movement
                efficiency_reward = 0.01
                tactical_reward += efficiency_reward

        # 4. Immediate reaction bonus for ball contact situations
        if (
            ball_vel[0] < 0 and ball_pos[0] < 0.2 and abs(ball_pos[1] - player_pos[1]) < 0.1
        ):  # Close to contact
            reaction_bonus = 0.08
            tactical_reward += reaction_bonus

        # 5. Defensive positioning when ball is distant
        if ball_pos[0] > 0.7:  # Ball on opponent side
            center_distance = abs(player_pos[1] - 0.5)  # Distance from center
            if center_distance < 0.25:  # Stay reasonably centered
                defensive_reward = (0.25 - center_distance) / 0.25 * 0.05
                tactical_reward += defensive_reward

        return tactical_reward

    def _normalize_position_axis(self, value: float, extent: float) -> float:
        if extent > 1.0 and abs(value) > 1.0:
            return value / extent
        return value

    def _normalize_velocity_axis(self, value: float) -> float:
        if abs(value) > 1.0:
            return value / game_config.MAX_BALL_SPEED
        return value

    def calculate_strategic_reward(
        self,
        episode_rewards: list[float],
        episode_observations: list[dict[str, Any]],
        episode_actions: list[Action],
    ) -> list[float]:
        """
        Calculate strategic rewards for episode-end learning.

        Focus areas:
        - Match outcome influence
        - Successful rally sequences
        - Long-term positioning strategy
        - Sequence completion bonuses

        Args:
            episode_rewards: List of immediate rewards for each step
            episode_observations: List of observations for each step
            episode_actions: List of actions for each step

        Returns:
            List of strategic rewards with proper credit assignment
        """
        if not episode_rewards:
            return []

        # Calculate base discounted rewards for temporal credit assignment
        discounted_rewards = self._calculate_discounted_rewards(episode_rewards)

        # Calculate strategic bonuses for each step
        strategic_rewards = []
        for i, base_reward in enumerate(discounted_rewards):
            strategic_reward = base_reward

            # 1. Rally contribution bonus - actions that led to successful ball returns
            rally_bonus = self._calculate_rally_contribution_bonus(i, episode_observations)
            strategic_reward += rally_bonus

            # 2. Match outcome influence - how actions affected final result
            outcome_bonus = self._calculate_match_outcome_bonus(
                i, episode_observations, episode_rewards
            )
            strategic_reward += outcome_bonus

            # 3. Defensive strategy bonus - maintaining good court positioning
            defensive_bonus = self._calculate_defensive_strategy_bonus(
                i, episode_observations, episode_actions
            )
            strategic_reward += defensive_bonus

            # 4. Sequence coherence bonus - reward consistent tactical execution
            coherence_bonus = self._calculate_sequence_coherence_bonus(
                i, episode_actions, episode_observations
            )
            strategic_reward += coherence_bonus

            strategic_rewards.append(strategic_reward)

        return strategic_rewards

    def _calculate_rally_contribution_bonus(
        self, step: int, observations: list[dict[str, Any]]
    ) -> float:
        """Calculate bonus for actions that contributed to successful rallies."""
        # Look ahead to find ball contact events
        rally_bonus = 0.0
        look_ahead_window = min(25, len(observations) - step - 1)

        for future_step in range(step + 1, step + 1 + look_ahead_window):
            if future_step >= len(observations):
                break

            obs = observations[future_step]
            events = obs.get("events", [])

            # Check for successful ball contact
            if self._has_successful_contact_event(events):
                # Earlier actions in the sequence get progressively more credit
                time_to_contact = future_step - step
                if time_to_contact <= 15:  # Within reasonable sequence length
                    credit_factor = max(0.1, 1.0 - (time_to_contact / 15.0))
                    rally_bonus += 2.5 * credit_factor
                break  # Only credit for first successful contact in sequence

        return rally_bonus

    def _has_successful_contact_event(self, events: Any) -> bool:
        """Return whether canonical or legacy event payloads include this agent's contact."""
        if isinstance(events, dict):
            return self._has_player1_contact(
                events.get("paddle_hits")
            ) or self._has_player1_contact(events.get("rotating_paddle_hits"))

        if isinstance(events, list):
            return any(
                "ball_hit" in str(event).lower()
                or "contact" in str(event).lower()
                or "paddle_hit" in str(event).lower()
                for event in events
            )

        return False

    def _has_player1_contact(self, bucket: Any) -> bool:
        """Return whether a contact bucket contains canonical player-1 contact."""
        if isinstance(bucket, list):
            return any(self._is_player1_contact_event(event) for event in bucket)
        if isinstance(bucket, dict):
            return self._is_player1_contact_event(bucket)
        return False

    def _is_player1_contact_event(self, event: Any) -> bool:
        if not isinstance(event, dict):
            return False
        return event.get("player") == 1

    def _calculate_match_outcome_bonus(
        self, step: int, observations: list[dict[str, Any]], rewards: list[float]
    ) -> float:
        """Calculate bonus based on contribution to match outcome."""
        if len(observations) < 20:  # Too short to evaluate match outcome
            return 0.0

        # Analyze final observations for match outcome
        total_final_reward = sum(rewards[-10:])  # Sum of final rewards

        # Determine if match was won or lost based on reward patterns
        match_outcome = 0.0
        if total_final_reward > 20:  # Likely won the match
            match_outcome = 1.5
        elif total_final_reward < -20:  # Likely lost the match
            match_outcome = -0.8

        if abs(match_outcome) > 0.1:
            # Distribute outcome influence across the episode
            # Later actions have more influence on the outcome
            episode_length = len(observations)
            time_factor = (step / episode_length) ** 1.5  # Exponential weighting towards end
            influence_bonus = match_outcome * time_factor * 0.7
            return float(influence_bonus)

        return 0.0

    def _calculate_defensive_strategy_bonus(
        self, step: int, observations: list[dict[str, Any]], actions: list[Action]
    ) -> float:
        """Reward maintaining good defensive positioning over time."""
        if step < 8:  # Need sufficient history
            return 0.0

        # Analyze positioning strategy over recent history
        analysis_window = min(8, step + 1)
        recent_obs = observations[step - analysis_window + 1 : step + 1]

        good_defensive_positions = 0
        total_defensive_situations = 0

        for obs in recent_obs:
            raw_ball_pos = obs.get("ball_pos", [0.5, 0.5])
            raw_player_pos = obs.get("player_pos", [0.0, 0.5])
            raw_ball_vel = obs.get("ball_vel", [0.0, 0.0])
            field_width = float(obs.get("field_width", 1.0))
            field_height = float(obs.get("field_height", 1.0))

            ball_pos = [
                self._normalize_position_axis(float(raw_ball_pos[0]), field_width),
                self._normalize_position_axis(float(raw_ball_pos[1]), field_height),
            ]
            player_pos = [
                self._normalize_position_axis(float(raw_player_pos[0]), field_width),
                self._normalize_position_axis(float(raw_player_pos[1]), field_height),
            ]
            ball_vel = [
                self._normalize_velocity_axis(float(raw_ball_vel[0])),
                self._normalize_velocity_axis(float(raw_ball_vel[1])),
            ]

            # Identify defensive situations (ball far away or moving away)
            if ball_pos[0] > 0.6 or ball_vel[0] > 0.1:
                total_defensive_situations += 1

                # Good defensive position: stay reasonably centered
                center_distance = abs(player_pos[1] - 0.5)
                if center_distance < 0.3:
                    good_defensive_positions += 1

        if total_defensive_situations > 0:
            defensive_ratio = good_defensive_positions / total_defensive_situations
            strategy_bonus = defensive_ratio * 0.4
            return strategy_bonus

        return 0.0

    def _calculate_sequence_coherence_bonus(
        self, step: int, actions: list[Action], observations: list[dict[str, Any]]
    ) -> float:
        """Reward coherent tactical sequences (avoid erratic movement patterns)."""
        if step < 6:  # Need sufficient sequence length
            return 0.0

        # Analyze recent movement patterns
        sequence_length = min(6, step + 1)
        recent_actions = actions[step - sequence_length + 1 : step + 1]

        # Calculate movement consistency
        y_movements = [action.move_y for action in recent_actions]

        # Penalize excessive oscillation
        oscillation_penalty = 0.0
        direction_changes = 0
        for i in range(1, len(y_movements)):
            if (
                len(y_movements) > i and y_movements[i] * y_movements[i - 1] < -0.1
            ):  # Direction change
                direction_changes += 1

        if direction_changes > 2:  # Too many direction changes
            oscillation_penalty = -0.15 * (direction_changes - 2)

        # Reward smooth, purposeful movement
        if direction_changes <= 1:  # Consistent movement direction
            consistency_bonus = 0.1
        else:
            consistency_bonus = 0.0

        return consistency_bonus + oscillation_penalty

    def _calculate_discounted_rewards(self, rewards: list[float]) -> list[float]:
        """Calculate discounted cumulative rewards for proper credit assignment."""
        if not rewards:
            return []

        discounted_rewards = []
        cumulative_reward = 0.0

        # Calculate in reverse order (from end to beginning)
        for reward in reversed(rewards):
            cumulative_reward = reward + self.gamma * cumulative_reward
            discounted_rewards.append(cumulative_reward)

        # Reverse to get chronological order
        discounted_rewards.reverse()

        return discounted_rewards


class DQNNetwork(nn.Module):
    """Deep Q-Network with stability improvements"""

    def __init__(
        self,
        input_size: int,
        hidden_size: int = 512,
        output_size: int = 9,
        layer_size: int = 3,
        use_normalization: bool = True,
        dropout_rate: float = 0.0,
    ):
        """
        Args:
            input_size: Input state size
            hidden_size: Hidden layers size
            output_size: Number of possible actions
            dropout_rate: Dropout probability for hidden layers. Defaults to disabled.
        """
        super().__init__()
        self.dropout_rate = dropout_rate

        # More stable architecture with less aggressive progressive reduction
        layer_sizes = [hidden_size // (2**i) for i in range(layer_size)]

        # Deeper and stable architecture
        self.fc_layers = nn.ModuleList()
        for i in range(layer_size):
            in_features = input_size if i == 0 else layer_sizes[i - 1]
            out_features = layer_sizes[i]
            self.fc_layers.append(nn.Linear(in_features, out_features))

        self.output_layer = nn.Linear(layer_sizes[-1], output_size)

        # Batch normalization to improve stability
        self.use_normalization = use_normalization
        if use_normalization:
            # self.batch_norms = nn.ModuleList()
            self.layer_norms = nn.ModuleList()
            for i in range(layer_size):
                # self.batch_norms.append(nn.BatchNorm1d(layer_sizes[i]))
                self.layer_norms.append(nn.LayerNorm(layer_sizes[i]))
        else:
            # self.batch_norms = nn.ModuleList([nn.Identity() for _ in range(layer_size)])
            self.layer_norms = nn.ModuleList([nn.Identity() for _ in range(layer_size)])

        # Dropout is configurable and disabled by default for deterministic DQN targets.
        self.dropouts = nn.ModuleList()
        for _ in range(layer_size):
            self.dropouts.append(nn.Dropout(dropout_rate))

        # Xavier initialization for stability
        self._initialize_weights()

    def _initialize_weights(self) -> None:
        """Xavier weight initialization"""
        for m in self.modules():
            if isinstance(m, nn.Linear):
                # nn.init.xavier_uniform_(m.weight)
                nn.init.kaiming_normal_(m.weight, mode="fan_in", nonlinearity="relu")
                nn.init.constant_(m.bias, 0.01)

    def forward(self, x: torch.Tensor) -> torch.Tensor:
        """Forward pass with normalization and dropout"""
        x = x.to(device)

        for fc, ln, dropout in zip(self.fc_layers, self.layer_norms, self.dropouts):
            x = fc(x)
            if self.use_normalization:
                x = ln(x)
            x = F.relu(x)
            x = dropout(x)

        # Output layer
        x = self.output_layer(x)
        return x

    def set_dropout_rate(self, dropout_rate: float) -> None:
        """Update dropout probability without changing checkpoint state shape."""
        self.dropout_rate = dropout_rate
        for dropout in self.dropouts:
            dropout.p = dropout_rate


class PrioritizedReplayBuffer:
    """Replay buffer with prioritized sampling based on TD error"""

    def __init__(
        self, capacity: int, alpha: float = 0.6, beta: float = 0.4, beta_increment: float = 0.001
    ):
        self.capacity = capacity
        self.alpha = alpha  # priority
        self.beta = beta  # importance correction
        self.beta_increment = beta_increment

        self.buffer: deque[Transition] = deque(maxlen=capacity)
        self.priorities: deque[float] = deque(maxlen=capacity)
        self.epsilon: float = 1e-6  # to avoid zero priorities

    def add(
        self,
        state: np.ndarray,
        action: int,
        reward: float,
        next_state: np.ndarray,
        done: bool,
        td_error: float | None = None,
    ) -> None:
        """Add a transition with priority based on TD error"""
        if td_error is None:
            # If no TD error provided, use maximum priority
            priority = max(self.priorities) if self.priorities else 1.0
        else:
            priority = abs(td_error) + self.epsilon

        transition = Transition(state, action, reward, next_state, done)
        self.buffer.append(transition)
        self.priorities.append(priority**self.alpha)

    def sample(self, batch_size: int) -> tuple[list[Transition], np.ndarray, np.ndarray]:
        """Prioritized sampling"""
        if len(self.buffer) < batch_size:
            batch_size = len(self.buffer)

        # Calculate probabilities
        priorities = np.array(self.priorities)
        probabilities = priorities / priorities.sum()

        # Sampling
        indices = np.random.choice(len(self.buffer), batch_size, p=probabilities)
        samples = [self.buffer[i] for i in indices]

        # Importance weights
        weights = (len(self.buffer) * probabilities[indices]) ** (-self.beta)
        weights = weights / weights.max()  # normalization

        # Increment beta
        self.beta = min(1.0, self.beta + self.beta_increment)

        return samples, weights, indices

    def update_priorities(self, indices: np.ndarray, td_errors: np.ndarray) -> None:
        """Update priorities with new TD errors"""
        for idx, td_error in zip(indices, td_errors, strict=False):
            priority = (abs(td_error) + self.epsilon) ** self.alpha
            self.priorities[idx] = priority

    def __len__(self) -> int:
        return len(self.buffer)


class ReplayBuffer:
    """Simple replay buffer for comparison"""

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
        """Add a transition"""
        transition = Transition(state, action, reward, next_state, done)
        self.buffer.append(transition)

    def sample(self, batch_size: int) -> list[Transition]:
        """Random sampling"""
        return random.sample(self.buffer, min(batch_size, len(self.buffer)))

    def __len__(self) -> int:
        return len(self.buffer)


class DQNAgent(AIPlayer):
    """DQN Agent with stability improvements"""

    uses_player1_dqn_frame = True

    def __init__(
        self,
        state_size: int = 32,
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
        train_frequency: int = 10,
        min_replay_size: int = 1000,
        training_mode: str = "episode_end",
        reward_normalization: bool = True,
        # Dual-scale training parameters
        enable_dual_scale_training: bool = False,
        tactical_train_frequency: int = 10,
        tactical_learning_rate: float | None = None,
        strategic_learning_rate: float | None = None,
        dropout_rate: float = 0.0,
        name: str = "DQN AI",
    ):
        """
        Professional DQN Agent with hybrid and dual-scale training capabilities.

        Args:
            state_size: Size of the state space
            action_size: Number of possible actions
            lr: Base learning rate
            gamma: Discount factor
            epsilon: Initial exploration probability
            epsilon_min: Minimum exploration probability
            epsilon_decay: Epsilon decay rate
            batch_size: Batch size for training
            memory_size: Replay buffer size
            use_prioritized_replay: Whether to use prioritized experience replay
            tau: Soft update coefficient for target network
            train_frequency: Training frequency (1 = every step, 10 = every 10 steps)
            min_replay_size: Minimum buffer size before starting training
            training_mode: "episode_end" for better credit assignment, "step_by_step" for immediate training
            reward_normalization: Whether to normalize rewards for episode-end training
            enable_dual_scale_training: Enable advanced dual-scale learning (tactical + strategic)
            tactical_train_frequency: How often to perform tactical training (default: every 10 steps)
            tactical_learning_rate: Learning rate for tactical training (default: lr * 0.3)
            strategic_learning_rate: Learning rate for strategic training (default: lr * 1.5)
            dropout_rate: Dropout probability for DQN hidden layers. Defaults to disabled.
            name: Agent name
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
        self.tau = tau
        self.train_frequency = train_frequency
        self.min_replay_size = min(memory_size, min_replay_size)
        self.dropout_rate = dropout_rate

        # Hybrid training configuration
        self.training_mode = training_mode
        self.reward_normalization = reward_normalization

        # Dual-scale training configuration
        self.enable_dual_scale_training = enable_dual_scale_training
        self.tactical_train_frequency = tactical_train_frequency
        self.tactical_learning_rate = (
            tactical_learning_rate if tactical_learning_rate is not None else lr * 0.3
        )
        self.strategic_learning_rate = (
            strategic_learning_rate if strategic_learning_rate is not None else lr * 1.5
        )

        # Episode buffer for episode-end and strategic training
        self.episode_buffer: list[dict[str, Any]] = []
        self.episode_rewards: list[float] = []

        # Decision-time state/action waiting for the environment step result.
        self.last_state: np.ndarray | None = None
        self.last_action: int | None = None
        self.training_enabled = True
        self.exploration_enabled = True

        # Step counters for controlling training frequency
        self.step_count = 0
        self.tactical_step_count = 0

        # Neural networks
        self.q_network = DQNNetwork(state_size, 512, action_size, dropout_rate=dropout_rate).to(
            device
        )
        self.target_network = DQNNetwork(
            state_size, 512, action_size, dropout_rate=dropout_rate
        ).to(device)

        # Initialize optimizers based on training mode
        if self.enable_dual_scale_training:
            # Separate optimizers for tactical and strategic learning
            self.tactical_optimizer = optim.Adam(
                self.q_network.parameters(),
                lr=self.tactical_learning_rate,
                weight_decay=1e-4,
                eps=1e-4,
            )
            self.strategic_optimizer = optim.Adam(
                self.q_network.parameters(),
                lr=self.strategic_learning_rate,
                weight_decay=1e-3,
                eps=1e-4,
            )
            # Keep original optimizer for compatibility
            self.optimizer = self.strategic_optimizer

            # Initialize reward calculator
            self.reward_calculator = HybridRewardCalculator(self.gamma)

            # Dual-scale training statistics
            self.tactical_loss_history: list[float] = []
            self.strategic_loss_history: list[float] = []
        else:
            # Standard single optimizer
            self.optimizer = optim.Adam(
                self.q_network.parameters(), lr=lr, weight_decay=1e-3, eps=1e-4
            )

        # Learning rate scheduler
        self.scheduler = optim.lr_scheduler.ReduceLROnPlateau(
            self.optimizer, mode="min", factor=0.5, patience=100
        )

        # Replay buffer
        self.memory: ReplayBuffer | PrioritizedReplayBuffer
        if use_prioritized_replay:
            self.memory = PrioritizedReplayBuffer(memory_size)
        else:
            self.memory = ReplayBuffer(memory_size)

        self.use_prioritized_replay = use_prioritized_replay

        # Copy initial weights
        self.update_target_network()

        # Training statistics
        self.training_step = 0
        self.loss_history: list[float] = []
        self.reward_history: list[float] = []

    def get_action(self, observation: dict[str, Any] | None, explore: bool | None = None) -> Action:
        """
        Required interface: converts observation to action

        Args:
            observation: Formatted game observation
            explore: Whether to use epsilon-greedy exploration. Defaults to the agent's training mode.

        Returns:
            Action: Action to perform
        """
        if observation is None:
            self._clear_pending_transition()
            return Action(move_x=0.0, move_y=0.0)
        # Convert observation to state vector
        state = self._observation_to_state(observation)

        # Get numeric action
        action_idx = self.act(state, explore=explore)
        if self.training_enabled:
            self.last_state = state.copy()
            self.last_action = action_idx

        # Convert to game Action
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
        Advanced training interface supporting hybrid and dual-scale learning.

        Training modes:
        - Standard hybrid: "episode_end" or "step_by_step"
        - Dual-scale: Advanced combination of tactical + strategic learning

        Args:
            observation: New observation from the environment
            action: Action performed
            reward: Reward received
            done: Whether the episode is finished
            info: Additional information
        """
        current_state = self._observation_to_state(observation)

        if not self.training_enabled:
            self.current_episode_reward += reward
            self._clear_pending_transition()
            return

        # Route to appropriate training method
        if self.enable_dual_scale_training:
            self._on_step_dual_scale(current_state, action, reward, done, info, observation)
        else:
            # Standard hybrid training
            if self.training_mode == "episode_end":
                self._on_step_episode_mode(current_state, action, reward, done, info)
            else:
                self._on_step_traditional(current_state, action, reward, done, info)

        # Update episode reward (common to all modes)
        self.current_episode_reward += reward

        self._clear_pending_transition()

    def _on_step_dual_scale(
        self,
        current_state: np.ndarray,
        action: Action,
        reward: float,
        done: bool,
        info: dict[str, Any],
        observation: dict[str, Any],
    ) -> None:
        """
        Handle dual-scale training: simultaneous tactical and strategic learning.

        This method implements the core dual-scale training logic:
        - Tactical training: Frequent updates with low LR for immediate feedback
        - Strategic training: Episode-end updates with high LR for long-term credit assignment
        """

        # Always store experience for both training types
        if self.last_state is not None and self.last_action is not None:
            strategic_observation = observation.copy()
            strategic_observation["events"] = info.get("events", {})

            # Store for strategic episode-end training
            strategic_experience = {
                "state": self.last_state.copy(),
                "action": self.last_action,
                "reward": reward,
                "next_state": current_state.copy(),
                "done": done,
                "observation": strategic_observation,
            }
            self.episode_buffer.append(strategic_experience)
            self.episode_rewards.append(reward)

            # Calculate tactical reward for immediate training
            tactical_reward = self.reward_calculator.calculate_tactical_reward(
                observation, action, reward
            )

            # Store tactical experience in main replay buffer
            self.memory.add(self.last_state, self.last_action, tactical_reward, current_state, done)

            self.tactical_step_count += 1

            # Tactical training (frequent, conservative learning rate)
            if (
                len(self.memory) >= self.min_replay_size
                and self.tactical_step_count % self.tactical_train_frequency == 0
            ):
                tactical_loss = self._train_tactical()
                if tactical_loss is not None:
                    self.tactical_loss_history.append(tactical_loss)

        # Strategic training at episode end (less frequent, aggressive learning rate)
        if done and len(self.episode_buffer) > 0:
            strategic_loss = self._train_strategic_on_episode()
            if strategic_loss is not None:
                self.strategic_loss_history.append(strategic_loss)
            self._reset_episode_buffer()

    def _train_tactical(self) -> float | None:
        """
        Perform tactical training with conservative learning rate.

        Focus: Immediate feedback, ball tracking, positioning optimization.
        Frequency: Every N steps (e.g., every 10 steps)
        Learning Rate: Lower (e.g., 0.0003)
        """
        if len(self.memory) < self.batch_size:
            return None

        # Sample experiences for tactical training
        if self.use_prioritized_replay:
            sampled_data = cast(
                tuple[list[Transition], np.ndarray, np.ndarray],
                self.memory.sample(self.batch_size),
            )
            experiences, weights_arr, indices = sampled_data
            weights = torch.FloatTensor(weights_arr).to(device)
        else:
            experiences = cast(list[Transition], self.memory.sample(self.batch_size))
            weights = torch.ones(len(experiences)).to(device)
            indices = None

        # Compute tactical loss
        loss = self._compute_dqn_loss(experiences, weights)

        # Tactical optimization (conservative)
        self.tactical_optimizer.zero_grad()
        loss.backward()
        torch.nn.utils.clip_grad_norm_(self.q_network.parameters(), max_norm=1.0)
        self.tactical_optimizer.step()

        # Update priorities if using prioritized replay
        if self.use_prioritized_replay and indices is not None:
            # Recalculate TD errors for priority updates
            with torch.no_grad():
                td_errors = self._compute_td_errors(experiences)
                # Cast memory to PrioritizedReplayBuffer since we're in the prioritized branch
                prb = cast("PrioritizedReplayBuffer", self.memory)
                prb.update_priorities(indices, td_errors.cpu().numpy())

        # Gentle target network update for tactical training
        self.soft_update_target_network()

        return float(loss.item())

    def _train_strategic_on_episode(self) -> float | None:
        """
        Perform strategic training with aggressive learning rate at episode end.

        Focus: Long-term credit assignment, match outcomes, sequence completion.
        Frequency: Once per episode
        Learning Rate: Higher (e.g., 0.0015)
        """
        if len(self.episode_buffer) == 0:
            return None

        # Calculate enhanced strategic rewards
        episode_observations = [exp["observation"] for exp in self.episode_buffer]
        episode_actions = [ACTION_MAPPING[exp["action"]] for exp in self.episode_buffer]

        strategic_rewards = self.reward_calculator.calculate_strategic_reward(
            self.episode_rewards, episode_observations, episode_actions
        )

        # Create strategic training experiences
        strategic_experiences = []
        for i, experience in enumerate(self.episode_buffer):
            strategic_exp = Transition(
                experience["state"],
                experience["action"],
                strategic_rewards[i],  # Enhanced reward with credit assignment
                experience["next_state"],
                experience["done"],
            )
            strategic_experiences.append(strategic_exp)

        # Strategic training with multiple updates for stronger learning
        total_loss = 0.0
        num_updates = 0

        # Perform multiple training iterations for strategic learning
        iterations = max(1, len(strategic_experiences) // self.batch_size)
        for _ in range(min(iterations, 3)):  # Limit to prevent overfitting
            if len(strategic_experiences) >= self.batch_size:
                # Sample from strategic experiences
                batch_experiences = random.sample(strategic_experiences, self.batch_size)
                weights = torch.ones(len(batch_experiences)).to(device)

                # Compute strategic loss
                loss = self._compute_dqn_loss(batch_experiences, weights)

                # Strategic optimization (aggressive)
                self.strategic_optimizer.zero_grad()
                loss.backward()
                torch.nn.utils.clip_grad_norm_(self.q_network.parameters(), max_norm=2.0)
                self.strategic_optimizer.step()

                total_loss += loss.item()
                num_updates += 1

        # Multiple target network updates for strategic training
        for _ in range(2):  # More significant target network updates
            self.soft_update_target_network()

        # Update training statistics
        if num_updates > 0:
            avg_episode_reward = float(np.mean(self.episode_rewards))
            self.reward_history.append(avg_episode_reward)
            return total_loss / num_updates

        return None

    def _compute_dqn_loss(
        self, experiences: list[Transition], weights: torch.Tensor
    ) -> torch.Tensor:
        """
        Compute Double DQN loss for given experiences with importance sampling weights.

        Args:
            experiences: List of transitions
            weights: Importance sampling weights for prioritized replay

        Returns:
            Computed loss tensor
        """
        # Extract data from experiences
        states = torch.FloatTensor(np.array([e.state for e in experiences])).to(device)
        actions = torch.LongTensor([e.action for e in experiences]).to(device)
        rewards = torch.FloatTensor([e.reward for e in experiences]).to(device)
        next_states = torch.FloatTensor(np.array([e.next_state for e in experiences])).to(device)
        dones = torch.BoolTensor([e.done for e in experiences]).to(device)

        with _temporary_eval_mode(self.q_network, self.target_network):
            # Current Q-values keep gradients, but dropout stays inactive.
            current_q_values = self.q_network(states).gather(1, actions.unsqueeze(1))

            # Double DQN target values
            with torch.no_grad():
                # Action selection with main network
                next_actions = self.q_network(next_states).argmax(1).unsqueeze(1)
                # Value estimation with target network
                next_q_values = self.target_network(next_states).gather(1, next_actions)
                target_q_values = rewards.unsqueeze(1) + (
                    self.gamma * next_q_values * (~dones).unsqueeze(1)
                )

        # Compute weighted MSE loss
        td_errors = target_q_values - current_q_values
        weighted_loss: torch.Tensor = (weights.unsqueeze(1) * td_errors.pow(2)).mean()

        return weighted_loss

    def _compute_td_errors(self, experiences: list[Transition]) -> torch.Tensor:
        """Compute TD errors for priority updates in prioritized replay."""
        # Extract data from transitions
        states = torch.FloatTensor(np.array([e.state for e in experiences])).to(device)
        actions = torch.LongTensor([e.action for e in experiences]).to(device)
        rewards = torch.FloatTensor([e.reward for e in experiences]).to(device)
        next_states = torch.FloatTensor(np.array([e.next_state for e in experiences])).to(device)
        dones = torch.BoolTensor([e.done for e in experiences]).to(device)

        with _temporary_eval_mode(self.q_network, self.target_network), torch.no_grad():
            current_q_values = self.q_network(states).gather(1, actions.unsqueeze(1))
            next_actions = self.q_network(next_states).argmax(1).unsqueeze(1)
            next_q_values = self.target_network(next_states).gather(1, next_actions)
            target_q_values = rewards.unsqueeze(1) + (
                self.gamma * next_q_values * (~dones).unsqueeze(1)
            )
            td_errors = torch.abs(target_q_values - current_q_values).squeeze()

        return td_errors

    def _on_step_episode_mode(
        self,
        current_state: np.ndarray,
        action: Action,
        reward: float,
        done: bool,
        info: dict[str, Any],
    ) -> None:
        """Handle step processing for episode-end training mode."""
        # Store experience in episode buffer
        if self.last_state is not None and self.last_action is not None:
            experience = {
                "state": self.last_state.copy(),
                "action": self.last_action,
                "reward": reward,
                "next_state": current_state.copy(),
                "done": done,
            }
            self.episode_buffer.append(experience)
            self.episode_rewards.append(reward)

        # Train at episode end
        if done and len(self.episode_buffer) > 0:
            self._train_on_episode()
            self._reset_episode_buffer()

    def _on_step_traditional(
        self,
        current_state: np.ndarray,
        action: Action,
        reward: float,
        done: bool,
        info: dict[str, Any],
    ) -> None:
        """Handle step processing for traditional step-by-step training mode."""
        if self.last_state is not None and self.last_action is not None:
            # Store experience immediately
            self.remember(self.last_state, self.last_action, reward, current_state, done)
            self.step_count += 1

            # Train according to frequency
            if (
                len(self.memory) >= self.min_replay_size
                and self.step_count % self.train_frequency == 0
            ):
                loss = self.replay()
                if loss is not None:
                    self.update_learning_rate(reward)

    def _observation_to_state(self, observation: dict[str, Any]) -> np.ndarray:
        """
        Converts a game observation to a state vector for the network.
        Extended version including bonuses, paddle sizes, score, etc.

        Args:
            observation: Formatted game observation

        Returns:
            Normalized state vector of variable dimension depending on active bonuses
        """
        # Base state (8 dimensions)
        base_state = [
            observation["ball_pos"][0],  # Ball X position
            observation["ball_pos"][1],  # Ball Y position
            observation["ball_vel"][0],  # Ball X velocity
            observation["ball_vel"][1],  # Ball Y velocity
            observation["player_pos"][0],  # Player X position
            observation["player_pos"][1],  # Player Y position
            observation["opponent_pos"][0],  # Opponent X position
            observation["opponent_pos"][1],  # Opponent Y position
            observation["opponent_previous_pos"][0],  # Opponent previous X position
            observation["opponent_previous_pos"][1],  # Opponent previous Y position
            observation["field_width"],  # Field width
            observation["field_height"],  # Field height
        ]

        # Additional important information (5 dimensions)
        extra_state = [
            self._get_paddle_size(
                observation.get("player_paddle_size", 1.0)
            ),  # Player paddle height
            self._get_paddle_size(
                observation.get("opponent_paddle_size", 1.0)
            ),  # Opponent paddle height
            observation.get("score_diff", 0.0) / 10.0,  # Score difference (normalized)
            observation.get("time_elapsed", 0.0) / 300.0,  # Time elapsed (normalized to 5 min)
            len(observation.get("bonuses", [])) / 5.0,  # Number of active bonuses (normalized)
        ]

        # Information about active bonuses (up to 3 bonuses x 3 info = 9 dimensions max)
        bonus_state = []
        bonuses = observation.get("bonuses", [])
        max_bonuses = 3  # Limit number of bonuses considered

        for i in range(max_bonuses):
            if i < len(bonuses):
                bonus = bonuses[i]
                if len(bonus) >= 3:  # [x, y, type]
                    bonus_state.extend(
                        [
                            bonus[0],  # Bonus X position (already normalized)
                            bonus[1],  # Bonus Y position (already normalized)
                            bonus[2] / 3.0,  # Bonus type (normalized: 1,2,3 -> 0.33,0.67,1.0)
                        ]
                    )
                else:
                    bonus_state.extend([0.0, 0.0, 0.0])  # Empty bonus
            else:
                bonus_state.extend([0.0, 0.0, 0.0])  # No bonus

        # Information about rotating paddles (up to 2 paddles x 3 info = 6 dimensions max)
        rotating_paddle_state = []
        rotating_paddles = observation.get("rotating_paddles", [])
        max_rotating_paddles = 2

        for i in range(max_rotating_paddles):
            if i < len(rotating_paddles):
                rp = rotating_paddles[i]
                if len(rp) >= 3:  # [x, y, angle]
                    # Physics stores radians and lets the value grow unwrapped.
                    angle_rad = ((float(rp[2]) + np.pi) % (2 * np.pi)) - np.pi
                    normalized_angle = (angle_rad + np.pi) / (
                        2 * np.pi
                    )  # Normalize [-π, π] -> [0, 1]
                    rotating_paddle_state.extend(
                        [
                            rp[0],  # X position (already normalized)
                            rp[1],  # Y position (already normalized)
                            normalized_angle,  # Normalized angle ([-π, π] -> [0, 1])
                        ]
                    )
                else:
                    rotating_paddle_state.extend([0.0, 0.0, 0.0])
            else:
                rotating_paddle_state.extend([0.0, 0.0, 0.0])

        # Combine all states
        # Total: 12 (base) + 5 (extra) + 9 (bonus) + 6 (rotating) = 32 dimensions
        full_state = base_state + extra_state + bonus_state + rotating_paddle_state

        return np.array(full_state, dtype=np.float32)

    def _action_to_index(self, action: Action) -> int:
        """
        Converts an Action to a numeric index

        Args:
            action: Game action

        Returns:
            Action index (0-8)
        """
        # Find the corresponding action in the mapping
        for idx, mapped_action in ACTION_MAPPING.items():
            if (
                abs(mapped_action.move_x - action.move_x) < 0.1
                and abs(mapped_action.move_y - action.move_y) < 0.1
            ):
                return idx

        # Default action if not found
        return 0

    def _get_paddle_size(self, paddle_size_data: float | list | tuple) -> float:
        """
        Extracts paddle size from different possible formats

        Args:
            paddle_size_data: Can be float, list, or tuple

        Returns:
            Normalized paddle size (0.0 to 1.0)
        """
        if isinstance(paddle_size_data, list | tuple) and len(paddle_size_data) >= 2:
            size = paddle_size_data[1]  # Height (index 1)
        elif isinstance(paddle_size_data, int | float):
            size = float(paddle_size_data)
        else:
            size = 1.0  # Default value

        normalized_size = float(np.clip(size / game_config.PADDLE_HEIGHT, 0.0, 4.0))
        return normalized_size

    def _clear_pending_transition(self) -> None:
        """Clear the decision-time transition source after it has been consumed."""
        self.last_state = None
        self.last_action = None

    def on_episode_start(self) -> None:
        """Clear stale decision data when a new episode starts."""
        super().on_episode_start()
        self._clear_pending_transition()

    def on_episode_end(self) -> None:
        """Clear stale decision data when an episode ends."""
        self._clear_pending_transition()
        super().on_episode_end()

    def update_target_network(self) -> None:
        """Copy weights from main network to target network"""
        self.target_network.load_state_dict(self.q_network.state_dict())

    def set_training_mode(self, training: bool) -> None:
        """Enable or disable training behavior and module train/eval mode."""
        self.training_enabled = training
        self.exploration_enabled = training
        self.q_network.train(training)
        self.target_network.train(training)
        self._clear_pending_transition()

    def set_exploration_mode(self, explore: bool) -> None:
        """Enable or disable epsilon-greedy policy exploration independently."""
        self.exploration_enabled = explore

    def soft_update_target_network(self) -> None:
        """Soft update of target network"""
        for target_param, main_param in zip(
            self.target_network.parameters(), self.q_network.parameters(), strict=False
        ):
            target_param.data.copy_(
                self.tau * main_param.data + (1.0 - self.tau) * target_param.data
            )

    def _calculate_discounted_rewards(self, rewards: list[float]) -> list[float]:
        """
        Calculate discounted cumulative rewards for episode-end training.

        This method implements proper credit assignment by calculating the discounted
        cumulative reward for each action in the episode, helping the agent understand
        which actions contributed to the final outcome.

        Args:
            rewards: List of immediate rewards for each step in the episode

        Returns:
            List of discounted cumulative rewards for each step
        """
        if not rewards:
            return []

        discounted_rewards = []
        cumulative_reward = 0.0

        # Calculate discounted rewards in reverse order (from end to beginning)
        for reward in reversed(rewards):
            cumulative_reward = reward + self.gamma * cumulative_reward
            discounted_rewards.append(cumulative_reward)

        # Reverse to get chronological order
        discounted_rewards.reverse()

        # Normalize rewards if enabled (improves training stability)
        if self.reward_normalization and len(discounted_rewards) > 1:
            mean_reward = np.mean(discounted_rewards)
            std_reward = np.std(discounted_rewards)
            if std_reward > 1e-8:  # Avoid division by zero
                discounted_rewards = [
                    float((r - mean_reward) / std_reward) for r in discounted_rewards
                ]

        return discounted_rewards

    def set_training_mode_hybrid(self, mode: str) -> None:
        """
        Switch between training modes for different learning scenarios.

        Args:
            mode: "episode_end" for better credit assignment on sequential actions,
                  "step_by_step" for immediate feedback on individual actions
        """
        if mode not in ["episode_end", "step_by_step"]:
            raise ValueError(
                f"Invalid training mode: {mode}. Must be 'episode_end' or 'step_by_step'"
            )

        old_mode = self.training_mode
        self.training_mode = mode

        # Clear episode buffer when switching to step-by-step mode
        if mode == "step_by_step" and old_mode == "episode_end":
            self.episode_buffer.clear()
            self.episode_rewards.clear()

        print(f"🔄 Training mode switched: {old_mode} → {mode}")
        if mode == "episode_end":
            print(
                "   💡 Better credit assignment for sequential actions (paddle positioning, ball returns)"
            )
        else:
            print("   ⚡ Immediate training for quick iteration and debugging")

    def _train_on_episode(self) -> None:
        """
        Train the network on the complete episode using discounted rewards.

        This method provides better credit assignment by calculating cumulative
        discounted rewards for each action in the episode, helping the agent
        understand which early actions contributed to later successes.
        """
        if len(self.episode_buffer) == 0:
            return

        # Calculate discounted rewards for proper credit assignment
        discounted_rewards = self._calculate_discounted_rewards(self.episode_rewards)

        # Add all experiences to replay memory with discounted rewards
        for i, experience in enumerate(self.episode_buffer):
            # Use discounted reward instead of immediate reward
            self.memory.add(
                experience["state"],
                experience["action"],
                discounted_rewards[i],
                experience["next_state"],
                experience["done"],
            )

        # Perform multiple training iterations on the episode
        # Train more aggressively since we have better reward signals
        episode_length = len(self.episode_buffer)
        num_training_iterations = max(1, episode_length // self.batch_size)

        total_loss = 0.0
        successful_iterations = 0

        for _ in range(num_training_iterations):
            if len(self.memory) >= self.batch_size:
                loss = self.replay()
                if loss is not None:
                    total_loss += loss
                    successful_iterations += 1

        # Update learning rate based on episode performance
        if successful_iterations > 0:
            avg_episode_reward = float(np.mean(self.episode_rewards))
            self.update_learning_rate(avg_episode_reward)

            # Log episode training stats
            avg_loss = total_loss / successful_iterations
            self.loss_history.append(avg_loss)

            # Store episode statistics
            self.reward_history.append(avg_episode_reward)

    def _reset_episode_buffer(self) -> None:
        """Reset episode buffer for the next episode."""
        self.episode_buffer.clear()
        self.episode_rewards.clear()

    def remember(
        self, state: np.ndarray, action: int, reward: float, next_state: np.ndarray, done: bool
    ) -> None:
        """Store an experience in the replay memory"""
        self.memory.add(state, action, reward, next_state, done)

    def act(
        self,
        state: np.ndarray,
        explore: bool | None = None,
        *,
        training: bool | None = None,
    ) -> int:
        """Choose an action, optionally using epsilon-greedy exploration."""
        if training is not None:
            if explore is not None and explore != training:
                raise ValueError("act() received conflicting explore and training flags")
            explore = training

        if explore is None:
            explore = self.exploration_enabled

        if explore and random.random() < self.epsilon:
            return random.randrange(self.action_size)

        # Convert to PyTorch tensor
        state_tensor = torch.FloatTensor(state).unsqueeze(0).to(device)

        # Prediction with the network
        was_training = self.q_network.training
        self.q_network.eval()
        with torch.no_grad():
            q_values = self.q_network(state_tensor)
        self.q_network.train(was_training)

        return int(q_values.cpu().data.numpy().argmax())

    def replay(self) -> float | None:
        """Train the network with experience replay"""
        if not self.training_enabled:
            return None

        # Quick check of buffer size
        if len(self.memory) < self.batch_size:
            return None

        # Sampling
        indices: np.ndarray | None = None
        experiences: list[Transition]
        if self.use_prioritized_replay:
            sampled = cast(
                tuple[list[Transition], np.ndarray, np.ndarray],
                self.memory.sample(self.batch_size),
            )
            experiences, weights_arr, indices = sampled
            weights = torch.FloatTensor(weights_arr).to(device)
        else:
            experiences = cast(list[Transition], self.memory.sample(self.batch_size))
            weights = torch.ones(len(experiences)).to(device)

        # Data extraction (optimized to avoid PyTorch warnings)
        states = torch.FloatTensor(np.array([e.state for e in experiences])).to(device)
        actions = torch.LongTensor([e.action for e in experiences]).to(device)
        rewards = torch.FloatTensor([e.reward for e in experiences]).to(device)
        next_states = torch.FloatTensor(np.array([e.next_state for e in experiences])).to(device)
        dones = torch.BoolTensor([e.done for e in experiences]).to(device)

        with _temporary_eval_mode(self.q_network, self.target_network):
            # Current Q-values keep gradients, but dropout stays inactive.
            current_q_values = self.q_network(states).gather(1, actions.unsqueeze(1))

            # Double DQN: action selection with main network, evaluation with target network
            with torch.no_grad():
                next_actions = self.q_network(next_states).argmax(1).unsqueeze(1)
                next_q_values = self.target_network(next_states).gather(1, next_actions)
                target_q_values = rewards.unsqueeze(1) + (
                    self.gamma * next_q_values * (~dones).unsqueeze(1)
                )

        # Calculate loss with importance weighting
        td_errors = target_q_values - current_q_values
        loss = (weights.unsqueeze(1) * td_errors.pow(2)).mean()

        # Optimization with gradient clipping
        self.optimizer.zero_grad()
        loss.backward()
        torch.nn.utils.clip_grad_norm_(self.q_network.parameters(), max_norm=1.0)

        self.optimizer.step()

        # Update priorities if prioritized replay
        if self.use_prioritized_replay and indices is not None:
            td_errors_np = td_errors.detach().cpu().numpy().flatten()
            prb = cast("PrioritizedReplayBuffer", self.memory)
            prb.update_priorities(indices, td_errors_np)

        # Soft update of target network
        self.soft_update_target_network()

        # Epsilon decay with adaptive scheduling
        if self.epsilon > self.epsilon_min:
            # Slower decay at the start of training
            decay_factor = self.epsilon_decay
            self.epsilon = max(self.epsilon_min, self.epsilon * decay_factor)

        self.training_step += 1
        self.loss_history.append(loss.item())

        return float(loss.item())

    def save_model(self, filepath: str) -> None:
        """Save the model with hybrid training parameters"""
        torch.save(
            {
                "q_network_state_dict": self.q_network.state_dict(),
                "target_network_state_dict": self.target_network.state_dict(),
                "optimizer_state_dict": self.optimizer.state_dict(),
                "epsilon": self.epsilon,
                "training_step": self.training_step,
                "step_count": self.step_count,
                "loss_history": self.loss_history,
                "reward_history": self.reward_history,
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
                    "tau": self.tau,
                    "train_frequency": self.train_frequency,
                    "min_replay_size": self.min_replay_size,
                    "training_mode": self.training_mode,
                    "reward_normalization": self.reward_normalization,
                    "use_prioritized_replay": self.use_prioritized_replay,
                },
            },
            filepath,
        )

    def load_model(self, filepath: str) -> None:
        """Load the model with hybrid training parameters"""
        checkpoint = safe_torch_load(filepath, map_location=device)
        validate_dqn_checkpoint(checkpoint)

        self.q_network.load_state_dict(checkpoint["q_network_state_dict"])
        self.target_network.load_state_dict(checkpoint["target_network_state_dict"])
        self.optimizer.load_state_dict(checkpoint["optimizer_state_dict"])
        self.epsilon = checkpoint["epsilon"]
        self.training_step = checkpoint["training_step"]
        self.step_count = checkpoint.get("step_count", 0)  # Compatibility with old models
        self.loss_history = checkpoint["loss_history"]
        self.reward_history = checkpoint["reward_history"]

        # Load hyperparameters if available
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
            self.train_frequency = hyperparams.get("train_frequency", 10)  # Default value
            self.min_replay_size = hyperparams.get("min_replay_size", 1000)  # Default value
            self.training_mode = hyperparams.get(
                "training_mode", "episode_end"
            )  # Default to episode_end
            self.reward_normalization = hyperparams.get(
                "reward_normalization", True
            )  # Default enabled
            self.use_prioritized_replay = hyperparams["use_prioritized_replay"]
            self.dropout_rate = hyperparams.get("dropout_rate", self.dropout_rate)
            self.q_network.set_dropout_rate(self.dropout_rate)
            self.target_network.set_dropout_rate(self.dropout_rate)

        # Reset episode buffer when loading (don't carry over partial episodes)
        self._reset_episode_buffer()
        self._clear_pending_transition()

    def update_learning_rate(self, reward: float) -> None:
        """Update learning rate based on performance"""
        self.scheduler.step(reward)

    def get_exploration_rate(self) -> float:
        """Return the current exploration rate"""
        return self.epsilon

    def get_training_stats(self) -> dict:
        """Return comprehensive training statistics including hybrid mode info"""
        stats = {
            "training_step": self.training_step,
            "step_count": self.step_count,
            "train_frequency": self.train_frequency,
            "min_replay_size": self.min_replay_size,
            "memory_size": len(self.memory),
            "training_mode": self.training_mode,
            "reward_normalization": self.reward_normalization,
            "episode_buffer_size": len(self.episode_buffer),
            "epsilon": self.epsilon,
            "avg_loss": float(np.mean(self.loss_history[-100:])) if self.loss_history else 0.0,
            "avg_reward": float(np.mean(self.reward_history[-100:]))
            if self.reward_history
            else 0.0,
            "current_lr": self.optimizer.param_groups[0]["lr"],
        }

        # Add dual-scale training statistics
        if self.enable_dual_scale_training:
            stats.update(
                {
                    "dual_scale_training": True,
                    "tactical_step_count": self.tactical_step_count,
                    "tactical_train_frequency": self.tactical_train_frequency,
                    "tactical_lr": self.tactical_learning_rate,
                    "strategic_lr": self.strategic_learning_rate,
                    "tactical_optimizer_lr": self.tactical_optimizer.param_groups[0]["lr"],
                    "strategic_optimizer_lr": self.strategic_optimizer.param_groups[0]["lr"],
                }
            )
        else:
            stats["dual_scale_training"] = False

        return stats


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
