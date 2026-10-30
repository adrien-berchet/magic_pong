"""Deep Q-Network (DQN) AI implementation for Magic Pong using PyTorch.

Vanilla DQN with Double DQN action selection (van Hasselt et al., 2015).
"""

import copy
import random
from collections import deque
from collections import namedtuple
from os import PathLike
from typing import Any
from typing import BinaryIO
from typing import cast

import numpy as np
import torch
import torch.nn as nn
import torch.nn.functional as F
import torch.optim as optim

from magic_pong.ai.action_shaping import GeometricShieldConfig
from magic_pong.ai.action_shaping import shape_action
from magic_pong.ai.geometry import GEOMETRY_CONTEXT_FEATURES
from magic_pong.ai.geometry import GEOMETRY_CONTEXT_HORIZON
from magic_pong.ai.geometry import get_state_layout
from magic_pong.ai.geometry import write_contact_features_scalar
from magic_pong.ai.interface import AIPlayer
from magic_pong.ai.models.dqn_checkpoint import GEOMETRY_CONTEXT_STATE_SIZE
from magic_pong.ai.models.dqn_checkpoint import SCORE_CONTEXT_FEATURES
from magic_pong.ai.models.dqn_checkpoint import SCORE_CONTEXT_NORMALIZER
from magic_pong.ai.models.dqn_checkpoint import SCORE_CONTEXT_STATE_SIZE
from magic_pong.ai.models.dqn_checkpoint import build_dqn_checkpoint_metadata
from magic_pong.ai.models.dqn_checkpoint import safe_torch_load
from magic_pong.ai.models.dqn_checkpoint import validate_dqn_checkpoint
from magic_pong.ai.policy_anchor import PolicyAnchor
from magic_pong.core.entities import Action
from magic_pong.utils.config import game_config

# Transition tuple for replay buffer
Transition = namedtuple("Transition", ("state", "action", "reward", "next_state", "done"))

device = torch.device("cuda" if torch.cuda.is_available() else "cpu")


def resolve_layer_widths(
    hidden_size: int, layer_size: int, hidden_layers: list[int] | None = None
) -> list[int]:
    """Return the per-layer neuron counts for a DQN network.

    If ``hidden_layers`` is given, it is used verbatim (fully manual
    architecture). Otherwise widths are derived from ``hidden_size``
    (first layer) by halving geometrically for ``layer_size`` layers.
    """
    if hidden_layers:
        return list(hidden_layers)
    return [hidden_size // (2**i) for i in range(layer_size)]


class DQNNetwork(nn.Module):
    """Deep Q-Network: stack of Linear→ReLU layers with optional dueling architecture."""

    score_context_weight: nn.Parameter | None
    geometry_context_weight: nn.Parameter | None
    advantage_layer: nn.Linear | None
    value_layer: nn.Linear | None
    output_layer: nn.Linear | None
    log_tau: nn.Parameter | None

    def __init__(
        self,
        input_size: int,
        hidden_size: int = 256,
        output_size: int = 9,
        layer_size: int = 2,
        hidden_layers: list[int] | None = None,
        score_context: bool = False,
        geometry_context: bool = False,
        dueling: bool = False,
    ):
        super().__init__()
        layer_sizes = resolve_layer_widths(hidden_size, layer_size, hidden_layers)
        layout = get_state_layout(score_context=score_context, geometry_context=geometry_context)
        if (score_context or geometry_context) and input_size != layout.state_size:
            raise ValueError(
                f"Networks with score_context={score_context}, geometry_context={geometry_context} "
                f"require {layout.state_size} input values"
            )
        legacy_input_size = 30 if (score_context or geometry_context) else input_size

        self.score_context = score_context
        self.geometry_context = geometry_context
        self.layout = layout
        self.dueling = dueling
        self.fc_layers = nn.ModuleList()
        for i, width in enumerate(layer_sizes):
            in_features = legacy_input_size if i == 0 else layer_sizes[i - 1]
            self.fc_layers.append(nn.Linear(in_features, width))

        self.advantage_layer = None
        self.value_layer = None
        self.output_layer = None
        self.log_tau = None

        if dueling:
            self.advantage_layer = nn.Linear(layer_sizes[-1], output_size)
            self.value_layer = nn.Linear(layer_sizes[-1], 1)
            self.log_tau = nn.Parameter(torch.tensor(0.0))
        else:
            self.output_layer = nn.Linear(layer_sizes[-1], output_size)

        self._initialize_weights()
        # Allocate after legacy initialization; adding context must consume no RNG.
        self.register_parameter("score_context_weight", None)
        self.register_parameter("geometry_context_weight", None)
        if score_context:
            self.enable_score_context()
        if geometry_context:
            self.enable_geometry_context()

    def enable_score_context(self) -> None:
        """Add a zero projection without changing the existing layers or RNG."""
        if self.score_context_weight is None:
            first = cast(nn.Linear, self.fc_layers[0])
            if first.in_features != 30:
                raise ValueError("Score context can only extend a 30-input network")
            self.score_context_weight = nn.Parameter(
                first.weight.new_zeros((first.out_features, len(SCORE_CONTEXT_FEATURES)))
            )
            self.score_context = True
            self.layout = get_state_layout(
                score_context=True, geometry_context=self.geometry_context
            )

    def enable_geometry_context(self) -> None:
        """Add a zero projection without changing the existing layers or RNG."""
        if self.geometry_context_weight is None:
            first = cast(nn.Linear, self.fc_layers[0])
            if first.in_features != 30:
                raise ValueError("Geometry context can only extend a 30-input network")
            self.geometry_context_weight = nn.Parameter(
                first.weight.new_zeros((first.out_features, len(GEOMETRY_CONTEXT_FEATURES)))
            )
            self.geometry_context = True
            self.layout = get_state_layout(score_context=self.score_context, geometry_context=True)

    def enable_dueling(
        self,
        tau: float = 0.03,
        warm_start: bool = True,
        q_semantics: str = "logits",
    ) -> None:
        """Convert this network to Dueling DQN with advantage and value streams."""
        if self.dueling:
            return
        assert self.output_layer is not None
        last_layer = cast(nn.Linear, self.fc_layers[-1])
        last_width = last_layer.out_features
        output_size = self.output_layer.out_features
        device = self.output_layer.weight.device

        self.advantage_layer = nn.Linear(last_width, output_size).to(device)
        self.value_layer = nn.Linear(last_width, 1).to(device)

        if warm_start:
            if q_semantics == "logits":
                self.advantage_layer.weight.data.copy_(self.output_layer.weight.data)
                self.advantage_layer.bias.data.copy_(self.output_layer.bias.data)
                nn.init.zeros_(self.value_layer.weight)
                nn.init.zeros_(self.value_layer.bias)
                self.log_tau = nn.Parameter(
                    torch.tensor(float(np.log(tau)), device=device, dtype=torch.float32)
                )
            else:  # returns
                self.advantage_layer.weight.data.copy_(self.output_layer.weight.data)
                self.advantage_layer.bias.data.copy_(self.output_layer.bias.data)
                self.value_layer.weight.data.copy_(
                    self.output_layer.weight.data.mean(dim=0, keepdim=True)
                )
                self.value_layer.bias.data.copy_(
                    self.output_layer.bias.data.mean(dim=0, keepdim=True)
                )
                self.log_tau = nn.Parameter(torch.tensor(0.0, device=device, dtype=torch.float32))
        else:
            nn.init.kaiming_normal_(self.advantage_layer.weight, mode="fan_in", nonlinearity="relu")
            nn.init.constant_(self.advantage_layer.bias, 0.01)
            nn.init.kaiming_normal_(self.value_layer.weight, mode="fan_in", nonlinearity="relu")
            nn.init.constant_(self.value_layer.bias, 0.01)
            self.log_tau = nn.Parameter(torch.tensor(0.0, device=device, dtype=torch.float32))

        self.output_layer = None
        self.dueling = True

    def fold_to_standard(self) -> "DQNNetwork":
        """Fold dueling advantage, value and tau into an exact equivalent standard output layer."""
        if not self.dueling:
            return copy.deepcopy(self)
        assert self.advantage_layer is not None and self.value_layer is not None
        tau = float(torch.exp(self.log_tau).item()) if self.log_tau is not None else 1.0
        w_adv = tau * self.advantage_layer.weight.data
        b_adv = tau * self.advantage_layer.bias.data
        w_val = self.value_layer.weight.data
        b_val = self.value_layer.bias.data

        w_folded = w_adv - w_adv.mean(dim=0, keepdim=True) + w_val
        b_folded = b_adv - b_adv.mean() + b_val.squeeze()

        device = self.advantage_layer.weight.device
        first_layer = cast(nn.Linear, self.fc_layers[0]) if len(self.fc_layers) > 0 else None
        in_size = (
            self.layout.state_size
            if (self.score_context or self.geometry_context)
            else (
                first_layer.in_features
                if first_layer is not None
                else self.advantage_layer.in_features
            )
        )
        folded_net = DQNNetwork(
            input_size=in_size,
            hidden_layers=[cast(nn.Linear, fc).out_features for fc in self.fc_layers],
            output_size=self.advantage_layer.out_features,
            score_context=self.score_context,
            geometry_context=self.geometry_context,
            dueling=False,
        ).to(device)

        for i, fc_module in enumerate(self.fc_layers):
            fc = cast(nn.Linear, fc_module)
            target_fc = cast(nn.Linear, folded_net.fc_layers[i])
            target_fc.weight.data.copy_(fc.weight.data)
            target_fc.bias.data.copy_(fc.bias.data)
        if self.score_context_weight is not None and folded_net.score_context_weight is not None:
            folded_net.score_context_weight.data.copy_(self.score_context_weight.data)
        if (
            self.geometry_context_weight is not None
            and folded_net.geometry_context_weight is not None
        ):
            folded_net.geometry_context_weight.data.copy_(self.geometry_context_weight.data)

        assert folded_net.output_layer is not None
        folded_net.output_layer.weight.data.copy_(w_folded)
        folded_net.output_layer.bias.data.copy_(b_folded)
        return folded_net

    def _initialize_weights(self) -> None:
        for m in self.modules():
            if isinstance(m, nn.Linear):
                nn.init.kaiming_normal_(m.weight, mode="fan_in", nonlinearity="relu")
                nn.init.constant_(m.bias, 0.01)

    def trunk(self, x: torch.Tensor) -> torch.Tensor:
        """Extract hidden features from the shared MLP trunk."""
        if self.score_context_weight is not None or self.geometry_context_weight is not None:
            h = self.fc_layers[0](x[..., :30].contiguous())
            if self.score_context_weight is not None and self.layout.score_slice is not None:
                h = h + F.linear(x[..., self.layout.score_slice], self.score_context_weight)
            if self.geometry_context_weight is not None and self.layout.geometry_slice is not None:
                h = h + F.linear(x[..., self.layout.geometry_slice], self.geometry_context_weight)
            h = F.relu(h)
            for i in range(1, len(self.fc_layers)):
                h = F.relu(self.fc_layers[i](h))
        else:
            h = x
            for fc in self.fc_layers:
                h = F.relu(fc(h))
        return h

    def raw_advantage_logits(self, x: torch.Tensor) -> torch.Tensor:
        """Compute unscaled advantage/action logits (without value shift or tau multiplier)."""
        h = self.trunk(x)
        if self.dueling:
            assert self.advantage_layer is not None
            return self.advantage_layer(h)
        assert self.output_layer is not None
        return self.output_layer(h)

    def forward(self, x: torch.Tensor) -> torch.Tensor:
        h = self.trunk(x)
        if self.dueling:
            assert self.advantage_layer is not None and self.value_layer is not None
            tau = torch.exp(self.log_tau) if self.log_tau is not None else 1.0
            adv = tau * self.advantage_layer(h)
            val = self.value_layer(h)
            return val + (adv - adv.mean(dim=-1, keepdim=True))

        assert self.output_layer is not None
        return self.output_layer(h)


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
        hidden_size: int = 256,
        layer_size: int = 2,
        hidden_layers: list[int] | None = None,
        name: str = "DQN AI",
        terminal_on_goal: bool = False,
        score_context: bool = False,
        geometry_context: bool = False,
        dueling: bool = False,
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
            hidden_size: Width of the first hidden layer (ignored if hidden_layers is set)
            layer_size: Number of hidden layers (ignored if hidden_layers is set)
            hidden_layers: Explicit per-layer neuron counts, e.g. [256, 128, 64].
                Overrides hidden_size/layer_size with a fully manual architecture.
            name: Agent display name
            terminal_on_goal: Stop the learning return at every goal without ending the match.
            score_context: Append own score, opponent score and match target (33 inputs).
            geometry_context: Append predicted intercept y, time to intercept, own paddle vel y, opponent paddle vel y (34 or 37 inputs).
            dueling: Enable Dueling DQN architecture (advantage + value streams).
        """
        super().__init__(name)

        layout = get_state_layout(score_context=score_context, geometry_context=geometry_context)
        if (score_context or geometry_context) and state_size not in (30, layout.state_size):
            raise ValueError(f"Context extensions require 30 or {layout.state_size} input values")
        self.score_context = score_context
        self.geometry_context = geometry_context
        self.layout = layout
        self.state_size = layout.state_size if (score_context or geometry_context) else state_size
        state_size = self.state_size
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
        self.terminal_on_goal = terminal_on_goal
        self.dueling = dueling
        self.dueling_trained = False
        self.q_semantics = "logits"
        # Training-only auxiliary state is owned and checkpointed by the frame runner.
        self.policy_anchor: PolicyAnchor | None = None

        # Decision-time state/action waiting for the environment step result
        self.last_state: np.ndarray | None = None
        self.last_action: int | None = None
        self.training_enabled = True
        self.exploration_enabled = True
        # Kept for pretraining compatibility (pretraining.py reads/writes this)
        self._last_chosen_action: int | None = None

        self.step_count = 0
        self.hidden_layers = list(hidden_layers) if hidden_layers else None
        layer_widths = resolve_layer_widths(hidden_size, layer_size, self.hidden_layers)
        # Kept for display/summary purposes (dashboard, checkpoint metadata) even when
        # hidden_layers overrides the actual per-layer widths.
        self.hidden_size = layer_widths[0]
        self.layer_size = len(layer_widths)

        self.q_network = DQNNetwork(
            state_size,
            output_size=action_size,
            hidden_layers=layer_widths,
            score_context=score_context,
            geometry_context=geometry_context,
            dueling=dueling,
        ).to(device)
        self.target_network = DQNNetwork(
            state_size,
            output_size=action_size,
            hidden_layers=layer_widths,
            score_context=score_context,
            geometry_context=geometry_context,
            dueling=dueling,
        ).to(device)

        self.optimizer = optim.Adam(self.q_network.parameters(), lr=lr)

        self.memory = ReplayBuffer(memory_size)

        self.update_target_network()

        self.training_step = 0
        self.loss_history: list[float] = []
        self.reward_history: list[float] = []
        self.grad_norm_history: list[float] = []
        self.action_shaper_config: GeometricShieldConfig | None = None

    def enable_score_context(self) -> None:
        """Upgrade a weights-only warm start; empty replay and a fresh optimizer are required."""
        if self.score_context:
            return
        allowed_sizes = (30, GEOMETRY_CONTEXT_STATE_SIZE)
        if self.state_size not in allowed_sizes or len(self.memory):
            raise ValueError(
                f"Score-context migration requires a 30- or {GEOMETRY_CONTEXT_STATE_SIZE}-input agent with empty replay"
            )
        self.q_network.enable_score_context()
        self.target_network.enable_score_context()
        self.score_context = True
        self.layout = get_state_layout(score_context=True, geometry_context=self.geometry_context)
        self.state_size = self.layout.state_size
        self.optimizer = optim.Adam(self.q_network.parameters(), lr=self.lr)
        self._clear_pending_transition()

    def enable_geometry_context(self) -> None:
        """Upgrade a weights-only warm start; empty replay and a fresh optimizer are required."""
        if self.geometry_context:
            return
        allowed_sizes = (30, SCORE_CONTEXT_STATE_SIZE)
        if self.state_size not in allowed_sizes or len(self.memory):
            raise ValueError(
                f"Geometry-context migration requires a 30- or {SCORE_CONTEXT_STATE_SIZE}-input agent with empty replay"
            )
        self.q_network.enable_geometry_context()
        self.target_network.enable_geometry_context()
        self.geometry_context = True
        self.layout = get_state_layout(score_context=self.score_context, geometry_context=True)
        self.state_size = self.layout.state_size
        self.optimizer = optim.Adam(self.q_network.parameters(), lr=self.lr)
        self._clear_pending_transition()

    def enable_dueling(
        self,
        tau: float = 0.03,
        warm_start: bool = True,
        q_semantics: str = "logits",
    ) -> None:
        """Convert this agent to Dueling DQN with advantage and value streams."""
        self.q_network.enable_dueling(tau=tau, warm_start=warm_start, q_semantics=q_semantics)
        self.target_network.enable_dueling(tau=tau, warm_start=warm_start, q_semantics=q_semantics)
        self.target_network.load_state_dict(self.q_network.state_dict())
        self.dueling = True
        self.dueling_trained = True
        self.q_semantics = "returns"
        self.optimizer = optim.Adam(self.q_network.parameters(), lr=self.lr)

    def freeze_for_phase1(self) -> None:
        """Phase 1: freeze trunk and advantage; train only value head and log_tau."""
        if not self.q_network.dueling:
            raise ValueError("Must be dueling to freeze for Phase 1")
        for name, param in self.q_network.named_parameters():
            if "value_layer" in name or name == "log_tau":
                param.requires_grad = True
            else:
                param.requires_grad = False
        phase1_params = [p for p in self.q_network.parameters() if p.requires_grad]
        self.optimizer = optim.Adam(phase1_params, lr=self.lr)

    def unfreeze_for_phase2(self, lr: float | None = None) -> None:
        """Phase 2: unfreeze all parameters for end-to-end fine-tuning."""
        for param in self.q_network.parameters():
            param.requires_grad = True
        if lr is not None:
            self.lr = lr
        self.optimizer = optim.Adam(self.q_network.parameters(), lr=self.lr)

    def configure_action_shaper(self, config: GeometricShieldConfig | None) -> None:
        """Configure inference-time action shaping (geometric barrier and soft-x).

        Args:
            config: GeometricShieldConfig instance, or None to disable shaping.

        Raises:
            ValueError: If shaping is enabled without geometry context, or if
                training or exploration mode is active.
        """
        if config is not None:
            if not self.geometry_context:
                raise ValueError("Action shaping requires geometry_context=True")
            if self.training_enabled or self.exploration_enabled:
                raise ValueError(
                    "Action shaping cannot be configured while training or exploration is enabled"
                )
        self.action_shaper_config = config

    def get_action(self, observation: dict[str, Any] | None, explore: bool | None = None) -> Action:
        """Convert observation to action via the Q-network (or epsilon-greedy)."""
        if observation is None:
            self._clear_pending_transition()
            self._last_chosen_action = None
            return Action(move_x=0.0, move_y=0.0)

        if self.action_shaper_config is not None:
            if self.training_enabled or self.exploration_enabled or (explore is True):
                raise ValueError("Action shaping cannot be used while training or exploring")
            state = self._observation_to_state(observation)
            state_tensor = torch.as_tensor(state).unsqueeze(0).to(device)
            was_training = self.q_network.training
            self.q_network.eval()
            with torch.no_grad():
                q_values = self.q_network(state_tensor).cpu().numpy().ravel()
            self.q_network.train(was_training)
            action_idx = int(q_values.argmax())
            self._last_chosen_action = action_idx
            return shape_action(q_values, state, self.layout, self.action_shaper_config)

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
        if self.action_shaper_config is not None and self.training_enabled:
            raise ValueError("Action shaping cannot be used while training")
        current_state = self._observation_to_state(observation)
        transition_done = bool(done)
        if "terminated" in info or "truncated" in info:
            transition_done = bool(
                info.get("terminated", done and not info.get("truncated", False))
            )

        if self.terminal_on_goal and info.get("events", {}).get("goals"):
            transition_done = True

        if self.training_enabled and self.last_state is not None and self.last_action is not None:
            self.remember(
                self.last_state,
                self.last_action,
                reward,
                current_state,
                transition_done,
            )
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
        Convert an observation to 30 legacy values, optionally followed by score and geometry context.

        Dimensions: 10 base + 5 extra + 9 bonus (3×3) + 6 rotating paddle (2×3) = 30
        """
        if self.geometry_context:
            required_keys = (
                "ball_pos",
                "ball_vel",
                "player_pos",
                "opponent_pos",
            )
            for k in required_keys:
                if k not in observation or observation[k] is None:
                    raise ValueError(f"Missing required geometry context key in observation: {k}")

        # Legacy unmarked observations are raw. Never infer units from a mutable
        # global flag, nor normalize observations twice when the producer did it.
        normalized = observation.get("positions_normalized", False)
        _fw = 1.0 if normalized else float(observation.get("field_width", game_config.FIELD_WIDTH))
        _fh = (
            1.0 if normalized else float(observation.get("field_height", game_config.FIELD_HEIGHT))
        )
        _ms = (
            1.0
            if normalized
            else float(observation.get("max_ball_speed", game_config.MAX_BALL_SPEED))
        )

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
        if self.score_context:
            try:
                context = np.asarray(
                    [observation[k] for k in SCORE_CONTEXT_FEATURES], dtype=np.float32
                )
            except (KeyError, TypeError, ValueError) as exc:
                raise ValueError("Missing or invalid score context in observation") from exc
            if (
                context.shape != (3,)
                or not np.isfinite(context).all()
                or (context[:2] < 0).any()
                or context[2] <= 0
            ):
                raise ValueError(
                    "Invalid score context: finite nonnegative scores and positive target required"
                )
            full_state.extend(context / SCORE_CONTEXT_NORMALIZER)

        if self.geometry_context:
            required_keys = (
                "ball_pos",
                "ball_vel",
                "player_pos",
                "opponent_pos",
            )
            for k in required_keys:
                if k not in observation or observation[k] is None:
                    raise ValueError(f"Missing required geometry context key in observation: {k}")

            geom_check = [
                observation["ball_pos"][0],
                observation["ball_pos"][1],
                observation["ball_vel"][0],
                observation["ball_vel"][1],
                observation["player_pos"][0],
                observation["player_pos"][1],
                observation["opponent_pos"][0],
                observation["opponent_pos"][1],
            ]
            if not all(np.isfinite(x) for x in geom_check):
                raise ValueError("Non-finite values in observation geometry fields")

            state_30 = np.asarray(full_state[:30], dtype=np.float32)
            if not np.isfinite(state_30).all():
                raise ValueError(
                    "Non-finite values in canonical 30D state for geometry calculation"
                )

            fw = float(observation.get("field_width", game_config.FIELD_WIDTH))
            fh = float(observation.get("field_height", game_config.FIELD_HEIGHT))
            ms = float(observation.get("max_ball_speed", game_config.MAX_BALL_SPEED))
            pw = float(observation.get("paddle_width", game_config.PADDLE_WIDTH))
            ph = float(game_config.PADDLE_HEIGHT)
            br = float(observation.get("ball_radius", game_config.BALL_RADIUS))
            ps = float(observation.get("paddle_speed", game_config.PADDLE_SPEED))

            out_geom = np.empty(16, dtype=np.float32)
            write_contact_features_scalar(
                state_30, out_geom, 0, fw, fh, ms, pw, ph, ps, br, GEOMETRY_CONTEXT_HORIZON
            )
            full_state.extend(out_geom)

        return np.array(full_state, dtype=np.float32)

    def _get_paddle_height_pixels(self, paddle_size_data: Any) -> float:
        if isinstance(paddle_size_data, list | tuple) and len(paddle_size_data) >= 2:
            return float(paddle_size_data[1])
        elif isinstance(paddle_size_data, (int, float)):
            size = float(paddle_size_data)
            if size > 4.0:
                return size
            return size * game_config.PADDLE_HEIGHT
        return float(game_config.PADDLE_HEIGHT)

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
        if training and self.action_shaper_config is not None:
            raise ValueError("Cannot enable training mode while action shaping is configured")
        self.training_enabled = training
        self.exploration_enabled = training
        self.q_network.train(training)
        self.target_network.eval()
        self._clear_pending_transition()

    def set_exploration_mode(self, explore: bool) -> None:
        """Enable or disable epsilon-greedy exploration independently."""
        if explore and self.action_shaper_config is not None:
            raise ValueError("Cannot enable exploration mode while action shaping is configured")
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
            last.state,
            last.action,
            last.reward + extra_reward,
            last.next_state,
            last.done,
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

        td_loss = self._compute_dqn_loss(experiences, weights)
        loss = td_loss
        if self.policy_anchor is not None and self.policy_anchor.active:
            loss = loss + self.policy_anchor.loss(self.q_network, self.batch_size, td_loss)

        self.optimizer.zero_grad()
        loss.backward()
        grad_norm = torch.nn.utils.clip_grad_norm_(self.q_network.parameters(), max_norm=1.0)
        self.grad_norm_history.append(float(grad_norm))
        self.optimizer.step()

        if self.epsilon > self.epsilon_min:
            self.epsilon = max(self.epsilon_min, self.epsilon * self.epsilon_decay)

        self.training_step += 1
        self.loss_history.append(td_loss.item())

        if self.training_step % self.target_update_frequency == 0:
            self.update_target_network()

        return float(loss.item())

    def save_model(self, filepath: str | PathLike[str] | BinaryIO) -> None:
        """Save checkpoint to disk."""
        if self.q_network.dueling:
            folded_q = self.q_network.fold_to_standard()
            folded_target = self.target_network.fold_to_standard()
            dummy_opt = optim.Adam(folded_q.parameters(), lr=self.lr)
            q_dict = folded_q.state_dict()
            target_dict = folded_target.state_dict()
            opt_dict = dummy_opt.state_dict()
            dueling_trained = True
            q_semantics = "returns"
        else:
            q_dict = self.q_network.state_dict()
            target_dict = self.target_network.state_dict()
            opt_dict = self.optimizer.state_dict()
            dueling_trained = getattr(self, "dueling_trained", False)
            q_semantics = getattr(self, "q_semantics", "logits")

        metadata = build_dqn_checkpoint_metadata(
            state_size=self.state_size,
            action_size=self.action_size,
            score_context=self.score_context,
            geometry_context=self.geometry_context,
        )
        if dueling_trained:
            metadata["dueling_trained"] = True
        if q_semantics:
            metadata["q_semantics"] = q_semantics

        torch.save(
            {
                "q_network_state_dict": q_dict,
                "target_network_state_dict": target_dict,
                "optimizer_state_dict": opt_dict,
                "epsilon": self.epsilon,
                "training_step": self.training_step,
                "step_count": self.step_count,
                "loss_history": self.loss_history[-1000:],
                "reward_history": self.reward_history[-1000:],
                "grad_norm_history": self.grad_norm_history[-1000:],
                "metadata": metadata,
                "hyperparameters": {
                    "state_size": self.state_size,
                    "action_size": self.action_size,
                    "lr": self.lr,
                    "gamma": self.gamma,
                    "terminal_on_goal": self.terminal_on_goal,
                    "score_context": self.score_context,
                    "geometry_context": self.geometry_context,
                    "epsilon_min": self.epsilon_min,
                    "epsilon_decay": self.epsilon_decay,
                    "batch_size": self.batch_size,
                    "hidden_size": self.hidden_size,
                    "layer_size": self.layer_size,
                    "hidden_layers": self.hidden_layers,
                    "dueling": False if self.q_network.dueling else self.dueling,
                },
            },
            filepath,
        )

    def load_model(self, filepath: str | PathLike[str] | BinaryIO) -> None:
        """Load checkpoint from disk."""
        checkpoint = safe_torch_load(filepath, map_location=device)
        validate_dqn_checkpoint(checkpoint)
        self.policy_anchor = None
        self.action_shaper_config = None
        self.terminal_on_goal = checkpoint.get("hyperparameters", {}).get("terminal_on_goal", False)

        if "hyperparameters" in checkpoint:
            hyperparams = checkpoint["hyperparameters"]
            self.state_size = hyperparams["state_size"]
            self.score_context = hyperparams.get("score_context", False)
            self.geometry_context = hyperparams.get("geometry_context", False)
            self.layout = get_state_layout(
                score_context=self.score_context,
                geometry_context=self.geometry_context,
            )
            self.action_size = hyperparams["action_size"]
            self.lr = hyperparams["lr"]
            self.gamma = hyperparams["gamma"]
            self.epsilon_min = hyperparams["epsilon_min"]
            self.epsilon_decay = hyperparams["epsilon_decay"]
            self.batch_size = hyperparams["batch_size"]
            self.hidden_layers = hyperparams.get("hidden_layers")
            # Rebuild the networks to match the checkpoint's own architecture:
            # hidden_size/layer_size/hidden_layers only describe the network
            # actually built at __init__ time, so a checkpoint saved with a
            # different architecture than this instance's constructor defaults
            # would otherwise fail to load_state_dict below on a shape mismatch.
            layer_widths = resolve_layer_widths(
                hyperparams.get("hidden_size", 256),
                hyperparams.get("layer_size", 2),
                self.hidden_layers,
            )
            self.hidden_size = layer_widths[0]
            self.layer_size = len(layer_widths)
            self.dueling = hyperparams.get("dueling", False)
            self.dueling_trained = checkpoint.get("metadata", {}).get("dueling_trained", False)
            self.q_semantics = checkpoint.get("metadata", {}).get("q_semantics", "logits")
            self.q_network = DQNNetwork(
                self.state_size,
                output_size=self.action_size,
                hidden_layers=layer_widths,
                score_context=self.score_context,
                geometry_context=self.geometry_context,
                dueling=self.dueling,
            ).to(device)
            self.target_network = DQNNetwork(
                self.state_size,
                output_size=self.action_size,
                hidden_layers=layer_widths,
                score_context=self.score_context,
                geometry_context=self.geometry_context,
                dueling=self.dueling,
            ).to(device)
            self.optimizer = optim.Adam(self.q_network.parameters(), lr=self.lr)

        self.q_network.load_state_dict(checkpoint["q_network_state_dict"])
        self.target_network.load_state_dict(checkpoint["target_network_state_dict"])
        self.optimizer.load_state_dict(checkpoint["optimizer_state_dict"])
        self.epsilon = checkpoint["epsilon"]
        self.training_step = checkpoint["training_step"]
        self.step_count = checkpoint.get("step_count", 0)
        self.loss_history = checkpoint["loss_history"]
        self.reward_history = checkpoint["reward_history"]
        self.grad_norm_history = checkpoint.get("grad_norm_history", [])

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

    @property
    def param_count(self) -> int:
        """Total number of trainable parameters in the Q-network."""
        return sum(p.numel() for p in self.q_network.parameters())

    def benchmark_inference(self, n: int = 500) -> float:
        """Return mean inference time in microseconds over n forward passes."""
        import time

        obs = np.zeros(self.state_size, dtype=np.float32)
        for _ in range(10):
            self.act(obs)
        t0 = time.perf_counter()
        for _ in range(n):
            self.act(obs)
        return (time.perf_counter() - t0) / n * 1e6


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
