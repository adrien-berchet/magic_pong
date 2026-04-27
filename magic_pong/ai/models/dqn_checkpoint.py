"""Shared DQN checkpoint contract and validation helpers."""

from __future__ import annotations

from collections.abc import Mapping
from dataclasses import dataclass
from typing import Any

CHECKPOINT_SCHEMA_VERSION = 1
EXPECTED_MODEL_TYPE = "magic_pong.dqn"
EXPECTED_STATE_SCHEMA = "magic_pong.dqn_state.v1"
EXPECTED_OBSERVATION_FRAME = "player1_dqn_frame"
EXPECTED_ACTION_FRAME = "player1_dqn_frame"
EXPECTED_STATE_SIZE = 32
EXPECTED_ACTION_SIZE = 9

METADATA_KEY = "metadata"

REQUIRED_CHECKPOINT_KEYS = (
    "q_network_state_dict",
    "target_network_state_dict",
    "optimizer_state_dict",
    "epsilon",
    "training_step",
    "loss_history",
    "reward_history",
    "hyperparameters",
)

REQUIRED_HYPERPARAMETER_KEYS = (
    "state_size",
    "action_size",
    "lr",
    "gamma",
    "epsilon_min",
    "epsilon_decay",
    "batch_size",
    "tau",
    "use_prioritized_replay",
)

REQUIRED_METADATA_KEYS = (
    "schema_version",
    "model_type",
    "state_schema",
    "observation_frame",
    "action_frame",
    "state_size",
    "action_size",
)

EXPECTED_DQN_STATE_DICT_SHAPES = {
    "fc_layers.0.weight": (512, EXPECTED_STATE_SIZE),
    "fc_layers.0.bias": (512,),
    "fc_layers.1.weight": (256, 512),
    "fc_layers.1.bias": (256,),
    "fc_layers.2.weight": (128, 256),
    "fc_layers.2.bias": (128,),
    "layer_norms.0.weight": (512,),
    "layer_norms.0.bias": (512,),
    "layer_norms.1.weight": (256,),
    "layer_norms.1.bias": (256,),
    "layer_norms.2.weight": (128,),
    "layer_norms.2.bias": (128,),
    "output_layer.weight": (EXPECTED_ACTION_SIZE, 128),
    "output_layer.bias": (EXPECTED_ACTION_SIZE,),
}
EXPECTED_DQN_PARAMETER_COUNT = len(EXPECTED_DQN_STATE_DICT_SHAPES)


class DQNCheckpointError(ValueError):
    """Raised when a DQN checkpoint does not match the supported contract."""


@dataclass(frozen=True)
class DQNCheckpointValidation:
    """Structured validation result for UI and loading code."""

    valid: bool
    error: str | None
    warnings: tuple[str, ...]
    metadata: Mapping[str, Any] | None
    hyperparameters: Mapping[str, Any] | None
    training_step: Any
    epsilon: Any

    def as_dict(self) -> dict[str, Any]:
        """Return a plain dict for existing GUI data plumbing."""
        return {
            "valid": self.valid,
            "error": self.error,
            "warnings": list(self.warnings),
            "metadata": dict(self.metadata) if self.metadata is not None else None,
            "hyperparameters": (
                dict(self.hyperparameters) if self.hyperparameters is not None else None
            ),
            "training_step": self.training_step,
            "epsilon": self.epsilon,
        }


def build_dqn_checkpoint_metadata(
    *, state_size: int = EXPECTED_STATE_SIZE, action_size: int = EXPECTED_ACTION_SIZE
) -> dict[str, Any]:
    """Build metadata identifying the persisted DQN contract."""
    return {
        "schema_version": CHECKPOINT_SCHEMA_VERSION,
        "model_type": EXPECTED_MODEL_TYPE,
        "state_schema": EXPECTED_STATE_SCHEMA,
        "observation_frame": EXPECTED_OBSERVATION_FRAME,
        "action_frame": EXPECTED_ACTION_FRAME,
        "state_size": state_size,
        "action_size": action_size,
    }


def safe_torch_load(filepath: str, *, map_location: Any = "cpu") -> Any:
    """
    Load a torch checkpoint with safe weights-only loading when available.

    PyTorch versions before the ``weights_only`` argument are supported by
    falling back only when that argument is rejected by the loader signature.
    """
    import torch

    try:
        return torch.load(filepath, map_location=map_location, weights_only=True)
    except TypeError as exc:
        if "weights_only" not in str(exc):
            raise
        return torch.load(filepath, map_location=map_location)


def validate_dqn_checkpoint(checkpoint: Any) -> DQNCheckpointValidation:
    """Validate a checkpoint and raise ``DQNCheckpointError`` on incompatibility."""
    if not isinstance(checkpoint, Mapping):
        raise DQNCheckpointError("Invalid DQN checkpoint: expected a mapping payload")

    missing_keys = [key for key in REQUIRED_CHECKPOINT_KEYS if key not in checkpoint]
    if missing_keys:
        raise DQNCheckpointError(f"Missing required checkpoint data: {', '.join(missing_keys)}")

    hyperparameters = _validate_hyperparameters(checkpoint["hyperparameters"])
    metadata = validate_dqn_checkpoint_metadata(checkpoint.get(METADATA_KEY), hyperparameters)
    _validate_network_state_dict("q_network_state_dict", checkpoint["q_network_state_dict"])
    _validate_network_state_dict(
        "target_network_state_dict", checkpoint["target_network_state_dict"]
    )
    _validate_optimizer_state_dict(checkpoint["optimizer_state_dict"])

    return DQNCheckpointValidation(
        valid=True,
        error=None,
        warnings=(),
        metadata=metadata,
        hyperparameters=hyperparameters,
        training_step=checkpoint.get("training_step"),
        epsilon=checkpoint.get("epsilon"),
    )


def get_dqn_checkpoint_validation_info(checkpoint: Any) -> DQNCheckpointValidation:
    """Return structured validation info without raising contract errors."""
    checkpoint_mapping = checkpoint if isinstance(checkpoint, Mapping) else {}
    try:
        return validate_dqn_checkpoint(checkpoint)
    except DQNCheckpointError as exc:
        metadata = checkpoint_mapping.get(METADATA_KEY)
        hyperparameters = checkpoint_mapping.get("hyperparameters")
        return DQNCheckpointValidation(
            valid=False,
            error=str(exc),
            warnings=(),
            metadata=metadata if isinstance(metadata, Mapping) else None,
            hyperparameters=hyperparameters if isinstance(hyperparameters, Mapping) else None,
            training_step=checkpoint_mapping.get("training_step"),
            epsilon=checkpoint_mapping.get("epsilon"),
        )


def validate_dqn_checkpoint_metadata(
    metadata: Any, hyperparameters: Mapping[str, Any] | None = None
) -> Mapping[str, Any]:
    """Validate metadata against the current DQN checkpoint contract."""
    if metadata is None:
        raise DQNCheckpointError(
            "Legacy DQN checkpoint without metadata is not supported by default. "
            "Re-save the model with current Magic Pong training code to add checkpoint metadata."
        )
    if not isinstance(metadata, Mapping):
        raise DQNCheckpointError("Invalid DQN checkpoint metadata: expected a mapping")

    missing_keys = [key for key in REQUIRED_METADATA_KEYS if key not in metadata]
    if missing_keys:
        raise DQNCheckpointError(f"Missing DQN checkpoint metadata: {', '.join(missing_keys)}")

    _validate_expected_value(
        "schema_version", metadata["schema_version"], CHECKPOINT_SCHEMA_VERSION
    )
    _validate_expected_value("model_type", metadata["model_type"], EXPECTED_MODEL_TYPE)
    _validate_expected_value("state_schema", metadata["state_schema"], EXPECTED_STATE_SCHEMA)
    _validate_expected_value(
        "observation_frame", metadata["observation_frame"], EXPECTED_OBSERVATION_FRAME
    )
    _validate_expected_value("action_frame", metadata["action_frame"], EXPECTED_ACTION_FRAME)
    _validate_expected_value("state_size", metadata["state_size"], EXPECTED_STATE_SIZE)
    _validate_expected_value("action_size", metadata["action_size"], EXPECTED_ACTION_SIZE)

    if hyperparameters is not None:
        _validate_expected_value(
            "hyperparameters.state_size",
            hyperparameters.get("state_size"),
            EXPECTED_STATE_SIZE,
        )
        _validate_expected_value(
            "hyperparameters.action_size",
            hyperparameters.get("action_size"),
            EXPECTED_ACTION_SIZE,
        )

    return metadata


def _validate_hyperparameters(hyperparameters: Any) -> Mapping[str, Any]:
    if not isinstance(hyperparameters, Mapping):
        raise DQNCheckpointError("Invalid DQN checkpoint hyperparameters: expected a mapping")

    missing_hyperparameters = [
        key for key in REQUIRED_HYPERPARAMETER_KEYS if key not in hyperparameters
    ]
    if missing_hyperparameters:
        raise DQNCheckpointError(f"Missing hyperparameters: {', '.join(missing_hyperparameters)}")

    _validate_expected_value(
        "hyperparameters.state_size", hyperparameters["state_size"], EXPECTED_STATE_SIZE
    )
    _validate_expected_value(
        "hyperparameters.action_size", hyperparameters["action_size"], EXPECTED_ACTION_SIZE
    )

    return hyperparameters


def _validate_network_state_dict(name: str, state_dict: Any) -> None:
    if not isinstance(state_dict, Mapping):
        raise DQNCheckpointError(f"Invalid DQN checkpoint {name}: expected a mapping")
    if not state_dict:
        raise DQNCheckpointError(f"Invalid DQN checkpoint {name}: state dict is empty")

    missing_keys = [key for key in EXPECTED_DQN_STATE_DICT_SHAPES if key not in state_dict]
    if missing_keys:
        raise DQNCheckpointError(
            f"Invalid DQN checkpoint {name}: missing network weights {', '.join(missing_keys)}"
        )

    unexpected_keys = [key for key in state_dict if key not in EXPECTED_DQN_STATE_DICT_SHAPES]
    if unexpected_keys:
        raise DQNCheckpointError(
            f"Invalid DQN checkpoint {name}: unexpected network weights "
            f"{', '.join(unexpected_keys)}"
        )

    wrong_shapes = []
    for key, expected_shape in EXPECTED_DQN_STATE_DICT_SHAPES.items():
        actual_shape = _tensor_shape(state_dict[key])
        if actual_shape != expected_shape:
            wrong_shapes.append(f"{key} expected {expected_shape}, got {actual_shape}")

    if wrong_shapes:
        raise DQNCheckpointError(
            f"Invalid DQN checkpoint {name}: incompatible network tensor shapes "
            f"{'; '.join(wrong_shapes)}"
        )


def _tensor_shape(value: Any) -> tuple[int, ...] | None:
    shape = getattr(value, "shape", None)
    if shape is None:
        return None
    try:
        return tuple(int(dimension) for dimension in shape)
    except TypeError:
        return None


def _validate_optimizer_state_dict(optimizer_state_dict: Any) -> None:
    if not isinstance(optimizer_state_dict, Mapping):
        raise DQNCheckpointError("Invalid DQN checkpoint optimizer_state_dict: expected a mapping")

    missing_keys = [key for key in ("state", "param_groups") if key not in optimizer_state_dict]
    if missing_keys:
        raise DQNCheckpointError(
            f"Invalid DQN checkpoint optimizer_state_dict: missing {', '.join(missing_keys)}"
        )

    if not isinstance(optimizer_state_dict["state"], Mapping):
        raise DQNCheckpointError(
            "Invalid DQN checkpoint optimizer_state_dict.state: expected a mapping"
        )
    if not isinstance(optimizer_state_dict["param_groups"], list):
        raise DQNCheckpointError(
            "Invalid DQN checkpoint optimizer_state_dict.param_groups: expected a list"
        )

    param_groups = optimizer_state_dict["param_groups"]
    if len(param_groups) != 1:
        raise DQNCheckpointError(
            "Invalid DQN checkpoint optimizer_state_dict.param_groups: expected one "
            f"parameter group, got {len(param_groups)}"
        )

    param_group = param_groups[0]
    if not isinstance(param_group, Mapping):
        raise DQNCheckpointError(
            "Invalid DQN checkpoint optimizer_state_dict.param_groups[0]: expected a mapping"
        )

    params = param_group.get("params")
    if not isinstance(params, list):
        raise DQNCheckpointError(
            "Invalid DQN checkpoint optimizer_state_dict.param_groups[0].params: expected a list"
        )
    if len(params) != EXPECTED_DQN_PARAMETER_COUNT:
        raise DQNCheckpointError(
            "Invalid DQN checkpoint optimizer_state_dict.param_groups[0].params: "
            f"expected {EXPECTED_DQN_PARAMETER_COUNT} parameters, got {len(params)}"
        )


def _validate_expected_value(name: str, actual: Any, expected: Any) -> None:
    if actual != expected:
        raise DQNCheckpointError(
            f"Unsupported DQN checkpoint {name}: expected {expected!r}, got {actual!r}"
        )
