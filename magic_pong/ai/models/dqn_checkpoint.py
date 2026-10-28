"""Shared DQN checkpoint contract and validation helpers."""

from __future__ import annotations

from collections.abc import Mapping
from dataclasses import dataclass
from os import PathLike
from typing import Any
from typing import BinaryIO

from magic_pong.ai.geometry import get_state_layout

CHECKPOINT_SCHEMA_VERSION = 4
EXPECTED_MODEL_TYPE = "magic_pong.dqn"
EXPECTED_STATE_SCHEMA = "magic_pong.dqn_state.v4"
EXPECTED_OBSERVATION_FRAME = "player1_dqn_frame"
EXPECTED_ACTION_FRAME = "player1_dqn_frame"
EXPECTED_STATE_SIZE = 30
EXPECTED_ACTION_SIZE = 9
SCORE_CONTEXT_STATE_SIZE = 33
SCORE_CONTEXT_SCHEMA_VERSION = 5
SCORE_CONTEXT_STATE_SCHEMA = "magic_pong.dqn_state.v5_score_context"
SCORE_CONTEXT_FEATURES = ("own_score", "opponent_score", "match_target")
SCORE_CONTEXT_NORMALIZER = 11.0

GEOMETRY_CONTEXT_SCHEMA_VERSION = 6
GEOMETRY_CONTEXT_STATE_SCHEMA = "magic_pong.dqn_state.v6"
GEOMETRY_CONTEXT_STATE_SIZE = 34
FULL_CONTEXT_STATE_SIZE = 37
GEOMETRY_CONTEXT_FEATURES = (
    "predicted_intercept_y",
    "time_to_intercept",
    "own_paddle_vel_y",
    "opponent_paddle_vel_y",
)
GEOMETRY_CONTEXT_HORIZON = 1.5
GEOMETRY_CONTEXT_VERSION = 1
# Network first-layer input equals the observation size (vanilla DQN, no history)
EXPECTED_NETWORK_INPUT_SIZE = EXPECTED_STATE_SIZE

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
    "fc_layers.0.weight": (256, EXPECTED_NETWORK_INPUT_SIZE),
    "fc_layers.0.bias": (256,),
    "fc_layers.1.weight": (128, 256),
    "fc_layers.1.bias": (128,),
    "output_layer.weight": (EXPECTED_ACTION_SIZE, 128),
    "output_layer.bias": (EXPECTED_ACTION_SIZE,),
}
EXPECTED_DQN_PARAMETER_COUNT = len(EXPECTED_DQN_STATE_DICT_SHAPES)


def _compute_state_dict_shapes(
    hidden_size: int,
    layer_size: int,
    state_size: int = EXPECTED_STATE_SIZE,
    action_size: int = EXPECTED_ACTION_SIZE,
    hidden_layers: list[int] | None = None,
    score_context: bool = False,
    geometry_context: bool = False,
) -> dict[str, tuple[int, ...]]:
    """Return the expected state-dict key→shape mapping for a given architecture.

    If ``hidden_layers`` is given, it is used verbatim as the per-layer widths
    (manual architecture). Otherwise widths are derived from hidden_size/layer_size
    by halving geometrically.
    """
    layer_sizes = (
        list(hidden_layers) if hidden_layers else [hidden_size // (2**i) for i in range(layer_size)]
    )
    shapes: dict[str, tuple[int, ...]] = {}
    for i, width in enumerate(layer_sizes):
        in_features = (
            (EXPECTED_STATE_SIZE if (score_context or geometry_context) else state_size)
            if i == 0
            else layer_sizes[i - 1]
        )
        shapes[f"fc_layers.{i}.weight"] = (width, in_features)
        shapes[f"fc_layers.{i}.bias"] = (width,)
    shapes["output_layer.weight"] = (action_size, layer_sizes[-1])
    shapes["output_layer.bias"] = (action_size,)
    if score_context:
        shapes["score_context_weight"] = (layer_sizes[0], len(SCORE_CONTEXT_FEATURES))
    if geometry_context:
        shapes["geometry_context_weight"] = (layer_sizes[0], len(GEOMETRY_CONTEXT_FEATURES))
    return shapes


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
    *,
    state_size: int = EXPECTED_STATE_SIZE,
    action_size: int = EXPECTED_ACTION_SIZE,
    score_context: bool = False,
    geometry_context: bool = False,
) -> dict[str, Any]:
    """Build metadata identifying the persisted DQN contract."""
    from magic_pong.utils.config import game_config

    layout = get_state_layout(score_context=score_context, geometry_context=geometry_context)
    if geometry_context:
        schema_version = GEOMETRY_CONTEXT_SCHEMA_VERSION
        state_schema = GEOMETRY_CONTEXT_STATE_SCHEMA
    elif score_context:
        schema_version = SCORE_CONTEXT_SCHEMA_VERSION
        state_schema = SCORE_CONTEXT_STATE_SCHEMA
    else:
        schema_version = CHECKPOINT_SCHEMA_VERSION
        state_schema = EXPECTED_STATE_SCHEMA

    metadata: dict[str, Any] = {
        "schema_version": schema_version,
        "model_type": EXPECTED_MODEL_TYPE,
        "state_schema": state_schema,
        "observation_frame": EXPECTED_OBSERVATION_FRAME,
        "action_frame": EXPECTED_ACTION_FRAME,
        "state_size": state_size,
        "action_size": action_size,
    }
    if score_context or geometry_context:
        metadata["legacy_state_size"] = EXPECTED_STATE_SIZE
        metadata["input_blocks"] = list(layout.input_blocks)

    if score_context:
        metadata.update(
            score_context_size=len(SCORE_CONTEXT_FEATURES),
            score_context_enabled=True,
            score_context_features=list(SCORE_CONTEXT_FEATURES),
            score_context_normalizer=SCORE_CONTEXT_NORMALIZER,
        )
    if geometry_context:
        metadata.update(
            geometry_context_size=len(GEOMETRY_CONTEXT_FEATURES),
            geometry_context_enabled=True,
            geometry_context_features=list(GEOMETRY_CONTEXT_FEATURES),
            geometry_context_horizon=GEOMETRY_CONTEXT_HORIZON,
            geometry_context_velocity_normalizer=float(game_config.PADDLE_SPEED),
            geometry_context_version=GEOMETRY_CONTEXT_VERSION,
        )
    return metadata


def safe_torch_load(filepath: str | PathLike[str] | BinaryIO, *, map_location: Any = "cpu") -> Any:
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
    score_context = hyperparameters.get("score_context", False)
    geometry_context = hyperparameters.get("geometry_context", False)
    hidden_size: int = int(hyperparameters.get("hidden_size", 256))
    layer_size: int = int(hyperparameters.get("layer_size", 2))
    raw_hidden_layers = hyperparameters.get("hidden_layers")
    hidden_layers: list[int] | None = (
        [int(w) for w in raw_hidden_layers] if raw_hidden_layers else None
    )
    _validate_network_state_dict(
        "q_network_state_dict",
        checkpoint["q_network_state_dict"],
        hidden_size=hidden_size,
        layer_size=layer_size,
        has_explicit_layer_size="layer_size" in hyperparameters,
        hidden_layers=hidden_layers,
        score_context=score_context,
        geometry_context=geometry_context,
    )
    _validate_network_state_dict(
        "target_network_state_dict",
        checkpoint["target_network_state_dict"],
        hidden_size=hidden_size,
        layer_size=layer_size,
        has_explicit_layer_size="layer_size" in hyperparameters,
        hidden_layers=hidden_layers,
        score_context=score_context,
        geometry_context=geometry_context,
    )
    expected_shapes = _compute_state_dict_shapes(
        hidden_size,
        layer_size,
        hidden_layers=hidden_layers,
        score_context=score_context,
        geometry_context=geometry_context,
    )
    _validate_optimizer_state_dict(
        checkpoint["optimizer_state_dict"], expected_param_count=len(expected_shapes)
    )

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
            hyperparameters=(hyperparameters if isinstance(hyperparameters, Mapping) else None),
            training_step=checkpoint_mapping.get("training_step"),
            epsilon=checkpoint_mapping.get("epsilon"),
        )


def validate_dqn_checkpoint_metadata(
    metadata: Any, hyperparameters: Mapping[str, Any] | None = None
) -> Mapping[str, Any]:
    """Validate metadata against the current DQN checkpoint contract."""
    from magic_pong.utils.config import game_config

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

    schema_version = metadata["schema_version"]
    if schema_version not in (
        CHECKPOINT_SCHEMA_VERSION,
        SCORE_CONTEXT_SCHEMA_VERSION,
        GEOMETRY_CONTEXT_SCHEMA_VERSION,
    ):
        raise DQNCheckpointError(f"Unsupported DQN checkpoint schema_version: {schema_version}")

    geometry_context = schema_version == GEOMETRY_CONTEXT_SCHEMA_VERSION
    if geometry_context:
        _validate_expected_value(
            "state_schema", metadata["state_schema"], GEOMETRY_CONTEXT_STATE_SCHEMA
        )
        input_blocks = metadata.get("input_blocks")
        if not isinstance(input_blocks, (list, tuple)):
            raise DQNCheckpointError(
                "Invalid DQN checkpoint metadata: expected input_blocks list for v6"
            )
        allowed_sequences = [
            ["base", "geometry_context"],
            ["base", "score_context", "geometry_context"],
        ]
        if list(input_blocks) not in allowed_sequences:
            raise DQNCheckpointError(
                f"Invalid DQN checkpoint input_blocks: got {input_blocks}, expected one of {allowed_sequences}"
            )
        score_context = "score_context" in input_blocks
        expected_layout = get_state_layout(
            score_context=score_context, geometry_context=geometry_context
        )
        expected_state_size = expected_layout.state_size
    elif schema_version == SCORE_CONTEXT_SCHEMA_VERSION:
        score_context = True
        _validate_expected_value(
            "state_schema", metadata["state_schema"], SCORE_CONTEXT_STATE_SCHEMA
        )
        expected_state_size = SCORE_CONTEXT_STATE_SIZE
    else:
        score_context = False
        _validate_expected_value("state_schema", metadata["state_schema"], EXPECTED_STATE_SCHEMA)
        expected_state_size = EXPECTED_STATE_SIZE

    _validate_expected_value("model_type", metadata["model_type"], EXPECTED_MODEL_TYPE)
    _validate_expected_value(
        "observation_frame", metadata["observation_frame"], EXPECTED_OBSERVATION_FRAME
    )
    _validate_expected_value("action_frame", metadata["action_frame"], EXPECTED_ACTION_FRAME)
    _validate_expected_value("state_size", metadata["state_size"], expected_state_size)
    _validate_expected_value("action_size", metadata["action_size"], EXPECTED_ACTION_SIZE)

    if score_context:
        for key, expected in {
            "legacy_state_size": EXPECTED_STATE_SIZE,
            "score_context_size": len(SCORE_CONTEXT_FEATURES),
            "score_context_enabled": True,
            "score_context_features": list(SCORE_CONTEXT_FEATURES),
            "score_context_normalizer": SCORE_CONTEXT_NORMALIZER,
        }.items():
            _validate_expected_value(key, metadata.get(key), expected)

    if geometry_context:
        for key, expected in {
            "legacy_state_size": EXPECTED_STATE_SIZE,
            "geometry_context_size": len(GEOMETRY_CONTEXT_FEATURES),
            "geometry_context_enabled": True,
            "geometry_context_features": list(GEOMETRY_CONTEXT_FEATURES),
            "geometry_context_horizon": GEOMETRY_CONTEXT_HORIZON,
            "geometry_context_velocity_normalizer": float(game_config.PADDLE_SPEED),
            "geometry_context_version": GEOMETRY_CONTEXT_VERSION,
        }.items():
            _validate_expected_value(key, metadata.get(key), expected)

    if hyperparameters is not None:
        _validate_expected_value(
            "hyperparameters.score_context",
            hyperparameters.get("score_context", False),
            score_context,
        )
        _validate_expected_value(
            "hyperparameters.geometry_context",
            hyperparameters.get("geometry_context", False),
            geometry_context,
        )
        _validate_expected_value(
            "hyperparameters.state_size",
            hyperparameters.get("state_size"),
            expected_state_size,
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

    score_context = hyperparameters.get("score_context", False)
    if not isinstance(score_context, bool):
        raise DQNCheckpointError("hyperparameters.score_context must be boolean")
    geometry_context = hyperparameters.get("geometry_context", False)
    if not isinstance(geometry_context, bool):
        raise DQNCheckpointError("hyperparameters.geometry_context must be boolean")

    expected_state_size = get_state_layout(
        score_context=score_context, geometry_context=geometry_context
    ).state_size
    _validate_expected_value(
        "hyperparameters.state_size",
        hyperparameters["state_size"],
        expected_state_size,
    )
    _validate_expected_value(
        "hyperparameters.action_size",
        hyperparameters["action_size"],
        EXPECTED_ACTION_SIZE,
    )

    return hyperparameters


def _validate_network_state_dict(
    name: str,
    state_dict: Any,
    hidden_size: int = 256,
    layer_size: int = 2,
    has_explicit_layer_size: bool = False,
    hidden_layers: list[int] | None = None,
    score_context: bool = False,
    geometry_context: bool = False,
) -> None:
    if not isinstance(state_dict, Mapping):
        raise DQNCheckpointError(f"Invalid DQN checkpoint {name}: expected a mapping")
    if not state_dict:
        raise DQNCheckpointError(f"Invalid DQN checkpoint {name}: state dict is empty")

    # Schema v3 checkpoints contain LayerNorm weights; schema v4 removed them.
    if "layer_norms.0.weight" in state_dict:
        raise DQNCheckpointError(
            f"Incompatible DQN checkpoint {name}: contains LayerNorm weights "
            "(layer_norms.*) from schema v3. The architecture is now vanilla DQN "
            "(no normalization). Use warm_start_from_v1_checkpoint to transfer weights."
        )

    # These legacy-format heuristics only apply to the halving hidden_size/layer_size
    # scheme; an explicit hidden_layers declaration is trusted as-is.
    if hidden_layers is None:
        # Old schema v1/v2 checkpoints have fc_layers.2.* but no explicit layer_size in
        # hyperparameters. New checkpoints with layer_size >= 3 declare it explicitly.
        if "fc_layers.2.weight" in state_dict and not has_explicit_layer_size:
            raise DQNCheckpointError(
                f"Incompatible DQN checkpoint {name}: contains a 3-layer network "
                "(fc_layers.2.*) from schema v1/v2 without an explicit layer_size declaration. "
                "Inputs were also unnormalized then. Retrain from scratch."
            )

        # Old hidden_size=512 checkpoints have no explicit hidden_size declaration.
        first_layer = state_dict.get("fc_layers.0.weight")
        if first_layer is not None:
            shape = _tensor_shape(first_layer)
            if shape is not None and shape[0] == 512 and hidden_size != 512:
                raise DQNCheckpointError(
                    f"Incompatible DQN checkpoint {name}: state dict has hidden_size=512 "
                    f"but declared hidden_size={hidden_size}. Retrain from scratch."
                )

    expected_shapes = _compute_state_dict_shapes(
        hidden_size,
        layer_size,
        hidden_layers=hidden_layers,
        score_context=score_context,
        geometry_context=geometry_context,
    )

    missing_keys = [key for key in expected_shapes if key not in state_dict]
    if missing_keys:
        raise DQNCheckpointError(
            f"Invalid DQN checkpoint {name}: missing network weights {', '.join(missing_keys)}"
        )

    unexpected_keys = [key for key in state_dict if key not in expected_shapes]
    if unexpected_keys:
        raise DQNCheckpointError(
            f"Invalid DQN checkpoint {name}: unexpected network weights "
            f"{', '.join(unexpected_keys)}"
        )

    wrong_shapes = []
    for key, expected_shape in expected_shapes.items():
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


def _validate_optimizer_state_dict(
    optimizer_state_dict: Any, expected_param_count: int = EXPECTED_DQN_PARAMETER_COUNT
) -> None:
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
    if len(params) != expected_param_count:
        raise DQNCheckpointError(
            "Invalid DQN checkpoint optimizer_state_dict.param_groups[0].params: "
            f"expected {expected_param_count} parameters, got {len(params)}"
        )


def _validate_expected_value(name: str, actual: Any, expected: Any) -> None:
    if actual != expected:
        raise DQNCheckpointError(
            f"Unsupported DQN checkpoint {name}: expected {expected!r}, got {actual!r}"
        )
