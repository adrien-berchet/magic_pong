"""Model conversion utilities between experimental architectures and standard DQN checkpoints."""

from __future__ import annotations

import copy
from os import PathLike
from typing import Any

import torch

from magic_pong.ai.models.dqn_checkpoint import build_dqn_checkpoint_metadata
from magic_pong.ai.models.dqn_checkpoint import safe_torch_load
from magic_pong.ai.models.dqn_checkpoint import validate_dqn_checkpoint


def convert_probe_to_dqn_checkpoint(
    probe_weights: str | PathLike[str] | dict[str, torch.Tensor],
    output_path: str | PathLike[str] | None = None,
    *,
    original_checkpoint: str | PathLike[str] | dict[str, Any] | None = None,
    hidden_layers: list[int] | None = None,
) -> dict[str, Any]:
    """Convert a ProbeNetwork weights dict into a standard v7 DQN checkpoint payload."""
    weights = (
        safe_torch_load(probe_weights)
        if isinstance(probe_weights, (str, PathLike))
        else probe_weights
    )

    q_state_dict: dict[str, torch.Tensor] = {}
    for k, v in weights.items():
        if k.startswith("base."):
            q_state_dict[k[5:]] = v
        elif k == "projection":
            q_state_dict["geometry_context_weight"] = v
        else:
            q_state_dict[k] = v

    if "geometry_context_weight" not in q_state_dict:
        raise ValueError("Probe weights must contain 'projection' or 'geometry_context_weight'")

    hp: dict[str, Any] = {
        "action_size": 9,
        "lr": 0.001,
        "gamma": 0.99,
        "epsilon_min": 0.05,
        "epsilon_decay": 0.995,
        "batch_size": 32,
    }
    training_step = 0
    epsilon = 0.05

    if original_checkpoint is not None:
        orig = (
            safe_torch_load(original_checkpoint)
            if isinstance(original_checkpoint, (str, PathLike))
            else original_checkpoint
        )
        if isinstance(orig, dict):
            if "hyperparameters" in orig and isinstance(orig["hyperparameters"], dict):
                hp.update(orig["hyperparameters"])
            training_step = orig.get("training_step", training_step)
            epsilon = orig.get("epsilon", epsilon)

    hp["state_size"] = 46
    hp["geometry_context"] = True
    if hidden_layers is not None:
        hp["hidden_layers"] = list(hidden_layers)
    elif "hidden_layers" not in hp:
        layers = []
        i = 0
        while f"fc_layers.{i}.weight" in q_state_dict:
            layers.append(q_state_dict[f"fc_layers.{i}.weight"].shape[0])
            i += 1
        hp["hidden_layers"] = layers
        hp["hidden_size"] = layers[0] if layers else 256
        hp["layer_size"] = len(layers)

    metadata = build_dqn_checkpoint_metadata(state_size=46, geometry_context=True)

    dummy_params = [torch.nn.Parameter(torch.empty(1)) for _ in q_state_dict]
    opt = torch.optim.Adam(dummy_params, lr=hp.get("lr", 0.001))
    optimizer_state = opt.state_dict()

    checkpoint = {
        "metadata": metadata,
        "hyperparameters": hp,
        "q_network_state_dict": q_state_dict,
        "target_network_state_dict": copy.deepcopy(q_state_dict),
        "optimizer_state_dict": optimizer_state,
        "epsilon": epsilon,
        "training_step": training_step,
        "loss_history": [],
        "reward_history": [],
    }

    validate_dqn_checkpoint(checkpoint)

    if output_path is not None:
        torch.save(checkpoint, output_path)

    return checkpoint
