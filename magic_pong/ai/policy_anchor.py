"""Opt-in defensive action rehearsal for the frame-training experiment.

Only action ordering is anchored: no reward, bootstrap target, replay item or
absolute Q value is supplied by the reference. Playable checkpoints stay plain
DQN models; the frame trainer owns the bank and sampler continuation state.
"""

from __future__ import annotations

import hashlib
import io
import math
import random
from typing import Any

import torch


def action_margin_loss(
    q_values: torch.Tensor, actions: torch.Tensor, margin: float
) -> torch.Tensor:
    chosen = q_values.gather(1, actions[:, None]).squeeze(1)
    others = q_values.scatter(1, actions[:, None], -torch.inf).max(dim=1).values
    return (others - chosen + margin).relu().square().mean()


class PolicyAnchor:
    def __init__(
        self,
        bank_bytes: bytes,
        expected_sha256: str,
        *,
        weight: float,
        margin: float,
        seed: int,
        state_size: int,
    ) -> None:
        if hashlib.sha256(bank_bytes).hexdigest() != expected_sha256:
            raise ValueError("anchor bank hash differs from configuration")
        if not math.isfinite(weight) or weight <= 0 or not math.isfinite(margin) or margin <= 0:
            raise ValueError("anchor weight and margin must be finite and positive")
        bank = torch.load(io.BytesIO(bank_bytes), map_location="cpu", weights_only=True)
        if bank.get("schema") != "magic_pong.policy_anchor_bank.v1":
            raise ValueError("unsupported anchor bank schema")
        self.states, self.actions, self.strata = bank["states"], bank["actions"], bank["strata"]
        if self.states.ndim != 2 or self.states.shape[1] != state_size:
            raise ValueError("anchor bank observation width differs from agent")
        count = len(self.states)
        if (
            not count
            or self.states.dtype != torch.float32
            or self.states.requires_grad
            or not torch.isfinite(self.states).all()
            or self.actions.shape != (count,)
            or self.strata.shape != (count,)
            or self.actions.dtype != torch.int64
            or self.strata.dtype != torch.int64
            or self.actions.min() < 0
            or self.actions.max() >= 9
        ):
            raise ValueError("invalid anchor bank tensors")
        labels = sorted(self.strata.unique().tolist())
        if labels != list(range(len(labels))):
            raise ValueError("anchor strata must be contiguous nonnegative integers")
        self.groups = [(self.strata == label).nonzero().flatten().tolist() for label in labels]
        if len({len(group) for group in self.groups}) != 1:
            raise ValueError("anchor bank strata must be balanced")
        self.bank_bytes, self.sha256 = bank_bytes, expected_sha256
        self.metadata = bank.get("metadata", {})
        self.weight, self.margin = weight, margin
        self.rng = random.Random(seed)
        self.active = False
        self.updates = 0
        self.td_sum = self.weighted_sum = 0.0

    def sample_indices(self, batch_size: int) -> list[int]:
        if batch_size % len(self.groups):
            raise ValueError("anchor batch size must be divisible by number of strata")
        return [
            self.rng.choice(group)
            for group in self.groups
            for _ in range(batch_size // len(self.groups))
        ]

    def loss(
        self, network: torch.nn.Module, batch_size: int, td_loss: torch.Tensor
    ) -> torch.Tensor:
        indices = self.sample_indices(batch_size)
        device = next(network.parameters()).device
        q = network(self.states[indices].to(device))
        loss = self.weight * action_margin_loss(q, self.actions[indices].to(device), self.margin)
        self.updates += 1
        self.td_sum += float(td_loss.detach())
        self.weighted_sum += float(loss.detach())
        return loss

    def summary(self) -> dict[str, Any]:
        return {
            "bank_sha256": self.sha256,
            "states": len(self.states),
            "strata": len(self.groups),
            "updates": self.updates,
            "td_sum": self.td_sum,
            "weighted_sum": self.weighted_sum,
        }

    def state_dict(self) -> dict[str, Any]:
        return {
            **self.summary(),
            "bank_bytes": self.bank_bytes,
            "rng": self.rng.getstate(),
            "weight": self.weight,
            "margin": self.margin,
        }

    def restore(self, state: dict[str, Any]) -> None:
        if (
            state["bank_sha256"] != self.sha256
            or state["bank_bytes"] != self.bank_bytes
            or state["weight"] != self.weight
            or state["margin"] != self.margin
        ):
            raise ValueError("anchor continuation differs from configuration")
        self.rng.setstate(state["rng"])
        self.updates, self.td_sum, self.weighted_sum = (
            state["updates"],
            state["td_sum"],
            state["weighted_sum"],
        )
