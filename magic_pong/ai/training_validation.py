"""Competitive checkpoint selection shared by training entry points."""

from __future__ import annotations

import json
from collections.abc import Callable
from collections.abc import Mapping
from pathlib import Path
from typing import Any

from magic_pong.ai import competitive


def _seeds(value: str | tuple[int, ...]) -> tuple[int, ...]:
    if isinstance(value, str):
        try:
            return tuple(int(part.strip()) for part in value.split(",") if part.strip())
        except ValueError as exc:
            raise ValueError("Evaluation seeds must be comma-separated integers") from exc
    return tuple(value)


def validation_config_from_args(
    args: Any, extra_opponents: tuple[str, ...] = ()
) -> competitive.CompetitiveConfig:
    defaults = competitive.CompetitiveConfig()
    opponents = getattr(args, "validation_opponents", None)
    if isinstance(opponents, str):
        opponents = tuple(part.strip() for part in opponents.split(",") if part.strip())
    opponents = tuple(dict.fromkeys((*tuple(opponents or defaults.opponents), *extra_opponents)))
    return competitive.CompetitiveConfig(
        opponents=opponents,
        seeds=_seeds(getattr(args, "validation_seeds", defaults.seeds)),
        held_out_seeds=_seeds(getattr(args, "test_seeds", defaults.held_out_seeds)),
        max_steps=getattr(args, "validation_max_steps", 18000),
        max_score=getattr(args, "validation_max_score", 11),
    )


class CheckpointSelector:
    """Save only candidates that improve a fixed, frozen validation protocol.

    Existing checkpoints are re-evaluated under this run's protocol before any
    replacement. The caller must keep neural opponent factories fixed for the
    selector's lifetime. Held-out seeds are never used for checkpoint selection.
    """

    def __init__(
        self,
        path: Path,
        config: competitive.CompetitiveConfig,
        interval: int,
        opponent_factories: Mapping[str, Callable[..., Any]] | None = None,
        *,
        patience_baseline_episode: int = 0,
    ) -> None:
        self.path = Path(path)
        self.config = config
        self.interval = interval
        self.opponent_factories = dict(opponent_factories or {})
        self.best_result: Any = None
        self.last_evaluated_episode: int | None = None
        self.last_improved_episode = patience_baseline_episode
        self.best_path_this_run: Path | None = None
        self.best_report_path_this_run: Path | None = None
        self.selected_best_this_run = False
        self._incumbent_checked = False

    def _payload(
        self,
        result: Any,
        episode: int,
        *,
        selected: bool,
        incumbent: bool = False,
    ) -> dict[str, Any]:
        payload = competitive.competitive_result_to_dict(result)
        payload["training_episode"] = episode
        payload["selected"] = selected
        payload["incumbent"] = incumbent
        payload["checkpoint_path"] = str(self.path)
        if incumbent:
            payload["checked_training_episode"] = episode
        return payload

    def _write_best_report(self, payload: dict[str, Any]) -> None:
        report = self.path.with_suffix(".validation.json")
        temporary_report = report.with_suffix(".json.tmp")
        temporary_report.write_text(json.dumps(payload, indent=2) + "\n")
        temporary_report.replace(report)
        self.best_report_path_this_run = report

    def consider(self, agent: Any, episode: int, *, force: bool = False) -> bool:
        if self.interval <= 0 or episode == self.last_evaluated_episode:
            return False
        if not force and episode % self.interval:
            return False

        if not self._incumbent_checked:
            if self.path.exists():
                self.best_result = competitive.evaluate_competitive_checkpoint(
                    self.path, self.config, opponent_factories=self.opponent_factories
                )
                self.best_path_this_run = self.path
            self._incumbent_checked = True

        result = competitive.evaluate_competitive_agent(
            agent, self.config, opponent_factories=self.opponent_factories
        )
        self.last_evaluated_episode = episode
        improved = self.best_result is None or competitive.is_better(result, self.best_result)
        payload = self._payload(result, episode, selected=improved)
        self.path.parent.mkdir(parents=True, exist_ok=True)
        with self.path.with_suffix(".validation.jsonl").open("a") as stream:
            stream.write(json.dumps(payload) + "\n")

        if improved:
            temporary = self.path.with_suffix(self.path.suffix + ".tmp")
            agent.save_model(str(temporary))
            temporary.replace(self.path)
            self._write_best_report(payload)
            self.best_result = result
            self.best_path_this_run = self.path
            self.selected_best_this_run = True
            self.last_improved_episode = episode
        elif self.best_result is not None and self.best_path_this_run is not None:
            self._write_best_report(
                self._payload(self.best_result, episode, selected=False, incumbent=True)
            )
        rank = competitive.competitive_rank(result)
        print(
            f"[validation] episode={episode} worst lower bound={rank[0]:.3f}"
            f" {'new champion' if improved else 'incumbent retained'}"
        )
        return improved
