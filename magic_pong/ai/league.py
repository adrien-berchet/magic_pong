"""Persistent, weakness-weighted opponents for competitive training.

A league directory has one writer. Its manifest stores sampling state, outcomes,
and immutable opponent checkpoints, so restarting a run does not erase history.
Managed snapshots are bounded; explicitly imported independent/exploiter opponents
are durable and their count is controlled by the caller.
"""

from __future__ import annotations

import hashlib
import json
import math
import random
import shutil
import uuid
from collections.abc import Callable
from collections.abc import Iterable
from collections.abc import Mapping
from dataclasses import asdict
from dataclasses import dataclass
from dataclasses import field
from functools import partial
from pathlib import Path
from typing import Any

from magic_pong.ai.evaluation import preserve_global_rng_state
from magic_pong.ai.models.simple_ai import BUILTIN_OPPONENTS
from magic_pong.ai.models.simple_ai import create_ai


@dataclass
class LeagueEntry:
    id: str
    kind: str
    path: str | None = None
    role: str = "builtin"
    sequence: int = 0
    champion: bool = False
    protected: bool = False
    games: int = 0
    wins: int = 0
    timeouts: int = 0
    metadata: dict[str, Any] = field(default_factory=dict)


def _default_opponent_factory(name: str) -> Any:
    if not name.startswith("dqn:"):
        return create_ai(name)
    from magic_pong.ai.models.dqn_ai import DQNAgent

    agent = DQNAgent()
    agent.load_model(name.removeprefix("dqn:"))
    return agent


def _tuple_tree(value: Any) -> Any:
    """Restore Random.getstate() after its JSON list representation."""
    return tuple(_tuple_tree(item) for item in value) if isinstance(value, list) else value


class OpponentLeague:
    """Cover all built-ins, recent policies, historical champions and specialists.

    ``coverage_floor`` reserves that fraction of probability mass for uniform
    coverage. Remaining mass favors opponents against which the learner has a
    low smoothed match win rate. Timeouts always count as non-wins.

    ``recent_size + champion_size + historical_size`` is the managed snapshot
    limit. Explicit benchmark references take priority within that limit. Leave
    at least one unprotected slot so new candidates can still enter the league.
    """

    def __init__(
        self,
        snapshot_dir: Path,
        *,
        seed: int = 42,
        rng: random.Random | None = None,
        base_weights: Mapping[str, float] | None = None,
        recent_size: int = 5,
        champion_size: int = 5,
        historical_size: int = 5,
        snapshot_weight: float = 0.5,
        coverage_floor: float = 0.2,
        opponent_factory: Callable[[str], Any] | None = None,
    ) -> None:
        if recent_size < 1 or champion_size < 0 or historical_size < 0:
            raise ValueError("recent_size must be positive; other pool sizes must be nonnegative")
        if not 0 <= snapshot_weight <= 1 or not 0 < coverage_floor <= 1:
            raise ValueError("snapshot_weight must be in [0, 1]; coverage_floor in (0, 1]")
        weights = dict(base_weights or {})
        if set(weights) - set(BUILTIN_OPPONENTS):
            raise ValueError("base_weights must contain only built-in opponent names")
        if any(not math.isfinite(value) or value < 0 for value in weights.values()):
            raise ValueError("base_weights must be finite and nonnegative")
        self.snapshot_dir = Path(snapshot_dir).resolve()
        self.snapshot_dir.mkdir(parents=True, exist_ok=True)
        self.manifest_path = self.snapshot_dir / "league.json"
        self.rng = rng if rng is not None else random.Random(seed)
        self.recent_size = recent_size
        self.champion_size = champion_size
        self.historical_size = historical_size
        self.snapshot_weight = snapshot_weight
        self.coverage_floor = coverage_floor
        # Missing built-ins receive a modest baseline; the uniform floor also
        # guarantees coverage when the caller explicitly assigns zero weight.
        self.base_weights = {name: weights.get(name, 1.0) for name in BUILTIN_OPPONENTS}
        self.opponent_factory = opponent_factory or _default_opponent_factory
        self.entries: dict[str, LeagueEntry] = {}
        self._cache: dict[str, Any] = {}
        self._counter = 0
        self.champion_id: str | None = None
        if self.manifest_path.exists():
            self._load()
        for name in BUILTIN_OPPONENTS:
            self.entries.setdefault(name, LeagueEntry(id=name, kind="builtin"))
        if self._protected_count() >= self.pool_size:
            raise ValueError(
                "pool capacity must leave one slot beyond protected benchmark references"
            )
        self._prune()
        self._save()

    @property
    def pool_size(self) -> int:
        return self.recent_size + self.champion_size + self.historical_size

    def _protected_count(self) -> int:
        return sum(entry.kind == "snapshot" and entry.protected for entry in self.entries.values())

    def _load(self) -> None:
        payload = json.loads(self.manifest_path.read_text())
        if payload.get("version") != 1:
            raise ValueError("Unsupported league manifest version")
        self._counter = int(payload["counter"])
        self.champion_id = payload.get("champion_id")
        for raw in payload["entries"]:
            entry = LeagueEntry(**raw)
            if entry.kind not in {"builtin", "snapshot", "external"}:
                raise ValueError(f"Unknown league entry kind: {entry.kind}")
            if entry.path is not None:
                path = Path(entry.path).resolve()
                # Every non-builtin entry is an owned immutable copy. Refuse a
                # hand-edited manifest that could cause pruning outside its dir.
                if path.parent != self.snapshot_dir:
                    raise ValueError(f"League checkpoint outside its directory: {path}")
                if not path.is_file():
                    raise FileNotFoundError(f"League checkpoint missing: {path}")
                digest = hashlib.sha256(path.read_bytes()).hexdigest()
                if entry.metadata.get("sha256", digest) != digest:
                    raise ValueError(f"League checkpoint contents changed: {path}")
                entry.metadata["sha256"] = digest
            self.entries[entry.id] = entry
        self.rng.setstate(_tuple_tree(payload["rng_state"]))

    def _save(self) -> None:
        payload = {
            "version": 1,
            "counter": self._counter,
            "champion_id": self.champion_id,
            "rng_state": self.rng.getstate(),
            "entries": [asdict(entry) for entry in self.entries.values()],
        }
        temporary = self.manifest_path.with_suffix(f".{uuid.uuid4().hex}.tmp")
        try:
            temporary.write_text(json.dumps(payload, indent=2, sort_keys=True) + "\n")
            temporary.replace(self.manifest_path)
        finally:
            temporary.unlink(missing_ok=True)

    def _allocate(self, prefix: str) -> tuple[str, Path]:
        # O_EXCL also avoids overwriting orphan files left by interrupted saves.
        while True:
            self._counter += 1
            identifier = f"{prefix}:{self._counter:06d}"
            path = self.snapshot_dir / f"{prefix}_{self._counter:06d}.pth"
            try:
                with path.open("xb"):
                    pass
            except FileExistsError:
                continue
            return identifier, path

    def add_snapshot(
        self,
        agent: Any,
        *,
        role: str = "snapshot",
        champion: bool = False,
        metadata: Mapping[str, Any] | None = None,
    ) -> str:
        """Save a candidate without modifying the live learner's modes."""
        metadata = dict(metadata or {})
        json.dumps(metadata)  # Fail before creating a file for unserializable metadata.
        identifier, path = self._allocate("snapshot")
        try:
            with preserve_global_rng_state():
                agent.save_model(str(path))
            metadata["sha256"] = hashlib.sha256(path.read_bytes()).hexdigest()
        except BaseException:
            path.unlink(missing_ok=True)
            raise
        self.entries[identifier] = LeagueEntry(
            id=identifier,
            kind="snapshot",
            path=str(path),
            role=role,
            sequence=self._counter,
            champion=champion,
            metadata=metadata,
        )
        if champion:
            self.champion_id = identifier
        self._prune()
        self._save()
        return identifier

    def add_checkpoint(
        self,
        path: Path | str,
        *,
        role: str = "independent",
        metadata: Mapping[str, Any] | None = None,
    ) -> str:
        """Import an independent policy or exploiter as an immutable owned copy.

        Originals are never removed or modified, including files outside the
        league directory. Duplicate source/role imports reuse the existing entry.
        """
        source = Path(path).resolve()
        if not source.is_file():
            raise FileNotFoundError(source)
        digest = hashlib.sha256(source.read_bytes()).hexdigest()
        for entry in self.entries.values():
            if (
                entry.kind == "external"
                and entry.metadata.get("source") == str(source)
                and entry.metadata.get("source_sha256") == digest
                and entry.role == role
            ):
                return entry.id
        details = {**dict(metadata or {}), "source": str(source), "source_sha256": digest}
        json.dumps(details)
        identifier, destination = self._allocate("external")
        try:
            shutil.copyfile(source, destination)
            details["sha256"] = hashlib.sha256(destination.read_bytes()).hexdigest()
            opponent = self._load_frozen_opponent(f"dqn:{destination}")
        except BaseException:
            destination.unlink(missing_ok=True)
            raise
        self.entries[identifier] = LeagueEntry(
            id=identifier,
            kind="external",
            path=str(destination),
            role=role,
            sequence=self._counter,
            protected=True,
            metadata=details,
        )
        self._cache[identifier] = opponent
        self._save()
        return identifier

    def promote(self, identifier: str) -> None:
        """Mark a validated candidate as current champion and retain its history."""
        entry = self.entries[identifier]
        if entry.kind == "builtin":
            raise ValueError("Only neural checkpoints can be promoted")
        entry.champion = True
        self.champion_id = identifier
        self._prune()
        self._save()

    def protect(self, identifier: str) -> None:
        """Pin a fixed benchmark checkpoint, retaining one candidate slot."""
        entry = self.entries[identifier]
        if (
            entry.kind == "snapshot"
            and not entry.protected
            and self._protected_count() >= self.pool_size - 1
        ):
            raise ValueError("Benchmark references must leave one slot for new snapshots")
        entry.protected = True
        self._save()

    @staticmethod
    def _diversity(entry: LeagueEntry, retained: list[LeagueEntry]) -> tuple[float, float, int]:
        """Prefer distinct evaluation profiles, roles, then spaced training ages."""
        role_novelty = float(all(other.role != entry.role for other in retained))
        profile = entry.metadata.get("performance_profile", {})
        distances = []
        if isinstance(profile, dict):
            for other in retained:
                previous = other.metadata.get("performance_profile", {})
                if isinstance(previous, dict) and profile.keys() == previous.keys() and profile:
                    distances.append(
                        sum((float(profile[key]) - float(previous[key])) ** 2 for key in profile)
                    )
        profile_distance = min(distances, default=0.0)
        age_distance = min(
            (abs(entry.sequence - other.sequence) for other in retained), default=entry.sequence
        )
        return role_novelty + profile_distance, float(age_distance), -entry.sequence

    def _prune(self) -> None:
        snapshots = sorted(
            (entry for entry in self.entries.values() if entry.kind == "snapshot"),
            key=lambda entry: entry.sequence,
        )
        retained = [entry for entry in snapshots if entry.protected]

        def keep(entry: LeagueEntry) -> None:
            if entry not in retained and len(retained) < self.pool_size:
                retained.append(entry)

        # Most recent policies remain available even when a benchmark pins old ones.
        for entry in reversed(snapshots[-self.recent_size :]):
            keep(entry)
        if self.champion_id in self.entries:
            current = self.entries[self.champion_id]
            if current.kind == "snapshot":
                keep(current)
        champions = [entry for entry in snapshots if entry.champion]
        for entry in reversed(champions[-self.champion_size :] if self.champion_size else []):
            keep(entry)
        candidates = [entry for entry in snapshots if entry not in retained]
        while candidates and len(retained) < self.pool_size:
            chosen = max(candidates, key=lambda entry: self._diversity(entry, retained))
            keep(chosen)
            candidates.remove(chosen)
        obsolete = [entry for entry in snapshots if entry not in retained]
        # Persist references before unlinking; a crash may leave harmless orphan
        # files, but cannot leave a committed manifest pointing at pruned files.
        for entry in obsolete:
            del self.entries[entry.id]
            self._cache.pop(entry.id, None)
        if obsolete:
            self._save()
            for entry in obsolete:
                if entry.path is not None:
                    Path(entry.path).unlink(missing_ok=True)

    def probabilities(self, only_ids: Iterable[str] | None = None) -> dict[str, float]:
        identifiers = list(dict.fromkeys(only_ids)) if only_ids is not None else list(self.entries)
        if not identifiers:
            raise ValueError("Opponent pool cannot be empty")
        entries = [self.entries[identifier] for identifier in identifiers]
        builtins = [entry for entry in entries if entry.kind == "builtin"]
        neural = [entry for entry in entries if entry.kind != "builtin"]
        neural_share = self.snapshot_weight if builtins and neural else float(bool(neural))
        total_base = sum(self.base_weights[entry.id] for entry in builtins) or len(builtins)
        scores = {}
        for entry in entries:
            if entry.kind == "builtin":
                weight = (1 - neural_share) * self.base_weights[entry.id] / total_base
            else:
                weight = neural_share / len(neural)
            # Beta(1, 1) smoothing avoids unstable probabilities from one match.
            weakness = 1.0 - (entry.wins + 1) / (entry.games + 2)
            scores[entry.id] = weight * weakness**2
        total = sum(scores.values())
        floor = self.coverage_floor / len(entries)
        return {
            identifier: floor
            + (1 - self.coverage_floor) * (score / total if total else 1 / len(entries))
            for identifier, score in scores.items()
        }

    def next_opponent(self, *, only_ids: Iterable[str] | None = None) -> tuple[str, Any]:
        probabilities = self.probabilities(only_ids)
        identifier = self.rng.choices(
            list(probabilities), weights=list(probabilities.values()), k=1
        )[0]
        self._save()
        return identifier, self._get_opponent(identifier)

    def record_outcome(self, identifier: str, *, won: bool, timeout: bool = False) -> None:
        entry = self.entries[identifier]
        entry.games += 1
        entry.wins += int(won and not timeout)
        entry.timeouts += int(timeout)
        self._save()

    @staticmethod
    def _freeze(opponent: Any) -> Any:
        toggle = getattr(opponent, "set_training_mode", None)
        if callable(toggle):
            toggle(False)
        if hasattr(opponent, "epsilon"):
            opponent.epsilon = 0.0
        for attribute in ("q_network", "target_network"):
            network = getattr(opponent, attribute, None)
            if network is not None:
                network.eval()
                network.requires_grad_(False)
        return opponent

    def _get_opponent(self, identifier: str) -> Any:
        entry = self.entries[identifier]
        if entry.kind == "builtin":
            with preserve_global_rng_state():
                return self.opponent_factory(identifier)
        if identifier not in self._cache:
            self._cache[identifier] = self._load_frozen_opponent(f"dqn:{entry.path}")
        return self._cache[identifier]

    def _load_frozen_opponent(self, name: str) -> Any:
        with preserve_global_rng_state():
            return self._freeze(self.opponent_factory(name))

    def benchmark_identity(self, identifier: str) -> str:
        """Bind evaluation protocol names to immutable checkpoint contents."""
        entry = self.entries[identifier]
        if entry.kind == "builtin":
            return identifier
        return f"{identifier}@sha256:{entry.metadata['sha256']}"

    def snapshot_references(self) -> tuple[str, ...]:
        """Capture historical and diverse opponents while reserving a live slot."""
        entries = [entry for entry in self.entries.values() if entry.kind != "builtin"]
        externals = [entry.id for entry in entries if entry.kind == "external"]
        snapshots = [entry for entry in entries if entry.kind == "snapshot"]
        # Prefer already fixed references and champions, then older diverse models.
        snapshots.sort(key=lambda entry: (not entry.protected, not entry.champion, entry.sequence))
        # Fixed references use historical slots; leave room for both recent
        # candidates and newly promoted champions throughout a long run.
        reference_limit = max(1, self.historical_size)
        reference_limit = min(reference_limit, self.pool_size - 1)
        return tuple(externals + [entry.id for entry in snapshots[:reference_limit]])

    def benchmark_factory(self, ids: Iterable[str] | None = None) -> Callable[..., Any]:
        factories = self.benchmark_factories(ids)

        def build(identifier: str, **_kwargs: Any) -> Any:
            if identifier in BUILTIN_OPPONENTS:
                return self.opponent_factory(identifier)
            return factories[identifier]()

        return build

    def benchmark_factories(
        self,
        ids: Iterable[str] | None = None,
        *,
        content_ids: bool = False,
    ) -> dict[str, Callable[[], Any]]:
        """Freeze the opponent set for an entire candidate-comparison protocol."""
        identifiers = tuple(self.snapshot_references() if ids is None else dict.fromkeys(ids))
        new_pins = sum(
            self.entries[identifier].kind == "snapshot" and not self.entries[identifier].protected
            for identifier in identifiers
        )
        if self._protected_count() + new_pins >= self.pool_size:
            raise ValueError("Benchmark references must leave one slot for new snapshots")
        for identifier in identifiers:
            self.entries[identifier].protected = True
        self._save()
        return {
            (self.benchmark_identity(identifier) if content_ids else identifier): (
                partial(self._get_opponent, identifier)
            )
            for identifier in identifiers
        }

    def cleanup(self) -> None:
        """Release cached agents; persistent checkpoint history survives shutdown."""
        self._save()
        self._cache.clear()
