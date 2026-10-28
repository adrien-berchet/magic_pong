"""Frame-budget training with continuous phase handoffs and boundary-exact resume.

Playable .pth models remain ordinary DQN checkpoints. Internal .training.pt
bundles include replay, RNG, league and trainer state, and resume only from a
completed episode. No live simulation state is silently discarded on resume.
"""

from __future__ import annotations

import hashlib
import io
import json
import math
import random
import uuid
from argparse import Namespace
from collections.abc import Iterable
from dataclasses import asdict
from dataclasses import dataclass
from pathlib import Path
from typing import Any
from typing import cast

import numpy as np
import torch

from magic_pong.ai.competitive import CompetitiveConfig
from magic_pong.ai.competitive import competitive_result_to_dict
from magic_pong.ai.competitive import evaluate_competitive_agent
from magic_pong.ai.evaluation import _seed_everything
from magic_pong.ai.evaluation import preserve_global_rng_state
from magic_pong.ai.league import OpponentLeague
from magic_pong.ai.models.dqn_ai import DQNAgent
from magic_pong.ai.models.dqn_ai import ReplayBuffer
from magic_pong.ai.pilot import PilotConfig
from magic_pong.ai.pilot import _isolated_training
from magic_pong.ai.pilot import _network_hash
from magic_pong.ai.policy_anchor import PolicyAnchor
from magic_pong.core.game_engine import GameEngine
from magic_pong.utils.config import ai_config
from magic_pong.utils.config import game_config

PHASES = ("curriculum", "anneal", "selfplay")
STRATA = ((1, False), (1, True), (2, False), (2, True))
SCHEMA = "magic_pong.frame_training.v1"


@dataclass(frozen=True)
class FramePipelineConfig:
    phase_frames: tuple[int, int, int] = (1_000_000, 1_000_000, 2_000_000)
    seed: int = 44
    epsilon_starts: tuple[float, float, float] = (0.30, 0.10, 0.20)
    epsilon_ends: tuple[float, float, float] = (0.10, 0.05, 0.05)
    decay_fraction: float = 0.8
    shaping_floor: float = 0.0
    learning_rate: float = 0.0003
    gamma: float = 0.99
    memory_size: int = 20000
    batch_size: int = 64
    min_replay_size: int = 1000
    train_frequency: int = 10
    target_update_frequency: int = 1000
    hidden_layers: tuple[int, ...] = (256, 128)
    max_episode_steps: int = 10000
    training_scores: tuple[int, ...] = (1, 3, 5, 11)
    terminal_on_goal: bool = False
    score_context: bool = False
    geometry_context: bool = False
    anchor_weight: float = 0.0
    anchor_margin: float = 0.01
    anchor_bank_sha256: str = ""
    curriculum_opponent: str = "follow_ball"
    anneal_opponent: str = "follow_ball"
    selfplay_opponent: str = "league"
    reward_shaping: str = "legacy"
    validation_interval: int = 100000
    snapshot_interval: int = 100000
    checkpoint_interval: int = 50000
    validation_opponents: tuple[str, ...] = (
        "follow_ball",
        "defensive",
        "aggressive",
        "random",
        "predictive",
        "dummy",
        "training_dummy",
        "tracker",
        "striker",
    )
    validation_seeds: tuple[int, ...] = (101, 102, 103)
    validation_max_steps: int = 18000
    validation_max_score: int = 11
    regression_patience: int = 3
    regression_margin: float = 0.15
    legacy_control: bool = False

    def __post_init__(self) -> None:
        if (
            not math.isfinite(self.anchor_weight)
            or self.anchor_weight < 0
            or not math.isfinite(self.anchor_margin)
            or self.anchor_margin <= 0
            or bool(self.anchor_weight) != bool(self.anchor_bank_sha256)
            or (
                self.anchor_bank_sha256
                and (
                    len(self.anchor_bank_sha256) != 64
                    or any(c not in "0123456789abcdef" for c in self.anchor_bank_sha256)
                )
            )
            or (self.anchor_weight and self.legacy_control)
        ):
            raise ValueError("invalid anchor configuration")
        if len(self.phase_frames) != 3 or any(n <= 0 or n % 4 for n in self.phase_frames):
            raise ValueError("phase_frames must contain three positive multiples of four")
        if not 0 < self.decay_fraction < 1:
            raise ValueError("decay_fraction must be between zero and one")
        if not math.isfinite(self.shaping_floor) or not 0 <= self.shaping_floor <= 1:
            raise ValueError("shaping_floor must be a finite value between zero and one")
        for values in (self.epsilon_starts, self.epsilon_ends):
            if len(values) != 3 or any(not math.isfinite(x) or not 0 <= x <= 1 for x in values):
                raise ValueError("epsilon schedules need three finite probabilities")
        if any(a < b for a, b in zip(self.epsilon_starts, self.epsilon_ends)):
            raise ValueError("epsilon schedules must decay")
        positive = (
            self.memory_size,
            self.batch_size,
            self.min_replay_size,
            self.train_frequency,
            self.target_update_frequency,
            self.max_episode_steps,
            self.validation_interval,
            self.snapshot_interval,
            self.checkpoint_interval,
        )
        if (
            any(n <= 0 for n in positive)
            or self.batch_size > self.memory_size
            or self.min_replay_size > self.memory_size
        ):
            raise ValueError("training sizes and intervals must be positive and fit replay")
        if self.steady_reward_tail_frames < self.memory_size:
            raise ValueError("anneal steady-reward tail must cover at least one replay capacity")
        if not self.training_scores or any(n <= 0 for n in self.training_scores):
            raise ValueError("training_scores must contain positive score targets")
        if self.reward_shaping not in ("legacy", "phase3"):
            raise ValueError("unsupported reward shaping mode")
        if self.selfplay_opponent not in ("league", "follow_ball"):
            raise ValueError("selfplay_opponent must be league or follow_ball")
        if (
            not math.isfinite(self.learning_rate)
            or self.learning_rate <= 0
            or not 0 < self.gamma <= 1
        ):
            raise ValueError("invalid learning rate or gamma")
        if self.regression_patience < 0 or not 0 <= self.regression_margin <= 1:
            raise ValueError("invalid regression guard")
        self.evaluation_config()

    @property
    def zero_reward_tail(self) -> int:
        return self.steady_reward_tail_frames if self.shaping_floor == 0 else 0

    @property
    def steady_reward_tail_frames(self) -> int:
        return self.phase_frames[1] - math.ceil(self.phase_frames[1] * self.decay_fraction)

    def epsilon(self, phase: int, frames: int) -> float:
        progress = min(1.0, max(0, frames) / (self.phase_frames[phase] * self.decay_fraction))
        return self.epsilon_starts[phase] + progress * (
            self.epsilon_ends[phase] - self.epsilon_starts[phase]
        )

    def shaping(self, phase: int, frames: int) -> float:
        if phase == 0:
            return 1.0
        if phase == 2:
            return self.shaping_floor
        fraction = 1.0 if self.legacy_control else self.decay_fraction
        remaining = max(0.0, 1.0 - frames / (self.phase_frames[phase] * fraction))
        return self.shaping_floor + (1.0 - self.shaping_floor) * remaining

    def evaluation_config(self, extra: Iterable[str] = ()) -> CompetitiveConfig:
        return CompetitiveConfig(
            opponents=self.validation_opponents + tuple(extra),
            seeds=self.validation_seeds,
            max_steps=self.validation_max_steps,
            max_score=self.validation_max_score,
        )


def source_hashes() -> dict[str, str]:
    root = Path(__file__).resolve().parents[2]
    files = sorted((root / "magic_pong").rglob("*.py")) + [root / "train_optimized.py"]
    return {str(p.relative_to(root)): hashlib.sha256(p.read_bytes()).hexdigest() for p in files}


def config_from_args(args: Namespace) -> FramePipelineConfig:
    return FramePipelineConfig(
        phase_frames=(
            args.pipeline_curriculum_frames,
            args.pipeline_anneal_frames,
            args.pipeline_selfplay_frames,
        ),
        seed=args.seed,
        # FramePipelineConfig validates the lengths and values in __post_init__.
        epsilon_starts=cast(
            tuple[float, float, float],
            tuple(float(x) for x in args.frame_epsilon_starts.split(",")),
        ),
        epsilon_ends=cast(
            tuple[float, float, float], tuple(float(x) for x in args.frame_epsilon_ends.split(","))
        ),
        decay_fraction=args.frame_decay_fraction,
        shaping_floor=args.frame_shaping_floor,
        terminal_on_goal=args.frame_terminal_on_goal,
        score_context=args.frame_score_context,
        geometry_context=getattr(args, "frame_geometry_context", False),
        learning_rate=args.learning_rate,
        gamma=args.gamma,
        memory_size=args.memory_size,
        batch_size=args.batch_size,
        min_replay_size=args.min_replay_size,
        train_frequency=args.train_frequency,
        hidden_layers=tuple(
            args.hidden_layers or [max(1, args.hidden_size // 2**i) for i in range(args.layer_size)]
        ),
        max_episode_steps=args.max_steps_per_episode,
        training_scores=tuple(int(x) for x in args.training_score_mix.split(",")),
        curriculum_opponent=args.phase1_opponent,
        anneal_opponent=args.pipeline_anneal_opponent or args.phase2_opponent,
        selfplay_opponent=args.frame_selfplay_opponent,
        reward_shaping=args.reward_shaping,
        validation_interval=args.frame_validation_interval,
        snapshot_interval=args.frame_snapshot_interval,
        checkpoint_interval=args.frame_checkpoint_interval,
        validation_opponents=tuple(args.validation_opponents.split(",")),
        validation_seeds=tuple(int(x) for x in args.validation_seeds.split(",")),
        validation_max_steps=args.validation_max_steps,
        validation_max_score=args.validation_max_score,
        regression_patience=args.frame_regression_patience,
    )


def model_bytes(agent: DQNAgent) -> bytes:
    stream = io.BytesIO()
    agent.save_model(stream)
    return stream.getvalue()


def load_model_bytes(agent: DQNAgent, data: bytes) -> None:
    agent.load_model(io.BytesIO(data))


def _write_json(path: Path, value: dict[str, Any]) -> None:
    temporary = path.with_suffix(path.suffix + ".tmp")
    temporary.write_text(json.dumps(value, indent=2) + "\n")
    temporary.replace(path)


def _save_bundle(path: Path, value: dict[str, Any]) -> None:
    temporary = path.with_suffix(path.suffix + ".tmp")
    torch.save(value, temporary)
    temporary.replace(path)


def _write_model(path: Path, data: bytes) -> None:
    temporary = path.with_suffix(path.suffix + ".tmp")
    temporary.write_bytes(data)
    temporary.replace(path)


def _rng_state() -> dict[str, Any]:
    # The global RandomState uses MT19937 and returns the legacy five-tuple.
    state = cast(tuple[str, np.ndarray, int, int, float], np.random.get_state())
    return {
        "python": random.getstate(),
        "numpy": (state[0], state[1].tolist(), *state[2:]),
        "torch": torch.get_rng_state(),
    }


def _restore_rng(state: dict[str, Any]) -> None:
    random.setstate(state["python"])
    n = state["numpy"]
    np.random.set_state((n[0], np.asarray(n[1], dtype=np.uint32), *n[2:]))
    torch.set_rng_state(state["torch"])


def _replay_state(agent: DQNAgent) -> dict[str, Any]:
    entries = list(agent.memory.buffer)
    return {
        "capacity": agent.memory.buffer.maxlen,
        "states": torch.from_numpy(np.array([e.state for e in entries], dtype=np.float32)),
        "actions": [e.action for e in entries],
        "rewards": [e.reward for e in entries],
        "next_states": torch.from_numpy(
            np.array([e.next_state for e in entries], dtype=np.float32)
        ),
        "dones": [e.done for e in entries],
    }


def _restore_replay(agent: DQNAgent, data: dict[str, Any]) -> None:
    agent.memory = ReplayBuffer(data["capacity"])
    for s, a, r, ns, done in zip(
        data["states"],
        data["actions"],
        data["rewards"],
        data["next_states"],
        data["dones"],
        strict=True,
    ):
        agent.memory.add(s.numpy().copy(), a, r, ns.numpy().copy(), done)


def _league_bundle(league: OpponentLeague) -> dict[str, Any]:
    return {
        "manifest": json.loads(league.manifest_path.read_text()),
        "files": {
            Path(e.path).name: Path(e.path).read_bytes() for e in league.entries.values() if e.path
        },
    }


def _restore_league(data: dict[str, Any], directory: Path) -> None:
    # A new owned directory avoids mutating a league that advanced after this save.
    directory.mkdir()
    manifest = json.loads(json.dumps(data["manifest"]))
    for entry in manifest["entries"]:
        if entry["path"]:
            name = Path(entry["path"]).name
            contents = data["files"][name]
            if hashlib.sha256(contents).hexdigest() != entry["metadata"]["sha256"]:
                raise ValueError("bundled league snapshot hash mismatch")
            (directory / name).write_bytes(contents)
            entry["path"] = str((directory / name).resolve())
    (directory / "league.json").write_text(json.dumps(manifest) + "\n")


def _fresh_agent(config: FramePipelineConfig, checkpoint: str | Path | None) -> DQNAgent:
    agent = DQNAgent(
        hidden_layers=list(config.hidden_layers),
        memory_size=config.memory_size,
        batch_size=config.batch_size,
        min_replay_size=config.min_replay_size,
        train_frequency=config.train_frequency,
        target_update_frequency=config.target_update_frequency,
        score_context=config.score_context,
        geometry_context=config.geometry_context,
    )
    if checkpoint:
        agent.load_model(str(checkpoint))
    if config.score_context and not agent.score_context:
        agent.enable_score_context()
    if agent.score_context != config.score_context:
        raise ValueError("Cannot discard score context from a trained checkpoint")
    if config.geometry_context and not agent.geometry_context:
        agent.enable_geometry_context()
    if agent.geometry_context != config.geometry_context:
        raise ValueError("Cannot discard geometry context from a trained checkpoint")
    # A warm start is a declared new experiment; exact continuation uses the bundle.
    agent.lr, agent.gamma, agent.batch_size = config.learning_rate, config.gamma, config.batch_size
    agent.terminal_on_goal = config.terminal_on_goal
    agent.optimizer = torch.optim.Adam(agent.q_network.parameters(), lr=config.learning_rate)
    agent.update_target_network()
    agent.step_count = agent.training_step = 0
    agent.loss_history, agent.reward_history, agent.grad_norm_history = [], [], []
    agent.epsilon, agent.epsilon_min = config.epsilon_starts[0], min(config.epsilon_ends)
    agent.epsilon_decay = 0.998 if config.legacy_control else 1.0
    return agent


def _builtin_rate(report: dict[str, Any], opponents: Iterable[str]) -> float:
    cells = [c for c in report["matchups"] if c["opponent"] in opponents]
    return sum(c["wins"] for c in cells) / sum(c["matches"] for c in cells)


def run_frame_pipeline(
    config: FramePipelineConfig,
    output_dir: str | Path,
    *,
    initial_checkpoint: str | Path | None = None,
    resume: str | Path | None = None,
    stop_after_episodes: int | None = None,
    anchor_bank: str | Path | None = None,
    retain_phase_bundles: bool = False,
) -> dict[str, Any]:
    """Run or resume; optional stop_after_episodes pauses at a reproducible boundary."""
    directory = Path(output_dir).resolve()
    original = None
    hashes = source_hashes()
    if resume:
        original = torch.load(resume, map_location="cpu", weights_only=True)
        if original.get("schema") != SCHEMA:
            raise ValueError("unsupported training bundle schema")
        original["config"].setdefault("shaping_floor", 0.0)
        original["config"].setdefault("selfplay_opponent", "league")
        original["config"].setdefault("terminal_on_goal", False)
        original["config"].setdefault("score_context", False)
        original["config"].setdefault("geometry_context", False)
        original["config"].setdefault("anchor_weight", 0.0)
        original["config"].setdefault("anchor_margin", 0.01)
        original["config"].setdefault("anchor_bank_sha256", "")
        if original["config"] != asdict(config):
            raise ValueError("resume config differs; start a new warm-start experiment instead")
        if original["source_hashes"] != hashes:
            raise ValueError("resume source/physics differs; start a new adaptation run")
        if (
            directory != Path(resume).resolve().parent
            and directory.exists()
            and any(directory.iterdir())
        ):
            raise FileExistsError("resume relocation requires an empty output directory")
    elif directory.exists() and any(directory.iterdir()):
        raise FileExistsError(
            "output already contains a run; use explicit resume or a new directory"
        )
    directory.mkdir(parents=True, exist_ok=True)
    isolation = PilotConfig(threads=1)
    with _isolated_training(isolation):
        ai_config.REWARD_SHAPING_MODE = config.reward_shaping
        _seed_everything(config.seed)
        agent = _fresh_agent(config, initial_checkpoint if not original else None)
        if original:
            load_model_bytes(agent, original["model"])
            _restore_replay(agent, original["replay"])
            league_dir = directory / ("league_resume_" + uuid.uuid4().hex)
            _restore_league(original["league"], league_dir)
        else:
            league_dir = directory / "league"
        if config.anchor_weight:
            if original:
                bank_bytes = original["anchor"]["bank_bytes"]
            elif anchor_bank is not None:
                bank_bytes = Path(anchor_bank).read_bytes()
            else:
                raise ValueError("a new anchored run requires an anchor bank")
            anchor = PolicyAnchor(
                bank_bytes,
                config.anchor_bank_sha256,
                weight=config.anchor_weight,
                margin=config.anchor_margin,
                seed=config.seed + 910000,
                state_size=agent.state_size,
            )
            if config.batch_size % len(anchor.groups):
                raise ValueError("anchor batch size must be divisible by number of strata")
            if original:
                anchor.restore(original["anchor"])
            agent.policy_anchor = anchor
        elif anchor_bank is not None:
            raise ValueError("anchor bank supplied to an unanchored run")
        league = OpponentLeague(league_dir, seed=config.seed)
        if original:
            state = original
        else:
            initial = model_bytes(agent)
            (directory / "initial.pth").write_bytes(initial)
            reference = league.add_checkpoint(directory / "initial.pth", role="initial_reference")
            state = {
                "schema": SCHEMA,
                "config": asdict(config),
                "source_hashes": hashes,
                "runtime": {"torch": str(torch.__version__), "device": "cpu", "threads": 1},
                "initial_checkpoint": str(initial_checkpoint) if initial_checkpoint else None,
                "resolved_game_config": game_config.model_dump(),
                "resolved_ai_config": ai_config.model_dump(),
                "initial_sha256": hashlib.sha256(initial).hexdigest(),
                "initial_network_sha256": _network_hash(agent),
                "initial_replay_size": len(agent.memory),
                "initial_optimizer_size": len(agent.optimizer.state),
                "references": [reference],
                "progress": {
                    "phase": 0,
                    "phase_frames": 0,
                    "total_frames": 0,
                    "episodes": 0,
                    "stratum_frames": [0, 0, 0, 0],
                    "zero_reward_frames": 0,
                    "steady_reward_tail_frames": 0,
                },
                "next_validation": config.validation_interval,
                "next_snapshot": config.snapshot_interval,
                "next_checkpoint": config.checkpoint_interval,
                "last_validation": -1,
                "best": None,
                "phase_best": None,
                "phase_results": [],
                "evaluations": [],
                "episodes": [],
                "bad_validations": 0,
                "status": "running",
            }
        factories = league.benchmark_factories(state["references"], content_ids=True)
        evaluation_config = config.evaluation_config(factories)
        progress = state["progress"]
        progress.setdefault(
            "steady_reward_tail_frames",
            (
                progress["zero_reward_frames"]
                if progress["phase"] == 1 and not config.legacy_control
                else 0
            ),
        )
        if original:
            if state["runtime"] != {"torch": str(torch.__version__), "device": "cpu", "threads": 1}:
                raise ValueError("resume runtime differs")
            _restore_rng(state["rng"])
        else:
            _seed_everything(config.seed)

        def evaluate() -> None:
            if state["last_validation"] == progress["total_frames"]:
                return
            report = competitive_result_to_dict(
                evaluate_competitive_agent(agent, evaluation_config, opponent_factories=factories)
            )
            weights = model_bytes(agent)
            report.update(
                env_frames=progress["total_frames"],
                phase=PHASES[min(progress["phase"], 2)],
                model_sha256=hashlib.sha256(weights).hexdigest(),
                network_sha256=_network_hash(agent),
            )
            report["builtin_win_rate"] = _builtin_rate(report, config.validation_opponents)
            candidate_path = directory / f"candidate_{progress['total_frames']:09d}.pth"
            _write_model(candidate_path, weights)
            report["checkpoint_path"] = str(candidate_path)
            selected = state["best"] is None or report["rank"] > state["best"]["report"]["rank"]
            if selected:
                state["best"] = {"model": weights, "report": report}
            if (
                state["phase_best"] is None
                or report["rank"] > state["phase_best"]["report"]["rank"]
            ):
                state["phase_best"] = {"model": weights, "report": report}
            decline = (
                report["builtin_win_rate"]
                < state["best"]["report"]["builtin_win_rate"] - config.regression_margin
            )
            state["bad_validations"] = state["bad_validations"] + 1 if decline else 0
            state["last_validation"] = progress["total_frames"]
            state["last_report"] = report
            state["evaluations"].append(
                {
                    "frames": progress["total_frames"],
                    "selected": selected,
                    "phase": report["phase"],
                    "rank": report["rank"],
                    "builtin_win_rate": report["builtin_win_rate"],
                }
            )
            _write_json(directory / f"validation_{progress['total_frames']:09d}.json", report)
            print(
                f"validation frames={progress['total_frames']} builtin_win_rate={report['builtin_win_rate']:.1%} selected={selected}",
                flush=True,
            )

        def checkpoint(status: str) -> dict[str, Any]:
            state["status"] = status
            state["model"] = model_bytes(agent)
            state["replay"] = _replay_state(agent)
            state["rng"] = _rng_state()
            state["league"] = _league_bundle(league)
            if agent.policy_anchor is not None:
                state["anchor"] = agent.policy_anchor.state_dict()
            _save_bundle(directory / "latest.training.pt", state)
            if (
                retain_phase_bundles
                and progress["phase"] in (1, 2)
                and progress["phase_frames"] == 0
            ):
                _save_bundle(directory / f"phase_{progress['phase']}_start.training.pt", state)
            _write_model(directory / "latest.pth", state["model"])
            if state["best"]:
                _write_model(directory / "best.pth", state["best"]["model"])
                _write_json(directory / "best.validation.json", state["best"]["report"])
            summary = {
                k: state[k]
                for k in (
                    "schema",
                    "config",
                    "source_hashes",
                    "runtime",
                    "progress",
                    "phase_results",
                    "evaluations",
                    "status",
                )
            }
            summary["episodes"] = state["episodes"]
            if agent.policy_anchor is not None:
                summary["anchor"] = agent.policy_anchor.summary()
            summary["best_frames"] = (
                state["best"]["report"]["env_frames"] if state["best"] else None
            )
            summary["best_builtin_win_rate"] = (
                state["best"]["report"]["builtin_win_rate"] if state["best"] else None
            )
            summary["initial_sha256"] = state["initial_sha256"]
            for key in ("initial_network_sha256", "initial_replay_size", "initial_optimizer_size"):
                summary[key] = state[key]
            _write_json(directory / "training.json", summary)
            return summary

        try:
            if original and state["status"] in ("complete", "regressed"):
                return checkpoint(state["status"])
            if not original:
                evaluate()
                checkpoint("running")
            state["status"] = "running"
            while progress["phase"] < 3:
                phase = progress["phase"]
                if agent.policy_anchor is not None:
                    agent.policy_anchor.active = phase == 2
                quota = config.phase_frames[phase] // 4
                stratum = min(range(4), key=lambda i: progress["stratum_frames"][i])
                side, bonuses = STRATA[stratum]
                remaining = quota - progress["stratum_frames"][stratum]
                game_config.BONUSES_ENABLED = bonuses
                game_config.MAX_SCORE = random.choice(config.training_scores)
                if phase < 2:
                    opponent_id = (
                        config.curriculum_opponent if phase == 0 else config.anneal_opponent
                    )
                    opponent = league._get_opponent(opponent_id)
                elif config.selfplay_opponent != "league":
                    opponent_id = config.selfplay_opponent
                    opponent = league._get_opponent(opponent_id)
                else:
                    opponent_id, opponent = league.next_opponent()
                engine = GameEngine(headless=True)
                engine.set_players(*((agent, opponent) if side == 1 else (opponent, agent)))
                agent.on_episode_start()
                engine.start_game()
                limit = min(remaining, config.max_episode_steps)
                engine.ai_environment.max_steps = limit
                frames = goals_for = goals_against = 0
                td_terminals = nonterminal_goal_td_terminals = 0
                goal_transitions = []
                reward = 0.0
                start = progress["phase_frames"]
                try:
                    while engine.is_running() and frames < limit:
                        if not config.legacy_control:
                            agent.epsilon = config.epsilon(phase, progress["phase_frames"])
                        ai_config.MICRO_REWARD_SCALE = config.shaping(
                            phase, progress["phase_frames"]
                        )
                        result = engine.update(
                            dt=game_config.GAME_SPEED_MULTIPLIER / game_config.FPS
                        )
                        if ai_config.MICRO_REWARD_SCALE == 0:
                            progress["zero_reward_frames"] += 1
                        if (
                            phase == 1
                            and progress["phase_frames"]
                            >= math.ceil(config.phase_frames[1] * config.decay_fraction)
                            and ai_config.MICRO_REWARD_SCALE == config.shaping_floor
                        ):
                            progress["steady_reward_tail_frames"] += 1
                        frames += 1
                        progress["phase_frames"] += 1
                        progress["total_frames"] += 1
                        progress["stratum_frames"][stratum] += 1
                        reward += result["rewards"][f"player{side}"]
                        # Read the mask actually stored by the learner, separately
                        # from simulator termination/truncation. No RNG is consumed.
                        td_terminal = bool(agent.memory.buffer[-1].done)
                        td_terminals += td_terminal
                        if result["events"].get("goals"):
                            physical_terminal = bool(result["info"]["terminated"])
                            nonterminal_goal_td_terminals += td_terminal and not physical_terminal
                            goal_transitions.append(
                                {
                                    "frame": progress["total_frames"],
                                    "terminated": physical_terminal,
                                    "truncated": bool(result["info"]["truncated"]),
                                    "td_terminal": td_terminal,
                                    "reward": result["rewards"][f"player{side}"],
                                    "scorers": [g["player"] for g in result["events"]["goals"]],
                                }
                            )
                        for goal in result["events"].get("goals", []):
                            goals_for += goal["player"] == side
                            goals_against += goal["player"] != side
                        if result["done"]:
                            break
                finally:
                    if engine.is_running():
                        engine.stop_game()
                if not frames:
                    raise RuntimeError("environment made no progress")
                if not config.legacy_control:
                    agent.epsilon = config.epsilon(phase, progress["phase_frames"])
                ai_config.MICRO_REWARD_SCALE = config.shaping(phase, progress["phase_frames"])
                progress["episodes"] += 1
                winner = result["info"].get("winner", 0)
                if phase == 2:
                    league.record_outcome(opponent_id, won=winner == side, timeout=winner == 0)
                state["episodes"].append(
                    {
                        "episode": progress["episodes"],
                        "phase": PHASES[phase],
                        "start_phase_frames": start,
                        "end_frames": progress["total_frames"],
                        "frames": frames,
                        "max_score": game_config.MAX_SCORE,
                        "step_limit": limit,
                        "episode_done": bool(result["done"]),
                        "terminated": bool(result["info"]["terminated"]),
                        "truncated": bool(result["info"]["truncated"]),
                        "td_terminals": td_terminals,
                        "nonterminal_goal_td_terminals": nonterminal_goal_td_terminals,
                        "goal_transitions": goal_transitions,
                        "side": side,
                        "bonuses": bonuses,
                        "opponent": opponent_id,
                        "winner": winner,
                        "goals_for": goals_for,
                        "goals_against": goals_against,
                        "reward": reward,
                        "epsilon": agent.epsilon,
                        "shaping": ai_config.MICRO_REWARD_SCALE,
                        "td_loss": (
                            float(np.mean(agent.loss_history[-100:]))
                            if agent.loss_history
                            else None
                        ),
                    }
                )
                at_end = progress["phase_frames"] == config.phase_frames[phase]
                if progress["total_frames"] >= state["next_validation"] or at_end:
                    evaluate()
                    while state["next_validation"] <= progress["total_frames"]:
                        state["next_validation"] += config.validation_interval
                if (
                    config.regression_patience
                    and state["bad_validations"] >= config.regression_patience
                ):
                    return checkpoint("regressed")
                if progress["total_frames"] >= state["next_snapshot"]:
                    if phase == 2:
                        league.add_snapshot(agent, metadata={"frames": progress["total_frames"]})
                    while state["next_snapshot"] <= progress["total_frames"]:
                        state["next_snapshot"] += config.snapshot_interval
                if at_end:
                    if phase == 1 and not config.legacy_control:
                        assert progress["steady_reward_tail_frames"] >= config.memory_size
                    (directory / f"{PHASES[phase]}_final.pth").write_bytes(model_bytes(agent))
                    state["phase_results"].append(
                        {
                            "phase": PHASES[phase],
                            "end_frames": progress["total_frames"],
                            "agent_frames": agent.step_count,
                            "updates": agent.training_step,
                            "zero_reward_frames": progress["zero_reward_frames"],
                            "steady_reward_tail_frames": progress["steady_reward_tail_frames"],
                            "shaping_end": ai_config.MICRO_REWARD_SCALE,
                            "epsilon_end": agent.epsilon,
                        }
                    )
                    if config.legacy_control and phase < 2:
                        with preserve_global_rng_state():
                            load_model_bytes(agent, state["phase_best"]["model"])
                    else:
                        # The next phase starts with these exact weights. Its
                        # initial candidate is already evaluated under the same
                        # fixed protocol; no RNG-affecting repeat is necessary.
                        state["phase_best"] = {
                            "model": model_bytes(agent),
                            "report": state["last_report"],
                        }
                    progress.update(
                        phase=phase + 1,
                        phase_frames=0,
                        stratum_frames=[0, 0, 0, 0],
                        zero_reward_frames=0,
                        steady_reward_tail_frames=0,
                    )
                if progress["phase"] == 3:
                    (directory / "final.pth").write_bytes(model_bytes(agent))
                    return checkpoint("complete")
                if stop_after_episodes is not None and progress["episodes"] >= stop_after_episodes:
                    return checkpoint("paused")
                if progress["total_frames"] >= state["next_checkpoint"] or at_end:
                    while state["next_checkpoint"] <= progress["total_frames"]:
                        state["next_checkpoint"] += config.checkpoint_interval
                    checkpoint("running")
                    print(
                        f"training frames={progress['total_frames']}/{sum(config.phase_frames)} phase={PHASES[progress['phase']]} epsilon={agent.epsilon:.3f}",
                        flush=True,
                    )
        except KeyboardInterrupt:
            # The latest atomic bundle is a valid earlier boundary. Never save a
            # partially played episode as if it could be resumed without physics.
            print("Interrupted; resume from the last saved episode boundary.", flush=True)
            raise
        finally:
            league.cleanup()
        raise RuntimeError("training exited without a completed checkpoint boundary")
