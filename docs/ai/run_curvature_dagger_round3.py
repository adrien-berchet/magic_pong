"""Option A: Curvature & Bounce-Targeted DAgger (Round 3) in Magic Pong.

Executes a two-stage targeted DAgger protocol across training seeds 11, 22, 33,
combining high-urgency and wall-bounce filtering, hybrid planner-barrier projection,
and strict non-Striker KL policy anchoring.
"""

from __future__ import annotations

import os

os.environ["CUDA_VISIBLE_DEVICES"] = ""

import argparse
import copy
import hashlib
import json
import math
import multiprocessing as mp
import random
import shutil
from pathlib import Path
from typing import Any

import numpy as np
import torch
import torch.nn.functional as F

from magic_pong.ai.action_shaping import GeometricShieldConfig
from magic_pong.ai.agent_adapter import to_player1_dqn_observation
from magic_pong.ai.competitive import CompetitiveConfig
from magic_pong.ai.competitive import evaluate_competitive_agent
from magic_pong.ai.curvature_dagger import compute_concession_per_hit
from magic_pong.ai.curvature_dagger import filter_targeted_tiers
from magic_pong.ai.curvature_dagger import project_planner_target
from magic_pong.ai.geometry import compute_geometry_features_from_physics_batch
from magic_pong.ai.models.dqn_ai import DQNAgent
from magic_pong.ai.models.dqn_ai import DQNNetwork
from magic_pong.ai.models.planner_ai import make_teacher
from magic_pong.ai.models.simple_ai import create_ai
from magic_pong.core.game_engine import TrainingManager
from magic_pong.utils.config import game_config_tmp

ROOT = Path("models/curvature_dagger_round3_2026-10-30")
PROMOTED_PATH = Path("models/dqn_v7_promoted.pth")
LEGACY_PID_DIR = Path("models/planner_imitation_diagnostics_2026-09-21")
SCHEMA = "magic_pong.curvature_dagger_round3.v1"

SETTINGS: dict[str, Any] = {
    "training_seeds": [11, 22, 33],
    # Collection seeds
    "stage1_collection_seeds": list(range(797100, 797108)),  # 8 seeds: 32 promoted, 32 shield
    "anchor_collection_seeds": list(range(797100, 797104)),  # 4 seeds: 48 non-striker matches
    "stage2_collection_seeds": list(
        range(797108, 797116)
    ),  # 8 seeds: 32 unshielded stage 2 matches
    # Validation seeds
    "val_non_striker_seeds": list(range(798100, 798104)),  # 4 seeds: 48 matches (6k frames)
    "val_striker_seeds": list(range(798100, 798116)),  # 16 seeds: 64 matches (18k frames)
    # Heldout and Screen seeds
    "heldout_seeds": list(range(800100, 800108)),  # 8 seeds: 128 matches (6k frames)
    "long_screen_seeds": list(range(801100, 801116)),  # 16 seeds: 64 matches each (18k frames)
    "sides": [1, 2],
    "bonuses": [False, True],
    "opponents": ["follow_ball", "aggressive", "tracker", "striker"],
    "non_striker_opponents": ["follow_ball", "aggressive", "tracker"],
    # Optimization
    "learning_rate": 2e-4,
    "gradient_clip": 10.0,
    "batch_targeted": 64,
    "batch_anchor": 192,
    "anchor_lambda": 1.0,
    "stage1_updates": 6000,
    "stage2_updates": 6000,
    "checkpoint_updates": [1500, 3000, 6000],
}


def sha(path: Path) -> str:
    h = hashlib.sha256()
    with open(path, "rb") as f:
        while chunk := f.read(65536):
            h.update(chunk)
    return h.hexdigest()


def write_json(path: Path, data: Any) -> None:
    path.parent.mkdir(parents=True, exist_ok=True)
    temp = path.with_suffix(".tmp")
    with open(temp, "w") as f:
        json.dump(data, f, indent=2, sort_keys=True)
    temp.replace(path)


def assert_splits(settings: dict[str, Any]) -> None:
    """Ensure strict seed isolation across historical and new evaluation splits."""
    consumed_past = {
        680100,
        680101,
        680102,
        680103,
        690100,
        690101,
        690102,
        690103,
        *range(700100, 700108),
        *range(710100, 710104),
        *range(720100, 720108),
        *range(730100, 730116),
        *range(740100, 740104),
        *range(750100, 750116),
        *range(760100, 760108),
        *range(770100, 770116),
        *range(795100, 795108),  # pilot dev
        *range(796100, 796108),  # pilot dev eval
    }
    new_seeds = set(
        settings["stage1_collection_seeds"]
        + settings["anchor_collection_seeds"]
        + settings["stage2_collection_seeds"]
        + settings["val_non_striker_seeds"]
        + settings["val_striker_seeds"]
        + settings["heldout_seeds"]
        + settings["long_screen_seeds"]
    )
    if consumed_past & new_seeds:
        overlap = sorted(consumed_past & new_seeds)
        raise ValueError(f"Reused historical experiment seeds: {overlap}")

    col_set = set(settings["stage1_collection_seeds"] + settings["stage2_collection_seeds"])
    val_set = set(settings["val_non_striker_seeds"] + settings["val_striker_seeds"])
    held_set = set(settings["heldout_seeds"])
    screen_set = set(settings["long_screen_seeds"])

    if len(col_set & val_set) > 0 or len(col_set & held_set) > 0 or len(col_set & screen_set) > 0:
        raise ValueError("Collection seeds overlap with evaluation panels!")
    if len(val_set & held_set) > 0 or len(val_set & screen_set) > 0:
        raise ValueError("Validation seeds overlap with heldout/screen panels!")
    if len(held_set & screen_set) > 0:
        raise ValueError("Heldout seeds overlap with long-game screen seeds!")


class InstrumentedTrainingManager(TrainingManager):
    """TrainingManager wrapping game_engine.update to log opponent paddle hits."""

    def __init__(self, *args: Any, **kwargs: Any) -> None:
        super().__init__(*args, **kwargs)
        self.recorded_hits: dict[str, int] = {"p1_hits": 0, "p2_hits": 0}

    def train_episode(self, player1: Any, player2: Any, max_steps: int = 10000) -> dict[str, Any]:
        self.recorded_hits["p1_hits"] = 0
        self.recorded_hits["p2_hits"] = 0
        orig_update = self.game_engine.update

        def wrapped_update(*u_args: Any, **u_kwargs: Any) -> dict[str, Any]:
            res = orig_update(*u_args, **u_kwargs)
            ev = res.get("events", {})
            for h in ev.get("paddle_hits", []):
                if h["player"] == 1:
                    self.recorded_hits["p1_hits"] += 1
                elif h["player"] == 2:
                    self.recorded_hits["p2_hits"] += 1
            return res

        self.game_engine.update = wrapped_update  # type: ignore[assignment]
        try:
            stats = super().train_episode(player1, player2, max_steps=max_steps)
            stats["recorded_hits"] = dict(self.recorded_hits)
            return stats
        finally:
            self.game_engine.update = orig_update  # type: ignore[assignment]


def _eval_chunk_task(task: tuple[str, CompetitiveConfig, str, bool]) -> list[dict[str, Any]]:
    """Worker task executing evaluate_competitive_agent on a seed chunk."""
    ckpt_path, cfg, split, shield_on = task
    torch.set_num_threads(1)
    agent = DQNAgent()
    agent.load_model(ckpt_path)
    agent.set_training_mode(False)
    agent.set_exploration_mode(False)
    if shield_on:
        agent.configure_action_shaper(GeometricShieldConfig())

    managers: list[InstrumentedTrainingManager] = []

    def factory(headless: bool = True) -> InstrumentedTrainingManager:
        mgr = InstrumentedTrainingManager(headless=headless)
        managers.append(mgr)
        return mgr

    res = evaluate_competitive_agent(
        agent,
        cfg,
        split=split,
        training_manager_factory=factory,
    )

    matches_dict = []
    for i, m in enumerate(res.matches):
        hits_info = managers[i].recorded_hits if i < len(managers) else {"p1_hits": 0, "p2_hits": 0}
        agent_hits = hits_info["p1_hits"] if m.player_side == 1 else hits_info["p2_hits"]
        opp_hits = hits_info["p2_hits"] if m.player_side == 1 else hits_info["p1_hits"]
        matches_dict.append(
            {
                "opponent": m.opponent,
                "seed": m.seed,
                "player_side": m.player_side,
                "bonuses_enabled": m.bonuses_enabled,
                "agent_won": m.agent_won,
                "winner": m.winner,
                "steps": m.steps,
                "goals_for": m.goals_for,
                "goals_against": m.goals_against,
                "timeout": m.timeout,
                "agent_paddle_hits": agent_hits,
                "opp_paddle_hits": opp_hits,
            }
        )
    return matches_dict


def evaluate_agent_instrumented(
    ckpt_path: str | Path,
    config: CompetitiveConfig,
    split: str,
    shield_on: bool = False,
    workers: int = 8,
) -> dict[str, Any]:
    """Evaluate agent using official harness across parallel seed chunks."""
    seeds = config.match_seeds(split)
    if not seeds:
        return {}

    # Chunk seeds across workers
    chunk_size = max(1, math.ceil(len(seeds) / workers))
    tasks = []
    for i in range(0, len(seeds), chunk_size):
        seed_chunk = tuple(seeds[i : i + chunk_size])
        sub_cfg = CompetitiveConfig(
            opponents=config.opponents,
            seeds=seed_chunk if split == "validation" else config.seeds,
            held_out_seeds=seed_chunk if split == "held_out" else config.held_out_seeds,
            player_sides=config.player_sides,
            bonus_modes=config.bonus_modes,
            max_steps=config.max_steps,
            max_score=config.max_score,
        )
        tasks.append((str(ckpt_path), sub_cfg, split, shield_on))

    if len(tasks) == 1:
        matches_dict = _eval_chunk_task(tasks[0])
    else:
        with mp.Pool(min(len(tasks), workers)) as pool:
            chunk_results = pool.map(_eval_chunk_task, tasks)
        matches_dict = sum(chunk_results, [])

    # Compute aggregate stats
    gf = sum(m["goals_for"] for m in matches_dict)
    ga = sum(m["goals_against"] for m in matches_dict)
    st = sum(m["steps"] for m in matches_dict)
    opp_h = sum(m["opp_paddle_hits"] for m in matches_dict)
    wins = sum(m["agent_won"] for m in matches_dict)
    losses = sum(not m["agent_won"] and not m["timeout"] for m in matches_dict)
    timeouts = sum(m["timeout"] for m in matches_dict)

    point_share = float(gf / max(1, gf + ga))
    ga_1k = float(1000.0 * ga / max(1, st))
    gf_1k = float(1000.0 * gf / max(1, st))
    concede_per_hit = compute_concession_per_hit(ga, opp_h)

    # Bonus breakdowns
    b_on = [m for m in matches_dict if m["bonuses_enabled"]]
    b_off = [m for m in matches_dict if not m["bonuses_enabled"]]

    return {
        "n": len(matches_dict),
        "wins": wins,
        "losses": losses,
        "timeouts": timeouts,
        "goals_for": gf,
        "goals_against": ga,
        "steps": st,
        "opp_paddle_hits": opp_h,
        "point_share": point_share,
        "ga_1k": ga_1k,
        "gf_1k": gf_1k,
        "ga_per_1k": ga_1k,
        "gf_per_1k": gf_1k,
        "concede_per_hit": concede_per_hit,
        "bonus_on": {
            "gf": sum(m["goals_for"] for m in b_on),
            "ga": sum(m["goals_against"] for m in b_on),
            "opp_hits": sum(m["opp_paddle_hits"] for m in b_on),
            "concede_per_hit": compute_concession_per_hit(
                sum(m["goals_against"] for m in b_on), sum(m["opp_paddle_hits"] for m in b_on)
            ),
        },
        "bonus_off": {
            "gf": sum(m["goals_for"] for m in b_off),
            "ga": sum(m["goals_against"] for m in b_off),
            "opp_hits": sum(m["opp_paddle_hits"] for m in b_off),
            "concede_per_hit": compute_concession_per_hit(
                sum(m["goals_against"] for m in b_off), sum(m["opp_paddle_hits"] for m in b_off)
            ),
        },
        "matches": matches_dict,
    }


def collect_match(task: tuple[str, str, str, int, int, bool, int, bool, Path]) -> str:
    """Run one match, extracting frame-level state, planner labels, hits, and goals."""
    arm, ckpt_path, opp_name, seed, side, bonus, max_steps, label_teacher, out_dir = task
    out_path = out_dir / f"{arm}__{opp_name}__{seed}_{side}_{int(bonus)}.npz"
    if out_path.exists():
        return str(out_path)

    torch.set_num_threads(1)
    random.seed(seed)
    np.random.seed(seed % (2**32))
    torch.manual_seed(seed)

    agent = DQNAgent()
    agent.load_model(ckpt_path)
    agent.set_training_mode(False)
    agent.set_exploration_mode(False)

    if arm == "shield":
        agent.configure_action_shaper(GeometricShieldConfig())

    controller = agent
    teacher = make_teacher(90, 30, backend="fast") if label_teacher else None
    opponent = create_ai(opp_name)

    states: list[np.ndarray] = []
    logits: list[np.ndarray] = []
    values: list[np.ndarray] = []
    ga_flags: list[np.ndarray] = []
    hit_for: list[bool] = []
    hit_against: list[bool] = []
    goal_for: list[bool] = []
    goal_against: list[bool] = []

    with game_config_tmp(BONUSES_ENABLED=bonus, MAX_SCORE=3):
        mgr = TrainingManager(headless=True)
        engine = mgr.game_engine
        p1, p2 = (controller, opponent) if side == 1 else (opponent, controller)
        engine.set_players(p1, p2)
        engine.start_game()
        engine.ai_environment.max_steps = max_steps

        step = 0
        while engine.is_running() and step < max_steps:
            ph = engine.physics_engine
            raw_obs = engine.ai_environment.observation_processor.process_game_state(
                ph.get_game_state(), side
            )
            obs = to_player1_dqn_observation(raw_obs, side)
            s = agent._observation_to_state(obs)

            with torch.no_grad():
                lg = agent.q_network(torch.as_tensor(s)).cpu().numpy()

            if teacher is not None:
                dec = teacher.label(ph, side)
                v = np.asarray(dec.values, dtype=np.float32)
                ga_act = np.array([dec.outcomes[a].goals_against > 0 for a in range(9)], dtype=bool)
            else:
                v = np.full(9, np.nan, dtype=np.float32)
                ga_act = np.zeros(9, dtype=bool)

            res = engine.update()
            step += 1
            ev = res.get("events", {})
            hits = [h["player"] for h in ev.get("paddle_hits", [])]
            goals = [g["player"] for g in ev.get("goals", [])]

            states.append(s.copy())
            logits.append(lg)
            values.append(v)
            ga_flags.append(ga_act)
            hit_for.append(side in hits)
            hit_against.append((3 - side) in hits)
            goal_for.append(side in goals)
            goal_against.append((3 - side) in goals)

            if res.get("done"):
                break

        mgr.cleanup()

    np.savez_compressed(
        out_path,
        state=np.array(states, dtype=np.float32),
        logits=np.array(logits, dtype=np.float32),
        values=np.array(values, dtype=np.float32),
        ga_flag=np.array(ga_flags, dtype=bool),
        hit_for=np.array(hit_for, dtype=bool),
        hit_against=np.array(hit_against, dtype=bool),
        goal_for=np.array(goal_for, dtype=bool),
        goal_against=np.array(goal_against, dtype=bool),
    )
    return str(out_path)


def load_anchor_bank(trace_dir: Path) -> torch.Tensor:
    """Build anchor dataset from legacy non-Striker dataset + promoted non-Striker matches."""
    # 1. Promoted non-Striker matches
    anchor_states = []
    for opp in ("tracker", "aggressive", "follow_ball"):
        pattern = f"promoted_anchor__{opp}__*.npz"
        for p in sorted(trace_dir.glob(pattern)):
            d = np.load(p)
            anchor_states.append(d["state"])

    # 2. Legacy non-Striker demonstration dataset (seed 22 train split)
    leg_npz = LEGACY_PID_DIR / "seed_22/dataset.npz"
    leg_col = json.loads((LEGACY_PID_DIR / "seed_22/collection.json").read_text())["matches"]
    with np.load(leg_npz) as f:
        leg_raw = {k: f[k] for k in f.files}

    opp_map = {m["index"]: m["opponent"] for m in leg_col}
    split_map = {m["index"]: m["split"] for m in leg_col}
    side_map = {m["index"]: m["side"] for m in leg_col}

    opponents = np.array([opp_map[i] for i in leg_raw["match"]])
    splits = np.array([split_map[i] for i in leg_raw["match"]])
    sides = np.array([side_map[i] for i in leg_raw["match"]], dtype=np.int32)

    keep = (opponents != "striker") & (splits == "train")
    leg_geom = compute_geometry_features_from_physics_batch(
        leg_raw["physics"][keep], leg_raw["params"][keep], sides[keep]
    )
    leg_states = np.concatenate([leg_raw["states"][keep][:, :30], leg_geom], axis=-1)
    anchor_states.append(leg_states)

    combined = np.concatenate(anchor_states, axis=0).astype(np.float32)
    return torch.as_tensor(combined)


def build_targeted_dataset(
    trace_paths: list[Path],
    kind: str,  # 'hybrid' or 'planner'
    frozen_net: DQNNetwork,
    rng: np.random.Generator,
) -> tuple[torch.Tensor, torch.Tensor, int]:
    """Extract targeted tiers, project targets if hybrid, and return (states, targets, fallback_count)."""
    xs: list[np.ndarray] = []
    vs: list[np.ndarray] = []

    for p in trace_paths:
        d = np.load(p)
        st = d["state"]
        val = d["values"]
        ga_flags = d["goal_against"]
        ga_actions = d["ga_flag"]
        hit_against = d["hit_against"]

        tier_mask, base_mask = filter_targeted_tiers(
            st, val, ga_flags, ga_actions, hit_against_flags=hit_against
        )

        xs.append(st[tier_mask])
        vs.append(val[tier_mask])

        # 20% contrastive base states
        base_indices = np.flatnonzero(base_mask)
        if len(base_indices) > 0:
            n_base = max(1, int(tier_mask.sum()) // 5)
            n_sample = min(len(base_indices), n_base)
            chosen_base = rng.choice(base_indices, n_sample, replace=False)
            xs.append(st[chosen_base])
            vs.append(val[chosen_base])

    x_all = torch.as_tensor(np.concatenate(xs, axis=0), dtype=torch.float32)
    v_all = torch.as_tensor(np.concatenate(vs, axis=0), dtype=torch.float32)

    with torch.no_grad():
        frozen_lg = frozen_net(x_all)

    if kind == "hybrid":
        targets, fallbacks = project_planner_target(v_all, x_all, frozen_lg, temperature=0.03)
    else:
        soft = torch.softmax(v_all / 0.03, dim=1)
        chosen = F.one_hot(v_all.argmax(dim=1), 9).to(soft.dtype)
        targets = 0.9 * soft + 0.1 * chosen
        fallbacks = 0

    return x_all, targets, fallbacks


def train_candidate(
    candidate_name: str,
    kind: str,
    seed: int,
    targeted_paths: list[Path],
    anchor_tensor: torch.Tensor,
    n_updates: int,
    out_dir: Path,
) -> dict[str, Any]:
    """Train candidate checkpoint starting from dqn_v7_promoted with CE + KL anchor."""
    torch.set_num_threads(4)
    torch.manual_seed(seed)
    rng = np.random.default_rng(seed)

    # Base agent starting from promoted
    agent = DQNAgent()
    agent.load_model(str(PROMOTED_PATH))
    q_net = agent.q_network
    frozen_net = copy.deepcopy(q_net).eval()
    for p in frozen_net.parameters():
        p.requires_grad_(False)

    print(
        f"[{candidate_name}] Extracting targeted tiers from {len(targeted_paths)} traces...",
        flush=True,
    )
    x_targeted, targets, fallbacks = build_targeted_dataset(targeted_paths, kind, frozen_net, rng)
    print(
        f"[{candidate_name}] Retained {len(x_targeted):,} targeted rows (fallbacks: {fallbacks})",
        flush=True,
    )

    # Fresh Adam optimizer
    optimizer = torch.optim.Adam(q_net.parameters(), lr=SETTINGS["learning_rate"])
    opt_params = {p for g in optimizer.param_groups for p in g["params"]}
    assert {p for p in q_net.parameters() if p.requires_grad} == opt_params
    assert q_net.geometry_context_weight in opt_params

    b_t = SETTINGS["batch_targeted"]
    b_a = SETTINGS["batch_anchor"]
    lam = SETTINGS["anchor_lambda"]

    saved_checkpoints: dict[int, str] = {}
    history = []

    print(
        f"[{candidate_name}] Beginning {n_updates} updates (lr={SETTINGS['learning_rate']}, lam={lam})...",
        flush=True,
    )
    for u in range(1, n_updates + 1):
        idx_t = torch.randint(len(x_targeted), (b_t,))
        idx_a = torch.randint(len(anchor_tensor), (b_a,))

        batch_xt = x_targeted[idx_t]
        batch_yt = targets[idx_t]
        batch_xa = anchor_tensor[idx_a]

        ce = -(batch_yt * F.log_softmax(q_net(batch_xt), dim=1)).sum(dim=1).mean()

        with torch.no_grad():
            teacher_p = F.softmax(frozen_net(batch_xa), dim=1)
        student_logp = F.log_softmax(q_net(batch_xa), dim=1)
        kl = (teacher_p * (teacher_p.clamp_min(1e-9).log() - student_logp)).sum(dim=1).mean()

        loss = ce + lam * kl
        optimizer.zero_grad()
        loss.backward()
        torch.nn.utils.clip_grad_norm_(q_net.parameters(), SETTINGS["gradient_clip"])
        optimizer.step()

        if u in SETTINGS["checkpoint_updates"]:
            ckpt_path = out_dir / f"{candidate_name}_u{u}.pth"
            # Update target network and save through DQNAgent
            agent.update_target_network()
            agent.save_model(str(ckpt_path))
            saved_checkpoints[u] = str(ckpt_path)
            history.append(
                {
                    "update": u,
                    "ce_loss": float(ce.detach().cpu()),
                    "kl_loss": float(kl.detach().cpu()),
                }
            )
            print(
                f"[{candidate_name}] Update {u}/{n_updates}: CE={ce:.4f}, KL={kl:.4f} -> Saved {ckpt_path.name}",
                flush=True,
            )

    return {
        "candidate": candidate_name,
        "kind": kind,
        "seed": seed,
        "n_targeted": len(x_targeted),
        "fallbacks": fallbacks,
        "checkpoints": saved_checkpoints,
        "history": history,
    }


def main() -> None:
    parser = argparse.ArgumentParser(
        description="Option A: Curvature & Bounce-Targeted DAgger (Round 3)"
    )
    parser.add_argument(
        "--skip-baseline", action="store_true", help="Skip Step 0 baseline evaluation if cached"
    )
    parser.add_argument(
        "--workers", type=int, default=min(8, mp.cpu_count()), help="Multiprocessing workers"
    )
    args = parser.parse_args()

    assert_splits(SETTINGS)
    ROOT.mkdir(parents=True, exist_ok=True)
    traces_dir = ROOT / "traces"
    traces_dir.mkdir(parents=True, exist_ok=True)
    val_dir = ROOT / "validation"
    val_dir.mkdir(parents=True, exist_ok=True)
    heldout_dir = ROOT / "heldout"
    heldout_dir.mkdir(parents=True, exist_ok=True)

    print("=====================================================================")
    print(" Option A: Curvature & Bounce-Targeted DAgger (Round 3) Pipeline")
    print("=====================================================================")

    # -------------------------------------------------------------------------
    # Step 0: Baseline Empirical Benchmarks (Bit-exact promoted baseline)
    # -------------------------------------------------------------------------
    baseline_cache = ROOT / "baseline_benchmarks.json"
    if baseline_cache.exists() and args.skip_baseline:
        print(f"\n[Step 0] Loading cached baseline benchmarks from {baseline_cache}...")
        baseline = json.loads(baseline_cache.read_text())
    else:
        print("\n[Step 0] Benchmarking dqn_v7_promoted.pth across all evaluation panels...")
        base_agent = DQNAgent()
        base_agent.load_model(str(PROMOTED_PATH))

        # 1. Validation Non-Striker
        val_ns_cfg = CompetitiveConfig(
            opponents=tuple(SETTINGS["non_striker_opponents"]),
            seeds=tuple(SETTINGS["val_non_striker_seeds"]),
            player_sides=tuple(SETTINGS["sides"]),
            bonus_modes=tuple(SETTINGS["bonuses"]),
            max_steps=6000,
            max_score=3,
        )
        print("  Evaluating Baseline on Validation Non-Striker Panel (48 matches)...")
        res_val_ns = evaluate_agent_instrumented(
            PROMOTED_PATH, val_ns_cfg, "validation", workers=args.workers
        )

        # 2. Validation Striker 18k
        val_str_cfg = CompetitiveConfig(
            opponents=("striker",),
            seeds=tuple(SETTINGS["val_striker_seeds"]),
            player_sides=tuple(SETTINGS["sides"]),
            bonus_modes=tuple(SETTINGS["bonuses"]),
            max_steps=18000,
            max_score=3,
        )
        print("  Evaluating Baseline on Validation Striker 18k Panel (64 matches)...")
        res_val_str = evaluate_agent_instrumented(
            PROMOTED_PATH, val_str_cfg, "validation", workers=args.workers
        )

        # 3. Heldout Standard Panel 6k
        held_cfg = CompetitiveConfig(
            opponents=tuple(SETTINGS["opponents"]),
            held_out_seeds=tuple(SETTINGS["heldout_seeds"]),
            player_sides=tuple(SETTINGS["sides"]),
            bonus_modes=tuple(SETTINGS["bonuses"]),
            max_steps=6000,
            max_score=3,
        )
        print("  Evaluating Baseline on Heldout Standard Panel (128 matches)...")
        res_held = evaluate_agent_instrumented(
            PROMOTED_PATH, held_cfg, "held_out", workers=args.workers
        )

        # 4. Tracker Long-Game Screen 18k
        track_cfg = CompetitiveConfig(
            opponents=("tracker",),
            held_out_seeds=tuple(SETTINGS["long_screen_seeds"]),
            player_sides=tuple(SETTINGS["sides"]),
            bonus_modes=tuple(SETTINGS["bonuses"]),
            max_steps=18000,
            max_score=3,
        )
        print("  Evaluating Baseline on Tracker Long-Game Screen (64 matches)...")
        res_track = evaluate_agent_instrumented(
            PROMOTED_PATH, track_cfg, "held_out", workers=args.workers
        )

        # 5. Striker Long-Game Screen 18k
        striker_cfg = CompetitiveConfig(
            opponents=("striker",),
            held_out_seeds=tuple(SETTINGS["long_screen_seeds"]),
            player_sides=tuple(SETTINGS["sides"]),
            bonus_modes=tuple(SETTINGS["bonuses"]),
            max_steps=18000,
            max_score=3,
        )
        print("  Evaluating Baseline on Striker Long-Game Screen (64 matches)...")
        res_striker = evaluate_agent_instrumented(
            PROMOTED_PATH, striker_cfg, "held_out", workers=args.workers
        )

        baseline = {
            "validation_non_striker": res_val_ns,
            "validation_striker": res_val_str,
            "heldout_standard": res_held,
            "tracker_long_screen": res_track,
            "striker_long_screen": res_striker,
        }
        write_json(baseline_cache, baseline)

    print(
        f"  Baseline Validation Non-Striker Losses: {baseline['validation_non_striker']['losses']}/48"
    )
    print(
        f"  Baseline Validation Striker Share: {baseline['validation_striker']['point_share']:.2%}, GA/1k: {baseline['validation_striker']['ga_per_1k']:.4f}, Concede/Hit: {baseline['validation_striker']['concede_per_hit']:.2%}"
    )
    print(
        f"  Baseline Striker Screen Share: {baseline['striker_long_screen']['point_share']:.2%}, GA/1k: {baseline['striker_long_screen']['ga_per_1k']:.4f}, Concede/Hit: {baseline['striker_long_screen']['concede_per_hit']:.2%}"
    )

    # -------------------------------------------------------------------------
    # Step 1: Stage 1 Trace Collection
    # -------------------------------------------------------------------------
    print("\n[Step 1] Collecting Stage 1 Match Traces...")
    collect_tasks = []
    # Striker matches: promoted and shield on stage 1 seeds
    for s in SETTINGS["stage1_collection_seeds"]:
        for side in SETTINGS["sides"]:
            for b in SETTINGS["bonuses"]:
                collect_tasks.append(
                    ("promoted", str(PROMOTED_PATH), "striker", s, side, b, 18000, True, traces_dir)
                )
                collect_tasks.append(
                    ("shield", str(PROMOTED_PATH), "striker", s, side, b, 18000, True, traces_dir)
                )

    # Anchor matches: promoted on anchor seeds against non-Striker
    for opp in SETTINGS["non_striker_opponents"]:
        for s in SETTINGS["anchor_collection_seeds"]:
            for side in SETTINGS["sides"]:
                for b in SETTINGS["bonuses"]:
                    collect_tasks.append(
                        (
                            "promoted_anchor",
                            str(PROMOTED_PATH),
                            opp,
                            s,
                            side,
                            b,
                            6000,
                            False,
                            traces_dir,
                        )
                    )

    print(f"  Executing {len(collect_tasks)} collection tasks across {args.workers} workers...")
    with mp.Pool(args.workers) as pool:
        for i, _path in enumerate(pool.imap_unordered(collect_match, collect_tasks)):
            if (i + 1) % 16 == 0 or (i + 1) == len(collect_tasks):
                print(f"    Completed trace {i + 1}/{len(collect_tasks)}", flush=True)

    # Build anchor tensor
    print("  Assembling anchor tensor...")
    anchor_tensor = load_anchor_bank(traces_dir)
    print(f"  Anchor tensor assembled: {len(anchor_tensor):,} frames")

    # -------------------------------------------------------------------------
    # Step 2: Stage 1 Training
    # -------------------------------------------------------------------------
    print("\n[Step 2] Training Stage 1 Candidates (Arm C & Arm A)...")
    s1_promoted_paths = sorted(traces_dir.glob("promoted__striker__*.npz"))
    s1_shield_paths = sorted(traces_dir.glob("shield__striker__*.npz"))
    s1_hybrid_paths = s1_promoted_paths + s1_shield_paths

    stage1_results: dict[str, Any] = {}
    for seed in SETTINGS["training_seeds"]:
        # Arm C (Hybrid)
        c_name = f"ArmC_Stage1_seed{seed}"
        res_c = train_candidate(
            c_name,
            kind="hybrid",
            seed=seed,
            targeted_paths=s1_hybrid_paths,
            anchor_tensor=anchor_tensor,
            n_updates=SETTINGS["stage1_updates"],
            out_dir=ROOT,
        )
        stage1_results[c_name] = res_c

        # Arm A (Comparator)
        a_name = f"ArmA_Stage1_seed{seed}"
        res_a = train_candidate(
            a_name,
            kind="planner",
            seed=seed,
            targeted_paths=s1_promoted_paths,
            anchor_tensor=anchor_tensor,
            n_updates=1500,  # Comparator matched to 1.5k pilot
            out_dir=ROOT,
        )
        stage1_results[a_name] = res_a

    # -------------------------------------------------------------------------
    # Step 3: Stage 1 Validation Evaluation & Selection
    # -------------------------------------------------------------------------
    print("\n[Step 3] Evaluating Stage 1 Checkpoints on Validation Panels...")
    val_ns_cfg = CompetitiveConfig(
        opponents=tuple(SETTINGS["non_striker_opponents"]),
        seeds=tuple(SETTINGS["val_non_striker_seeds"]),
        player_sides=tuple(SETTINGS["sides"]),
        bonus_modes=tuple(SETTINGS["bonuses"]),
        max_steps=6000,
        max_score=3,
    )
    val_str_cfg = CompetitiveConfig(
        opponents=("striker",),
        seeds=tuple(SETTINGS["val_striker_seeds"]),
        player_sides=tuple(SETTINGS["sides"]),
        bonus_modes=tuple(SETTINGS["bonuses"]),
        max_steps=18000,
        max_score=3,
    )

    stage1_evals: dict[str, dict[str, Any]] = {}
    best_stage1_per_seed: dict[int, str] = {}

    for seed in SETTINGS["training_seeds"]:
        c_name = f"ArmC_Stage1_seed{seed}"
        ckpts = stage1_results[c_name]["checkpoints"]
        seed_candidates = []

        for u, ckpt_p in ckpts.items():
            cand_key = f"{c_name}_u{u}"
            agent = DQNAgent()
            agent.load_model(ckpt_p)
            print(f"  Evaluating {cand_key} on Validation Non-Striker & Striker...")
            ns_res = evaluate_agent_instrumented(
                ckpt_p, val_ns_cfg, "validation", workers=args.workers
            )
            str_res = evaluate_agent_instrumented(
                ckpt_p, val_str_cfg, "validation", workers=args.workers
            )
            eval_record = {
                "candidate": cand_key,
                "checkpoint": ckpt_p,
                "seed": seed,
                "update": u,
                "non_striker_losses": ns_res["losses"],
                "striker_share": str_res["point_share"],
                "striker_gf_1k": str_res["gf_1k"],
                "striker_ga_1k": str_res["ga_1k"],
                "striker_concede_per_hit": str_res["concede_per_hit"],
                "ns_report": ns_res,
                "str_report": str_res,
            }
            stage1_evals[cand_key] = eval_record
            seed_candidates.append(eval_record)

        # Selection rules on Stage 1
        # 1. Filter: non-Striker losses <= baseline
        base_ns_losses = baseline["validation_non_striker"]["losses"]
        base_str_gf_1k = baseline["validation_striker"]["gf_1k"]
        surviving = [c for c in seed_candidates if c["non_striker_losses"] <= base_ns_losses]

        # 2. Anti-stalling filter: GF/1k >= baseline GF/1k
        anti_stalled = [c for c in surviving if c["striker_gf_1k"] >= base_str_gf_1k]
        pool_to_rank = anti_stalled if anti_stalled else surviving
        if not pool_to_rank:
            pool_to_rank = seed_candidates

        # 3. Rank by Striker Point Share (descending), breaking ties by lower concede per hit
        best_cand = max(
            pool_to_rank,
            key=lambda c: (c["striker_share"], -c["striker_concede_per_hit"], -c["striker_ga_1k"]),
        )
        best_stage1_per_seed[seed] = best_cand["checkpoint"]
        print(
            f"  [Seed {seed}] Best Stage 1 Candidate: {best_cand['candidate']} (Share: {best_cand['striker_share']:.2%}, GA/1k: {best_cand['striker_ga_1k']:.4f})"
        )

    # -------------------------------------------------------------------------
    # Step 4: Stage 2 Collection & Retraining
    # -------------------------------------------------------------------------
    print("\n[Step 4] Stage 2: Collecting Unshielded Traces with Best Stage 1 Checkpoints...")
    s2_collect_tasks = []
    for seed in SETTINGS["training_seeds"]:
        s1_best_path = best_stage1_per_seed[seed]
        for s in SETTINGS["stage2_collection_seeds"]:
            for side in SETTINGS["sides"]:
                for b in SETTINGS["bonuses"]:
                    s2_collect_tasks.append(
                        (
                            f"stage2_seed{seed}",
                            s1_best_path,
                            "striker",
                            s,
                            side,
                            b,
                            18000,
                            True,
                            traces_dir,
                        )
                    )

    print(
        f"  Executing {len(s2_collect_tasks)} Stage 2 collection tasks across {args.workers} workers..."
    )
    with mp.Pool(args.workers) as pool:
        for i, _path in enumerate(pool.imap_unordered(collect_match, s2_collect_tasks)):
            if (i + 1) % 16 == 0 or (i + 1) == len(s2_collect_tasks):
                print(f"    Completed trace {i + 1}/{len(s2_collect_tasks)}", flush=True)

    print("\n[Step 5] Stage 2: Retraining Arm C with Pooled Data...")
    stage2_results: dict[str, Any] = {}
    for seed in SETTINGS["training_seeds"]:
        s2_seed_paths = sorted(traces_dir.glob(f"stage2_seed{seed}__striker__*.npz"))
        pooled_paths = s1_hybrid_paths + s2_seed_paths
        c2_name = f"ArmC_Stage2_seed{seed}"
        res_c2 = train_candidate(
            c2_name,
            kind="hybrid",
            seed=seed,
            targeted_paths=pooled_paths,
            anchor_tensor=anchor_tensor,
            n_updates=SETTINGS["stage2_updates"],
            out_dir=ROOT,
        )
        stage2_results[c2_name] = res_c2

    # -------------------------------------------------------------------------
    # Step 6: Full Validation Evaluation & Primary Candidate Selection
    # -------------------------------------------------------------------------
    print("\n[Step 6] Final Validation Ranking across all Stage 1 & Stage 2 Checkpoints...")
    all_evals: dict[str, dict[str, Any]] = dict(stage1_evals)
    selected_per_seed: dict[int, dict[str, Any]] = {}

    for seed in SETTINGS["training_seeds"]:
        c2_name = f"ArmC_Stage2_seed{seed}"
        ckpts = stage2_results[c2_name]["checkpoints"]
        for u, ckpt_p in ckpts.items():
            cand_key = f"{c2_name}_u{u}"
            agent = DQNAgent()
            agent.load_model(ckpt_p)
            print(f"  Evaluating {cand_key} on Validation Non-Striker & Striker...")
            ns_res = evaluate_agent_instrumented(
                ckpt_p, val_ns_cfg, "validation", workers=args.workers
            )
            str_res = evaluate_agent_instrumented(
                ckpt_p, val_str_cfg, "validation", workers=args.workers
            )
            all_evals[cand_key] = {
                "candidate": cand_key,
                "checkpoint": ckpt_p,
                "seed": seed,
                "update": u,
                "stage": 2,
                "non_striker_losses": ns_res["losses"],
                "striker_share": str_res["point_share"],
                "striker_gf_1k": str_res["gf_1k"],
                "striker_ga_1k": str_res["ga_1k"],
                "striker_concede_per_hit": str_res["concede_per_hit"],
                "ns_report": ns_res,
                "str_report": str_res,
            }

        # Select overall best checkpoint for this seed from Stage 1 and Stage 2
        all_seed_cands = [
            c for c in all_evals.values() if c["seed"] == seed and c["candidate"].startswith("ArmC")
        ]
        base_ns_losses = baseline["validation_non_striker"]["losses"]
        base_str_gf_1k = baseline["validation_striker"]["gf_1k"]

        surviving = [c for c in all_seed_cands if c["non_striker_losses"] <= base_ns_losses]
        anti_stalled = [c for c in surviving if c["striker_gf_1k"] >= base_str_gf_1k]
        pool = anti_stalled if anti_stalled else surviving
        if not pool:
            pool = all_seed_cands

        best_cand = max(
            pool,
            key=lambda c: (c["striker_share"], -c["striker_concede_per_hit"], -c["striker_ga_1k"]),
        )
        selected_per_seed[seed] = best_cand
        print(
            f"  ==> Selected Primary Checkpoint for Seed {seed}: {best_cand['candidate']} (Share: {best_cand['striker_share']:.2%}, GA/1k: {best_cand['striker_ga_1k']:.4f}, Concede/Hit: {best_cand['striker_concede_per_hit']:.2%})"
        )

    # -------------------------------------------------------------------------
    # Step 7: Final Held-Out Panel Evaluation & Gate Checks
    # -------------------------------------------------------------------------
    print("\n[Step 7] Running Official Gate Evaluation on Fresh Disjoint Panels...")
    held_cfg = CompetitiveConfig(
        opponents=tuple(SETTINGS["opponents"]),
        held_out_seeds=tuple(SETTINGS["heldout_seeds"]),
        player_sides=tuple(SETTINGS["sides"]),
        bonus_modes=tuple(SETTINGS["bonuses"]),
        max_steps=6000,
        max_score=3,
    )
    track_cfg = CompetitiveConfig(
        opponents=("tracker",),
        held_out_seeds=tuple(SETTINGS["long_screen_seeds"]),
        player_sides=tuple(SETTINGS["sides"]),
        bonus_modes=tuple(SETTINGS["bonuses"]),
        max_steps=18000,
        max_score=3,
    )
    striker_cfg = CompetitiveConfig(
        opponents=("striker",),
        held_out_seeds=tuple(SETTINGS["long_screen_seeds"]),
        player_sides=tuple(SETTINGS["sides"]),
        bonus_modes=tuple(SETTINGS["bonuses"]),
        max_steps=18000,
        max_score=3,
    )

    final_evaluations: dict[int, dict[str, Any]] = {}
    for seed in SETTINGS["training_seeds"]:
        best = selected_per_seed[seed]
        ckpt_path = best["checkpoint"]
        agent = DQNAgent()
        agent.load_model(ckpt_path)

        print(
            f"\n  [Seed {seed}] Evaluating {best['candidate']} on Heldout Standard Panel (128 matches)..."
        )
        held_res = evaluate_agent_instrumented(
            ckpt_path, held_cfg, "held_out", workers=args.workers
        )

        print(
            f"  [Seed {seed}] Evaluating {best['candidate']} on Tracker 18k Screen (64 matches)..."
        )
        track_res = evaluate_agent_instrumented(
            ckpt_path, track_cfg, "held_out", workers=args.workers
        )

        print(
            f"  [Seed {seed}] Evaluating {best['candidate']} on Striker 18k Screen (64 matches)..."
        )
        striker_res = evaluate_agent_instrumented(
            ckpt_path, striker_cfg, "held_out", workers=args.workers
        )

        # Also evaluate with runtime shield on as requested for GUI parity reporting
        print(f"  [Seed {seed}] Evaluating {best['candidate']} WITH SHIELD ON for reporting...")
        striker_shield_res = evaluate_agent_instrumented(
            ckpt_path, striker_cfg, "held_out", shield_on=True, workers=args.workers
        )

        final_evaluations[seed] = {
            "selected_candidate": best["candidate"],
            "checkpoint": str(ckpt_path),
            "heldout": held_res,
            "tracker_long": track_res,
            "striker_long": striker_res,
            "striker_shield_long": striker_shield_res,
        }

    # Evaluate Arm A comparator on Heldout and Screens for seed 22
    arma_ckpt = stage1_results["ArmA_Stage1_seed22"]["checkpoints"][1500]
    print("\n  Evaluating Arm A Comparator on Striker 18k Screen...")
    arma_striker_res = evaluate_agent_instrumented(
        arma_ckpt, striker_cfg, "held_out", workers=args.workers
    )
    arma_held_res = evaluate_agent_instrumented(
        arma_ckpt, held_cfg, "held_out", workers=args.workers
    )

    # -------------------------------------------------------------------------
    # Step 8: Gate Verification & Summary
    # -------------------------------------------------------------------------
    print("\n=====================================================================")
    print(" Official Gate Verification & Gating Decisions")
    print("=====================================================================")

    base_track = baseline["tracker_long_screen"]
    base_str = baseline["striker_long_screen"]

    # Gate 1: Non-Regression on Held-Out Panel
    # <= 3 non-Striker losses per seed (out of 96 non-Striker matches), <= 11/288 pooled, Tracker <= 1
    g1_passes = {}
    for seed in SETTINGS["training_seeds"]:
        m_list = final_evaluations[seed]["heldout"]["matches"]
        ns_losses = sum(
            1
            for m in m_list
            if m["opponent"] != "striker" and not m["agent_won"] and not m["timeout"]
        )
        track_losses = sum(
            1
            for m in m_list
            if m["opponent"] == "tracker" and not m["agent_won"] and not m["timeout"]
        )
        g1_pass = (ns_losses <= 3) and (track_losses <= 1)
        g1_passes[seed] = (g1_pass, ns_losses, track_losses)

    pooled_ns_losses = sum(g1_passes[s][1] for s in SETTINGS["training_seeds"])
    gate1_pass = all(g1_passes[s][0] for s in SETTINGS["training_seeds"]) and (
        pooled_ns_losses <= 11
    )

    # Gate 2: Tracker 18k Screen
    # <= 4/64 losses per seed, win rate >= 85%, GF >= 90% of paired baseline
    g2_passes = {}
    for seed in SETTINGS["training_seeds"]:
        tr = final_evaluations[seed]["tracker_long"]
        t_losses = tr["losses"]
        t_win_rate = float(tr["wins"] / max(1, tr["n"]))
        t_gf_ratio = float(tr["goals_for"] / max(1, base_track["goals_for"]))
        g2_pass = (t_losses <= 4) and (t_win_rate >= 0.85) and (t_gf_ratio >= 0.90)
        g2_passes[seed] = (g2_pass, t_losses, t_win_rate, t_gf_ratio)

    gate2_pass = all(g2_passes[s][0] for s in SETTINGS["training_seeds"])

    # Gate 3: Striker 18k Screen
    # Pooled share >= 25%, each seed >= 15%, GA/1k < paired baseline, concede per hit <= 2.5%
    g3_passes = {}
    pooled_str_gf = sum(
        final_evaluations[s]["striker_long"]["goals_for"] for s in SETTINGS["training_seeds"]
    )
    pooled_str_ga = sum(
        final_evaluations[s]["striker_long"]["goals_against"] for s in SETTINGS["training_seeds"]
    )
    pooled_str_share = float(pooled_str_gf / max(1, pooled_str_gf + pooled_str_ga))
    pooled_str_steps = sum(
        final_evaluations[s]["striker_long"]["steps"] for s in SETTINGS["training_seeds"]
    )
    pooled_str_ga_1k = float(1000.0 * pooled_str_ga / max(1, pooled_str_steps))
    pooled_opp_hits = sum(
        final_evaluations[s]["striker_long"]["opp_paddle_hits"] for s in SETTINGS["training_seeds"]
    )
    pooled_concede_per_hit = compute_concession_per_hit(pooled_str_ga, pooled_opp_hits)

    for seed in SETTINGS["training_seeds"]:
        sr = final_evaluations[seed]["striker_long"]
        s_pass = (
            (sr["point_share"] >= 0.15)
            and (sr["ga_per_1k"] < base_str["ga_per_1k"])
            and (sr["concede_per_hit"] <= 0.025)
        )
        g3_passes[seed] = (s_pass, sr["point_share"], sr["ga_per_1k"], sr["concede_per_hit"])

    gate3_pass = (
        (pooled_str_share >= 0.25)
        and (pooled_str_ga_1k < base_str["ga_per_1k"])
        and (pooled_concede_per_hit <= 0.025)
        and all(g3_passes[s][0] for s in SETTINGS["training_seeds"])
    )

    # Gate 4: Anti-Stalling
    # Striker GF/1k >= baseline GF/1k for every seed
    g4_passes = {}
    for seed in SETTINGS["training_seeds"]:
        sr = final_evaluations[seed]["striker_long"]
        g4_pass = sr["gf_per_1k"] >= base_str["gf_per_1k"]
        g4_passes[seed] = (g4_pass, sr["gf_per_1k"], base_str["gf_per_1k"])

    gate4_pass = all(g4_passes[s][0] for s in SETTINGS["training_seeds"])

    all_gates_passed = gate1_pass and gate2_pass and gate3_pass and gate4_pass

    print(f"\nGate 1 (Non-Regression): {'PASS' if gate1_pass else 'FAIL'}")
    for s in SETTINGS["training_seeds"]:
        print(
            f"  Seed {s}: Non-Striker Losses = {g1_passes[s][1]}/96 (<=3), Tracker = {g1_passes[s][2]}/32 (<=1)"
        )
    print(f"  Pooled Non-Striker Losses: {pooled_ns_losses}/288 (<=11)")

    print(f"\nGate 2 (Tracker 18k): {'PASS' if gate2_pass else 'FAIL'}")
    for s in SETTINGS["training_seeds"]:
        print(
            f"  Seed {s}: Losses = {g2_passes[s][1]}/64 (<=4), WinRate = {g2_passes[s][2]:.1%} (>=85%), GF/BaseGF = {g2_passes[s][3]:.2f} (>=0.90)"
        )

    print(f"\nGate 3 (Striker 18k): {'PASS' if gate3_pass else 'FAIL'}")
    for s in SETTINGS["training_seeds"]:
        print(
            f"  Seed {s}: Share = {g3_passes[s][1]:.2%} (>=15%), GA/1k = {g3_passes[s][2]:.4f} (<{base_str['ga_per_1k']:.4f}), Concede/Hit = {g3_passes[s][3]:.2%} (<=2.5%)"
        )
    print(
        f"  Pooled Striker: Share = {pooled_str_share:.2%} (>=25%), GA/1k = {pooled_str_ga_1k:.4f} (<{base_str['ga_per_1k']:.4f}), Concede/Hit = {pooled_concede_per_hit:.2%} (<=2.5%)"
    )

    print(f"\nGate 4 (Anti-Stalling): {'PASS' if gate4_pass else 'FAIL'}")
    for s in SETTINGS["training_seeds"]:
        print(f"  Seed {s}: GF/1k = {g4_passes[s][1]:.4f} (>= {g4_passes[s][2]:.4f})")

    # Promotion Decision
    best_validation_seed = max(
        SETTINGS["training_seeds"],
        key=lambda s: (
            selected_per_seed[s]["striker_share"],
            -selected_per_seed[s]["striker_ga_1k"],
        ),
    )
    promoted_cand = selected_per_seed[best_validation_seed]

    if all_gates_passed:
        promotion_decision = "PROMOTED"
        promoted_target_path = ROOT / "dqn_v7_promoted.pth"
        shutil.copy2(promoted_cand["checkpoint"], promoted_target_path)
        print(
            f"\n🏆 PROMOTION DECISION: ALL GATES PASSED! Checkpoint {promoted_cand['candidate']} PROMOTED -> {promoted_target_path}"
        )
    else:
        promotion_decision = "DO NOT PROMOTE"
        print(
            "\n⛔ PROMOTION DECISION: DO NOT PROMOTE (models/dqn_v7_promoted.pth remains production champion)"
        )

    # Save final report JSON
    final_report = {
        "schema": SCHEMA,
        "promotion_decision": promotion_decision,
        "all_gates_passed": all_gates_passed,
        "best_validation_seed": best_validation_seed,
        "promoted_candidate": promoted_cand,
        "gate_results": {
            "gate1_non_regression": {
                "pass": gate1_pass,
                "details": {s: g1_passes[s] for s in SETTINGS["training_seeds"]},
                "pooled_losses": pooled_ns_losses,
            },
            "gate2_tracker_long": {
                "pass": gate2_pass,
                "details": {s: g2_passes[s] for s in SETTINGS["training_seeds"]},
            },
            "gate3_striker_long": {
                "pass": gate3_pass,
                "pooled_share": pooled_str_share,
                "pooled_ga_1k": pooled_str_ga_1k,
                "pooled_concede_per_hit": pooled_concede_per_hit,
                "details": {s: g3_passes[s] for s in SETTINGS["training_seeds"]},
            },
            "gate4_anti_stalling": {
                "pass": gate4_pass,
                "details": {s: g4_passes[s] for s in SETTINGS["training_seeds"]},
            },
        },
        "baseline_benchmarks": baseline,
        "selected_candidates_per_seed": selected_per_seed,
        "final_evaluations": final_evaluations,
        "arma_comparator": {
            "striker_long": arma_striker_res,
            "heldout": arma_held_res,
        },
    }
    report_path = ROOT / "curvature_dagger_round3_report.json"
    write_json(report_path, final_report)
    print(f"\nFull report JSON saved to: {report_path}")


if __name__ == "__main__":
    main()
