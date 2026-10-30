"""Option 1: Vector RL Fine-Tuning of models/dqn_v7_promoted.pth.

Deliberated with Claude CLI:
- Opponent Mixture: 40% Striker, 25% Tracker, 10% Aggressive, 10% Self-play, 10% Planner, 5% Follow-ball
- Discounting & Reward: gamma = 0.997, hit_reward = 0.0 (prevents stalling incentive)
- Target Schedule: Hard copy every 500,000 frames (~3.9k updates)
- Selective Anchor: KL divergence penalty (lambda = 0.1) on raw advantage logits against frozen
  promoted teacher on non-Striker / non-Planner transitions
- Architecture: Dueling DQN with analytical fold-on-save
- Training: Phase 1 (500k frames, lr 2e-4) value calibration + Phase 2 (1M frames, lr 2e-5) fine-tuning
- Candidate Selection: Validation seeds (750100..750115) with 18k-frame Striker matches,
  filtered by non-Striker losses <= baseline, ranked primarily by Striker point share
- Non-Regression Gate: Held-out evaluation across 3 RL seeds (11, 22, 33).
"""

from __future__ import annotations

import json
import shutil
import time
from pathlib import Path
from typing import Any

import torch

from magic_pong.ai.competitive import CompetitiveConfig
from magic_pong.ai.competitive import evaluate_competitive_agent
from magic_pong.ai.models.dqn_ai import DQNAgent
from magic_pong.ai.vector_training import VectorDQNTrainer
from magic_pong.ai.vector_training import VectorTrainingConfig

ROOT = Path("models/vector_rl_promoted_finetuning_2026-10-29")
BASE_CHECKPOINT = Path("models/dqn_v7_promoted.pth")

SETTINGS: dict[str, Any] = {
    "training_seeds": [11, 22, 33],
    "validation_non_striker_seeds": [750100, 750101, 750102, 750103],
    "validation_striker_seeds": list(range(750100, 750116)),
    "heldout_seeds": list(range(760100, 760108)),
    "scoring_seeds": list(range(770100, 770116)),
    "sides": [1, 2],
    "bonuses": [False, True],
    "total_frames": 1_500_000,
    "phase1_frames": 500_000,
    "phase1_lr": 2e-4,
    "phase2_lr": 2e-5,
    "warmup_updates": 2000,
    "target_update_frames": 500_000,
    "gamma": 0.997,
    "hit_reward": 0.0,
    "anchor_lambda": 0.1,
    "epsilon_start": 0.05,
    "epsilon_end": 0.01,
    "epsilon_decay_frames": 1_000_000,
    "batch_size": 256,
    "replay_capacity": 250_000,
    "min_replay": 10_000,
    "n_envs": 128,
    "action_repeat": 4,
    "warm_start_tau": 0.03,
    "checkpoint_interval_frames": 250_000,
    "opponent_mix": {
        "striker": 4.0,
        "tracker": 2.5,
        "aggressive": 1.0,
        "self": 1.0,
        "planner": 1.0,
        "follow_ball": 0.5,
    },
}


def compute_scoring_metrics(
    matches: list[dict[str, Any]], opponent: str | None = None
) -> dict[str, Any]:
    filtered = [m for m in matches if opponent is None or m.get("opponent") == opponent]
    gf = sum(m.get("goals_for", 0) for m in filtered)
    ga = sum(m.get("goals_against", 0) for m in filtered)
    steps = sum(m.get("steps", 0) for m in filtered)
    n = len(filtered)
    wins = sum(m.get("agent_won", False) for m in filtered)
    timeouts = sum(m.get("timeout", False) for m in filtered)
    losses = n - wins - timeouts
    gf_1k = 1000.0 * gf / max(1, steps)
    ga_1k = 1000.0 * ga / max(1, steps)
    point_share = float(gf / max(1, (gf + ga)))
    return {
        "matches": n,
        "wins": wins,
        "losses": losses,
        "timeouts": timeouts,
        "goals_for": gf,
        "goals_against": ga,
        "steps": steps,
        "goals_for_per_1000_frames": gf_1k,
        "goals_against_per_1000_frames": ga_1k,
        "point_share": point_share,
    }


def evaluate_checkpoint(
    checkpoint_path: Path, cfg: CompetitiveConfig, split: str = "held_out"
) -> dict[str, Any]:
    agent = DQNAgent(state_size=46, hidden_layers=[256, 128], geometry_context=True)
    agent.load_model(str(checkpoint_path))
    res = evaluate_competitive_agent(agent, cfg, split=split)
    matches_dict = [
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
        }
        for m in res.matches
    ]
    rates: dict[str, Any] = {"total": compute_scoring_metrics(matches_dict)}
    for opp in cfg.opponents:
        rates[opp] = compute_scoring_metrics(matches_dict, opp)
    return {
        "matches": matches_dict,
        "scoring_rates": rates,
    }


def run_validation_evaluation(checkpoint_path: Path) -> dict[str, Any]:
    """Run validation evaluation: non-Striker (6k frames) + Striker (18k frames)."""
    cache_path = checkpoint_path.parent / f"{checkpoint_path.stem}_val_eval.json"
    if cache_path.exists():
        try:
            return json.loads(cache_path.read_text())
        except Exception:
            pass

    # 1. Non-Striker validation
    non_striker_cfg = CompetitiveConfig(
        opponents=("follow_ball", "aggressive", "tracker"),
        seeds=tuple(SETTINGS["validation_non_striker_seeds"]),
        player_sides=tuple(SETTINGS["sides"]),
        bonus_modes=tuple(SETTINGS["bonuses"]),
        max_steps=6000,
        max_score=3,
    )
    non_striker_eval = evaluate_checkpoint(checkpoint_path, non_striker_cfg, split="validation")

    # 2. Striker validation (18k frames)
    striker_cfg = CompetitiveConfig(
        opponents=("striker",),
        seeds=tuple(SETTINGS["validation_striker_seeds"]),
        player_sides=tuple(SETTINGS["sides"]),
        bonus_modes=tuple(SETTINGS["bonuses"]),
        max_steps=18000,
        max_score=3,
    )
    striker_eval = evaluate_checkpoint(checkpoint_path, striker_cfg, split="validation")

    non_striker_losses = non_striker_eval["scoring_rates"]["total"]["losses"]
    striker_rates = striker_eval["scoring_rates"]["striker"]

    res = {
        "checkpoint": str(checkpoint_path),
        "non_striker_losses": non_striker_losses,
        "striker_gf": striker_rates["goals_for"],
        "striker_ga": striker_rates["goals_against"],
        "striker_point_share": striker_rates["point_share"],
        "striker_ga_per_1k": striker_rates["goals_against_per_1000_frames"],
        "non_striker_eval": non_striker_eval,
        "striker_eval": striker_eval,
    }
    cache_path.write_text(json.dumps(res, indent=2))
    return res


def train_seed(seed: int, base_checkpoint: Path, output_dir: Path) -> Path:
    output_dir.mkdir(parents=True, exist_ok=True)
    final_model_path = output_dir / "final.pth"
    if final_model_path.exists():
        print(f"[Seed {seed}] Already trained at {final_model_path}")
        return final_model_path

    print("\n==================================================================")
    print(f" Starting training: seed {seed} with Dueling + Non-Striker Policy Anchor")
    print(f" Base checkpoint: {base_checkpoint}")
    print("==================================================================")

    config = VectorTrainingConfig(
        total_frames=SETTINGS["total_frames"],
        n_envs=SETTINGS["n_envs"],
        action_repeat=SETTINGS["action_repeat"],
        batch_size=SETTINGS["batch_size"],
        replay_capacity=SETTINGS["replay_capacity"],
        min_replay=SETTINGS["min_replay"],
        target_update_frames=SETTINGS["target_update_frames"],
        epsilon_start=SETTINGS["epsilon_start"],
        epsilon_end=SETTINGS["epsilon_end"],
        epsilon_decay_frames=SETTINGS["epsilon_decay_frames"],
        hidden_layers=(256, 128),
        geometry_context=True,
        terminal_on_goal=True,
        gamma=SETTINGS["gamma"],
        hit_reward=SETTINGS["hit_reward"],
        anchor_lambda=SETTINGS["anchor_lambda"],
        opponent_mix=SETTINGS["opponent_mix"],
        dueling=True,
        warm_start_tau=SETTINGS["warm_start_tau"],
        phase1_frames=SETTINGS["phase1_frames"],
        phase1_lr=SETTINGS["phase1_lr"],
        phase2_lr=SETTINGS["phase2_lr"],
        warmup_updates=SETTINGS["warmup_updates"],
        initial_checkpoint=str(base_checkpoint),
        seed=seed,
        device="cuda" if torch.cuda.is_available() else "cpu",
        checkpoint_interval_frames=SETTINGS["checkpoint_interval_frames"],
        eval_interval_frames=1_000_000_000,  # Evaluated separately via validation suite
    )

    t0 = time.time()
    trainer = VectorDQNTrainer(config, output_dir)
    summary = trainer.run(verbose=True, log_interval_frames=250_000)
    trainer.save("final.pth")
    elapsed = time.time() - t0
    print(
        f"[Seed {seed}] Finished {summary['frames']:,} frames in {elapsed:.1f}s "
        f"({summary.get('frames_per_second', 0):.0f} fps)"
    )
    return final_model_path


def main() -> None:
    ROOT.mkdir(parents=True, exist_ok=True)

    if not BASE_CHECKPOINT.exists():
        raise FileNotFoundError(f"Missing base checkpoint: {BASE_CHECKPOINT}")

    # =========================================================================
    # Step 0: Measure Baseline on the exact models/dqn_v7_promoted.pth
    # =========================================================================
    baseline_cache_file = ROOT / "baseline_measurements.json"
    if baseline_cache_file.exists():
        print(f"Loading baseline measurements from {baseline_cache_file}")
        baseline_data = json.loads(baseline_cache_file.read_text())
    else:
        print("\n==================================================================")
        print(" Step 0: Measuring exact baseline for models/dqn_v7_promoted.pth")
        print("==================================================================")

        # 1. Validation baseline
        print("Evaluating baseline on Validation Panel...")
        val_baseline = run_validation_evaluation(BASE_CHECKPOINT)

        # 2. Heldout baseline (6k frames)
        print("Evaluating baseline on Held-out Panel...")
        heldout_cfg = CompetitiveConfig(
            opponents=("follow_ball", "aggressive", "tracker", "striker"),
            held_out_seeds=tuple(SETTINGS["heldout_seeds"]),
            player_sides=tuple(SETTINGS["sides"]),
            bonus_modes=tuple(SETTINGS["bonuses"]),
            max_steps=6000,
            max_score=3,
        )
        heldout_baseline = evaluate_checkpoint(BASE_CHECKPOINT, heldout_cfg, split="held_out")

        # 3. Striker long-game screen (18k frames)
        print("Evaluating baseline on Striker Long-Game Screen...")
        striker_long_cfg = CompetitiveConfig(
            opponents=("striker",),
            held_out_seeds=tuple(SETTINGS["scoring_seeds"]),
            player_sides=tuple(SETTINGS["sides"]),
            bonus_modes=tuple(SETTINGS["bonuses"]),
            max_steps=18000,
            max_score=3,
        )
        striker_long_baseline = evaluate_checkpoint(
            BASE_CHECKPOINT, striker_long_cfg, split="held_out"
        )

        # 4. Tracker long-game screen (18k frames)
        print("Evaluating baseline on Tracker Long-Game Screen...")
        tracker_long_cfg = CompetitiveConfig(
            opponents=("tracker",),
            held_out_seeds=tuple(SETTINGS["scoring_seeds"]),
            player_sides=tuple(SETTINGS["sides"]),
            bonus_modes=tuple(SETTINGS["bonuses"]),
            max_steps=18000,
            max_score=3,
        )
        tracker_long_baseline = evaluate_checkpoint(
            BASE_CHECKPOINT, tracker_long_cfg, split="held_out"
        )

        baseline_data = {
            "validation": val_baseline,
            "heldout": heldout_baseline,
            "striker_long": striker_long_baseline,
            "tracker_long": tracker_long_baseline,
        }
        baseline_cache_file.write_text(json.dumps(baseline_data, indent=2))

    val_base_non_striker_losses = baseline_data["validation"]["non_striker_losses"]
    heldout_base_non_striker_losses = (
        baseline_data["heldout"]["scoring_rates"]["total"]["losses"]
        - baseline_data["heldout"]["scoring_rates"]["striker"]["losses"]
    )
    striker_base_rates = baseline_data["striker_long"]["scoring_rates"]["striker"]
    tracker_base_losses = baseline_data["tracker_long"]["scoring_rates"]["tracker"]["losses"]

    print("\n--- BASELINE EMPIRICAL BENCHMARKS ---")
    print(f"Validation Non-Striker Losses: {val_base_non_striker_losses}")
    print(
        f"Validation Striker Point Share: {baseline_data['validation']['striker_point_share']:.2%}"
    )
    print(f"Heldout Non-Striker Losses (per 96 matches): {heldout_base_non_striker_losses}")
    print(
        f"Striker Long Screen: GF={striker_base_rates['goals_for']}, GA={striker_base_rates['goals_against']}, "
        f"Point Share={striker_base_rates['point_share']:.2%}, GA/1k={striker_base_rates['goals_against_per_1000_frames']:.4f}"
    )
    print(f"Tracker Long Screen Losses (per 64 matches): {tracker_base_losses}")

    # =========================================================================
    # Step 1: Train all seeds and validate checkpoints
    # =========================================================================
    seed_results: dict[int, dict[str, Any]] = {}

    for seed in SETTINGS["training_seeds"]:
        seed_dir = ROOT / f"seed_{seed}"
        train_seed(seed, BASE_CHECKPOINT, seed_dir)

        # Discover all saved checkpoints for this seed
        ckpt_files = sorted(seed_dir.glob("checkpoint_*.pth"))
        final_pth = seed_dir / "final.pth"
        if final_pth.exists() and final_pth not in ckpt_files:
            ckpt_files.append(final_pth)

        print(f"\n[Seed {seed}] Evaluating {len(ckpt_files)} checkpoints on Validation Panel...")
        val_reports: list[dict[str, Any]] = []
        for ckpt in ckpt_files:
            val_rep = run_validation_evaluation(ckpt)
            val_reports.append(val_rep)
            print(
                f"  [{ckpt.name}] non_striker_loss={val_rep['non_striker_losses']} "
                f"(max {val_base_non_striker_losses}), "
                f"striker_point_share={val_rep['striker_point_share']:.2%}, "
                f"GA/1k={val_rep['striker_ga_per_1k']:.4f}"
            )

        # Rank candidate checkpoints for this seed:
        # Filter: non_striker_losses <= baseline
        # Primary: striker_point_share (descending)
        # Tie-break: striker_ga_per_1k (ascending)
        filtered_candidates = [
            r for r in val_reports if r["non_striker_losses"] <= val_base_non_striker_losses
        ]

        if not filtered_candidates:
            print(f"[Seed {seed}] WARNING: No candidates passed validation filter! Using fallback.")
            best_candidate = min(val_reports, key=lambda r: r["non_striker_losses"])
        else:
            best_candidate = max(
                filtered_candidates,
                key=lambda r: (r["striker_point_share"], -r["striker_ga_per_1k"]),
            )

        best_ckpt_path = Path(best_candidate["checkpoint"])
        print(
            f"[Seed {seed}] Selected Best Checkpoint: {best_ckpt_path.name} "
            f"(Point share: {best_candidate['striker_point_share']:.2%}, "
            f"GA/1k: {best_candidate['striker_ga_per_1k']:.4f})"
        )

        selected_dest = seed_dir / "best_val.pth"
        shutil.copy2(best_ckpt_path, selected_dest)

        seed_results[seed] = {
            "best_checkpoint": str(selected_dest),
            "original_checkpoint_name": best_ckpt_path.name,
            "validation_report": best_candidate,
            "all_validation_reports": val_reports,
        }

    # =========================================================================
    # Step 2: Full Held-Out Panel & Long-Game Screen Evaluations
    # =========================================================================
    print("\n==================================================================")
    print(" Step 2: Evaluating Best Validation Checkpoints on Held-out Panels")
    print("==================================================================")

    heldout_cfg = CompetitiveConfig(
        opponents=("follow_ball", "aggressive", "tracker", "striker"),
        held_out_seeds=tuple(SETTINGS["heldout_seeds"]),
        player_sides=tuple(SETTINGS["sides"]),
        bonus_modes=tuple(SETTINGS["bonuses"]),
        max_steps=6000,
        max_score=3,
    )
    striker_long_cfg = CompetitiveConfig(
        opponents=("striker",),
        held_out_seeds=tuple(SETTINGS["scoring_seeds"]),
        player_sides=tuple(SETTINGS["sides"]),
        bonus_modes=tuple(SETTINGS["bonuses"]),
        max_steps=18000,
        max_score=3,
    )
    tracker_long_cfg = CompetitiveConfig(
        opponents=("tracker",),
        held_out_seeds=tuple(SETTINGS["scoring_seeds"]),
        player_sides=tuple(SETTINGS["sides"]),
        bonus_modes=tuple(SETTINGS["bonuses"]),
        max_steps=18000,
        max_score=3,
    )

    for seed, data in seed_results.items():
        seed_dir = ROOT / f"seed_{seed}"
        ckpt_path = Path(data["best_checkpoint"])

        h_cache = seed_dir / "heldout_eval.json"
        if h_cache.exists():
            print(f"\n[Seed {seed}] Loading cached Heldout Standard Panel from {h_cache}...")
            h_eval = json.loads(h_cache.read_text())
        else:
            print(f"\n[Seed {seed}] Evaluating on Heldout Standard Panel...")
            h_eval = evaluate_checkpoint(ckpt_path, heldout_cfg, split="held_out")
            h_cache.write_text(json.dumps(h_eval, indent=2))

        s_cache = seed_dir / "striker_long_eval.json"
        if s_cache.exists():
            print(f"[Seed {seed}] Loading cached Striker Long-Game Screen from {s_cache}...")
            s_long_eval = json.loads(s_cache.read_text())
        else:
            print(f"[Seed {seed}] Evaluating on Striker Long-Game Screen...")
            s_long_eval = evaluate_checkpoint(ckpt_path, striker_long_cfg, split="held_out")
            s_cache.write_text(json.dumps(s_long_eval, indent=2))

        t_cache = seed_dir / "tracker_long_eval.json"
        if t_cache.exists():
            print(f"[Seed {seed}] Loading cached Tracker Long-Game Screen from {t_cache}...")
            t_long_eval = json.loads(t_cache.read_text())
        else:
            print(f"[Seed {seed}] Evaluating on Tracker Long-Game Screen...")
            t_long_eval = evaluate_checkpoint(ckpt_path, tracker_long_cfg, split="held_out")
            t_cache.write_text(json.dumps(t_long_eval, indent=2))

        data["heldout_eval"] = h_eval
        data["striker_long_eval"] = s_long_eval
        data["tracker_long_eval"] = t_long_eval

    # =========================================================================
    # Step 3: Non-Regression & Promotion Gate Verification
    # =========================================================================
    print("\n==================================================================")
    print(" Step 3: Verifying Non-Regression Gates")
    print("==================================================================")

    passed_per_seed: dict[int, bool] = {}
    pooled_non_striker_losses = 0
    pooled_striker_gf = 0
    pooled_striker_ga = 0
    pooled_striker_steps = 0

    for seed, data in seed_results.items():
        h_rates = data["heldout_eval"]["scoring_rates"]
        tracker_heldout_losses = h_rates["tracker"]["losses"]
        non_striker_losses = (
            h_rates["follow_ball"]["losses"]
            + h_rates["aggressive"]["losses"]
            + h_rates["tracker"]["losses"]
        )
        pooled_non_striker_losses += non_striker_losses

        t_long_losses = data["tracker_long_eval"]["scoring_rates"]["tracker"]["losses"]

        s_rates = data["striker_long_eval"]["scoring_rates"]["striker"]
        pooled_striker_gf += s_rates["goals_for"]
        pooled_striker_ga += s_rates["goals_against"]
        pooled_striker_steps += s_rates["steps"]

        # Per-seed checks:
        # 1. Tracker heldout losses <= 1 per 32 matches
        # 2. Tracker long-game losses <= 4 per 64 matches
        seed_pass = (tracker_heldout_losses <= 1) and (t_long_losses <= 4)
        passed_per_seed[seed] = seed_pass
        print(
            f"[Seed {seed}] Tracker heldout losses: {tracker_heldout_losses}/32 (gate <= 1), "
            f"Tracker long-game losses: {t_long_losses}/64 (gate <= 4) -> "
            f"{'PASS' if seed_pass else 'FAIL'}"
        )

    # Pooled checks:
    # 1. Pooled non-Striker losses <= 3 * baseline + 2
    # 2. Pooled Striker GA / 1k < baseline
    # 3. Pooled Striker point share >= baseline point share
    pooled_striker_ga_1k = 1000.0 * pooled_striker_ga / max(1, pooled_striker_steps)
    pooled_striker_share = pooled_striker_gf / max(1, (pooled_striker_gf + pooled_striker_ga))

    gate_pooled_non_striker_max = 3 * heldout_base_non_striker_losses + 2
    pass_pooled_non_striker = pooled_non_striker_losses <= gate_pooled_non_striker_max
    pass_striker_ga_1k = pooled_striker_ga_1k < striker_base_rates["goals_against_per_1000_frames"]
    pass_striker_share = pooled_striker_share >= striker_base_rates["point_share"]

    all_seeds_passed = all(passed_per_seed.values())
    promotion_eligible = (
        all_seeds_passed and pass_pooled_non_striker and pass_striker_ga_1k and pass_striker_share
    )

    print("\n--- POOLED GATE RESULTS ---")
    print(
        f"Pooled Non-Striker Losses: {pooled_non_striker_losses} / {gate_pooled_non_striker_max} -> "
        f"{'PASS' if pass_pooled_non_striker else 'FAIL'}"
    )
    print(
        f"Pooled Striker GA/1k: {pooled_striker_ga_1k:.4f} vs baseline {striker_base_rates['goals_against_per_1000_frames']:.4f} -> "
        f"{'PASS' if pass_striker_ga_1k else 'FAIL'}"
    )
    print(
        f"Pooled Striker Point Share: {pooled_striker_share:.2%} vs baseline {striker_base_rates['point_share']:.2%} -> "
        f"{'PASS' if pass_striker_share else 'FAIL'}"
    )
    print(f"All 3 Seeds Passed Per-Seed Gates: {all_seeds_passed}")
    print(
        f"\n>>> FINAL PROMOTION DECISION: {'PROMOTED' if promotion_eligible else 'DO NOT PROMOTE'} <<<"
    )

    best_seed = max(
        seed_results.keys(),
        key=lambda s: (
            seed_results[s]["validation_report"]["striker_point_share"],
            -seed_results[s]["validation_report"]["striker_ga_per_1k"],
        ),
    )
    print(f"Top Validation Ranked Seed: Seed {best_seed}")

    if promotion_eligible:
        promoted_dest = Path("models/dqn_v7_promoted_rl.pth")
        shutil.copy2(seed_results[best_seed]["best_checkpoint"], promoted_dest)
        print(f"Copied promoted checkpoint to {promoted_dest}")

    # Save summary report
    summary_report = {
        "timestamp": time.strftime("%Y-%m-%d %H:%M:%S"),
        "promotion_eligible": promotion_eligible,
        "best_seed": best_seed,
        "baseline": baseline_data,
        "seed_results": seed_results,
        "pooled": {
            "passed_per_seed": passed_per_seed,
            "all_seeds_passed": all_seeds_passed,
            "pooled_non_striker_losses": pooled_non_striker_losses,
            "gate_pooled_non_striker_max": gate_pooled_non_striker_max,
            "pass_pooled_non_striker": pass_pooled_non_striker,
            "pooled_striker_ga_1k": pooled_striker_ga_1k,
            "baseline_striker_ga_1k": striker_base_rates["goals_against_per_1000_frames"],
            "pass_striker_ga_1k": pass_striker_ga_1k,
            "pooled_striker_share": pooled_striker_share,
            "baseline_striker_share": striker_base_rates["point_share"],
            "pass_striker_share": pass_striker_share,
        },
    }

    report_path = ROOT / "vector_rl_finetuning_report.json"
    report_path.write_text(json.dumps(summary_report, indent=2))
    print(f"\nSaved complete report to {report_path}")


if __name__ == "__main__":
    main()
