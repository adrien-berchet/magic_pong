"""Matched frame-budget comparison of legacy and corrected training schedules.

Launch one process per seed/variant. Both branches use identical starting
weights, fresh optimizer/replay, balanced side/bonus frame quotas and opponents.
Only schedule/handoff behavior differs. Regression stopping is disabled so each
branch receives the same fixed budget; held-out evaluation remains untouched.
"""

from __future__ import annotations

import argparse
from pathlib import Path

from magic_pong.ai.scheduled_training import FramePipelineConfig
from magic_pong.ai.scheduled_training import run_frame_pipeline


def main(argv: list[str] | None = None) -> int:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--initial-checkpoint", type=Path)
    parser.add_argument("--output-dir", type=Path, required=True)
    parser.add_argument("--seed", type=int, choices=(11, 22, 33), required=True)
    parser.add_argument("--variant", choices=("legacy", "corrected"), required=True)
    parser.add_argument("--resume", action="store_true")
    args = parser.parse_args(argv)
    if not args.resume and args.initial_checkpoint is None:
        parser.error("--initial-checkpoint is required for a new pilot branch")
    config = FramePipelineConfig(
        phase_frames=(250000, 250000, 500000),
        seed=args.seed,
        training_scores=(1,),
        validation_interval=250000,
        legacy_control=args.variant == "legacy",
        regression_patience=0,
    )
    directory = args.output_dir / f"seed_{args.seed}" / args.variant
    result = run_frame_pipeline(
        config,
        directory,
        initial_checkpoint=None if args.resume else args.initial_checkpoint,
        resume=directory / "latest.training.pt" if args.resume else None,
    )
    print(f"{args.seed} {args.variant}: {result['status']}", flush=True)
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
