#!/usr/bin/env python3
"""Generate visual debug artifacts for Magic Pong AI training."""

from __future__ import annotations

import argparse
from typing import Any

from magic_pong.ai.models.simple_ai import create_ai
from magic_pong.ai.visual_debug import capture_pretraining_run
from magic_pong.ai.visual_debug import capture_training_episode
from magic_pong.ai.visual_debug import capture_trajectory_debug
from magic_pong.ai.visual_debug import load_debug_json
from magic_pong.ai.visual_debug import render_pretraining_run_png
from magic_pong.ai.visual_debug import render_trajectory_png
from magic_pong.ai.visual_debug import replay_training_trace
from magic_pong.ai.visual_debug import save_debug_json


def _build_dqn_agent(model_path: str | None = None) -> Any:
    from magic_pong.ai.models.dqn_ai import DQNAgent
    from magic_pong.ai.models.dqn_checkpoint import EXPECTED_ACTION_SIZE
    from magic_pong.ai.models.dqn_checkpoint import EXPECTED_STATE_SIZE

    agent = DQNAgent(
        state_size=EXPECTED_STATE_SIZE,
        action_size=EXPECTED_ACTION_SIZE,
        include_prev_action_in_state=True,
        name="VisualDebugDQN",
    )
    if model_path:
        agent.load_model(model_path)
    agent.set_training_mode(False)
    agent.set_exploration_mode(False)
    return agent


def _add_common_pretraining_args(parser: argparse.ArgumentParser) -> None:
    parser.add_argument("--seed", type=int, default=None, help="Random seed for reproducibility")
    parser.add_argument("--player-id", type=int, choices=[1, 2], default=1, help="AI player side")
    parser.add_argument(
        "--json",
        type=str,
        default=None,
        help="Optional path to save the raw capture JSON next to the image",
    )


def _cmd_trajectory(args: argparse.Namespace) -> None:
    data = capture_trajectory_debug(player_id=args.player_id, seed=args.seed)
    image_path = render_trajectory_png(data, args.output)
    print(f"Trajectory image saved: {image_path}")
    if args.json:
        json_path = save_debug_json(data, args.json)
        print(f"Trajectory JSON saved: {json_path}")


def _cmd_pretraining_run(args: argparse.Namespace) -> None:
    agent = _build_dqn_agent(args.model)
    data = capture_pretraining_run(
        agent,
        player_id=args.player_id,
        seed=args.seed,
        explore=args.explore,
    )
    image_path = render_pretraining_run_png(data, args.output)
    print(f"Pretraining debug image saved: {image_path}")
    if args.json:
        json_path = save_debug_json(data, args.json)
        print(f"Pretraining debug JSON saved: {json_path}")


def _cmd_replay_training(args: argparse.Namespace) -> None:
    agent = _build_dqn_agent(args.model)
    opponent = create_ai(args.opponent, name=f"Opponent_{args.opponent}")
    trace = capture_training_episode(
        agent,
        opponent,
        player_id=args.player_id,
        seed=args.seed,
        max_steps=args.max_steps,
        initial_ball_direction=args.ball_direction,
        initial_ball_angle=args.ball_angle,
        train_agent=args.train_agent,
    )
    trace_path = save_debug_json(trace, args.trace)
    print(f"Training replay trace saved: {trace_path}")
    print(
        "Summary: "
        f"steps={trace['summary']['steps']} "
        f"winner={trace['summary']['winner']} "
        f"reward_frames={len(trace['reward_frames'])}"
    )
    if args.replay:
        replay_training_trace(trace, fps=args.fps)


def _cmd_replay_trace(args: argparse.Namespace) -> None:
    trace = load_debug_json(args.trace)
    replay_training_trace(trace, fps=args.fps)


def build_parser() -> argparse.ArgumentParser:
    parser = argparse.ArgumentParser(
        description="Generate visual debug outputs for Magic Pong AI training",
        formatter_class=argparse.ArgumentDefaultsHelpFormatter,
    )
    subparsers = parser.add_subparsers(dest="command", required=True)

    trajectory = subparsers.add_parser(
        "trajectory",
        help="Save a PNG of one random pretraining trajectory and optimal interception point",
    )
    _add_common_pretraining_args(trajectory)
    trajectory.add_argument(
        "--output",
        type=str,
        default="debug/trajectory.png",
        help="PNG output path",
    )
    trajectory.set_defaults(func=_cmd_trajectory)

    pretraining = subparsers.add_parser(
        "pretraining-run",
        help="Save a PNG of one pretraining model decision, action, and reward",
    )
    _add_common_pretraining_args(pretraining)
    pretraining.add_argument(
        "--model",
        type=str,
        default=None,
        help="Optional DQN checkpoint path; omitted uses a fresh untrained DQN",
    )
    pretraining.add_argument(
        "--explore",
        action="store_true",
        help="Use epsilon exploration for the selected action",
    )
    pretraining.add_argument(
        "--output",
        type=str,
        default="debug/pretraining_run.png",
        help="PNG output path",
    )
    pretraining.set_defaults(func=_cmd_pretraining_run)

    replay_training = subparsers.add_parser(
        "replay-training",
        help="Capture one training episode versus an opponent as a replay trace",
    )
    replay_training.add_argument("--seed", type=int, default=None)
    replay_training.add_argument("--player-id", type=int, choices=[1, 2], default=1)
    replay_training.add_argument(
        "--model",
        type=str,
        default=None,
        help="Optional DQN checkpoint path; omitted uses a fresh untrained DQN",
    )
    replay_training.add_argument(
        "--opponent",
        choices=[
            "follow_ball",
            "defensive",
            "aggressive",
            "random",
            "predictive",
            "dummy",
            "training_dummy",
        ],
        default="follow_ball",
    )
    replay_training.add_argument("--max-steps", type=int, default=1000)
    replay_training.add_argument("--ball-direction", type=int, choices=[-1, 0, 1], default=0)
    replay_training.add_argument(
        "--ball-angle",
        type=float,
        default=None,
        help="Initial ball angle in radians",
    )
    replay_training.add_argument(
        "--trace",
        type=str,
        default="debug/training_replay.json",
        help="JSON trace output path",
    )
    replay_training.add_argument("--replay", action="store_true", help="Open replay after capture")
    replay_training.add_argument("--fps", type=int, default=30, help="Replay frame rate")
    replay_training.add_argument(
        "--train-agent",
        action="store_true",
        help="Allow the agent to learn while capturing the episode",
    )
    replay_training.set_defaults(func=_cmd_replay_training)

    replay_trace = subparsers.add_parser(
        "replay-trace",
        help="Replay an existing training trace JSON file",
    )
    replay_trace.add_argument("trace", type=str)
    replay_trace.add_argument("--fps", type=int, default=30, help="Replay frame rate")
    replay_trace.set_defaults(func=_cmd_replay_trace)

    return parser


def main() -> None:
    args = build_parser().parse_args()
    args.func(args)


if __name__ == "__main__":
    main()
