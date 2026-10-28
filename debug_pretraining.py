#!/usr/bin/env python3
"""
Debug tool for the pretraining reward signal.

For each of the 9 actions, shows:
  - the simulated Y movement
  - the optimal interception Y
  - the resulting proximity reward

Usage:
    python debug_pretraining.py              # one random scenario
    python debug_pretraining.py --n 20      # 20 random scenarios, summary at end
    python debug_pretraining.py --seed 42   # reproducible run
    python debug_pretraining.py --y_only    # match y-only pretraining reward
"""

import argparse
import random

import numpy as np

from magic_pong.ai.models.dqn_ai import ACTION_MAPPING
from magic_pong.ai.pretraining import create_pretrainer
from magic_pong.utils.config import ai_config
from magic_pong.utils.config import game_config


def _action_label(a):
    action = ACTION_MAPPING[a]
    dirs = []
    if action.move_y < 0:
        dirs.append("UP")
    elif action.move_y > 0:
        dirs.append("DN")
    if action.move_x < 0:
        dirs.append("LT")
    elif action.move_x > 0:
        dirs.append("RT")
    return f"{a}({'·'.join(dirs) or 'STAY':7s})"


def debug_scenario(pretrainer, player_id=1, scenario_idx=None):
    dt = game_config.GAME_SPEED_MULTIPLIER / game_config.FPS

    ball_state = pretrainer.generate_random_ball_state(player_id)
    game_state = pretrainer.create_game_state_from_ball_state(ball_state, player_id)
    initial_paddle_pos = game_state[f"player{player_id}_position"]

    bx, by = ball_state["ball_position"]
    vx, vy = ball_state["ball_velocity"]
    px, py = initial_paddle_pos
    ph = game_config.PADDLE_HEIGHT

    # Find optimal interception point
    ai_config.USE_PROXIMITY_REWARD = True
    pretrainer.reward_calculator._calculate_proximity_reward(game_state, player_id)
    opt = pretrainer.reward_calculator.optimal_points.get(player_id, {})
    opt_pos = opt.get("position")

    prefix = f"[#{scenario_idx}] " if scenario_idx is not None else ""
    print(f"\n{prefix}Ball  x={bx:6.1f} y={by:6.1f}  vel_x={vx:+7.1f} vel_y={vy:+7.1f}")
    print(f"{'':8s}Paddle  x={px:6.1f} y={py:6.1f}  center_y={py + ph / 2:6.1f}")
    if opt_pos is not None:
        print(
            f"{'':8s}Optimal interception y={opt_pos[1]:6.1f}  "
            f"(delta_y from paddle center: {opt_pos[1] - (py + ph / 2):+.1f})"
        )
    else:
        print("        Optimal point: None (ball not approaching)")

    print()
    if pretrainer.y_only:
        print(
            f"  {'Action':12s}  {'train_as':12s}  {'sim_y':>6s}  {'new_py':>7s}"
            f"  {'old_dist':>8s}  {'new_dist':>8s}  reward"
        )
        print(f"  {'-' * 12}  {'-' * 12}  {'-' * 6}  {'-' * 7}  {'-' * 8}  {'-' * 8}  ------")
    else:
        print(
            f"  {'Action':12s}  {'sim_y':>6s}  {'new_py':>7s}"
            f"  {'old_dist':>8s}  {'new_dist':>8s}  reward"
        )
        print(f"  {'-' * 12}  {'-' * 6}  {'-' * 7}  {'-' * 8}  {'-' * 8}  ------")

    rows = []
    rewards = {}
    for action_idx in range(9):
        action = ACTION_MAPPING[action_idx]
        training_action_idx = pretrainer._action_index_for_training(action_idx)
        sim_action_obj = (
            type(action)(move_x=0.0, move_y=action.move_y) if pretrainer.y_only else action
        )

        new_paddle_pos = pretrainer.simulate_paddle_movement(
            initial_paddle_pos, sim_action_obj, player_id=player_id, dt=dt
        )

        # Reset to initial state and re-initialise the distance baseline
        gs_copy = dict(game_state)
        ai_config.USE_PROXIMITY_REWARD = True
        pretrainer.reward_calculator._calculate_proximity_reward(gs_copy, player_id)
        prev_dist = pretrainer.reward_calculator.last_ball_distance.get(player_id)

        # Apply paddle movement and compute new distance
        gs_copy[f"player{player_id}_prev_position"] = initial_paddle_pos
        gs_copy[f"player{player_id}_last_position"] = initial_paddle_pos
        gs_copy[f"player{player_id}_position"] = new_paddle_pos

        ai_config.USE_PROXIMITY_REWARD = True
        reward, _ = pretrainer.calculate_optimal_position_reward(gs_copy, player_id, dt=dt)
        new_dist = pretrainer.reward_calculator.last_ball_distance.get(player_id)
        ai_config.USE_PROXIMITY_REWARD = False

        rewards[action_idx] = reward
        rows.append(
            (
                action_idx,
                training_action_idx,
                action.move_y,
                new_paddle_pos[1],
                prev_dist or 0,
                new_dist or 0,
                reward,
            )
        )

    best_reward = max(rewards.values()) if rewards else 0.0
    for action_idx, training_action_idx, move_y, new_y, prev_dist, new_dist, reward in rows:
        marker = " <-- best" if np.isclose(reward, best_reward) else ""
        if pretrainer.y_only:
            print(
                f"  {_action_label(action_idx):12s}  {_action_label(training_action_idx):12s}"
                f"  {move_y:+6.1f}  {new_y:7.1f}"
                f"  {prev_dist:8.1f}  {new_dist:8.1f}  {reward:+.4f}{marker}"
            )
        else:
            print(
                f"  {_action_label(action_idx):12s}  {move_y:+6.1f}  {new_y:7.1f}"
                f"  {prev_dist:8.1f}  {new_dist:8.1f}  {reward:+.4f}{marker}"
            )

    print()
    return rewards


def main():
    parser = argparse.ArgumentParser(description="Debug pretraining reward signal")
    parser.add_argument("--n", type=int, default=1, help="Number of scenarios to run")
    parser.add_argument("--seed", type=int, default=None, help="Random seed")
    parser.add_argument("--player_id", type=int, default=1, choices=[1, 2])
    parser.add_argument(
        "--y_only",
        action="store_true",
        help="Use the same y-only pretraining objective as train_optimized.py --pretrain",
    )
    args = parser.parse_args()

    if args.seed is not None:
        random.seed(args.seed)
        np.random.seed(args.seed)

    pretrainer = create_pretrainer(y_only=args.y_only)

    positive_counts = dict.fromkeys(range(9), 0)
    negative_counts = dict.fromkeys(range(9), 0)

    for i in range(args.n):
        rewards = debug_scenario(
            pretrainer,
            player_id=args.player_id,
            scenario_idx=(i + 1) if args.n > 1 else None,
        )
        summary_rewards = (
            {action_idx: rewards[action_idx] for action_idx in (0, 1, 2)}
            if pretrainer.y_only
            else rewards
        )
        for a, r in summary_rewards.items():
            if r > 0:
                positive_counts[a] += 1
            elif r < 0:
                negative_counts[a] += 1

    if args.n > 1:
        print(f"\n{'=' * 60}")
        summary_kind = "training-label counts" if pretrainer.y_only else "action counts"
        print(f"Summary over {args.n} scenarios (y_only={pretrainer.y_only}, {summary_kind}):")
        print(f"  {'Action':12s}  {'positive':>9s}  {'negative':>9s}  {'net':>5s}")
        print(f"  {'-' * 12}  {'-' * 9}  {'-' * 9}  {'-' * 5}")
        for a in range(9):
            p = positive_counts[a]
            n = negative_counts[a]
            print(f"  {_action_label(a):12s}  {p:9d}  {n:9d}  {p - n:+5d}")


if __name__ == "__main__":
    main()
