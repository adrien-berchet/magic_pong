#!/usr/bin/env python3
"""
Diagnose positional bias and action oscillation in a trained DQN checkpoint.

Plays a loaded DQN checkpoint against the simple heuristic opponents headlessly
and reports:
  - The paddle's vertical position distribution (to spot corner-camping).
  - How often the paddle reverses vertical direction frame-to-frame (to spot
    Q-value flicker / "many opposite actions").
  - Hit rate broken down by the ball's vertical position at the moment of
    contact or miss (to spot a region of the field the agent struggles with).
"""

import argparse
from collections import defaultdict

import numpy as np

from magic_pong.ai.models.dqn_ai import DQNAgent
from magic_pong.ai.models.dqn_checkpoint import EXPECTED_ACTION_SIZE
from magic_pong.ai.models.dqn_checkpoint import EXPECTED_STATE_SIZE
from magic_pong.ai.models.simple_ai import create_ai
from magic_pong.core.game_engine import GameEngine
from magic_pong.utils.config import game_config

DEFAULT_OPPONENTS = ["follow_ball", "defensive", "aggressive", "random", "predictive"]
NUM_Y_BUCKETS = 5


def run_episode(
    game_engine: GameEngine, dqn_agent: DQNAgent, opponent, dqn_player_id: int, max_steps: int
) -> tuple[list[float], int, int, dict[int, int], dict[int, int]]:
    """Play one episode and collect positional/behavioral diagnostics."""
    players = (dqn_agent, opponent) if dqn_player_id == 1 else (opponent, dqn_agent)
    game_engine.set_players(*players)
    game_engine.start_game()

    y_positions: list[float] = []
    reversals = 0
    last_sign = 0
    bucket_hits: dict[int, int] = defaultdict(int)
    bucket_misses: dict[int, int] = defaultdict(int)

    steps = 0
    while game_engine.is_running() and steps < max_steps:
        pre_ball_y = game_engine.get_game_state()["ball_position"][1] / game_config.FIELD_HEIGHT
        result = game_engine.update()
        game_state = result["game_state"]

        paddle_size = game_state.get(
            f"player{dqn_player_id}_paddle_size", game_config.PADDLE_HEIGHT
        )
        raw_center_y = game_state[f"player{dqn_player_id}_position"][1] + paddle_size / 2
        norm_y = raw_center_y / game_config.FIELD_HEIGHT

        if y_positions:
            delta = norm_y - y_positions[-1]
            sign = 1 if delta > 1e-4 else (-1 if delta < -1e-4 else 0)
            if sign != 0:
                if last_sign != 0 and sign != last_sign:
                    reversals += 1
                last_sign = sign
        y_positions.append(norm_y)

        bucket = min(int(pre_ball_y * NUM_Y_BUCKETS), NUM_Y_BUCKETS - 1)
        for hit in result["events"].get("paddle_hits", []):
            if hit.get("player") == dqn_player_id:
                bucket_hits[bucket] += 1
        for goal in result["events"].get("goals", []):
            if goal.get("player") != dqn_player_id:
                bucket_misses[bucket] += 1

        steps += 1
        if result["done"]:
            break

    return y_positions, reversals, steps, bucket_hits, bucket_misses


def main() -> None:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--model_path", required=True, help="Path to the DQN checkpoint to load")
    parser.add_argument(
        "--opponents", default=",".join(DEFAULT_OPPONENTS), help="Comma-separated opponent names"
    )
    parser.add_argument("--episodes", type=int, default=100, help="Episodes to play per opponent")
    parser.add_argument(
        "--player_id", type=int, choices=[1, 2], default=1, help="Side the DQN agent plays on"
    )
    parser.add_argument("--max_steps", type=int, default=2000, help="Max steps per episode")
    parser.add_argument(
        "--action_hysteresis_margin",
        type=float,
        default=0.0,
        help="Greedy action hysteresis margin to test (0.0 = plain argmax)",
    )
    args = parser.parse_args()

    opponent_names = [name.strip() for name in args.opponents.split(",") if name.strip()]

    agent = DQNAgent(state_size=EXPECTED_STATE_SIZE, action_size=EXPECTED_ACTION_SIZE)
    agent.load_model(args.model_path)
    agent.set_training_mode(False)
    agent.set_action_hysteresis(args.action_hysteresis_margin)

    game_engine = GameEngine(headless=True)

    all_y_positions: list[float] = []
    total_reversals = 0
    total_steps = 0
    bucket_hits_total: dict[int, int] = defaultdict(int)
    bucket_misses_total: dict[int, int] = defaultdict(int)

    for opponent_name in opponent_names:
        for _ in range(args.episodes):
            opponent = create_ai(opponent_name)
            y_positions, reversals, steps, bucket_hits, bucket_misses = run_episode(
                game_engine, agent, opponent, args.player_id, args.max_steps
            )
            all_y_positions.extend(y_positions)
            total_reversals += reversals
            total_steps += steps
            for bucket, count in bucket_hits.items():
                bucket_hits_total[bucket] += count
            for bucket, count in bucket_misses.items():
                bucket_misses_total[bucket] += count
        print(f"Finished {args.episodes} episodes vs {opponent_name}")

    y_array = np.array(all_y_positions)
    print("\n=== Paddle Y position distribution (0=top edge, 1=bottom edge) ===")
    hist, edges = np.histogram(y_array, bins=10, range=(0.0, 1.0))
    for count, lo, hi in zip(hist, edges[:-1], edges[1:], strict=False):
        pct = 100.0 * count / len(y_array)
        bar = "#" * int(pct / 2)
        print(f"  [{lo:.1f}, {hi:.1f}): {pct:5.1f}%  {bar}")
    print(f"  mean={y_array.mean():.3f}  std={y_array.std():.3f}")

    print(f"\n=== Action reversals: {total_reversals} over {total_steps} steps ===")
    print(f"  {100.0 * total_reversals / total_steps:.2f} reversals per 100 steps")

    print("\n=== Hit rate by ball-Y position at moment of contact/miss ===")
    for bucket in range(NUM_Y_BUCKETS):
        hits = bucket_hits_total.get(bucket, 0)
        misses = bucket_misses_total.get(bucket, 0)
        total = hits + misses
        rate = 100.0 * hits / total if total else float("nan")
        lo = bucket / NUM_Y_BUCKETS
        hi = (bucket + 1) / NUM_Y_BUCKETS
        print(f"  ball_y in [{lo:.1f}, {hi:.1f}): {rate:5.1f}% hit rate (n={total})")


if __name__ == "__main__":
    main()
