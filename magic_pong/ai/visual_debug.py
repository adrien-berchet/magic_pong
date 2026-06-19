"""Visual debug capture and replay helpers for Magic Pong AI training."""

from __future__ import annotations

import json
import random
from collections.abc import Mapping
from contextlib import contextmanager
from pathlib import Path
from typing import Any

import numpy as np

from magic_pong.ai.agent_adapter import adapt_action_for_world
from magic_pong.ai.agent_adapter import adapt_info_for_agent
from magic_pong.ai.agent_adapter import adapt_observation_for_agent
from magic_pong.ai.interface import AIPlayer
from magic_pong.ai.models.dqn_ai import ACTION_MAPPING
from magic_pong.ai.pretraining import OptimalPointPretrainer
from magic_pong.ai.pretraining import create_pretrainer
from magic_pong.core.entities import Action
from magic_pong.core.entities import Player
from magic_pong.core.game_engine import GameEngine
from magic_pong.core.game_engine import _call_player_hook
from magic_pong.utils.config import ai_config
from magic_pong.utils.config import game_config

EVENT_LABELS = {
    "goals": "goal",
    "paddle_hits": "paddle hit",
    "rotating_paddle_hits": "rotating hit",
    "bonus_collected": "bonus",
    "wall_bounces": "wall",
}


def set_visual_debug_seed(seed: int | None) -> None:
    """Seed Python, NumPy, and torch when available for repeatable debug captures."""
    if seed is None:
        return

    random.seed(seed)
    np.random.seed(seed)
    try:
        import torch
    except ImportError:
        return
    torch.manual_seed(seed)
    if torch.cuda.is_available():
        torch.cuda.manual_seed_all(seed)


def action_label(action_index: int | None, action: Action | None = None) -> str:
    """Return a compact human-readable label for a discrete action."""
    if action_index is not None and action_index in ACTION_MAPPING:
        action = ACTION_MAPPING[action_index]
    if action is None:
        return "unknown"

    parts = []
    if action.move_y < 0:
        parts.append("up")
    elif action.move_y > 0:
        parts.append("down")
    if action.move_x < 0:
        parts.append("left")
    elif action.move_x > 0:
        parts.append("right")
    label = "+".join(parts) if parts else "stay"
    if action_index is None:
        return label
    return f"{action_index}:{label}"


def action_to_dict(action: Action, action_index: int | None = None) -> dict[str, Any]:
    """Serialize an action for debug output."""
    return {
        "index": action_index,
        "label": action_label(action_index, action),
        "move_x": float(action.move_x),
        "move_y": float(action.move_y),
    }


def action_index_for_action(action: Action) -> int | None:
    """Return the nearest discrete DQN action index for an arbitrary action."""
    best_index = None
    best_distance = float("inf")
    for index, mapped_action in ACTION_MAPPING.items():
        distance = (mapped_action.move_x - action.move_x) ** 2 + (
            mapped_action.move_y - action.move_y
        ) ** 2
        if distance < best_distance:
            best_index = index
            best_distance = distance
    return best_index


def capture_trajectory_debug(
    *,
    pretrainer: OptimalPointPretrainer | None = None,
    player_id: int = 1,
    seed: int | None = None,
) -> dict[str, Any]:
    """Capture one random pretraining trajectory and its optimal interception point."""
    set_visual_debug_seed(seed)
    pretrainer = pretrainer or create_pretrainer(y_only=False)
    pretrainer.reward_calculator.reset()

    ball_state = pretrainer.generate_random_ball_state(player_id)
    game_state = pretrainer.create_game_state_from_ball_state(ball_state, player_id)
    optimal_point = _prime_and_get_optimal_point(pretrainer, game_state, player_id)
    trajectory = _simulate_trajectory_from_state(pretrainer, game_state)

    return _jsonify(
        {
            "kind": "trajectory",
            "seed": seed,
            "player_id": player_id,
            "game_state": game_state,
            "ball_state": ball_state,
            "trajectory": trajectory,
            "optimal_point": optimal_point,
            "config": _config_summary(),
        }
    )


def capture_pretraining_run(
    agent: Any,
    *,
    pretrainer: OptimalPointPretrainer | None = None,
    player_id: int = 1,
    seed: int | None = None,
    explore: bool = False,
) -> dict[str, Any]:
    """Capture one model decision in the pretraining objective."""
    set_visual_debug_seed(seed)
    pretrainer = pretrainer or create_pretrainer(y_only=False)
    pretrainer.reward_calculator.reset()

    dt = game_config.GAME_SPEED_MULTIPLIER / game_config.FPS
    ball_state = pretrainer.generate_random_ball_state(player_id)
    initial_game_state = pretrainer.create_game_state_from_ball_state(ball_state, player_id)
    initial_paddle_pos = initial_game_state[f"player{player_id}_position"]

    optimal_point = _prime_and_get_optimal_point(pretrainer, initial_game_state, player_id)
    previous_distance = pretrainer.reward_calculator.last_ball_distance.get(player_id)

    with _temporary_agent_decision_state(agent):
        action_index = _select_pretraining_action(
            agent, pretrainer, initial_game_state, player_id, explore
        )
    action = ACTION_MAPPING[action_index]
    simulated_action = Action(move_x=0.0, move_y=action.move_y) if pretrainer.y_only else action
    new_paddle_pos = pretrainer.simulate_paddle_movement(
        initial_paddle_pos, simulated_action, player_id=player_id, dt=dt
    )

    final_game_state = dict(initial_game_state)
    final_game_state[f"player{player_id}_prev_position"] = initial_paddle_pos
    final_game_state[f"player{player_id}_last_position"] = initial_paddle_pos
    final_game_state[f"player{player_id}_position"] = new_paddle_pos

    reward, reward_info = pretrainer.calculate_optimal_position_reward(
        final_game_state, player_id, dt=dt
    )
    new_distance = pretrainer.reward_calculator.last_ball_distance.get(player_id)
    trajectory = _simulate_trajectory_from_state(pretrainer, initial_game_state)

    return _jsonify(
        {
            "kind": "pretraining_run",
            "seed": seed,
            "player_id": player_id,
            "initial_game_state": initial_game_state,
            "final_game_state": final_game_state,
            "ball_state": ball_state,
            "trajectory": trajectory,
            "optimal_point": optimal_point,
            "action": action_to_dict(action, action_index),
            "simulated_action": action_to_dict(simulated_action, action_index),
            "reward": float(reward),
            "reward_info": reward_info,
            "previous_distance": previous_distance,
            "new_distance": new_distance,
            "config": _config_summary(),
        }
    )


def capture_training_episode(
    agent: Player,
    opponent: Player,
    *,
    player_id: int = 1,
    seed: int | None = None,
    max_steps: int = 1000,
    initial_ball_direction: int = 0,
    initial_ball_angle: float | None = None,
    train_agent: bool = False,
) -> dict[str, Any]:
    """Run and capture one complete episode for later replay."""
    if player_id not in (1, 2):
        raise ValueError("player_id must be 1 or 2")

    set_visual_debug_seed(seed)
    fixed_dt = game_config.GAME_SPEED_MULTIPLIER / game_config.FPS

    with _temporary_training_mode(agent, train_agent):
        engine = GameEngine(headless=True)
        player1 = agent if player_id == 1 else opponent
        player2 = opponent if player_id == 1 else agent

        engine.set_players(player1, player2)
        engine.start_game()
        engine.ai_environment.max_steps = max_steps
        if initial_ball_direction != 0 or initial_ball_angle is not None:
            engine.ai_environment.physics_engine.reset_ball(
                initial_ball_direction, initial_ball_angle
            )

        frames: list[dict[str, Any]] = []
        totals = {"player1": 0.0, "player2": 0.0}
        winner = 0

        while engine.is_running() and len(frames) < max_steps:
            action1 = engine._get_player_action(engine.player1, 1) or Action(0.0, 0.0)
            action2 = engine._get_player_action(engine.player2, 2) or Action(0.0, 0.0)

            obs1, obs2, reward1, reward2, done, info = engine.ai_environment.step(
                action1, action2, dt=fixed_dt
            )
            _notify_step(engine.player1, obs1, action1, reward1, done, info, 1)
            _notify_step(engine.player2, obs2, action2, reward2, done, info, 2)

            totals["player1"] += reward1
            totals["player2"] += reward2
            frame = _build_training_frame(
                len(frames), action1, action2, reward1, reward2, done, info
            )
            frames.append(frame)

            if done:
                winner = int(info.get("winner", 0))
                engine.running = False
                break

        if engine.is_running():
            engine.running = False
        _call_player_hook(engine.player1, "on_episode_end")
        _call_player_hook(engine.player2, "on_episode_end")

    reward_frames = [frame["index"] for frame in frames if frame["is_reward_frame"]]
    final_score = frames[-1]["score"] if frames else [0, 0]
    trace = {
        "kind": "training_replay",
        "metadata": {
            "seed": seed,
            "player_id": player_id,
            "opponent": getattr(opponent, "name", type(opponent).__name__),
            "max_steps": max_steps,
            "initial_ball_direction": initial_ball_direction,
            "initial_ball_angle": initial_ball_angle,
            "dt": fixed_dt,
            "config": _config_summary(),
        },
        "frames": frames,
        "reward_frames": reward_frames,
        "summary": {
            "winner": winner,
            "steps": len(frames),
            "total_reward_p1": totals["player1"],
            "total_reward_p2": totals["player2"],
            "final_score": final_score,
        },
    }
    return _jsonify(trace)


def mark_reward_frames(trace: dict[str, Any]) -> dict[str, Any]:
    """Recompute reward-frame markers for an existing trace."""
    reward_frames = []
    for index, frame in enumerate(trace.get("frames", [])):
        frame["is_reward_frame"] = is_reward_frame(frame)
        frame.setdefault("index", index)
        if frame["is_reward_frame"]:
            reward_frames.append(frame["index"])
    trace["reward_frames"] = reward_frames
    return trace


def is_reward_frame(frame: Mapping[str, Any]) -> bool:
    """Return True when a frame carries a reward, penalty, or game event."""
    rewards = frame.get("rewards", {})
    if isinstance(rewards, Mapping):
        for reward in rewards.values():
            if abs(float(reward)) > 1e-12:
                return True

    events = frame.get("events", {})
    if isinstance(events, Mapping):
        return any(bool(bucket) for bucket in events.values())
    return False


def save_debug_json(data: Mapping[str, Any], path: str | Path) -> Path:
    """Write debug data as pretty JSON."""
    output_path = Path(path)
    output_path.parent.mkdir(parents=True, exist_ok=True)
    with output_path.open("w", encoding="utf-8") as handle:
        json.dump(_jsonify(data), handle, indent=2)
    return output_path


def load_debug_json(path: str | Path) -> dict[str, Any]:
    """Load debug JSON data."""
    with Path(path).open(encoding="utf-8") as handle:
        data = json.load(handle)
    if isinstance(data, dict) and data.get("kind") == "training_replay":
        mark_reward_frames(data)
    return data


def render_trajectory_png(data: Mapping[str, Any], output_path: str | Path) -> Path:
    """Render a trajectory capture as a PNG image."""
    plt = _get_pyplot()
    fig, ax = plt.subplots(figsize=(10, 7))
    _draw_matplotlib_field(ax, data.get("config", {}))
    _draw_matplotlib_game_state(ax, data["game_state"], player_id=int(data["player_id"]))
    _draw_matplotlib_trajectory(ax, data.get("trajectory", []))
    _draw_matplotlib_optimal_point(ax, data.get("optimal_point"), int(data["player_id"]))

    ball_pos = data["game_state"]["ball_position"]
    ball_vel = data["game_state"]["ball_velocity"]
    ax.set_title(
        "Pretraining trajectory "
        f"seed={data.get('seed')} ball=({ball_pos[0]:.1f},{ball_pos[1]:.1f}) "
        f"vel=({ball_vel[0]:+.1f},{ball_vel[1]:+.1f})"
    )
    ax.legend(loc="lower right")
    return _save_figure(fig, output_path)


def render_pretraining_run_png(data: Mapping[str, Any], output_path: str | Path) -> Path:
    """Render a pretraining decision capture as a PNG image."""
    plt = _get_pyplot()
    fig, ax = plt.subplots(figsize=(10, 7))
    player_id = int(data["player_id"])
    _draw_matplotlib_field(ax, data.get("config", {}))
    _draw_matplotlib_game_state(ax, data["initial_game_state"], player_id=player_id)
    _draw_matplotlib_trajectory(ax, data.get("trajectory", []))
    _draw_matplotlib_optimal_point(ax, data.get("optimal_point"), player_id)
    _draw_matplotlib_final_paddle(ax, data["final_game_state"], player_id)

    initial_pos = data["initial_game_state"][f"player{player_id}_position"]
    final_pos = data["final_game_state"][f"player{player_id}_position"]
    paddle_height = float(data["initial_game_state"].get(f"player{player_id}_paddle_size", 0.0))
    start = (
        float(initial_pos[0]) + game_config.PADDLE_WIDTH / 2,
        float(initial_pos[1]) + paddle_height / 2,
    )
    end = (
        float(final_pos[0]) + game_config.PADDLE_WIDTH / 2,
        float(final_pos[1]) + paddle_height / 2,
    )
    ax.annotate(
        "",
        xy=end,
        xytext=start,
        arrowprops={"arrowstyle": "->", "color": "#00d4ff", "linewidth": 2.0},
    )

    action = data.get("action", {})
    reward = float(data.get("reward", 0.0))
    text = (
        f"Action: {action.get('label', 'unknown')}\n"
        f"Reward: {reward:+.4f}\n"
        f"Distance: {data.get('previous_distance', 0):.2f} -> "
        f"{data.get('new_distance', 0):.2f}"
    )
    ax.text(
        0.02,
        0.98,
        text,
        transform=ax.transAxes,
        va="top",
        ha="left",
        color="white",
        bbox={"boxstyle": "round,pad=0.45", "facecolor": "#111827", "alpha": 0.88},
    )
    ax.set_title(f"Pretraining decision seed={data.get('seed')}")
    ax.legend(loc="lower right")
    return _save_figure(fig, output_path)


def replay_training_trace(trace: Mapping[str, Any], *, fps: int = 30) -> None:
    """Replay a captured training trace in a Pygame window."""
    import pygame

    frames = trace.get("frames", [])
    if not frames:
        raise ValueError("Trace has no frames to replay")

    width, height = _trace_field_size(trace)
    pygame.init()
    pygame.font.init()
    screen = pygame.display.set_mode((width, height))
    pygame.display.set_caption("Magic Pong Training Replay")
    clock = pygame.time.Clock()
    font = pygame.font.Font(None, 24)
    large_font = pygame.font.Font(None, 34)

    index = 0
    paused = False
    running = True
    while running:
        for event in pygame.event.get():
            if event.type == pygame.QUIT:
                running = False
            elif event.type == pygame.KEYDOWN:
                if event.key in (pygame.K_ESCAPE, pygame.K_q):
                    running = False
                elif event.key == pygame.K_SPACE:
                    paused = not paused
                elif event.key in (pygame.K_RIGHT, pygame.K_PERIOD):
                    paused = True
                    index = min(index + 1, len(frames) - 1)
                elif event.key in (pygame.K_LEFT, pygame.K_COMMA):
                    paused = True
                    index = max(index - 1, 0)
                elif event.key == pygame.K_n:
                    paused = True
                    index = _next_reward_index(trace, index)
                elif event.key == pygame.K_p:
                    paused = True
                    index = _previous_reward_index(trace, index)

        draw_training_replay_frame(screen, trace, index, font=font, large_font=large_font)
        pygame.display.flip()

        if not paused:
            if index < len(frames) - 1:
                index += 1
            else:
                paused = True
        clock.tick(fps)

    pygame.quit()


def draw_training_replay_frame(
    surface: Any,
    trace: Mapping[str, Any],
    frame_index: int,
    *,
    font: Any | None = None,
    large_font: Any | None = None,
) -> None:
    """Draw one replay frame onto an existing Pygame surface."""
    import pygame

    frames = trace.get("frames", [])
    if not frames:
        raise ValueError("Trace has no frames")
    frame_index = max(0, min(frame_index, len(frames) - 1))
    frame = frames[frame_index]
    game_state = frame["game_state"]

    width, height = surface.get_size()
    if font is None:
        pygame.font.init()
        font = pygame.font.Font(None, 24)
    if large_font is None:
        large_font = pygame.font.Font(None, 34)

    surface.fill((6, 10, 16))
    center_x = width // 2
    pygame.draw.line(surface, (76, 86, 102), (center_x, 0), (center_x, height), 2)
    pygame.draw.circle(surface, (76, 86, 102), (center_x, height // 2), 50, 2)

    _draw_pygame_paddle(surface, game_state, 1)
    _draw_pygame_paddle(surface, game_state, 2)
    _draw_pygame_ball(surface, game_state)
    _draw_pygame_optimal_points(surface, frame)
    _draw_pygame_events(surface, frame)
    _draw_pygame_overlay(surface, trace, frame, frame_index, font, large_font)


def _select_pretraining_action(
    agent: Any,
    pretrainer: OptimalPointPretrainer,
    game_state: dict[str, Any],
    player_id: int,
    explore: bool,
) -> int:
    if hasattr(agent, "_observation_to_state") and hasattr(agent, "act"):
        observation = pretrainer._game_state_to_observation(game_state, player_id)
        state = agent._observation_to_state(observation)
        return int(agent.act(state, explore=explore))

    if hasattr(agent, "get_action"):
        observation = pretrainer._game_state_to_observation(game_state, player_id)
        action = agent.get_action(observation)
        action_index = action_index_for_action(action)
        if action_index is not None:
            return action_index

    raise TypeError("agent must expose DQN act/_observation_to_state or get_action")


def _prime_and_get_optimal_point(
    pretrainer: OptimalPointPretrainer, game_state: dict[str, Any], player_id: int
) -> dict[str, Any] | None:
    original_proximity = ai_config.USE_PROXIMITY_REWARD
    ai_config.USE_PROXIMITY_REWARD = True
    try:
        pretrainer.reward_calculator._calculate_proximity_reward(game_state, player_id)
        return _jsonify(pretrainer.reward_calculator.get_optimal_points().get(player_id))
    finally:
        ai_config.USE_PROXIMITY_REWARD = original_proximity


def _simulate_trajectory_from_state(
    pretrainer: OptimalPointPretrainer,
    game_state: Mapping[str, Any],
) -> list[dict[str, Any]]:
    ball_pos = tuple(game_state["ball_position"])
    ball_vel = tuple(game_state["ball_velocity"])
    field_bounds = tuple(
        game_state.get("field_bounds", (0, game_config.FIELD_WIDTH, 0, game_config.FIELD_HEIGHT))
    )
    points = pretrainer.reward_calculator._simulate_ball_trajectory(
        ball_pos, ball_vel, field_bounds, max_time=None
    )
    return [
        {"position": [float(position[0]), float(position[1])], "time": float(time_step)}
        for position, time_step in points
    ]


def _notify_step(
    player: Player | None,
    observation: dict[str, Any],
    action: Action,
    reward: float,
    done: bool,
    info: dict[str, Any],
    player_id: int,
) -> None:
    if player is None:
        return
    agent_observation = adapt_observation_for_agent(observation, player_id, player)
    agent_action = adapt_action_for_world(action, player_id, player)
    agent_info = adapt_info_for_agent(info, player_id, player)
    _call_player_hook(player, "on_step", agent_observation, agent_action, reward, done, agent_info)


def _build_training_frame(
    index: int,
    action1: Action,
    action2: Action,
    reward1: float,
    reward2: float,
    done: bool,
    info: Mapping[str, Any],
) -> dict[str, Any]:
    game_state = info["game_state"]
    frame = {
        "index": index,
        "game_state": game_state,
        "actions": {
            "player1": action_to_dict(action1, action_index_for_action(action1)),
            "player2": action_to_dict(action2, action_index_for_action(action2)),
        },
        "rewards": {"player1": float(reward1), "player2": float(reward2)},
        "events": info.get("events", {}),
        "score": game_state.get("score", [0, 0]),
        "done": bool(done),
        "winner": int(info.get("winner", 0)),
        "optimal_points": info.get("optimal_points", {}),
    }
    frame["is_reward_frame"] = is_reward_frame(frame)
    return _jsonify(frame)


@contextmanager
def _temporary_training_mode(player: Player, training: bool) -> Any:
    previous_training = getattr(player, "training_enabled", None)
    previous_exploration = getattr(player, "exploration_enabled", None)
    previous_last_chosen_action = getattr(player, "_last_chosen_action", None)
    previous_last_state = getattr(player, "last_state", None)
    previous_last_action = getattr(player, "last_action", None)
    if isinstance(player, AIPlayer) and hasattr(player, "set_training_mode"):
        player.set_training_mode(training)
    try:
        yield
    finally:
        if previous_training is not None and hasattr(player, "set_training_mode"):
            player.set_training_mode(bool(previous_training))
        if previous_exploration is not None and hasattr(player, "set_exploration_mode"):
            player.set_exploration_mode(bool(previous_exploration))
        if hasattr(player, "_last_chosen_action"):
            player._last_chosen_action = previous_last_chosen_action
        if hasattr(player, "last_state"):
            player.last_state = previous_last_state
        if hasattr(player, "last_action"):
            player.last_action = previous_last_action


@contextmanager
def _temporary_agent_decision_state(agent: Any) -> Any:
    previous_last_chosen_action = getattr(agent, "_last_chosen_action", None)
    previous_last_state = getattr(agent, "last_state", None)
    previous_last_action = getattr(agent, "last_action", None)
    try:
        yield
    finally:
        if hasattr(agent, "_last_chosen_action"):
            agent._last_chosen_action = previous_last_chosen_action
        if hasattr(agent, "last_state"):
            agent.last_state = previous_last_state
        if hasattr(agent, "last_action"):
            agent.last_action = previous_last_action


def _jsonify(value: Any) -> Any:
    if isinstance(value, np.ndarray):
        return _jsonify(value.tolist())
    if isinstance(value, np.generic):
        return value.item()
    if isinstance(value, Mapping):
        return {str(key): _jsonify(item) for key, item in value.items()}
    if isinstance(value, tuple | list):
        return [_jsonify(item) for item in value]
    return value


def _config_summary() -> dict[str, Any]:
    return {
        "field_width": float(game_config.FIELD_WIDTH),
        "field_height": float(game_config.FIELD_HEIGHT),
        "paddle_width": float(game_config.PADDLE_WIDTH),
        "paddle_height": float(game_config.PADDLE_HEIGHT),
        "ball_radius": float(game_config.BALL_RADIUS),
        "fps": float(game_config.FPS),
        "game_speed_multiplier": float(game_config.GAME_SPEED_MULTIPLIER),
    }


def _get_pyplot() -> Any:
    import matplotlib

    matplotlib.use("Agg")
    import matplotlib.pyplot as plt

    return plt


def _save_figure(fig: Any, output_path: str | Path) -> Path:
    path = Path(output_path)
    path.parent.mkdir(parents=True, exist_ok=True)
    fig.savefig(path, dpi=160, bbox_inches="tight")
    fig.clear()
    return path


def _draw_matplotlib_field(ax: Any, config: Mapping[str, Any]) -> None:
    width = float(config.get("field_width", game_config.FIELD_WIDTH))
    height = float(config.get("field_height", game_config.FIELD_HEIGHT))
    ax.set_facecolor("#06101a")
    ax.set_xlim(0, width)
    ax.set_ylim(height, 0)
    ax.set_aspect("equal", adjustable="box")
    ax.axvline(width / 2, color="#526173", linewidth=1.5)
    ax.add_patch(
        _matplotlib_patches().Circle(
            (width / 2, height / 2), 50, fill=False, edgecolor="#526173", linewidth=1.5
        )
    )
    ax.set_xlabel("x")
    ax.set_ylabel("y")
    ax.grid(color="#1f2937", alpha=0.3)


def _draw_matplotlib_game_state(ax: Any, game_state: Mapping[str, Any], *, player_id: int) -> None:
    patches = _matplotlib_patches()
    for pid, color in (
        (1, "#4ade80" if player_id == 1 else "#9ca3af"),
        (2, "#4ade80" if player_id == 2 else "#9ca3af"),
    ):
        pos = game_state[f"player{pid}_position"]
        height = float(game_state.get(f"player{pid}_paddle_size", game_config.PADDLE_HEIGHT))
        rect = patches.Rectangle(
            (float(pos[0]), float(pos[1])),
            game_config.PADDLE_WIDTH,
            height,
            linewidth=1.5,
            edgecolor=color,
            facecolor=color,
            alpha=0.75,
            label=f"P{pid} paddle",
        )
        ax.add_patch(rect)

    ball_pos = game_state["ball_position"]
    ax.add_patch(
        patches.Circle(
            (float(ball_pos[0]), float(ball_pos[1])),
            game_config.BALL_RADIUS,
            facecolor="#f8fafc",
            edgecolor="#ffffff",
            label="ball",
        )
    )


def _draw_matplotlib_final_paddle(ax: Any, game_state: Mapping[str, Any], player_id: int) -> None:
    patches = _matplotlib_patches()
    pos = game_state[f"player{player_id}_position"]
    height = float(game_state.get(f"player{player_id}_paddle_size", game_config.PADDLE_HEIGHT))
    ax.add_patch(
        patches.Rectangle(
            (float(pos[0]), float(pos[1])),
            game_config.PADDLE_WIDTH,
            height,
            linewidth=2.5,
            edgecolor="#00d4ff",
            facecolor="none",
            label="paddle after action",
        )
    )


def _draw_matplotlib_trajectory(ax: Any, trajectory: Any) -> None:
    if not trajectory:
        return
    xs = [float(point["position"][0]) for point in trajectory]
    ys = [float(point["position"][1]) for point in trajectory]
    ax.plot(xs, ys, color="#facc15", linewidth=2.0, label="ball trajectory")
    ax.scatter(xs[-1:], ys[-1:], color="#f59e0b", s=35, label="trajectory end")


def _draw_matplotlib_optimal_point(
    ax: Any, optimal_point: Mapping[str, Any] | None, player_id: int
) -> None:
    if not optimal_point:
        return
    position = optimal_point.get("position")
    if not position:
        return
    color = "#ef4444" if player_id == 1 else "#22c55e"
    ball_interception = optimal_point.get("ball_interception_position")
    if ball_interception:
        ax.scatter(
            [float(ball_interception[0])],
            [float(ball_interception[1])],
            color="#f97316",
            s=110,
            marker="o",
            linewidths=2,
            facecolors="none",
            label="ball intercept",
        )
    ax.scatter(
        [float(position[0])],
        [float(position[1])],
        color=color,
        s=140,
        marker="x",
        linewidths=3,
        label="paddle target",
    )


def _matplotlib_patches() -> Any:
    import matplotlib.patches as patches

    return patches


def _trace_field_size(trace: Mapping[str, Any]) -> tuple[int, int]:
    config = (
        trace.get("metadata", {}).get("config", {})
        if isinstance(trace.get("metadata"), Mapping)
        else {}
    )
    width = int(config.get("field_width", game_config.FIELD_WIDTH))
    height = int(config.get("field_height", game_config.FIELD_HEIGHT))
    return width, height


def _next_reward_index(trace: Mapping[str, Any], current_index: int) -> int:
    for index in trace.get("reward_frames", []):
        if int(index) > current_index:
            return int(index)
    return current_index


def _previous_reward_index(trace: Mapping[str, Any], current_index: int) -> int:
    for index in reversed(trace.get("reward_frames", [])):
        if int(index) < current_index:
            return int(index)
    return current_index


def _draw_pygame_paddle(surface: Any, game_state: Mapping[str, Any], player_id: int) -> None:
    import pygame

    pos = game_state[f"player{player_id}_position"]
    height = int(game_state.get(f"player{player_id}_paddle_size", game_config.PADDLE_HEIGHT))
    color = (74, 222, 128) if player_id == 1 else (96, 165, 250)
    rect = pygame.Rect(int(pos[0]), int(pos[1]), int(game_config.PADDLE_WIDTH), height)
    pygame.draw.rect(surface, color, rect)


def _draw_pygame_ball(surface: Any, game_state: Mapping[str, Any]) -> None:
    import pygame

    pos = game_state["ball_position"]
    pygame.draw.circle(
        surface,
        (248, 250, 252),
        (int(pos[0]), int(pos[1])),
        int(game_config.BALL_RADIUS),
    )


def _draw_pygame_optimal_points(surface: Any, frame: Mapping[str, Any]) -> None:
    import pygame

    points = frame.get("optimal_points", {})
    if not isinstance(points, Mapping):
        return
    for raw_player_id, point_data in points.items():
        if not point_data:
            continue
        position = point_data.get("position")
        if not position:
            continue
        player_id = int(raw_player_id)
        color = (239, 68, 68) if player_id == 1 else (34, 197, 94)
        ball_interception = point_data.get("ball_interception_position")
        if ball_interception:
            pygame.draw.circle(
                surface,
                (249, 115, 22),
                (int(ball_interception[0]), int(ball_interception[1])),
                12,
                2,
            )
        pygame.draw.circle(surface, color, (int(position[0]), int(position[1])), 11, 3)
        pygame.draw.line(
            surface,
            color,
            (int(position[0]) - 14, int(position[1])),
            (int(position[0]) + 14, int(position[1])),
            2,
        )
        pygame.draw.line(
            surface,
            color,
            (int(position[0]), int(position[1]) - 14),
            (int(position[0]), int(position[1]) + 14),
            2,
        )


def _draw_pygame_events(surface: Any, frame: Mapping[str, Any]) -> None:
    import pygame

    if not frame.get("is_reward_frame"):
        return
    overlay = pygame.Surface(surface.get_size(), pygame.SRCALPHA)
    rewards = frame.get("rewards", {})
    p1 = float(rewards.get("player1", 0.0)) if isinstance(rewards, Mapping) else 0.0
    p2 = float(rewards.get("player2", 0.0)) if isinstance(rewards, Mapping) else 0.0
    if p1 < 0 or p2 < 0:
        color = (239, 68, 68, 52)
    elif p1 > 0 or p2 > 0:
        color = (34, 197, 94, 46)
    else:
        color = (250, 204, 21, 42)
    overlay.fill(color)
    surface.blit(overlay, (0, 0))


def _draw_pygame_overlay(
    surface: Any,
    trace: Mapping[str, Any],
    frame: Mapping[str, Any],
    frame_index: int,
    font: Any,
    large_font: Any,
) -> None:
    import pygame

    width, _height = surface.get_size()
    panel = pygame.Surface((min(520, width - 24), 142), pygame.SRCALPHA)
    panel.fill((15, 23, 42, 220))
    surface.blit(panel, (12, 12))

    score = frame.get("score", [0, 0])
    title = f"Frame {frame_index + 1}/{len(trace.get('frames', []))}  Score {score[0]}-{score[1]}"
    surface.blit(large_font.render(title, True, (248, 250, 252)), (24, 22))

    rewards = frame.get("rewards", {})
    actions = frame.get("actions", {})
    p1_action = (
        actions.get("player1", {}).get("label", "?") if isinstance(actions, Mapping) else "?"
    )
    p2_action = (
        actions.get("player2", {}).get("label", "?") if isinstance(actions, Mapping) else "?"
    )
    lines = [
        f"Rewards: P1 {float(rewards.get('player1', 0.0)):+.3f}  "
        f"P2 {float(rewards.get('player2', 0.0)):+.3f}",
        f"Actions: P1 {p1_action}  P2 {p2_action}",
        f"Events: {_event_summary(frame.get('events', {})) or 'none'}",
        "Space pause  Left/Right step  P/N reward frame  Q/Esc quit",
    ]
    y = 58
    for line in lines:
        surface.blit(font.render(line, True, (203, 213, 225)), (24, y))
        y += 22


def _event_summary(events: Any) -> str:
    if not isinstance(events, Mapping):
        return ""
    labels = []
    for key, bucket in events.items():
        if bucket:
            labels.append(f"{EVENT_LABELS.get(str(key), str(key))}x{len(bucket)}")
    return ", ".join(labels)
