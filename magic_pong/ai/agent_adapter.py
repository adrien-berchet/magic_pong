"""
Agent-facing observation/action adapters.

DQN checkpoints were trained from the player-1/left-side frame.  Keep that
contract local to DQN-like agents so existing heuristic AIs continue to see the
current world-relative observations.
"""

import math
from copy import deepcopy
from typing import Any

from magic_pong.core.entities import Action
from magic_pong.utils.config import ai_config
from magic_pong.utils.config import game_config

DQN_PLAYER1_FRAME_ATTR = "uses_player1_dqn_frame"
DQN_CONTACT_EVENT_KEYS = ("paddle_hits", "rotating_paddle_hits")


def uses_player1_dqn_frame(agent: object | None) -> bool:
    """Return whether an agent expects player-1/left-side DQN semantics."""
    return bool(getattr(agent, DQN_PLAYER1_FRAME_ATTR, False))


def adapt_observation_for_agent(
    observation: dict[str, Any], player_id: int, agent: object | None
) -> dict[str, Any]:
    """Adapt an observation before passing it to an opted-in DQN-like agent."""
    if not uses_player1_dqn_frame(agent):
        return observation

    return to_player1_dqn_observation(observation, player_id)


def adapt_action_for_world(action: Action, player_id: int, agent: object | None) -> Action:
    """Adapt an action between world/physics space and DQN canonical frame.

    The transform is its own inverse (flipping move_x for player 2 is symmetric),
    so this function serves both directions for opted-in DQN-like agents.
    """
    if not uses_player1_dqn_frame(agent):
        return action

    return from_player1_dqn_action(action, player_id)


def adapt_info_for_agent(
    info: dict[str, Any], player_id: int, agent: object | None
) -> dict[str, Any]:
    """Adapt step info before passing it to an opted-in DQN-like agent."""
    if not uses_player1_dqn_frame(agent):
        return info

    adapted = info.copy()
    events = info.get("events")
    if isinstance(events, dict):
        adapted["events"] = _filter_contact_events_for_agent(events, player_id)

    return adapted


def to_player1_dqn_observation(observation: dict[str, Any], player_id: int) -> dict[str, Any]:
    """
    Convert a player-relative observation to the DQN checkpoint frame.

    Player 1 is already in the canonical left-side frame. Player 2 observations
    are mirrored horizontally so the DQN still sees itself as the left player.
    """
    _validate_player_id(player_id)
    if player_id == 1:
        return observation

    adapted = deepcopy(observation)

    _mirror_position_x(adapted, "ball_pos", paddle_like=False)
    _mirror_position_x(adapted, "player_pos", paddle_like=True)
    _mirror_position_x(adapted, "opponent_pos", paddle_like=True)
    _mirror_position_x(adapted, "opponent_previous_pos", paddle_like=True)

    ball_vel = adapted.get("ball_vel")
    if ball_vel is not None and _has_xy(ball_vel):
        ball_vel[0] = -float(ball_vel[0])

    for bonus in adapted.get("bonuses", []):
        if _has_xy(bonus):
            bonus[0] = _mirror_x(float(bonus[0]), paddle_like=False, observation=adapted)

    for rotating_paddle in adapted.get("rotating_paddles", []):
        if _has_xy(rotating_paddle):
            rotating_paddle[0] = _mirror_x(
                float(rotating_paddle[0]), paddle_like=False, observation=adapted
            )
            if len(rotating_paddle) >= 3:
                rotating_paddle[2] = _mirror_angle(float(rotating_paddle[2]))

    return adapted


def to_player1_frame_x(x: float, player_id: int, x_extent: float) -> float:
    """Mirror an x coordinate into the canonical player-1/left-side frame."""
    _validate_player_id(player_id)
    if player_id == 1:
        return x
    return x_extent - x


def to_player1_frame_vx(vx: float, player_id: int) -> float:
    """Mirror a horizontal velocity into the canonical player-1/left-side frame."""
    _validate_player_id(player_id)
    if player_id == 1:
        return vx
    return -vx


def from_player1_dqn_action(action: Action, player_id: int) -> Action:
    """Convert a DQN action from canonical player-1 frame to world space."""
    _validate_player_id(player_id)
    if player_id == 1:
        return action
    return Action(move_x=-action.move_x, move_y=action.move_y)


def _filter_contact_events_for_agent(events: dict[str, Any], player_id: int) -> dict[str, Any]:
    filtered = deepcopy(events)
    for key in DQN_CONTACT_EVENT_KEYS:
        if key not in events:
            continue
        filtered[key] = _canonicalize_agent_contact_bucket(events[key], player_id)
    return filtered


def _canonicalize_agent_contact_bucket(bucket: Any, player_id: int) -> Any:
    if isinstance(bucket, list):
        return [
            _canonicalize_contact_event(event)
            for event in bucket
            if _event_belongs_to_player(event, player_id)
        ]
    if isinstance(bucket, dict):
        if _event_belongs_to_player(bucket, player_id):
            return _canonicalize_contact_event(bucket)
        return {}
    return bucket


def _canonicalize_contact_event(event: Any) -> Any:
    if isinstance(event, dict):
        canonical_event = deepcopy(event)
        canonical_event["player"] = 1
        return canonical_event
    return event


def _event_belongs_to_player(event: Any, player_id: int) -> bool:
    if not isinstance(event, dict):
        return True
    if "player" not in event:
        return True
    return event["player"] == player_id


def _validate_player_id(player_id: int) -> None:
    if player_id not in (1, 2):
        raise ValueError(f"Invalid player_id: {player_id}. Expected 1 or 2.")


def _mirror_position_x(observation: dict[str, Any], key: str, *, paddle_like: bool) -> None:
    value = observation.get(key)
    if value is not None and _has_xy(value):
        value[0] = _mirror_x(float(value[0]), paddle_like=paddle_like, observation=observation)


def _has_xy(value: object) -> bool:
    return isinstance(value, list) and len(value) >= 2


def _mirror_x(x: float, *, paddle_like: bool, observation: dict[str, Any]) -> float:
    x_extent = _x_extent(observation)
    if paddle_like:
        return x_extent - x - _paddle_width_extent(observation)
    return to_player1_frame_x(x, 2, x_extent)


def _mirror_angle(angle: float) -> float:
    mirrored = math.pi - angle
    return ((mirrored + math.pi) % (2 * math.pi)) - math.pi


def _x_extent(observation: dict[str, Any]) -> float:
    if ai_config.NORMALIZE_POSITIONS:
        return 1.0
    return float(observation.get("field_width", game_config.FIELD_WIDTH))


def _paddle_width_extent(observation: dict[str, Any]) -> float:
    if ai_config.NORMALIZE_POSITIONS:
        field_width = float(observation.get("field_width", game_config.FIELD_WIDTH))
        return game_config.PADDLE_WIDTH / field_width
    return game_config.PADDLE_WIDTH
