"""Record a side-by-side or sequential video of model learning stages."""

from __future__ import annotations

import argparse
import os
import subprocess
from pathlib import Path
from typing import Any

# Ensure headless execution before importing pygame
os.environ["SDL_VIDEODRIVER"] = "dummy"

import pygame  # noqa: E402

from magic_pong.ai.models.dqn_ai import DQNAgent  # noqa: E402
from magic_pong.ai.models.simple_ai import create_ai  # noqa: E402
from magic_pong.core.game_engine import GameEngine  # noqa: E402
from magic_pong.gui.pygame_renderer import PygameRenderer  # noqa: E402
from magic_pong.utils.config import game_config  # noqa: E402

BANNER_HEIGHT = 80


def draw_stage_header(
    canvas: pygame.Surface,
    width: int,
    banner_height: int,
    stage_idx: int,
    total_stages: int,
    stage_tag: str,
    title: str,
    subtitle: str,
    font_bold: pygame.font.Font,
    font_small: pygame.font.Font,
) -> None:
    """Draw a dedicated presentation banner completely above the game field."""
    banner_rect = pygame.Rect(0, 0, width, banner_height)
    canvas.fill((9, 13, 22), banner_rect)

    # Separator line between banner and game field
    pygame.draw.line(canvas, (45, 60, 85), (0, banner_height - 1), (width, banner_height - 1), 2)

    # Badge Pill
    badge_text = f"STAGE {stage_idx}/{total_stages} • {stage_tag.upper()}"
    badge_surf = font_bold.render(badge_text, True, (255, 214, 95))
    badge_bg = pygame.Surface(
        (badge_surf.get_width() + 16, badge_surf.get_height() + 8), pygame.SRCALPHA
    )
    badge_bg.fill((30, 40, 56))
    pygame.draw.rect(badge_bg, (255, 214, 95), badge_bg.get_rect(), 1, border_radius=4)

    b_x, b_y = 20, 12
    canvas.blit(badge_bg, (b_x, b_y))
    canvas.blit(badge_surf, (b_x + 8, b_y + 4))

    # Title text
    title_surf = font_bold.render(f"—  {title}", True, (245, 250, 255))
    canvas.blit(title_surf, (b_x + badge_bg.get_width() + 14, b_y + 4))

    # Subtitle / behavior description
    sub_surf = font_small.render(subtitle, True, (155, 175, 205))
    canvas.blit(sub_surf, (22, 46))


def draw_paddle_labels(
    screen: pygame.Surface,
    game_state: dict[str, Any],
    font: pygame.font.Font,
) -> None:
    """Draw floating indicator tags above the paddles to clearly differentiate them."""
    p1_pos = game_state["player1_position"]
    p2_pos = game_state["player2_position"]

    # Player 1 (Left) - AI Model Under Test
    p1_label = font.render("◀ AI MODEL", True, (0, 255, 200))
    p1_bg = pygame.Surface((p1_label.get_width() + 10, p1_label.get_height() + 4), pygame.SRCALPHA)
    p1_bg.fill((0, 30, 25, 200))
    pygame.draw.rect(p1_bg, (0, 255, 200), p1_bg.get_rect(), 1, border_radius=4)

    y1 = max(10, int(p1_pos[1]) - 22)
    screen.blit(p1_bg, (int(p1_pos[0]) + 20, y1))
    screen.blit(p1_label, (int(p1_pos[0]) + 25, y1 + 2))

    # Player 2 (Right) - Sparring Opponent
    p2_label = font.render("OPPONENT ▶", True, (160, 175, 195))
    p2_bg = pygame.Surface((p2_label.get_width() + 10, p2_label.get_height() + 4), pygame.SRCALPHA)
    p2_bg.fill((20, 25, 35, 200))
    pygame.draw.rect(p2_bg, (100, 115, 135), p2_bg.get_rect(), 1, border_radius=4)

    y2 = max(10, int(p2_pos[1]) - 22)
    x2 = int(p2_pos[0]) - p2_bg.get_width() - 10
    screen.blit(p2_bg, (x2, y2))
    screen.blit(p2_label, (x2 + 5, y2 + 2))


def record_progression_video(
    stages: list[dict[str, Any]],
    output_path: Path,
    seconds_per_stage: float = 8.0,
    opponent_type: str = "tracker",
    fps: int = 60,
    banner_height: int = BANNER_HEIGHT,
) -> None:
    """Simulate each stage and pipe rendered frames to ffmpeg."""
    field_width = game_config.FIELD_WIDTH
    field_height = game_config.FIELD_HEIGHT
    canvas_width = field_width
    canvas_height = field_height + banner_height
    frames_per_stage = int(seconds_per_stage * fps)

    renderer = PygameRenderer()
    canvas = pygame.Surface((canvas_width, canvas_height))

    font_bold = pygame.font.Font(None, 24)
    font_small = pygame.font.Font(None, 19)
    font_tag = pygame.font.Font(None, 16)

    # Setup ffmpeg process
    cmd = [
        "ffmpeg",
        "-y",
        "-f",
        "rawvideo",
        "-vcodec",
        "rawvideo",
        "-s",
        f"{canvas_width}x{canvas_height}",
        "-pix_fmt",
        "rgb24",
        "-r",
        str(fps),
        "-i",
        "-",
        "-c:v",
        "libx264",
        "-pix_fmt",
        "yuv420p",
        "-preset",
        "medium",
        "-crf",
        "18",
        str(output_path),
    ]

    print(f"🎬 Starting video render: {output_path}")
    print(
        f"   Video Resolution: {canvas_width}x{canvas_height} (Field: {field_width}x{field_height}, Header: {banner_height}px) @ {fps} FPS"
    )
    print(
        f"   Stages: {len(stages)} ({seconds_per_stage}s each, {len(stages) * seconds_per_stage:.1f}s total)"
    )

    ffmpeg_proc = subprocess.Popen(
        cmd, stdin=subprocess.PIPE, stdout=subprocess.DEVNULL, stderr=subprocess.PIPE
    )
    assert ffmpeg_proc.stdin is not None

    try:
        for idx, stage in enumerate(stages, start=1):
            ckpt_path = stage["checkpoint"]
            print(f"\n▶ [{idx}/{len(stages)}] Recording: {stage['title']}")

            # Create DQN Agent
            agent = DQNAgent(
                geometry_context=True,
                name=stage["title"],
            )
            if ckpt_path and os.path.isfile(ckpt_path):
                agent.load_model(ckpt_path)

            # Check if this stage should use true random exploration
            is_random_stage = stage.get("explore_random", False)
            if is_random_stage:
                agent.epsilon = 1.0
                agent.set_training_mode(False)
                agent.set_exploration_mode(True)
            else:
                agent.set_training_mode(False)
                agent.set_exploration_mode(False)

            # Optional geometric shield for the final stage if desired
            if stage.get("shield"):
                from magic_pong.ai.action_shaping import GeometricShieldConfig

                agent.configure_action_shaper(GeometricShieldConfig(margin=20.0, soft_x_tau=0.5))

            opponent = create_ai(opponent_type, name=f"Opponent ({opponent_type})")

            engine = GameEngine(headless=True)
            engine.set_players(agent, opponent)
            engine.start_game()
            # Serve ball towards Player 1 (the model) to immediately test its defense
            engine.physics_engine.reset_ball(-1)

            # Record frames for this stage
            for f in range(frames_per_stage):
                res = engine.update()
                game_state = res["game_state"]

                # 1. Render game field cleanly on renderer.screen (800x600)
                renderer.render_game_state(game_state)

                # 2. Draw floating paddle labels (AI Model vs Opponent)
                draw_paddle_labels(renderer.screen, game_state, font_tag)

                # 3. Blit game field to canvas below the banner
                canvas.blit(renderer.screen, (0, banner_height))

                # 4. Draw the header banner completely outside the field at the top
                draw_stage_header(
                    canvas,
                    canvas_width,
                    banner_height,
                    idx,
                    len(stages),
                    stage["tag"],
                    stage["title"],
                    stage["subtitle"],
                    font_bold,
                    font_small,
                )

                # Send composite canvas frame to ffmpeg
                raw_bytes = pygame.image.tobytes(canvas, "RGB")
                ffmpeg_proc.stdin.write(raw_bytes)

                if (f + 1) % (fps * 2) == 0:
                    print(f"   Rendered {(f + 1) / fps:.1f}s / {seconds_per_stage:.1f}s...")

        ffmpeg_proc.stdin.close()
        ffmpeg_proc.wait()
        print(
            f"\n✅ Video successfully generated: {output_path} ({os.path.getsize(output_path) / (1024 * 1024):.2f} MB)"
        )

    except Exception as e:
        ffmpeg_proc.kill()
        raise e


def main() -> None:
    parser = argparse.ArgumentParser(
        description="Record a 4-stage model learning progression video."
    )
    parser.add_argument(
        "--output", type=str, default="learning_progression.mp4", help="Output MP4 video file path"
    )
    parser.add_argument(
        "--seconds_per_stage", type=float, default=8.0, help="Seconds to record per stage"
    )
    parser.add_argument(
        "--opponent",
        type=str,
        default="tracker",
        choices=["follow_ball", "tracker", "aggressive", "striker", "defensive"],
        help="Sparring opponent AI",
    )
    parser.add_argument(
        "--stage1",
        type=str,
        default="models/progression_demo/stage1_random_untrained.pth",
        help="Stage 1 model checkpoint",
    )
    parser.add_argument(
        "--stage2",
        type=str,
        default="models/progression_demo/checkpoint_0000500387.pth",
        help="Stage 2 model checkpoint",
    )
    parser.add_argument(
        "--stage3",
        type=str,
        default="models/progression_demo/checkpoint_0003000427.pth",
        help="Stage 3 model checkpoint",
    )
    parser.add_argument(
        "--stage4", type=str, default="models/dqn_v7_promoted.pth", help="Stage 4 model checkpoint"
    )
    args = parser.parse_args()

    stages = [
        {
            "tag": "0 Frames",
            "title": "Random Exploration (Untrained Weights)",
            "subtitle": "Left paddle is the AI Model: random uncoordinated actions (eps=1.0), misses the ball.",
            "checkpoint": args.stage1,
            "explore_random": True,
            "shield": False,
        },
        {
            "tag": "500k Frames",
            "title": "Directional Tracking (Coarse & Erratic)",
            "subtitle": "Learns to chase the ball vertically; jittery movements and reacts too late on bounces.",
            "checkpoint": args.stage2,
            "explore_random": False,
            "shield": False,
        },
        {
            "tag": "3M Frames",
            "title": "Basic Rallying Competency (Intermediate)",
            "subtitle": "Sustains standard rallies on direct shots; vulnerable to sharp wall reflections.",
            "checkpoint": args.stage3,
            "explore_random": False,
            "shield": False,
        },
        {
            "tag": "Promoted Champion",
            "title": "Champion Mastery (Multi-Bounce Geometry)",
            "subtitle": "Anticipates wall bounces in advance, sharp corner cuts, and impenetrable defense.",
            "checkpoint": args.stage4,
            "explore_random": False,
            "shield": True,
        },
    ]

    record_progression_video(
        stages=stages,
        output_path=Path(args.output),
        seconds_per_stage=args.seconds_per_stage,
        opponent_type=args.opponent,
    )


if __name__ == "__main__":
    main()
