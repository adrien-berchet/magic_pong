"""
PyGame renderer for Magic Pong game
"""

import math
from typing import Any

import pygame

from magic_pong.core.entities import Ball
from magic_pong.core.entities import Bonus
from magic_pong.core.entities import BonusType
from magic_pong.core.entities import Paddle
from magic_pong.core.entities import RotatingPaddle
from magic_pong.utils.config import ai_config
from magic_pong.utils.config import game_config


class PygameRenderer:
    """PyGame-based renderer for Magic Pong"""

    def __init__(self, width: int | None = None, height: int | None = None):
        """Initialize the PyGame renderer"""
        self.width = width or game_config.FIELD_WIDTH
        self.height = height or game_config.FIELD_HEIGHT

        # Initialize PyGame
        pygame.init()

        # Create the display
        self.screen = pygame.display.set_mode((self.width, self.height))
        pygame.display.set_caption("Magic Pong")

        # Clock for controlling frame rate
        self.clock = pygame.time.Clock()

        # Default bonus colors
        default_bonus_colors: dict[str, tuple[int, int, int]] = {
            "enlarge_paddle": (0, 255, 0),
            "shrink_opponent": (255, 0, 0),
            "rotating_paddle": (0, 0, 255),
        }

        # Colors - use typed dictionary
        self.background_color: tuple[int, int, int] = game_config.BACKGROUND_COLOR
        self.ball_color: tuple[int, int, int] = game_config.BALL_COLOR
        self.paddle_color: tuple[int, int, int] = game_config.PADDLE_COLOR
        self.bonus_colors: dict[str, tuple[int, int, int]] = (
            game_config.BONUS_COLORS if game_config.BONUS_COLORS else default_bonus_colors
        )
        self.text_color: tuple[int, int, int] = (255, 255, 255)
        self.line_color: tuple[int, int, int] = (100, 100, 100)
        self.optimal_point_p1_color: tuple[int, int, int] = (255, 100, 100)
        self.optimal_point_p2_color: tuple[int, int, int] = (100, 255, 100)

        # Font for text rendering
        self.font_large = pygame.font.Font(None, 74)
        self.font_medium = pygame.font.Font(None, 48)
        self.font_small = pygame.font.Font(None, 36)

        # UI state
        self.show_fps = False
        self.show_debug = False

    def clear_screen(self) -> None:
        """Clear the screen with background color"""
        self.screen.fill(self.background_color)

    def draw_field(self) -> None:
        """Draw the game field (center line, borders)"""
        # Center line
        center_x = self.width // 2
        pygame.draw.line(self.screen, self.line_color, (center_x, 0), (center_x, self.height), 2)

        # Center circle
        pygame.draw.circle(self.screen, self.line_color, (center_x, self.height // 2), 50, 2)

    def draw_ball(self, ball: Ball) -> None:
        """Draw the game ball"""
        pos = (int(ball.position.x), int(ball.position.y))
        pygame.draw.circle(self.screen, self.ball_color, pos, int(ball.radius))

    def draw_optimal_point(
        self, position: tuple, player_id: int, is_approaching: bool = False
    ) -> None:
        """Draw an optimal interception point as a virtual ball"""
        if not ai_config.SHOW_OPTIMAL_POINTS_GUI:
            return

        pos = (int(position[0]), int(position[1]))
        color = self.optimal_point_p1_color if player_id == 1 else self.optimal_point_p2_color

        # Draw the optimal point as a semi-transparent circle
        radius = 12 if is_approaching else 8

        # Create a surface for alpha blending
        point_surface = pygame.Surface((radius * 2, radius * 2), pygame.SRCALPHA)

        # Draw filled circle with transparency
        alpha = 180 if is_approaching else 120
        color_with_alpha = (*color, alpha)
        pygame.draw.circle(point_surface, color_with_alpha, (radius, radius), radius)

        # Draw border for better visibility
        border_color = (*color, 255)
        pygame.draw.circle(point_surface, border_color, (radius, radius), radius, 2)

        # Add pulsing effect for approaching balls
        if is_approaching:
            pulse = abs(math.sin(pygame.time.get_ticks() * 0.008)) * 3
            pygame.draw.circle(
                point_surface, border_color, (radius, radius), int(radius + pulse), 1
            )

        # Blit to main screen
        self.screen.blit(point_surface, (pos[0] - radius, pos[1] - radius))

    def draw_paddle(self, paddle: Paddle) -> None:
        """Draw a player paddle"""
        rect = pygame.Rect(
            int(paddle.position.x), int(paddle.position.y), int(paddle.width), int(paddle.height)
        )
        pygame.draw.rect(self.screen, self.paddle_color, rect)

        # Draw size effect indicator
        if paddle.size_effect_timer > 0:
            # Draw a glowing effect
            glow_color = (255, 255, 0) if paddle.height > paddle.original_height else (255, 0, 0)
            glow_rect = pygame.Rect(rect.x - 3, rect.y - 3, rect.width + 6, rect.height + 6)
            pygame.draw.rect(self.screen, glow_color, glow_rect, 3)

    def draw_rotating_paddle(self, rpaddle: RotatingPaddle) -> None:
        """Draw a rotating paddle"""
        segments = rpaddle.get_line_segments()
        for p1, p2 in segments:
            pygame.draw.line(
                self.screen,
                (0, 150, 255),  # Blue color for rotating paddle
                (int(p1.x), int(p1.y)),
                (int(p2.x), int(p2.y)),
                int(rpaddle.thickness),
            )

    def draw_bonus(self, bonus: Bonus) -> None:
        """Draw a bonus item"""
        color = self.bonus_colors.get(bonus.type.value, (255, 255, 255))

        # Draw bonus as a diamond shape
        half_size = bonus.size // 2
        center_x = int(bonus.position.x)
        center_y = int(bonus.position.y)

        points = [
            (center_x, center_y - half_size),  # Top
            (center_x + half_size, center_y),  # Right
            (center_x, center_y + half_size),  # Bottom
            (center_x - half_size, center_y),  # Left
        ]

        pygame.draw.polygon(self.screen, color, points)

        # Add pulsing effect
        pulse = abs(math.sin(pygame.time.get_ticks() * 0.005)) * 5
        pygame.draw.polygon(self.screen, color, points, int(2 + pulse))

    def draw_score(self, score: tuple[int, int]) -> None:
        """Draw the current score"""
        score_text = f"{score[0]}  -  {score[1]}"
        text_surface = self.font_large.render(score_text, True, self.text_color)
        text_rect = text_surface.get_rect()
        text_rect.centerx = self.width // 2
        text_rect.top = 20
        self.screen.blit(text_surface, text_rect)

    def draw_ui_info(self, info: dict[str, Any]) -> None:
        """Draw additional UI information"""
        y_offset = 20

        if self.show_fps:
            fps = self.clock.get_fps()
            fps_text = f"FPS: {fps:.0f}"
            fps_surface = self.font_small.render(fps_text, True, self.text_color)
            self.screen.blit(fps_surface, (10, y_offset))
            y_offset += 30

        if self.show_debug and "debug_info" in info:
            for key, value in info["debug_info"].items():
                debug_text = f"{key}: {value}"
                debug_surface = self.font_small.render(debug_text, True, self.text_color)
                self.screen.blit(debug_surface, (10, y_offset))
                y_offset += 25

    def draw_game_over(self, winner: int, score: tuple[int, int]) -> None:
        """Draw game over screen"""
        # Semi-transparent overlay
        overlay = pygame.Surface((self.width, self.height))
        overlay.set_alpha(180)
        overlay.fill((0, 0, 0))
        self.screen.blit(overlay, (0, 0))

        # Winner text
        if winner == 0:
            winner_text = "Draw!"
        else:
            winner_text = f"Player {winner} wins!"

        winner_surface = self.font_large.render(winner_text, True, self.text_color)
        winner_rect = winner_surface.get_rect()
        winner_rect.center = (self.width // 2, self.height // 2 - 50)
        self.screen.blit(winner_surface, winner_rect)

        # Final score
        score_text = f"Final score: {score[0]} - {score[1]}"
        score_surface = self.font_medium.render(score_text, True, self.text_color)
        score_rect = score_surface.get_rect()
        score_rect.center = (self.width // 2, self.height // 2 + 20)
        self.screen.blit(score_surface, score_rect)

        # Instructions
        restart_text = "Press SPACE to play again or ESC to quit"
        restart_surface = self.font_small.render(restart_text, True, self.text_color)
        restart_rect = restart_surface.get_rect()
        restart_rect.center = (self.width // 2, self.height // 2 + 80)
        self.screen.blit(restart_surface, restart_rect)

    def draw_pause_screen(self) -> None:
        """Draw pause screen"""
        # Semi-transparent overlay
        overlay = pygame.Surface((self.width, self.height))
        overlay.set_alpha(128)
        overlay.fill((0, 0, 0))
        self.screen.blit(overlay, (0, 0))

        # Pause text
        pause_surface = self.font_large.render("PAUSE", True, self.text_color)
        pause_rect = pause_surface.get_rect()
        pause_rect.center = (self.width // 2, self.height // 2)
        self.screen.blit(pause_surface, pause_rect)

        # Instructions
        instructions = "Press P or SPACE to continue"
        inst_surface = self.font_small.render(instructions, True, self.text_color)
        inst_rect = inst_surface.get_rect()
        inst_rect.center = (self.width // 2, self.height // 2 + 60)
        self.screen.blit(inst_surface, inst_rect)

    def draw_menu(self, selected_option: int = 0) -> None:
        """Draw the main menu"""
        self.clear_screen()

        # Title
        title_surface = self.font_large.render("MAGIC PONG", True, self.text_color)
        title_rect = title_surface.get_rect()
        title_rect.center = (self.width // 2, self.height // 4)
        self.screen.blit(title_surface, title_rect)

        # Menu options
        options = [
            "1 vs 1 (Two players)",
            "1 vs AI (Player vs AI)",
            "Play vs Trained Model",
            "AI vs AI (Demonstration)",
            "Settings",
            "Quit",
        ]

        for i, option in enumerate(options):
            color = (255, 255, 0) if i == selected_option else self.text_color
            option_surface = self.font_medium.render(option, True, color)
            option_rect = option_surface.get_rect()
            option_rect.center = (self.width // 2, self.height // 2 + i * 60)
            self.screen.blit(option_surface, option_rect)

        # Instructions
        instructions = "Use UP/DOWN arrows and ENTER to select"
        inst_surface = self.font_small.render(instructions, True, self.text_color)
        inst_rect = inst_surface.get_rect()
        inst_rect.center = (self.width // 2, self.height * 3 // 4 + 50)
        self.screen.blit(inst_surface, inst_rect)

    def draw_model_selection_menu(self, available_models: list, selected_model: int = 0) -> None:
        """Draw the model selection menu"""
        self.clear_screen()

        # Title
        title_surface = self.font_large.render("SELECT TRAINED MODEL", True, self.text_color)
        title_rect = title_surface.get_rect()
        title_rect.center = (self.width // 2, self.height // 6)
        self.screen.blit(title_surface, title_rect)

        # Check if models are available
        if not available_models:
            no_models_surface = self.font_medium.render(
                "No trained models found!", True, (255, 100, 100)
            )
            no_models_rect = no_models_surface.get_rect()
            no_models_rect.center = (self.width // 2, self.height // 2)
            self.screen.blit(no_models_surface, no_models_rect)

            back_surface = self.font_small.render(
                "Press ESCAPE to return to main menu", True, self.text_color
            )
            back_rect = back_surface.get_rect()
            back_rect.center = (self.width // 2, self.height // 2 + 60)
            self.screen.blit(back_surface, back_rect)
            return

        # Display models list
        start_y = self.height // 3
        max_visible = min(6, len(available_models))  # Show max 6 models at once

        # Calculate scroll offset if there are more models than can fit
        scroll_offset = 0
        if selected_model >= max_visible - 1:
            scroll_offset = selected_model - max_visible + 1

        for i in range(max_visible):
            model_index = i + scroll_offset
            if model_index >= len(available_models):
                break

            model = available_models[model_index]
            is_selected = model_index == selected_model

            # Model name
            color = (255, 255, 0) if is_selected else self.text_color
            model_surface = self.font_medium.render(model["name"], True, color)
            model_rect = model_surface.get_rect()
            model_rect.center = (self.width // 2, start_y + i * 50)
            self.screen.blit(model_surface, model_rect)

            # Model file name (smaller text)
            if is_selected:
                file_surface = self.font_small.render(f"({model['file']})", True, (180, 180, 180))
                file_rect = file_surface.get_rect()
                file_rect.center = (self.width // 2, start_y + i * 50 + 25)
                self.screen.blit(file_surface, file_rect)

        # Scroll indicators
        if scroll_offset > 0:
            up_arrow = self.font_small.render("↑ More models above", True, (150, 150, 150))
            up_rect = up_arrow.get_rect()
            up_rect.center = (self.width // 2, start_y - 30)
            self.screen.blit(up_arrow, up_rect)

        if scroll_offset + max_visible < len(available_models):
            down_arrow = self.font_small.render("↓ More models below", True, (150, 150, 150))
            down_rect = down_arrow.get_rect()
            down_rect.center = (self.width // 2, start_y + max_visible * 50 + 20)
            self.screen.blit(down_arrow, down_rect)

        # Instructions
        instructions = [
            "Use UP/DOWN arrows to select",
            "Press ENTER to load model",
            "Press ESCAPE to return to main menu",
        ]

        for i, instruction in enumerate(instructions):
            inst_surface = self.font_small.render(instruction, True, self.text_color)
            inst_rect = inst_surface.get_rect()
            inst_rect.center = (self.width // 2, self.height * 4 // 5 + i * 25)
            self.screen.blit(inst_surface, inst_rect)

    def draw_error_message(self, message: str) -> None:
        """Draw an error message overlay"""
        # Semi-transparent background
        overlay = pygame.Surface((self.width, self.height))
        overlay.set_alpha(128)
        overlay.fill((0, 0, 0))
        self.screen.blit(overlay, (0, 0))

        # Error box background
        box_width = min(600, self.width - 100)
        box_height = 120
        box_x = (self.width - box_width) // 2
        box_y = (self.height - box_height) // 2

        error_box = pygame.Surface((box_width, box_height))
        error_box.fill((139, 0, 0))  # Dark red
        pygame.draw.rect(error_box, (255, 0, 0), error_box.get_rect(), 3)  # Red border
        self.screen.blit(error_box, (box_x, box_y))

        # Error title
        title_surface = self.font_medium.render("ERROR", True, (255, 255, 255))
        title_rect = title_surface.get_rect()
        title_rect.center = (self.width // 2, box_y + 25)
        self.screen.blit(title_surface, title_rect)

        # Error message (word wrap for long messages)
        words = message.split(" ")
        lines = []
        current_line: list[str] = []

        for word in words:
            test_line = " ".join(current_line + [word])
            text_width = self.font_small.size(test_line)[0]

            if text_width <= box_width - 40:  # 20px margin on each side
                current_line.append(word)
            else:
                if current_line:
                    lines.append(" ".join(current_line))
                current_line = [word]

        if current_line:
            lines.append(" ".join(current_line))

        # Display message lines
        for i, line in enumerate(lines[:3]):  # Max 3 lines
            line_surface = self.font_small.render(line, True, (255, 255, 255))
            line_rect = line_surface.get_rect()
            line_rect.center = (self.width // 2, box_y + 55 + i * 20)
            self.screen.blit(line_surface, line_rect)

    def draw_controls_help(self) -> None:
        """Draw controls help overlay"""
        help_lines = [
            "CONTROLS:",
            "",
            "Player 1 (Left):",
            "  W/S - Up/Down",
            "  A/D - Left/Right",
            "",
            "Player 2 (Right):",
            "  ↑/↓ - Up/Down",
            "  ←/→ - Left/Right",
            "",
            "General:",
            "  P - Pause",
            "  F1 - Show this help",
            "  F2 - Show FPS",
            "  F3 - Debug mode",
            "  ESC - Main menu",
        ]

        # Background
        help_width = 400
        help_height = len(help_lines) * 25 + 40
        help_x = (self.width - help_width) // 2
        help_y = (self.height - help_height) // 2

        help_bg = pygame.Surface((help_width, help_height))
        help_bg.set_alpha(220)
        help_bg.fill((20, 20, 20))
        self.screen.blit(help_bg, (help_x, help_y))

        # Border
        pygame.draw.rect(self.screen, self.text_color, (help_x, help_y, help_width, help_height), 2)

        # Text
        for i, line in enumerate(help_lines):
            if line.startswith("CONTROLS:") or line.startswith("General:"):
                color = (255, 255, 0)
                font = self.font_medium
            elif line.startswith("Player"):
                color = (0, 255, 255)
                font = self.font_small
            else:
                color = self.text_color
                font = self.font_small

            if line.strip():  # Skip empty lines
                text_surface = font.render(line, True, color)
                self.screen.blit(text_surface, (help_x + 20, help_y + 20 + i * 25))

    def render_game_state(
        self, game_state: dict[str, Any], additional_info: dict[str, Any] | None = None
    ) -> None:
        """Render the complete game state"""
        self.clear_screen()
        self.draw_field()

        # Draw ball
        ball_pos = game_state["ball_position"]
        ball_vel = game_state["ball_velocity"]
        ball = Ball(ball_pos[0], ball_pos[1], ball_vel[0], ball_vel[1])
        self.draw_ball(ball)

        # Draw optimal points (before paddles for better visual layering)
        if additional_info and "optimal_points" in additional_info:
            optimal_points = additional_info["optimal_points"]
            for player_id, point_data in optimal_points.items():
                if point_data and "position" in point_data:
                    ball_vel_data = point_data.get("ball_velocity", (0, 0))
                    # Check if ball is moving towards this player
                    is_approaching = (player_id == 1 and ball_vel_data[0] < 0) or (
                        player_id == 2 and ball_vel_data[0] > 0
                    )
                    self.draw_optimal_point(point_data["position"], player_id, is_approaching)

        # Draw paddles
        p1_pos = game_state["player1_position"]
        p2_pos = game_state["player2_position"]

        paddle1 = Paddle(p1_pos[0], p1_pos[1], 1)
        paddle1.height = game_state["player1_paddle_size"]
        paddle2 = Paddle(p2_pos[0], p2_pos[1], 2)
        paddle2.height = game_state["player2_paddle_size"]

        self.draw_paddle(paddle1)
        self.draw_paddle(paddle2)

        # Draw bonuses
        for bonus_data in game_state["active_bonuses"]:
            bonus_pos, bonus_type = bonus_data[0:2], bonus_data[2]
            bonus = Bonus(bonus_pos[0], bonus_pos[1], BonusType(bonus_type))
            self.draw_bonus(bonus)

        # Draw rotating paddles
        for rp_data in game_state["rotating_paddles"]:
            rp_pos, rp_angle = rp_data[0:2], rp_data[2]
            # Note: We'd need player_id from game state for proper RotatingPaddle creation
            rpaddle = RotatingPaddle(rp_pos[0], rp_pos[1], 1)  # Assume player 1 for now
            rpaddle.angle = rp_angle
            self.draw_rotating_paddle(rpaddle)

        # Draw score
        score = game_state["score"]
        self.draw_score(score)

        # Draw additional UI info
        if additional_info:
            self.draw_ui_info(additional_info)

    def present(self) -> None:
        """Present the rendered frame"""
        pygame.display.flip()

    def update(self, fps: int | None = None) -> None:
        """Update the display and maintain frame rate"""
        fps = fps or game_config.FPS
        self.clock.tick(fps)

    def toggle_fps_display(self) -> None:
        """Toggle FPS display"""
        self.show_fps = not self.show_fps

    def toggle_debug_display(self) -> None:
        """Toggle debug information display"""
        self.show_debug = not self.show_debug

    def draw_config_category_menu(self, categories: list, selected_index: int) -> None:
        """Draw the configuration category selection menu"""
        self.clear_screen()

        # Title
        title_surface = self.font_large.render("SETTINGS", True, self.text_color)
        title_rect = title_surface.get_rect()
        title_rect.center = (self.width // 2, self.height // 6)
        self.screen.blit(title_surface, title_rect)

        # Categories
        start_y = self.height // 3
        for i, category_name in enumerate(categories):
            color = (255, 255, 0) if i == selected_index else self.text_color
            category_surface = self.font_medium.render(category_name, True, color)
            category_rect = category_surface.get_rect()
            category_rect.center = (self.width // 2, start_y + i * 60)
            self.screen.blit(category_surface, category_rect)

        # Instructions
        instructions = [
            "Use UP/DOWN to navigate",
            "Press ENTER to select category",
            "Press S to SAVE configuration",
            "Press R to RESET to defaults",
            "Press ESC to return to menu",
        ]

        inst_y = self.height * 4 // 5
        for i, instruction in enumerate(instructions):
            inst_surface = self.font_small.render(instruction, True, self.text_color)
            inst_rect = inst_surface.get_rect()
            inst_rect.center = (self.width // 2, inst_y + i * 25)
            self.screen.blit(inst_surface, inst_rect)

    def draw_config_option_menu(
        self,
        category_name: str,
        options: list[dict],
        selected_index: int,
        is_editing: bool = False,
    ) -> None:
        """Draw the configuration options menu for a specific category"""
        self.clear_screen()

        # Title
        title_surface = self.font_large.render(f"{category_name} Settings", True, self.text_color)
        title_rect = title_surface.get_rect()
        title_rect.center = (self.width // 2, 60)
        self.screen.blit(title_surface, title_rect)

        # Options
        start_y = 140
        max_visible = 8  # Maximum options visible at once

        # Calculate scroll offset
        scroll_offset = 0
        if selected_index >= max_visible - 1:
            scroll_offset = selected_index - max_visible + 1

        for i in range(max_visible):
            option_index = i + scroll_offset
            if option_index >= len(options):
                break

            option = options[option_index]
            is_selected = option_index == selected_index
            y_pos = start_y + i * 55

            # Option label
            label_color = (255, 255, 0) if is_selected else self.text_color
            if is_editing and is_selected:
                label_color = (0, 255, 0)  # Green when editing

            label_surface = self.font_medium.render(option["label"], True, label_color)
            label_rect = label_surface.get_rect()
            label_rect.midleft = (100, y_pos)
            self.screen.blit(label_surface, label_rect)

            # Option value
            value_text = self._format_config_value(option["value"], option["field_type"])
            value_color = (0, 255, 0) if (is_editing and is_selected) else (200, 200, 200)
            value_surface = self.font_medium.render(value_text, True, value_color)
            value_rect = value_surface.get_rect()
            value_rect.midright = (self.width - 100, y_pos)
            self.screen.blit(value_surface, value_rect)

            # Show range for selected numeric options
            if is_selected and option["field_type"] == "numeric":
                range_text = f"[{option['min_value']:.0f} - {option['max_value']:.0f}]"
                range_surface = self.font_small.render(range_text, True, (150, 150, 150))
                range_rect = range_surface.get_rect()
                range_rect.midright = (self.width - 100, y_pos + 25)
                self.screen.blit(range_surface, range_rect)

        # Scroll indicators
        if scroll_offset > 0:
            up_arrow = self.font_small.render("↑ More options above", True, (150, 150, 150))
            up_rect = up_arrow.get_rect()
            up_rect.center = (self.width // 2, start_y - 20)
            self.screen.blit(up_arrow, up_rect)

        if scroll_offset + max_visible < len(options):
            down_arrow = self.font_small.render("↓ More options below", True, (150, 150, 150))
            down_rect = down_arrow.get_rect()
            down_rect.center = (self.width // 2, start_y + max_visible * 55 + 10)
            self.screen.blit(down_arrow, down_rect)

        # Instructions
        inst_y = self.height - 120
        if is_editing:
            instructions = [
                "LEFT/RIGHT to change value",
                "ENTER to confirm",
                "ESC to cancel",
            ]
        else:
            instructions = [
                "UP/DOWN to navigate",
                "ENTER to edit value",
                "ESC to return",
            ]

        for i, instruction in enumerate(instructions):
            inst_surface = self.font_small.render(instruction, True, self.text_color)
            inst_rect = inst_surface.get_rect()
            inst_rect.center = (self.width // 2, inst_y + i * 25)
            self.screen.blit(inst_surface, inst_rect)

    def _format_config_value(self, value: Any, field_type: str) -> str:
        """Format a configuration value for display"""
        if field_type == "boolean":
            return "ON" if value else "OFF"
        elif field_type == "numeric":
            if isinstance(value, float):
                return f"{value:.2f}"
            return str(value)
        elif field_type == "selection":
            return str(value).upper()
        else:
            return str(value)

    def draw_confirmation_dialog(self, message: str, title: str = "Confirm") -> None:
        """Draw a confirmation dialog"""
        # Semi-transparent overlay
        overlay = pygame.Surface((self.width, self.height))
        overlay.set_alpha(180)
        overlay.fill((0, 0, 0))
        self.screen.blit(overlay, (0, 0))

        # Dialog box
        box_width = min(500, self.width - 100)
        box_height = 200
        box_x = (self.width - box_width) // 2
        box_y = (self.height - box_height) // 2

        dialog_box = pygame.Surface((box_width, box_height))
        dialog_box.fill((40, 40, 40))
        pygame.draw.rect(dialog_box, (255, 255, 0), dialog_box.get_rect(), 3)
        self.screen.blit(dialog_box, (box_x, box_y))

        # Title
        title_surface = self.font_medium.render(title, True, (255, 255, 0))
        title_rect = title_surface.get_rect()
        title_rect.center = (self.width // 2, box_y + 40)
        self.screen.blit(title_surface, title_rect)

        # Message (word wrap)
        words = message.split(" ")
        lines = []
        current_line: list[str] = []

        for word in words:
            test_line = " ".join(current_line + [word])
            text_width = self.font_small.size(test_line)[0]

            if text_width <= box_width - 40:
                current_line.append(word)
            else:
                if current_line:
                    lines.append(" ".join(current_line))
                current_line = [word]

        if current_line:
            lines.append(" ".join(current_line))

        # Display message
        for i, line in enumerate(lines[:2]):  # Max 2 lines
            line_surface = self.font_small.render(line, True, self.text_color)
            line_rect = line_surface.get_rect()
            line_rect.center = (self.width // 2, box_y + 85 + i * 30)
            self.screen.blit(line_surface, line_rect)

        # Instructions
        confirm_text = "Press ENTER to confirm, ESC to cancel"
        confirm_surface = self.font_small.render(confirm_text, True, (200, 200, 200))
        confirm_rect = confirm_surface.get_rect()
        confirm_rect.center = (self.width // 2, box_y + box_height - 30)
        self.screen.blit(confirm_surface, confirm_rect)

    def cleanup(self) -> None:
        """Clean up PyGame resources"""
        pygame.quit()
