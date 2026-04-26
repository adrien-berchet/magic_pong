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
        self.font_config_title = pygame.font.Font(None, 54)
        self.font_config_section = pygame.font.Font(None, 34)
        self.font_config_body = pygame.font.Font(None, 28)
        self.font_config_hint = pygame.font.Font(None, 24)
        self.font_config_micro = pygame.font.Font(None, 20)

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

    def draw_pause_screen(self, actions: list[dict[str, Any]], selected_action: int = 0) -> None:
        """Draw pause action menu overlay."""
        # Semi-transparent overlay
        overlay = pygame.Surface((self.width, self.height))
        overlay.set_alpha(172)
        overlay.fill((0, 0, 0))
        self.screen.blit(overlay, (0, 0))

        panel_width = min(760, self.width - 48)
        panel_height = min(430, self.height - 48)
        panel_rect = pygame.Rect(0, 0, panel_width, panel_height)
        panel_rect.center = (self.width // 2, self.height // 2)
        self._draw_panel(panel_rect, (9, 13, 20), (58, 70, 88))

        title_surface = self.font_config_title.render("Paused", True, self.text_color)
        self.screen.blit(title_surface, (panel_rect.x + 24, panel_rect.y + 20))

        subtitle_surface = self.font_config_hint.render(
            "Choose an action for the current match", True, (154, 166, 184)
        )
        self.screen.blit(subtitle_surface, (panel_rect.x + 24, panel_rect.y + 62))

        body_top = panel_rect.y + 98
        footer_rect = pygame.Rect(
            panel_rect.x + 18, panel_rect.bottom - 64, panel_rect.width - 36, 46
        )
        list_rect = pygame.Rect(panel_rect.x + 18, body_top, 220, footer_rect.y - body_top - 14)
        detail_rect = pygame.Rect(
            list_rect.right + 14,
            body_top,
            panel_rect.right - list_rect.right - 32,
            list_rect.height,
        )

        self._draw_panel(list_rect, (14, 18, 27), (52, 63, 80))
        self._draw_panel(detail_rect, (11, 15, 23), (48, 58, 74))

        self._draw_option_rows(actions, selected_action, list_rect, "ACTIONS")
        self._draw_menu_detail(actions, selected_action, detail_rect)
        self._draw_hint_bar(
            footer_rect,
            [
                ("UP/DOWN", "Navigate"),
                ("ENTER", "Select"),
                ("P/SPACE", "Resume"),
                ("ESC", "Main menu"),
            ],
        )

    def draw_menu(self, menu_options: list[dict[str, Any]], selected_option: int = 0) -> None:
        """Draw the main menu as a two-panel control surface."""
        self.screen.fill((7, 10, 15))

        margin = max(18, min(32, self.width // 28))
        gap = max(12, min(18, self.width // 48))
        footer_height = 56
        body_top = 88
        body_bottom = self.height - footer_height - 18
        body_height = max(260, body_bottom - body_top)
        sidebar_width = max(210, min(280, self.width // 3))

        sidebar_rect = pygame.Rect(margin, body_top, sidebar_width, body_height)
        detail_rect = pygame.Rect(
            sidebar_rect.right + gap,
            body_top,
            self.width - sidebar_rect.right - gap - margin,
            body_height,
        )
        footer_rect = pygame.Rect(
            margin,
            self.height - footer_height - 10,
            self.width - 2 * margin,
            footer_height,
        )

        title_surface = self.font_config_title.render("Magic Pong", True, self.text_color)
        self.screen.blit(title_surface, (margin, 24))

        subtitle_surface = self.font_config_hint.render(
            "Choose a mode or open the control panel", True, (154, 166, 184)
        )
        subtitle_rect = subtitle_surface.get_rect()
        subtitle_rect.midleft = (margin + 214, 47)
        self.screen.blit(subtitle_surface, subtitle_rect)

        self._draw_panel(sidebar_rect, (14, 18, 27), (52, 63, 80))
        self._draw_panel(detail_rect, (11, 15, 23), (48, 58, 74))

        self._draw_option_rows(menu_options, selected_option, sidebar_rect, "MODES")
        self._draw_menu_detail(menu_options, selected_option, detail_rect)
        self._draw_hint_bar(
            footer_rect,
            [
                ("UP/DOWN", "Navigate"),
                ("ENTER/SPACE", "Select"),
                ("ESC", "Quit"),
            ],
        )

    def _draw_option_rows(
        self,
        options: list[dict[str, Any]],
        selected_index: int,
        panel_rect: pygame.Rect,
        heading: str,
    ) -> None:
        """Draw selectable menu rows inside a panel."""
        heading_surface = self.font_config_micro.render(heading, True, (126, 138, 156))
        self.screen.blit(heading_surface, (panel_rect.x + 16, panel_rect.y + 16))

        if not options:
            empty_surface = self.font_config_body.render("No options", True, (170, 180, 195))
            empty_rect = empty_surface.get_rect(center=panel_rect.center)
            self.screen.blit(empty_surface, empty_rect)
            return

        selected_index = max(0, min(selected_index, len(options) - 1))
        row_height = 52
        list_top = panel_rect.y + 46
        available_height = panel_rect.bottom - list_top - 12
        max_visible = max(1, available_height // row_height)
        max_scroll = max(0, len(options) - max_visible)
        scroll_offset = min(max(0, selected_index - max_visible + 1), max_scroll)

        for visible_index in range(max_visible):
            option_index = visible_index + scroll_offset
            if option_index >= len(options):
                break

            option = options[option_index]
            is_selected = option_index == selected_index
            is_available = bool(option.get("available", True))
            row_rect = pygame.Rect(
                panel_rect.x + 10,
                list_top + visible_index * row_height,
                panel_rect.width - 20,
                row_height - 7,
            )

            if is_selected:
                pygame.draw.rect(self.screen, (32, 41, 55), row_rect, border_radius=8)
                border_color = (255, 214, 95) if is_available else (255, 146, 92)
                pygame.draw.rect(self.screen, border_color, row_rect, 1, border_radius=8)
                accent_rect = pygame.Rect(row_rect.x, row_rect.y + 7, 4, row_rect.height - 14)
                pygame.draw.rect(self.screen, border_color, accent_rect, border_radius=2)

            label_color = (255, 224, 118) if is_selected else (190, 198, 210)
            if not is_available:
                label_color = (255, 168, 118) if is_selected else (158, 132, 128)

            label = self._truncate_text(
                str(option.get("label", "Option")), self.font_config_body, row_rect.width - 28
            )
            label_surface = self.font_config_body.render(label, True, label_color)
            self.screen.blit(label_surface, (row_rect.x + 14, row_rect.y + 7))

            status = str(option.get("status", ""))
            if status:
                status_color = (150, 160, 176) if is_available else (196, 130, 116)
                status_text = self._truncate_text(
                    status, self.font_config_micro, row_rect.width - 28
                )
                status_surface = self.font_config_micro.render(status_text, True, status_color)
                self.screen.blit(status_surface, (row_rect.x + 14, row_rect.y + 28))

        if scroll_offset > 0:
            indicator = self.font_config_micro.render("More above", True, (140, 150, 166))
            self.screen.blit(
                indicator, (panel_rect.right - indicator.get_width() - 18, list_top - 18)
            )

        if scroll_offset + max_visible < len(options):
            indicator = self.font_config_micro.render("More below", True, (140, 150, 166))
            self.screen.blit(
                indicator,
                (
                    panel_rect.right - indicator.get_width() - 18,
                    panel_rect.bottom - indicator.get_height() - 8,
                ),
            )

    def _draw_menu_detail(
        self, options: list[dict[str, Any]], selected_index: int, detail_rect: pygame.Rect
    ) -> None:
        """Draw detail information for the selected menu/action option."""
        if not options:
            return

        selected_index = max(0, min(selected_index, len(options) - 1))
        option = options[selected_index]
        is_available = bool(option.get("available", True))

        title = str(option.get("title") or option.get("label") or "Option")
        title = self._truncate_text(title, self.font_config_section, detail_rect.width - 48)
        title_surface = self.font_config_section.render(title, True, self.text_color)
        self.screen.blit(title_surface, (detail_rect.x + 24, detail_rect.y + 24))

        status = str(option.get("status", ""))
        if status:
            status_color = (92, 232, 146) if is_available else (255, 146, 92)
            status_width = min(
                max(92, self.font_config_micro.size(status)[0] + 28),
                detail_rect.width - 48,
            )
            self._draw_status_badge(
                status,
                status_color,
                pygame.Rect(detail_rect.x + 24, detail_rect.y + 64, status_width, 28),
            )

        description = str(option.get("description", ""))
        description_y = detail_rect.y + 108
        for line in self._wrap_text(description, self.font_config_hint, detail_rect.width - 48)[:4]:
            line_surface = self.font_config_hint.render(line, True, (190, 198, 210))
            self.screen.blit(line_surface, (detail_rect.x + 24, description_y))
            description_y += 25

        details = option.get("details", [])
        detail_rows_rect = pygame.Rect(
            detail_rect.x + 24,
            description_y + 12,
            detail_rect.width - 48,
            detail_rect.bottom - description_y - 32,
        )
        if isinstance(details, list):
            self._draw_detail_rows(details, detail_rows_rect)

    def _draw_detail_rows(self, details: list[dict[str, Any]], rows_rect: pygame.Rect) -> None:
        """Draw compact key/value detail rows."""
        row_height = 44
        max_visible = max(0, rows_rect.height // (row_height + 8))

        for index, detail in enumerate(details[:max_visible]):
            row_rect = pygame.Rect(
                rows_rect.x,
                rows_rect.y + index * (row_height + 8),
                rows_rect.width,
                row_height,
            )
            pygame.draw.rect(self.screen, (18, 24, 34), row_rect, border_radius=8)
            pygame.draw.rect(self.screen, (48, 58, 74), row_rect, 1, border_radius=8)

            label = str(detail.get("label", ""))
            value = str(detail.get("value", ""))
            label_width = min(130, max(82, row_rect.width // 3))
            value_width = max(1, row_rect.width - label_width - 34)

            label_text = self._truncate_text(label, self.font_config_micro, label_width)
            value_text = self._truncate_text(value, self.font_config_hint, value_width)

            label_surface = self.font_config_micro.render(label_text.upper(), True, (126, 138, 156))
            label_rect = label_surface.get_rect(midleft=(row_rect.x + 14, row_rect.centery))
            self.screen.blit(label_surface, label_rect)

            value_surface = self.font_config_hint.render(value_text, True, (220, 226, 236))
            value_rect = value_surface.get_rect(
                midleft=(row_rect.x + label_width + 20, row_rect.centery)
            )
            self.screen.blit(value_surface, value_rect)

    def draw_model_selection_menu(
        self, available_models: list[dict[str, Any]], selected_model: int = 0
    ) -> None:
        """Draw the trained model browser as a two-panel control surface."""
        self.screen.fill((7, 10, 15))

        model_options = [self._normalize_model_option(model) for model in available_models]

        margin = max(18, min(32, self.width // 28))
        gap = max(12, min(18, self.width // 48))
        footer_height = 56
        body_top = 88
        body_bottom = self.height - footer_height - 18
        body_height = max(260, body_bottom - body_top)
        sidebar_width = max(260, min(340, self.width // 3))

        sidebar_rect = pygame.Rect(margin, body_top, sidebar_width, body_height)
        detail_rect = pygame.Rect(
            sidebar_rect.right + gap,
            body_top,
            self.width - sidebar_rect.right - gap - margin,
            body_height,
        )
        footer_rect = pygame.Rect(
            margin,
            self.height - footer_height - 10,
            self.width - 2 * margin,
            footer_height,
        )

        title_surface = self.font_config_title.render("Model Browser", True, self.text_color)
        self.screen.blit(title_surface, (margin, 24))

        subtitle_surface = self.font_config_hint.render(
            "Choose a trained checkpoint for the DQN opponent", True, (154, 166, 184)
        )
        subtitle_rect = subtitle_surface.get_rect()
        subtitle_rect.midleft = (margin + 250, 47)
        self.screen.blit(subtitle_surface, subtitle_rect)

        self._draw_panel(sidebar_rect, (14, 18, 27), (52, 63, 80))
        self._draw_panel(detail_rect, (11, 15, 23), (48, 58, 74))

        if model_options:
            selected_model = max(0, min(selected_model, len(model_options) - 1))
            self._draw_option_rows(model_options, selected_model, sidebar_rect, "MODELS")
            self._draw_model_detail(model_options, selected_model, detail_rect)
            hints = [
                ("UP/DOWN", "Navigate"),
                ("ENTER/SPACE", "Load"),
                ("ESC", "Back"),
            ]
        else:
            self._draw_option_rows([], 0, sidebar_rect, "MODELS")
            self._draw_empty_model_detail(detail_rect)
            hints = [("ESC", "Main menu")]

        self._draw_hint_bar(footer_rect, hints)

    def _normalize_model_option(self, model: dict[str, Any]) -> dict[str, Any]:
        """Accept both raw discovery rows and renderer-ready model rows."""
        option = dict(model)
        label = str(option.get("label") or option.get("name") or option.get("file") or "Model")
        path = str(option.get("path") or "")
        file_name = str(option.get("file") or (path.rsplit("/", 1)[-1] if path else "Unknown"))

        option.setdefault("label", label)
        option.setdefault("title", label)
        option.setdefault("file", file_name)
        option.setdefault("description", "Saved model checkpoint discovered locally.")
        option.setdefault("status", "Discovered")
        option.setdefault("available", True)
        option.setdefault(
            "details",
            [
                {"label": "Display name", "value": label},
                {"label": "File", "value": file_name},
                {"label": "Path", "value": path or "Unknown"},
            ],
        )
        return option

    def _draw_model_detail(
        self, models: list[dict[str, Any]], selected_index: int, detail_rect: pygame.Rect
    ) -> None:
        """Draw the selected model metadata panel."""
        if not models:
            return

        selected_index = max(0, min(selected_index, len(models) - 1))
        model = models[selected_index]
        is_available = bool(model.get("available", True))
        status = str(model.get("status", ""))
        title = str(model.get("title") or model.get("label") or "Model")

        title_text = self._truncate_text(title, self.font_config_section, detail_rect.width - 48)
        title_surface = self.font_config_section.render(title_text, True, self.text_color)
        self.screen.blit(title_surface, (detail_rect.x + 24, detail_rect.y + 22))

        file_name = str(model.get("file", ""))
        if file_name:
            file_text = self._truncate_text(
                file_name, self.font_config_hint, detail_rect.width - 48
            )
            file_surface = self.font_config_hint.render(file_text, True, (154, 166, 184))
            self.screen.blit(file_surface, (detail_rect.x + 24, detail_rect.y + 54))

        if status:
            if not is_available:
                status_color = (255, 146, 92)
            elif status.lower() == "valid":
                status_color = (92, 232, 146)
            else:
                status_color = (100, 170, 255)
            status_width = min(
                max(92, self.font_config_micro.size(status)[0] + 28),
                detail_rect.width - 48,
            )
            self._draw_status_badge(
                status,
                status_color,
                pygame.Rect(detail_rect.x + 24, detail_rect.y + 84, status_width, 28),
            )

        description = str(model.get("description", ""))
        description_y = detail_rect.y + 126
        for line in self._wrap_text(description, self.font_config_hint, detail_rect.width - 48)[:3]:
            line_surface = self.font_config_hint.render(line, True, (190, 198, 210))
            self.screen.blit(line_surface, (detail_rect.x + 24, description_y))
            description_y += 24

        details = model.get("details", [])
        if isinstance(details, list):
            rows_rect = pygame.Rect(
                detail_rect.x + 24,
                description_y + 12,
                detail_rect.width - 48,
                detail_rect.bottom - description_y - 32,
            )
            self._draw_model_detail_rows(details, rows_rect)

    def _draw_empty_model_detail(self, detail_rect: pygame.Rect) -> None:
        """Draw the empty state for the model browser."""
        title_surface = self.font_config_section.render("No Trained Models", True, self.text_color)
        self.screen.blit(title_surface, (detail_rect.x + 24, detail_rect.y + 24))

        self._draw_status_badge(
            "No models",
            (255, 146, 92),
            pygame.Rect(detail_rect.x + 24, detail_rect.y + 64, 104, 28),
        )

        description = "No .pth checkpoints were discovered in the local models directory."
        description_y = detail_rect.y + 108
        for line in self._wrap_text(description, self.font_config_hint, detail_rect.width - 48)[:3]:
            line_surface = self.font_config_hint.render(line, True, (190, 198, 210))
            self.screen.blit(line_surface, (detail_rect.x + 24, description_y))
            description_y += 24

        rows_rect = pygame.Rect(
            detail_rect.x + 24,
            description_y + 12,
            detail_rect.width - 48,
            detail_rect.bottom - description_y - 32,
        )
        self._draw_model_detail_rows(
            [
                {"label": "Location", "value": "models/"},
                {"label": "Expected file", "value": "*.pth"},
                {"label": "Back", "value": "Press ESC to return to the main menu"},
            ],
            rows_rect,
        )

    def _draw_model_detail_rows(
        self, details: list[dict[str, Any]], rows_rect: pygame.Rect
    ) -> None:
        """Draw compact model metadata rows."""
        row_height = 34
        gap = 6
        max_visible = max(0, rows_rect.height // (row_height + gap))

        for index, detail in enumerate(details[:max_visible]):
            row_rect = pygame.Rect(
                rows_rect.x,
                rows_rect.y + index * (row_height + gap),
                rows_rect.width,
                row_height,
            )
            pygame.draw.rect(self.screen, (18, 24, 34), row_rect, border_radius=8)
            pygame.draw.rect(self.screen, (48, 58, 74), row_rect, 1, border_radius=8)

            label = str(detail.get("label", ""))
            value = str(detail.get("value", ""))
            tone = str(detail.get("tone", ""))
            label_width = min(148, max(92, row_rect.width // 3))
            value_width = max(1, row_rect.width - label_width - 34)

            label_text = self._truncate_text(label, self.font_config_micro, label_width)
            value_text = self._truncate_text(value, self.font_config_hint, value_width)

            value_color = (220, 226, 236)
            if tone == "error":
                value_color = (255, 168, 118)
            elif tone == "success":
                value_color = (92, 232, 146)

            label_surface = self.font_config_micro.render(label_text.upper(), True, (126, 138, 156))
            label_rect = label_surface.get_rect(midleft=(row_rect.x + 14, row_rect.centery))
            self.screen.blit(label_surface, label_rect)

            value_surface = self.font_config_hint.render(value_text, True, value_color)
            value_rect = value_surface.get_rect(
                midleft=(row_rect.x + label_width + 20, row_rect.centery)
            )
            self.screen.blit(value_surface, value_rect)

        if len(details) > max_visible and max_visible > 0:
            overflow = self.font_config_micro.render("More metadata hidden", True, (140, 150, 166))
            overflow_rect = overflow.get_rect(bottomright=(rows_rect.right, rows_rect.bottom))
            self.screen.blit(overflow, overflow_rect)

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
        """Draw the configuration category selection panel."""
        layout = self._draw_settings_shell(categories, selected_index, "Choose a settings group")
        content_rect = layout["content"]

        selected_name = categories[selected_index] if categories else "Settings"
        title_surface = self.font_config_section.render("Control Panel", True, self.text_color)
        self.screen.blit(title_surface, (content_rect.x + 24, content_rect.y + 24))

        prompt = "Select a category to edit its options."
        prompt_surface = self.font_config_body.render(prompt, True, (190, 198, 210))
        self.screen.blit(prompt_surface, (content_rect.x + 24, content_rect.y + 64))

        preview_rect = pygame.Rect(
            content_rect.x + 24,
            content_rect.y + 120,
            content_rect.width - 48,
            min(128, content_rect.height - 168),
        )
        pygame.draw.rect(self.screen, (22, 27, 36), preview_rect, border_radius=8)
        pygame.draw.rect(self.screen, (58, 70, 88), preview_rect, 1, border_radius=8)

        preview_title = self.font_config_body.render(selected_name, True, (255, 220, 95))
        self.screen.blit(preview_title, (preview_rect.x + 18, preview_rect.y + 18))

        preview_text = "Press ENTER or SPACE to open this group."
        preview_surface = self.font_config_hint.render(preview_text, True, (168, 176, 190))
        self.screen.blit(preview_surface, (preview_rect.x + 18, preview_rect.y + 55))

        self._draw_hint_bar(
            layout["footer"],
            [
                ("UP/DOWN", "Navigate"),
                ("ENTER", "Open"),
                ("S", "Save"),
                ("R", "Reset"),
                ("ESC", "Back"),
            ],
        )

    def draw_config_option_menu(
        self,
        category_name: str,
        options: list[dict],
        selected_index: int,
        is_editing: bool = False,
        categories: list | None = None,
        selected_category_index: int = 0,
    ) -> None:
        """Draw the configuration options control panel for a category."""
        sidebar_categories = categories or [category_name]
        layout = self._draw_settings_shell(
            sidebar_categories,
            selected_category_index,
            f"{category_name} settings",
        )
        content_rect = layout["content"]

        header_y = content_rect.y + 18
        title_surface = self.font_config_section.render(category_name, True, self.text_color)
        self.screen.blit(title_surface, (content_rect.x + 22, header_y))

        status_text = "EDITING" if is_editing else "READY"
        status_color = (70, 210, 130) if is_editing else (100, 170, 255)
        self._draw_status_badge(
            status_text,
            status_color,
            pygame.Rect(content_rect.right - 112, header_y + 2, 86, 28),
        )

        row_height = 68
        list_top = content_rect.y + 62
        available_height = content_rect.bottom - list_top - 14
        max_visible = max(1, available_height // row_height)

        max_scroll = max(0, len(options) - max_visible)
        scroll_offset = min(max(0, selected_index - max_visible + 1), max_scroll)

        for visible_index in range(max_visible):
            option_index = visible_index + scroll_offset
            if option_index >= len(options):
                break

            option = options[option_index]
            row_rect = pygame.Rect(
                content_rect.x + 16,
                list_top + visible_index * row_height,
                content_rect.width - 32,
                row_height - 8,
            )
            self._draw_config_option_row(
                option,
                row_rect,
                option_index == selected_index,
                is_editing and option_index == selected_index,
            )

        if scroll_offset > 0:
            indicator = self.font_config_micro.render("More above", True, (140, 150, 166))
            self.screen.blit(
                indicator, (content_rect.right - indicator.get_width() - 24, list_top - 18)
            )

        if scroll_offset + max_visible < len(options):
            indicator = self.font_config_micro.render("More below", True, (140, 150, 166))
            self.screen.blit(
                indicator,
                (
                    content_rect.right - indicator.get_width() - 24,
                    content_rect.bottom - indicator.get_height() - 8,
                ),
            )

        selected_field_type = ""
        if options and 0 <= selected_index < len(options):
            selected_field_type = options[selected_index]["field_type"]

        if is_editing and selected_field_type == "boolean":
            hints = [
                ("LEFT/RIGHT", "Change"),
                ("SPACE", "Toggle"),
                ("ENTER", "Confirm"),
                ("ESC", "Cancel"),
            ]
        elif is_editing:
            hints = [
                ("LEFT/RIGHT", "Change"),
                ("ENTER/SPACE", "Confirm"),
                ("ESC", "Cancel"),
            ]
        else:
            hints = [
                ("UP/DOWN", "Navigate"),
                ("ENTER", "Edit"),
                ("S", "Save"),
                ("R", "Reset"),
                ("ESC", "Back"),
            ]

        self._draw_hint_bar(layout["footer"], hints)

    def _format_config_value(self, value: Any, field_type: str) -> str:
        """Format a configuration value for display"""
        if field_type == "boolean":
            return "ON" if value else "OFF"
        elif field_type == "numeric":
            if isinstance(value, float) and value.is_integer():
                return str(int(value))
            if isinstance(value, float):
                return f"{value:.2f}".rstrip("0").rstrip(".")
            return str(value)
        elif field_type == "selection":
            return str(value).upper()
        else:
            return str(value)

    def _draw_settings_shell(
        self, categories: list, selected_index: int, subtitle: str
    ) -> dict[str, pygame.Rect]:
        """Draw the shared settings frame and category sidebar."""
        self.screen.fill((7, 10, 15))

        margin = max(18, min(30, self.width // 28))
        gap = max(12, min(18, self.width // 48))
        footer_height = 56
        body_top = 82
        body_bottom = self.height - footer_height - 18
        body_height = max(220, body_bottom - body_top)
        sidebar_width = max(150, min(220, self.width // 4))

        sidebar_rect = pygame.Rect(margin, body_top, sidebar_width, body_height)
        content_rect = pygame.Rect(
            sidebar_rect.right + gap,
            body_top,
            self.width - sidebar_rect.right - gap - margin,
            body_height,
        )
        footer_rect = pygame.Rect(
            margin,
            self.height - footer_height - 10,
            self.width - 2 * margin,
            footer_height,
        )

        title_surface = self.font_config_title.render("Settings", True, self.text_color)
        self.screen.blit(title_surface, (margin, 24))

        subtitle_text = self._truncate_text(
            subtitle, self.font_config_hint, max(120, self.width - 260)
        )
        subtitle_surface = self.font_config_hint.render(subtitle_text, True, (154, 166, 184))
        subtitle_rect = subtitle_surface.get_rect()
        subtitle_rect.midleft = (margin + 190, 47)
        self.screen.blit(subtitle_surface, subtitle_rect)

        self._draw_panel(sidebar_rect, (14, 18, 27), (52, 63, 80))
        self._draw_panel(content_rect, (11, 15, 23), (48, 58, 74))

        nav_label = self.font_config_micro.render("CATEGORIES", True, (126, 138, 156))
        self.screen.blit(nav_label, (sidebar_rect.x + 16, sidebar_rect.y + 16))

        row_y = sidebar_rect.y + 46
        row_height = 42
        for index, category in enumerate(categories):
            row_rect = pygame.Rect(
                sidebar_rect.x + 10,
                row_y + index * row_height,
                sidebar_rect.width - 20,
                row_height - 6,
            )
            is_selected = index == selected_index
            if is_selected:
                pygame.draw.rect(self.screen, (32, 41, 55), row_rect, border_radius=8)
                pygame.draw.rect(self.screen, (255, 214, 95), row_rect, 1, border_radius=8)
                accent_rect = pygame.Rect(row_rect.x, row_rect.y + 7, 4, row_rect.height - 14)
                pygame.draw.rect(self.screen, (255, 214, 95), accent_rect, border_radius=2)

            color = self.text_color if is_selected else (170, 180, 195)
            category_text = self._truncate_text(
                category, self.font_config_body, row_rect.width - 22
            )
            category_surface = self.font_config_body.render(category_text, True, color)
            category_rect = category_surface.get_rect()
            category_rect.midleft = (row_rect.x + 14, row_rect.centery)
            self.screen.blit(category_surface, category_rect)

        return {"sidebar": sidebar_rect, "content": content_rect, "footer": footer_rect}

    def _draw_panel(
        self,
        rect: pygame.Rect,
        fill_color: tuple[int, int, int],
        border_color: tuple[int, int, int],
    ) -> None:
        """Draw a bordered panel."""
        pygame.draw.rect(self.screen, fill_color, rect, border_radius=8)
        pygame.draw.rect(self.screen, border_color, rect, 1, border_radius=8)

    def _draw_status_badge(self, text: str, color: tuple[int, int, int], rect: pygame.Rect) -> None:
        """Draw the current settings panel state."""
        pygame.draw.rect(self.screen, (22, 28, 38), rect, border_radius=6)
        pygame.draw.rect(self.screen, color, rect, 1, border_radius=6)

        text_surface = self.font_config_micro.render(text, True, color)
        text_rect = text_surface.get_rect(center=rect.center)
        self.screen.blit(text_surface, text_rect)

    def _draw_config_option_row(
        self,
        option: dict,
        row_rect: pygame.Rect,
        is_selected: bool,
        is_editing: bool,
    ) -> None:
        """Draw one settings option row with an appropriate value control."""
        fill_color = (26, 32, 43) if is_selected else (14, 19, 29)
        border_color = (74, 212, 132) if is_editing else (255, 214, 95)
        muted_color = (150, 160, 176)

        pygame.draw.rect(self.screen, fill_color, row_rect, border_radius=8)
        if is_selected:
            pygame.draw.rect(self.screen, border_color, row_rect, 2, border_radius=8)

        control_width = min(240, max(176, int(row_rect.width * 0.44)))
        control_rect = pygame.Rect(
            row_rect.right - control_width - 14,
            row_rect.y + 10,
            control_width,
            row_rect.height - 20,
        )
        label_width = control_rect.x - row_rect.x - 28

        label_color = (255, 224, 118) if is_selected else self.text_color
        if is_editing:
            label_color = (92, 232, 146)

        label_text = self._truncate_text(option["label"], self.font_config_body, label_width)
        label_surface = self.font_config_body.render(label_text, True, label_color)
        self.screen.blit(label_surface, (row_rect.x + 14, row_rect.y + 10))

        description = option.get("description") or ""
        description_text = self._truncate_text(description, self.font_config_hint, label_width)
        description_surface = self.font_config_hint.render(description_text, True, muted_color)
        self.screen.blit(description_surface, (row_rect.x + 14, row_rect.y + 35))

        self._draw_config_value_control(option, control_rect, is_selected, is_editing)

    def _draw_config_value_control(
        self,
        option: dict,
        control_rect: pygame.Rect,
        is_selected: bool,
        is_editing: bool,
    ) -> None:
        """Draw a type-specific control for a setting value."""
        field_type = option["field_type"]
        if field_type == "boolean":
            self._draw_boolean_toggle(control_rect, bool(option["value"]), is_editing)
        elif field_type == "numeric":
            self._draw_numeric_slider(option, control_rect, is_selected, is_editing)
        elif field_type == "selection":
            self._draw_selection_segments(option, control_rect, is_editing)
        else:
            value_text = self._format_config_value(option["value"], field_type)
            color = (92, 232, 146) if is_editing else (210, 218, 228)
            value_text = self._truncate_text(value_text, self.font_config_body, control_rect.width)
            value_surface = self.font_config_body.render(value_text, True, color)
            value_rect = value_surface.get_rect(midright=(control_rect.right, control_rect.centery))
            self.screen.blit(value_surface, value_rect)

    def _draw_boolean_toggle(
        self, control_rect: pygame.Rect, value: bool, is_editing: bool
    ) -> None:
        """Draw a boolean as an ON/OFF toggle."""
        toggle_width = min(118, control_rect.width)
        toggle_height = 32
        toggle_rect = pygame.Rect(
            control_rect.right - toggle_width,
            control_rect.centery - toggle_height // 2,
            toggle_width,
            toggle_height,
        )
        fill_color = (50, 144, 95) if value else (62, 70, 82)
        border_color = (92, 232, 146) if is_editing else (112, 124, 142)
        pygame.draw.rect(self.screen, fill_color, toggle_rect, border_radius=toggle_height // 2)
        pygame.draw.rect(
            self.screen, border_color, toggle_rect, 2, border_radius=toggle_height // 2
        )

        knob_radius = 11
        knob_x = toggle_rect.right - 18 if value else toggle_rect.x + 18
        pygame.draw.circle(self.screen, (235, 240, 246), (knob_x, toggle_rect.centery), knob_radius)

        value_text = "ON" if value else "OFF"
        text_color = (236, 244, 240) if value else (218, 224, 232)
        text_surface = self.font_config_micro.render(value_text, True, text_color)
        text_x = toggle_rect.x + 18 if value else toggle_rect.right - text_surface.get_width() - 18
        text_rect = text_surface.get_rect(midleft=(text_x, toggle_rect.centery))
        self.screen.blit(text_surface, text_rect)

    def _draw_numeric_slider(
        self, option: dict, control_rect: pygame.Rect, is_selected: bool, is_editing: bool
    ) -> None:
        """Draw a numeric value with slider context."""
        value = option["value"]
        min_value = option.get("min_value")
        max_value = option.get("max_value")
        value_text = self._format_config_value(value, "numeric")
        value_color = (92, 232, 146) if is_editing else (220, 226, 236)

        value_surface = self.font_config_body.render(value_text, True, value_color)
        value_rect = value_surface.get_rect(topright=(control_rect.right, control_rect.y - 1))
        self.screen.blit(value_surface, value_rect)

        track_width = control_rect.width
        track_rect = pygame.Rect(control_rect.x, control_rect.y + 28, track_width, 6)
        pygame.draw.rect(self.screen, (50, 58, 70), track_rect, border_radius=3)

        if min_value is not None and max_value is not None and max_value != min_value:
            percent = (float(value) - float(min_value)) / (float(max_value) - float(min_value))
            percent = max(0.0, min(1.0, percent))
            fill_width = max(4, int(track_rect.width * percent))
            fill_rect = pygame.Rect(track_rect.x, track_rect.y, fill_width, track_rect.height)
            fill_color = (92, 232, 146) if is_editing else (100, 170, 255)
            pygame.draw.rect(self.screen, fill_color, fill_rect, border_radius=3)
            thumb_x = track_rect.x + fill_width
            thumb_color = (
                (92, 232, 146) if is_editing else (255, 214, 95) if is_selected else (190, 198, 210)
            )
            pygame.draw.circle(self.screen, thumb_color, (thumb_x, track_rect.centery), 6)

            min_text = self._format_config_value(min_value, "numeric")
            max_text = self._format_config_value(max_value, "numeric")
            min_surface = self.font_config_micro.render(min_text, True, (132, 142, 158))
            max_surface = self.font_config_micro.render(max_text, True, (132, 142, 158))
            self.screen.blit(min_surface, (track_rect.x, track_rect.bottom + 4))
            max_rect = max_surface.get_rect(topright=(track_rect.right, track_rect.bottom + 4))
            self.screen.blit(max_surface, max_rect)

    def _draw_selection_segments(
        self, option: dict, control_rect: pygame.Rect, is_editing: bool
    ) -> None:
        """Draw a selection value as compact segmented choices."""
        choices = option.get("choices") or []
        if not choices:
            value_text = self._format_config_value(option["value"], "selection")
            value_surface = self.font_config_body.render(value_text, True, (220, 226, 236))
            value_rect = value_surface.get_rect(midright=(control_rect.right, control_rect.centery))
            self.screen.blit(value_surface, value_rect)
            return

        segment_height = 30
        segment_rect = pygame.Rect(
            control_rect.x,
            control_rect.centery - segment_height // 2,
            control_rect.width,
            segment_height,
        )
        segment_width = max(1, segment_rect.width // len(choices))
        selected_value = option["value"]

        for index, choice in enumerate(choices):
            part_rect = pygame.Rect(
                segment_rect.x + index * segment_width,
                segment_rect.y,
                segment_width,
                segment_height,
            )
            if index == len(choices) - 1:
                part_rect.width = segment_rect.right - part_rect.x

            is_current = choice == selected_value
            fill_color = (50, 144, 95) if is_current else (22, 28, 38)
            border_color = (92, 232, 146) if is_editing and is_current else (74, 86, 104)
            pygame.draw.rect(self.screen, fill_color, part_rect)
            pygame.draw.rect(self.screen, border_color, part_rect, 1)

            label = self._truncate_text(
                str(choice).upper(), self.font_config_micro, part_rect.width - 8
            )
            text_color = (240, 246, 244) if is_current else (170, 180, 195)
            choice_surface = self.font_config_micro.render(label, True, text_color)
            choice_rect = choice_surface.get_rect(center=part_rect.center)
            self.screen.blit(choice_surface, choice_rect)

    def _draw_hint_bar(self, footer_rect: pygame.Rect, hints: list[tuple[str, str]]) -> None:
        """Draw compact keyboard hints in the settings footer."""
        self._draw_panel(footer_rect, (12, 16, 24), (44, 54, 70))

        x = footer_rect.x + 12
        y = footer_rect.y + 13
        chip_height = 30
        gap = 8

        for key, action in hints:
            label = f"{key} {action}"
            label_width = self.font_config_hint.size(label)[0]
            chip_width = label_width + 18

            if x + chip_width > footer_rect.right - 12:
                x = footer_rect.x + 12
                y += chip_height + 6
                if y + chip_height > footer_rect.bottom - 6:
                    break

            chip_rect = pygame.Rect(x, y, chip_width, chip_height)
            pygame.draw.rect(self.screen, (24, 30, 40), chip_rect, border_radius=6)
            pygame.draw.rect(self.screen, (58, 70, 88), chip_rect, 1, border_radius=6)

            label_surface = self.font_config_hint.render(label, True, (202, 210, 222))
            label_rect = label_surface.get_rect(center=chip_rect.center)
            self.screen.blit(label_surface, label_rect)

            x += chip_width + gap

    def _truncate_text(self, text: str, font: pygame.font.Font, max_width: int) -> str:
        """Trim text to fit a single line."""
        if font.size(text)[0] <= max_width:
            return text

        suffix = "..."
        suffix_width = font.size(suffix)[0]
        available_width = max_width - suffix_width
        if available_width <= 0:
            return suffix

        trimmed = text
        while trimmed and font.size(trimmed)[0] > available_width:
            trimmed = trimmed[:-1]

        return trimmed.rstrip() + suffix

    def _wrap_text(self, text: str, font: pygame.font.Font, max_width: int) -> list[str]:
        """Wrap text to fit within a maximum width."""
        if not text:
            return []

        lines: list[str] = []
        current_line: list[str] = []

        for word in text.split():
            if font.size(word)[0] > max_width:
                if current_line:
                    lines.append(" ".join(current_line))
                    current_line = []
                lines.append(self._truncate_text(word, font, max_width))
                continue

            test_line = " ".join(current_line + [word])
            if font.size(test_line)[0] <= max_width:
                current_line.append(word)
            else:
                if current_line:
                    lines.append(" ".join(current_line))
                current_line = [word]

        if current_line:
            lines.append(" ".join(current_line))

        return lines

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
