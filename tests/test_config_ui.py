"""Tests for the settings menu application flow."""

from types import SimpleNamespace
from typing import Any

import pygame

import magic_pong.gui.game_app as game_app_module
from magic_pong.gui.game_app import GameState
from magic_pong.gui.game_app import MagicPongApp
from magic_pong.utils.config_manager import CONFIG_CATEGORIES


class DummyRenderer:
    """Renderer stand-in for input-flow tests."""

    show_debug = False

    def cleanup(self) -> None:
        pass


def key_event(key: int) -> Any:
    """Create the minimal event shape consumed by app input handlers."""
    return SimpleNamespace(type=pygame.KEYDOWN, key=key)


def create_app(monkeypatch: Any) -> MagicPongApp:
    """Create the app without opening a pygame window."""
    monkeypatch.setattr(game_app_module, "PygameRenderer", DummyRenderer)
    return MagicPongApp()


def test_config_option_data_includes_control_panel_metadata(monkeypatch: Any) -> None:
    app = create_app(monkeypatch)

    options_data = app._build_config_options_data(CONFIG_CATEGORIES[0])

    assert options_data[0]["label"] == "Ball Speed"
    assert options_data[0]["field_type"] == "numeric"
    assert options_data[0]["step"] == 25.0
    assert options_data[0]["description"] == "Initial ball speed"
    assert "choices" in options_data[0]


def test_save_confirmation_from_options_returns_to_options(monkeypatch: Any) -> None:
    app = create_app(monkeypatch)

    app.handle_config_category_input(key_event(pygame.K_RETURN))
    assert app.state == GameState.CONFIG_OPTIONS
    assert app.config_current_category is CONFIG_CATEGORIES[0]

    app.handle_config_options_input(key_event(pygame.K_s))
    assert app.state == GameState.CONFIG_CONFIRM
    assert app.config_confirm_action == "save"
    assert app.config_confirm_return_state == GameState.CONFIG_OPTIONS

    app.handle_config_confirm_input(key_event(pygame.K_ESCAPE))
    assert app.state == GameState.CONFIG_OPTIONS
    assert app.config_confirm_action is None
    assert app.config_confirm_return_state == GameState.CONFIG_CATEGORY
    assert app.config_current_category is CONFIG_CATEGORIES[0]


def test_space_confirms_non_boolean_edit_without_toggling(monkeypatch: Any) -> None:
    app = create_app(monkeypatch)

    app.handle_config_category_input(key_event(pygame.K_RETURN))
    app.handle_config_options_input(key_event(pygame.K_RETURN))
    assert app.config_is_editing

    app.handle_config_options_input(key_event(pygame.K_SPACE))
    assert not app.config_is_editing
    assert app.config_edit_backup is None
