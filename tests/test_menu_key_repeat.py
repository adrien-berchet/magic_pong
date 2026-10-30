"""Tests for the repeat of arrow keys held down in the GUI menus."""

from collections.abc import Callable
from types import SimpleNamespace
from typing import Any

import pygame
import pytest

import magic_pong.gui.game_app as game_app_module
from magic_pong.gui.game_app import GameState
from magic_pong.gui.game_app import MagicPongApp
from magic_pong.gui.key_repeat import KeyRepeater
from magic_pong.utils.config import game_config
from magic_pong.utils.config_manager import CONFIG_CATEGORIES
from magic_pong.utils.config_manager import ConfigCategory
from magic_pong.utils.config_manager import ConfigFieldType
from magic_pong.utils.config_manager import get_config_value
from magic_pong.utils.config_manager import set_config_value

DELAY_MS = game_app_module.MENU_KEY_REPEAT_DELAY_MS
INTERVAL_MS = game_app_module.MENU_KEY_REPEAT_INTERVAL_MS


class DummyRenderer:
    """Renderer stand-in for input-flow tests."""

    show_debug = False

    def cleanup(self) -> None:
        pass


class MenuHarness:
    """An app driven by key events on a manual clock."""

    def __init__(self, monkeypatch: Any) -> None:
        monkeypatch.setattr(game_app_module, "PygameRenderer", DummyRenderer)
        monkeypatch.setattr(game_app_module.pygame.time, "get_ticks", lambda: self.now_ms)
        self.now_ms = 1000
        self.app = MagicPongApp()

    def press(self, key: int) -> None:
        self.app._handle_event(SimpleNamespace(type=pygame.KEYDOWN, key=key))

    def release(self, key: int) -> None:
        self.app._handle_event(SimpleNamespace(type=pygame.KEYUP, key=key))

    def advance(self, duration_ms: int, frame_ms: int = 10) -> None:
        """Run the per-frame repeat update for the given duration."""
        for _ in range(duration_ms // frame_ms):
            self.now_ms += frame_ms
            self.app._update_menu_key_repeat()


@pytest.fixture
def harness(monkeypatch: Any) -> MenuHarness:
    return MenuHarness(monkeypatch)


def _category_with(field_type: ConfigFieldType) -> tuple[ConfigCategory, int]:
    """Find a configuration option of the given type."""
    for category in CONFIG_CATEGORIES:
        for index, option in enumerate(category.options):
            if option.field_type == field_type:
                return category, index
    raise AssertionError(f"No option of type {field_type}")


def _start_editing(app: MagicPongApp, category: ConfigCategory, index: int) -> None:
    app.state = GameState.CONFIG_OPTIONS
    app.config_current_category = category
    app.config_option_selected = index
    app.config_is_editing = True


class TestKeyRepeater:
    def test_nothing_is_repeated_before_the_delay(self) -> None:
        repeater = KeyRepeater(delay_ms=300, interval_ms=100)
        repeater.press(1, now_ms=0)

        assert repeater.poll(299) is None

    def test_key_repeats_after_the_delay_then_at_each_interval(self) -> None:
        repeater = KeyRepeater(delay_ms=300, interval_ms=100)
        repeater.press(1, now_ms=0)

        assert repeater.poll(300) == 1
        assert repeater.poll(399) is None
        assert repeater.poll(400) == 1

    def test_no_key_held_returns_nothing(self) -> None:
        assert KeyRepeater(delay_ms=300, interval_ms=100).poll(10_000) is None

    def test_releasing_the_key_stops_the_repeat(self) -> None:
        repeater = KeyRepeater(delay_ms=300, interval_ms=100)
        repeater.press(1, now_ms=0)
        repeater.release(1)

        assert repeater.poll(1000) is None

    def test_releasing_another_key_keeps_the_repeat(self) -> None:
        repeater = KeyRepeater(delay_ms=300, interval_ms=100)
        repeater.press(1, now_ms=0)
        repeater.release(2)

        assert repeater.poll(300) == 1

    def test_last_pressed_key_takes_over_and_restarts_the_delay(self) -> None:
        repeater = KeyRepeater(delay_ms=300, interval_ms=100)
        repeater.press(1, now_ms=0)
        repeater.press(2, now_ms=200)

        assert repeater.poll(499) is None
        assert repeater.poll(500) == 2

    def test_cancel_stops_the_repeat(self) -> None:
        repeater = KeyRepeater(delay_ms=300, interval_ms=100)
        repeater.press(1, now_ms=0)
        repeater.cancel()

        assert repeater.poll(1000) is None

    def test_a_stalled_frame_gives_one_repeat_not_a_burst(self) -> None:
        repeater = KeyRepeater(delay_ms=300, interval_ms=100)
        repeater.press(1, now_ms=0)

        assert repeater.poll(5000) == 1
        assert repeater.poll(5001) is None
        assert repeater.poll(5100) == 1


def _main_menu(app: MagicPongApp) -> Callable[[], int]:
    return lambda: app.menu_selected


def _human_vs_ai(app: MagicPongApp) -> Callable[[], int]:
    app.state = GameState.HUMAN_VS_AI_SELECT
    return lambda: app._human_vs_ai_idx


def _watch_p1(app: MagicPongApp) -> Callable[[], int]:
    app.state = GameState.WATCH_P1_SELECT
    return lambda: app._watch_p1_idx


def _watch_p2(app: MagicPongApp) -> Callable[[], int]:
    app.state = GameState.WATCH_P2_SELECT
    return lambda: app._watch_p2_idx


def _model_selection(app: MagicPongApp) -> Callable[[], int]:
    app.state = GameState.MODEL_SELECTION
    app.available_models = [{"name": str(index), "path": str(index)} for index in range(30)]
    return lambda: app.model_selected


def _config_category(app: MagicPongApp) -> Callable[[], int]:
    app.state = GameState.CONFIG_CATEGORY
    return lambda: app.config_category_selected


def _config_options(app: MagicPongApp) -> Callable[[], int]:
    app.state = GameState.CONFIG_OPTIONS
    app.config_current_category = max(CONFIG_CATEGORIES, key=lambda c: len(c.options))
    return lambda: app.config_option_selected


def _pause(app: MagicPongApp) -> Callable[[], int]:
    app.state = GameState.PAUSED
    return lambda: app.pause_selected


def _game_over(app: MagicPongApp) -> Callable[[], int]:
    app.state = GameState.GAME_OVER
    return lambda: app.game_over_selected


@pytest.mark.parametrize(
    "enter_menu",
    [
        _main_menu,
        _human_vs_ai,
        _watch_p1,
        _watch_p2,
        _model_selection,
        _config_category,
        _config_options,
        _pause,
        _game_over,
    ],
)
def test_holding_an_arrow_keeps_navigating_every_menu(
    harness: MenuHarness, enter_menu: Callable[[MagicPongApp], Callable[[], int]]
) -> None:
    selection = enter_menu(harness.app)
    harness.press(pygame.K_DOWN)
    assert selection() == 1

    harness.advance(DELAY_MS - 10)
    assert selection() == 1

    harness.advance(10)
    assert selection() == 2

    # The third position may wrap around in the shortest menus
    harness.advance(INTERVAL_MS)
    assert selection() != 2


def test_holding_up_navigates_backwards(harness: MenuHarness) -> None:
    harness.app.state = GameState.MODEL_SELECTION
    harness.app.available_models = [{"name": str(i), "path": str(i)} for i in range(30)]
    harness.app.model_selected = 10

    harness.press(pygame.K_UP)
    harness.advance(DELAY_MS)

    assert harness.app.model_selected == 8


def test_a_short_tap_moves_only_once(harness: MenuHarness) -> None:
    harness.press(pygame.K_DOWN)
    harness.advance(DELAY_MS // 2)
    harness.release(pygame.K_DOWN)
    harness.advance(DELAY_MS * 4)

    assert harness.app.menu_selected == 1


def test_releasing_the_arrow_stops_the_navigation(harness: MenuHarness) -> None:
    harness.app.state = GameState.MODEL_SELECTION
    harness.app.available_models = [{"name": str(i), "path": str(i)} for i in range(30)]
    harness.press(pygame.K_DOWN)
    harness.advance(DELAY_MS)
    selected = harness.app.model_selected
    harness.release(pygame.K_DOWN)
    harness.advance(DELAY_MS * 4)

    assert selected > 1
    assert harness.app.model_selected == selected


@pytest.mark.parametrize("key", [pygame.K_RETURN, pygame.K_SPACE, pygame.K_ESCAPE, pygame.K_s])
def test_other_keys_do_not_repeat(harness: MenuHarness, key: int) -> None:
    harness.app.state = GameState.CONFIG_CATEGORY
    harness.press(key)
    state = harness.app.state

    harness.advance(DELAY_MS * 4)

    # A repeat would start editing an option (ENTER, SPACE) or quit the app (ESCAPE)
    assert harness.app.state == state
    assert not harness.app.config_is_editing
    assert harness.app.running


def test_opening_a_submenu_stops_the_arrow_repeat(harness: MenuHarness) -> None:
    harness.app.state = GameState.CONFIG_CATEGORY
    harness.press(pygame.K_DOWN)
    harness.press(pygame.K_RETURN)
    assert harness.app.state == GameState.CONFIG_OPTIONS

    harness.advance(DELAY_MS * 4)

    assert harness.app.config_option_selected == 0


def test_arrow_held_during_gameplay_does_not_navigate_the_game_over_menu(
    harness: MenuHarness,
) -> None:
    harness.app.state = GameState.PLAYING
    harness.press(pygame.K_DOWN)
    harness.app.state = GameState.GAME_OVER

    harness.advance(DELAY_MS * 4)

    assert harness.app.game_over_selected == 0


def test_arrow_held_when_pausing_does_not_navigate_the_pause_menu(harness: MenuHarness) -> None:
    harness.app.state = GameState.PLAYING
    harness.press(pygame.K_DOWN)
    harness.app.state = GameState.PAUSED

    harness.advance(DELAY_MS * 4)

    assert harness.app.pause_selected == 0


def test_arrow_held_when_resuming_the_game_is_not_repeated(harness: MenuHarness) -> None:
    harness.app.state = GameState.PAUSED
    harness.press(pygame.K_DOWN)
    harness.press(pygame.K_p)
    assert harness.app.state == GameState.PLAYING
    selected = harness.app.pause_selected

    harness.advance(DELAY_MS * 4)

    assert harness.app.pause_selected == selected


def test_help_overlay_stops_the_arrow_repeat(harness: MenuHarness) -> None:
    harness.app.state = GameState.PAUSED
    harness.press(pygame.K_DOWN)
    harness.press(pygame.K_F1)
    assert harness.app.show_help

    harness.advance(DELAY_MS * 4)

    assert harness.app.pause_selected == 1


def test_arrow_pressed_under_the_help_overlay_is_not_repeated(harness: MenuHarness) -> None:
    harness.app.state = GameState.PAUSED
    harness.app.show_help = True
    harness.press(pygame.K_DOWN)
    harness.app.show_help = False

    harness.advance(DELAY_MS * 4)

    assert harness.app.pause_selected == 0


def test_holding_an_arrow_adjusts_a_numeric_option_repeatedly(harness: MenuHarness) -> None:
    category, index = _category_with(ConfigFieldType.NUMERIC)
    option = category.options[index]
    initial = get_config_value(game_config, option.key)
    set_config_value(game_config, option.key, option.min_value)
    try:
        _start_editing(harness.app, category, index)

        harness.press(pygame.K_RIGHT)
        harness.advance(DELAY_MS)

        value = get_config_value(game_config, option.key)
        assert value == min(option.max_value, option.min_value + 2 * option.step)
    finally:
        set_config_value(game_config, option.key, initial)


def test_holding_an_arrow_does_not_flip_a_boolean_option_repeatedly(
    harness: MenuHarness,
) -> None:
    category, index = _category_with(ConfigFieldType.BOOLEAN)
    option = category.options[index]
    initial = get_config_value(game_config, option.key)
    try:
        _start_editing(harness.app, category, index)

        harness.press(pygame.K_RIGHT)
        harness.advance(DELAY_MS * 4)

        assert get_config_value(game_config, option.key) is (not initial)
    finally:
        set_config_value(game_config, option.key, initial)


def test_vertical_arrow_held_when_editing_starts_does_not_move_the_selection(
    harness: MenuHarness,
) -> None:
    harness.app.state = GameState.CONFIG_OPTIONS
    harness.app.config_current_category = max(CONFIG_CATEGORIES, key=lambda c: len(c.options))
    harness.press(pygame.K_DOWN)
    harness.press(pygame.K_RETURN)
    assert harness.app.config_is_editing
    harness.press(pygame.K_ESCAPE)

    harness.advance(DELAY_MS * 4)

    assert harness.app.config_option_selected == 1


def test_horizontal_arrow_outside_of_editing_keeps_the_vertical_repeat(
    harness: MenuHarness,
) -> None:
    harness.app.state = GameState.MODEL_SELECTION
    harness.app.available_models = [{"name": str(i), "path": str(i)} for i in range(30)]
    harness.press(pygame.K_DOWN)
    harness.press(pygame.K_LEFT)

    harness.advance(DELAY_MS)

    assert harness.app.model_selected == 2


def test_run_loop_repeats_a_held_arrow(harness: MenuHarness, monkeypatch: Any) -> None:
    """The main loop feeds the repeat from one frame to the next."""
    app = harness.app
    frames = [
        [SimpleNamespace(type=pygame.KEYDOWN, key=pygame.K_DOWN)],
        *([[]] * ((DELAY_MS // 10) + 1)),
    ]
    remaining = iter(frames)

    def next_events() -> list[Any]:
        harness.now_ms += 10
        try:
            return next(remaining)
        except StopIteration:
            app.running = False
            return []

    monkeypatch.setattr(game_app_module.pygame.event, "get", next_events)
    monkeypatch.setattr(app, "update", lambda: None)
    monkeypatch.setattr(app, "render", lambda: None)
    monkeypatch.setattr(app, "cleanup", lambda: None)
    monkeypatch.setattr(app.renderer, "update", lambda: None, raising=False)

    app.run()

    assert app.menu_selected == 2
