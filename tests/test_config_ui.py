"""Tests for the settings menu application flow."""

from pathlib import Path
from types import ModuleType
from types import SimpleNamespace
from typing import Any

import pygame

import magic_pong.gui.game_app as game_app_module
from magic_pong.gui.game_app import GameMode
from magic_pong.gui.game_app import GameState
from magic_pong.gui.game_app import MagicPongApp
from magic_pong.gui.game_app import PauseAction
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


def pause_action_index(app: MagicPongApp, action: PauseAction) -> int:
    """Return the configured index for a pause action."""
    return next(index for index, item in enumerate(app.pause_actions) if item.action == action)


def menu_mode_index(app: MagicPongApp, mode: GameMode) -> int:
    """Return the configured index for a main menu mode."""
    return next(index for index, item in enumerate(app.menu_options) if item.mode == mode)


def test_main_menu_data_flags_trained_model_without_models(monkeypatch: Any) -> None:
    app = create_app(monkeypatch)
    app.available_models = []

    menu_data = app._build_main_menu_data()
    trained_model = next(item for item in menu_data if item["mode"] == GameMode.LOAD_MODEL.value)

    assert trained_model["label"] == "Trained Model"
    assert trained_model["available"] is False
    assert trained_model["status"] == "No models available"
    assert {"label": "Availability", "value": "No models available"} in trained_model["details"]


def test_main_menu_trained_model_opens_empty_model_browser(monkeypatch: Any) -> None:
    app = create_app(monkeypatch)
    app.available_models = []
    app.menu_selected = menu_mode_index(app, GameMode.LOAD_MODEL)

    app.handle_menu_input(key_event(pygame.K_RETURN))

    assert app.state == GameState.MODEL_SELECTION
    assert app.model_selected == 0


def test_model_selection_data_includes_browser_metadata(monkeypatch: Any, tmp_path: Path) -> None:
    app = create_app(monkeypatch)
    model_path = tmp_path / "trained_agent.pth"
    model_path.write_bytes(b"0" * 2048)
    app.available_models = [
        {"name": "Trained Agent", "path": str(model_path), "file": model_path.name}
    ]

    model_data = app._build_model_selection_data()
    details = {item["label"]: item["value"] for item in model_data[0]["details"]}

    assert model_data[0]["label"] == "Trained Agent"
    assert model_data[0]["status"] == "File found"
    assert model_data[0]["available"] is True
    assert details["File"] == "trained_agent.pth"
    assert details["Location"] == str(tmp_path)
    assert details["Availability"] == "File found"
    assert details["File size"] == "2.0 KB"


def test_model_info_rejects_checkpoint_missing_loader_keys(
    monkeypatch: Any, tmp_path: Path
) -> None:
    app = create_app(monkeypatch)
    model_path = tmp_path / "incomplete_agent.pth"
    model_path.write_bytes(b"0" * 2048)

    fake_torch = ModuleType("torch")
    fake_torch.load = lambda *_args, **_kwargs: {
        "q_network_state_dict": {},
        "target_network_state_dict": {},
    }
    monkeypatch.setitem(__import__("sys").modules, "torch", fake_torch)

    model_info = app._load_model_info(str(model_path))

    assert model_info["valid"] is False
    assert "optimizer_state_dict" in model_info["error"]


def test_model_info_rejects_checkpoint_missing_loader_hyperparameters(
    monkeypatch: Any, tmp_path: Path
) -> None:
    app = create_app(monkeypatch)
    model_path = tmp_path / "missing_hyperparams_agent.pth"
    model_path.write_bytes(b"0" * 2048)

    fake_torch = ModuleType("torch")
    fake_torch.load = lambda *_args, **_kwargs: {
        "q_network_state_dict": {},
        "target_network_state_dict": {},
        "optimizer_state_dict": {},
        "epsilon": 0.0,
        "training_step": 1,
        "loss_history": [],
        "reward_history": [],
        "hyperparameters": {"state_size": 32, "action_size": 9},
    }
    monkeypatch.setitem(__import__("sys").modules, "torch", fake_torch)

    model_info = app._load_model_info(str(model_path))

    assert model_info["valid"] is False
    assert "lr" in model_info["error"]


def test_model_selection_data_marks_pth_directory_unavailable(
    monkeypatch: Any, tmp_path: Path
) -> None:
    app = create_app(monkeypatch)
    model_path = tmp_path / "directory_model.pth"
    model_path.mkdir()
    app.available_models = [
        {"name": "Directory Model", "path": str(model_path), "file": model_path.name}
    ]

    model_data = app._build_model_selection_data()
    details = {item["label"]: item["value"] for item in model_data[0]["details"]}

    assert model_data[0]["status"] == "Missing file"
    assert model_data[0]["available"] is False
    assert details["Availability"] == "Missing file"
    assert "File size" not in details


def test_model_selection_data_marks_cached_invalid_model(monkeypatch: Any, tmp_path: Path) -> None:
    app = create_app(monkeypatch)
    model_path = tmp_path / "invalid_agent.pth"
    model_path.write_bytes(b"0" * 2048)
    app.available_models = [
        {"name": "Invalid Agent", "path": str(model_path), "file": model_path.name}
    ]
    app.model_info_by_path[str(model_path)] = {
        "path": str(model_path),
        "valid": False,
        "error": "Missing required data: q_network_state_dict",
    }

    model_data = app._build_model_selection_data()
    details = {item["label"]: item["value"] for item in model_data[0]["details"]}

    assert model_data[0]["status"] == "Invalid"
    assert model_data[0]["available"] is False
    assert details["Validity"] == "Failed validation"
    assert details["Error"] == "Missing required data: q_network_state_dict"


def test_model_selection_data_includes_cached_training_metadata(
    monkeypatch: Any, tmp_path: Path
) -> None:
    app = create_app(monkeypatch)
    model_path = tmp_path / "valid_agent.pth"
    model_path.write_bytes(b"0" * 2048)
    app.available_models = [
        {"name": "Valid Agent", "path": str(model_path), "file": model_path.name}
    ]
    app.model_info_by_path[str(model_path)] = {
        "path": str(model_path),
        "valid": True,
        "training_step": 12000,
        "epsilon": 0.012345,
        "hyperparameters": {"state_size": 32, "action_size": 9, "lr": 0.001},
    }

    model_data = app._build_model_selection_data()
    details = {item["label"]: item["value"] for item in model_data[0]["details"]}

    assert model_data[0]["status"] == "Valid"
    assert model_data[0]["available"] is True
    assert details["Validity"] == "Valid checkpoint"
    assert details["Training step"] == "12000"
    assert details["Epsilon"] == "0.01235"
    assert "state=32" in details["Hyperparameters"]
    assert "actions=9" in details["Hyperparameters"]
    assert "lr=0.001" in details["Hyperparameters"]


def test_model_selection_no_models_input_is_safe_and_esc_returns(
    monkeypatch: Any,
) -> None:
    app = create_app(monkeypatch)
    app.state = GameState.MODEL_SELECTION
    app.available_models = []

    app.handle_model_selection_input(key_event(pygame.K_UP))
    app.handle_model_selection_input(key_event(pygame.K_DOWN))

    assert app.model_selected == 0

    app.handle_model_selection_input(key_event(pygame.K_RETURN))

    assert app.error_message == "No trained models found in models/"

    app.handle_model_selection_input(key_event(pygame.K_ESCAPE))

    assert app.state == GameState.MENU


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


def test_pause_space_resumes_even_when_action_selection_moved(monkeypatch: Any) -> None:
    app = create_app(monkeypatch)
    app.state = GameState.PAUSED
    app.game_engine.paused = True
    app.pause_selected = pause_action_index(app, PauseAction.QUIT)

    app.handle_pause_input(key_event(pygame.K_SPACE))

    assert app.state == GameState.PLAYING
    assert app.game_engine.paused is False


def test_pause_navigation_enter_selects_restart(monkeypatch: Any) -> None:
    app = create_app(monkeypatch)
    app.state = GameState.PAUSED
    app.current_mode = GameMode.ONE_VS_AI
    restarted = False

    def fake_restart_game() -> None:
        nonlocal restarted
        restarted = True

    monkeypatch.setattr(app, "restart_game", fake_restart_game)

    app.handle_pause_input(key_event(pygame.K_DOWN))
    assert app.pause_selected == pause_action_index(app, PauseAction.RESTART)

    app.handle_pause_input(key_event(pygame.K_RETURN))

    assert restarted is True


def test_pause_settings_returns_to_pause_from_category_screen(monkeypatch: Any) -> None:
    app = create_app(monkeypatch)
    app.state = GameState.PAUSED
    app.game_engine.paused = True
    app.pause_selected = pause_action_index(app, PauseAction.SETTINGS)

    app.handle_pause_input(key_event(pygame.K_RETURN))

    assert app.state == GameState.CONFIG_CATEGORY
    assert app.config_exit_state == GameState.PAUSED

    app.handle_config_category_input(key_event(pygame.K_ESCAPE))

    assert app.state == GameState.PAUSED
    assert app.game_engine.paused is True
