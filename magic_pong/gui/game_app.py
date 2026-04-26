"""
Main game application with PyGame GUI
"""

import os
import sys
import traceback
from dataclasses import dataclass
from enum import Enum
from typing import Any
from typing import Literal

import pygame

from magic_pong.ai.models.simple_ai import DefensiveAI
from magic_pong.ai.models.simple_ai import FollowBallAI as SimpleAI
from magic_pong.core.entities import Player
from magic_pong.core.game_engine import GameEngine
from magic_pong.gui.human_player import HumanPlayer
from magic_pong.gui.human_player import InputManager
from magic_pong.gui.human_player import create_human_players
from magic_pong.gui.pygame_renderer import PygameRenderer
from magic_pong.utils.config import game_config
from magic_pong.utils.config import load_config_from_file
from magic_pong.utils.config_manager import CONFIG_CATEGORIES
from magic_pong.utils.config_manager import ConfigCategory
from magic_pong.utils.config_manager import ConfigFieldType
from magic_pong.utils.config_manager import get_config_value
from magic_pong.utils.config_manager import set_config_value


class GameMode(Enum):
    """Available game modes"""

    MENU = "menu"
    ONE_VS_ONE = "1v1"
    ONE_VS_AI = "1vAI"
    LOAD_MODEL = "load_model"
    AI_DEMO = "demo"
    CONFIG = "config"


@dataclass(frozen=True)
class MainMenuOption:
    """Main menu option plus detail-panel copy."""

    label: str
    mode: GameMode | None
    title: str
    description: str
    details: tuple[tuple[str, str], ...]


class GameState(Enum):
    """Current application state"""

    MENU = "menu"
    MODEL_SELECTION = "model_selection"
    CONFIG_CATEGORY = "config_category"
    CONFIG_OPTIONS = "config_options"
    CONFIG_CONFIRM = "config_confirm"
    PLAYING = "playing"
    PAUSED = "paused"
    GAME_OVER = "game_over"
    HELP = "help"


class PauseAction(Enum):
    """Actions available from the pause menu."""

    RESUME = "resume"
    RESTART = "restart"
    SETTINGS = "settings"
    MAIN_MENU = "main_menu"
    QUIT = "quit"


@dataclass(frozen=True)
class PauseMenuAction:
    """Pause menu action plus detail-panel copy."""

    label: str
    action: PauseAction
    title: str
    description: str
    details: tuple[tuple[str, str], ...]


class MagicPongApp:
    """Main application class for Magic Pong with PyGame GUI"""

    def __init__(self) -> None:
        """Initialize the application"""
        # Initialize game components
        self.game_engine = GameEngine(headless=False)
        self.renderer = PygameRenderer()
        self.input_manager = InputManager()

        # Application state
        self.state = GameState.MENU
        self.current_mode = GameMode.MENU
        self.running = True

        # Menu state
        self.menu_selected = 0
        self.menu_options = [
            MainMenuOption(
                "1 vs 1",
                GameMode.ONE_VS_ONE,
                "Local Duel",
                "Two players share the keyboard for a direct Magic Pong match.",
                (
                    ("Players", "Player 1 vs Player 2"),
                    ("Controls", "WASD and arrow keys"),
                    ("Status", "Ready"),
                ),
            ),
            MainMenuOption(
                "1 vs AI",
                GameMode.ONE_VS_AI,
                "Player vs Simple AI",
                "Practice against the built-in defensive computer opponent.",
                (
                    ("Players", "Human vs Simple AI"),
                    ("Controls", "Player uses WASD"),
                    ("Status", "Ready"),
                ),
            ),
            MainMenuOption(
                "Trained Model",
                GameMode.LOAD_MODEL,
                "Play vs Trained Model",
                "Challenge a saved DQN model discovered from the local models directory.",
                (
                    ("Players", "Human vs trained DQN"),
                    ("Controls", "Player uses WASD"),
                ),
            ),
            MainMenuOption(
                "AI Demo",
                GameMode.AI_DEMO,
                "AI vs AI Demonstration",
                "Watch the simple and defensive AI players compete without human input.",
                (
                    ("Players", "Simple AI vs Defensive AI"),
                    ("Controls", "Watch mode"),
                    ("Status", "Ready"),
                ),
            ),
            MainMenuOption(
                "Settings",
                GameMode.CONFIG,
                "Settings",
                "Adjust gameplay, controls, display, and advanced tuning values.",
                (
                    ("Categories", "Gameplay, Controls, Display, Advanced"),
                    ("Status", "Available"),
                ),
            ),
            MainMenuOption(
                "Quit",
                None,
                "Quit Magic Pong",
                "Close the application.",
                (("Status", "Exit application"),),
            ),
        ]

        # Pause menu state
        self.pause_selected = 0
        self.pause_actions = [
            PauseMenuAction(
                "Resume",
                PauseAction.RESUME,
                "Resume Game",
                "Return to the current rally without resetting the match.",
                (("Fast keys", "P or SPACE"),),
            ),
            PauseMenuAction(
                "Restart",
                PauseAction.RESTART,
                "Restart Match",
                "Reset the current mode and start a fresh match immediately.",
                (("Effect", "Current score and rally reset"),),
            ),
            PauseMenuAction(
                "Settings",
                PauseAction.SETTINGS,
                "Settings",
                "Open the control panel while keeping the current game paused.",
                (("Return", "ESC from settings returns here"),),
            ),
            PauseMenuAction(
                "Main Menu",
                PauseAction.MAIN_MENU,
                "Main Menu",
                "Stop the current match and return to mode selection.",
                (("Effect", "Current game ends"),),
            ),
            PauseMenuAction(
                "Quit",
                PauseAction.QUIT,
                "Quit Magic Pong",
                "Close the application from the pause menu.",
                (("Effect", "Application exits"),),
            ),
        ]

        # Game state
        self.game_over_timer = 0.0
        self.show_help = False

        # Model selection state
        self.model_selected = 0
        self.available_models: list[dict[str, Any]] = []
        self.selected_model_path = None
        self.model_info: dict[str, Any] | None = None
        self.model_info_by_path: dict[str, dict[str, Any]] = {}
        self.error_message: str | None = None
        self.error_timer: float = 0.0

        # Players
        self.human_players: dict[int, HumanPlayer | None] = {1: None, 2: None}

        # Configuration state
        self.config_category_selected = 0
        self.config_option_selected = 0
        self.config_is_editing = False
        self.config_current_category: ConfigCategory | None = None
        self.config_edit_backup: Any | None = None
        self.config_confirm_action: Literal["save", "reset"] | None = None
        self.config_confirm_return_state = GameState.CONFIG_CATEGORY
        self.config_exit_state = GameState.MENU

        print("Magic Pong initialized successfully!")
        print("Use arrows to navigate and ENTER to select")

        # Initialize available models
        self._discover_models()

        # Try to load saved configuration
        if load_config_from_file():
            print("Loaded saved configuration")

    def start_game_mode(self, mode: GameMode) -> None:
        """Start a specific game mode"""
        self.current_mode = mode

        # Create human players based on mode
        self.human_players = create_human_players(mode.value)

        # Clear input manager and add human players
        self.input_manager = InputManager()
        for player_id, player in self.human_players.items():
            if player is not None:
                self.input_manager.add_player(player, player_id)

        # Create AI players as needed

        ai_players: dict[int, Player | None] = {1: None, 2: None}

        if mode == GameMode.ONE_VS_AI:
            # Human vs AI
            ai_players[2] = SimpleAI("Simple AI")

        elif mode == GameMode.LOAD_MODEL:
            # Human vs Trained AI
            if not self.selected_model_path:
                print("No model selected, cannot start game")
                self.state = GameState.MENU
                self.current_mode = GameMode.MENU
                return
            ai_players[2] = self._create_trained_ai(  # type: ignore[unreachable]
                self.selected_model_path
            )
            if ai_players[2] is None:
                print("Failed to load trained model, falling back to Simple AI")
                ai_players[2] = SimpleAI("Simple AI (fallback)")

        elif mode == GameMode.AI_DEMO:
            # AI vs AI
            ai_players[1] = SimpleAI("AI 1")
            ai_players[2] = DefensiveAI("AI 2")

        # Set players in game engine
        player1 = self.human_players[1] or ai_players[1]
        player2 = self.human_players[2] or ai_players[2]

        self.game_engine.set_players(player1, player2)
        self.game_engine.start_game()

        # Change state to playing
        self.state = GameState.PLAYING

        print(f"Game mode {mode.value} started!")

    def _discover_models(self) -> None:
        """Discover available trained models"""
        models_dir = "models"
        if not os.path.exists(models_dir):
            print("Models directory not found")
            return

        self.available_models = []

        # Look for .pth files in models directory
        for file in os.listdir(models_dir):
            if file.endswith(".pth") and not file.endswith("training.png"):
                model_path = os.path.join(models_dir, file)
                if not os.path.isfile(model_path):
                    continue
                # Extract model name without extension for display
                model_name = file.replace(".pth", "").replace("_", " ").title()
                self.available_models.append({"name": model_name, "path": model_path, "file": file})

        # Sort models by name for better UX
        self.available_models.sort(key=lambda x: x["name"])

        print(f"Discovered {len(self.available_models)} trained models")

    def _load_model_info(self, model_path: str | None) -> dict:
        """Load basic information about a model"""
        try:
            import torch

            # Check if file exists
            if model_path is None or not os.path.isfile(model_path):
                return {"path": model_path, "valid": False, "error": "File not found"}

            # Check file size
            file_size = os.path.getsize(model_path)
            if file_size < 1024:  # Less than 1KB probably corrupted
                return {
                    "path": model_path,
                    "valid": False,
                    "error": "File too small, possibly corrupted",
                }

            # Try to load the checkpoint
            checkpoint = torch.load(model_path, map_location="cpu")

            # Validate checkpoint structure
            required_keys = [
                "q_network_state_dict",
                "target_network_state_dict",
                "optimizer_state_dict",
                "epsilon",
                "training_step",
                "loss_history",
                "reward_history",
            ]
            missing_keys = [key for key in required_keys if key not in checkpoint]

            if missing_keys:
                return {
                    "path": model_path,
                    "valid": False,
                    "error": f"Missing required data: {', '.join(missing_keys)}",
                }

            # Get hyperparameters for compatibility check
            hyperparams = checkpoint.get("hyperparameters", {})
            required_hyperparams = [
                "state_size",
                "action_size",
                "lr",
                "gamma",
                "epsilon_min",
                "epsilon_decay",
                "batch_size",
                "tau",
                "use_prioritized_replay",
            ]
            missing_hyperparams = [key for key in required_hyperparams if key not in hyperparams]

            if missing_hyperparams:
                return {
                    "path": model_path,
                    "valid": False,
                    "error": f"Missing hyperparameters: {', '.join(missing_hyperparams)}",
                }

            expected_state_size = 32
            expected_action_size = 9

            actual_state_size = hyperparams.get("state_size", expected_state_size)
            actual_action_size = hyperparams.get("action_size", expected_action_size)

            if (
                actual_state_size != expected_state_size
                or actual_action_size != expected_action_size
            ):
                return {
                    "path": model_path,
                    "valid": False,
                    "error": f"Incompatible model architecture: expected {expected_state_size}x{expected_action_size}, got {actual_state_size}x{actual_action_size}",
                }

            info = {
                "path": model_path,
                "training_step": checkpoint.get("training_step", "Unknown"),
                "epsilon": checkpoint.get("epsilon", "Unknown"),
                "hyperparameters": hyperparams,
                "file_size_mb": round(file_size / (1024 * 1024), 2),
                "valid": True,
                "error": None,
            }

            return info

        except Exception as e:
            error_msg = str(e)
            # Make error messages more user-friendly
            if "No module named 'torch'" in error_msg:
                error_msg = "PyTorch not installed. Please install with: pip install torch"
            elif "corrupted" in error_msg.lower():
                error_msg = "File appears to be corrupted"
            elif "permission" in error_msg.lower():
                error_msg = "Permission denied accessing file"

            return {"path": model_path, "valid": False, "error": error_msg}

    def _load_and_cache_model_info(self, model_path: str | None) -> dict[str, Any]:
        """Load model metadata once for selection/load flows and cache it by path."""
        model_info = self._load_model_info(model_path)
        if model_path is not None:
            self.model_info_by_path[model_path] = model_info
        return model_info

    def _create_trained_ai(self, model_path: str) -> Player | None:
        """Create a DQN AI player from a trained model"""
        try:
            # First validate the model
            model_info = self.model_info_by_path.get(model_path)
            if model_info is None:
                model_info = self._load_and_cache_model_info(model_path)
            self.model_info = model_info
            if not model_info["valid"]:
                self._show_error_message(f"Cannot load model: {model_info['error']}")
                return None

            # Get hyperparameters from model if available
            hyperparams = model_info.get("hyperparameters", {})
            state_size = hyperparams.get("state_size", 32)
            action_size = hyperparams.get("action_size", 9)

            # Create DQN agent with correct parameters
            from magic_pong.ai.models.dqn_ai import DQNAgent

            agent = DQNAgent(state_size=state_size, action_size=action_size, name="Trained DQN AI")

            # Load the trained model
            agent.load_model(model_path)

            # Set to evaluation mode (no exploration)
            agent.epsilon = 0.0
            agent.set_training_mode(False)

            training_steps = model_info.get("training_step", "Unknown")
            file_size = model_info.get("file_size_mb", "Unknown")
            print(f"Successfully loaded trained model: {os.path.basename(model_path)}")
            print(f"  Training steps: {training_steps}")
            print(f"  File size: {file_size} MB")
            print(f"  Network architecture: {state_size} → {action_size}")

            return agent

        except Exception as e:
            error_msg = f"Failed to create AI from model: {str(e)}"
            print(error_msg)
            self._show_error_message(error_msg)
            return None

    def _show_error_message(self, message: str, duration: float = 3.0) -> None:
        """Show an error message to the user"""
        self.error_message = message
        self.error_timer = duration
        print(f"Error: {message}")

    def open_config_menu(self, return_state: GameState = GameState.MENU) -> None:
        """Open settings and remember where ESC from the category screen should return."""
        self.config_exit_state = return_state
        self.config_category_selected = 0
        self.config_option_selected = 0
        self.config_current_category = None
        self.config_is_editing = False
        self.config_edit_backup = None
        self.state = GameState.CONFIG_CATEGORY
        print("Entering configuration menu")

    def handle_menu_input(self, event: pygame.event.Event) -> None:
        """Handle input in menu state"""
        if event.type == pygame.KEYDOWN:
            if event.key == pygame.K_UP:
                self.menu_selected = (self.menu_selected - 1) % len(self.menu_options)
            elif event.key == pygame.K_DOWN:
                self.menu_selected = (self.menu_selected + 1) % len(self.menu_options)
            elif event.key == pygame.K_RETURN or event.key == pygame.K_SPACE:
                selected_mode = self.menu_options[self.menu_selected].mode
                if selected_mode is None:
                    # Quit option
                    self.running = False
                elif selected_mode == GameMode.CONFIG:
                    # Enter configuration menu
                    self.open_config_menu(GameState.MENU)
                elif selected_mode == GameMode.LOAD_MODEL:
                    # Switch to model selection screen
                    self.state = GameState.MODEL_SELECTION
                    self.model_selected = 0
                    if self.available_models:
                        print("Entering model selection mode")
                    else:
                        print("No trained models found!")
                else:
                    self.start_game_mode(selected_mode)
            elif event.key == pygame.K_ESCAPE:
                self.running = False

    def handle_model_selection_input(self, event: pygame.event.Event) -> None:
        """Handle input in model selection state"""
        if event.type == pygame.KEYDOWN:
            if event.key == pygame.K_UP:
                if self.available_models:
                    self.model_selected = (self.model_selected - 1) % len(self.available_models)
            elif event.key == pygame.K_DOWN:
                if self.available_models:
                    self.model_selected = (self.model_selected + 1) % len(self.available_models)
            elif event.key == pygame.K_RETURN or event.key == pygame.K_SPACE:
                # Load selected model and start game
                if self.available_models:
                    selected_model = self.available_models[self.model_selected]
                    self.selected_model_path = selected_model["path"]
                    self.model_info = self._load_and_cache_model_info(self.selected_model_path)

                    if self.model_info["valid"]:
                        print(f"Loading model: {selected_model['name']}")
                        self.start_game_mode(GameMode.LOAD_MODEL)
                    else:
                        error_msg = (
                            f"Cannot load '{selected_model['name']}': {self.model_info['error']}"
                        )
                        self._show_error_message(error_msg, 4.0)
                        print(f"Error: {error_msg}")
                else:
                    self._show_error_message("No trained models found in models/", 3.0)
            elif event.key == pygame.K_ESCAPE:
                # Return to main menu
                self.state = GameState.MENU
                print("Returning to main menu")

    def handle_game_input(self, event: pygame.event.Event) -> None:
        """Handle input during gameplay"""
        action = self.input_manager.handle_event(event)

        if action == "menu":
            self.return_to_menu()
        elif action == "pause":
            self.toggle_pause()
        elif action == "help":
            self.show_help = not self.show_help
        elif action == "toggle_fps":
            self.renderer.toggle_fps_display()
        elif action == "toggle_debug":
            self.renderer.toggle_debug_display()
        elif action == "restart":
            if self.state == GameState.GAME_OVER:
                self.restart_game()
        elif action == "quit":
            self.running = False

    def handle_pause_input(self, event: pygame.event.Event) -> None:
        """Handle input in pause state"""
        if event.type == pygame.KEYDOWN:
            if event.key == pygame.K_UP:
                self.pause_selected = (self.pause_selected - 1) % len(self.pause_actions)
            elif event.key == pygame.K_DOWN:
                self.pause_selected = (self.pause_selected + 1) % len(self.pause_actions)
            elif event.key == pygame.K_RETURN:
                selected_action = self.pause_actions[self.pause_selected].action
                self._activate_pause_action(selected_action)
            elif event.key == pygame.K_p or event.key == pygame.K_SPACE:
                self.toggle_pause()
            elif event.key == pygame.K_ESCAPE:
                self.return_to_menu()

    def _activate_pause_action(self, action: PauseAction) -> None:
        """Apply a selected pause menu action."""
        if action == PauseAction.RESUME:
            self.toggle_pause()
        elif action == PauseAction.RESTART:
            self.restart_game()
        elif action == PauseAction.SETTINGS:
            self.open_config_menu(GameState.PAUSED)
        elif action == PauseAction.MAIN_MENU:
            self.return_to_menu()
        elif action == PauseAction.QUIT:
            self.running = False

    def handle_game_over_input(self, event: pygame.event.Event) -> None:
        """Handle input in game over state"""
        if event.type == pygame.KEYDOWN:
            if event.key == pygame.K_SPACE:
                self.restart_game()
            elif event.key == pygame.K_ESCAPE:
                self.return_to_menu()

    def handle_help_input(self, event: pygame.event.Event) -> None:
        """Handle input when help is shown"""
        if event.type == pygame.KEYDOWN:
            if event.key == pygame.K_F1 or event.key == pygame.K_ESCAPE:
                self.show_help = False

    def handle_config_category_input(self, event: pygame.event.Event) -> None:
        """Handle input in configuration category selection menu"""
        if event.type == pygame.KEYDOWN:
            if event.key == pygame.K_UP:
                self.config_category_selected = (self.config_category_selected - 1) % len(
                    CONFIG_CATEGORIES
                )
            elif event.key == pygame.K_DOWN:
                self.config_category_selected = (self.config_category_selected + 1) % len(
                    CONFIG_CATEGORIES
                )
            elif event.key == pygame.K_RETURN or event.key == pygame.K_SPACE:
                # Enter selected category
                self.config_current_category = CONFIG_CATEGORIES[self.config_category_selected]
                self.config_option_selected = 0
                self.state = GameState.CONFIG_OPTIONS
                print(f"Entering {self.config_current_category.name} configuration")
            elif event.key == pygame.K_s:
                # Save configuration
                self.config_confirm_action = "save"
                self.config_confirm_return_state = GameState.CONFIG_CATEGORY
                self.state = GameState.CONFIG_CONFIRM
            elif event.key == pygame.K_r:
                # Reset to defaults
                self.config_confirm_action = "reset"
                self.config_confirm_return_state = GameState.CONFIG_CATEGORY
                self.state = GameState.CONFIG_CONFIRM
            elif event.key == pygame.K_ESCAPE:
                # Return to the screen that opened settings
                self.state = self.config_exit_state
                if self.state == GameState.PAUSED:
                    print("Returning to pause menu")
                else:
                    print("Returning to main menu")

    def handle_config_options_input(self, event: pygame.event.Event) -> None:
        """Handle input in configuration options menu"""
        if not self.config_current_category:
            return

        options = self.config_current_category.options

        if event.type == pygame.KEYDOWN:
            if self.config_is_editing:
                # Currently editing a value
                self._handle_config_edit_input(event)
            else:
                # Navigating options
                if event.key == pygame.K_UP:
                    self.config_option_selected = (self.config_option_selected - 1) % len(options)
                elif event.key == pygame.K_DOWN:
                    self.config_option_selected = (self.config_option_selected + 1) % len(options)
                elif event.key == pygame.K_RETURN or event.key == pygame.K_SPACE:
                    # Start editing current option
                    selected_option = options[self.config_option_selected]
                    self.config_edit_backup = get_config_value(game_config, selected_option.key)
                    self.config_is_editing = True
                    print(f"Editing {selected_option.label}")
                elif event.key == pygame.K_s:
                    # Save configuration
                    self.config_confirm_action = "save"
                    self.config_confirm_return_state = GameState.CONFIG_OPTIONS
                    self.state = GameState.CONFIG_CONFIRM
                elif event.key == pygame.K_r:
                    # Reset to defaults
                    self.config_confirm_action = "reset"
                    self.config_confirm_return_state = GameState.CONFIG_OPTIONS
                    self.state = GameState.CONFIG_CONFIRM
                elif event.key == pygame.K_ESCAPE:
                    # Return to category menu
                    self.state = GameState.CONFIG_CATEGORY
                    self.config_current_category = None
                    print("Returning to category menu")

    def _handle_config_edit_input(self, event: pygame.event.Event) -> None:
        """Handle input while editing a configuration value"""
        if not self.config_current_category:
            return

        option = self.config_current_category.options[self.config_option_selected]
        current_value = get_config_value(game_config, option.key)

        if event.key == pygame.K_RETURN or (
            event.key == pygame.K_SPACE and option.field_type != ConfigFieldType.BOOLEAN
        ):
            # Confirm edit
            self.config_is_editing = False
            self.config_edit_backup = None
            print(f"Updated {option.label} to {get_config_value(game_config, option.key)}")

        elif event.key == pygame.K_ESCAPE:
            # Cancel edit
            if self.config_edit_backup is not None:
                set_config_value(game_config, option.key, self.config_edit_backup)
            self.config_is_editing = False
            self.config_edit_backup = None
            print(f"Cancelled editing {option.label}")

        elif option.field_type == ConfigFieldType.BOOLEAN:
            # Toggle boolean value (SPACE or LEFT/RIGHT)
            if (
                event.key == pygame.K_SPACE
                or event.key == pygame.K_LEFT
                or event.key == pygame.K_RIGHT
            ):
                new_value = not current_value
                set_config_value(game_config, option.key, new_value)

        elif option.field_type == ConfigFieldType.NUMERIC:
            # Adjust numeric value
            if event.key == pygame.K_LEFT:
                new_value = max(option.min_value, current_value - option.step)
                set_config_value(game_config, option.key, new_value)
            elif event.key == pygame.K_RIGHT:
                new_value = min(option.max_value, current_value + option.step)
                set_config_value(game_config, option.key, new_value)

        elif option.field_type == ConfigFieldType.SELECTION:
            # Cycle through selections
            if option.choices and (event.key == pygame.K_LEFT or event.key == pygame.K_RIGHT):
                current_index = option.choices.index(current_value)
                if event.key == pygame.K_LEFT:
                    new_index = (current_index - 1) % len(option.choices)
                else:
                    new_index = (current_index + 1) % len(option.choices)
                set_config_value(game_config, option.key, option.choices[new_index])

    def handle_config_confirm_input(self, event: pygame.event.Event) -> None:
        """Handle input in configuration confirmation dialog"""
        if event.type == pygame.KEYDOWN:
            if event.key == pygame.K_RETURN or event.key == pygame.K_SPACE:
                # Confirm action
                if self.config_confirm_action == "save":
                    self._save_configuration()
                elif self.config_confirm_action == "reset":
                    self._reset_configuration()
                self.state = self.config_confirm_return_state
                self.config_confirm_action = None
                self.config_confirm_return_state = GameState.CONFIG_CATEGORY
            elif event.key == pygame.K_ESCAPE:
                # Cancel action
                self.state = self.config_confirm_return_state
                self.config_confirm_action = None
                self.config_confirm_return_state = GameState.CONFIG_CATEGORY
                print("Cancelled action")

    def _save_configuration(self) -> None:
        """Save current configuration to file"""
        try:
            game_config.save_to_file()
            print("Configuration saved successfully!")
            self._show_error_message("Configuration saved successfully!", 2.0)
        except Exception as e:
            error_msg = f"Failed to save configuration: {str(e)}"
            print(error_msg)
            self._show_error_message(error_msg, 3.0)

    def _reset_configuration(self) -> None:
        """Reset configuration to defaults"""
        try:
            game_config.reset_to_defaults()
            print("Configuration reset to defaults!")
            self._show_error_message("Configuration reset to defaults!", 2.0)
        except Exception as e:
            error_msg = f"Failed to reset configuration: {str(e)}"
            print(error_msg)
            self._show_error_message(error_msg, 3.0)

    def toggle_pause(self) -> None:
        """Toggle pause state"""
        if self.state == GameState.PLAYING:
            self.game_engine.pause_game()
            self.state = GameState.PAUSED
            self.pause_selected = 0
        elif self.state == GameState.PAUSED:
            self.game_engine.pause_game()
            self.state = GameState.PLAYING

    def return_to_menu(self) -> None:
        """Return to main menu"""
        self.game_engine.stop_game()
        self.state = GameState.MENU
        self.current_mode = GameMode.MENU
        self.pause_selected = 0
        self.config_exit_state = GameState.MENU
        self.show_help = False
        print("Return to main menu")

    def restart_game(self) -> None:
        """Restart current game"""
        if self.current_mode != GameMode.MENU:
            self.start_game_mode(self.current_mode)

    def update(self) -> None:
        """Update game logic"""
        if self.state == GameState.PLAYING:
            # Update human player inputs
            self.input_manager.update_players()

            # Update game engine
            result = self.game_engine.update()

            # Check for game over
            if result["done"]:
                self.state = GameState.GAME_OVER
                self.game_over_timer = 0.0

        elif self.state == GameState.GAME_OVER:
            self.game_over_timer += 1.0 / 60.0  # Assuming 60 FPS

        # Update error message timer
        if self.error_message and self.error_timer > 0:
            self.error_timer -= 1.0 / 60.0
            if self.error_timer <= 0:
                self.error_message = None

    def _build_main_menu_data(self) -> list[dict[str, Any]]:
        """Build renderer-friendly data for the main menu."""
        options_data = []
        model_count = len(self.available_models)

        for option in self.menu_options:
            details = [{"label": label, "value": value} for label, value in option.details]
            status = "Ready"
            available = True

            if option.mode == GameMode.LOAD_MODEL:
                available = model_count > 0
                if model_count == 1:
                    status = "1 model available"
                elif model_count > 1:
                    status = f"{model_count} models available"
                else:
                    status = "No models available"
                details.append({"label": "Availability", "value": status})
            elif option.mode == GameMode.CONFIG:
                status = "Available"
            elif option.mode is None:
                status = "Exit"

            options_data.append(
                {
                    "label": option.label,
                    "mode": option.mode.value if option.mode else None,
                    "title": option.title,
                    "description": option.description,
                    "details": details,
                    "status": status,
                    "available": available,
                }
            )

        return options_data

    def _build_pause_actions_data(self) -> list[dict[str, Any]]:
        """Build renderer-friendly data for the pause action menu."""
        actions_data = []

        for action in self.pause_actions:
            details = [{"label": label, "value": value} for label, value in action.details]
            if action.action in {PauseAction.RESTART, PauseAction.MAIN_MENU}:
                details.append({"label": "Current mode", "value": self.current_mode.value})

            actions_data.append(
                {
                    "label": action.label,
                    "action": action.action.value,
                    "title": action.title,
                    "description": action.description,
                    "details": details,
                    "status": "Paused",
                    "available": True,
                }
            )

        return actions_data

    def _build_model_selection_data(self) -> list[dict[str, Any]]:
        """Build renderer-friendly data for the trained model browser."""
        models_data = []

        for model in self.available_models:
            path = str(model.get("path") or "")
            file_name = str(model.get("file") or os.path.basename(path) or "Unknown file")
            display_name = str(
                model.get("name") or file_name.replace(".pth", "").replace("_", " ").title()
            )
            model_info = self.model_info_by_path.get(path)
            is_model_file = bool(path and os.path.isfile(path))
            available = is_model_file
            status = "File found" if is_model_file else "Missing file"
            description = (
                "Saved DQN checkpoint discovered locally. Select it to validate and load it "
                "for a player-vs-model match."
            )

            details = [
                {"label": "Display name", "value": display_name},
                {"label": "File", "value": file_name},
                {"label": "Location", "value": self._format_model_location(path)},
                {
                    "label": "Availability",
                    "value": status,
                    "tone": "success" if is_model_file else "error",
                },
            ]

            file_size = self._format_model_file_size(path)
            if file_size:
                details.append({"label": "File size", "value": file_size})

            if model_info is not None:
                is_valid = bool(model_info.get("valid"))
                if is_valid and is_model_file:
                    status = "Valid"
                    available = True
                    description = "Validated checkpoint ready to load as the trained DQN opponent."
                    details.append(
                        {"label": "Validity", "value": "Valid checkpoint", "tone": "success"}
                    )
                else:
                    status = "Invalid"
                    available = False
                    description = (
                        "This checkpoint was checked and cannot be loaded until the issue below "
                        "is resolved."
                    )
                    details.append(
                        {
                            "label": "Validity",
                            "value": "Failed validation",
                            "tone": "error",
                        }
                    )

                error = model_info.get("error")
                if error:
                    details.append({"label": "Error", "value": str(error), "tone": "error"})

                if "training_step" in model_info:
                    details.append(
                        {
                            "label": "Training step",
                            "value": self._format_model_value(model_info["training_step"]),
                        }
                    )
                if "epsilon" in model_info:
                    details.append(
                        {
                            "label": "Epsilon",
                            "value": self._format_model_value(model_info["epsilon"]),
                        }
                    )

                hyperparameters = self._format_model_hyperparameters(
                    model_info.get("hyperparameters")
                )
                if hyperparameters:
                    details.append({"label": "Hyperparameters", "value": hyperparameters})

            models_data.append(
                {
                    "label": display_name,
                    "title": display_name,
                    "file": file_name,
                    "path": path,
                    "description": description,
                    "details": details,
                    "status": status,
                    "available": available,
                }
            )

        return models_data

    def _format_model_location(self, model_path: str) -> str:
        """Return a compact directory hint for a model path."""
        if not model_path:
            return "Unknown"
        directory = os.path.dirname(model_path)
        return directory or "."

    def _format_model_file_size(self, model_path: str) -> str:
        """Return a display-friendly file size without loading the checkpoint."""
        if not model_path or not os.path.isfile(model_path):
            return ""

        try:
            file_size = os.path.getsize(model_path)
        except OSError:
            return ""

        if file_size < 1024:
            return f"{file_size} B"
        if file_size < 1024 * 1024:
            return f"{file_size / 1024:.1f} KB"
        return f"{file_size / (1024 * 1024):.2f} MB"

    def _format_model_value(self, value: Any) -> str:
        """Format model checkpoint values for compact display."""
        if isinstance(value, float):
            return f"{value:.4g}"
        return str(value)

    def _format_model_hyperparameters(self, hyperparameters: Any) -> str:
        """Format the most relevant model hyperparameters in one compact row."""
        if not isinstance(hyperparameters, dict):
            return ""
        if not hyperparameters:
            return "Unavailable"

        labels = {
            "state_size": "state",
            "action_size": "actions",
            "lr": "lr",
            "learning_rate": "lr",
            "gamma": "gamma",
            "epsilon_decay": "eps decay",
            "batch_size": "batch",
        }
        priority_keys = list(labels)
        parts = [
            f"{labels[key]}={self._format_model_value(hyperparameters[key])}"
            for key in priority_keys
            if key in hyperparameters
        ]

        remaining_keys = sorted(key for key in hyperparameters if key not in labels)
        for key in remaining_keys[: max(0, 5 - len(parts))]:
            parts.append(f"{key}={self._format_model_value(hyperparameters[key])}")

        return ", ".join(parts[:5])

    def _build_config_options_data(self, category: ConfigCategory) -> list[dict[str, Any]]:
        """Build renderer-friendly option data for a configuration category."""
        options_data = []
        for opt in category.options:
            options_data.append(
                {
                    "label": opt.label,
                    "value": get_config_value(game_config, opt.key),
                    "field_type": opt.field_type.value,
                    "min_value": opt.min_value,
                    "max_value": opt.max_value,
                    "step": opt.step,
                    "choices": opt.choices,
                    "description": opt.description,
                }
            )
        return options_data

    def render(self) -> None:
        """Render the current state"""
        if self.state == GameState.MENU:
            self.renderer.draw_menu(self._build_main_menu_data(), self.menu_selected)

        elif self.state == GameState.MODEL_SELECTION:
            self.renderer.draw_model_selection_menu(
                self._build_model_selection_data(), self.model_selected
            )

        elif self.state == GameState.CONFIG_CATEGORY:
            # Draw configuration category menu
            category_names = [cat.name for cat in CONFIG_CATEGORIES]
            self.renderer.draw_config_category_menu(category_names, self.config_category_selected)

        elif self.state == GameState.CONFIG_OPTIONS:
            # Draw configuration options menu
            if self.config_current_category:
                category_names = [cat.name for cat in CONFIG_CATEGORIES]
                options_data = self._build_config_options_data(self.config_current_category)
                self.renderer.draw_config_option_menu(
                    self.config_current_category.name,
                    options_data,
                    self.config_option_selected,
                    self.config_is_editing,
                    category_names,
                    self.config_category_selected,
                )

        elif self.state == GameState.CONFIG_CONFIRM:
            # Draw confirmation dialog
            if self.config_confirm_action == "save":
                message = "Save current configuration to file?"
                title = "Save Configuration"
            elif self.config_confirm_action == "reset":
                message = "Reset all settings to default values?"
                title = "Reset Configuration"
            else:
                message = "Confirm action?"
                title = "Confirm"

            category_names = [cat.name for cat in CONFIG_CATEGORIES]
            if (
                self.config_confirm_return_state == GameState.CONFIG_OPTIONS
                and self.config_current_category
            ):
                # First draw the options menu as background
                options_data = self._build_config_options_data(self.config_current_category)
                self.renderer.draw_config_option_menu(
                    self.config_current_category.name,
                    options_data,
                    self.config_option_selected,
                    self.config_is_editing,
                    category_names,
                    self.config_category_selected,
                )
            else:
                # First draw the category menu as background
                self.renderer.draw_config_category_menu(
                    category_names, self.config_category_selected
                )
            # Then overlay the confirmation dialog
            self.renderer.draw_confirmation_dialog(message, title)

        elif self.state in [GameState.PLAYING, GameState.PAUSED, GameState.GAME_OVER]:
            # Get game state and render
            game_state = self.game_engine.get_game_state()

            # Add debug info if needed
            additional_info = {}
            if self.renderer.show_debug:
                additional_info["debug_info"] = {
                    "Mode": self.current_mode.value,
                    "State": self.state.value,
                    "Players": len([p for p in self.human_players.values() if p is not None]),
                }

            self.renderer.render_game_state(game_state, additional_info)

            # Overlay pause screen
            if self.state == GameState.PAUSED:
                self.renderer.draw_pause_screen(
                    self._build_pause_actions_data(), self.pause_selected
                )

            # Overlay game over screen
            elif self.state == GameState.GAME_OVER:
                winner = self.game_engine.get_winner()
                score = game_state["score"]
                self.renderer.draw_game_over(winner, score)

        # Show help overlay if requested
        if self.show_help and self.state != GameState.MENU:
            self.renderer.draw_controls_help()

        # Show error message overlay if active
        if self.error_message:
            self.renderer.draw_error_message(self.error_message)

        # Present the frame
        self.renderer.present()

    def run(self) -> None:
        """Main application loop"""
        print("Starting Magic Pong...")

        try:
            while self.running:
                # Handle events
                for event in pygame.event.get():
                    if event.type == pygame.QUIT:
                        self.running = False
                        continue

                    # Route events based on current state
                    if self.show_help and self.state != GameState.MENU:
                        self.handle_help_input(event)
                    elif self.state == GameState.MENU:
                        self.handle_menu_input(event)
                    elif self.state == GameState.MODEL_SELECTION:
                        self.handle_model_selection_input(event)
                    elif self.state == GameState.CONFIG_CATEGORY:
                        self.handle_config_category_input(event)
                    elif self.state == GameState.CONFIG_OPTIONS:
                        self.handle_config_options_input(event)
                    elif self.state == GameState.CONFIG_CONFIRM:
                        self.handle_config_confirm_input(event)
                    elif self.state == GameState.PLAYING:
                        self.handle_game_input(event)
                    elif self.state == GameState.PAUSED:
                        self.handle_pause_input(event)
                    elif self.state == GameState.GAME_OVER:
                        self.handle_game_over_input(event)

                # Update game logic
                self.update()

                # Render frame
                self.render()

                # Control frame rate
                self.renderer.update()

        except Exception as e:
            print(f"Error during execution: {e}")
            traceback.print_exc()

        finally:
            self.cleanup()

    def cleanup(self) -> None:
        """Clean up resources"""
        print("Cleaning up resources...")
        self.game_engine.stop_game()
        self.renderer.cleanup()
        pygame.quit()
        print("Magic Pong closed properly.")


def main() -> None:
    """Main entry point"""
    try:
        app = MagicPongApp()
        app.run()
    except KeyboardInterrupt:
        print("\nUser interruption")
    except Exception as e:
        print(f"Fatal error: {e}")
        traceback.print_exc()
    finally:
        # Ensure pygame is properly closed
        try:
            pygame.quit()
        except Exception:
            pass
        sys.exit(0)


if __name__ == "__main__":
    main()
