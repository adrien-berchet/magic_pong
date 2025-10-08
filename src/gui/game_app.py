"""
Main game application with PyGame GUI
"""

import os
import sys
import traceback
from enum import Enum

import pygame
from magic_pong.ai.models.dqn_ai import DQNAgent
from magic_pong.ai.models.simple_ai import FollowBallAI as SimpleAI
from magic_pong.core.entities import Player
from magic_pong.core.game_engine import GameEngine
from magic_pong.gui.human_player import HumanPlayer, InputManager, create_human_players
from magic_pong.gui.pygame_renderer import PygameRenderer


class GameMode(Enum):
    """Available game modes"""

    MENU = "menu"
    ONE_VS_ONE = "1v1"
    ONE_VS_AI = "1vAI"
    LOAD_MODEL = "load_model"
    AI_DEMO = "demo"


class GameState(Enum):
    """Current application state"""

    MENU = "menu"
    MODEL_SELECTION = "model_selection"
    PLAYING = "playing"
    PAUSED = "paused"
    GAME_OVER = "game_over"
    HELP = "help"


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
            ("1 vs 1 (Two players)", GameMode.ONE_VS_ONE),
            ("1 vs AI (Player vs AI)", GameMode.ONE_VS_AI),
            ("Play vs Trained Model", GameMode.LOAD_MODEL),
            ("AI vs AI (Demonstration)", GameMode.AI_DEMO),
            ("Quit", None),
        ]

        # Game state
        self.game_over_timer = 0.0
        self.show_help = False

        # Model selection state
        self.model_selected = 0
        self.available_models: list[dict] = []
        self.selected_model_path = None
        self.model_info: dict | None = None
        self.error_message: str | None = None
        self.error_timer: float = 0.0

        # Players
        self.human_players: dict[int, HumanPlayer | None] = {1: None, 2: None}

        print("Magic Pong initialized successfully!")
        print("Use arrows to navigate and ENTER to select")

        # Initialize available models
        self._discover_models()

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
            ai_players[2] = SimpleAI("AI 2")

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
            if model_path is None or not os.path.exists(model_path):
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
            required_keys = ["q_network_state_dict", "target_network_state_dict"]
            missing_keys = [key for key in required_keys if key not in checkpoint]

            if missing_keys:
                return {
                    "path": model_path,
                    "valid": False,
                    "error": f'Missing required data: {", ".join(missing_keys)}',
                }

            # Get hyperparameters for compatibility check
            hyperparams = checkpoint.get("hyperparameters", {})
            expected_state_size = 28
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

    def _create_trained_ai(self, model_path: str) -> Player | None:
        """Create a DQN AI player from a trained model"""
        try:
            # First validate the model
            model_info = self._load_model_info(model_path)
            if not model_info["valid"]:
                self._show_error_message(f"Cannot load model: {model_info['error']}")
                return None

            # Get hyperparameters from model if available
            hyperparams = model_info.get("hyperparameters", {})
            state_size = hyperparams.get("state_size", 28)
            action_size = hyperparams.get("action_size", 9)

            # Create DQN agent with correct parameters
            agent = DQNAgent(
                state_size=state_size, action_size=action_size, player_id=2, name="Trained DQN AI"
            )

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

    def handle_menu_input(self, event: pygame.event.Event) -> None:
        """Handle input in menu state"""
        if event.type == pygame.KEYDOWN:
            if event.key == pygame.K_UP:
                self.menu_selected = (self.menu_selected - 1) % len(self.menu_options)
            elif event.key == pygame.K_DOWN:
                self.menu_selected = (self.menu_selected + 1) % len(self.menu_options)
            elif event.key == pygame.K_RETURN or event.key == pygame.K_SPACE:
                selected_mode = self.menu_options[self.menu_selected][1]
                if selected_mode is None:
                    # Quit option
                    self.running = False
                elif selected_mode == GameMode.LOAD_MODEL:
                    # Switch to model selection screen
                    if self.available_models:
                        self.state = GameState.MODEL_SELECTION
                        self.model_selected = 0
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
                self.model_selected = (self.model_selected - 1) % len(self.available_models)
            elif event.key == pygame.K_DOWN:
                self.model_selected = (self.model_selected + 1) % len(self.available_models)
            elif event.key == pygame.K_RETURN or event.key == pygame.K_SPACE:
                # Load selected model and start game
                if self.available_models:
                    selected_model = self.available_models[self.model_selected]
                    self.selected_model_path = selected_model["path"]
                    self.model_info = self._load_model_info(self.selected_model_path)

                    if self.model_info["valid"]:
                        print(f"Loading model: {selected_model['name']}")
                        self.start_game_mode(GameMode.LOAD_MODEL)
                    else:
                        error_msg = (
                            f"Cannot load '{selected_model['name']}': {self.model_info['error']}"
                        )
                        self._show_error_message(error_msg, 4.0)
                        print(f"Error: {error_msg}")
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
            if event.key == pygame.K_p or event.key == pygame.K_SPACE:
                self.toggle_pause()
            elif event.key == pygame.K_ESCAPE:
                self.return_to_menu()

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

    def toggle_pause(self) -> None:
        """Toggle pause state"""
        if self.state == GameState.PLAYING:
            self.game_engine.pause_game()
            self.state = GameState.PAUSED
        elif self.state == GameState.PAUSED:
            self.game_engine.pause_game()
            self.state = GameState.PLAYING

    def return_to_menu(self) -> None:
        """Return to main menu"""
        self.game_engine.stop_game()
        self.state = GameState.MENU
        self.current_mode = GameMode.MENU
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

    def render(self) -> None:
        """Render the current state"""
        if self.state == GameState.MENU:
            self.renderer.draw_menu(self.menu_selected)

        elif self.state == GameState.MODEL_SELECTION:
            self.renderer.draw_model_selection_menu(self.available_models, self.model_selected)

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
                self.renderer.draw_pause_screen()

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
