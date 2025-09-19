"""
Main game application with PyGame GUI
"""

import sys
import traceback
from enum import Enum

import pygame
from magic_pong.ai.examples.simple_ai import FollowBallAI as SimpleAI
from magic_pong.ai.interface import AIPlayer
from magic_pong.core.game_engine import GameEngine
from magic_pong.gui.human_player import HumanPlayer, InputManager, create_human_players
from magic_pong.gui.pygame_renderer import PygameRenderer


class GameMode(Enum):
    """Available game modes"""

    MENU = "menu"
    ONE_VS_ONE = "1v1"
    ONE_VS_AI = "1vAI"
    AI_DEMO = "demo"


class GameState(Enum):
    """Current application state"""

    MENU = "menu"
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
            ("AI vs AI (Demonstration)", GameMode.AI_DEMO),
            ("Quit", None),
        ]

        # Game state
        self.game_over_timer = 0.0
        self.show_help = False

        # Players
        self.human_players: dict[int, HumanPlayer | None] = {1: None, 2: None}

        print("Magic Pong initialized successfully!")
        print("Use arrows to navigate and ENTER to select")

    def start_game_mode(self, mode: GameMode) -> None:
        """Start a specific game mode"""
        self.current_mode = mode

        # Create human players based on mode
        self.human_players = create_human_players(mode.value)

        # Clear input manager and add human players
        self.input_manager = InputManager()
        for player in self.human_players.values():
            if player is not None:
                self.input_manager.add_player(player)

        # Create AI players as needed

        ai_players: dict[int, AIPlayer | None] = {1: None, 2: None}

        if mode == GameMode.ONE_VS_AI:
            # Human vs AI
            ai_players[2] = SimpleAI(2, "Simple AI")

        elif mode == GameMode.AI_DEMO:
            # AI vs AI
            ai_players[1] = SimpleAI(1, "AI 1")
            ai_players[2] = SimpleAI(2, "AI 2")

        # Set players in game engine
        player1 = self.human_players[1] or ai_players[1]
        player2 = self.human_players[2] or ai_players[2]

        self.game_engine.set_players(player1, player2)
        self.game_engine.start_game()

        # Change state to playing
        self.state = GameState.PLAYING

        print(f"Game mode {mode.value} started!")

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
                else:
                    self.start_game_mode(selected_mode)
            elif event.key == pygame.K_ESCAPE:
                self.running = False

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

    def render(self) -> None:
        """Render the current state"""
        if self.state == GameState.MENU:
            self.renderer.draw_menu(self.menu_selected)

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
