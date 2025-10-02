"""
Magic Pong main game engine
"""

import time
from typing import Any

import pygame
from magic_pong.ai.interface import AIPlayer, GameEnvironment
from magic_pong.core.entities import Action
from magic_pong.core.physics import PhysicsEngine
from magic_pong.utils.config import game_config


class GameEngine:
    """Main engine that orchestrates the game"""

    def __init__(self, headless: bool = False):
        self.headless = headless
        self.physics_engine = PhysicsEngine(game_config.FIELD_WIDTH, game_config.FIELD_HEIGHT)

        # Environment for AI
        self.ai_environment = GameEnvironment(self.physics_engine, headless)

        # Game state
        self.running = False
        self.paused = False
        self.last_update_time = 0.0

        # Players (can be human or AI)
        self.player1: AIPlayer | None = None
        self.player2: AIPlayer | None = None

        # Statistics
        self.total_games = 0
        self.game_stats = {
            "player1_wins": 0,
            "player2_wins": 0,
            "total_steps": 0,
            "average_game_length": 0.0,
        }

    def set_players(self, player1: AIPlayer | None, player2: AIPlayer | None) -> None:
        """Sets the players (human or AI)"""
        self.player1 = player1
        self.player2 = player2

        # Notify AI players of episode start
        if player1 and hasattr(player1, "on_episode_start"):
            player1.on_episode_start()
        if player2 and hasattr(player2, "on_episode_start"):
            player2.on_episode_start()

    def start_game(self) -> None:
        """Starts a new game"""
        self.running = True
        self.paused = False
        self.physics_engine.reset_game()
        self.last_update_time = time.time()

        # Reset AI environment
        self.ai_environment.reset()

    def stop_game(self) -> None:
        """Stops the current game"""
        self.running = False

        # Notify AI players of episode end
        if self.player1 and hasattr(self.player1, "on_episode_end"):
            self.player1.on_episode_end(0.0)
        if self.player2 and hasattr(self.player2, "on_episode_end"):
            self.player2.on_episode_end(0.0)

    def pause_game(self) -> None:
        """Pauses / resumes the game"""
        self.paused = not self.paused
        if not self.paused:
            self.last_update_time = time.time()

    def update(self, dt: float | None = None) -> dict[str, Any]:
        """
        Updates the game by one frame

        Args:
            dt: Delta time in seconds. If None, calculated automatically

        Returns:
            Dict containing events and game state
        """
        if not self.running or self.paused:
            return {"events": {}, "game_state": self.physics_engine.get_game_state()}

        # Calculate delta time
        if dt is None:
            current_time = time.time()
            dt = current_time - self.last_update_time
            self.last_update_time = current_time

        # Limit delta time to avoid large jumps
        dt = min(dt, 1.0 / 30.0)  # Max 30 FPS minimum

        # Get player actions
        action1 = self._get_player_action(self.player1, 1)
        action2 = self._get_player_action(self.player2, 2)

        # Update physics via AI environment
        obs1, obs2, reward1, reward2, done, info = self.ai_environment.step(action1, action2)

        # Notify AI players
        if self.player1 and hasattr(self.player1, "on_step"):
            # Create default action if None
            safe_action1 = action1 if action1 is not None else Action(move_x=0.0, move_y=0.0)
            self.player1.on_step(obs1, safe_action1, reward1, done, info)
        if self.player2 and hasattr(self.player2, "on_step"):
            # Create default action if None
            safe_action2 = action2 if action2 is not None else Action(move_x=0.0, move_y=0.0)
            self.player2.on_step(obs2, safe_action2, reward2, done, info)

        # Check for game end
        if done:
            self._handle_game_end(info)

        return {
            "events": info["events"],
            "game_state": info["game_state"],
            "observations": {"player1": obs1, "player2": obs2},
            "rewards": {"player1": reward1, "player2": reward2},
            "done": done,
            "info": info,
        }

    def _get_player_action(self, player: AIPlayer | None, player_id: int) -> Action | None:
        """Gets a player's action"""
        if player is None:
            return None

        if hasattr(player, "get_human_action"):
            # Human player (check this first!)
            human_action: Action | None = player.get_human_action()
            return human_action
        elif hasattr(player, "get_action"):
            # AI player
            game_state = self.physics_engine.get_game_state()
            observation = self.ai_environment.observation_processor.process_game_state(
                game_state, player_id
            )
            action: Action = player.get_action(observation)
            return action
        else:
            return None

    def _handle_game_end(self, info: dict[str, Any]) -> None:
        """Handles the end of a game"""
        winner = info.get("winner", 0)

        # Update statistics
        self.total_games += 1
        if winner == 1:
            self.game_stats["player1_wins"] += 1
        elif winner == 2:
            self.game_stats["player2_wins"] += 1

        self.game_stats["total_steps"] += info.get("step_count", 0)
        self.game_stats["average_game_length"] = self.game_stats["total_steps"] / self.total_games

        # Notify AI players
        if self.player1 and hasattr(self.player1, "on_episode_end"):
            final_reward = 1.0 if winner == 1 else -1.0 if winner == 2 else 0.0
            self.player1.on_episode_end(final_reward)
        if self.player2 and hasattr(self.player2, "on_episode_end"):
            final_reward = 1.0 if winner == 2 else -1.0 if winner == 1 else 0.0
            self.player2.on_episode_end(final_reward)

        # Stop the game
        self.running = False

    def get_game_state(self) -> dict[str, Any]:
        """Returns the complete game state"""
        return self.physics_engine.get_game_state()

    def get_stats(self) -> dict[str, Any]:
        """Returns game statistics"""
        stats: dict[str, Any] = self.game_stats.copy()

        # Add AI player stats
        if self.player1 and hasattr(self.player1, "get_stats"):
            stats["player1_ai_stats"] = self.player1.get_stats()
        if self.player2 and hasattr(self.player2, "get_stats"):
            stats["player2_ai_stats"] = self.player2.get_stats()

        return stats

    def reset_stats(self) -> None:
        """Resets statistics to zero"""
        self.total_games = 0
        self.game_stats = {
            "player1_wins": 0,
            "player2_wins": 0,
            "total_steps": 0,
            "average_game_length": 0.0,
        }

    def set_speed_multiplier(self, multiplier: float) -> None:
        """Sets the speed multiplier for training"""
        game_config.GAME_SPEED_MULTIPLIER = multiplier

    def is_running(self) -> bool:
        """Checks if the game is running"""
        return self.running

    def is_paused(self) -> bool:
        """Checks if the game is paused"""
        return self.paused

    def is_game_over(self) -> bool:
        """Checks if the game is over"""
        return self.physics_engine.is_game_over()

    def get_winner(self) -> int:
        """Returns the game winner"""
        return self.physics_engine.get_winner()


class TrainingManager:
    """Manager for AI training"""

    def __init__(
        self,
        headless: bool = True,
        initial_ball_direction: int = 0,
        initial_ball_angle: float | None = None,
        fast_gui: bool = False,
    ):
        self.game_engine = GameEngine(headless=headless)
        self.headless = headless
        self.renderer = None
        self.fast_gui = fast_gui
        self.initial_ball_direction = initial_ball_direction
        self.initial_ball_angle = initial_ball_angle

        # Initialize renderer if not headless
        if not headless:
            try:
                from magic_pong.gui.pygame_renderer import PygameRenderer

                self.renderer = PygameRenderer()
                print("🎮 GUI renderer initialized for training visualization " + "(fast mode enabled)" if fast_gui else "(normal speed)")
            except ImportError:
                print("⚠️  Warning: Could not import PygameRenderer, falling back to headless mode")
                self.headless = True

        self.training_stats: dict[str, Any] = {
            "episodes": 0,
            "total_steps": 0,
            "player1_wins": 0,
            "player2_wins": 0,
            "average_episode_length": 0.0,
            "average_rewards": {"player1": 0.0, "player2": 0.0},
        }

    def train_episode(
        self, player1: AIPlayer, player2: AIPlayer, max_steps: int = 10000
    ) -> dict[str, Any]:
        """
        Trains a complete episode

        Args:
            player1: First player (AI)
            player2: Second player (AI)
            max_steps: Maximum number of steps per episode

        Returns:
            Dict: Episode statistics
        """
        self.game_engine.set_players(player1, player2)
        self.game_engine.start_game()

        # Set initial ball direction if specified
        if self.initial_ball_direction != 0 or self.initial_ball_angle is not None:
            physics_engine = self.game_engine.ai_environment.physics_engine
            physics_engine.reset_ball(self.initial_ball_direction, self.initial_ball_angle)

        episode_stats: dict[str, Any] = {
            "steps": 0,
            "winner": 0,
            "total_reward_p1": 0.0,
            "total_reward_p2": 0.0,
            "events": [],
        }

        while self.game_engine.is_running() and episode_stats["steps"] < max_steps:
            # Handle pygame events if renderer is active
            if self.renderer is not None:
                for event in pygame.event.get():
                    if event.type == pygame.QUIT:
                        print("Training interrupted by user")
                        return episode_stats
                    elif event.type == pygame.KEYDOWN:
                        if event.key == pygame.K_ESCAPE:
                            print("Training interrupted by ESC key")
                            return episode_stats

            result = self.game_engine.update()

            # Render the game if renderer is available
            if self.renderer is not None:
                game_state = result["game_state"]

                # Add training info to display
                training_info = {
                    "episode": self.training_stats["episodes"] + 1,
                    "step": episode_stats["steps"],
                    "reward_p1": episode_stats["total_reward_p1"],
                    "reward_p2": episode_stats["total_reward_p2"],
                    "player1_wins": self.training_stats["player1_wins"],
                    "player2_wins": self.training_stats["player2_wins"],
                }
                data = {"training_info": training_info}
                if "optimal_points" in result["info"]:
                    data["optimal_points"] = result["info"]["optimal_points"]

                self.renderer.render_game_state(game_state, data)
                self.renderer.present()

                # Control frame rate to make visualization watchable
                self.renderer.update(1000 if self.fast_gui else game_config.FPS)
                # self.renderer.update(600)
                # print(f"+++++ fast_gui in train_episode: {self.fast_gui} ; actual fps: {self.renderer.clock.get_fps()}")  # Debug line

            episode_stats["steps"] += 1
            episode_stats["total_reward_p1"] += result["rewards"]["player1"]
            episode_stats["total_reward_p2"] += result["rewards"]["player2"]
            episode_stats["events"].extend(result["events"].get("goals", []))

            if result["done"]:
                episode_stats["winner"] = result["info"].get("winner", 0)
                break

        # Update training statistics
        self._update_training_stats(episode_stats)

        print("++++++++++++++++++++ fast_gui: ", self.fast_gui)  # Debug line

        return episode_stats

    def _update_training_stats(self, episode_stats: dict[str, Any]) -> None:
        """Updates training statistics"""
        self.training_stats["episodes"] += 1
        self.training_stats["total_steps"] += episode_stats["steps"]

        if episode_stats["winner"] == 1:
            self.training_stats["player1_wins"] += 1
        elif episode_stats["winner"] == 2:
            self.training_stats["player2_wins"] += 1

        # Moving average of rewards
        alpha = 0.01  # Smoothing factor
        self.training_stats["average_rewards"]["player1"] = (1 - alpha) * self.training_stats[
            "average_rewards"
        ]["player1"] + alpha * episode_stats["total_reward_p1"]
        self.training_stats["average_rewards"]["player2"] = (1 - alpha) * self.training_stats[
            "average_rewards"
        ]["player2"] + alpha * episode_stats["total_reward_p2"]

        self.training_stats["average_episode_length"] = (
            self.training_stats["total_steps"] / self.training_stats["episodes"]
        )

    def get_training_stats(self) -> dict[str, Any]:
        """Returns training statistics"""
        return self.training_stats.copy()

    def reset_training_stats(self) -> None:
        """Resets training statistics to zero"""
        self.training_stats = {
            "episodes": 0,
            "total_steps": 0,
            "player1_wins": 0,
            "player2_wins": 0,
            "average_episode_length": 0.0,
            "average_rewards": {"player1": 0.0, "player2": 0.0},
        }

    def set_ball_initial_direction(
        self, direction: int = 0, angle_rad: float | None = None
    ) -> None:
        """Set the initial ball direction for the next episodes"""
        self.initial_ball_direction = direction
        self.initial_ball_angle = angle_rad

    def cleanup(self) -> None:
        """Clean up resources, especially the renderer"""
        if self.renderer is not None:
            self.renderer.cleanup()
            self.renderer = None
