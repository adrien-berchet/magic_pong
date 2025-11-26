"""
Simple Magic Pong game example with graphical interface
"""

import sys

try:
    from magic_pong.gui.game_app import MagicPongApp
except ImportError as e:
    print(f"Error: Unable to import required modules: {e}")
    print("Make sure pygame is installed: pip install pygame")
    sys.exit(1)


def run_simple_game():
    """Launch a simple 1 vs 1 game"""
    print("Launching a Magic Pong 1 vs 1 game...")

    app = MagicPongApp()

    # Optional: start directly in 1v1 mode
    # app.start_game_mode(GameMode.ONE_VS_ONE)

    app.run()


if __name__ == "__main__":
    run_simple_game()
