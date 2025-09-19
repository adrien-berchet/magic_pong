"""
Examples of using different game modes of Magic Pong
"""

import sys

try:
    from magic_pong.gui.game_app import GameMode, MagicPongApp
except ImportError as e:
    print(f"Error: Unable to import required modules: {e}")
    print("Make sure pygame is installed: pip install pygame")
    print("And that you are in the correct directory.")
    sys.exit(1)


def demo_1v1():
    """Demo of 1 player vs 1 player mode"""
    print("=== DEMO: 1 Player vs 1 Player ===")
    print("Player 1 (Left): W/A/S/D")
    print("Player 2 (Right): Arrow keys")
    print("Press ESC to return to menu")
    print()

    app = MagicPongApp()
    app.start_game_mode(GameMode.ONE_VS_ONE)
    app.run()


def demo_1v_ai():
    """Demo of 1 player vs AI mode"""
    print("=== DEMO: 1 Player vs AI ===")
    print("Player (Left): W/A/S/D")
    print("AI (Right): Controlled automatically")
    print("Press ESC to return to menu")
    print()

    app = MagicPongApp()
    app.start_game_mode(GameMode.ONE_VS_AI)
    app.run()


def demo_ai_vs_ai():
    """Demo of AI vs AI mode"""
    print("=== DEMO: AI vs AI ===")
    print("Both players are controlled by AI")
    print("Watch the demonstration!")
    print("Press ESC to return to menu")
    print()

    app = MagicPongApp()
    app.start_game_mode(GameMode.AI_DEMO)
    app.run()


def show_menu():
    """Show main menu"""
    print("=== MAGIC PONG - EXAMPLES ===")
    print()
    print("Choose a demonstration mode:")
    print("1. 1 Player vs 1 Player")
    print("2. 1 Player vs AI")
    print("3. AI vs AI (Demonstration)")
    print("4. Full game menu")
    print("5. Quit")
    print()

    try:
        choice = input("Your choice (1-5): ").strip()

        if choice == "1":
            demo_1v1()
        elif choice == "2":
            demo_1v_ai()
        elif choice == "3":
            demo_ai_vs_ai()
        elif choice == "4":
            print("Launching full menu...")
            app = MagicPongApp()
            app.run()
        elif choice == "5":
            print("Goodbye!")
            return
        else:
            print("Invalid choice, please try again.")
            show_menu()

    except KeyboardInterrupt:
        print("\nGoodbye!")
    except Exception as e:
        print(f"Error: {e}")


if __name__ == "__main__":
    show_menu()
