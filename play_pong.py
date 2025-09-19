#!/usr/bin/env python3
"""
Main script to launch Magic Pong with PyGame graphical interface
"""

import importlib
import sys

try:
    from magic_pong.gui.game_app import main
    from magic_pong.utils.keyboard_layout import auto_configure_layout

except ImportError as e:
    print(f"Import error: {e}")
    print()
    print("Checking dependencies:")
    if importlib.util.find_spec("pygame") is not None:
        print("✓ pygame is installed")
    else:
        print("✗ pygame is not installed - pip install pygame")

    if importlib.util.find_spec("numpy") is not None:
        print("✓ numpy is installed")
    else:
        print("✗ numpy is not installed - pip install numpy")

if __name__ == "__main__":
    print("=== MAGIC PONG ===")
    print("Advanced Pong game with AI")
    print()

    try:
        # Automatic keyboard layout configuration
        layout = auto_configure_layout()
        print(f"Detected keyboard configuration: {layout.upper()}")

        print()
        print("CONTROLS:")
        print("  Player 1 (Left): Z/Q/S/D (AZERTY) or W/A/S/D (QWERTY)")
        print("  Player 2 (Right): Arrow keys")
        print("  P or SPACE: Pause")
        print("  F1: Help")
        print("  F2: Show FPS")
        print("  ESC: Main menu/Quit")
        print()
        print("Starting game...")
        print()

        main()
    except Exception as e:
        print(f"Error: {e}")
        import traceback

        traceback.print_exc()
    finally:
        sys.exit(1)
