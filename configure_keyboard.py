#!/usr/bin/env python3
"""
Utility to configure Magic Pong keyboard layout
"""

from magic_pong.utils.keyboard_layout import get_preferred_layout
from magic_pong.utils.keyboard_layout import list_available_layouts
from magic_pong.utils.keyboard_layout import set_preferred_layout
from magic_pong.utils.keyboard_layout import show_layout_help


def main():
    """Interface to configure keyboard layout"""
    print("=== MAGIC PONG KEYBOARD CONFIGURATION ===")
    print()

    current_layout = get_preferred_layout()
    layouts = list_available_layouts()

    print(f"Current layout: {layouts[current_layout]}")
    print()
    print("Available layouts:")
    for key, name in layouts.items():
        marker = " (current)" if key == current_layout else ""
        print(f"  {key}: {name}{marker}")

    print()
    print("Commands:")
    print("  help - Show key help")
    print("  set <layout> - Change layout (e.g. 'set azerty')")
    print("  quit - Exit")
    print()

    while True:
        try:
            command = input("magic_pong_config> ").strip().lower()

            if command == "quit" or command == "q":
                break

            elif command == "help" or command == "h":
                print()
                print(show_layout_help())

            elif command.startswith("set "):
                layout = command[4:].strip()
                if layout in layouts:
                    if set_preferred_layout(layout):
                        print(f"Layout changed to: {layouts[layout]}")
                        current_layout = layout
                    else:
                        print("Error changing layout")
                else:
                    print(f"Unknown layout: {layout}")
                    print(f"Available layouts: {', '.join(layouts.keys())}")

            elif command == "list" or command == "l":
                for key, name in layouts.items():
                    marker = " (current)" if key == current_layout else ""
                    print(f"  {key}: {name}{marker}")

            else:
                print("Unknown command. Type 'help' for help.")

        except KeyboardInterrupt:
            print("\nGoodbye!")
            break
        except EOFError:
            break


if __name__ == "__main__":
    main()
