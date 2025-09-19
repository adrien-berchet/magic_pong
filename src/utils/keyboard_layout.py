"""
Keyboard layout detection and management for Magic Pong
"""

import json
import locale
import os
from pathlib import Path

from magic_pong.utils.config import KEYBOARD_LAYOUTS, game_config


def detect_system_layout() -> str:
    """
    Detect the most likely keyboard layout based on system locale

    Returns:
        Keyboard layout name (default to 'qwerty' if detection fails)
    """
    try:
        # Try to get system locale
        system_locale = locale.getdefaultlocale()[0]
        if system_locale:
            system_locale = system_locale.lower()

            # Map common locales to keyboard layouts
            if system_locale.startswith("fr"):
                return "azerty"
            elif system_locale.startswith("de"):
                return "qwertz"
            else:
                return "qwerty"

    except Exception:
        pass

    # Fallback to environment variables
    lang = os.environ.get("LANG", "").lower()
    if "fr" in lang:
        return "azerty"
    elif "de" in lang:
        return "qwertz"

    return "qwerty"


def get_config_file_path() -> Path:
    """Get the path to the user configuration file"""
    home = Path.home()
    config_dir = home / ".config" / "magic_pong"
    config_dir.mkdir(parents=True, exist_ok=True)
    return config_dir / "user_config.json"


def load_user_preferences() -> dict:
    """Load user preferences from config file"""
    config_file = get_config_file_path()

    if config_file.exists():
        try:
            with open(config_file, encoding="utf-8") as f:
                return json.load(f)
        except (OSError, json.JSONDecodeError):
            pass

    return {}


def save_user_preferences(preferences: dict) -> None:
    """Save user preferences to config file"""
    config_file = get_config_file_path()

    try:
        with open(config_file, "w", encoding="utf-8") as f:
            json.dump(preferences, f, indent=2, ensure_ascii=False)
    except OSError:
        pass  # Silently fail if we can't write config


def get_preferred_layout() -> str:
    """
    Get the user's preferred keyboard layout

    Priority:
    1. User saved preference
    2. System detection
    3. Default configuration
    """
    # Try user preference first
    user_prefs = load_user_preferences()
    if "keyboard_layout" in user_prefs:
        layout = user_prefs["keyboard_layout"]
        if layout in KEYBOARD_LAYOUTS:
            return layout

    # Try system detection
    detected = detect_system_layout()
    if detected in KEYBOARD_LAYOUTS:
        return detected

    # Fallback to config default
    return game_config.KEYBOARD_LAYOUT


def set_preferred_layout(layout: str) -> bool:
    """
    Set the user's preferred keyboard layout

    Args:
        layout: Layout name (must be in KEYBOARD_LAYOUTS)

    Returns:
        True if successful, False otherwise
    """
    if layout not in KEYBOARD_LAYOUTS:
        return False

    user_prefs = load_user_preferences()
    user_prefs["keyboard_layout"] = layout
    save_user_preferences(user_prefs)

    # Update the game config
    game_config.KEYBOARD_LAYOUT = layout

    return True


def list_available_layouts() -> dict:
    """Get all available keyboard layouts"""
    return {name: layout.name for name, layout in KEYBOARD_LAYOUTS.items()}


def auto_configure_layout() -> str:
    """
    Automatically configure the best keyboard layout

    Returns:
        The selected layout name
    """
    preferred = get_preferred_layout()
    game_config.KEYBOARD_LAYOUT = preferred
    return preferred


def show_layout_help() -> str:
    """
    Generate help text showing current key mappings

    Returns:
        Formatted help text
    """
    layout = game_config.get_keyboard_layout()

    help_text = f"Configuration clavier actuelle : {layout.name}\n\n"
    help_text += "Contrôles WASD :\n"
    for action, key_name in layout.display_names.items():
        help_text += f"  {action}: {key_name}\n"

    help_text += "\nContrôles flèches :\n"
    help_text += "  haut: ↑\n"
    help_text += "  bas: ↓\n"
    help_text += "  gauche: ←\n"
    help_text += "  droite: →\n"

    help_text += f"\nLayouts disponibles : {', '.join(list_available_layouts().values())}\n"

    return help_text
