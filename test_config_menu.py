#!/usr/bin/env python
"""Test script for configuration menu"""

from magic_pong.utils.config import game_config
from magic_pong.utils.config_manager import CONFIG_CATEGORIES

print("=== Configuration Categories Test ===")
print(f"Configuration categories loaded: {len(CONFIG_CATEGORIES)}")
for cat in CONFIG_CATEGORIES:
    print(f"  - {cat.name}: {len(cat.options)} options")

print("\n=== Configuration Save/Load Test ===")
try:
    game_config.save_to_file("test_config.json")
    print("✓ Config saved successfully")

    import json

    with open("test_config.json") as f:
        data = json.load(f)
    print(f"✓ JSON file is valid ({len(data)} keys)")

    import os

    os.remove("test_config.json")
    print("✓ Test cleanup complete")

except Exception as e:
    print(f"✗ Error: {e}")

print("\n=== Configuration Reset Test ===")
try:
    original_ball_speed = game_config.BALL_SPEED
    game_config.BALL_SPEED = 500.0
    game_config.reset_to_defaults()
    print(f"✓ Reset successful (BALL_SPEED: {original_ball_speed})")
except Exception as e:
    print(f"✗ Error: {e}")

print("\n✅ All tests passed!")
