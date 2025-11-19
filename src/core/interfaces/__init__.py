"""
Core interfaces and protocols for Magic Pong

This module defines abstract interfaces that components must implement,
enabling loose coupling and easier testing/extension.
"""

from magic_pong.core.interfaces.physics import PhysicsBackend
from magic_pong.core.interfaces.player import PlayerProtocol
from magic_pong.core.interfaces.renderer import RendererProtocol

__all__ = ["PhysicsBackend", "PlayerProtocol", "RendererProtocol"]
