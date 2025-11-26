"""
AI Environment package - Modular RL training environment

This package contains the components for building RL training environments:
- observation: Observation builders for different state representations
- rewards: Reward calculators for different training strategies
- gym_wrapper: Gymnasium-compatible environment wrapper (legacy, see interface.py)
- factory: Easy environment creation with sensible defaults

NEW in Phase 2: Protocol-based environment creation!
"""

from magic_pong.ai.environment.factory import EnvironmentFactory, GameEnvironmentWrapper

__all__ = ["EnvironmentFactory", "GameEnvironmentWrapper"]
