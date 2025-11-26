# CLAUDE.md - Magic Pong AI Assistant Guide

This document provides comprehensive guidance for AI assistants (like Claude) working on the Magic Pong codebase.

## Project Overview

**Magic Pong** is an advanced Pong game specifically designed for artificial intelligence training and experimentation. It features:

- **Free 2D movement**: Players move in both X and Y axes within their half of the field
- **Advanced bonus system**: Paddle size changes, rotating paddles, and strategic gameplay elements
- **AI-first design**: Headless mode, fast training (up to 1000x speed), framework-agnostic interface
- **Modern Python architecture**: Type hints, modular design, comprehensive testing

**Tech Stack**: Python 3.10+, pygame, numpy, pydantic, PyTorch (optional), Gymnasium (optional)

## Repository Structure

```
magic_pong/
├── magic_pong/                   # Main source package
│   ├── core/                     # Core game engine
│   │   ├── entities.py          # Game entities (Ball, Paddle, Bonus, etc.)
│   │   ├── physics.py           # Physics engine and collision detection
│   │   ├── game_engine.py       # Main game orchestration
│   │   ├── collision.py         # Advanced collision algorithms
│   │   └── interfaces/          # Protocol definitions
│   │       ├── player.py        # PlayerProtocol for polymorphic players
│   │       ├── physics.py       # PhysicsEngine protocol
│   │       └── renderer.py      # Renderer protocol
│   ├── ai/                       # AI interface and implementations
│   │   ├── interface.py         # Base AIPlayer class (backward compat)
│   │   ├── pretraining.py       # Pretraining utilities
│   │   ├── interfaces/          # Protocol-based interfaces
│   │   │   ├── observation.py   # ObservationBuilder protocol
│   │   │   └── reward.py        # RewardCalculator protocol
│   │   ├── environment/         # Environment creation and management
│   │   │   └── factory.py       # EnvironmentFactory for easy setup
│   │   └── models/              # AI implementations (formerly examples/)
│   │       ├── simple_ai.py     # Rule-based AIs (Random, Defensive, etc.)
│   │       ├── dqn_ai.py        # Deep Q-Network implementation
│   │       ├── ai_vs_ai.py      # AI tournament and training scripts
│   │       └── train_dqn.py     # DQN training script
│   ├── gui/                      # Graphical interface
│   │   ├── game_app.py          # Main GUI application
│   │   ├── pygame_renderer.py   # Pygame rendering logic
│   │   └── human_player.py      # Human player controller
│   └── utils/                    # Utilities and configuration
│       ├── config.py            # Game and AI configuration
│       └── keyboard_layout.py   # Keyboard layout support (QWERTY/AZERTY/QWERTZ)
├── tests/                        # Unit tests
│   └── test_entities.py
├── examples/                     # Usage examples
│   ├── game_modes_demo.py
│   └── pygame_gui_example.py
├── train_optimized.py           # Optimized training script
├── train_pretrained_dqn.py      # Pretrained DQN training
├── play_pong.py                 # Play against AI
├── configure_keyboard.py        # Keyboard layout configuration
├── ARCHITECTURE_ANALYSIS.md     # Architecture documentation
├── AI_REFACTOR_PLAN.md          # AI module refactoring plan
├── USAGE_EXAMPLES.md            # Usage examples documentation
├── pyproject.toml               # Project metadata and tool config
├── requirements.txt             # Python dependencies
├── Makefile                     # Development commands
├── tox.ini                      # Multi-version testing config
├── .pre-commit-config.yaml      # Pre-commit hooks
├── .codespellignorelines        # Codespell ignore list
└── .github/workflows/ci.yml     # CI/CD pipeline

```

## Key Architecture Concepts

### 1. Core Game Components

**Entities** (`magic_pong/core/entities.py`):
- `Ball`: Game ball with position, velocity, collision detection
- `Paddle`: Player paddle with movement constraints and size effects
- `RotatingPaddle`: Bonus rotating paddle with angle and collision segments
- `Bonus`: Collectible bonuses (enlarge paddle, shrink opponent, rotating paddle)
- `Vector2D`: Simple 2D vector math
- `Action`: Player action with normalized movement (-1 to 1)
- `GameState`: Complete game state snapshot for AI

**Physics Engine** (`magic_pong/core/physics.py`):
- Handles all game physics, collisions, and bonus management
- Separates concerns between physics simulation and rendering
- Manages events (goals, hits, bonus collection) for AI reward calculation

**Game Engine** (`magic_pong/core/game_engine.py`):
- Orchestrates game flow and player interactions
- `TrainingManager`: Optimized headless training mode
- Supports variable speed for fast AI training

**Core Interfaces** (`magic_pong/core/interfaces/`):
- `PlayerProtocol`: Enables polymorphic player handling (human, AI, random)
- `PhysicsEngine`: Protocol for physics engine implementations
- `Renderer`: Protocol for different rendering backends

### 2. AI Interface (Refactored Architecture)

**Legacy Interface** (`magic_pong/ai/interface.py`):
- Maintained for backward compatibility
- Re-exports new protocol-based components

**Protocol-Based Interfaces** (`magic_pong/ai/interfaces/`):

```python
# Observation Builder Protocol
class ObservationBuilder(Protocol):
    def build_observation(self, game_state: dict, player_id: int) -> np.ndarray:
        """Build observation array from game state"""
        ...

    @property
    def observation_size(self) -> int:
        """Get observation dimension"""
        ...

# Reward Calculator Protocol
class RewardCalculator(Protocol):
    def calculate_reward(self, game_state: dict, events: dict, player_id: int) -> float:
        """Calculate reward for a player"""
        ...
```

**Implementations**:
- `VectorObservationBuilder`: Flat vector observations (positions, velocities)
- `SparseRewardCalculator`: Only goal-based rewards (+1/-1)
- `DenseRewardCalculator`: Includes hit rewards, bonus collection, proximity

**Environment Factory** (`magic_pong/ai/environment/factory.py`):
- Easy environment creation with custom components
- Experiment with different reward functions and observation spaces
- Sensible defaults for quick prototyping

```python
from magic_pong.ai.environment import EnvironmentFactory

# Simple creation with defaults
env = EnvironmentFactory.create_default(physics_engine)

# Custom reward function
env = EnvironmentFactory.create(
    physics_engine,
    reward_calculator=DenseRewardCalculator(hit_reward=0.2)
)
```

**Observation Structure**:
```python
{
    'ball_pos': [x, y],              # Normalized or absolute
    'ball_vel': [vx, vy],            # Ball velocity
    'player_pos': [x, y],            # Player position
    'opponent_pos': [x, y],          # Opponent position
    'player_paddle_size': float,     # Current paddle size
    'opponent_paddle_size': float,   # Opponent paddle size
    'bonuses': [[x, y, type], ...],  # Active bonuses
    'rotating_paddles': [[x, y, angle], ...],
    'score_diff': int,               # Score difference
    'time_elapsed': float            # Time since start
}
```

**Reward System**:
- `+1.0`: Score a point
- `-1.0`: Concede a point
- `+0.1`: Collect a bonus or hit the ball
- `+0.02`: Hit with rotating paddle
- Optional proximity rewards for approaching optimal interception points

### 3. Configuration System

All configuration is centralized in `magic_pong/utils/config.py`:

```python
from magic_pong.utils.config import game_config, ai_config

# Modify game settings
game_config.FIELD_WIDTH = 800
game_config.BALL_SPEED = 300.0
game_config.BONUSES_ENABLED = True

# Modify AI settings
ai_config.HEADLESS_MODE = True
ai_config.FAST_MODE_MULTIPLIER = 10.0
ai_config.USE_PROXIMITY_REWARD = False
```

## Development Workflows

### Setup and Installation

```bash
# Clone repository
git clone <repository_url>
cd magic_pong

# Development setup (recommended)
make dev-setup
# OR manually:
pip install -e ".[dev]"

# Install all dependencies including AI frameworks
pip install -e ".[all]"

# Install pre-commit hooks
make pre-commit-install
```

### Code Quality Standards

**This project enforces strict code quality**:

1. **Type Hints**: All functions must have complete type annotations
2. **Line Length**: Maximum 100 characters (Black format)
3. **Import Order**: Managed by Ruff (stdlib, third-party, local)
4. **Docstrings**: Required for public APIs (Google/NumPy style)

**Quality Tools**:
- **Ruff**: Fast linting and formatting (replaces flake8, isort, pyupgrade, black)
- **MyPy**: Static type checking with strict settings
- **Pytest**: Testing framework with coverage tracking
- **Codespell**: Spell checking for code and documentation
- **Pre-commit**: Automated quality checks before commits

### Common Development Commands

```bash
# Run tests
make test                  # Basic tests
make test-cov             # With coverage report
make test-all             # All Python versions (tox)

# Code quality
make lint                 # Check linting issues
make lint-fix             # Auto-fix linting issues
make format               # Format code with Black
make type-check           # MyPy type checking
make quality              # All checks (lint + format + types)
make quality-fix          # Auto-fix all fixable issues

# Pre-commit hooks
make pre-commit-run       # Run all pre-commit hooks
make pre-commit-update    # Update hook versions

# Cleanup
make clean                # Remove build artifacts and cache

# Run examples
make run-tournament       # AI tournament
python play_pong.py       # Play against AI
python train_optimized.py # Train DQN agent
```

### Testing Guidelines

**Location**: `tests/` directory

**Run tests**:
```bash
pytest tests/                                    # All tests
pytest tests/test_entities.py                   # Specific file
pytest tests/ -v --cov=magic_pong               # With coverage
pytest tests/ -m "not slow"                     # Skip slow tests
```

**Writing tests**:
- Use descriptive test names: `test_ball_bounces_on_wall_collision`
- Test one concept per test function
- Use fixtures for common setup
- Mark slow tests with `@pytest.mark.slow`

### Git Workflow

**Branches**:
- `main`: Stable releases
- `develop`: Active development
- Feature branches: `feature/your-feature-name`
- Claude branches: Auto-generated as `claude/claude-md-*`

**Commits**:
- Use clear, descriptive commit messages
- Format: `<type>: <description>` (e.g., `Feat: Add proximity reward`, `Fix: Ball collision bug`)
- Types: Feat, Fix, Docs, Test, Refactor, Style, Chore

**Pre-commit Hooks**:
- Automatically run on commit
- Checks: trailing whitespace, YAML syntax, Ruff linting, MyPy types, Codespell
- Fix issues before committing or use `git commit --no-verify` (discouraged)

### CI/CD Pipeline

**GitHub Actions** (`.github/workflows/ci.yml`):
- Runs on push to `main` or `develop` branches and all PRs
- Jobs:
  - **test**: Runs tests on Python 3.10, 3.11, 3.12
  - **lint**: Checks code formatting and linting
  - **type-check**: Validates type annotations
  - **coverage**: Generates coverage report (uploads to Codecov)

**All checks must pass** before merging PRs.

## AI Assistant Guidelines

### When Working on This Codebase

1. **Always read before modifying**: Never propose changes to files you haven't read
2. **Respect the architecture**: Maintain separation between core, ai, gui, and utils
3. **Type everything**: Add type hints to all new functions and classes
4. **Test your changes**: Add or update tests for new functionality
5. **Follow conventions**: Use existing code style and patterns
6. **Update documentation**: Modify README.md or this file if adding major features

### Code Style Requirements

```python
# GOOD: Type hints, clear names, proper formatting
def calculate_distance(pos1: Vector2D, pos2: Vector2D) -> float:
    """Calculate Euclidean distance between two positions."""
    dx = pos2.x - pos1.x
    dy = pos2.y - pos1.y
    return math.sqrt(dx * dx + dy * dy)

# BAD: No types, unclear, poor formatting
def calc(p1,p2):
    return math.sqrt((p2.x-p1.x)**2+(p2.y-p1.y)**2)
```

### Running Quality Checks Before Committing

**Always run before proposing changes**:
```bash
make quality-fix    # Auto-fix what can be fixed
make quality        # Verify all checks pass
make test          # Ensure tests pass
```

### Common Patterns

**Adding a new AI**:
1. Create class inheriting from `AIPlayer` in `magic_pong/ai/models/`
2. Implement `get_action()` and `on_step()` methods
3. Add example usage in `magic_pong/ai/models/ai_vs_ai.py`
4. Document in README.md

**Using the new protocol-based interfaces**:
```python
from magic_pong.ai.interfaces import ObservationBuilder, RewardCalculator
from magic_pong.ai.environment import EnvironmentFactory

# Create custom observation builder
class CustomObservationBuilder(ObservationBuilder):
    def build_observation(self, game_state, player_id):
        # Your custom observation logic
        return observation_array

    @property
    def observation_size(self) -> int:
        return 42  # Your observation dimension

# Use it with the factory
env = EnvironmentFactory.create(
    physics_engine,
    observation_builder=CustomObservationBuilder()
)
```

**Adding a new bonus type**:
1. Add enum value to `BonusType` in `magic_pong/core/entities.py`
2. Implement effect in `PhysicsEngine._apply_bonus_effect()` in `magic_pong/core/physics.py`
3. Add color to `game_config.BONUS_COLORS` in `magic_pong/utils/config.py`
4. Update rendering in `magic_pong/gui/pygame_renderer.py`

**Adding configuration options**:
1. Add field to `GameConfig` or `AIConfig` in `magic_pong/utils/config.py`
2. Use the config value in relevant code
3. Document in README.md with example usage

### Package Import Convention

**Always use absolute imports** from `magic_pong` package:
```python
# GOOD
from magic_pong.core.entities import Ball, Paddle, Action
from magic_pong.utils.config import game_config, ai_config

# BAD
from entities import Ball  # Relative imports
from ..utils.config import game_config
```

### File Organization

**Keep files focused**:
- `entities.py`: Only entity classes (Ball, Paddle, etc.)
- `physics.py`: Physics simulation and collision detection
- `game_engine.py`: Game loop and orchestration
- `interface.py`: AI interface classes

**When to create new files**:
- File exceeds ~500 lines
- New major feature with multiple related classes
- Clear separation of concerns (e.g., new rendering backend)

### Debugging and Development

**Debug configurations**:
```python
# Enable debug output for optimal points
ai_config.DEBUG_OPTIMAL_POINTS = True

# Show optimal points in GUI
ai_config.SHOW_OPTIMAL_POINTS_GUI = True

# Slow down for debugging
ai_config.FAST_MODE_MULTIPLIER = 1.0
```

**Common debugging patterns**:
```python
# Add debug prints with context
print(f"[DEBUG] Ball position: {ball.position.to_tuple()}, velocity: {ball.velocity.to_tuple()}")

# Use events for tracking
events = physics_engine.update(dt, action1, action2)
print(f"Events this step: {events}")
```

### Performance Considerations

1. **Headless training is critical**: Use `headless=True` for AI training
2. **Fast mode multiplier**: Set `ai_config.FAST_MODE_MULTIPLIER = 10.0` or higher
3. **Avoid rendering in training loops**: Don't call `render()` in headless mode
4. **Batch operations**: Process multiple episodes before checking progress

### Testing New Features

**Minimal test example**:
```python
def test_new_bonus_effect():
    """Test that new bonus applies correct effect."""
    paddle = Paddle(100, 200, player_id=1)
    original_size = paddle.height

    # Apply bonus effect
    paddle.apply_size_effect(multiplier=2.0, duration=5.0)

    assert paddle.height == original_size * 2.0
    assert paddle.size_effect_timer == 5.0
```

### When to Ask for Clarification

1. **Ambiguous requirements**: "Should proximity rewards apply to defensive positioning?"
2. **Architecture decisions**: "Should we add this to PhysicsEngine or GameEngine?"
3. **Breaking changes**: "This would change the observation format - is that acceptable?"
4. **Performance trade-offs**: "More accurate collision detection would slow training by 20%"

## Key Technical Details

### Coordinate System

- Origin (0, 0) is **top-left corner**
- X increases to the right
- Y increases downward
- Player 1 occupies left half (x: 0 to 400)
- Player 2 occupies right half (x: 400 to 800)

### Collision Detection

**Types implemented**:
- AABB (Axis-Aligned Bounding Box) for paddles and bonuses
- Circle-rectangle collision for ball-paddle
- Circle-line segment collision for rotating paddles
- Continuous collision detection to prevent tunneling

### Physics Simulation

- Fixed timestep: 1/60 second (60 FPS)
- Fast mode multiplies dt for accelerated training
- Velocity-based movement with constraints
- Bounce mechanics preserve momentum with optional acceleration

### Module Dependencies

**Dependency graph**:
```
core/interfaces/ (protocol definitions, no dependencies)
    ↓
entities.py (no dependencies except config)
    ↓
physics.py (uses entities, implements PhysicsEngine protocol)
    ↓
game_engine.py (uses physics, PlayerProtocol)
    ↓
ai/interfaces/ (observation & reward protocols)
    ↓
ai/environment/ (uses protocols, provides EnvironmentFactory)
    ↓
ai/models/ (AI implementations, use environment & interfaces)
```

**Key Design Principles**:
- **Protocol-based**: Use protocols for dependency inversion
- **No circular dependencies**: Keep strict module hierarchy
- **Separation of concerns**: Core, AI, and GUI are independent
- **Backward compatibility**: Legacy interfaces re-export new components

## Troubleshooting Common Issues

### Import Errors

```bash
# If imports fail, reinstall in editable mode
pip install -e .

# Check package structure
python -c "import magic_pong; print(magic_pong.__file__)"
```

### Type Checking Errors

```bash
# Run MyPy on specific file
mypy magic_pong/core/entities.py

# Common fixes:
# - Add type: ignore comments for third-party libraries
# - Use Optional[Type] for nullable values
# - Use Union[Type1, Type2] for multiple types
```

### Pre-commit Hook Failures

```bash
# Run hooks manually to see details
pre-commit run --all-files

# Update hooks if outdated
pre-commit autoupdate

# Skip hooks in emergency (not recommended)
git commit --no-verify
```

### Test Failures

```bash
# Run with verbose output
pytest tests/ -vv

# Run specific test with debugging
pytest tests/test_entities.py::test_ball_reset -vv -s

# Check coverage gaps
pytest tests/ --cov=magic_pong --cov-report=html
open htmlcov/index.html
```

## Working with Protocol-Based Architecture

The codebase has been refactored to use Python protocols for better extensibility and testability.

### Key Protocols

**PlayerProtocol** (`magic_pong/core/interfaces/player.py`):
- Enables polymorphic player handling
- All players (human, AI, random) implement the same interface
- Game engine doesn't need to know player type

**ObservationBuilder** (`magic_pong/ai/interfaces/observation.py`):
- Customize how game state is converted to observations
- Easy to experiment with different observation spaces
- Examples: vector-based, image-based, history-based

**RewardCalculator** (`magic_pong/ai/interfaces/reward.py`):
- Customize reward functions for training
- Examples: sparse (goals only), dense (hits + bonuses), shaped (proximity)
- Protocol-based design allows easy A/B testing

### Using the EnvironmentFactory

The factory pattern simplifies environment creation:

```python
from magic_pong.ai.environment import EnvironmentFactory
from magic_pong.ai.interfaces import DenseRewardCalculator

# Quick start with defaults
env = EnvironmentFactory.create_default(physics_engine)

# Custom configuration
env = EnvironmentFactory.create(
    physics=physics_engine,
    headless=True,
    player_id=1,
    reward_calculator=DenseRewardCalculator(
        goal_reward=1.0,
        hit_reward=0.2,
        bonus_reward=0.3
    )
)
```

### Backward Compatibility

The refactoring maintains backward compatibility:
- `magic_pong/ai/interface.py` still exists and re-exports components
- Existing code using `AIPlayer`, `ObservationProcessor`, etc. continues to work
- New code should prefer protocol-based interfaces for better extensibility

## Resources and References

- **README.md**: User-facing documentation and quickstart
- **GUI_README.md**: GUI-specific documentation
- **ARCHITECTURE_ANALYSIS.md**: Detailed architecture analysis and design decisions
- **AI_REFACTOR_PLAN.md**: AI module refactoring plan and protocol-based design
- **USAGE_EXAMPLES.md**: Comprehensive usage examples
- **pyproject.toml**: All tool configurations
- **examples/**: Working code examples (game modes, pygame GUI)
- **magic_pong/ai/models/**: AI implementation patterns
- **magic_pong/ai/interfaces/**: Protocol definitions for extensibility

## Contributing Checklist

Before submitting changes:

- [ ] Code is properly type-annotated
- [ ] All tests pass (`make test`)
- [ ] Code is formatted (`make format`)
- [ ] Linting passes (`make lint`)
- [ ] Type checking passes (`make type-check`)
- [ ] New features have tests
- [ ] Documentation is updated (README.md, docstrings)
- [ ] Commit messages are descriptive
- [ ] No debug print statements left in code
- [ ] Pre-commit hooks are installed and passing

## Version Information

- **Python**: 3.10, 3.11, 3.12 (3.10+ required)
- **License**: MIT
- **Package Name**: magic-pong
- **Current Version**: 0.1.0 (Alpha)
- **Author**: Adrien Berchet
- **Key Dependencies**:
  - pygame >= 2.5.0 (game rendering)
  - numpy >= 1.24.0 (numerical operations)
  - pydantic >= 2.0.0 (data validation)
  - torch >= 2.0.0 (optional, for neural networks)
  - gymnasium >= 0.29.0 (optional, for RL environments)

---

**Last Updated**: 2025-11-26 (Post-refactoring to protocol-based architecture)

This document should be updated when:
- Major architectural changes occur
- New patterns or conventions are established
- Development workflow changes
- New tools are added to the pipeline
