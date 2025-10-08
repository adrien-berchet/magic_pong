# Magic Pong

An advanced Pong game specifically designed for artificial intelligence training, featuring advanced functionality and flexible architecture.

## Features

### Advanced Gameplay
- **Free movement**: Players can move freely within their half of the field (not just vertically)
- **Symmetric bonus system**:
  - Player paddle enlargement
  - Opponent paddle shrinking
  - Additional rotating paddle
- **Realistic physics** with bounces and effects

### AI Interface
- **Framework agnostic architecture**: Compatible with different AI frameworks (PyTorch, TensorFlow, etc.)
- **Headless mode**: Ultra-fast training without graphics display
- **Variable speed**: Acceleration up to 1000x for training
- **Configurable reward system**
- **Normalized observations** for learning

### Included AI Examples
- **RandomAI**: Random movements
- **FollowBallAI**: Follows the ball
- **DefensiveAI**: Defensive strategy
- **AggressiveAI**: Seeks bonuses and attacks
- **PredictiveAI**: Predicts ball trajectory

## Installation

```bash
# Clone the project
git clone <repository_url>
cd magic_pong

# Installation for usage
pip install -e .

# Installation for development (recommended)
pip install -e ".[dev]"

# Complete installation (all dependencies)
pip install -e ".[all]"
```

### Installation with Make

```bash
# Complete setup for development
make dev-setup
```

## Quick Usage

### AI vs AI Training

```python
from src.core.game_engine import TrainingManager
from src.ai.models.simple_ai import create_ai

# Create training manager
trainer = TrainingManager(headless=True)

# Create AIs
player1 = create_ai('aggressive', 1)
player2 = create_ai('defensive', 2)

# Train one episode
stats = trainer.train_episode(player1, player2)
print(f"Winner: Player {stats['winner']}")
```

### AI Tournament

```bash
cd magic_pong
python examples/ai_vs_ai.py --mode tournament
```

### Simple Training

```bash
cd magic_pong
python examples/ai_vs_ai.py --mode training
```

## Architecture

```
magic_pong/
├── src/
│   ├── core/           # Game engine and physics
│   ├── ai/             # AI interface and examples
│   ├── graphics/       # Graphics rendering (coming soon)
│   └── utils/          # Configuration and utilities
├── examples/           # Usage examples
└── docs/              # Documentation
```

### Main Components

- **PhysicsEngine**: Manages game physics, collisions, bonuses
- **GameEngine**: Orchestrates the game and manages players
- **TrainingManager**: Optimized for AI training
- **Player**: Abstract interface for players
- **GameEnvironment**: Environment compatible with RL frameworks

## Configuration

The game is highly configurable via [`src/utils/config.py`](src/utils/config.py):

```python
from src.utils.config import game_config, ai_config

# Game configuration
game_config.FIELD_WIDTH = 800
game_config.FIELD_HEIGHT = 600
game_config.BALL_SPEED = 300.0

# AI configuration
ai_config.HEADLESS_MODE = True
ai_config.FAST_MODE_MULTIPLIER = 10.0
```

## Create a Custom AI

```python
from src.ai.interface import Player
from src.core.entities import Action

class MyAI(Player):
    def get_action(self, observation):
        # Your logic here
        ball_pos = observation['ball_pos']
        player_pos = observation['player_pos']

        # Calculate action
        move_x = ball_pos[0] - player_pos[0]
        move_y = ball_pos[1] - player_pos[1]

        return Action(move_x, move_y)

    def on_step(self, observation, action, reward, done, info):
        # Learning here
        self.current_episode_reward += reward
```

## PyTorch Interface

```python
import torch
import torch.nn as nn
from src.ai.interface import Player

class PyTorchAI(Player):
    def __init__(self, player_id, model):
        super().__init__(player_id)
        self.model = model

    def get_action(self, observation):
        # Convert observation to tensor
        state = self._obs_to_tensor(observation)

        # Model prediction
        with torch.no_grad():
            action_probs = self.model(state)

        # Convert to Action
        return Action(
            move_x=action_probs[0].item(),
            move_y=action_probs[1].item()
        )
```

## AI Observations

The observation provided to each AI contains:

```python
observation = {
    'ball_pos': [x, y],                    # Ball position
    'ball_vel': [vx, vy],                  # Ball velocity
    'player_pos': [x, y],                  # Player position
    'opponent_pos': [x, y],                # Opponent position
    'player_paddle_size': float,           # Paddle size
    'opponent_paddle_size': float,         # Opponent paddle size
    'bonuses': [[x, y, type], ...],        # Active bonuses
    'rotating_paddles': [[x, y, angle]], # Rotating paddles
    'score_diff': int,                     # Score difference
    'time_elapsed': float                  # Elapsed time
}
```

## Reward System

- **+1.0**: Score a point
- **-1.0**: Concede a point
- **+0.1**: Collect a bonus
- **+0.01**: Touch the ball
- **+0.02**: Use a rotating paddle

## Performance

In headless mode with acceleration:
- **Normal speed**: ~60 FPS
- **Fast mode**: ~600-6000 FPS (10-100x faster)
- **Training**: Thousands of episodes per minute

## Example Results

Tournament between included AIs (20 games each):

```
Rankings:
1. aggressive: 52 victories
2. predictive: 48 victories
3. defensive: 31 victories
4. follow_ball: 28 victories
5. random: 1 victory
```

## Development

### Development Tools

This project uses Python development best practices with modern tools:

#### Tests
```bash
# Tests with pytest
make test

# Tests with code coverage
make test-cov

# Tests on all Python versions with tox
make test-all
tox
```

#### Code Quality
```bash
# Linting check (ruff)
make lint

# Automatic fixing of linting issues
make lint-fix

# Code formatting (black)
make format

# Format checking
make format-check

# Type checking (mypy)
make type-check

# All quality checks
make quality

# Automatic fixing of all issues
make quality-fix
```

#### Tox - Multi-version Testing
```bash
# Tests on Python 3.8
tox -e py38

# Tests on Python 3.9
tox -e py39

# Tests on Python 3.10
tox -e py310

# Tests on Python 3.11
tox -e py311

# Tests on Python 3.12
tox -e py312

# Linting with tox
tox -e lint

# Formatting with tox
tox -e format

# Type checking with tox
tox -e type-check
```

#### Pre-commit Hooks
```bash
# Install pre-commit hooks
make pre-commit-install

# Run pre-commit on all files
make pre-commit-run

# Update hooks
make pre-commit-update
```

#### Cleanup
```bash
# Clean temporary files
make clean
```

### Tool Configuration

Tools are configured in [`pyproject.toml`](pyproject.toml):

- **Black**: Automatic code formatting (100 character line)
- **Ruff**: Fast and modern linting (replaces flake8, isort, etc.)
- **MyPy**: Static type checking
- **Pytest**: Testing framework with code coverage
- **Tox**: Testing across multiple Python versions
- **Pre-commit**: Validation hooks before commit
- **GitHub Actions**: Automated continuous integration

### Code Structure

- **Clear separation** between business logic and display
- **Modular and extensible** architecture
- **Complete type hints** for better maintenance
- **Unit tests** with pytest and tox
- **Ready continuous integration** with configured tools

### Adding New Bonuses

```python
# In entities.py
class BonusType(Enum):
    MY_BONUS = "my_bonus"

# In physics.py
def _apply_bonus_effect(self, bonus_type, player):
    if bonus_type == BonusType.MY_BONUS:
        # Your effect here
        pass
```

## Roadmap

- [ ] Pygame graphical interface
- [ ] Network multiplayer mode
- [ ] Gymnasium integration
- [ ] Model save/load
- [ ] Advanced metrics and visualizations
- [ ] GPU support for training

## Contributing

Contributions are welcome! See [`CONTRIBUTING.md`](CONTRIBUTING.md) for guidelines.

## License

MIT License - see [`LICENSE`](LICENSE) for details.

## Author

Adrien Berchet - Magic Pong Project for AI Training

---

**Magic Pong** - Where AI learns to play! 🏓🤖
