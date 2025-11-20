# Magic Pong - Usage Examples

## Creating Training Environments (New Protocol-Based API)

### Quick Start - Default Configuration

```python
from magic_pong.core.physics import PhysicsEngine
from magic_pong.ai.environment import EnvironmentFactory
from magic_pong.utils.config import game_config

# Create physics engine
physics = PhysicsEngine(game_config.FIELD_WIDTH, game_config.FIELD_HEIGHT)

# Create environment with sensible defaults
env = EnvironmentFactory.create_default(physics, headless=True)

# Train your agent!
obs = env.reset()
for step in range(1000):
    action = your_agent.get_action(obs)  # Your RL agent
    obs, reward, done, info = env.step(action)
    if done:
        obs = env.reset()
```

### Sparse Rewards (Only Goals)

Good for testing if your agent can learn from minimal signal:

```python
# Only rewards +1 for scoring, -1 for opponent scoring
env = EnvironmentFactory.create_sparse(physics, headless=True)
```

### Custom Dense Rewards

Experiment with different reward shaping:

```python
from magic_pong.ai.interfaces import DenseRewardCalculator

env = EnvironmentFactory.create(
    physics,
    reward_calculator=DenseRewardCalculator(
        goal_reward=10.0,      # Big reward for goals
        lose_penalty=-10.0,    # Big penalty for conceding
        hit_reward=0.5,        # Reward paddle hits
        wall_reward=0.1,       # Small reward for wall bounces
        bonus_reward=0.2,      # Reward for collecting bonuses
    ),
    headless=True
)
```

### Temporal Observations (Frame Stacking)

Include history to help agent understand velocity and trajectory:

```python
from magic_pong.ai.interfaces import HistoryObservationBuilder

env = EnvironmentFactory.create(
    physics,
    observation_builder=HistoryObservationBuilder(
        history_length=3,  # Stack last 3 frames
        field_width=game_config.FIELD_WIDTH,
        field_height=game_config.FIELD_HEIGHT
    ),
    headless=True
)

# Observation is now 3x larger (3 frames stacked)
print(env.observation_space_size)  # 18 (6 features × 3 frames)
```

### Custom Reward Function

Create your own reward strategy:

```python
from magic_pong.ai.interfaces import RewardCalculator

class MyCustomRewardCalculator:
    """Reward based on ball proximity to opponent"""

    def calculate_reward(self, events, game_state, player_id):
        reward = 0.0

        # Big rewards for goals
        for goal_event in events.get("goals", []):
            if goal_event["player"] == player_id:
                reward += 5.0
            else:
                reward -= 5.0

        # Reward for keeping ball on opponent's side
        ball_x = game_state["ball_position"][0]
        field_width = game_state["field_bounds"][1]

        if player_id == 1 and ball_x > field_width / 2:
            reward += 0.01  # Ball on right side (opponent's side)
        elif player_id == 2 and ball_x < field_width / 2:
            reward += 0.01  # Ball on left side (opponent's side)

        return reward

    def reset(self):
        pass

# Use your custom reward
env = EnvironmentFactory.create(
    physics,
    reward_calculator=MyCustomRewardCalculator(),
    headless=True
)
```

### Mix and Match

Combine any observation builder with any reward calculator:

```python
from magic_pong.ai.interfaces import (
    HistoryObservationBuilder,
    SparseRewardCalculator
)

# Sparse rewards + history observations
env = EnvironmentFactory.create(
    physics,
    reward_calculator=SparseRewardCalculator(),
    observation_builder=HistoryObservationBuilder(history_length=5),
    headless=True
)
```

## Benefits of the New API

1. **Easy Experimentation** - Try different rewards in 1 line
2. **Clean Code** - Separate concerns (rewards vs observations)
3. **Reusable** - Share reward functions across projects
4. **Testable** - Test rewards and observations independently
5. **Extensible** - Add new strategies without touching environment code

## Backward Compatibility

Old code still works! The original `GameEnvironment` from `ai/interface.py` is unchanged:

```python
# This still works exactly as before
from magic_pong.ai.interface import GameEnvironment

env = GameEnvironment(physics, headless=True)
```

## Comparison: Old vs New

### Old Way (Still Works)
```python
from magic_pong.ai.interface import GameEnvironment

# Monolithic environment with hardcoded reward logic
env = GameEnvironment(physics, headless=True)

# To change rewards: modify GameEnvironment class ❌
# To change observations: modify ObservationProcessor ❌
```

### New Way (Recommended)
```python
from magic_pong.ai.environment import EnvironmentFactory
from magic_pong.ai.interfaces import DenseRewardCalculator

# Modular: inject custom components
env = EnvironmentFactory.create(
    physics,
    reward_calculator=DenseRewardCalculator(hit_reward=0.3)
)

# To change rewards: pass different calculator ✓
# To change observations: pass different builder ✓
```

## Next Steps

1. Try different reward functions to see what works best
2. Experiment with observation spaces (vector vs history vs image)
3. Share your custom reward/observation strategies!
4. Check out `src/ai/interfaces/` for all available implementations
