# AI Module Refactoring Plan - Phase 2

## Current State

`magic_pong/ai/interface.py` - 646 lines with mixed responsibilities:
- AIPlayer (52 lines) - Abstract AI player base class
- ObservationProcessor (90 lines) - Converts game state to observations
- RewardCalculator (386 lines) - Calculates rewards for training
- GameEnvironment (100 lines) - Gymnasium environment wrapper

## Goals

1. **Better Organization** - Split into focused modules
2. **Backward Compatibility** - Existing code continues to work
3. **New Protocol-Based APIs** - Use interfaces we created in Phase 1
4. **Easy Experimentation** - Factory pattern for quick setup

## New Structure

```
magic_pong/ai/
├── interfaces/              # ✓ Done in Phase 1
│   ├── observation.py       # ObservationBuilder protocol + implementations
│   └── reward.py            # RewardCalculator protocol + implementations
├── environment/             # NEW
│   ├── observation.py       # Move ObservationProcessor here (legacy)
│   ├── rewards.py           # Move RewardCalculator here (legacy)
│   ├── gym_wrapper.py       # Move GameEnvironment here
│   ├── factory.py           # NEW - Easy environment creation
│   └── __init__.py          # Exports
├── interface.py             # Keep for backward compat (re-exports)
└── ...existing files...
```

## Implementation Steps

### Step 1: Create Environment Package ✓
- [x] Create `magic_pong/ai/environment/` directory
- [x] Create `__init__.py`

### Step 2: Move & Reorganize Classes
- [ ] Move ObservationProcessor to `environment/observation.py`
- [ ] Move RewardCalculator to `environment/rewards.py`
- [ ] Move GameEnvironment to `environment/gym_wrapper.py`
- [ ] Keep AIPlayer in `interface.py` (it's small and core)

### Step 3: Create Factory
- [ ] `environment/factory.py` - Easy environment creation
- [ ] Support both old and new APIs
- [ ] Provide sensible defaults

### Step 4: Update interface.py
- [ ] Re-export moved classes for backward compatibility
- [ ] Add deprecation warnings (optional)

### Step 5: Test Everything
- [ ] Run all existing tests
- [ ] Verify backward compatibility
- [ ] Test new factory patterns

## Backward Compatibility Strategy

Old code will continue to work:
```python
# Still works!
from magic_pong.ai.interface import GameEnvironment, RewardCalculator

env = GameEnvironment(physics, headless=True)
```

New code can use better APIs:
```python
# New way - using protocols!
from magic_pong.ai.environment import EnvironmentFactory
from magic_pong.ai.interfaces import DenseRewardCalculator, VectorObservationBuilder

env = EnvironmentFactory.create(
    physics=physics,
    reward_calculator=DenseRewardCalculator(),
    observation_builder=VectorObservationBuilder(),
)
```

## Benefits

1. **Smaller Files** - Each file <200 lines, focused responsibility
2. **Easy to Find** - Clear module names (observation, rewards, gym_wrapper)
3. **Extensible** - Protocols make it easy to add new strategies
4. **Testable** - Can test components in isolation
5. **No Breaking Changes** - Existing code works unchanged
