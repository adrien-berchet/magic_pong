# Magic Pong Graphical Interface

## Description

This PyGame graphical interface allows you to play Magic Pong with keyboard controls for human players. It offers several game modes:

- **1 vs 1**: Two human players on the same keyboard
- **1 vs AI**: One human player against an AI
- **AI vs AI**: Demonstration with two AIs

## Installation

Make sure you have PyGame installed:

```bash
pip install pygame numpy
```

## Launch

### Method 1: Main script
```bash
python play_pong.py
```

### Method 2: GUI module
```bash
python -m src.gui.game_app
```

### Method 3: After package installation
```bash
magic-pong
# or
magic-pong-gui
```

### Method 4: Specific examples
```bash
python examples/pygame_gui_example.py
python examples/game_modes_demo.py
```

## Controls

### Player 1 (Left paddle)
- **W**: Move up
- **S**: Move down
- **A**: Move left
- **D**: Move right

### Player 2 (Right paddle)
- **↑**: Move up
- **↓**: Move down
- **←**: Move left
- **→**: Move right

### General controls
- **P** or **SPACE**: Pause/Resume
- **F1**: Show/Hide help
- **F2**: Show/Hide FPS
- **F3**: Debug mode
- **ESC**: Return to main menu
- **R**: Restart game (at game end)

## Game modes

### 1 vs 1 (Two players)
Two human players compete on the same keyboard. Player 1 uses WASD, Player 2 uses arrow keys.

### 1 vs AI (Player vs AI)
A human player (left, WASD controls) faces a simple AI that controls the right paddle.

### AI vs AI (Demonstration)
Two AIs compete automatically. Ideal mode for observing AI behavior or for demonstrations.

## Advanced features

- **Bonuses**: Bonuses appear randomly on the field
- **Rotating paddles**: Special bonus that creates rotating paddles
- **Visual effects**: Ball trails, highlight effects
- **Statistics display**: FPS, debug information
- **Pause/Resume**: Ability to pause at any time

## Code structure

- `pygame_renderer.py`: PyGame rendering management
- `human_player.py`: Human player and keyboard input management
- `game_app.py`: Main application and game loop
- `examples/`: Example scripts and demonstrations

## Customization

You can modify the configuration in `magic_pong/utils/config.py`:

- Field dimensions
- Game speeds
- Colors
- Controls and keys

## Troubleshooting

### Import error
- Check that pygame is installed: `pip install pygame`
- Make sure you're in the correct directory
- Verify that the `magic_pong` module is accessible

### Performance
- Use F2 to display FPS
- Reduce resolution if necessary
- Close other resource-intensive applications

### Unresponsive controls
- Make sure the game window has focus
- Check your keyboard configuration
- Restart the game if necessary

## Development

To add new features:

1. Modify `pygame_renderer.py` for display
2. Modify `human_player.py` for controls
3. Modify `game_app.py` for application logic

The code is structured modularly to facilitate extensions.
