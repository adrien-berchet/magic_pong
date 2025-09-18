"""
Configuration du jeu Magic Pong
"""

from dataclasses import dataclass


@dataclass
class GameConfig:
    """Configuration principale du jeu"""

    # Dimensions du terrain
    FIELD_WIDTH: int = 800
    FIELD_HEIGHT: int = 600

    # Physique de la balle
    BALL_RADIUS: float = 8.0
    BALL_SPEED: float = 300.0  # pixels par seconde
    BALL_SPEED_INCREASE: float = 1.05  # Facteur d'accélération après chaque rebond

    # Raquettes des joueurs
    PADDLE_WIDTH: float = 15.0
    PADDLE_HEIGHT: float = 80.0
    PADDLE_SPEED: float = 400.0  # pixels par seconde
    PADDLE_MARGIN: float = 50.0  # Distance du bord du terrain

    # Bonus
    BONUS_SIZE: float = 20.0
    BONUS_SPAWN_INTERVAL: float = 15.0  # secondes
    BONUS_DURATION: float = 10.0  # secondes
    PADDLE_SIZE_MULTIPLIER: float = 1.5  # Facteur d'agrandissement
    PADDLE_SIZE_REDUCER: float = 0.6  # Facteur de rétrécissement

    # Raquette tournante
    ROTATING_PADDLE_RADIUS: float = 40.0
    ROTATING_PADDLE_THICKNESS: float = 8.0
    ROTATING_PADDLE_SPEED: float = 2.0  # radians par seconde
    ROTATING_PADDLE_DURATION: float = 15.0  # secondes

    # Gameplay
    MAX_SCORE: int = 11
    GAME_SPEED_MULTIPLIER: float = 1.0  # Pour accélérer l'entraînement

    # Affichage
    FPS: int = 60
    BACKGROUND_COLOR: tuple[int, int, int] = (0, 0, 0)
    BALL_COLOR: tuple[int, int, int] = (255, 255, 255)
    PADDLE_COLOR: tuple[int, int, int] = (255, 255, 255)
    BONUS_COLORS: dict | None = None

    def __post_init__(self) -> None:
        if self.BONUS_COLORS is None:
            self.BONUS_COLORS = {
                "enlarge_paddle": (0, 255, 0),  # Vert
                "shrink_opponent": (255, 0, 0),  # Rouge
                "rotating_paddle": (0, 0, 255),  # Bleu
            }


@dataclass
class AIConfig:
    """Configuration pour l'interface IA"""

    # Observation space
    NORMALIZE_POSITIONS: bool = True  # Normaliser les positions entre -1 et 1
    INCLUDE_VELOCITY: bool = True  # Inclure la vélocité de la balle
    INCLUDE_HISTORY: bool = False  # Inclure l'historique des positions
    HISTORY_LENGTH: int = 3  # Nombre de frames d'historique

    # Reward system
    SCORE_REWARD: float = 1.0
    LOSE_PENALTY: float = -1.0
    BONUS_REWARD: float = 0.1
    WALL_HIT_REWARD: float = 0.01  # Petit bonus pour toucher la balle

    # Training
    MAX_EPISODE_STEPS: int = 10000
    HEADLESS_MODE: bool = False
    FAST_MODE_MULTIPLIER: float = 10.0  # Accélération en mode rapide


# Instance globale de configuration
game_config = GameConfig()
ai_config = AIConfig()
