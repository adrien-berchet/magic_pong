"""
Entités du jeu Magic Pong : balle, raquettes, bonus
"""

import math
from dataclasses import dataclass
from enum import Enum
from typing import Tuple, List, Optional
import numpy as np

from magic_pong.utils.config import game_config


class BonusType(Enum):
    """Types de bonus disponibles"""
    ENLARGE_PADDLE = "enlarge_paddle"
    SHRINK_OPPONENT = "shrink_opponent"
    ROTATING_PADDLE = "rotating_paddle"


@dataclass
class Vector2D:
    """Vecteur 2D simple pour les positions et vélocités"""
    x: float
    y: float

    def __add__(self, other: 'Vector2D') -> 'Vector2D':
        return Vector2D(self.x + other.x, self.y + other.y)

    def __sub__(self, other: 'Vector2D') -> 'Vector2D':
        return Vector2D(self.x - other.x, self.y - other.y)

    def __mul__(self, scalar: float) -> 'Vector2D':
        return Vector2D(self.x * scalar, self.y * scalar)

    def magnitude(self) -> float:
        return math.sqrt(self.x ** 2 + self.y ** 2)

    def normalize(self) -> 'Vector2D':
        mag = self.magnitude()
        if mag == 0:
            return Vector2D(0, 0)
        return Vector2D(self.x / mag, self.y / mag)

    def to_tuple(self) -> Tuple[float, float]:
        return (self.x, self.y)


class Ball:
    """Balle du jeu"""

    def __init__(self, x: float, y: float, vx: float, vy: float):
        self.position = Vector2D(x, y)
        self.velocity = Vector2D(vx, vy)
        self.radius = game_config.BALL_RADIUS
        self.last_paddle_hit: Optional[int] = None  # Pour éviter les rebonds multiples

    def update(self, dt: float) -> None:
        """Met à jour la position de la balle"""
        self.position = self.position + self.velocity * dt

    def reset_to_center(self, direction: int = 1) -> None:
        """Remet la balle au centre avec une direction donnée"""
        self.position = Vector2D(
            game_config.FIELD_WIDTH / 2,
            game_config.FIELD_HEIGHT / 2
        )

        # Direction aléatoire mais contrôlée
        angle = np.random.uniform(-math.pi/4, math.pi/4)  # ±45 degrés
        speed = game_config.BALL_SPEED

        self.velocity = Vector2D(
            direction * speed * math.cos(angle),
            speed * math.sin(angle)
        )

    def bounce_vertical(self) -> None:
        """Rebond vertical (murs haut/bas)"""
        self.velocity.y = -self.velocity.y

    def bounce_horizontal(self) -> None:
        """Rebond horizontal (raquettes)"""
        self.velocity.x = -self.velocity.x
        # Légère accélération après chaque rebond
        speed_increase = game_config.BALL_SPEED_INCREASE
        self.velocity = self.velocity * speed_increase


class Paddle:
    """Raquette de joueur"""

    def __init__(self, x: float, y: float, player_id: int):
        self.position = Vector2D(x, y)
        self.player_id = player_id
        self.width = game_config.PADDLE_WIDTH
        self.height = game_config.PADDLE_HEIGHT
        self.original_height = game_config.PADDLE_HEIGHT
        self.size_effect_timer = 0.0

        # Limites de mouvement selon le joueur
        if player_id == 1:  # Joueur gauche
            self.min_x = 0
            self.max_x = game_config.FIELD_WIDTH / 2 - self.width
        else:  # Joueur droite
            self.min_x = game_config.FIELD_WIDTH / 2
            self.max_x = game_config.FIELD_WIDTH - self.width

        self.min_y = 0
        self.max_y = game_config.FIELD_HEIGHT - self.height

    def update(self, dt: float) -> None:
        """Met à jour la raquette (effets temporaires)"""
        if self.size_effect_timer > 0:
            self.size_effect_timer -= dt
            if self.size_effect_timer <= 0:
                self.reset_size()

    def move(self, dx: float, dy: float, dt: float) -> None:
        """Déplace la raquette avec contraintes"""
        speed = game_config.PADDLE_SPEED * dt
        new_x = self.position.x + dx * speed
        new_y = self.position.y + dy * speed

        # Contraintes de mouvement
        self.position.x = max(self.min_x, min(self.max_x, new_x))
        self.position.y = max(self.min_y, min(self.max_y, new_y))

    def apply_size_effect(self, multiplier: float, duration: float) -> None:
        """Applique un effet de taille temporaire"""
        self.height = self.original_height * multiplier
        self.size_effect_timer = duration
        # Réajuster les limites Y
        self.max_y = game_config.FIELD_HEIGHT - self.height
        if self.position.y > self.max_y:
            self.position.y = self.max_y

    def reset_size(self) -> None:
        """Remet la taille normale"""
        self.height = self.original_height
        self.max_y = game_config.FIELD_HEIGHT - self.height

    def get_rect(self) -> Tuple[float, float, float, float]:
        """Retourne le rectangle de collision (x, y, width, height)"""
        return (self.position.x, self.position.y, self.width, self.height)


class RotatingPaddle:
    """Raquette tournante (bonus)"""

    def __init__(self, x: float, y: float, player_id: int):
        self.center = Vector2D(x, y)
        self.player_id = player_id
        self.radius = game_config.ROTATING_PADDLE_RADIUS
        self.thickness = game_config.ROTATING_PADDLE_THICKNESS
        self.angle = 0.0
        self.angular_speed = game_config.ROTATING_PADDLE_SPEED
        self.lifetime = game_config.ROTATING_PADDLE_DURATION

    def update(self, dt: float) -> bool:
        """Met à jour la rotation et la durée de vie"""
        self.angle += self.angular_speed * dt
        self.lifetime -= dt
        return self.lifetime > 0

    def get_line_segments(self) -> List[Tuple[Vector2D, Vector2D]]:
        """Retourne les segments de ligne pour la collision"""
        # Quatre segments formant un carré tournant
        segments = []
        for i in range(4):
            angle1 = self.angle + i * math.pi / 2
            angle2 = self.angle + (i + 1) * math.pi / 2

            p1 = Vector2D(
                self.center.x + self.radius * math.cos(angle1),
                self.center.y + self.radius * math.sin(angle1)
            )
            p2 = Vector2D(
                self.center.x + self.radius * math.cos(angle2),
                self.center.y + self.radius * math.sin(angle2)
            )
            segments.append((p1, p2))

        return segments


class Bonus:
    """Bonus qui apparaît sur le terrain"""

    def __init__(self, x: float, y: float, bonus_type: BonusType):
        self.position = Vector2D(x, y)
        self.type = bonus_type
        self.size = game_config.BONUS_SIZE
        self.collected = False
        self.lifetime = 30.0  # Disparaît après 30 secondes si pas collecté

    def update(self, dt: float) -> bool:
        """Met à jour le bonus, retourne False si expiré"""
        self.lifetime -= dt
        return self.lifetime > 0 and not self.collected

    def collect(self) -> BonusType:
        """Collecte le bonus"""
        self.collected = True
        return self.type

    def get_rect(self) -> Tuple[float, float, float, float]:
        """Retourne le rectangle de collision"""
        half_size = self.size / 2
        return (
            self.position.x - half_size,
            self.position.y - half_size,
            self.size,
            self.size
        )


@dataclass
class GameState:
    """État complet du jeu pour l'IA"""
    ball_position: Tuple[float, float]
    ball_velocity: Tuple[float, float]
    player1_position: Tuple[float, float]
    player2_position: Tuple[float, float]
    player1_paddle_size: float
    player2_paddle_size: float
    active_bonuses: List[Tuple[float, float, str]]  # position + type
    rotating_paddles: List[Tuple[float, float, float]]  # center + angle
    score: Tuple[int, int]
    time_elapsed: float
    field_bounds: Tuple[float, float, float, float] = (0, 800, 0, 600)


@dataclass
class Action:
    """Action d'un joueur"""
    move_x: float  # -1.0 à 1.0
    move_y: float  # -1.0 à 1.0

    def __post_init__(self):
        # Clamp les valeurs entre -1 et 1
        self.move_x = max(-1.0, min(1.0, self.move_x))
        self.move_y = max(-1.0, min(1.0, self.move_y))
