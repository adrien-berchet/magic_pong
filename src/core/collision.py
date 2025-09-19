"""
Système de détection de collisions pour Magic Pong
"""

import math

from magic_pong.core.entities import Ball, Bonus, Paddle, RotatingPaddle, Vector2D


def point_in_rect(point: Vector2D, rect: tuple[float, float, float, float]) -> bool:
    """Vérifie si un point est dans un rectangle"""
    x, y, width, height = rect
    return x <= point.x <= x + width and y <= point.y <= y + height


def circle_rect_collision(ball: Ball, rect: tuple[float, float, float, float]) -> bool:
    """Détecte la collision entre un cercle (balle) et un rectangle"""
    x, y, width, height = rect

    # Point le plus proche du rectangle au centre de la balle
    closest_x = max(x, min(ball.position.x, x + width))
    closest_y = max(y, min(ball.position.y, y + height))

    # Distance entre le centre de la balle et le point le plus proche
    distance = math.sqrt((ball.position.x - closest_x) ** 2 + (ball.position.y - closest_y) ** 2)

    return distance <= ball.radius


def circle_line_collision(ball: Ball, line_start: Vector2D, line_end: Vector2D) -> bool:
    """Détecte la collision entre un cercle et un segment de ligne"""
    # Vecteur de la ligne
    line_vec = line_end - line_start
    line_length = line_vec.magnitude()

    if line_length == 0:
        # Ligne de longueur nulle, vérifier la distance au point
        distance = (ball.position - line_start).magnitude()
        return distance <= ball.radius

    # Normaliser le vecteur de ligne
    line_unit = line_vec.normalize()

    # Vecteur du début de ligne au centre de la balle
    to_ball = ball.position - line_start

    # Projection du vecteur to_ball sur la ligne
    projection_length = to_ball.x * line_unit.x + to_ball.y * line_unit.y

    # Limiter la projection à la longueur de la ligne
    projection_length = max(0, min(line_length, projection_length))

    # Point le plus proche sur la ligne
    closest_point = line_start + line_unit * projection_length

    # Distance du centre de la balle au point le plus proche
    distance = (ball.position - closest_point).magnitude()

    return distance <= ball.radius


def get_paddle_collision_normal(ball: Ball, paddle: Paddle) -> Vector2D | None:
    """Calcule la normale de collision avec une raquette"""
    rect = paddle.get_rect()
    x, y, width, height = rect

    # Déterminer de quel côté la balle frappe la raquette
    ball_center_x = ball.position.x
    ball_center_y = ball.position.y

    # Centre du rectangle
    rect_center_x = x + width / 2
    rect_center_y = y + height / 2

    # Différences
    dx = ball_center_x - rect_center_x
    dy = ball_center_y - rect_center_y

    # Ratios pour déterminer le côté
    width_ratio = dx / (width / 2) if width > 0 else 0
    height_ratio = dy / (height / 2) if height > 0 else 0

    # Le côté avec le plus grand ratio absolu est celui touché
    if abs(width_ratio) > abs(height_ratio):
        # Collision horizontale (gauche ou droite)
        return Vector2D(1.0 if dx > 0 else -1.0, 0.0)
    else:
        # Collision verticale (haut ou bas)
        return Vector2D(0.0, 1.0 if dy > 0 else -1.0)


def apply_paddle_bounce(ball: Ball, paddle: Paddle) -> None:
    """Applique l'effet de rebond sur une raquette avec effet"""
    normal = get_paddle_collision_normal(ball, paddle)
    if not normal:
        return

    # Position relative de la balle sur la raquette (pour l'effet)
    paddle_center_y = paddle.position.y + paddle.height / 2
    relative_hit_pos = (ball.position.y - paddle_center_y) / (paddle.height / 2)
    relative_hit_pos = max(-1.0, min(1.0, relative_hit_pos))  # Clamp entre -1 et 1

    # Vitesse actuelle
    speed = ball.velocity.magnitude()

    if abs(normal.x) > abs(normal.y):
        # Rebond horizontal (raquette)
        ball.velocity.x = -ball.velocity.x
        # Ajouter un effet vertical basé sur la position de frappe
        ball.velocity.y += relative_hit_pos * speed * 0.3
    else:
        # Rebond vertical (mur)
        ball.velocity.y = -ball.velocity.y

    # Normaliser et remettre à la bonne vitesse
    ball.velocity = ball.velocity.normalize() * speed

    # Légère accélération
    ball.velocity = ball.velocity * 1.02


class CollisionDetector:
    """Gestionnaire principal des collisions"""

    def __init__(self) -> None:
        pass

    def check_ball_walls(self, ball: Ball, field_width: float, field_height: float) -> str:
        """Vérifie les collisions avec les murs. Retourne le type de collision."""
        # Murs haut et bas
        if ball.position.y - ball.radius <= 0:
            ball.position.y = ball.radius
            return "top"
        elif ball.position.y + ball.radius >= field_height:
            ball.position.y = field_height - ball.radius
            return "bottom"

        # Murs gauche et droite (but)
        if ball.position.x - ball.radius <= 0:
            return "left_goal"
        elif ball.position.x + ball.radius >= field_width:
            return "right_goal"

        return "none"

    def check_ball_paddle(self, ball: Ball, paddle: Paddle) -> bool:
        """Vérifie et traite la collision balle-raquette"""
        if ball.last_paddle_hit == paddle.player_id:
            # Éviter les rebonds multiples sur la même raquette
            # Vérifier si la balle s'éloigne de la raquette
            rect = paddle.get_rect()
            paddle_center_x = rect[0] + rect[2] / 2

            if paddle.player_id == 1:  # Raquette gauche
                if ball.velocity.x > 0:  # Balle s'éloigne vers la droite
                    ball.last_paddle_hit = None
            else:  # Raquette droite
                if ball.velocity.x < 0:  # Balle s'éloigne vers la gauche
                    ball.last_paddle_hit = None

            return False

        if circle_rect_collision(ball, paddle.get_rect()):
            apply_paddle_bounce(ball, paddle)
            ball.last_paddle_hit = paddle.player_id
            return True

        return False

    def check_ball_rotating_paddle(self, ball: Ball, rotating_paddle: RotatingPaddle) -> bool:
        """Vérifie la collision avec une raquette tournante"""
        segments = rotating_paddle.get_line_segments()

        for start, end in segments:
            if circle_line_collision(ball, start, end):
                # Calculer la normale de collision
                line_vec = end - start
                normal = Vector2D(-line_vec.y, line_vec.x).normalize()

                # Réfléchir la vitesse
                dot_product = ball.velocity.x * normal.x + ball.velocity.y * normal.y
                ball.velocity.x -= 2 * dot_product * normal.x
                ball.velocity.y -= 2 * dot_product * normal.y

                return True

        return False

    def check_player_bonus(self, paddle: Paddle, bonuses: list[Bonus]) -> list[Bonus]:
        """Vérifie les collisions joueur-bonus"""
        collected = []
        paddle_rect = paddle.get_rect()

        for bonus in bonuses:
            if not bonus.collected:
                bonus_rect = bonus.get_rect()

                # Vérification simple de chevauchement des rectangles
                if (
                    paddle_rect[0] < bonus_rect[0] + bonus_rect[2]
                    and paddle_rect[0] + paddle_rect[2] > bonus_rect[0]
                    and paddle_rect[1] < bonus_rect[1] + bonus_rect[3]
                    and paddle_rect[1] + paddle_rect[3] > bonus_rect[1]
                ):
                    bonus.collect()
                    collected.append(bonus)

        return collected
