"""
Tests pour les entités du jeu Magic Pong
"""

import os
import sys

sys.path.insert(0, os.path.join(os.path.dirname(__file__), "..", "src"))

from core.entities import Action, Ball, Paddle, Vector2D


class TestVector2D:
    """Tests pour la classe Vector2D"""

    def test_creation(self) -> None:
        """Test de création d'un vecteur"""
        v = Vector2D(3.0, 4.0)
        assert v.x == 3.0
        assert v.y == 4.0

    def test_addition(self) -> None:
        """Test de l'addition de vecteurs"""
        v1 = Vector2D(1.0, 2.0)
        v2 = Vector2D(3.0, 4.0)
        result = v1 + v2
        assert result.x == 4.0
        assert result.y == 6.0

    def test_soustraction(self) -> None:
        """Test de la soustraction de vecteurs"""
        v1 = Vector2D(5.0, 7.0)
        v2 = Vector2D(2.0, 3.0)
        result = v1 - v2
        assert result.x == 3.0
        assert result.y == 4.0

    def test_multiplication_scalaire(self) -> None:
        """Test de la multiplication par un scalaire"""
        v = Vector2D(2.0, 3.0)
        result = v * 2.5
        assert result.x == 5.0
        assert result.y == 7.5

    def test_magnitude(self) -> None:
        """Test du calcul de la magnitude"""
        v = Vector2D(3.0, 4.0)
        assert v.magnitude() == 5.0

    def test_magnitude_zero(self) -> None:
        """Test de la magnitude d'un vecteur nul"""
        v = Vector2D(0.0, 0.0)
        assert v.magnitude() == 0.0

    def test_normalize(self) -> None:
        """Test de la normalisation"""
        v = Vector2D(3.0, 4.0)
        normalized = v.normalize()
        assert abs(normalized.magnitude() - 1.0) < 1e-10
        assert abs(normalized.x - 0.6) < 1e-10
        assert abs(normalized.y - 0.8) < 1e-10

    def test_normalize_zero_vector(self) -> None:
        """Test de la normalisation d'un vecteur nul"""
        v = Vector2D(0.0, 0.0)
        normalized = v.normalize()
        assert normalized.x == 0.0
        assert normalized.y == 0.0

    def test_to_tuple(self) -> None:
        """Test de conversion en tuple"""
        v = Vector2D(1.5, 2.5)
        assert v.to_tuple() == (1.5, 2.5)


class TestAction:
    """Tests pour la classe Action"""

    def test_creation_valeurs_valides(self) -> None:
        """Test de création avec des valeurs valides"""
        action = Action(0.5, -0.3)
        assert action.move_x == 0.5
        assert action.move_y == -0.3

    def test_clamp_valeurs_trop_grandes(self) -> None:
        """Test du clamping des valeurs trop grandes"""
        action = Action(2.0, -1.5)
        assert action.move_x == 1.0
        assert action.move_y == -1.0

    def test_clamp_valeurs_limites(self) -> None:
        """Test des valeurs limites"""
        action = Action(1.0, -1.0)
        assert action.move_x == 1.0
        assert action.move_y == -1.0


class TestBall:
    """Tests pour la classe Ball"""

    def test_creation(self) -> None:
        """Test de création d'une balle"""
        ball = Ball(100.0, 200.0, 50.0, -30.0)
        assert ball.position.x == 100.0
        assert ball.position.y == 200.0
        assert ball.velocity.x == 50.0
        assert ball.velocity.y == -30.0

    def test_update_position(self) -> None:
        """Test de mise à jour de la position"""
        ball = Ball(0.0, 0.0, 100.0, 50.0)
        ball.update(0.1)  # 0.1 seconde
        assert ball.position.x == 10.0
        assert ball.position.y == 5.0

    def test_bounce_vertical(self) -> None:
        """Test du rebond vertical"""
        ball = Ball(0.0, 0.0, 100.0, 50.0)
        ball.bounce_vertical()
        assert ball.velocity.x == 100.0
        assert ball.velocity.y == -50.0

    def test_bounce_horizontal(self) -> None:
        """Test du rebond horizontal"""
        ball = Ball(0.0, 0.0, 100.0, 50.0)
        original_speed = ball.velocity.magnitude()
        ball.bounce_horizontal()
        assert ball.velocity.x == -100.0
        assert ball.velocity.y == 50.0
        # Vérifier que la vitesse a augmenté
        assert ball.velocity.magnitude() > original_speed


class TestPaddle:
    """Tests pour la classe Paddle"""

    def test_creation_joueur_gauche(self) -> None:
        """Test de création d'une raquette pour le joueur gauche"""
        paddle = Paddle(50.0, 100.0, 1)
        assert paddle.position.x == 50.0
        assert paddle.position.y == 100.0
        assert paddle.player_id == 1
        assert paddle.min_x == 0

    def test_creation_joueur_droite(self) -> None:
        """Test de création d'une raquette pour le joueur droite"""
        paddle = Paddle(700.0, 100.0, 2)
        assert paddle.position.x == 700.0
        assert paddle.position.y == 100.0
        assert paddle.player_id == 2
        # Le min_x devrait être la moitié de la largeur du terrain
        assert paddle.min_x > 0

    def test_get_rect(self) -> None:
        """Test de récupération du rectangle de collision"""
        paddle = Paddle(100.0, 200.0, 1)
        rect = paddle.get_rect()
        assert rect[0] == 100.0  # x
        assert rect[1] == 200.0  # y
        assert rect[2] == paddle.width
        assert rect[3] == paddle.height

    def test_apply_size_effect(self) -> None:
        """Test de l'application d'un effet de taille"""
        paddle = Paddle(100.0, 200.0, 1)
        original_height = paddle.height
        paddle.apply_size_effect(1.5, 5.0)
        assert paddle.height == original_height * 1.5
        assert paddle.size_effect_timer == 5.0

    def test_reset_size(self) -> None:
        """Test de la remise à la taille normale"""
        paddle = Paddle(100.0, 200.0, 1)
        original_height = paddle.height
        paddle.apply_size_effect(2.0, 5.0)
        paddle.reset_size()
        assert paddle.height == original_height
        assert paddle.size_effect_timer == 0.0
