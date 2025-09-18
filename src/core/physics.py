"""
Système de physique pour Magic Pong
"""

import random

from magic_pong.core.collision import CollisionDetector
from magic_pong.core.entities import Action, Ball, Bonus, BonusType, Paddle, RotatingPaddle
from magic_pong.utils.config import game_config


class BonusSpawner:
    """Gestionnaire d'apparition des bonus"""

    def __init__(self, field_width: float, field_height: float):
        self.field_width = field_width
        self.field_height = field_height
        self.spawn_timer = 0.0
        self.spawn_interval = game_config.BONUS_SPAWN_INTERVAL

    def update(self, dt: float, existing_bonuses: list[Bonus]) -> list[Bonus]:
        """Met à jour le spawner et retourne les nouveaux bonus"""
        self.spawn_timer += dt
        new_bonuses = []

        if self.spawn_timer >= self.spawn_interval:
            self.spawn_timer = 0.0

            # Ne pas spawner s'il y a déjà trop de bonus
            if len(existing_bonuses) < 4:
                new_bonuses = self._spawn_symmetric_bonuses()

        return new_bonuses

    def _spawn_symmetric_bonuses(self) -> list[Bonus]:
        """Spawne des bonus de manière symétrique"""
        bonuses = []

        # Choisir un type de bonus aléatoire
        bonus_type = random.choice(list(BonusType))

        # Position aléatoire dans la moitié gauche
        left_x = random.uniform(50, self.field_width / 2 - 50)
        y = random.uniform(50, self.field_height - 50)

        # Position symétrique dans la moitié droite
        right_x = self.field_width - left_x

        # Créer les bonus symétriques
        bonuses.append(Bonus(left_x, y, bonus_type))
        bonuses.append(Bonus(right_x, y, bonus_type))

        return bonuses


class PhysicsEngine:
    """Moteur de physique principal"""

    def __init__(self, field_width: float, field_height: float):
        self.field_width = field_width
        self.field_height = field_height
        self.collision_detector = CollisionDetector()
        self.bonus_spawner = BonusSpawner(field_width, field_height)

        # État du jeu
        self.ball = Ball(field_width / 2, field_height / 2, game_config.BALL_SPEED, 0)

        self.player1 = Paddle(
            game_config.PADDLE_MARGIN, field_height / 2 - game_config.PADDLE_HEIGHT / 2, 1
        )

        self.player2 = Paddle(
            field_width - game_config.PADDLE_MARGIN - game_config.PADDLE_WIDTH,
            field_height / 2 - game_config.PADDLE_HEIGHT / 2,
            2,
        )

        self.bonuses: list[Bonus] = []
        self.rotating_paddles: list[RotatingPaddle] = []
        self.score: list[int] = [0, 0]
        self.game_time = 0.0

        # Initialiser la balle avec une direction aléatoire
        self.reset_ball()

    def reset_ball(self, direction: int = 0) -> None:
        """Remet la balle au centre"""
        if direction == 0:
            direction = random.choice([-1, 1])
        self.ball.reset_to_center(direction)

    def update(self, dt: float, player1_action: Action, player2_action: Action) -> dict:
        """Met à jour la physique du jeu"""
        # Appliquer le multiplicateur de vitesse
        effective_dt = dt * game_config.GAME_SPEED_MULTIPLIER
        self.game_time += effective_dt

        # Déplacer les joueurs
        if player1_action:
            self.player1.move(player1_action.move_x, player1_action.move_y, effective_dt)
        if player2_action:
            self.player2.move(player2_action.move_x, player2_action.move_y, effective_dt)

        # Mettre à jour les entités
        self.ball.update(effective_dt)
        self.player1.update(effective_dt)
        self.player2.update(effective_dt)

        # Mettre à jour les raquettes tournantes
        self.rotating_paddles = [rp for rp in self.rotating_paddles if rp.update(effective_dt)]

        # Mettre à jour les bonus
        self.bonuses = [bonus for bonus in self.bonuses if bonus.update(effective_dt)]

        # Spawner de nouveaux bonus
        new_bonuses = self.bonus_spawner.update(effective_dt, self.bonuses)
        self.bonuses.extend(new_bonuses)

        # Vérifier les collisions
        events = self._check_collisions()

        return events

    def _check_collisions(self) -> dict:
        """Vérifie toutes les collisions et retourne les événements"""
        events: dict[str, list] = {
            "wall_bounces": [],
            "paddle_hits": [],
            "goals": [],
            "bonus_collected": [],
            "rotating_paddle_hits": [],
        }

        # Collisions avec les murs
        wall_collision = self.collision_detector.check_ball_walls(
            self.ball, self.field_width, self.field_height
        )

        if wall_collision == "top" or wall_collision == "bottom":
            self.ball.bounce_vertical()
            events["wall_bounces"].append(wall_collision)
        elif wall_collision == "left_goal":
            self.score[1] += 1  # Point pour le joueur 2
            events["goals"].append({"player": 2, "score": self.score.copy()})
            self.reset_ball(1)  # Relancer vers la droite
        elif wall_collision == "right_goal":
            self.score[0] += 1  # Point pour le joueur 1
            events["goals"].append({"player": 1, "score": self.score.copy()})
            self.reset_ball(-1)  # Relancer vers la gauche

        # Collisions avec les raquettes
        if self.collision_detector.check_ball_paddle(self.ball, self.player1):
            events["paddle_hits"].append({"player": 1})
        if self.collision_detector.check_ball_paddle(self.ball, self.player2):
            events["paddle_hits"].append({"player": 2})

        # Collisions avec les raquettes tournantes
        for rp in self.rotating_paddles:
            if self.collision_detector.check_ball_rotating_paddle(self.ball, rp):
                events["rotating_paddle_hits"].append({"player": rp.player_id})

        # Collisions joueur-bonus
        for player, paddle in [(1, self.player1), (2, self.player2)]:
            collected = self.collision_detector.check_player_bonus(paddle, self.bonuses)
            for bonus in collected:
                self._apply_bonus_effect(bonus.type, player)
                events["bonus_collected"].append({"player": player, "type": bonus.type.value})

        return events

    def _apply_bonus_effect(self, bonus_type: BonusType, player: int) -> None:
        """Applique l'effet d'un bonus"""
        if bonus_type == BonusType.ENLARGE_PADDLE:
            # Élargir la raquette du joueur
            paddle = self.player1 if player == 1 else self.player2
            paddle.apply_size_effect(game_config.PADDLE_SIZE_MULTIPLIER, game_config.BONUS_DURATION)

        elif bonus_type == BonusType.SHRINK_OPPONENT:
            # Rétrécir la raquette de l'adversaire
            opponent_paddle = self.player2 if player == 1 else self.player1
            opponent_paddle.apply_size_effect(
                game_config.PADDLE_SIZE_REDUCER, game_config.BONUS_DURATION
            )

        elif bonus_type == BonusType.ROTATING_PADDLE:
            # Ajouter une raquette tournante
            if player == 1:
                # Position dans la moitié gauche
                x = random.uniform(100, self.field_width / 2 - 100)
            else:
                # Position dans la moitié droite
                x = random.uniform(self.field_width / 2 + 100, self.field_width - 100)

            y = random.uniform(100, self.field_height - 100)

            rotating_paddle = RotatingPaddle(x, y, player)
            self.rotating_paddles.append(rotating_paddle)

    def get_game_state(self) -> dict:
        """Retourne l'état complet du jeu"""
        return {
            "ball_position": self.ball.position.to_tuple(),
            "ball_velocity": self.ball.velocity.to_tuple(),
            "player1_position": self.player1.position.to_tuple(),
            "player2_position": self.player2.position.to_tuple(),
            "player1_paddle_size": self.player1.height,
            "player2_paddle_size": self.player2.height,
            "active_bonuses": [
                (bonus.position.x, bonus.position.y, bonus.type.value)
                for bonus in self.bonuses
                if not bonus.collected
            ],
            "rotating_paddles": [
                (rp.center.x, rp.center.y, rp.angle) for rp in self.rotating_paddles
            ],
            "score": self.score.copy(),
            "time_elapsed": self.game_time,
            "field_bounds": (0, self.field_width, 0, self.field_height),
        }

    def is_game_over(self) -> bool:
        """Vérifie si la partie est terminée"""
        return max(self.score) >= game_config.MAX_SCORE

    def get_winner(self) -> int:
        """Retourne le gagnant (1 ou 2), ou 0 si pas de gagnant"""
        if self.score[0] >= game_config.MAX_SCORE:
            return 1
        elif self.score[1] >= game_config.MAX_SCORE:
            return 2
        return 0

    def reset_game(self) -> None:
        """Remet le jeu à zéro"""
        self.score = [0, 0]
        self.game_time = 0.0
        self.bonuses.clear()
        self.rotating_paddles.clear()

        # Remettre les raquettes à leur position initiale
        self.player1.position.y = self.field_height / 2 - game_config.PADDLE_HEIGHT / 2
        self.player2.position.y = self.field_height / 2 - game_config.PADDLE_HEIGHT / 2
        self.player1.reset_size()
        self.player2.reset_size()

        # Remettre la balle au centre
        self.reset_ball()
