# Magic Pong

Un jeu de Pong élaboré conçu spécialement pour l'entraînement d'intelligence artificielle, avec des fonctionnalités avancées et une architecture flexible.

## Fonctionnalités

### Gameplay Avancé
- **Mouvement libre** : Les joueurs peuvent se déplacer librement dans leur moitié de terrain (pas seulement verticalement)
- **Système de bonus symétriques** :
  - Élargissement de la raquette du joueur
  - Rétrécissement de la raquette adverse
  - Raquette tournante supplémentaire
- **Physique réaliste** avec rebonds et effets

### Interface IA
- **Architecture agnostique** : Compatible avec différents frameworks d'IA (PyTorch, TensorFlow, etc.)
- **Mode headless** : Entraînement ultra-rapide sans affichage graphique
- **Vitesse variable** : Accélération jusqu'à 1000x pour l'entraînement
- **Système de récompenses** configurable
- **Observations normalisées** pour l'apprentissage

### Exemples d'IA Inclus
- **RandomAI** : Mouvements aléatoires
- **FollowBallAI** : Suit la balle
- **DefensiveAI** : Stratégie défensive
- **AggressiveAI** : Cherche les bonus et attaque
- **PredictiveAI** : Prédit la trajectoire de la balle

## Installation

```bash
# Cloner le projet
git clone <repository_url>
cd magic_pong

# Installer les dépendances
pip install -r requirements.txt
```

## Utilisation Rapide

### Entraînement IA vs IA

```python
from src.core.game_engine import TrainingManager
from src.ai.examples.simple_ai import create_ai

# Créer le gestionnaire d'entraînement
trainer = TrainingManager(headless=True)

# Créer les IA
player1 = create_ai('aggressive', 1)
player2 = create_ai('defensive', 2)

# Entraîner un épisode
stats = trainer.train_episode(player1, player2)
print(f"Gagnant: Joueur {stats['winner']}")
```

### Tournoi d'IA

```bash
cd magic_pong
python examples/ai_vs_ai.py --mode tournament
```

### Entraînement Simple

```bash
cd magic_pong
python examples/ai_vs_ai.py --mode training
```

## Architecture

```
magic_pong/
├── src/
│   ├── core/           # Moteur de jeu et physique
│   ├── ai/             # Interface IA et exemples
│   ├── graphics/       # Rendu graphique (à venir)
│   └── utils/          # Configuration et utilitaires
├── examples/           # Exemples d'utilisation
└── docs/              # Documentation
```

### Composants Principaux

- **PhysicsEngine** : Gère la physique du jeu, collisions, bonus
- **GameEngine** : Orchestre le jeu et gère les joueurs
- **TrainingManager** : Optimisé pour l'entraînement d'IA
- **AIPlayer** : Interface abstraite pour les IA
- **GameEnvironment** : Environnement compatible avec les frameworks RL

## Configuration

Le jeu est hautement configurable via [`src/utils/config.py`](src/utils/config.py):

```python
from src.utils.config import game_config, ai_config

# Configuration du jeu
game_config.FIELD_WIDTH = 800
game_config.FIELD_HEIGHT = 600
game_config.BALL_SPEED = 300.0

# Configuration IA
ai_config.HEADLESS_MODE = True
ai_config.FAST_MODE_MULTIPLIER = 10.0
```

## Créer une IA Personnalisée

```python
from src.ai.interface import AIPlayer
from src.core.entities import Action

class MonIA(AIPlayer):
    def get_action(self, observation):
        # Votre logique ici
        ball_pos = observation['ball_pos']
        player_pos = observation['player_pos']

        # Calculer l'action
        move_x = ball_pos[0] - player_pos[0]
        move_y = ball_pos[1] - player_pos[1]

        return Action(move_x, move_y)

    def on_step(self, observation, action, reward, done, info):
        # Apprentissage ici
        self.current_episode_reward += reward
```

## Interface avec PyTorch

```python
import torch
import torch.nn as nn
from src.ai.interface import AIPlayer

class PyTorchAI(AIPlayer):
    def __init__(self, player_id, model):
        super().__init__(player_id)
        self.model = model

    def get_action(self, observation):
        # Convertir l'observation en tensor
        state = self._obs_to_tensor(observation)

        # Prédiction du modèle
        with torch.no_grad():
            action_probs = self.model(state)

        # Convertir en Action
        return Action(
            move_x=action_probs[0].item(),
            move_y=action_probs[1].item()
        )
```

## Observations pour l'IA

L'observation fournie à chaque IA contient :

```python
observation = {
    'ball_pos': [x, y],                    # Position de la balle
    'ball_vel': [vx, vy],                  # Vélocité de la balle
    'player_pos': [x, y],                  # Position du joueur
    'opponent_pos': [x, y],                # Position de l'adversaire
    'player_paddle_size': float,           # Taille de la raquette
    'opponent_paddle_size': float,         # Taille raquette adverse
    'bonuses': [[x, y, type], ...],        # Bonus actifs
    'rotating_paddles': [[x, y, angle]], # Raquettes tournantes
    'score_diff': int,                     # Différence de score
    'time_elapsed': float                  # Temps écoulé
}
```

## Système de Récompenses

- **+1.0** : Marquer un point
- **-1.0** : Encaisser un point
- **+0.1** : Collecter un bonus
- **+0.01** : Toucher la balle
- **+0.02** : Utiliser une raquette tournante

## Performance

En mode headless avec accélération :
- **Vitesse normale** : ~60 FPS
- **Mode rapide** : ~600-6000 FPS (10-100x plus rapide)
- **Entraînement** : Milliers d'épisodes par minute

## Exemples de Résultats

Tournoi entre les IA incluses (20 parties chacune) :

```
Classement:
1. aggressive: 52 victoires
2. predictive: 48 victoires
3. defensive: 31 victoires
4. follow_ball: 28 victoires
5. random: 1 victoire
```

## Développement

### Structure du Code

- **Séparation claire** entre logique métier et affichage
- **Architecture modulaire** et extensible
- **Type hints** complets pour une meilleure maintenance
- **Tests unitaires** (à venir)

### Ajouter de Nouveaux Bonus

```python
# Dans entities.py
class BonusType(Enum):
    MON_BONUS = "mon_bonus"

# Dans physics.py
def _apply_bonus_effect(self, bonus_type, player):
    if bonus_type == BonusType.MON_BONUS:
        # Votre effet ici
        pass
```

## Roadmap

- [ ] Interface graphique Pygame
- [ ] Mode multijoueur en réseau
- [ ] Intégration Gymnasium
- [ ] Sauvegarde/chargement de modèles
- [ ] Métriques avancées et visualisations
- [ ] Support GPU pour l'entraînement

## Contribution

Les contributions sont les bienvenues ! Voir [`CONTRIBUTING.md`](CONTRIBUTING.md) pour les guidelines.

## Licence

MIT License - voir [`LICENSE`](LICENSE) pour les détails.

## Auteur

Adrien Berchet - Projet Magic Pong pour l'entraînement d'IA

---

**Magic Pong** - Où l'IA apprend à jouer ! 🏓🤖
