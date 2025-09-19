"""
Exemple simple de jeu Magic Pong avec interface graphique
"""

import sys

try:
    from magic_pong.gui.game_app import MagicPongApp
except ImportError as e:
    print(f"Erreur: Impossible d'importer les modules requis: {e}")
    print("Assurez-vous que pygame est installé: pip install pygame")
    sys.exit(1)


def run_simple_game():
    """Lance une partie simple en mode 1 vs 1"""
    print("Lancement d'une partie Magic Pong 1 vs 1...")

    app = MagicPongApp()

    # Optionnel: démarrer directement en mode 1v1
    # app.start_game_mode(GameMode.ONE_VS_ONE)

    app.run()


if __name__ == "__main__":
    run_simple_game()
