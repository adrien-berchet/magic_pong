.PHONY: help install install-dev test test-all lint format type-check clean build docs
.DEFAULT_GOAL := help

help: ## Affiche cette aide
	@echo "Commandes disponibles :"
	@grep -E '^[a-zA-Z_-]+:.*?## .*$$' $(MAKEFILE_LIST) | sort | awk 'BEGIN {FS = ":.*?## "}; {printf "\033[36m%-20s\033[0m %s\n", $$1, $$2}'

install: ## Installe le package en mode développement
	pip install -e .

install-dev: ## Installe les dépendances de développement
	pip install -e ".[dev]"

install-all: ## Installe toutes les dépendances (dev, ai, viz)
	pip install -e ".[all]"

test: ## Lance les tests avec pytest
	pytest tests/ -v

test-cov: ## Lance les tests avec couverture de code
	pytest tests/ -v --cov=magic_pong --cov-report=term-missing --cov-report=html

test-all: ## Lance tous les tests avec tox
	tox

lint: ## Vérifie le code avec ruff
	ruff check src/ tests/

lint-fix: ## Corrige automatiquement les problèmes de linting
	ruff check --fix src/ tests/

format: ## Formate le code avec black
	black src/ tests/

format-check: ## Vérifie le formatage sans modifier les fichiers
	black --check --diff src/ tests/

type-check: ## Vérifie les types avec mypy
	mypy src/

quality: ## Lance tous les contrôles qualité (lint, format, type-check)
	$(MAKE) lint
	$(MAKE) format-check
	$(MAKE) type-check

quality-fix: ## Corrige automatiquement les problèmes de qualité
	$(MAKE) lint-fix
	$(MAKE) format

clean: ## Nettoie les fichiers temporaires
	find . -type f -name "*.pyc" -delete
	find . -type d -name "__pycache__" -delete
	find . -type d -name "*.egg-info" -exec rm -rf {} +
	rm -rf build/
	rm -rf dist/
	rm -rf .tox/
	rm -rf .coverage
	rm -rf htmlcov/
	rm -rf .pytest_cache/
	rm -rf .mypy_cache/

build: ## Construit le package
	python -m build

docs: ## Génère la documentation (si configurée)
	@echo "Documentation non configurée pour le moment"

dev-setup: install-dev ## Configuration complète pour le développement
	@echo "✅ Environnement de développement configuré !"
	@echo "Commandes utiles :"
	@echo "  make test          - Lance les tests"
	@echo "  make test-all      - Lance tous les tests avec tox"
	@echo "  make quality       - Vérifie la qualité du code"
	@echo "  make quality-fix   - Corrige automatiquement les problèmes"

# Commandes tox
tox-py310: ## Lance les tests avec Python 3.10
	tox -e py310

tox-py311: ## Lance les tests avec Python 3.11
	tox -e py311

tox-py312: ## Lance les tests avec Python 3.12
	tox -e py312

tox-lint: ## Lance le linting avec tox
	tox -e lint

tox-format: ## Lance le formatage avec tox
	tox -e format

tox-type-check: ## Lance la vérification de types avec tox
	tox -e type-check

# Commandes pour les exemples
# Commandes pre-commit
pre-commit-install: ## Installe les hooks pre-commit
	pre-commit install

pre-commit-run: ## Lance pre-commit sur tous les fichiers
	pre-commit run --all-files

pre-commit-update: ## Met à jour les hooks pre-commit
	pre-commit autoupdate

# Commandes pour les exemples
run-example: ## Lance un exemple d'IA vs IA
	python -m magic_pong.ai.models.ai_vs_ai

run-tournament: ## Lance un tournoi d'IA
	magic-pong-tournament
