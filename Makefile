.PHONY: help install install-dev test test-all lint format type-check clean build docs
.DEFAULT_GOAL := help

help: ## Display this help
	@echo "Available commands:"
	@grep -E '^[a-zA-Z_-]+:.*?## .*$$' $(MAKEFILE_LIST) | sort | awk 'BEGIN {FS = ":.*?## "}; {printf "\033[36m%-20s\033[0m %s\n", $$1, $$2}'

install: ## Install the package in development mode
	pip install -e .

install-dev: ## Install development dependencies
	pip install -e ".[dev]"

install-all: ## Install all dependencies (dev, ai, viz)
	pip install -e ".[all]"

test: ## Run tests with pytest
	pytest tests/ -v

test-cov: ## Run tests with code coverage
	pytest tests/ -v --cov=magic_pong --cov-report=term-missing --cov-report=html

test-all: ## Run all tests with tox
	tox

lint: ## Check code with ruff
	ruff check magic_pong/ tests/

lint-fix: ## Automatically fix linting issues
	ruff check --fix magic_pong/ tests/

format: ## Format code with black
	black magic_pong/ tests/

format-check: ## Check formatting without modifying files
	black --check --diff magic_pong/ tests/

type-check: ## Check types with mypy
	mypy magic_pong/

quality: ## Run all quality checks (lint, format, type-check)
	$(MAKE) lint
	$(MAKE) format-check
	$(MAKE) type-check

quality-fix: ## Automatically fix quality issues
	$(MAKE) lint-fix
	$(MAKE) format

clean: ## Clean temporary files
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

build: ## Build the package
	python -m build

docs: ## Generate documentation (if configured)
	@echo "Documentation not configured yet"

dev-setup: install-dev ## Complete setup for development
	@echo "✅ Development environment configured!"
	@echo "Useful commands:"
	@echo "  make test          - Run tests"
	@echo "  make test-all      - Run all tests with tox"
	@echo "  make quality       - Check code quality"
	@echo "  make quality-fix   - Automatically fix issues"

# Tox commands
tox-py310: ## Run tests with Python 3.10
	tox -e py310

tox-py311: ## Run tests with Python 3.11
	tox -e py311

tox-py312: ## Run tests with Python 3.12
	tox -e py312

tox-lint: ## Run linting with tox
	tox -e lint

tox-format: ## Run formatting with tox
	tox -e format

tox-type-check: ## Run type checking with tox
	tox -e type-check

# Pre-commit commands
pre-commit-install: ## Install pre-commit hooks
	pre-commit install

pre-commit-run: ## Run pre-commit on all files
	pre-commit run --all-files

pre-commit-update: ## Update pre-commit hooks
	pre-commit autoupdate

# Example commands
run-example: ## Run an AI vs AI example
	python -m magic_pong.ai.models.ai_vs_ai

run-tournament: ## Run an AI tournament
	magic-pong-tournament
