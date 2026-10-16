# Insurance Federated Learning Platform — self-documenting Makefile
# Run `make help` to list all targets.
# PYTHON is venv-aware; override: make test PYTHON=/path/to/python

PYTHON ?= $(shell if [ -x venv/bin/python ]; then echo venv/bin/python; \
	elif [ -x .venv/bin/python ]; then echo .venv/bin/python; \
	else echo python3; fi)
PYTEST ?= $(PYTHON) -m pytest
PIP ?= $(PYTHON) -m pip

.DEFAULT_GOAL := help

.PHONY: help install install-dev lint format typecheck test test-unit \
	test-integration test-slow coverage data train fl-all evaluate serve \
	docker-build docker-run docker-test compose-up compose-down ci clean

help: ## Show this help message
	@grep -E '^[a-zA-Z_-]+:.*## ' $(MAKEFILE_LIST) | \
		awk 'BEGIN {FS = ":.*## "}; {printf "  \033[36m%-20s\033[0m %s\n", $$1, $$2}'

install: ## Create venv and install the package
	$(PYTHON) -m pip install -e .

install-dev: ## Install package with dev extras (tests, linters)
	$(PIP) install -e ".[dev]"

lint: ## Run ruff + black --check + isort --check
	$(PYTHON) -m ruff check src/insurance_fl tests
	$(PYTHON) -m black --check src/insurance_fl tests
	$(PYTHON) -m isort --check-only src/insurance_fl tests

format: ## Auto-format with black + isort + ruff
	$(PYTHON) -m black src/insurance_fl tests
	$(PYTHON) -m isort src/insurance_fl tests
	$(PYTHON) -m ruff check --fix src/insurance_fl tests

typecheck: ## Run mypy over the new package
	$(PYTHON) -m mypy src/insurance_fl

test: ## Run the full test suite (excludes slow)
	$(PYTEST) -q -m "not slow"

test-unit: ## Run unit tests only
	$(PYTEST) -q tests/unit tests/test_structure.py

test-integration: ## Run integration tests only
	$(PYTEST) -q -m integration

test-slow: ## Run slow tests (cross-process determinism, full generation)
	$(PYTEST) -q -m slow

coverage: ## Run tests with branch coverage (fail under 85%)
	$(PYTEST) -q --cov=insurance_fl --cov-branch --cov-report=term-missing \
		--cov-fail-under=85

data: ## Generate synthetic datasets into data/processed
	$(PYTHON) scripts/generate_dataset.py

train: ## Run one FL training round-set from config
	$(PYTHON) scripts/run_federated_training.py

fl-all: ## Run every FL experiment (config/experiments.yaml)
	$(PYTHON) scripts/run_federated_training.py --experiments config/experiments.yaml

evaluate: ## Evaluate models and write reports/
	$(PYTHON) scripts/evaluate_models.py

serve: ## Run the FastAPI service locally
	$(PYTHON) scripts/deploy_api.py

docker-build: ## Build the Docker image
	docker build -t insurance-fl:latest .

docker-run: ## Run the built image
	docker run --rm insurance-fl:latest

docker-test: ## Run container tests
	docker run --rm insurance-fl:latest python -m pytest -q -m "not slow"

compose-up: ## Start the federated docker-compose stack
	docker compose up --build -d

compose-down: ## Stop the federated docker-compose stack
	docker compose down

ci: ## What CI runs locally: lint, typecheck, tests, build
	$(MAKE) lint
	$(MAKE) typecheck
	$(MAKE) test
	$(MAKE) coverage

clean: ## Remove caches and build artifacts
	rm -rf .pytest_cache .mypy_cache .ruff_cache .coverage htmlcov dist build
	find . -name '__pycache__' -type d -prune -exec rm -rf {} + 2>/dev/null || true
	find . -name '._*' -not -path './.git/*' -delete 2>/dev/null || true
