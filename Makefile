# Insurance Federated Learning Platform — self-documenting Makefile
# Run `make help` to list all targets.
# PYTHON is venv-aware; override: make test PYTHON=/path/to/python

# Coverage ratchet: single source of truth for the fail-under threshold.
# WAVE6-COVERAGE: raise to 85% by Wave 6 (see docs/PROGRAM_PLAN.md).
COVERAGE_THRESHOLD ?= 70

PYTHON ?= $(shell if [ -x venv/bin/python ]; then echo venv/bin/python; \
	elif [ -x .venv/bin/python ]; then echo .venv/bin/python; \
	else echo python3; fi)
PYTEST ?= $(PYTHON) -m pytest
PIP ?= $(PYTHON) -m pip
FLWR ?= $(dir $(PYTHON))flwr

.DEFAULT_GOAL := help

.PHONY: help install install-dev install-causal lint format typecheck test test-unit flwr-guard notebooks kernel \
	test-integration test-slow coverage data train fl-all evaluate serve \
	docker-build docker-run docker-test compose-up compose-down ci clean \
	_check_causal_deps

help: ## Show this help message
	@grep -E '^[a-zA-Z_-]+:.*## ' $(MAKEFILE_LIST) | \
		awk 'BEGIN {FS = ":.*## "}; {printf "  \033[36m%-20s\033[0m %s\n", $$1, $$2}'

install: ## Create venv and install the package
	$(PYTHON) -m pip install -e .

install-dev: ## Install package with dev extras (tests, linters)
	$(PIP) install -e ".[dev,causal]"

install-causal: ## Install causal inference + transportability dependencies
	$(PIP) install -e ".[causal]"

_check_causal_deps: ## Verify statsmodels and pgmpy are importable (fails with actionable message)
	@$(PYTHON) -c "import statsmodels; import pgmpy; from statsmodels.api import Logit, OLS; from pgmpy.base import DAG" 2>/dev/null || { \
		echo "ERROR: causal dependencies not installed." >&2; \
		echo "  Run: make install-causal" >&2; \
		exit 1; \
	}

lint: ## Run ruff + black --check + isort --check
	$(PYTHON) -m ruff check src/insurance_fl tests
	$(PYTHON) -m black --check src/insurance_fl tests
	$(PYTHON) -m isort --check-only src/insurance_fl tests

format: ## Auto-format with black + isort + ruff
	$(PYTHON) -m black src/insurance_fl tests
	$(PYTHON) -m isort src/insurance_fl tests
	$(PYTHON) -m ruff check --fix src/insurance_fl tests

typecheck: ## Run mypy over the new package (MYPY_OPTS for e.g. --cache-dir)
	$(PYTHON) -m mypy $(MYPY_OPTS) src/insurance_fl

test: ## Run the full test suite (excludes slow)
	$(PYTEST) -q -m "not slow"

test-unit: ## Run unit tests only
	$(PYTEST) -q tests/unit tests/test_structure.py

test-integration: ## Run integration tests only
	$(PYTEST) -q -m integration

test-slow: ## Run slow tests (cross-process determinism, full generation)
	$(PYTEST) -q -m slow

notebooks: ## Execute all notebooks in notebooks/
	$(PYTHON) scripts/run_notebooks.py

kernel: ## Register the repo venv as the 'insurance-fl' Jupyter kernel for notebook validation
	$(PYTHON) -m ipykernel install --user --name insurance-fl --display-name "insurance-fl (venv)"

flwr-guard: ## Run 1-round Flower smoke and fail if dp backend unavailable
	@tmp=$$(mktemp); \
	KMP_DUPLICATE_LIB_OK=True OMP_NUM_THREADS=1 $(FLWR) run . --stream --run-config num-server-rounds=1 > $$tmp 2>&1; rc=$$?; \
	if grep -qi "backend not available" $$tmp; then \
		echo "ERROR: dp-accounting backend not available in flwr run" >&2; \
		cat $$tmp >&2; \
		rm -f $$tmp; \
		exit 1; \
	fi; \
	rm -f $$tmp; \
	exit $$rc

coverage: ## Run tests with branch coverage (fail under 85%)
	$(PYTEST) -q --cov=insurance_fl --cov-branch --cov-report=term-missing \
		--cov-fail-under=$(COVERAGE_THRESHOLD)

data: ## Generate synthetic datasets into data/processed
	$(PYTHON) scripts/generate_dataset.py

# --- Causal inference targets (D2, D3) ---

dag-validate: _check_causal_deps data ## Validate the causal DAG (docs/causal/dag.yaml)
	$(PYTHON) -m insurance_fl.causal.dag_check

causal-estimate: _check_causal_deps data ## Estimate the default causal estimand (backdoor + DR)
	$(PYTHON) scripts/causal_estimate.py

causal-refute: _check_causal_deps data ## Run refutation tests: make causal-refute ESTIMAND=<name>
	$(PYTHON) scripts/causal_refute.py --estimand $(or $(ESTIMAND),has_claims__claim_risk_score)

causal-all: _check_causal_deps data dag-validate causal-estimate causal-refute ## Run all causal steps on default estimand
	@echo "=== CAUSAL ALL COMPLETE ==="

# --- Transportability targets (D4) ---

transport: _check_causal_deps data ## Transport outcome model: make transport SOURCE=de,fr TARGET=ch TASK=claim_risk_score
	$(PYTHON) scripts/transport.py --source-clients $(SOURCE) --target-client $(TARGET) --task $(TASK)

transport-all: _check_causal_deps data ## Run all-pairs transport matrix (4 clients x 3 targets = 12 pairs)
	$(PYTHON) scripts/transport_all.py

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

ci: ## What CI runs locally: lint, typecheck, tests, coverage, flwr smoke
	$(MAKE) lint
	$(MAKE) typecheck
	$(MAKE) test
	$(MAKE) coverage
	$(MAKE) flwr-guard

clean: ## Remove caches and build artifacts
	rm -rf .pytest_cache .mypy_cache .ruff_cache .coverage htmlcov dist build
	find . -name '__pycache__' -type d -prune -exec rm -rf {} + 2>/dev/null || true
	find . -name '._*' -not -path './.git/*' -delete 2>/dev/null || true
