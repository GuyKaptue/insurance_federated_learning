# Test Matrix

Strict mirror: `src/insurance_fl/<pkg>/<module>.py` → `tests/unit/<pkg>/test_<module>.py`.
Enforced by `tests/test_structure.py` (fails if any source module lacks its test
or any test lacks its source module). `__init__.py` files are excluded.

| Source module | Test module | Status |
|---|---|---|
| `src/insurance_fl/__init__.py` | *(excluded — package plumbing)* | n/a |
| `src/insurance_fl/config/__init__.py` | *(excluded)* | n/a |
| `src/insurance_fl/config/loader.py` | `tests/unit/config/test_loader.py` | ✅ green |
| `src/insurance_fl/data/__init__.py` | *(excluded)* | n/a |
| `src/insurance_fl/data/country_profiles.py` | `tests/unit/data/test_country_profiles.py` | ✅ green |
| `src/insurance_fl/data/generator.py` | `tests/unit/data/test_generator.py` | ✅ green (incl. cross-process determinism, `@slow`) |
| `src/insurance_fl/data/partition.py` | `tests/unit/data/test_partition.py` | ✅ green (BY_CUSTOMER_ID, BY_GROUP_ID, get_partition) |
| `src/insurance_fl/data/preprocess.py` | `tests/unit/data/test_preprocess.py` | ✅ green |
| `src/insurance_fl/data/quality.py` | `tests/unit/data/test_quality.py` | ✅ green |
| `src/insurance_fl/models/__init__.py` | *(excluded)* | n/a |
| `src/insurance_fl/models/multi_task_nn.py` | `tests/unit/models/test_multi_task_nn.py` | ✅ green |
| *`src/insurance_fl/py.typed`* | *(not Python — typing marker)* | n/a |

Cross-cutting: `tests/test_structure.py` — mirror enforcement.

## Makefile ↔ test mapping

| Makefile target | Command | Result (measured) |
|---|---|---|
| `make test` | `pytest -q -m "not slow"` | 270 passed, 1 deselected |
| `make test-unit` | `pytest -q tests/unit tests/test_structure.py` | 270 passed |
| `make test-slow` | `pytest -q -m slow` | 1 passed, 270 deselected |
| `make coverage` | `pytest --cov=insurance_fl --cov-branch --cov-fail-under=$(COVERAGE_THRESHOLD)` | 270 passed, **73.98% ≥ 70%** |
| `make ci` | `make lint && make typecheck && make test && make coverage && make flwr-guard` | 262 passed, 74.13% ≥ 70%, flwr-guard 0 backend warnings |

## Coverage (measured, `make coverage`)

| Module | Coverage |
|---|---|
| `config/loader.py` | 94% |
| `data/country_profiles.py` | 100% |
| `data/generator.py` | 91% |
| `data/partition.py` | 86% |
| `data/readiness.py` | 80% |
| `data/preprocess.py` | 94% |
| `data/quality.py` | 98% |
| `models/multi_task_nn.py` | 98% |
| **TOTAL (branch)** | **94.58%** (66 passed, 85% enforced) |
