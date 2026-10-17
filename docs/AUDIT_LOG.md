# AUDIT LOG — Insurance Federated Learning Platform

## Phase 0 — Orientation (Read-Only)

**Date:** 2026-10-16
**Engineer:** Cline (AI coding agent)
**Repository:** `/Volumes/Intenso/25.09.2026/fl/insurance_federated_learning`

---

### Environment

| Tool | Version |
|------|---------|
| OS | Darwin 25.5.0 (macOS), arm64 |
| Python | 3.13.10 (`/opt/homebrew/opt/python@3.13/bin/python3.13`) |
| pip | 25.3 |
| Docker | 29.4.3, build 055a478 |
| Docker Compose | v5.1.3 |

---

### Git Inventory

**Tracked files (23):**

| File | Purpose | Has Test | Status |
|------|---------|----------|--------|
| `.gitignore` | Git ignore patterns | N/A | Works |
| `European_Insurance_Federated_Learning_Project.md` | Project description document | No | Placeholder/hypothesis |
| `README.md` | User-facing documentation | No | Has broken fences, placeholder badges |
| `data/processed/european_insurance_federated/dataset_info.json` | Dataset metadata (file structure) | No | Works |
| `data/processed/european_insurance_federated/metadata.json` | Dataset metadata (client sizes, targets, stats) | No | Works |
| `docs/EIFD.md` | European Insurance Federated Dataset documentation | No | Works |
| `outputs/reports/analysis/centralized/quality_report_centralized.json` | Centralized quality report | No | Works (generated artifact) |
| `outputs/reports/analysis/centralized/quality_summary_centralized.json` | Centralized quality summary | No | Works (generated artifact) |
| `outputs/reports/analysis/clients/belgium_ageas/quality_report_belgium_ageas.json` | Belgium quality report | No | Works (generated artifact) |
| `outputs/reports/analysis/clients/france_axa/quality_report_france_axa.json` | France quality report | No | Works (generated artifact) |
| `outputs/reports/analysis/clients/germany_allianz/quality_report_germany_allianz.json` | Germany quality report | No | Works (generated artifact) |
| `outputs/reports/analysis/clients/switzerland_helvetia/quality_report_switzerland_helvetia.json` | Switzerland quality report | No | Works (generated artifact) |
| `outputs/reports/analysis/summary_report.json` | Analysis summary JSON | No | Works (generated artifact) |
| `outputs/reports/analysis/summary_report.md` | Analysis summary MD | No | Works (generated artifact) |
| `pyproject.toml` | Packaging + Flower config | No | Works (see gaps below) |
| `requirements.txt` | Runtime dependencies | No | Works |
| `scripts/__init__.py` | Scripts package init | No | Empty |
| `scripts/analyze_data.py` | Data analysis script | No | Works |
| `scripts/generate_dataset.py` | Dataset generation CLI | No | Works |
| `scripts/generator.sh` | Shell wrapper for generation + analysis | No | Works |
| `src/__init__.py` | src package init | No | Empty |
**Untracked files/directories (11):**

| File/Dir | Purpose | Has Test | Status |
|----------|---------|----------|--------|
| `PROMPT.md` | Master prompt (this project's instruction file) | N/A | Empty (0 bytes) |
| `catboost_info/` | CatBoost training artifacts | N/A | Generated artifact |
| `cli.py` | Root-level unified CLI (thin wrapper delegating to scripts/) | No | Works (large, 675 lines) |
| `data/processed/european_insurance_federated/pyproject.toml` | Stray pyproject inside data dir | No | Stray/misplaced |
| `data/processed/european_insurance_federated/src/` | Stray src directory inside data dir | No | Stray/misplaced |
| `docs/HLCMNN.md` | Additional documentation | No | Untracked |
| `scripts/cli.py` | Duplicate of root cli.py | No | Duplicate |
| `scripts/run.py` | Full training/audit/explainability pipeline | No | Works (1532 lines) |
| `scripts/run_distributed.py` | Distributed FL runner (server/client/all roles) | No | Works (1143 lines) |
| `src/core/configs/config.py` | Centralized Config dataclass (1145 lines) | No | Works |
| `src/core/explainability/` | SHAP, audit, persona explainability | No | Works |
| `src/core/federated/` | Flower SuperLink/SuperNode FL stack | No | Works |

**Python source files (38 total):**

```
cli.py
scripts/cli.py
scripts/analyze_data.py
scripts/generate_dataset.py
scripts/run.py
scripts/run_distributed.py
src/__init__.py
src/core/__init__.py
src/core/configs/__init__.py
src/core/configs/config.py
src/core/configs/paths.py
src/core/explainability/__init__.py
src/core/explainability/audit.py
src/core/explainability/persona.py
src/core/explainability/shap_explainer.py
src/core/explainability/utils.py
src/core/federated/aggregators/__init__.py
src/core/federated/aggregators/hybrid_aggregator.py
src/core/federated/aggregators/nn_aggregator.py
src/core/federated/aggregators/tree_aggregator.py
src/core/federated/client_app.py
src/core/federated/config.py
src/core/federated/serialization.py
src/core/federated/server_app.py
src/core/federated/strategies.py

### README vs Disk — Gap List

| README Claim | Disk Reality | Gap |
|-------------|-------------|-----|
| `pip install insurance-federated-learning` (PyPI) | Not published to PyPI | Missing package publication |
| `pip install -e ".[dev]"` | `pyproject.toml` has `[project.optional-dependencies] dev` but no tests exist | Partially works; dev deps install but nothing to test |
| `insurance-fl-generate` console script | `pyproject.toml` has no `[project.scripts]`; `setup.py` (referenced in README) does not exist | **Missing** — no console scripts defined |
| `insurance-fl-train --config config/federated_config.yaml` | `config/federated_config.yaml` does not exist | **Missing** config file |
| `insurance-fl-serve --host 0.0.0.0 --port 8000` | No `deploy_api.py` or FastAPI server script exists | **Missing** |
| `docs/installation.md`, `docs/usage.md`, `docs/api.md` | Do not exist | **Missing** documentation |
| `CONTRIBUTING.md` | Does not exist | **Missing** |
| `LICENSE` | Does not exist (pyproject declares MIT) | **Missing** |
| CI badge pointing to `.github/workflows/ci.yml` | Directory does not exist | **Missing** |
| codecov badge | No coverage configuration exists | **Missing** |
| PyPI badge, Zenodo DOI badge, ReadTheDocs badge | Services do not exist for this project | **Placeholder/invalid** |
| `docker/Dockerfile.prod` | Does not exist | **Missing** |
| `kubernetes/` manifests | Do not exist | **Missing** |
| `setup.py` console entry points `src.scripts.generate_dataset:main` | Module path `src.scripts` does not exist as a package | **Path mismatch** |
| Python `>=3.11,<3.13` (pyproject) vs `>=3.10,<3.13` (README setup.py) | Range mismatch between files | **Version mismatch** |
| README badge says Python 3.11+ | pyproject says `<3.13` (excludes 3.13) | **Version mismatch** |
| System Python is 3.13.10 | pyproject requires `<3.13` | **Incompatible with local Python** |

### Key Architectural Observations

1. **Two CLI entry points:** Root `cli.py` (675 lines) and `scripts/cli.py` (duplicate). Root `cli.py` is not importable as a package module (no `__init__.py` at root for it).

2. **Package layout ambiguity:** `pyproject.toml` uses `[tool.setuptools] packages = {find = {where = ["src"]}}` which finds `src.core`, `src.core.configs`, etc. But the README's `setup.py` references `src.scripts.*` which doesn't exist. No `src/insurance_fl/` top-level package exists.

3. **Python 3.13 incompatibility:** `pyproject.toml` requires `python_requires = ">=3.11,<3.13"` but local Python is 3.13.10. A clean `pip install -e ".[dev]"` will fail.

4. **No tests exist anywhere.** Zero test files in the repository.

5. **No Docker, no Makefile, no CI/CD.** None of the Phase 4-6 infrastructure exists.

6. **Data already generated:** Parquet files exist for 4 clients + centralized, with train/val/test/site splits. Total 30,000 samples.

7. **Serialization security:** `src/core/federated/utils.py` has pickle-based serialization with explicit security warnings. `src/core/federated/serialization.py` provides a safe pickle-free alternative using native library formats.

8. **Flower integration:** `pyproject.toml` defines `[tool.flwr.app.components]` pointing to `src.core.federated.server_app:app` and `src.core.federated.client_app:app`. These use the modern Flower SuperLink/SuperNode API.

### Suspects from Master Prompt — Preliminary Status

| # | Suspect | Preliminary Status | Evidence Needed |
|---|---------|-------------------|-----------------|
| 1 | tensorflow-federated pin | REFUTED (not in requirements.txt or pyproject.toml) | N/A — not present |
| 2 | setup.py console script path mismatch | CONFIRMED | `setup.py` doesn't exist; pyproject has no `[project.scripts]` |
| 3 | Python version mismatch | CONFIRMED | pyproject: `>=3.11,<3.13`; local: 3.13.10; README badge: 3.11+ |
| 4 | Non-deterministic `hash()` seeding | To verify | Need to inspect data_generator.py |
| 5 | `_truncated_normal` not actually truncated | To verify | Need to inspect data_generator.py |
| 6 | `dirichlet_partition` can go negative | To verify | Need to inspect data_generator.py |
| 7 | `inject_quality_issues` dtype issues | To verify | Need to inspect data_generator.py |
| 8 | Quality injection in CLI not generator | To verify | Need to inspect cli.py / scripts/ |
| 9 | Test assertions not matching generator output | To verify | No tests exist yet |
| 10 | Segment probabilities not income-dependent | To verify | Need to inspect data_generator.py |

---

## Phase 1 — Error Research (Reproduce Before Fixing)

**Date:** 2026-10-16

### Install Reproduction

Created fresh venv at `/Volumes/Intenso/fl_audit_venv` (Python 3.13.10).
### Error Register

| ID | Symptom (verbatim) | Root Cause | Planned Fix | Guarding Test | Status |
|----|-------------------|------------|-------------|---------------|--------|
| E1 | `pip install -e ".[dev]"` fails: `requires a different Python: 3.13.10 not in '<3.13,>=3.11'` | `pyproject.toml` requires-python excludes 3.13; local Python is 3.13.10 | Change to `>=3.11,<3.14`; align classifiers, CI matrix, Dockerfile, README badge | `test_python_version_compat` — verify `python_requires` includes current Python | CONFIRMED |
| E2 | No console scripts (`insurance-fl-generate`, etc.) defined | `pyproject.toml` has no `[project.scripts]`; `setup.py` (referenced in README) does not exist | Add `[project.scripts]` to pyproject.toml pointing to `scripts.*` modules; remove setup.py references | `test_console_scripts` — verify each console script `--help` works | CONFIRMED |
| E3 | `import insurance_fl` fails: `No module named 'insurance_fl'` | No `src/insurance_fl/` top-level package; layout is `src.core`, `src.synthetic` | Restructure to `src/insurance_fl/` layout with pyproject.toml as single packaging file | `test_import_insurance_fl` — verify `import insurance_fl` works | CONFIRMED |
| E4 | No tests exist anywhere | `tests/` directory does not exist | Create full test suite with one test module per source module | `tests/test_structure.py` — enforces mirror rule | CONFIRMED |
| E5 | Label leakage: `is_high_risk` computed from `claim_risk_score` (also a target) | `_generate_targets` line 505-508 uses `claim_risk / 100 * 1.5` in high_risk logit | Remove target-derived features from other target computations; document leakage-safe target definitions | `test_no_label_leakage` — verify targets don't depend on other targets | CONFIRMED |
| E6 | `_add_data_quality_issues` assigns NaN to integer columns | Line 573: `df.loc[mask, field] = np.nan` on columns that may be integer dtype | Cast columns to float before injection; verify dtypes after injection | `test_quality_injection_dtypes` — verify no integer column gets NaN without upcast | CONFIRMED |
| E7 | `_add_data_quality_issues` sets `driving_experience = age + 5` (inconsistent) | Line 586: driving_experience can exceed age | Fix to consistent value | `test_quality_injection_consistency` — verify driving_experience <= age | CONFIRMED |
| E8 | `src.core.federated/__init__.py` missing | Directory exists but no `__init__.py` | Add `__init__.py` exporting key symbols | `test_federated_init` — verify `import src.core.federated` works | CONFIRMED |
| E9 | Duplicate cli.py (root `cli.py` and `scripts/cli.py`) | Two files with same purpose | Consolidate to single CLI; remove duplicate | `test_no_duplicate_cli` — verify only one CLI entry point | CONFIRMED |
| E10 | `config/federated_config.yaml` referenced in README but doesn't exist | No YAML config file | Create `config/federated_config.yaml` with validated loader | `test_config_loader` — verify YAML config loads and validates | CONFIRMED |
| E11 | `deploy_api.py` / FastAPI server referenced in README but doesn't exist | No `scripts/deploy_api.py` | Create FastAPI deployment script with `/health`, `/predict/<task>`, `/metrics` | `test_api_health` — verify TestClient `/health` returns 200 | CONFIRMED |
| E12 | No Makefile, Dockerfile, docker-compose.yml, .github/workflows | Missing Phase 4-6 infrastructure | Create all infrastructure files | `test_makefile_targets` — verify `make help` lists all targets | CONFIRMED |
| E13 | No LICENSE file (pyproject declares MIT) | LICENSE missing | Add MIT LICENSE file | N/A (file existence) | CONFIRMED |
| E14 | README has broken code fence + placeholder badges | README lines ~73-75 have stray fence; badges point to non-existent services | Fix README structure; replace placeholder badges with real ones | `test_readme_structure` — basic validation | CONFIRMED |
| E15 | `docs/HLCMNN.md` untracked, `docs/installation.md`, `docs/usage.md`, `docs/api.md` missing | Documentation gaps | Create missing docs; track HLCMNN.md | N/A | CONFIRMED |
| E16 | `data/processed/european_insurance_federated/pyproject.toml` and `src/` are stray files inside data dir | Misplaced files | Remove stray files from data directory | `test_no_stray_files_in_data` | CONFIRMED |
| E17 | `PROMPT.md` is empty (0 bytes) | Empty instruction file | Remove or populate | N/A | CONFIRMED |

**Command:** `pip install -e ".[dev]"`
**Output:** `ERROR: Package 'insurance-federated-learning' requires a different Python: 3.13.10 not in '<3.13,>=3.11'`

### Suspect Confirmation Summary

| # | Suspect | Status | Evidence |
|---|---------|--------|----------|
| 1 | tensorflow-federated pin | REFUTED | Not in requirements.txt or pyproject.toml |
| 2 | setup.py console script path mismatch | CONFIRMED | setup.py doesn't exist; pyproject has no [project.scripts] |
| 3 | Python version mismatch | CONFIRMED | `pip install` fails with Python 3.13.10 not in `<3.13` |
| 4 | Non-deterministic `hash()` seeding | REFUTED | No `hash()` usage found anywhere in codebase |
| 5 | `_truncated_normal` not actually truncated | REFUTED | No `_truncated_normal` function found in codebase |
| 6 | `dirichlet_partition` can go negative | REFUTED | No dirichlet partitioning; uses `train_test_split` |
| 7 | `inject_quality_issues` dtype issues | CONFIRMED | `_add_data_quality_issues` assigns NaN to columns (E6) |
| 8 | Quality injection in CLI not generator | REFUTED | Quality injection IS in generator (`_add_data_quality_issues`) |
| 9 | Test assertions not matching generator output | N/A | No tests exist yet |
| 10 | Segment probabilities not income-dependent | PARTIAL | Income IS used in segment scoring (lines 519-528) |
| 11 | Mixed currencies without normalization | CONFIRMED (by design) | Currency column exists (EUR/CHF); no normalization column |
| 12 | Fabricated evidence in attached docs | CONFIRMED | Docs contain placeholder KL values, p-values, timings |
| 13 | Broken README code fence | CONFIRMED | README has stray fence + setup.py section |
| 14 | Import hygiene / missing `__init__.py` | CONFIRMED | `import insurance_fl` fails; `src.core.federated/__init__.py` missing |

### Additional Issues Found (not in master prompt suspect list)

| ID | Description | Location |
|----|-------------|----------|
| A1 | Label leakage: `is_high_risk` uses `claim_risk_score` (which is itself a target) | `data_generator.py:505-508` |
| A2 | `driving_experience = age + 5` is logically inconsistent (experience > age) | `data_generator.py:586` |
| A3 | `renewal_probability` and `risk_probability` columns added to targets but not in schema | `data_generator.py:502,516` |
| A4 | `segment_probabilities` stored as list in DataFrame cell (not vectorized) | `data_generator.py:560` |
| A5 | Python 3.13.10 incompatibility with many dependency version pins (numpy<2.0, pandas<2.3, etc.) | `pyproject.toml` lines 40-41 |

### Gate 1 Status: PASS

Error register written to `docs/AUDIT_LOG.md`. Every suspect from the master prompt is marked CONFIRMED or REFUTED with evidence. No files modified yet.

**Next:** Phase 2 — Restructure and fix. Will adopt `src/insurance_fl/` layout, fix packaging, implement missing modules, and add tests alongside each fix.
**Root cause:** `pyproject.toml` line 21: `requires-python = ">=3.11,<3.13"` excludes Python 3.13.

**Workaround:** `pip install -e . --ignore-requires-python --no-deps` succeeded, then installed runtime deps individually.

**Command:** `python -c "import insurance_fl"`
**Output:** `ModuleNotFoundError: No module named 'insurance_fl'`

**Root cause:** No `src/insurance_fl/` package exists. Layout is `src.core`, `src.synthetic` under `src/`.

**Command:** `pytest -q`
**Output:** pytest not found (dev deps not installed due to Python version constraint)
| 11 | Mixed currencies without normalization | To verify | Need to inspect data schema |
| 12 | Fabricated evidence in attached docs | CONFIRMED (hypothesis) | Docs contain placeholder KL values, p-values |
| 13 | Broken README code fence | CONFIRMED | README has stray fence + setup.py section |
| 14 | Import hygiene / missing `__init__.py` | PARTIAL | Most `__init__.py` exist; `src.core.federated/__init__.py` missing |

### Gate 0 Status: PASS

Inventory and gap list written to `docs/AUDIT_LOG.md`. No files modified yet.

**Next:** Phase 1 — Error research (reproduce before fixing). Will create a fresh venv, attempt `pip install -e ".[dev]"`, run console scripts, `pytest`, and confirm/refute each suspect with evidence.
src/core/federated/utils.py
src/core/loader/__init__.py
src/core/loader/data_loader.py
src/core/loader/data_preprocessor.py
src/core/loader/data_quality_analyzer.py
src/core/loader/data_schema.py
src/core/models/__init__.py
src/core/models/ensemble_model.py
src/core/models/hybrid_model.py
src/core/models/multi_task_nn.py
src/synthetic/__init__.py
src/synthetic/data_generator.py
```

**Missing entirely (no tests, no Makefile, no Docker, no CI):**
- `tests/` directory — does not exist
- `Makefile` — does not exist
- `Dockerfile` — does not exist
- `docker-compose.yml` — does not exist
- `.github/workflows/` — does not exist
- `LICENSE` — does not exist (pyproject claims MIT)
- `CONTRIBUTING.md` — does not exist
- `.editorconfig` — does not exist
- `.dockerignore` — does not exist
- `py.typed` — does not exist
- `setup.py` — referenced in README but does not exist on disk
- `config/federated_config.yaml` — referenced in README but does not exist
- `docs/installation.md`, `docs/usage.md`, `docs/api.md` — referenced in README but do not exist
- `docker/Dockerfile.prod`, `kubernetes/`, `monitoring/` — referenced in docs but do not exist
| `src/core/__init__.py` | core package init | No | Empty |
| `src/core/configs/__init__.py` | configs package init | No | Exports paths |
| `src/core/configs/paths.py` | Project path management | No | Works |
| `src/core/loader/__init__.py` | loader package init | No | Exports loader classes |
| `src/core/loader/data_loader.py` | DataLoader + DatasetSplit | No | Works |
| `src/core/loader/data_preprocessor.py` | DataPreprocessor | No | Works |
| `src/core/loader/data_quality_analyzer.py` | DataQualityAnalyzer + QualityMetrics | No | Works |
| `src/core/loader/data_schema.py` | DataSchema + ColumnMetadata + get_insurance_schema() | No | Works |
| `src/synthetic/__init__.py` | synthetic package init | No | Exports DataGenerator |

---

## Phase 2 (continued) – Repair corrupted modules, packaging, config

### Session resume after disk-full block (2026-10-16)

**Context:** previous session ended mid-Phase 2 with two modules corrupted by
interleaved edits (failed `compileall`) and the system disk at 96% full
(ENOSPC blocked all tool calls).

**Issue register additions (symptom → root cause → fix → guarding test):**

| ID | Symptom (verbatim) | Root cause | Fix | Guarding test |
|---|---|---|---|---|
| S-01 | `SyntaxError: source code string cannot contain null bytes` on `src/insurance_fl/data/._*.py` | macOS AppleDouble `._*` junk files in source tree | `find src -name '._*' -delete` (+ Makefile `clean` target removes them) | `compileall` runs in `make ci` path via lint/mypy |
| S-02 | `data/generator.py`, `data/preprocess.py` failed `compileall` (interleaved method bodies from interrupted edits) | Prior session wrote overlapping fragments | Full clean rewrite of both modules (deterministic per-client `crc32` seeding, vectorized targets, leakage-safe targets, fit-on-train preprocessor) | `tests/unit/data/test_generator.py`, `tests/unit/data/test_preprocess.py` |
| S-03 | `pandas.errors.IntCastingNaNError: Cannot convert non-finite values (NA or inf) to integer` at `generator.py:427` (only at N≥1000, `make data` gate) | Tenure-injection cast `cap.astype("int64")` where `cap = age - 18` and `age` contains injected NaNs; 1% mask ∩ NaN-age rows didn't trigger at test N=400 | Cast to `float64` (column already upcast), skip empty mask | `test_tenure_injection_with_nan_age_does_not_crash` (N=3000, forced NaN overlap) |
| S-04 | `dirichlet_partition(..., min_samples_per_client=5)` left clients below minimum | Redundancy loop transferred 1 sample then `break` | `while`-loop until minimum met or no donor surplus | `test_min_samples_per_client_honored` (partition test) |
| S-05 | `assert m.column_types["d"] in ("object","string")` → got `'str'` | pandas 3.0 reports string dtype as `str` | Assertion accepts `("object","string","str")` with comment | same test |
| S-06 | `test_quality_injection...` failed: `assert np.False_` on `tenure <= age - 18` | NaN comparisons are False → `.all()` fails on unknown rows | `dropna()` before invariant check (test bug, not source) | same test |
| S-07 | **`.gitignore:195 models/` silently ignored `src/insurance_fl/models/`** — whole model package was untracked | Overbroad artifact pattern matches any depth | Anchored to `/models/` (repo root only) | `git check-ignore` shows NOT_IGNORED; `tests/test_structure.py` walks the dir |
| S-08 | 9× `PydanticDeprecatedSince20: Pydantic V1 style @validator` warnings (removed in Pydantic V3) | V1-style validators while pydantic pinned `>=2,<3` | Migrated to `@field_validator` + `@classmethod` (`info.data` for cross-field) | Full suite runs with `-W error::DeprecationWarning` → clean |
| S-09 | `make test-integration` exit 5 (`66 deselected`) | No integration tests existed | Added `tests/integration/test_e2e_pipeline.py` (generate→preprocess→train→evaluate, + full-pipeline determinism) | `make test-integration` → 2 passed |

### Commands run and real outputs (this session)

```
$ make help PYTHON=/Volumes/Intenso/fl_audit_venv/bin/python
  ... 22 targets listed ...

$ make test PYTHON=/Volumes/Intenso/fl_audit_venv/bin/python
65 passed, 1 deselected in 11.61s

$ make test-unit PYTHON=/Volumes/Intenso/fl_audit_venv/bin/python
66 passed, 8 warnings in 15.54s

$ make test-integration PYTHON=/Volumes/Intenso/fl_audit_venv/bin/python
2 passed, 66 deselected, 9 warnings in 12.16s

$ make test-slow PYTHON=/Volumes/Intenso/fl_audit_venv/bin/python
1 passed, 65 deselected, 8 warnings in 4.51s

$ make coverage PYTHON=/Volumes/Intenso/fl_audit_venv/bin/python
TOTAL   749   20   192   31   95%
Required test coverage of 85% reached. Total coverage: 94.58%
66 passed in 17.06s

$ make lint PYTHON=/Volumes/Intenso/fl_audit_venv/bin/python
All checks passed!            (ruff)
All done! 17 files would be left unchanged.   (black)
isort --check-only: clean

$ make typecheck PYTHON=/Volumes/Intenso/fl_audit_venv/bin/python
Success: no issues found in 11 source files

$ make ci PYTHON=/Volumes/Intenso/fl_audit_venv/bin/python
All checks passed!
Success: no issues found in 11 source files
67 passed, 1 deselected in 12.97s
Required test coverage of 85% reached. Total coverage: 94.73%
68 passed in 10.81s

$ make data PYTHON=/Volumes/Intenso/fl_audit_venv/bin/python
Centralized dataset: 3,000 total samples
germany_allianz: 5000 | france_axa: 5000 | belgium_ageas: 5000 | switzerland_helvetia: 5000
Done.

$ python -m compileall -q src/insurance_fl   → COMPILE_ALL_OK (after ._ cleanup)

$ 3× flakiness check:
68 passed in 10.04s
68 passed in 8.53s
68 passed in 7.23s
```

### Phase status

- **Gate 2:** PASS (modules import; `scripts/generate_dataset.py --help` responds; config validates)
- **Gate 3 (partial):** PASS — every current module mirrored, `test_structure.py` green (both directions), 68 tests green 3×, coverage 94.7% ≥ 85%. Integration dir seeded with e2e pipeline (FL-simulation tests deferred to the fl/ package phase).
- **Gate 4 (partial):** PASS for `help`, `ci`, `data`, `test*`, `coverage`, `lint`, `typecheck`. Targets `train`/`fl-all`/`evaluate`/`serve` depend on scripts not yet implemented (next phases). Docker targets not yet validated (Docker availability TBD).
- `docs/TEST_MATRIX.md` created and kept in sync with measured numbers.

---

## Layer 1 / Task 0 — Flower App baseline federation (2026-10-17, branch `feat/task0-baseline-federation`)

**Scope:** Task_Specification Task 0 + suspect #10 (`src.core.federated.*` phantom paths) fix.

**Build evidence:**
- New modules: `src/insurance_fl/fl/{task,client_app,server_app,strategy,privacy,stacking}.py` (last three **empty-by-design**, guarded by tests until Tasks 3/4/6–9 fill them).
- `pyproject.toml [tool.flwr.app.components]` → `insurance_fl.fl.{server_app,client_app}:app` (was `src.core.federated.*`, never existed).
- flwr pin `>=1.29.0,<2.0.0` (D-ORCH-004; spec asked ≥1.30 — deviation recorded).
- Mirrored tests per `test_structure.py` rule: `tests/unit/fl/test_{task,client_app,server_app,strategy,privacy,stacking}.py` (orphan `test_fl_app.py` removed).
- Makefile `typecheck` accepts `MYPY_OPTS` (cache-dir override; volume-space incident D-ORCH-006).

**Validation run (all fresh this session):**
| Check | Command | Result |
|---|---|---|
| Lint | `make lint PYTHON=python3` (ruff+black+isort) | PASS |
| Types | `mypy --cache-dir=/tmp/mypy_layer1_cache src/insurance_fl` | PASS, 0 issues / 25 files |
| Tests | `make test PYTHON=python3` | **90 passed**, 1 deselected |
| Coverage | `make coverage PYTHON=python3` | **91 passed, 93.47% ≥ 85% required** |
| Full CI | `make ci PYTHON=python3 MYPY_OPTS=--cache-dir=/tmp/mypy_layer1_cache` | **exit 0** |
| Data prep | `python -m insurance_fl.fl.task --prepare` (512/client, seed 42, DQ injection OFF) | 4×512 generated, 307 train rows/client |
| **Federation** | `flwr run . --stream --run-config 'num-server-rounds=2 local-epochs=1'` | **exit 0** — 2 rounds in 3.30 s, 4/4 clients fit + 2/4 eval per round, **loss 19 275 381 → 17 328 602** |
| Evidence file | `docs/evidence/layer1_flwr_run_2rounds.log` | committed |

**Incidents resolved (details D-ORCH-006):**
1. SuperLink crash on start: stale `~/.flwr/local-superlink/state.db` (541 MB) with unknown alembic revision `dfdbe2a8f936` (foreign flwr version) → backed up `state.db.pre-1.29-backup-20261017`, fresh DB, run green.
2. exFAT volume reached 100% mid-CI (`No space left on device`) → cleared gitignored tool caches (`.mypy_cache` 1.9 G, `.ruff_cache`, `.hypothesis`, `__pycache__` — all regenerable, all listed in `make clean`); volume at **1.9 Gi free**; mypy cache redirected to `/tmp`. **`venv/` = 26 G (broken torch) remains the space lever — maintainer decision pending.** No user data deleted.