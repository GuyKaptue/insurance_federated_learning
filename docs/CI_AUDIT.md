# CI/CD Audit (Phase 0)

**Date:** 2026-11-19 (local audit, read-only inventory + measured timings)  
**Repository:** `/Users/guykaptue/Downloads/insurance_federated_learning`  
**Branch/HEAD:** `feat/task0-baseline-federation` @ `333a4a2` (9 commits ahead of `origin/feat/task0-baseline-federation`)  
**Scope:** inventory, measured local runtimes, verified Flower 1.31 CLI facts, gap table and phase plan. No placeholder steps proposed for CI; items that cannot be made real are listed as gaps/not done.

## 1. Inventory (exists)

| Area | Exists | Evidence (paths / commands) | Notes |
|---|---:|---|---|
| GitHub Actions workflows | Yes (1) | `.github/workflows/ci.yml` | Single job `test`; currently unstaged rewrite in working tree not committed (audit records committed state context separately). Committed workflow installs `.[dev,causal]` in prior form? (see git history/working diff) — remote failure reported: run `44521000019`, `resolution-too-deep` at install. |
| Makefile targets | Yes | `Makefile` | `lint`, `format`, `typecheck`, `test` (`-m "not slow"`), `test-unit`, `test-integration`, `test-slow`, `coverage` (fail-under 70), `flwr-guard`, `data`, `data-check`, `notebooks`, `kernel`, `ci` (lint→typecheck→test→coverage→flwr-guard), causal/transport helpers, docker/compose stubs. |
| pyproject extras / deps | Yes | `pyproject.toml` | `requires-python >=3.11,<3.13`; core deps incl. `flwr>=1.31,<1.32`, `torch`, `dp-accounting`, `mlflow`, etc. Extras: `dev` (pytest/ruff/black/isort/mypy/nbformat/nbconvert/ipykernel/hypothesis/...), `causal` (statsmodels, pgmpy), `train/rdp/privacy` (dp-accounting), `notebooks` (viz + nbclient/ipykernel/networkx). Pytest markers: `slow`, `integration`. |
| Test layout & markers | Yes | `tests/unit/**`, `tests/integration/**`, `tests/test_structure.py`, `tests/golden/*` | Integration-marked files observed: `tests/integration/test_tls.py` (`@pytest.mark.integration`). Slow-marked: 4 tests (MIA script, generator). Feature-leakage test skip-if-data-missing. No `tests/contracts`, `tests/privacy`, `tests/fairness` directories. |
| Scripts | Yes | `scripts/` | `validate_notebooks.py`, `run_notebooks.py`, `check_dataset_signal.py`, `run_mia.py`, `dp_sweep.py`, `eval_is_high_risk.py`, causal/transport scripts, `launch_tls.py`, etc. No `scripts/ci/*`. |
| Docker / Compose / container defs | Partial | `.dockerignore` only | **No** `Dockerfile*`, `docker-compose*`, `compose*.yml`, `Containerfile*` in repo. Make targets `docker-build/run/test/compose-up/down` exist but images/manifests absent. |
| Configs | Yes | `config/federated_config.yaml` | Flower app components/run_config in `pyproject.toml`. |
| Lock/constraints files | Partial (local/uncommitted state) | `constraints/ci-py311.txt` exists in working tree (untracked) | No committed `uv.lock`/`poetry.lock`/`Pipfile.lock`. Prior CI used unconstrained editable install (`.[dev,causal]`) and hit `resolution-too-deep` (remote). |
| Dependency manager | pip/setuptools (primary) | `pyproject.toml`, `Makefile`, CI; `uv 0.13.7` installed locally | No uv/poetry lock committed; `uv` available on dev machine for potential lock generation. |
| Notebook validation | Yes | `scripts/validate_notebooks.py`, `reports/notebook_validation.json` (tracked) | Temp/in-memory execution, env capture, `NB_MODE` support in common helpers; validator runs 11 notebooks. |
| Security/audit tooling in env | Partial | local checks | `actionlint 1.7.12` available; `mypy 1.10.0` configured; `gitleaks` not installed; `pip-audit` not installed in venv. |
| Platform | Yes | local evidence | Dev: macOS arm64, `./venv` Python 3.11.16, Flower 1.31.0, Docker 29.4.3. `podman` not installed locally. CI/cloud target: Linux amd64 (per plan). |

## 2. Measured local runtimes (decision input for PR vs nightly)

Measured on 2026-11-19 in project `./venv` (macOS arm64):

| Check | Command | Result | Wall time (real) | Notes |
|---|---|---:|---:|---|
| Unit (not slow) | `pytest -q tests/unit -m "not slow"` | 250 passed, 4 deselected | **~32.0 s** (32.04s timed) | PR-suitable. |
| Integration | `pytest -q tests/integration` | 47 passed | **~8.9 s** (8.93s timed) | PR-suitable. |
| Notebook validator | `python scripts/validate_notebooks.py` | 11/11 passed | **~69.4 s** (69.36s timed) | Borderline for PR; feasible as parallel PR job or nightly depending on budget. |
| Ruff | `ruff check src/insurance_fl tests` | pass | ~0.48 s | Fast. |
| Black check | `black --check src/insurance_fl tests` | pass | ~0.78 s | Fast. |
| Isort check | `isort --check-only src/insurance_fl tests` | pass | ~0.40 s | Fast. |
| Mypy | `mypy src/insurance_fl` | pass (46 files) | ~5.45 s | Configured (`mypy==1.10.0` pinned in dev). |

**Implication:** quality + unit + integration + artifact checks fit PR budget; notebooks can run in PR as a separate timed job or be moved to nightly if total wall time becomes problematic (currently ~70s alone).

## 3. Flower 1.31 facts (verified)

Source: `./venv/bin/flwr --version` → `Flower version: 1.31.0`; CLI help inspected locally.

| Item | Verified value / command | Notes |
|---|---|---|
| `flwr run` | `flwr run [APP] [SUPERLINK]` with `--stream`, `--run-config`, `--federation`, `--federation-config` | **Current repo usage:** simulation-oriented `flwr run . --stream ...` (pyproject app components; evidence logs in `docs/evidence/`). |
| `flower-superlink` | installed (`./venv/bin/flower-superlink`) | Key flags: `--insecure`, TLS cert/key/CA (`--ssl-*`), `--serverappio-api-address` (default **0.0.0.0:9091**, deprecated alias `--simulationio-api-address`), `--fleet-api-type {grpc-rere,grpc-adapter,rest}`, `--fleet-api-address`, `--control-api-address` (default **0.0.0.0:9093**), `--simulation`, health server, auth flags. Fleet default bind address not explicitly stated in captured help text beyond option presence — **must be confirmed at implementation time** (or set explicitly in compose/scripts). |
| `flower-supernode` | installed (`./venv/bin/flower-supernode`) | Connects to SuperLink Fleet API via `--superlink`; transport `--grpc-rere|--grpc-adapter|--rest`; `--root-certificates`; `--node-config`; `--clientappio-api-address` (default **0.0.0.0:9094**); `--insecure`. |
| `flwr-serverapp` | installed | `--serverappio-api-address` default **127.0.0.1:9091**; token/token-file; TLS/root cert flags. |
| `flwr-clientapp` | installed | `--clientappio-api-address` default **127.0.0.1:9094**. |
| Known ports (from help) | **9091** ServerAppIo, **9093** Control API (SuperLink default), **9094** ClientAppIo | Fleet API port: not pinned by default text captured; `scripts/launch_tls.py` currently assumes SuperLink at `127.0.0.1:8000` (example/dry-run helper) — **port/command assumptions must be re-verified against deployment mode**. |
| `flwr-superlink` / `flwr-supernode` console scripts | not present as those exact names | Use `flower-superlink` / `flower-supernode`. |
| TLS helper | `scripts/launch_tls.py`, `generate_creds.py` | Helper prints commands with `--root-certificates`/`--ssl-*`; needs alignment with verified 1.31 deployment flags/ports before containerization. |

## 4. Gap table

| Area | Exists | Works (evidence) | Gap | Phase |
|---|---|---|---|---|
| CI workflow | Yes (`.github/workflows/ci.yml`) | **Not verified remotely for HEAD**; prior remote `44521000019` failed `resolution-too-deep`; local uncommitted changes pending | Rewrite with real independent jobs + gate, pinned action SHAs, permissions/concurrency/timeouts/caching; install required extras; no placeholders | 1 |
| Dependency install reproducibility | Partial (constraints file in WT, pyproject bounds adjusted locally uncommitted) | Local dry-run/installs improved in venv context; **Linux-generated/validated lock or constraints not yet committed/proven** | Generate/commit Linux-valid constraints or lock (uv/pip-compile on Linux), fix backtracking offender with evidence, verify in fresh Linux env | 1 |
| Quality checks | Yes (Makefile/tools) | `ruff/black/isort/mypy` pass locally | Ensure CI runs same checks; ruff config has deprecation warning (`lint.isort`) to clean up opportunistically | 1 |
| Unit/integration tests | Yes | 250 pass (unit, not slow), 47 pass (integration) locally | Wire to CI jobs with correct extras and markers | 1 |
| Privacy invariants job | Tests exist (unit/integration privacy/accounting) | Not isolated as CI job; budget-gate tests not yet implemented | Add dedicated job; include "privacy extra installed" check; extend when budget gate lands | 1 / after C |
| Artifact/provenance checks | Yes (`make data-check`, provenance tests, manifest/label-logic guard) | `data-check`/tests exist locally | Add CI job generating canonical data then running checks; ensure no reliance on committed parquets | 1 |
| Notebook validation | Yes | 11/11 in ~69s locally | Add to PR (timed) or nightly based on budget; set `NB_MODE=light` as appropriate | 1 or nightly |
| Security scanning | Partial | no `gitleaks`/`pip-audit` in env; no ignore policy files | Add real secret scan + `pip-audit` against real lock/constraints with documented ignores only if needed | 1 |
| Action pinning / supply chain | No | actions referenced by tags in committed workflow | Pin all third-party actions to full commit SHAs (resolved via `gh api`, deref annotated tags), keep version comments | 1 |
| Containerization | No | N/A | Multi-stage Dockerfile (server/client), strict `.dockerignore`, runtime-only data mounts, no baked datasets | 2 |
| 4-client Compose isolation | No | N/A | Per-client networks (client-server only), per-client RO mounts, server no data, verified Flower commands/ports, healthchecks/limits, TLS profile | 2 |
| Container isolation tests | No | N/A | Real failing-capable checks (client-to-client unreachable, server image/data isolation, non-root, partition counts, 2-round run/audit/budget gate) | 2 |
| Nightly full federated run | No | N/A | `schedule` + `workflow_dispatch`, full 4-client container run, full notebook mode, DP utility checks; no publishing | 2 |
| Podman rootless | No (tool missing locally) | N/A | CI job on ubuntu-latest, rootless run, host uid check, document limits if `podman-compose` insufficient | 3 |
| Image security/release | No | N/A | Trivy (HIGH/CRITICAL, dated ignores) + SBOM artifact; no publish/sign/registry creds; `docs/RELEASE_DESIGN.md` "not implemented" | 4 |
| Privacy budget gate & v2 MIA/DP artifacts | No (by audit) | not implemented/regenerated | Required before Phase 2 validation claims; ordering constraint from plan | pre-2 |
| Governance docs | Partial | `docs/CI_AUDIT.md` created this phase | `docs/RELEASE_DESIGN.md`, container/runbooks later; maintain implementation status/checkpoints per project protocol | 4+ |

## 5. Phase plan (ordered) and dependencies

1. **Phase 0 (this commit):** audit only — `docs/CI_AUDIT.md`.
2. **Phase 1:** make PR CI real and green (deps/backtracking fixed with Linux-verified lock/constraints, workflow rewrite, pinned actions, real security/artifact/privacy-invariant checks, run evidence). **Stop and report** after green run, timings, nightly moves.
3. **Blocker before Phase 2:** commit privacy budget gate and v2 MIA/DP artifacts (separate workstream per master prompt).
4. **Phase 2:** containers + isolated 4-client Compose + container CI/nightly.
5. **Phase 3:** Podman rootless validation.
6. **Phase 4:** Trivy/SBOM + `docs/RELEASE_DESIGN.md` (no publish/sign).

## 6. Not done / unverified in Phase 0

- No remote CI logs fetched; prior failure details based on reported run `44521000019`.
- No Linux container install verification executed in this phase.
- Fleet API default bind port and deployment-mode TLS command matrix not fully pinned from defaults (must set/verify explicitly).
- `gitleaks`/`pip-audit` not present locally; tool selection/versions for CI to be resolved with pinned actions/binaries.
- Other unstaged working-tree fixes (workflow/constraints/code quality/type fixes) intentionally **not** included in this audit-only commit.

## 7. Phase 1 execution (2026-11-19, clean-tree evidence)

### 7.1 Unstaged-change triage (completed before commits)

`git diff --stat` (pre-triage working tree): 14 files, 57 insertions(+), 52 deletions(-).

| File(s) | Disposition | Evidence |
|---|---|---|
| `src/insurance_fl/data/generator.py` | Committed (`4ef7e95`) — behavior-neutral | Import sort, explicit `cast`/`np.array(..., copy=True)` for mypy; `label_logic_sha()` unchanged (`03051d931c560108`); `make data-check` PASS on modified tree. |
| `src/insurance_fl/data/quality.py` | Committed (`4ef7e95`) — behavior-neutral | Lambda/line-break reformatting only. |
| `src/insurance_fl/fl/privacy.py` | Committed (`4ef7e95`) — behavior-neutral | Formatting, `RdpAccountantWrapper` forward-ref annotations, `bool(...)` cast in `would_exceed` for mypy; no logic changes. |
| `src/insurance_fl/fl/task.py` | Committed (`4ef7e95`) — behavior-neutral | Import cleanup, formatting, explicit ndarray casts for mypy; no feature/policy logic changes. |
| `pyproject.toml` | Committed (`61b699c`) | `pytest>=7.2.0,<10.0.0`, `pytest-cov>=4.0.0,<8.0.0` (dev extra) to align with resolved Linux toolchain. |
| `tests/integration/test_causal_transport_regression.py`, `tests/integration/test_rdp_accounting.py`, `tests/unit/data/test_signal_and_generator.py`, `tests/unit/fl/test_feature_leakage.py`, `tests/unit/fl/test_task.py`, `tests/unit/scripts/test_mia_script.py` | Committed (`61b699c`, lint-only portion) | Import/format/comment/whitespace fixes only; no assertions/markers/parametrizations removed or moved. |
| `tests/unit/data/test_quality.py`, `tests/unit/fl/test_privacy.py` | Committed (`9344252`) | Formatting/import-order fixes; **one assertion added** in `tests/unit/fl/test_privacy.py::test_state_persistence_roundtrip`: `assert math.isfinite(eps1["epsilon"])` (strengthens check; no tests removed/moved/renamed). |
| `.github/workflows/ci.yml` (prior unstaged edits) | Discarded, then rewritten in place | Replaced by real Phase 1 workflow (see §7.4). |
| `constraints/ci-py311.txt` (untracked, macOS-derived) | Deleted, never committed | Replaced by Linux-generated `constraints/ci-linux-py311.txt`. |
| `reports/notebook_validation.json` | Restored (no commit) | Validator run side effect; not part of triage. |

### 7.2 Test-count investigation (requested)

| Scope | `d04506f` collected not-slow test IDs (`pytest --collect-only -q -m "not slow"`, `./venv/bin/python -m pytest`) | Clean HEAD (`9344252`) | Delta |
|---|---:|---:|---|
| `tests/unit` | 249 | 249 | 0 (no disappeared/added IDs) |
| `tests/integration` | 47 | 47 | 0 (no disappeared/added IDs) |

- Execution totals on clean HEAD: **unit 250 passed, 4 deselected**; **integration 47 passed**. (Collection summary lines differ in formatting across runs; ID lists above are the compared artifact.)
- Earlier reported baseline “296 not-slow passed / 45 passed” was **not reproduced** at `d04506f` or current HEAD in this environment; no tests were removed or weakened during triage. The only assertion change was an addition in `test_state_persistence_roundtrip` (§7.1).

### 7.3 Clean-tree measurements (after triage commits, `git status` clean except untracked `constraints/` before staging)

| Check | Command | Result | Wall time (real) |
|---|---|---:|---:|
| Unit (not slow) | `python -m pytest -q -m "not slow" tests/unit` | 250 passed, 4 deselected | ~49.7 s |
| Integration | `python -m pytest -q tests/integration` | 47 passed | ~8.1 s |
| Notebook validator | `python scripts/validate_notebooks.py` | 11/11 passed | ~77.9 s |
| Ruff | `python -m ruff check src/insurance_fl tests` | pass (config deprecation warning: `isort` -> `lint.isort`) | fast |
| Black | `python -m black --check src/insurance_fl tests` | 101 files unchanged | ~0.36 s |
| Isort | `python -m isort --check-only src/insurance_fl tests` | pass | ~0.21 s |
| Mypy | `python -m mypy src/insurance_fl` | 46 source files, no issues | ~0.73 s |

`make data-check`: PASS, `LABEL_LOGIC: ok (sha=03051d931c560108)`.

### 7.4 Linux constraints and CI workflow

- **Constraints:** `constraints/ci-linux-py311.txt` generated via `uv pip compile pyproject.toml --extra dev --extra causal --extra privacy --python-version 3.11 --python-platform x86_64-unknown-linux-gnu -o constraints/ci-linux-py311.txt` (807 lines). No resolver conflict reported. Key pins present: `flwr==1.31.0`, `pytest==9.2.0`, `pytest-cov==7.1.0`, `dp-accounting==0.6.0`, `statsmodels==0.14.6`, `pgmpy==0.1.25`, `mypy==1.10.0`.
- **Docker install verification:** not executed locally — Docker daemon unavailable (`Cannot connect to the Docker daemon...`). First GitHub Actions run is the Linux install verification.
- **Workflow:** `.github/workflows/ci.yml` rewritten in place (no duplicate). Jobs: `quality` (ruff/black/isort/mypy), `unit` (not slow + coverage `--cov-fail-under=70`), `integration`, `artifacts` (generate data, `make data-check`, provenance tests, `make flwr-guard`), `notebooks` (`NB_MODE=light`), `privacy-invariants` (dp-accounting import + privacy/provenance/RDP tests), `security` (gitleaks + `pip-audit -r constraints/ci-linux-py311.txt --strict`), `ci-gate` (`needs` all, `if: always()`). All jobs: `ubuntu-latest`, least-privilege `permissions: contents: read`, `concurrency` cancel-in-progress, timeouts, pip cache keyed on `constraints/ci-linux-py311.txt`, install `-c constraints/ci-linux-py311.txt -e ".[dev,causal,privacy,notebooks]"`.
- **Action pins (full SHAs):** `actions/checkout@11bd71901bbe5b1630ceea73d27597364c9af683?` — resolved via `gh api`: checkout `11d5960a326750d5838078e36cf38b85af677262` (v4), setup-python `a26af69be951a213d495a4c3e4e4022e16d87065` (v5), cache `0057852bfaa89a56745cba8c7296529d2fc39830` (v4, not used in workflow), upload-artifact `ea165f8d65b6e75b540449e92b4886f43607fa02` (v4, not used), gitleaks-action dereferenced annotated tag to `ff98106e4c7b2bc287b24eaf42907196329070c7` (v2).
- **`actionlint`:** `actionlint .github/workflows/ci.yml` produced no output (exit success).

### 7.5 Flower command names (corrected/verified)

Verified binaries and defaults (Flower 1.31.0, `./venv`): `flower-superlink`, `flower-supernode`, `flwr run`, `flwr-serverapp`, `flwr-clientapp`. No `flwr-superlink`/`flwr-supernode` console scripts. Defaults from help: SuperLink ServerAppIo `0.0.0.0:9091`, Control API `0.0.0.0:9093`; SuperNode ClientAppIo `0.0.0.0:9094`; `flwr-serverapp` `127.0.0.1:9091`; `flwr-clientapp` `127.0.0.1:9094`. Fleet API address remains deployment-specific/unpinned in captured defaults; `scripts/launch_tls.py` example (`127.0.0.1:8000`) must be re-verified in Phase 2.
