% ---
% Version: v1.0 (original from root, unmodified)
% Canonical source: PROMPT.md
% Created by orchestrator O1 on 2026-10-16
% ---

# Master Prompt: Insurance Federated Learning Platform

Paste everything below the line into Claude Code (or Claude with repo access) from the repository root. Attach `README.md` and `European_Insurance_Federated_Learning_Project.md` as context.

---

## ROLE

You are a senior ML platform engineer (federated learning, MLOps, Python packaging, DevOps). You are working on the repository `GuyKaptue/insurance-federated-learning`: a federated learning platform for insurance analytics across four clients (Germany/Allianz, France/AXA, Belgium/Ageas, Switzerland/Helvetia) with five ML tasks (2 regression, 2 binary classification, 1 multi-class).

## MISSION

In strict order, and one phase at a time:

1. Analyze the complete project and research every error or inconsistency.
2. Fix and fully update the implementation.
3. Make every module and file have a matching test module.
4. Implement a Makefile, Docker, and GitHub Actions CI/CD.
5. Run the full test suite.
6. Run all federated learning experiments.
7. Deliver an evidence-based final report.

## NON-NEGOTIABLE RULES

1. **Evidence over claims.** Never state that something works, passes, or produces a number unless you ran it and saw the output. Quote the real command and the real output. Do not copy expected outputs from the attached documents (for example `7 passed in 3.42s`, the KL values, `stat=0.42`); those are unverified placeholders. Reproduce or replace them with measured values.
2. **Treat the attached documents as hypotheses, not truth.** Verify every claim about package versions, APIs, and file existence yourself (use `pip index versions <pkg>`, `pip install --dry-run`, official docs, PyPI, and the repo itself).
3. **Step by step with gates.** Finish a phase, show its gate evidence, and only then move on. If a gate fails, fix and re-run until it passes. Do not skip ahead.
4. **Root cause, not patches.** For each error, record symptom, root cause, fix, and the test that now guards against it. Never silence a failure by deleting or weakening a test, adding `xfail`/`skip` without justification, or loosening a threshold without explaining why.
5. **Small, reviewable commits** on a branch `chore/full-audit`: one commit per phase or logical fix, conventional-commit messages.
6. **Reproducibility.** Pin or bound dependencies, seed all randomness, and make runs deterministic across processes.
7. **No secrets, no real personal data.** All data is synthetic. Do not add credentials to the repo, Docker images, or CI logs.
8. **Ask only when blocked.** Otherwise decide, document the decision in `docs/DECISIONS.md`, and continue.
9. **Keep a running log** in `docs/AUDIT_LOG.md` (phase, commands, outputs summary, issues found, issues fixed).

---

## PHASE 0 – Orientation (read-only)

1. Print the full tree (`git ls-files`, plus untracked files) and the Python/OS/Docker versions available.
2. Read every file: README, `setup.py` and/or `pyproject.toml`, `requirements*.txt`, source, scripts, tests, configs, workflows, Docker/K8s files.
3. Produce an inventory table: file, purpose, has-test (yes/no), status (works / broken / missing / placeholder).
4. Compare what the README promises with what exists on disk. List every gap (for example referenced files such as `docker/Dockerfile.prod`, `kubernetes/`, `CONTRIBUTING.md`, `LICENSE`, `docs/*.md`, `.github/workflows/ci.yml`, `config/federated_config.yaml` that do not exist).

**Gate 0:** inventory and gap list written to `docs/AUDIT_LOG.md`. Nothing modified yet.

## PHASE 1 – Error research (reproduce before fixing)

Fresh virtualenv, then try the documented install and run paths exactly as a new user would: `pip install -e ".[dev]"`, `insurance-fl-generate`, `insurance-fl-train`, `insurance-fl-serve`, `pytest`.

Record every failure verbatim. In addition, investigate these **specific suspects** found in review of the attached documents. Confirm or refute each with a reproduction:

| # | Suspect | How to verify |
| --- | --- | --- |
| 1 | `tensorflow-federated` pinned to a version that may not exist, and TFF is heavy and may conflict with the Flower/PyTorch stack | `pip index versions tensorflow-federated`; check Python and TF compatibility. Decide: drop TFF (recommended) or isolate it. Document the decision. |
| 2 | `setup.py` console entry points use `src.scripts...` while the package is `insurance_fl` (module path mismatch); `setup.py` and `pyproject.toml` duplicate each other | Install and run each console script. Keep a single source of truth (`pyproject.toml`) and remove `setup.py`. |
| 3 | Python version mismatch: README badge says 3.11+, one config says `>=3.11,<3.13`, another `>=3.10,<3.13` | Pick one range, align badges, classifiers, CI matrix, Dockerfile. |
| 4 | **Non-deterministic seeding**: `np.random.default_rng(cfg.seed + hash(country) % 10_000)` uses Python's `hash()` on a string, which is salted per process (PYTHONHASHSEED) | Run generation twice in separate processes and compare checksums. Fix with a stable hash (for example `zlib.crc32`) and add a determinism test. |
| 5 | `_truncated_normal` computes `a, b` but never uses them and just clips (this piles mass at the bounds and is not truncated-normal) | Check boundary mass. Use `scipy.stats.truncnorm` or rejection sampling. Add a distribution test. |
| 6 | `dirichlet_partition`: `splits[-1] = len(idx) - splits[:-1].sum()` can go negative; floor rounding drops samples; comment about `LinAlgError` for small alpha is dubious | Fuzz with many seeds/alphas; assert every index is assigned exactly once and no split is negative. Use the cumulative-proportion split method. |
| 7 | `inject_quality_issues`: assigning NaN into integer columns, outlier injection based on a max that may itself be NaN or an earlier outlier, tenure forced to `age + 5` | Test dtypes after injection, parquet round-trip, and that the injected rates match configuration. |
| 8 | `GeneratorConfig.inject_quality_issues` flag exists, but quality injection is done in the CLI rather than the generator | Make behavior consistent and tested. |
| 9 | Tests assert relationships the generator never creates (for example `age` vs `tenure_years` correlation `> 0.1`, but tenure is drawn independently; Swiss income `> 1.3×` German mean is plausible but threshold unverified; income–credit-score `> 0.2`) | Run them, observe actual values, then either improve the generative process (preferred, documented) or correct the assertion with justification. |
| 10 | Segment probabilities: `seg_logits = segment_probs * (1 + income/mean)` is not a meaningful income-dependent mapping, and a per-row Python loop with `rng.choice` is slow | Vectorize and make income genuinely shift segment odds. Test the monotonic effect. |
| 11 | Mixed currencies (EUR vs CHF) mixed into combined data without normalization | Decide: keep local currency with an explicit column and document, or add a normalized column. Test it. |
| 12 | The attached analysis contains fabricated-looking evidence (KL values, p-values, timings, "7 passed") | Replace with real measured outputs stored under `reports/`. |
| 13 | README has a broken code fence (a stray closing fence followed by a `setup.py` section and git commands inside the same file) and placeholder badges (PyPI, Zenodo DOI `XXXXXX`, ReadTheDocs, codecov, downloads) for things that do not exist | Fix the structure; remove or replace badges that point to non-existent services. |
| 14 | `import` hygiene: missing `__init__.py`, scripts importing with the wrong package path, `find_packages(where="src")` returning `[]` | Verify with `python -c "import insurance_fl"` and `pip show -f`. |

Also search for anything else: type errors (`mypy`), lint (`flake8`/`ruff`), unused code, mutable defaults, dtype issues, leakage (target-derived features such as `high_risk` computed from `claim_risk_score` and `credit_score` that are also model inputs), and label leakage between tasks.

**Gate 1:** `docs/AUDIT_LOG.md` contains an error register: ID, symptom (verbatim output), root cause, planned fix, planned guarding test. Every suspect above is marked CONFIRMED or REFUTED with evidence.

## PHASE 2 – Restructure and fix

1. Adopt the `src/insurance_fl/` layout with `pyproject.toml` as the single packaging file (setuptools, `[project.scripts]` pointing to `insurance_fl.scripts.*`). Add `py.typed`, `LICENSE`, `CONTRIBUTING.md`, `.gitignore`, `.dockerignore`, `.editorconfig`, and pre-commit config (`ruff`, `black`, `isort`, `mypy`).
2. Implement and fix, one module at a time (each with its test written alongside):
   - `data/country_profiles.py` (add schema validation, for example with pydantic, so malformed profiles fail loudly).
   - `data/generator.py` (deterministic, vectorized, documented correlations, five tasks' targets defined clearly: 2 regression, 2 binary, 1 multi-class).
   - `data/quality.py`, `data/partition.py`, and a `data/preprocess.py` (imputation, scaling, encoding, leakage-safe, fit on train only).
   - `models/` for the multi-task / per-task models (PyTorch, CPU-friendly) with a clean `get_weights` / `set_weights` interface.
   - `fl/client.py`, `fl/server.py`, `fl/strategies.py` using **Flower** (`flwr`). Support FedAvg and FedProx at minimum, optionally differential privacy (clipping + noise) behind a config flag. Provide both in-process simulation and real gRPC server/client modes.
   - `config/federated_config.yaml` (validated config loader).
   - `scripts/generate_dataset.py`, `run_federated_training.py`, `evaluate_models.py`, `deploy_api.py` (FastAPI with `/health`, `/predict/<task>`, `/metrics`).
3. Every public function has type hints and a docstring. No dead code.

**Gate 2:** `pip install -e ".[dev]"` succeeds in a clean venv; `python -c "import insurance_fl"` works; all five console scripts respond to `--help`.

## PHASE 3 – One test module per source module

Create and enforce a strict mirror: `src/insurance_fl/<pkg>/<module>.py` maps to `tests/unit/<pkg>/test_<module>.py`.

1. Generate the full mapping table (source file, test file, status) in `docs/TEST_MATRIX.md`.
2. Add `tests/test_structure.py`: a test that walks `src/insurance_fl/**/*.py` (excluding `__init__.py`) and **fails** if any module lacks its mirrored test file. This makes the rule self-enforcing.
3. For each module write: happy-path tests, edge cases, error handling, determinism (same seed gives identical output across processes), property-based tests with `hypothesis` for partitioning and quality injection, and dtype/schema checks.
4. Integration tests (`tests/integration/`): generate, preprocess, 2-round FL simulation on a tiny dataset for each task, evaluate, serve API with `TestClient`.
5. Mark slow tests (`@pytest.mark.slow`); keep the default run under a few minutes.
6. Coverage target: at least 85% line coverage with branch coverage on, enforced in config (`--cov-fail-under=85`). Report honestly if you cannot reach it, and why.

**Gate 3:** `pytest -q` is green; `tests/test_structure.py` passes; coverage report attached.

## PHASE 4 – Makefile

Create a self-documenting `Makefile` (`make help` lists targets from `##` comments), using `.PHONY`, a venv-aware `PYTHON` variable, and no hard-coded absolute paths. Required targets:

| Target | Purpose |
| --- | --- |
| `install` / `install-dev` | create venv and install |
| `lint` / `format` / `typecheck` | ruff, black, isort, mypy |
| `test` / `test-unit` / `test-integration` / `test-slow` / `coverage` | pytest variants |
| `data` | generate synthetic datasets into `data/processed` |
| `train` | one FL run from config |
| `fl-all` | every FL experiment (see Phase 7) |
| `evaluate` | evaluate and write `reports/` |
| `serve` | run FastAPI locally |
| `docker-build` / `docker-run` / `docker-test` / `compose-up` / `compose-down` | container workflow |
| `ci` | what CI runs locally: lint, typecheck, tests, build |
| `clean` | remove caches and build artifacts |

**Gate 4:** `make help`, `make ci`, and `make data` run successfully; paste the output.

## PHASE 5 – Docker

1. `Dockerfile`: multi-stage, slim Python base matching the supported version, non-root user, `PYTHONDONTWRITEBYTECODE`, `PYTHONUNBUFFERED`, pip cache mounts, `HEALTHCHECK`, pinned base image tag, CPU-only PyTorch wheels.
2. `docker-compose.yml`: services `fl-server`, `fl-client-de`, `fl-client-fr`, `fl-client-be`, `fl-client-ch`, `api`, each client mounting only its own country's data (this demonstrates data locality). Optional profile for Prometheus and Grafana only if you actually create their configs.
3. `.dockerignore` excluding data, venvs, caches, `.git`.
4. If K8s manifests are kept, they must be real and validated (`kubectl apply --dry-run=client`, or `kubeconform`); otherwise remove all references to them from the README.
5. Build and run: `docker build`, `docker compose up` for a short multi-container FL run, and confirm a clean exit with metrics logged. If Docker is unavailable in your environment, say so explicitly and validate with `docker compose config` and `hadolint` instead; do not claim a build succeeded.

**Gate 5:** image builds, container tests pass, a compose FL run completes (or the limitation is stated with the static validation that was done).

## PHASE 6 – GitHub Actions CI/CD

Create `.github/workflows/`:

- `ci.yml`: triggers on push and pull\_request. Jobs: `lint` (ruff, black --check, isort --check), `typecheck` (mypy), `test` (matrix over the supported Python versions; coverage artifact; codecov upload only if the token is configured), `fl-smoke` (2-round simulation on tiny data), `docker` (buildx build with layer cache, run container tests, no push on PRs).
- `release.yml`: on tag `v*`: build sdist and wheel, run tests, publish to PyPI with trusted publishing (OIDC) only if the maintainer enables it, push image to GHCR, create a GitHub Release with notes.
- `codeql.yml` and Dependabot config (`.github/dependabot.yml`) for pip, GitHub Actions, and Docker.

Use `concurrency` to cancel superseded runs, `permissions` set to least privilege, pinned action versions (major tag or SHA), and dependency caching. Validate with `actionlint` and, if possible, `act`. Fix the README badges so each points to a workflow or service that really exists.

**Gate 6:** `actionlint` clean; workflows reviewed; local `make ci` equals the CI steps.

## PHASE 7 – Run all federated learning

Define the experiment grid in `config/experiments.yaml` and run it with `make fl-all`:

- **Tasks (5):** the 2 regression, 2 binary, and 1 multi-class targets (for example claim\_risk\_score, CLV, renewal, high\_risk, segment).
- **Strategies:** local-only baseline per client, centralized baseline (pooled data, upper bound), FedAvg, FedProx.
- **Heterogeneity:** the native country non-IID setting plus Dirichlet alpha in {0.1, 0.3, 1.0}.
- **Privacy:** DP off and on (report epsilon/delta if DP is implemented and accounted correctly; otherwise say it is not formally accounted).
- **Seeds:** at least 3 seeds; report mean and standard deviation.
- **Rounds:** configurable (for example 20), early-stopping optional.

For each run save: config, seed, per-round metrics, final metrics per client and global, and wall time under `reports/runs/<run_id>/`. Metrics: MAE/RMSE/R2 (regression), AUC/F1/PR-AUC (binary), macro-F1/accuracy (multi-class), plus a fairness-style per-country breakdown.

Then produce `reports/summary.md` with comparison tables and plots (convergence curves, per-country performance, FL vs local vs centralized). Interpret the results honestly: where FL helps, where it does not, and whether non-IID degrades convergence. Compute the real non-IID measures (for example per-feature KS statistics between countries and label-distribution divergence) instead of quoting the placeholder numbers.

**Gate 7:** all runs complete without errors; `reports/summary.md` generated from actual outputs.

## PHASE 8 – Full regression and final report

1. Fresh clone into a temp dir, then run: `make install-dev && make ci && make data && make fl-all`; then `docker build` and the compose run.
2. Re-run the full test suite three times to check for flakiness; fix any flaky test (seed, tolerance, ordering).
3. Update README: accurate install, quick start, architecture diagram (Mermaid), the Makefile targets, Docker usage, CI badges that really work, results table copied from `reports/summary.md`, known limitations.
4. Write `docs/FINAL_REPORT.md` containing:
   - Executive summary (what was broken, what was fixed).
   - Error register with final status (all items from Phase 1) and the test guarding each.
   - Test matrix (every module mapped to its test) and coverage numbers.
   - CI/CD and Docker description with run evidence.
   - FL experiment results and honest interpretation.
   - Remaining risks and recommended next steps.
5. Open a pull request from `chore/full-audit` with the report as the description.

**Gate 8:** the checklist below is entirely ticked with evidence.

---

## DEFINITION OF DONE (all must be true, each with evidence)

- [ ] Clean-venv install works; all console scripts run.
- [ ] No broken or missing file is referenced by the README, Docker, or CI.
- [ ] Every source module has a mirrored test module, enforced by `tests/test_structure.py`.
- [ ] Full test suite green three times in a row; coverage threshold met.
- [ ] Generation is deterministic across processes.
- [ ] `make help`, `make ci`, `make fl-all` work.
- [ ] Docker image builds and runs; compose FL run completes (or limitation documented).
- [ ] GitHub Actions workflows pass `actionlint` and cover lint, types, tests, FL smoke, Docker, release.
- [ ] All FL experiments run; results in `reports/summary.md` come from measured outputs.
- [ ] `docs/FINAL_REPORT.md` and an updated README delivered.

## OUTPUT FORMAT FOR EACH PHASE

At the end of every phase, reply with:

1. **What I did** (files created/changed).
2. **Commands run and key output** (real, trimmed).
3. **Errors found and how they were fixed.**
4. **Gate status** (PASS/FAIL, with evidence).
5. **Next phase** (and stop if a gate failed).

Start now with **Phase 0**. Do not modify any file until Phase 1 is complete.
