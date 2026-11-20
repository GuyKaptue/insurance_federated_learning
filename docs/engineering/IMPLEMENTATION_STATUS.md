# Implementation Status — Insurance Federated Learning Platform

## Snapshot
- **PROJECT:** Insurance Federated Learning Platform
- **CURRENT BRANCH:** `feat/task0-baseline-federation` (local commits ahead of origin include Part 1, Part 2 identity/guard, and incident/SecAgg+ honesty records; push/CI status recorded separately)
- **CURRENT COMMIT (HEAD, local):** `51d96c7` + pending follow-up fix commit (incident/SecAgg+ honesty/docs/test records)
- **LAST VERIFIED COMMIT (local validation):** working tree at `6299b71` + pending incident/SecAgg+ honesty/docs/test updates (not yet committed)
- **LATEST CI (evidence):** [44609880224](https://github.com/GuyKaptue/insurance_federated_learning/actions/runs/44609880224) — `quality/unit/integration/artifacts/notebooks/privacy-invariants` success, **security failure** (`pip-audit`: `ray 2.55.1` PYSEC-2026-2273; 108 ignored via blanket policy), `ci-gate` failure. Prior: [44608982208](https://github.com/GuyKaptue/insurance_federated_learning/actions/runs/44608982208), [44595947290](https://github.com/GuyKaptue/insurance_federated_learning/actions/runs/44595947290).
- **CURRENT STAGE:** Part 2 (identity/incident/SecAgg+ honesty) finalization before Part 3 security gate
- **LAST COMPLETED SUBSTAGE:** Part 1 committed (`c3be175`), `.dockerignore` tracked (`4d5ff02`), Part 2 identity fix + tests + guard committed (`3ee2266`, `31deb5e`, `fec2a24`, `f5043a6`, `6299b71`)

## Decisions (authoritative)
- **D-ORCH-009 (2026-11-20):** Stage 0 approvals (branch reattach, ray/simulation audit policy no Flower upgrade, xgboost parity ≤0.005 AUC/client, MIA pre-registered fallback, containers after C+D only, delete/ignore macOS constraints, archive legacy status docs, `node_id%4` correctness blocker).
- **D-ORCH-010 (2026-11-20):** Client-mapping incident — Flower simulation `node_id % 4` misassigned/collided in 3/3 sampled runs; all pre-fix Flower app-path results invalid for 4-partition identity/per-client attribution claims; in-process FedAvg+FedBN reference artifacts unaffected. Evidence: `docs/evidence/client_mapping_incident.json`, `docs/evidence/flower_path_inventory.md`.
- **D-PRIV-006 (2026-11-20):** SecAgg+ `PARTIALLY_IMPLEMENTED`; `server_app.main(secagg-enabled=true)` not wired/verified for multi-round Flower simulation orchestration and now fails closed (`NotImplementedError`, no hang/plaintext fallback). SecAgg+ protects parameter updates; per-client metrics and `num_examples` can remain visible to server/strategy in legacy-compat workflow path.

## Verified baseline evidence (selected)
- Canonical v2 data/manifest: `docs/evidence/dataset_manifest.json` (`label_logic_sha=03051d931c560108`, `config_hash=e93bcefe91a0b25a`, seed 42, 4×5000).
- 24-feature leakage-safe surface: identical ordered features across 4 clients incl. fixed missing indicators; no target/ID/leakage (`src/insurance_fl/fl/task.py`, `tests/unit/fl/test_feature_leakage.py`).
- Test-split Bayes: `reports/is_high_risk_auc.json:test_split_bayes` — all clients/models `ok` vs Bayes+2*SE (`any_flagged: False`).
- Fairness: `artifacts/fairness_disparity.json` point ~0.034716, 95% CI [0.015891, 0.044514, 0.080282], **inconclusive** (CI crosses 0.05). **Artifact verified, wording sweep open.**
- Calibration comparison present; **raw probabilities remain default**.

## Flower-path incident inventory (summary)
- **Invalid for 4-partition identity/per-client attribution claims (pre-fix):** `docs/evidence/layer1_flwr_run_2rounds.log`, `docs/evidence/layer2_flwr_run_3rounds.log`, `runs/test_final13/fl.log`, pre-fix Flower 4-client notebook/run outputs (see `docs/evidence/flower_path_inventory.md`).
- **Unaffected by mapping bug:** in-process FedAvg+FedBN shipped-model reference artifacts including `reports/is_high_risk_auc.json`, `artifacts/fairness_disparity.json` (as reported). `reports/stale_v1/*` remain stale for separate reasons.
- **Post-fix participation proof:** `make flwr-guard` with `require-all-clients=true` enforces 4 distinct partition mappings and 4 fit/eval client IDs once each (`scripts/check_flwr_guard_identity.py`).

## Capability status
| Capability | Status | Evidence / notes |
|---|---|---|
| Dataset v2 generation, manifest, label-logic guard | IMPLEMENTED_AND_VERIFIED | manifest/metadata, `check_dataset_signal.py`, provenance tests |
| 24-feature ordered surface & leakage controls | IMPLEMENTED_AND_VERIFIED | `task.py` policy, `get_feature_names`, `test_feature_leakage.py` |
| Flower server/client app wiring & 1-round sim smoke | IMPLEMENTED_AND_VERIFIED (smoke path) | `server_app.py`, `client_app.py`, post-fix `make flwr-guard` identity checks pass (4 mappings, 4 fit, 4 eval) |
| Four-client distinct-partition execution in simulation | IMPLEMENTED_AND_VERIFIED (simulation, post-fix) | `resolve_client_id` from `partition-id/num-partitions` (`src/insurance_fl/fl/task.py`), client reports `client_id` (`client_app.py`), server learns `node_id→client_id` from replies with conflict/unknown/duplicate rejection + audit events, dropout-tolerant default, `require_all_clients` for proof (`strategy.py`, `server_app.py`), guard enforces all 4 in fit+eval (`Makefile`, `scripts/check_flwr_guard_identity.py`). Tests: `tests/unit/fl/test_client_identity.py` (7), `tests/unit/fl/test_client_app.py`, `tests/unit/fl/test_server_app.py` incl. secagg fail-closed, `tests/unit/fl/test_strategy.py`. Pre-fix Flower app-path evidence invalid (D-ORCH-010). Simulation-only; reported `client_id` not a deployment security control. |
| Multi-task NN + Kendall weighting | IMPLEMENTED_BUT_UNVERIFIED | implementation/tests exist; invariants/claims to be asserted per audit rigor |
| FedBN (local BN not aggregated) | IMPLEMENTED_BUT_UNVERIFIED | local BN persisted/restored in client; explicit cross-round non-aggregation/state-leak tests pending |
| Boosters/stacking/beta fusion (xgboost 2.1.4) | IMPLEMENTED_BUT_UNVERIFIED | code/tests present; **parity vs 1.7.6 (≤0.005 AUC/client) and lock/provenance review pending** |
| RDP accounting mechanics (dp-accounting) | IMPLEMENTED_AND_VERIFIED | `RdpAccountantWrapper` mechanics + privacy unit/integration tests; **accounting-only, not enforcement** |
| Privacy budget pre-release gate/enforcement/resume | MISSING | `project_epsilon/would_exceed` not used to block configure/release/aggregate; no `privacy_budget_exhausted`, no atomic resume/`resumed_from_round`, no strategy-level refusal tests (Stage C) |
| SecAgg+ | PARTIALLY_IMPLEMENTED | Flower 1.31 `SecAggPlusWorkflow`/`secaggplus_mod` integration and protocol-level tests exist (`tests/integration/test_secagg.py`, 11 passed). **End-to-end Flower simulation orchestration via `server_app.main` not verified/wired:** `secagg-enabled=true` fails closed with `NotImplementedError` (no hang, no plaintext fallback) — `tests/unit/fl/test_server_app.py::test_server_app_main_secagg_enabled_fails_closed`. Strategy disables identity learning/stratification and selects all nodes in secagg mode (`test_secagg_mode_identity_map_inactive_and_all_nodes_selected`). **Server visibility (D-PRIV-006):** SecAgg+ protects parameter updates; per-client metrics (e.g. `client_id`, train/eval metrics) and `num_examples` can remain visible to server/strategy in legacy-compat workflow path (`secaggplus_workflow.py` ~536–538,651–665: per-client `FitRes`/metrics passed to `strategy.aggregate_fit` with aggregated parameters). **Not DP.** No committed end-to-end SecAgg Flower-app run log. |
| Local DP | PARTIALLY_IMPLEMENTED | mod wiring exists; mechanism/sensitivity/clip/accounting correctness in-run to be proven |
| MIA evaluation (v2 artifact) | PARTIALLY_IMPLEMENTED | code/tests exist; `reports/stale_v1/mia_evaluation.json` only, honest control/power/loss-gap + v2 regen pending (Stage D) |
| DP utility curve (v2 artifact) | PARTIALLY_IMPLEMENTED | script exists; `reports/stale_v1/dp_utility_curve.json` only, v2 regen pending (Stage D) |
| Fairness metrics/uncertainty | IMPLEMENTED_AND_VERIFIED (artifact) | `artifacts/fairness_disparity.json` evidence; **status must remain `inconclusive`**; wording sweep open |
| Calibration (raw default) | IMPLEMENTED_BUT_UNVERIFIED | comparison evidence present; runtime/docs/notebook consistency to confirm |
| TLS/mTLS runtime enforcement | PARTIALLY_IMPLEMENTED | helper/cert scripts and TLS tests exist; end-to-end Flower 1.31 enforcement/authz/negative matrix/rotation pending |
| Structured audit trail | IMPLEMENTED_BUT_UNVERIFIED | logger/events implemented and wired; completeness/continuity/tamper-evidence/retention enforcement not proven |
| Causal inference & transportability | IMPLEMENTED_BUT_UNVERIFIED | modules/tests present; claims must remain evidence-bound and predictive vs causal kept separate |
| CI/CD (truthful green gate) | PARTIALLY_IMPLEMENTED | workflow/jobs exist; latest run not fully green/meaningful (security/ray policy pending) |
| Docker/Compose, Podman, Cloud/K8s, clean-clone procedure | MISSING | blocked/deferred per roadmap and decisions |

## Tests executed (local, this checkpoint)
- `$ ./venv/bin/python -m pytest tests/unit -m "not slow" -q` → **258 passed, 4 deselected**
- `$ ./venv/bin/python -m pytest tests/integration -q` → **47 passed**
- `$ PATH=... make flwr-guard GUARD_DATA=data/processed/european_insurance_federated` → **OK** (`Run finished 1 round(s)... flwr-guard identity checks: OK (4 mappings, 4 fit, 4 evaluate)`)
- `$ ./venv/bin/python -m ruff check ...` (changed Part 2 files) → **All checks passed**
- `$ ./venv/bin/python -m black --check ...` / `isort --check-only ...` (changed Part 2 files) → **OK**
- `$ ./venv/bin/python -m mypy src/insurance_fl` → **Success: no issues found in 46 source files**
- `$ ./venv/bin/python scripts/validate_notebooks.py` → **env issue (local):** validator/kernel executable not in repo venv; not caused by Part 2 code changes (`reports/notebook_validation.json` reverted). CI notebooks job was green on 44609880224.

## Open issues / to-do (ordered)
1. **Part 3 (security gate, next):** `simulation` extra only for `artifacts`; `constraints/runtime-linux-py311.txt` + blocking runtime audit (`pip-audit -r ... --no-deps --disable-pip --strict`), full-lock audit report-only (JSON artifact), delete `security/pip-audit-ignore.txt`, per-package findings for both locks, bump fixable vulns with tests after each, read PYSEC-2026-2273 and either prove disabled + add expiring justified ignore with `scripts/ci/check_audit_ignores.py` or leave red, negative control; push + `gh run watch --exit-status` + per-job table.
2. **SecAgg+ end-to-end Flower 1.31 simulation run/log:** wire/verify multi-round SecAggPlusWorkflow orchestration through Flower app path (client manager/init/round loop as required), produce committed run evidence/log, and remove `NotImplementedError` only when proven. **Acceptance:** `flwr run . --stream --run-config 'secagg-enabled=true ...'` (or equivalent verified command) completes without hang/error, with documented server-visible fields consistent with D-PRIV-006.
3. **xgboost parity table & lock-diff review** (≤0.005 AUC/client vs 1.7.6 baseline or rerun/update artifacts+decision), library provenance, deterministic params.
4. **Consistency sweep:** raw calibration default, fairness `inconclusive` with Bayes decomposition (0.0290/0.0347/~0.0057), 24 vs 26 features; round-1 loss scale diagnosis (~8e6).
5. **Stage C (D8):** privacy pre-release projected budget gate, `_account_round`, atomic save/resume + `resumed_from_round`, no unauthorized configure/aggregate on exceed, `privacy_budget_exhausted`, refusal/kill-resume/keys/strategy-level tests.
6. **Stage D:** honest MIA control (pre-registered fallback) + regenerated v2 `reports/mia_evaluation.json`, `reports/dp_utility_curve.json` with provenance/schema.
7. **Rerun cheap/required Flower-path evidence** post-fix before citation (see inventory).

## Known limitations
- 4-client mapping is **simulation-only** and proven for post-fix Flower simulation path; reported `client_id` is **not** a deployment security control (Stage F authenticated registration required).
- Privacy accounting is **not budget-enforced**; SecAgg+ ≠ DP; TLS ≠ authorization.
- SecAgg+ runtime orchestration through `server_app.main` is not wired/verified (fails closed).
- Fairness **inconclusive** (point 0.0347, 95% CI crosses 0.05). Raw calibration default.
- 4-client single-host simulation is not proof of multi-org/production deployment; bitwise reproducibility not guaranteed.

## Uncommitted changes
- None for Part 2 incident/SecAgg+ honesty record at this checkpoint (pending commit/push may exist during active edits).

## Blockers
- Ray PYSEC-2026-2273 vs `flwr[simulation]==1.31.0` pin (policy-driven resolution in Part 3).
- Stage C/D required before container validation claims depending on enforcement/artifacts.

## Risks
- R1–R3: budget not enforced; security gate misleading; sim identity/isolation mistaken for deployment security.
- R4: **resolved for post-fix simulation path** (four-client identity proven in guard/tests); residual deployment-security risk remains.
- R5: xgboost 2.1.4 drift risk pending parity proof.
- R6: stale/inconsistent research claims (MIA/DP stale, calibration/fairness wording, pre-fix Flower evidence).
- R7: CI flakiness/perf anomalies (raylet disk warnings in local runs).
- R8: **round-1 distributed loss ~8e6** — diagnosis pending.
- R9: addressed post-fix in guard (`require-all-clients` + fit/eval identity checks).
- R10: reproducibility/resume gaps. R11: hygiene/docs divergence. R12: overclaiming readiness.

## Continuation checkpoint
- **LAST COMPLETED SUBSTEP:** Part 2 identity fix committed (`3ee2266`, `31deb5e`, `fec2a24`, `f5043a6`, `6299b71`) and incident/SecAgg+ honesty record committed (`51d96c7`); local validation for Part 2 scope: unit 258 passed/4 deselected, integration 47 passed, flwr-guard identity OK (4 mappings, 4 fit, 4 eval), mypy clean, lint/format clean on changed files. Notebook validator failed locally due environment (validator/kernel not in repo venv), not a code regression; CI notebooks was green on 44609880224.
- **NEXT STAGE:** push latest commits, record CI per-job state, then **Part 3 security gate**.
- **EXACT NEXT ACTION:** push branch, `gh run watch --exit-status`, record per-job table/durations, then execute Part 3 security gate steps (simulation/runtime/full audits, ignore cleanup, fixes, PYSEC handling, negative control) and push again.
- **FILES TO INSPECT BEFORE CONTINUING:** `docs/DECISIONS.md`, `docs/engineering/IMPLEMENTATION_STATUS.md`, `docs/evidence/client_mapping_incident.json`, `docs/evidence/flower_path_inventory.md`, `.github/workflows/ci.yml`, `pyproject.toml`, `constraints/ci-linux-py311.txt`, `security/pip-audit-ignore.txt`.
