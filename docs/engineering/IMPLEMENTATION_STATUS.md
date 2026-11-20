# Implementation Status — Insurance Federated Learning Platform

## Snapshot
- **PROJECT:** Insurance Federated Learning Platform
- **CURRENT BRANCH:** `feat/task0-baseline-federation` (reattached at `efc174a` for Part 1; HEAD was detached but synced)
- **CURRENT COMMIT:** `efc174a` (working changes: Part 1 hygiene/docs in progress)
- **LAST VERIFIED COMMIT:** `efc174a` (code baseline); CI not fully green/truthful at this commit (see below)
- **LATEST CI (evidence):** [44609880224](https://github.com/GuyKaptue/insurance_federated_learning/actions/runs/44609880224) — `quality/unit/integration/artifacts/notebooks/privacy-invariants` success, **security failure** (`pip-audit`: `ray 2.55.1` PYSEC-2026-2273; 108 ignored via blanket policy) and `ci-gate` failure. Prior: [44608982208](https://github.com/GuyKaptue/insurance_federated_learning/actions/runs/44608982208), [44595947290](https://github.com/GuyKaptue/insurance_federated_learning/actions/runs/44595947290).
- **CURRENT STAGE:** Stage A closeout / Part 1–3 execution (authorized)
- **LAST COMPLETED STAGE:** Stage 0 audit (approved 2026-11-20; see `docs/DECISIONS.md` D-ORCH-009)

## Decisions (authoritative for current work)
- See **`docs/DECISIONS.md` — D-ORCH-009 (Stage 0, 2026-11-20)**: branch reattach, ray/simulation audit policy (no Flower upgrade), xgboost parity ≤0.005 AUC/client, MIA pre-registered fallback criterion, containers after C+D only (no publish/sign/deploy), delete/ignore macOS constraints, archive legacy status docs, `node_id%4` correctness blocker.

## Verified baseline evidence (selected)
- Canonical v2 data/manifest: `docs/evidence/dataset_manifest.json` (`label_logic_sha=03051d931c560108`, `config_hash=e93bcefe91a0b25a`, seed 42, 4×5000). `make data-check`/provenance tests exist and pass in CI contexts when data generated.
- 24-feature leakage-safe surface: identical ordered features across 4 clients incl. fixed missing indicators; no target/ID/leakage (`src/insurance_fl/fl/task.py`, `tests/unit/fl/test_feature_leakage.py`).
- Test-split Bayes: `reports/is_high_risk_auc.json:test_split_bayes` — all clients/models `ok` vs Bayes+2*SE (`any_flagged: False`).
- Fairness: `artifacts/fairness_disparity.json` point ~0.034716, 95% CI [0.015891, 0.044514, 0.080282], **inconclusive** (CI crosses 0.05). **Artifact verified, wording sweep open.**
- Calibration comparison present; **raw probabilities remain default** (no uniform improvement shipped) — runtime/docs/notebook consistency sweep open.

## Capability status (one status per row; audit rigor)
| Capability | Status | Evidence / notes |
|---|---|---|
| Dataset v2 generation, manifest, label-logic guard | IMPLEMENTED_AND_VERIFIED | manifest/metadata, `check_dataset_signal.py`, provenance tests, CI generation+checks |
| 24-feature ordered surface & leakage controls | IMPLEMENTED_AND_VERIFIED | `task.py` policy, `get_feature_names`, `test_feature_leakage.py` |
| Flower server/client app wiring & 1-round sim smoke | IMPLEMENTED_AND_VERIFIED (smoke path) | `server_app.py`, `client_app.py`, `flwr-guard` completes 1 round in CI 44609880224; deployment topology not proven |
| Four-client distinct-partition execution in simulation | IMPLEMENTED_BUT_UNVERIFIED | `CLIENT_IDS`/4 partitions exist but identity currently derived via `node_id%4` in client and strategy; **Part 2 must prove/fix 1:1 mapping and guard all-4 fit+eval** |
| Multi-task NN + Kendall weighting | IMPLEMENTED_BUT_UNVERIFIED | implementation/tests exist; specific invariants/claims to be named and asserted per audit rigor |
| FedBN (local BN not aggregated) | IMPLEMENTED_BUT_UNVERIFIED | local BN persisted/restored in client; explicit cross-round non-aggregation/state-leak tests pending |
| Boosters/stacking/beta fusion (xgboost 2.1.4) | IMPLEMENTED_BUT_UNVERIFIED | code/tests present; **parity vs 1.7.6 (≤0.005 AUC/client) and lock/provenance review pending** |
| RDP accounting mechanics (dp-accounting) | IMPLEMENTED_AND_VERIFIED | `RdpAccountantWrapper` mechanics + privacy unit/integration tests; **accounting-only, not enforcement** |
| Privacy budget pre-release gate/enforcement/resume | MISSING | `project_epsilon/would_exceed` not used to block configure/release/aggregate; no `privacy_budget_exhausted`, no atomic resume/`resumed_from_round` in FL loop, no strategy-level refusal tests (Stage C) |
| SecAgg+ | PARTIALLY_IMPLEMENTED | workflow/mod + tests exist; enabled runtime proof/dropout handling/params pending; **not DP** |
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

## Tests executed (recent verified)
- Unit (not slow, collected): **250 / 4 deselected** (`tests/unit`).
- Integration (collected): **47** (`tests/integration`).
- CI 44609880224: quality/unit/integration/notebooks/privacy-invariants/artifacts **success**; security/ci-gate **failure** (see snapshot).

## Open issues / to-do (ordered)
1. **Part 1 (this commit):** hygiene/docs/decisions/status corrections.
2. **Part 2:** prove/fix 4-client identity end-to-end (client `partition-id`, server identity learned from replies, no modulo; reject conflicts), guard all-4 fit+eval, tests.
3. **Part 3:** security gate — `simulation` extra scoping, runtime-blocking + full-lock report-only audits, remove blanket ignore, remediate fixable vulns, justified PYSEC-2026-2273 handling per policy, negative control, green CI.
4. **Next approved steps:** xgboost parity table & lock-diff; consistency sweep (calibration raw, fairness inconclusive with decomposition, 24 vs 26) & round-1 loss scale diagnosis; Stage C (privacy pre-release gate, `_account_round`, atomic resume, tests); Stage D (MIA control + v2 MIA/DP artifacts).

## Known limitations
- 4-client mapping currently unproven/fix pending (simulation-only). Reported `client_id` in simulation is **not** a security control; real deployment requires authenticated registration (Stage F).
- Privacy accounting is **not budget-enforced**; SecAgg+ ≠ DP; TLS ≠ authorization.
- Fairness **inconclusive** (point 0.0347, 95% CI crosses 0.05). Raw calibration default.
- 4-client single-host simulation is not proof of multi-org/production deployment; bitwise reproducibility not guaranteed.

## Uncommitted changes
- Part 1 hygiene/docs files staged for commit (this checkpoint).

## Blockers
- Ray PYSEC-2026-2273 vs `flwr[simulation]==1.31.0` pin (policy-driven resolution in Part 3).
- Stage C/D required before container validation claims depending on enforcement/artifacts.

## Risks (register delta)
- R1–R3: budget not enforced; security gate misleading; sim identity/isolation mistaken for deployment security.
- R4: **four-client mapping not 1:1** (`node_id%4` in client and strategy) — correctness blocker.
- R5: xgboost 2.1.4 drift risk pending parity proof.
- R6: stale/inconsistent research claims (MIA/DP stale, calibration/fairness wording).
- R7: CI flakiness/perf anomalies.
- R8: **round-1 distributed loss ~8e6** (unscaled regression target may dominate multitask loss and DP clipped updates) — diagnosis pending.
- R9: **guard evaluates 2/4 clients** in observed runs — insufficient participation proof.
- R10: reproducibility/resume gaps. R11: hygiene/docs divergence. R12: overclaiming readiness.

## Continuation checkpoint
- **LAST COMPLETED SUBSTEP:** Stage 0 audit approved; Part 1 (branch reattach, constraints ignore, legacy doc archived, decisions/status corrected) prepared.
- **NEXT STAGE:** Part 2 — prove/fix four-client identity (client+server), tests, guard all-4.
- **EXACT NEXT ACTION:** commit Part 1 docs/hygiene, then inspect `flwr` 1.31 `context.node_config` in simulation and patch `task.py`/`client_app.py`/`strategy.py` + tests + `Makefile` guard per D-ORCH-009.
- **FILES TO INSPECT BEFORE CONTINUING:** `docs/DECISIONS.md`, `docs/engineering/IMPLEMENTATION_STATUS.md`, `src/insurance_fl/fl/{task.py,client_app.py,server_app.py,strategy.py}`, `tests/unit/fl/test_client_app.py`, `tests/unit/fl/test_strategy.py`, `Makefile`, `.github/workflows/ci.yml`.
