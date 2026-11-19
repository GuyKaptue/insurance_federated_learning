# Implementation Status — Insurance Federated Learning Platform

## Snapshot
- **PROJECT:** Insurance Federated Learning Platform
- **CURRENT BRANCH:** `feat/task0-baseline-federation`
- **CURRENT COMMIT:** (updated per stage)
- **LAST VERIFIED COMMIT:** `ab1b9c6` (pre-Stage A continuation baseline; audit completed read-only)
- **CURRENT STAGE:** Stage A (baseline validation & consistency) — in progress
- **LAST COMPLETED STAGE:** Stage 0 (repository audit)

## Completed implementations
- Stage 0 audit delivered (read-only): repo/git state, architecture, capability inventory, CI/test baseline, security/privacy posture, gaps, risks, roadmap, dependencies.
- Stage A (partial): test-split Bayes sanity analysis added to `reports/is_high_risk_auc.json:test_split_bayes` (all models ok, no AUC > Bayes+2*SE); 24-feature surface verified identical/ordered across 4 clients with explicit missingness indicators and no leakage; feature-leakage and structure tests strengthened.

## Tests executed
- `./venv/bin/python -m pytest -q tests/test_structure.py` → **1 failed, 2 passed** at audit time (orphan allowlist for `tests/unit/fl/test_feature_leakage.py`); fixed in Stage A.
- `./venv/bin/python -m pytest -q tests/test_structure.py tests/unit/fl/test_feature_leakage.py` → **4 passed** (Stage A).
- `./venv/bin/python -m pytest -q tests/unit/test_artifact_provenance.py tests/unit/data/test_signal_and_generator.py` → pending/recorded below.

## Artifacts generated/updated
- `reports/is_high_risk_auc.json`: added `test_split_bayes` (per-client Bayes test AUC, Hanley–McNeil SE, 2*SE flag rule, split/dup checks, model comparison) and feature-surface note.
- `tests/unit/fl/test_feature_leakage.py`: asserts identical ordered 24-feature list across all 4 clients, explicit indicators, forbidden exclusions.
- `tests/test_structure.py`: allowlist updated for `fl/test_feature_leakage.py`.

## Open issues
- CI `resolution-too-deep` (`pip install -e ".[dev,causal]"`) unresolved; `privacy`/`notebooks` extras not installed in workflow; 8+ commits unpushed (remote unverified).
- Privacy budget pre-release gate/enforcement, `_account_round`, atomic resume integration, `privacy_budget_exhausted`/`resumed_from_round` not implemented.
- Fresh v2 `reports/mia_evaluation.json` and `reports/dp_utility_curve.json` missing (stale_v1 only); MIA overfit control not yet strengthened to honest target.
- Calibration/fairness wording consistency sweep across README/notebooks/dashboards to be completed and verified (artifacts/DECISIONS currently aligned: raw default, fairness inconclusive).
- No Docker/Podman/Compose/K8s/cloud deployment assets; TLS runtime enforcement/negative matrix incomplete; clean-clone script missing.

## Known limitations
- `node_id % 4` client identity is simulation-only (not cert-bound/authz).
- SecAgg+ is not DP; RDP accounting is conservative (q=1 headline) and not enforced by strategy loop yet.
- Fairness status **inconclusive** (point 0.0347, 95% CI [0.0159, 0.0803] crosses 0.05, conservative max-min bootstrap).
- 4-client single-host simulation is not proof of multi-org/production deployment.
- Bitwise reproducibility not guaranteed (documented characterization pending).

## Uncommitted changes
- (none after Stage A commit/checkpoint)

## Blockers
- Push target/remote approval context: only `origin=https://github.com/GuyKaptue/insurance_federated_learning.git` configured (decision needed only if different from user intent).
- CI dependency resolution strategy choice may affect H-min (bounds vs constraints vs uv).

## Next stage
- **Stage A completion checks:** finish consistency sweep (README/notebooks), run relevant provenance/structure tests, commit/checkpoint.
- **Stage H-min:** diagnose/fix CI dependency backtracking, include required extras, validate fresh install, push and confirm green run.
- **Stage C:** implement/test privacy budget gate and restart-safe accounting in Flower 1.31 path.

## Exact next action
- Complete Stage A consistency verification and commit `docs/engineering/IMPLEMENTATION_STATUS.md` + any remaining doc fixes; then run targeted tests and proceed to H-min/C per evidence.

## Files to inspect before continuing
- `docs/engineering/IMPLEMENTATION_STATUS.md`
- `reports/is_high_risk_auc.json` (`test_split_bayes`, calibration, summary)
- `artifacts/fairness_disparity.json`
- `tests/test_structure.py`, `tests/unit/fl/test_feature_leakage.py`
- `src/insurance_fl/fl/{strategy.py,server_app.py,privacy.py,task.py}`
- `.github/workflows/ci.yml`, `pyproject.toml`, `Makefile`
- `docs/evidence/dataset_manifest.json`, `docs/DECISIONS.md`
