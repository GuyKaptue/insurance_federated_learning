# Notebooks Audit (Modernization)

## Phase 1: Discovery summary

### Environment (observed)
- Python: 3.13.10
- Flower: 1.43.0 (installed; project states Flower pinned to 1.29.x baseline/D-ORCH-004 — version drift noted)
- Key packages present: numpy 2.4.2, pandas 3.0.5, scipy 1.18.1, scikit-learn 1.9.1, matplotlib 3.11.2, seaborn 0.13.2, plotly 6.9.0, torch 2.15.0, xgboost 3.3.0, lightgbm 4.6.0, catboost 1.2.10, shap 0.52.0, dp-accounting 0.6.0, nbformat 5.11.1, nbclient 0.11.0, ipykernel 7.4.0, networkx 3.6.1, cryptography 50.0.1, statsmodels 0.14.6
- Missing at discovery time: `pgmpy` not importable (causal project APIs may fail without extras; fallbacks used where possible)
- Data available: `data/processed/european_insurance_federated/federated/{germany_allianz,france_axa,belgium_ageas,switzerland_helvetia}` with parquet splits
- Evidence/artifacts present: `docs/evidence/layer1_flwr_run_2rounds.log`, `docs/evidence/layer2_flwr_run_3rounds.log`, `reports/dp_utility_curve.json`, `reports/mia_evaluation.json`, `artifacts/fairness_disparity.json`, `docs/causal/dag.yaml`

### Baseline findings (verified)
- All 11 notebooks existed with same filenames. `synthetic_data_generation.ipynb` was corrupted/invalid JSON at start of modernization and repaired.
- `transportability_quickstart.ipynb` contained copied synthetic-data content (title/cells) and no transport analysis; rewritten.
- `tls_credentials.ipynb` previously used hardcoded `/tmp/tls_notebook_certs`; rewritten to use `tempfile.TemporaryDirectory()`.
- `fairness_evaluation.ipynb` previously contained invented metrics (`Germany: 0.8, ...`) and `stratified_sample` preview only; rewritten to compute real per-country metrics from predictions.
- `dp_accounting_rdp.ipynb` previously used stale/possibly unavailable API patterns; rewritten against current `insurance_fl.fl.privacy` API (`PrivacyConfig`, `RdpAccountantWrapper.step/epsilon/project_epsilon/state_dict/save_state/load_state`).
- `notebooks/README.md` previously claimed all notebooks executed with `<5 min`; rewritten with status tied to validation evidence.

### API facts (real signatures inspected)
- `insurance_fl.fl.privacy`: `PrivacyConfig(noise_multiplier, delta, target_delta, ...)`, `RdpAccountantWrapper(cfg, state_path=None)` with `step()`, `compose()`, `epsilon() -> dict`, `project_epsilon(additional_releases=1)`, `would_exceed()`, `state_dict()`, `from_state(cfg,state)`, `save_state(path)`, `load_state(path,cfg)`, `build_privacy_config(run_config)`, `PrivacyBackendUnavailable`.
- `insurance_fl.fl.fairness`: `CLIENT_COUNTRY_MAP`, `FairnessConfig`, `group_clients_by_country`, `stratified_sample(...) -> (selected, uncovered)`, `compute_disparity(country_metrics, threshold_pct=0.05)`.
- `insurance_fl.evaluation.mia`: `compute_features(model,x,y,task_type)`, `tpr_at_fpr(y_true,y_scores,target_fpr)`, `run_mia(...)`.
- `insurance_fl.data.quality`: `DataQualityAnalyzer.analyze(df)` and `overall_quality_score()`; note: `analyze` may fail on unhashable array-valued object columns (e.g. `segment_probabilities` ndarray) due to `duplicated()` — notebooks sanitize such columns before analysis.
- `insurance_fl.fl.dp`: `LocalDPConfig`, `build_local_dp_mod(config)` returns modifier or `None` when disabled.
- `insurance_fl.transport.transport`: `measure_covariate_shift`, `transport_outcome_model`, `transport_causal_effect`, `TransportReport`; module imports `require_causal_deps()` (pgmpy required for module import path) — transport notebook avoids direct module import and uses local shift/weighting computations.
- `insurance_fl.causal.truth`: `inject_truth(df, seed, base_effect, ...)`, `ground_truth_ate(df, ...)`, `ml_target_columns()`, `ml_feature_columns(df)`.
- `insurance_fl.causal.estimands`: `CausalEstimand.from_dag(...)`, `estimate_backdoor(df,estimand)`, `estimate_doubly_robust(df,estimand)` (may require causal deps/pgmpy in this environment).
- `insurance_fl.causal.dag_check`: `load_dag`, `validate_dag`, `build_pgmpy_dag` (pgmpy-dependent).
- `insurance_fl.fl.boosters`: `train_boosters_for_client(client_id, x_train, y_train, x_val, y_val, output_dir, tasks, n_folds, seed)`.
- `insurance_fl.fl.stacking`: `fit_stacking_meta(oof_predictions, y_true, fallback=True)`, `predict_stacked`, `save/load_stacking_meta`, `validate_alpha`.
- `insurance_fl.fl.hybrid`: `optimize_beta(nn_preds, booster_preds, y_true, task_type, task_name) -> (beta_claim|None, diagnostics)` with informative/uninformative status.
- Flower: the latest validator used 1.31.0; `SecAggPlusWorkflow` parameter construction is exercised, while end-to-end app orchestration remains unverified and fails closed.

## Before/after (approximate, by modernization pass)

| Notebook | Before cells | After cells | Figures (save_fig calls) | Notes |
|---|---:|---:|---:|---|
| `synthetic_data_generation.ipynb` | corrupted stub | 40 | 18 | Rebuilt: quality, distributions, correlations, PCA/t-SNE, TSTR AUC, DCR, seed stability |
| `dp_accounting_rdp.ipynb` | 3 | 29 | 10 | Current privacy API, RDP theory/cross-check, restart/fail-closed demos |
| `local_dp.ipynb` | 4 | 21 | 8 | RR/Laplace/Gaussian experiments, utility curves, project LDP API |
| `membership_inference.ipynb` | 4 | 19 | 8 | Artifact + computed MIA metrics/plots, project MIA utils |
| `security_secagg.ipynb` | 4 | 26 | 10 | Educational masking/dropout/SSS/overhead, Flower SecAgg params |
| `fairness_evaluation.ipynb` | 3 | 24 | 12 | Real per-country metrics, bootstrap disparity, ROC/calibration/sweep |
| `flower_federated_learning.ipynb` | 4 | 24 | 8 | Log parsing, FedAvg reference sim, heterogeneity/comm cost |
| `tls_credentials.ipynb` | 3 | 29 | 5 | Tempdir cert gen/verify/negative test, inspection, timelines/diagrams |
| `hybrid_boosters_stacking.ipynb` | 3 | 24 | 12 | NN/boosters/stacking/β fusion, leakage-free splits, diagnostics |
| `causal_inference_quickstart.ipynb` | 6 | 24 | 10 | DAG, backdoor/AIPW fallbacks, truth validation, balance/CATE/sensitivity |
| `transportability_quickstart.ipynb` | 4 (copied) | 24 | 12 | Full rewrite: LOCO shift/weights/ESS/transported ATE |

## Validation matrix

| Notebook | Structure | Execution (NB_MODE=light) | Kernel | Result |
|---|---|---|---|---|
| `calibration_comparison.ipynb` | ok | passed (3s) | insurance-fl | **passed** |
| `causal_inference_quickstart.ipynb` | ok | passed (5s) | insurance-fl | **passed** |
| `client_identity_and_isolation.ipynb` | ok | passed (3s) | insurance-fl | **passed** |
| `dp_accounting_rdp.ipynb` | ok | passed (5s) | insurance-fl | **passed** |
| `dp_utility_tradeoff.ipynb` | ok | passed (3s) | insurance-fl | **passed** |
| `fairness_evaluation.ipynb` | ok | passed (4s) | insurance-fl | **passed** |
| `feature_surface_and_leakage.ipynb` | ok | passed (4s) | insurance-fl | **passed** |
| `flower_federated_learning.ipynb` | ok | passed (4s) | insurance-fl | **passed** |
| `hybrid_boosters_stacking.ipynb` | ok | passed (13s) | insurance-fl | **passed** |
| `local_dp.ipynb` | ok | passed (4s) | insurance-fl | **passed** |
| `membership_inference.ipynb` | ok | passed (5s) | insurance-fl | **passed** |
| `mia_power_analysis.ipynb` | ok | passed (4s) | insurance-fl | **passed** |
| `privacy_budget_gate.ipynb` | ok | passed (3s) | insurance-fl | **passed** |
| `round1_loss_diagnosis.ipynb` | ok | passed (3s) | insurance-fl | **passed** |
| `security_secagg.ipynb` | ok | passed (4s) | insurance-fl | **passed** |
| `synthetic_data_generation.ipynb` | ok | passed (7s) | insurance-fl | **passed** |
| `test_split_bayes_sanity.ipynb` | ok | passed (3s) | insurance-fl | **passed** |
| `tls_credentials.ipynb` | ok | passed (4s) | insurance-fl | **passed** |
| `transportability_quickstart.ipynb` | ok | passed (8s) | insurance-fl | **passed** |

Validator report: `reports/notebook_validation.json`; Python 3.11.16, Flower 1.31.0, mode `light`. Measured total: 19/19 passed, 0 failed.

## Dependency report
- **Reused (project/env):** numpy, pandas, scipy, scikit-learn, matplotlib, seaborn/plotly (available), torch/xgboost/lightgbm/catboost (available), dp-accounting, statsmodels, networkx, cryptography, flwr, nbformat/nbclient/ipykernel.
- **New dependencies added:** none beyond existing `dev` edits already present in working tree (`nbformat`, `nbconvert`, `ipykernel` added to `dev` extra in current diff). No `notebooks` extra was added yet; recommend adding a `notebooks` extra (e.g. matplotlib, seaborn, plotly, nbformat, nbclient, ipykernel, networkx) with justification if maintainers want explicit install group.
- **Optional (handled with skips/messages):** `openssl` CLI, `pgmpy`, `cryptography` structured parsing, `torch` feature-demo paths, xgboost/lightgbm/catboost fallbacks, SHAP not required.

## Remaining limitations and bugs found in `src/` (not fixed here)
- `insurance_fl.data.quality.DataQualityAnalyzer.analyze` can fail on unhashable object values (e.g. ndarray in `segment_probabilities`) because `df.duplicated()` requires hashable elements. Notebooks sanitize defensively; consider making analyzer robust to unhashable cells.
- `insurance_fl.transport.transport` imports `require_causal_deps()` at module import time (requires `pgmpy`), which makes transport utilities unavailable without causal deps; notebooks implemented transport diagnostics without importing this module.
- Causal estimator/DAG utilities that depend on `pgmpy` are unavailable in this environment (`pgmpy` missing); notebooks use fallbacks where possible.
- Pre-existing privacy/fairness integration concerns noted in prompt context (e.g. state persistence wiring in strategy/server, fairness artifact guard semantics) were not modified in `src/` (no notebook-blocking bug required a `src/` edit for this pass).
- The latest light-mode run is recorded in `reports/notebook_validation.json`; light mode is not evidence that full mode was executed.
- Committed notebook outputs policy: notebooks were modernized and some executed ad hoc; outputs may be inconsistent/stale across files. Recommended: clear outputs or re-execute all via validation and commit consistently.

## Acceptance criteria status
- 11 notebooks, same filenames: **met**.
- `transportability_quickstart.ipynb` performs transport analysis: **met**.
- No fabricated numbers intended; computed or artifact-labeled: **met by design in rewrites** (review recommended).
- No hardcoded machine paths: **met in rewritten notebooks/helper**.
- ≥4 distinct interpreted figures per notebook (by `save_fig` count): **met** (all >=5 in current files).
- README catalog and validation JSON reflect the latest measured light-mode run: **met for light mode**; full mode is not claimed.
- One-commit-per-notebook and final `git log --oneline`: **not performed in this session** (working tree contains broad uncommitted/modified changes).
