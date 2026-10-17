# Task Specification: Hybrid Multi-Task Federated Learning Platform for European Insurance

## Document Purpose

This task specification translates the professional project description into an actionable, sequenced implementation plan. It defines **what** must be built, **in what order**, **with which references**, and **how each task is validated**. Each task is self-contained, has clear inputs/outputs, and maps to the architectural tiers defined in the project description (Hybrid Multi-Task Learning, Federated Learning, Privacy, Fairness, Stacking).

---

## Task 0: Project Scaffolding and Baseline Federation

**Objective**: Establish a working Flower App that runs end-to-end in simulation, providing the foundation for all subsequent tasks.

**Scope**:
- Create the `insurance_fl/` package structure (`client_app.py`, `server_app.py`, `task.py`, `stacking.py`, `strategy.py`, `privacy.py`).
- Implement a minimal `ClientApp` that trains a placeholder NN on local synthetic data and returns parameters.
- Implement a minimal `ServerApp` using `FedAvg` that runs 2 rounds in simulation.
- Verify `flwr run . --stream` completes without error.

**References**:
- Flower modern App architecture: `https://flower.ai/docs/framework/`
- Scaffolding baseline: `https://preview.flower.ai/apps/mainthread/federated-fraud-detection`

**Deliverables**:
- Working `flwr run` output showing 2 completed rounds.
- `pyproject.toml` with pinned dependencies (`flwr>=1.30`, `torch`, `xgboost`, `lightgbm`, `catboost`, `scipy`).

**Validation**: `flwr run .` exits with code 0; logs show round completion and aggregated loss.

**Dependencies**: None.

---

## Task 1: Multi-Task Neural Network (MTNN)

**Objective**: Implement the shared-encoder multi-head NN with uncertainty weighting and FedBN-compatible BatchNorm.

**Scope**:
- Shared encoder: 3 dense layers (256 → 128 → 64), ReLU, BatchNorm, Dropout 0.3.
- Task heads: 2 regression (Dense→1), 2 binary (Dense→1+sigmoid), 1 multi-class (Dense→3+softmax).
- Kendall et al. (2018) learnable uncertainty weights to balance task losses.
- Mark BatchNorm layers as **local** (FedBN) — excluded from federated aggregation.
- Unit test: forward pass with 33-feature input returns 5 task outputs with correct shapes.

**References**:
- Flower framework (ClientApp integration): `https://flower.ai/docs/framework/`
- FedBN concept: standard literature (Hsieh et al., 2020).

**Deliverables**:
- `insurance_fl/task.py` containing `MultiTaskNN` class and `train_nn()` / `evaluate_nn()` functions.
- Unit tests in `tests/test_mtnn.py`.

**Validation**: `pytest tests/test_mtnn.py` passes; manual forward pass confirms 5 outputs.

**Dependencies**: Task 0.

---

## Task 2: Booster Pre-Training (XGBoost, LightGBM, CatBoost)

**Objective**: Implement per-client, per-task training of the three boosters with early stopping and artifact persistence.

**Scope**:
- `train_xgb(task, X_train, y_train, X_val, y_val)` → saves `xgboost_{task}.json`.
- `train_lgbm(...)` → saves `lightgbm_{task}.txt`.
- `train_cat(...)` → saves `catboost_{task}.cbm`.
- Handle both regression (RMSE) and classification (log-loss) tasks.
- CLI entrypoint `insurance-fl-pretrain` that runs the full pre-training for a client partition.

**References**:
- MultiModel-Regression-Framework (training patterns): `https://github.com/git-abhisharma/MultiModel-Regression-Framework`
- Flower XGBoost app (serialization patterns): `https://preview.flower.ai/apps/mainthread/federated-fraud-detection`

**Deliverables**:
- `insurance_fl/task.py` extended with booster training functions.
- `insurance_fl/scripts/pretrain.py` CLI.
- Artifacts in `models/client_{cid}/`.

**Validation**: Run `insurance-fl-pretrain --partition-id 0`; verify 15 artifact files (3 boosters × 5 tasks).

**Dependencies**: Task 0.

---

## Task 3: Stacking Meta-Learner (NNLS)

**Objective**: Implement OOF generation and the non-negative Ridge / NNLS meta-learner that combines the three boosters.

**Scope**:
- `generate_oof_predictions(models, X, y, k_folds=5)` → returns OOF prediction matrix.
- `fit_stacking_meta(oof, y, positive=True)` → solves \(\arg\min_{\alpha_i \ge 0} \text{Loss}(\sum \alpha_i \hat{y}_i^{(\text{OOF})}, y)\) using `scipy.optimize.nnls` or `LinearRegression(positive=True)`.
- **Overfitting guard**: evaluate on a separate hold-out; if Stacking underperforms the best single booster, fall back to that booster (\(\alpha=1\), others \(=0\)).
- Save `stacking_meta_{task}.json` with \(\{\alpha_{\text{XGB}}, \alpha_{\text{LGBM}}, \alpha_{\text{CAT}}\}\), OOF RMSE/log-loss, and fallback flag.

**References**:
- RocketStack (OOF and non-negative pattern): `https://pypi.org/project/rocketstack-ensemble-ai/`
- Thalassemia IEEE paper (federated stacking blueprint): `https://ieeexplore.ieee.org/document/11546283`

**Deliverables**:
- `insurance_fl/stacking.py` with OOF generation, NNLS fit, and fallback logic.
- Unit tests verifying \(\alpha_i \ge 0\) and the fallback triggers correctly.

**Validation**: `pytest tests/test_stacking.py` passes; manual run on synthetic data confirms α constraint and sum unconstrained.

**Dependencies**: Task 2.

---

## Task 4: HybridAggregation Strategy (NN Federated, Boosters Local)

**Objective**: Implement the custom Flower strategy that aggregates only the NN while keeping boosters and Stacking α local.

**Scope**:
- Extend `FedAvg` with `HybridAggregation`.
- Aggregate NN weights weighted by `num-examples`, with optional FedProx proximal term (`proximal_mu=0.01`).
- Exclude BatchNorm layers from aggregation (FedBN).
- Log per-task metrics via `MetricRecord`.
- Maintain a global synthetic holdout for β optimization.

**References**:
- Flower strategy docs: `https://flower.ai/docs/framework/`

**Deliverables**:
- `insurance_fl/strategy.py` with `HybridAggregation` class.
- Integration test: 2-client simulation with mock data, verify NN weights aggregate and boosters remain untouched.

**Validation**: `flwr run` completes with the custom strategy; server logs show NN aggregation only.

**Dependencies**: Task 1, Task 3.

---

## Task 5: Level-2 Hybrid Fusion (β Optimization)

**Objective**: Implement the post-federated β optimization that combines NN and BoosterStack predictions per task.

**Scope**:
- `optimize_hybrid_beta(nn_model, stacked_boosters, global_holdout)` → per-task \(\beta \in [0,1]\) minimizing RMSE/log-loss.
- `combine_predictions(nn_pred, booster_pred, beta)` → \(\hat{y}_{\text{hybrid}} = \beta \hat{y}_{\text{NN}} + (1-\beta) \hat{y}_{\text{BoosterStack}}\).
- Save `hybrid_fusion_{task}.json` with β and holdout metrics.

**References**:
- ESG HNB-Stack IEEE paper (fusion logic): `https://ieeexplore.ieee.org/document/11482408`

**Deliverables**:
- `insurance_fl/task.py` extended with `optimize_hybrid_beta()` and `combine_predictions()`.
- Test verifying β=0 → pure booster, β=1 → pure NN.

**Validation**: `pytest tests/test_fusion.py` passes; β values saved per task.

**Dependencies**: Task 4.

---

## Task 6: TLS Configuration

**Objective**: Secure all SuperLink ↔ SuperNode and User ↔ SuperLink communications with TLS mutual authentication.

**Scope**:
- `insurance_fl/scripts/generate_creds.py` producing CA, server, and client certificates.
- Document SuperLink launch with `--ssl-ca-certfile`, `--ssl-certfile`, `--ssl-keyfile`.
- Document SuperNode launch with `--root-certificates`.
- Document `flwr run --root-certificates` for the user CLI.

**References**:
- Flower TLS docs: `https://flower.ai/docs/framework/`

**Deliverables**:
- `certificates/` directory (generated at setup time, git-ignored).
- `docs/deployment.md` with TLS commands.

**Validation**: Manual test — SuperLink/SuperNode handshake succeeds with TLS; `--insecure` connection is rejected.

**Dependencies**: Task 0.

---

## Task 7: Secure Aggregation (SecAgg+) for NN Updates

**Objective**: Wrap the NN aggregation path with SecAgg+ so the server only sees the aggregate.

**Scope**:
- Server: configure `SecAggPlusWorkflow(reconstruction_threshold=3, clipping_range=8.0, quantization_range=2**22)`.
- Client: wrap `ClientApp` with `secaggplus_mod`.
- Verify boosters and Stacking α never enter the SecAgg+ pipeline.

**References**:
- Flower SecAgg+ docs: `https://flower.ai/docs/framework/`

**Deliverables**:
- `insurance_fl/server_app.py` updated to use `SecAggPlusWorkflow`.
- `insurance_fl/client_app.py` updated with `secaggplus_mod`.

**Validation**: `flwr run` completes with SecAgg+ enabled; logs confirm masked/unmasked stages.

**Dependencies**: Task 4, Task 6.

---

## Task 8: Differential Privacy (LocalDpMod) — Optional

**Objective**: Add client-side DP noise to NN updates before SecAgg+ masking.

**Scope**:
- Configure `LocalDpMod(clipping_norm=1.0, sensitivity=1.0, epsilon=2.0, delta=1e-5)`.
- Ensure `local_dp_mod` runs **before** `secaggplus_mod` in the `mods` list.
- Document the utility-privacy tradeoff; treat as preview-phase feature.

**References**:
- Flower LocalDpMod docs: `https://flower.ai/docs/framework/`

**Deliverables**:
- `insurance_fl/privacy.py` with DP configuration.
- `docs/privacy.md` explaining the DP setup and caveats.

**Validation**: `flwr run` with DP enabled completes; ε spent logged. Compare accuracy vs. non-DP baseline.

**Dependencies**: Task 7.

---

## Task 9: Privacy Accounting (RDP Accountant) — Monitoring Layer

**Objective**: Track cumulative ε expenditure per round as a monitoring layer.

**Scope**:
- Integrate `RdpAccountant` with `PrivacyConfig(max_epsilon=10.0, target_delta=1e-5)`.
- Call `accountant.compose(GaussianPrivacyEvent(...))` after each round's aggregation.
- Log `privacy/epsilon_cumulative` to MLflow.
- Treat `would_exceed` as advisory, not hard-fail.

**References**:
- Flower privacy accounting docs: `https://flower.ai/docs/framework/`

**Deliverables**:
- `insurance_fl/privacy.py` extended with accountant integration.
- MLflow dashboard showing ε per round.

**Validation**: Manual run with 5 rounds; ε increases monotonically and is logged.

**Dependencies**: Task 8.

---

## Task 10: Fairness (QFedAvg or Stratified Sampling)

**Objective**: Ensure equitable performance across the four European countries (Germany, France, Belgium, Switzerland).

**Scope** (choose one):

**Option A (Fairness priority)**:
- Replace FedProx with `QFedAvg(q=0.1, client_learning_rate=...)`.
- Accept loss of proximal term stabilization.

**Option B (Stability priority — recommended)**:
- Keep FedProx proximal term.
- Implement **stratified client sampling**: each round samples evenly across countries.
- Monitor per-country metrics; verify disparity < threshold.

**References**:
- Flower QFedAvg docs: `https://flower.ai/docs/framework/`

**Deliverables**:
- `insurance_fl/strategy.py` updated with chosen fairness mechanism.
- `docs/fairness.md` documenting the choice and validation results.

**Validation**: Per-country metrics logged; disparity metric computed and reported.

**Dependencies**: Task 4.

---

## Task 11: Full Integration and End-to-End Test

**Objective**: Combine all components into a single Flower app that runs end-to-end.

**Scope**:
- `ClientApp` with MTNN training, booster loading, Stacking evaluation, DP mod, SecAgg+ mod.
- `ServerApp` with `HybridAggregation` + `SecAggPlusWorkflow` + RDP accountant.
- Full pipeline: pre-train boosters → federated NN training → β optimization → final hybrid model saved.
- Integration test: 4-client simulation with synthetic data, 5 rounds.

**References**: All previous tasks.

**Deliverables**:
- Complete `insurance_fl/` package.
- `tests/test_integration.py` with end-to-end assertion.
- `docs/architecture.md` with the final diagram.

**Validation**: `flwr run . --stream` completes; final artifacts (`nn_final.pt`, `stacking_meta_*.json`, `hybrid_fusion_*.json`) exist and pass schema checks.

**Dependencies**: Tasks 1–10.

---

## Task 12: Documentation and Deployment Guide

**Objective**: Produce operator-facing documentation for production deployment.

**Scope**:
- `README.md`: overview, quickstart, architecture diagram.
- `docs/deployment.md`: TLS setup, SuperLink/SuperNode launch, monitoring.
- `docs/privacy.md`: DP, SecAgg+, accounting caveats.
- `docs/fairness.md`: fairness mechanism choice and validation.
- `docs/stacking.md`: mathematical definition, α constraints, fallback logic.

**References**: All previous tasks.

**Deliverables**: Complete documentation set.

**Validation**: A new engineer can follow `docs/deployment.md` to launch a TLS-secured SuperLink/SuperNode pair and run 2 rounds.

**Dependencies**: Task 11.

---

## Task Dependency Graph

```
Task 0 (Scaffolding)
├── Task 1 (MTNN)
├── Task 2 (Boosters)
│   └── Task 3 (Stacking NNLS)
│       └── Task 5 (β Fusion)
├── Task 4 (HybridAggregation)
│   ├── Task 7 (SecAgg+)
│   │   └── Task 8 (LocalDP)
│   │       └── Task 9 (RDP Accounting)
│   └── Task 10 (Fairness)
└── Task 6 (TLS)

Task 11 (Full Integration) ← Tasks 1, 3, 5, 7, 9, 10
Task 12 (Documentation) ← Task 11
```

---

## Critical Constraints and Risks

| Constraint | Impact | Mitigation |
|-----------|--------|------------|
| QFedAvg and FedProx are mutually exclusive | Must choose one aggregation path | Default to FedProx + stratified sampling (Option B) |
| DP features are preview-phase | Production use not recommended | Enable as monitoring; validate empirically; consult Flower team |
| RDP accounting does not cover metrics | ε tracking incomplete | Suppress metrics logging or document the gap |
| No ready-made federated Stacking implementation exists | Custom development required | Use Thalassemia paper as blueprint + RocketStack for NNLS logic |
| SecAgg+ clipping_range affects utility | Too-small range degrades model | Tune empirically; start with `clipping_range=8.0` |
| Booster + Stacking α never federated | Cross-client learning limited to NN | Deliberate privacy decision; document tradeoff |

---

## Success Criteria

The project is complete when:

1. A TLS-secured SuperLink/SuperNode pair runs the full pipeline end-to-end.
2. The MTNN trains federated across 4 clients with FedBN and FedProx.
3. The three-booster Stacking Ensemble trains locally per client with NNLS α and overfitting guard.
4. Level-2 β fusion is optimized on a global holdout and saved per task.
5. SecAgg+ masks NN updates; the server never sees individual client contributions.
6. RDP accountant logs cumulative ε per round (advisory).
7. Fairness mechanism (stratified sampling or QFedAvg) produces per-country metrics within acceptable disparity.
8. All documentation (`README.md`, `docs/*.md`) is complete and validated by a new engineer.

This task specification provides the sequenced, reference-linked implementation plan that operationalizes the professional project description into an industry-grade hybrid federated learning platform.