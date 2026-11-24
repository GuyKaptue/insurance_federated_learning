# Hybrid Multi-Task Federated Learning Platform for European Insurance

## Executive Summary

This document specifies the **industry-grade implementation** of the European Insurance Federated Learning Platform. It extends the original scope to a **full professional system** built on **Flower's modern App-based architecture (v1.30+)**, secured with **TLS mutual authentication**, enhanced with **Secure Aggregation (SecAgg+)**, **Differential Privacy (LocalDpMod)**, **Privacy Accounting (RDP Accountant)**, and **Fairness (QFedAvg or stratified sampling)** — and powered by a **hybrid multi-task learning** design that combines:

- A **shared-encoder multi-head Neural Network (NN)** — the federated primary learner.
- A **Stacked Booster Ensemble** comprising **XGBoost, LightGBM, and CatBoost**, fused by a **non-negative Ridge meta-learner** (Stacked Generalization), with all three boosters participating in the final prediction.

The **hybrid** concept is formally defined as a **three-tier model fusion**:

1. **Base tier (Level 0)** — XGBoost, LightGBM, CatBoost trained locally per client per task.
2. **Booster meta tier (Level 1)** — a non-negative Ridge meta-learner that learns \(\alpha_{\text{XGB}}, \alpha_{\text{LGBM}}, \alpha_{\text{CAT}} \ge 0\) (unconstrained sum) on out-of-fold predictions.
3. **Hybrid fusion tier (Level 2)** — a learned \(\beta\)-weighted combination of the NN output and the Stacked Booster output, optimized on a global held-out validation set after federated training.

The privacy and fairness layer is **layered on top** of the hybrid architecture:

- **SecAgg+** protects NN parameter updates cryptographically (server sees only the aggregate).
- **LocalDpMod** optionally adds client-side differential privacy noise before SecAgg+ masking.
- **RDP Accountant** tracks cumulative ε expenditure as a monitoring layer.
- **QFedAvg** (or stratified sampling) ensures equitable performance across the four European countries.

This design captures the representational flexibility of deep learning, the tabular-data superiority of gradient boosting, the mathematical rigor of Stacked Generalization, and the security guarantees of modern federated privacy mechanisms — all within Flower's production-ready deployment runtime.

---

## 1. Hybrid Multi-Task Learning Architecture

### 1.1 Definition of "Hybrid" (Updated)

In this platform, **hybrid** refers to the coexistence and hierarchical combination of **two model families across three fusion tiers**:

| Tier | Component | Model Family | Role |
|------|-----------|--------------|------|
| **Level 0** | XGBoost, LightGBM, CatBoost | Gradient-boosted trees | Specialized tabular learners per task, trained locally per client |
| **Level 1** | Non-negative Ridge Stacking meta-learner | Linear meta-model | Learns \(\alpha_{\text{XGB}}, \alpha_{\text{LGBM}}, \alpha_{\text{CAT}}\) to combine boosters optimally |
| **Neural** | Shared Encoder + Multi-Head NN | Neural network | Learns common representations across all 5 tasks |
| **Level 2** | \(\beta\)-weighted hybrid fusion | Meta-combination | Combines NN and Stacked Booster outputs per task |

**Key upgrade from the original specification**: The booster component is no longer a fixed-\(\alpha\) CatBoost–LightGBM blend. It is a **true Stacking Ensemble** over all three boosters, with a **non-negative Ridge meta-learner** trained on out-of-fold predictions to avoid leakage. This is mathematically superior, empirically validated for insurance premium prediction, and aligns with industry-standard ensemble practice.

### 1.2 Mathematical Definition of the Stacked Booster Ensemble

Let the three base boosters be \(f_{\text{XGB}}, f_{\text{LGBM}}, f_{\text{CAT}}\). For each task and input \(\mathbf{x}\):

\[
\hat{y}_{\text{XGB}} = f_{\text{XGB}}(\mathbf{x}), \quad
\hat{y}_{\text{LGBM}} = f_{\text{LGBM}}(\mathbf{x}), \quad
\hat{y}_{\text{CAT}} = f_{\text{CAT}}(\mathbf{x})
\]

The **meta-feature vector** is:

\[
\mathbf{z} = [\hat{y}_{\text{XGB}}, \hat{y}_{\text{LGBM}}, \hat{y}_{\text{CAT}}]
\]

The **Stacked Booster Ensemble** is the Level-1 prediction:

\[
\hat{y}_{\text{BoosterStack}} = \alpha_{\text{XGB}} \cdot \hat{y}_{\text{XGB}}
+ \alpha_{\text{LGBM}} \cdot \hat{y}_{\text{LGBM}}
+ \alpha_{\text{CAT}} \cdot \hat{y}_{\text{CAT}}
\]

with the **constraint**:

\[
\alpha_{\text{XGB}}, \alpha_{\text{LGBM}}, \alpha_{\text{CAT}} \ge 0
\]

**Range and sum constraint**:

- Each \(\alpha_i \in [0, \infty)\) — non-negative, **no upper bound**.
- The sum \(\sum_i \alpha_i\) is **not forced to equal 1**. This is the critical distinction from simple blending; it allows the meta-learner to scale the ensemble output appropriately and is the hallmark of true Stacked Generalization.

**Meta-learner training** (per client, per task):

\[
(\alpha_{\text{XGB}}^*, \alpha_{\text{LGBM}}^*, \alpha_{\text{CAT}}^*) =
\arg\min_{\alpha_i \ge 0} \ \text{Loss}\left( \sum_{i} \alpha_i \hat{y}_i^{(\text{OOF})}, \ y \right)
\]

where \(\hat{y}_i^{(\text{OOF})}\) are **out-of-fold predictions** from a \(K\)-fold cross-validation on the client's local data, ensuring the meta-learner never sees predictions produced by base models trained on the same fold (preventing leakage). For regression tasks, the loss is RMSE; for classification tasks, log-loss.

**Implementation note**: Use `scipy.optimize.nnls` (Non-Negative Least Squares) or a non-negative Ridge solver (e.g., `sklearn.linear_model.LinearRegression(positive=True)` for OLS-equivalent, or a custom NNLS-with-\(L_2\) routine) to solve the constrained problem. Standard `sklearn.ridge.Ridge` does not support `positive=True` natively, so use `LinearRegression(positive=True)` or wrap `nnls` with a regularization term.

### 1.3 Multi-Task Neural Network Design

The NN follows the shared-encoder design refined for production:

```
Input (33 features)
       │
       ▼
┌─────────────────┐
│  Shared Encoder │  (3 dense layers: 256 → 128 → 64, ReLU, BatchNorm, Dropout 0.3)
└────────┬────────┘
         │ (64-dim shared representation)
         ▼
┌─────────────────────────────────────────────┐
│              Task-Specific Heads            │
├──────────────┬──────────────┬───────────────┤
│ Regression   │ Binary       │ Multi-Class   │
│ Head (2×)    │ Head (2×)    │ Head (1×)     │
│ Dense→1      │ Dense→1+σ    │ Dense→3+softmax│
└──────────────┴──────────────┴───────────────┘
```

**Production refinements**:

- **Gradient normalization**: Task-specific loss scaling prevents any single task from dominating the shared encoder during federated aggregation.
- **Uncertainty weighting** (Kendall et al., 2018): Learnable task uncertainty parameters automatically balance regression and classification losses.
- **Federated BatchNorm (FedBN)**: BatchNorm layers remain local to each client, handling non-IID feature distributions across Germany, France, Belgium, and Switzerland.

### 1.4 Booster Pre-Training and Stacking Optimization

Before federated training begins, each client **pre-trains** its local boosters and the Stacking meta-learner on its own data partition.

**Pre-training procedure** (per client, per task):

1. **XGBoost**: Train with early stopping on a validation split; save `xgboost_{task}.json`.
2. **LightGBM**: Train with early stopping; save `lightgbm_{task}.txt`.
3. **CatBoost**: Train with early stopping; save `catboost_{task}.cbm`.
4. **Generate OOF predictions**: Run \(K\)-fold CV (\(K=5\)) for each booster; collect out-of-fold predictions \(\hat{y}_{\text{XGB}}^{(\text{OOF})}, \hat{y}_{\text{LGBM}}^{(\text{OOF})}, \hat{y}_{\text{CAT}}^{(\text{OOF})}\).
5. **Solve NNLS for \(\alpha\)**: Fit the non-negative Ridge meta-learner on the OOF meta-features to obtain \(\alpha_{\text{XGB}}^*, \alpha_{\text{LGBM}}^*, \alpha_{\text{CAT}}^*\).
6. **Overfitting guard**: Evaluate the Stacked Ensemble on a **separate hold-out fold** not used for early stopping or NNLS. If the ensemble's validation performance is worse than the best single booster, fall back to the best single booster (i.e., set its \(\alpha = 1\), others \(= 0\)).
7. **Save artifacts**: `stacking_meta_{task}.json` containing \(\{\alpha_{\text{XGB}}, \alpha_{\text{LGBM}}, \alpha_{\text{CAT}}\}\), OOF RMSE/log-loss, and fallback flag.

**Saved artifacts per client**: `xgboost_{task}.json`, `lightgbm_{task}.txt`, `catboost_{task}.cbm`, `stacking_meta_{task}.json`.

### 1.5 Hybrid Combination at Inference (Level 2)

For each task, the final prediction is:

\[
\hat{y}_{\text{hybrid}} = \beta \cdot \hat{y}_{\text{NN}} + (1-\beta) \cdot \hat{y}_{\text{BoosterStack}}
\]

where:

\[
\hat{y}_{\text{BoosterStack}} = \alpha_{\text{XGB}} \hat{y}_{\text{XGB}}
+ \alpha_{\text{LGBM}} \hat{y}_{\text{LGBM}}
+ \alpha_{\text{CAT}} \hat{y}_{\text{CAT}}
\]

- \(\beta \in [0, 1]\) is determined **per task** on a **global held-out validation set** (synthetic data not used in any client's training) after federated training completes.
- \(\alpha_{\text{XGB}}, \alpha_{\text{LGBM}}, \alpha_{\text{CAT}} \ge 0\) are learned locally per client via NNLS and are **not aggregated** (boosters remain local — see §2.4).

This three-tier weighting (\(\alpha\) within boosters via Stacking, \(\beta\) between NN and boosters) provides fine-grained, mathematically principled control over the bias–variance tradeoff for each task.

---

## 2. Federated Learning with Modern Flower, TLS, and Privacy Mechanisms

### 2.1 Flower Architecture Alignment

The platform adopts Flower's **modern App-based architecture** (SuperLink / SuperNode / ServerApp / ClientApp) rather than the legacy `NumPyClient` pattern. This provides:

- **Separation of concerns**: `ServerApp` defines the strategy and workflow; `ClientApp` defines local training/evaluation logic.
- **Simulation and deployment parity**: The same app runs in simulation (`flwr run`) or deployment (`flower-superlink` / `flower-supernode`) without code changes.
- **Framework-agnostic client**: Flower supports XGBoost, LightGBM, CatBoost, and PyTorch/TensorFlow in the same federation.

**Project structure** (following Flower App conventions):

```
insurance_fl/
├── pyproject.toml
├── insurance_fl/
│   ├── __init__.py
│   ├── client_app.py      # ClientApp: local NN training + booster/Stacking inference
│   ├── server_app.py      # ServerApp: hybrid aggregation strategy + SecAgg+ workflow
│   ├── task.py            # Model definitions, data loading, booster & Stacking utilities
│   ├── stacking.py        # NNLS meta-learner + OOF generation
│   ├── strategy.py        # Custom Flower strategy (HybridAggregation)
│   └── privacy.py         # RDP accountant integration + privacy config
└── certificates/          # CA, server, client certs (TLS)
```

### 2.2 TLS Certificate Configuration

All SuperLink ↔ SuperNode and User ↔ SuperLink communications are secured with TLS, replacing `--insecure` flags.

**Certificate generation** (per organization PKI standards; the following is a prototyping template):

```bash
# Generate CA, server, and client certificates
python generate_creds.py
# Output: certificates/ca.crt, server.pem, server.key, client.pem, client.key
```

**SuperLink launch (TLS-enabled)**:

```bash
flower-superlink \
    --ssl-ca-certfile certificates/ca.crt \
    --ssl-certfile certificates/server.pem \
    --ssl-keyfile certificates/server.key
```

**SuperNode launch (TLS-enabled)**:

```bash
flower-supernode \
    --root-certificates certificates/ca.crt \
    --superlink <superlink-host>:8000 \
    --node-config="partition-id=0 num-partitions=4"
```

**User CLI connection**:

```bash
flwr run . <superlink-connection> --stream \
    --root-certificates certificates/ca.crt
```

**Production note**: Generate unique key pairs for each service (SuperLink Fleet API, Runtime API, each SuperNode) following organizational PKI standards. Self-signed certificates are acceptable only for prototyping and internal research.

### 2.3 Secure Aggregation (SecAgg+) — Production-Ready

Flower provides a production-ready implementation of the **SecAgg+ protocol** through `SecAggPlusWorkflow` on the server and `secaggplus_mod` on the client . It ensures the server only sees the aggregated NN update, never individual client contributions.

**How it works (four stages)** :

1. **Setup**: Server sends SecAgg+ configuration to clients; clients generate and send public keys.
2. **Share Keys**: Clients broadcast public keys and send encrypted private key shares.
3. **Collect Masked Vectors**: Clients send masked model parameters; the server cannot inspect individual updates.
4. **Unmask**: Private key shares are collected to unmask and aggregate the model parameters.

**Integration with your architecture**:

```python
from flwr.server.workflow.secure_aggregation.secaggplus_workflow import SecAggPlusWorkflow

# ServerApp: wrap HybridAggregation with SecAgg+ workflow
workflow = SecAggPlusWorkflow(
    reconstruction_threshold=2,   # Minimum shares to reconstruct (tolerates dropouts)
    max_weight=1000.0,
    clipping_range=8.0,           # Clip params to [-8, 8] before quantization
    quantization_range=2**22,     # Quantize floats to integers for crypto operations
)
```

```python
from flwr.client.mod import secaggplus_mod

# ClientApp: wrap with secaggplus_mod
app = ClientApp(
    client_fn=client_fn,
    mods=[secaggplus_mod],
)
```

**Compatibility with HybridAggregation**: The `SecAggPlusWorkflow` handles masking/unmasking *before* the strategy's `aggregate_train` is called. Your `HybridAggregation` strategy receives the **already-unmasked aggregated NN weights** — FedAvg/FedProx logic remains unchanged. The boosters and Stacking α never enter this pipeline (they remain local).

**Critical parameters** :
- `reconstruction_threshold`: Higher = better privacy but less dropout tolerance. For 4 clients, `threshold=3` is reasonable.
- `clipping_range`: Must be large enough to preserve model utility but small enough for quantization precision.
- `quantization_range`: `2**22` is the default and suitable for most NN parameters.

### 2.4 Differential Privacy (LocalDpMod) — Preview

Flower provides **LocalDpMod** for client-side differential privacy . It clips NN parameter updates and adds Gaussian noise *before* transmission.

**Integration**:

```python
from flwr.client.mod import LocalDpMod

local_dp_mod = LocalDpMod(
    clipping_norm=1.0,
    sensitivity=1.0,
    epsilon=2.0,
    delta=1e-5,
)

app = ClientApp(
    client_fn=client_fn,
    mods=[local_dp_mod, secaggplus_mod],  # DP first, then SecAgg+ masking
)
```

**Ordering is critical**: LocalDpMod must run **before** secaggplus_mod. The DP noise is added to the clipped update, then SecAgg+ masks it for the server. Combining DP and SecAgg+ increases total noise; tune ε conservatively.

**Important caveat**: Flower's DP features are in a **"preview phase"** . For production deployment with sensitive insurance data, validate the utility-privacy tradeoff empirically and consider consulting the Flower team.

### 2.5 Privacy Accounting (RDP Accountant) — Preview

Flower's **RDP Accountant** tracks cumulative ε expenditure across rounds using Google's `dp-accounting` package .

**Key methods** :
- `compose(event)`: Record a privacy event (one round's DP release).
- `get_epsilon(delta)`: Return cumulative ε at the supplied δ.
- `would_exceed(event, max_epsilon)`: Check if composing an event would breach the budget (non-mutating).

**Integration with HybridAggregation**:

```python
from flwr.serverapp import RdpAccountant, PrivacyConfig

config = PrivacyConfig(
    max_epsilon=10.0,
    target_delta=1e-5,
    population_size=total_clients,
)
accountant = RdpAccountant(config)

# In strategy after each round's aggregation:
accountant.compose(GaussianPrivacyEvent(
    noise_multiplier=noise_mult,
    sample_size=num_participants,
    population_size=total_clients,
))
epsilon_spent = accountant.get_epsilon(delta=1e-5)
log_metric("privacy/epsilon_cumulative", epsilon_spent)
```

**Caveats from documentation analysis**:
- Accounting currently covers only the **noised model arrays**, not data-dependent metrics (loss/accuracy) .
- The `would_exceed` check is **advisory** — it does not automatically halt training. You must implement the enforcement logic.

**Recommendation**: Enable RDP accounting as a **monitoring layer** (log ε per round). Treat automatic budget enforcement as experimental until the feature stabilizes.

### 2.6 Fairness: QFedAvg vs. Stratified Sampling

Flower provides **`QFedAvg`** (q-Fair Federated Learning), which modifies aggregation to prioritize fairness across clients with heterogeneous data distributions .

**QFedAvg parameters** :
- `client_learning_rate`: Used to approximate the Lipschitz constant.
- `q`: Fairness degree (0 = standard FedAvg, higher = more fairness).
- `train_loss_key`: MetricRecord key for training loss.

**Critical constraint**: **QFedAvg and FedProx cannot be combined** — they are distinct aggregation algorithms. You must choose:

- **Option A (Fairness priority)**: Use `QFedAvg` for NN aggregation, drop FedProx. Accept potential non-IID instability for equitable performance.
- **Option B (Stability priority)**: Keep FedProx proximal term, implement fairness via **stratified client sampling** (ensure each of the four countries contributes equitably each round). This preserves FedProx's stability benefits while addressing fairness at the sampling layer.

**Recommendation for this project**: Given the four-country non-IID scenario and the insurance domain's regulatory emphasis on equitable treatment, **Option B (FedProx + stratified sampling)** is the safer default. QFedAvg can be evaluated as an experimental alternative if per-country performance disparities persist after stratified sampling.

### 2.7 Booster and Stacking Privacy Decision (Unchanged)

**Design decision**: Boosters (XGBoost, LightGBM, CatBoost) and their Stacking meta-learner (\(\alpha\) coefficients) remain **local to each client** and are **not federated**. Only the NN is federated.

**Rationale**:

- Tree-based models are not amenable to gradient-based federated averaging; their parameters (split thresholds, leaf values) do not admit a meaningful FedAvg operation.
- The Stacking α coefficients encode information about the local data distribution (which booster works best locally). Aggregating them would leak distributional information.
- SecAgg+ and DP mechanisms are designed for **NN parameter vectors**, not tree ensembles.

This is a deliberate privacy–utility tradeoff: cross-client learning flows through the NN (protected by SecAgg+ + optional LocalDP), while boosters remain private and specialized per client.

**Future extension (out of scope for this release)**: Federated Stacking via server-coordinated OOF prediction exchange with secure aggregation — flagged as a research direction.

---

## 3. Custom Federated Strategy: HybridAggregation (Updated)

### 3.1 Strategy Design

The server uses a **custom Flower strategy** (`HybridAggregation`) that extends `FedAvg` to handle the hybrid architecture, **wrapped by SecAgg+ workflow** for NN privacy:

**Responsibilities**:

1. Aggregate NN weights from participating clients using weighted FedAvg (weighted by `num-examples`), optionally with FedProx proximal term.
2. Support **stratified client sampling** for fairness (Option B) or **QFedAvg** as alternative aggregation (Option A).
3. Maintain a **global hybrid validation set** (synthetic holdout) for Level-2 β optimization after training.
4. Log per-task metrics (NN, Stacked Booster, hybrid) and privacy accounting (ε per round) to MLflow.

**Key parameters**:

| Parameter | Default | Description |
|-----------|---------|-------------|
| `fraction_fit` | 1.0 | All 4 clients participate each round |
| `fraction_evaluate` | 1.0 | Full evaluation participation |
| `proximal_mu` | 0.01 | FedProx proximal term (0 disables) |
| `num_rounds` | 100 | Total federated rounds |
| `beta_optimization_round` | 100 | Round after which β is computed |
| `fairness_mode` | `"stratified"` | `"stratified"` (sampling) or `"qfedavg"` (aggregation) |
| `secagg_threshold` | 3 | SecAgg+ reconstruction threshold (4 clients) |
| `dp_epsilon` | 2.0 | LocalDpMod ε (if enabled) |
| `dp_delta` | 1e-5 | LocalDpMod δ |
| `max_epsilon` | 10.0 | Privacy budget ceiling (advisory) |

### 3.2 ServerApp Implementation Sketch (Updated)

```python
from flwr.serverapp import ServerApp, Grid, Context, RdpAccountant, PrivacyConfig
from flwr.server.workflow.secure_aggregation.secaggplus_workflow import SecAggPlusWorkflow
from insurance_fl.strategy import HybridAggregation
from insurance_fl.privacy import build_privacy_config

app = ServerApp()

@app.main()
def main(grid: Grid, context: Context) -> None:
    num_rounds = context.run_config["num-server-rounds"]

    # Privacy accounting
    privacy_config = build_privacy_config(context.run_config)
    accountant = RdpAccountant(privacy_config)

    # SecAgg+ workflow (NN privacy)
    secagg_workflow = SecAggPlusWorkflow(
        reconstruction_threshold=context.run_config["secagg-threshold"],
        clipping_range=8.0,
        quantization_range=2**22,
    )

    # Hybrid strategy
    strategy = HybridAggregation(
        fraction_fit=1.0,
        fraction_evaluate=1.0,
        proximal_mu=0.01,
        fairness_mode=context.run_config["fairness-mode"],
        accountant=accountant,  # for per-round ε logging
    )

    # Wrap strategy with SecAgg+ workflow
    result = strategy.start(
        grid=grid,
        initial_arrays=initial_nn_arrays,
        num_rounds=num_rounds,
        workflow=secagg_workflow,
    )

    # Level-2 β optimization on global holdout
    beta_weights = optimize_hybrid_beta(result.arrays, global_holdout)

    # Save final ε spent
    final_epsilon = accountant.get_epsilon(delta=privacy_config.target_delta)
    save_final_model(result.arrays, beta_weights, final_epsilon)
```

### 3.3 ClientApp Implementation Sketch (Updated)

```python
from flwr.clientapp import ClientApp
from flwr.client.mod import LocalDpMod, secaggplus_mod
from insurance_fl.task import (
    load_local_data, train_nn, evaluate_nn,
    load_stacked_boosters, evaluate_stacked_boosters,
)

# Privacy modifiers: DP first, then SecAgg+ masking
local_dp_mod = LocalDpMod(
    clipping_norm=1.0,
    sensitivity=1.0,
    epsilon=2.0,
    delta=1e-5,
)

app = ClientApp()

@app.train()
def train(msg, context):
    partition_id = context.node_config["partition-id"]
    train_loader = load_local_data(partition_id)

    updated_params, num_examples, metrics = train_nn(
        msg.content["arrays"].to_torch_state_dict(),
        train_loader,
    )
    return Message(
        content=RecordDict({
            "arrays": ArrayRecord(updated_params),
            "metrics": MetricRecord(metrics),
        }),
        reply_to=msg,
    )

@app.evaluate()
def evaluate(msg, context):
    partition_id = context.node_config["partition-id"]
    val_loader = load_local_data(partition_id, split="val")

    nn_metrics = evaluate_nn(msg.content["arrays"], val_loader)
    stacked = load_stacked_boosters(partition_id)
    booster_metrics = evaluate_stacked_boosters(stacked, val_loader)
    hybrid_metrics = combine_predictions(nn_metrics, booster_metrics)

    return Message(
        content=RecordDict({"metrics": MetricRecord(hybrid_metrics)}),
        reply_to=msg,
    )

# Apply modifiers: LocalDpMod → secaggplus_mod → ClientApp
app = ClientApp(
    client_fn=client_fn,
    mods=[local_dp_mod, secaggplus_mod],
)
```

---

## 4. Deployment and Security Checklist (Final)

| Layer | Item | Implementation | Maturity |
|-------|------|---------------|----------|
| **Transport** | TLS mutual auth | `--ssl-*` flags, `--root-certificates` | ✅ Production |
| **NN Privacy** | Secure Aggregation | `SecAggPlusWorkflow` + `secaggplus_mod` | ✅ Production  |
| **NN Privacy** | Local DP | `LocalDpMod(epsilon=2.0, delta=1e-5)` | ⚠️ Preview  |
| **NN Privacy** | Privacy Accounting | `RdpAccountant` + `PrivacyConfig` | ⚠️ Preview  |
| **NN Aggregation** | Non-IID Stability | FedProx proximal term | ✅ Available |
| **NN Aggregation** | Fairness (Option A) | `QFedAvg(q=0.1)` | ✅ Available  |
| **Fairness** | Fairness (Option B) | Stratified client sampling + FedProx | ✅ This document |
| **Boosters + Stacking** | — | **Local only, never federated** | ✅ By design |
| **Stacking α** | Constraint | \(\alpha_i \ge 0\), sum unconstrained, NNLS | ✅ This document |
| **Stacking guard** | Overfitting | OOF predictions + hold-out fallback | ✅ This document |
| **β optimization** | Level-2 fusion | Global holdout, post-federated | ✅ This document |

---

## 5. Key Recommendations from Flower Documentation Analysis (Final)

1. **Use the modern App-based architecture** (`ClientApp`/`ServerApp`) rather than legacy `NumPyClient`. This aligns with Flower's current direction and provides simulation/deployment parity.

2. **Enable TLS from the start** in any multi-machine or production-like setup.

3. **Adopt SecAgg+ as the primary NN privacy mechanism.** It is production-ready, compatible with your architecture, and provides cryptographic guarantees without the utility loss of local DP . Wrap the `ClientApp` with `secaggplus_mod` and configure the `ServerApp` with `SecAggPlusWorkflow`.

4. **Enable LocalDpMod only if regulatory requirements mandate formal DP guarantees.** Combining DP with SecAgg+ increases total noise; validate the utility-privacy tradeoff empirically. Treat DP features as preview-phase .

5. **Enable privacy accounting as a monitoring layer, not a hard gate.** Use `RdpAccountant` to log cumulative ε per round. Do not enable automatic budget enforcement in production until the preview feature stabilizes .

6. **Choose fairness mechanism deliberately.** If fairness across countries is a regulatory requirement, use `QFedAvg` and accept the loss of FedProx stabilization. If non-IID stability is the priority, implement fairness via **stratified client sampling** while keeping FedProx.

7. **Maintain the local-only design for boosters and Stacking α.** This is both a **privacy decision** (no distributional leakage from α) and a **compatibility decision** (SecAgg+ and DP accounting are designed for NN parameter vectors, not tree ensembles).

8. **Monitor round-level metrics** through Flower's `MetricRecord` mechanism. Ensure `num-examples` is always returned so FedAvg can weight aggregation correctly.

9. **For production certificate management**, follow organizational PKI standards; do not reuse self-signed certificates across services.

10. **Adopt Stacking (not fixed blending) for the Booster Ensemble** — use **NNLS on OOF predictions**, enforce **\(\alpha_i \ge 0\) with unconstrained sum**, and always apply the **fallback-to-best-single-booster guard**.

---

## 6. Conclusion

This final specification transforms the original project into an **industry-grade hybrid multi-task federated learning platform** with:

- **Flower modern architecture** and **TLS-secured mutual authentication**,
- **Secure Aggregation (SecAgg+)** for cryptographic privacy of NN updates ,
- **Optional LocalDpMod** for client-side differential privacy ,
- **RDP Accountant** for privacy budget monitoring ,
- **QFedAvg or stratified sampling** for fairness across four European countries ,
- A **Hybrid Multi-Task Neural Network** as the federated primary learner,
- A **three-booster Stacking Ensemble** (XGBoost + LightGBM + CatBoost) fused by a **non-negative Ridge / NNLS meta-learner**,
- **OOF-based training** of the Stacking meta-learner and a **fallback-to-best-single-booster guard**,
- A **learned β weighting** at Level 2 between NN and Stacked Booster outputs,
- **FedBN + FedProx** for robust non-IID training,
- **Deliberate privacy–utility tradeoff**: federate only the NN; keep boosters and their Stacking meta-learner local.

The resulting system preserves the original business objectives (privacy-preserving collaboration, multi-task efficiency, regulatory compliance) while delivering a **production-ready, security-hardened, mathematically rigorous hybrid-model federated learning platform** suitable for enterprise deployment in the European insurance industry.

**The only hard constraint** is that **QFedAvg and FedProx cannot be combined**; you must choose one for the NN aggregation path. All other capabilities — TLS, SecAgg+, LocalDP, RDP accounting, Stacking, and hybrid fusion — can coexist in the same Flower federation.