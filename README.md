# 🏦 Insurance Federated Learning Platform

[![Python Version](https://img.shields.io/badge/python-3.11+-blue.svg)](https://www.python.org/)
[![License: MIT](https://img.shields.io/badge/License-MIT-yellow.svg)](https://opensource.org/licenses/MIT)
[![TensorFlow Federated](https://img.shields.io/badge/TFF-0.52+-orange.svg)](https://www.tensorflow.org/federated)
[![Code style: black](https://img.shields.io/badge/code%20style-black-000000.svg)](https://github.com/psf/black)
[![CI](https://github.com/GuyKaptue/insurance-federated-learning/actions/workflows/ci.yml/badge.svg)](https://github.com/GuyKaptue/insurance-federated-learning/actions/workflows/ci.yml)
[![codecov](https://codecov.io/gh/GuyKaptue/insurance-federated-learning/branch/main/graph/badge.svg)](https://codecov.io/gh/GuyKaptue/insurance-federated-learning)
[![GitHub release](https://img.shields.io/github/release/GuyKaptue/insurance-federated-learning.svg)](https://GitHub.com/GuyKaptue/insurance-federated-learning/releases/)
[![GitHub last commit](https://img.shields.io/github/last-commit/GuyKaptue/insurance-federated-learning.svg)](https://github.com/GuyKaptue/insurance-federated-learning/commits/main)
[![GitHub stars](https://img.shields.io/github/stars/GuyKaptue/insurance-federated-learning.svg)](https://github.com/GuyKaptue/insurance-federated-learning/stargazers)
[![GitHub forks](https://img.shields.io/github/forks/GuyKaptue/insurance-federated-learning.svg)](https://github.com/GuyKaptue/insurance-federated-learning/network)
[![GitHub issues](https://img.shields.io/github/issues/GuyKaptue/insurance-federated-learning.svg)](https://github.com/GuyKaptue/insurance-federated-learning/issues)

[![PyPI version](https://badge.fury.io/py/insurance-federated-learning.svg)](https://badge.fury.io/py/insurance-federated-learning)
[![Downloads](https://pepy.tech/badge/insurance-federated-learning)](https://pepy.tech/project/insurance-federated-learning)
[![Documentation](https://img.shields.io/badge/docs-passing-brightgreen.svg)](https://insurance-federated-learning.readthedocs.io)
[![DOI](https://zenodo.org/badge/DOI/10.5281/zenodo.XXXXXX.svg)](https://doi.org/10.5281/zenodo.XXXXXX)

## 📌 Overview

Master entry point. Cross-border federated learning platform for insurance analytics across 4 European countries:
- 🇩🇪 **Germany** (Allianz)
- 🇫🇷 **France** (AXA)
- 🇧🇪 **Belgium** (Ageas)
- 🇨🇭 **Switzerland** (Helvetia)

All experiments use **synthetic data** (5,000 samples per client) generated with real-world statistics; no real customer data is used. Results below are verified on the current working tree (see `docs/AUDIT_LOG.md`).

## Pipeline

| Stage | What | Why | How to run | Results link |
|-------|------|-----|------------|--------------|
| 1. Synthetic Data Generation | 20,000 synthetic records (5,000/client, 33 features, 5 targets) | Realistic, privacy-safe benchmark | `make data` (official: `insurance-fl-generate --samples 5000 --seed 42 --output data/processed/european_insurance_federated`, backed by `insurance_fl.data.generator.EuropeanInsuranceFederatedDataset`) | `docs/EIFD.md`, `docs/evidence/dataset_manifest.json` (when present) |
| 2. TLS Certificate Generation | CA + server + client certs via OpenSSL | Transport security for local SuperLink/SuperNode | `python generate_creds.py --output-dir certificates` | `tests/integration/test_tls.py` |
| 3. Security (SecAgg+) | Cryptographic masking of parameter updates (partially implemented) | Parameter updates protected by protocol when correctly wired; **not end-to-end Flower-app runtime evidence** | `pytest tests/integration/test_secagg.py` (protocol-level); `server_app.main(secagg-enabled=true)` fails closed (`NotImplementedError`) | `tests/integration/test_secagg.py`, `tests/unit/fl/test_server_app.py` (see D-PRIV-006; no committed end-to-end SecAgg run log) |
| 4. Fairness | Country-stratified sampling + disparity D = max(m_c) - min(m_c) | Detect disparate impact | `pytest tests/integration/test_fairness.py` | `artifacts/fairness_disparity.json` (v2 `is_high_risk`: point 0.0347, joint 95% CI [0.0159, 0.0803], **inconclusive**) |
| 5. MIA | Black-box shadow-model MIA on is_high_risk | Empirical leakage check | `python scripts/run_mia.py` | `reports/mia_evaluation.json` (not committed) - AUC 0.6032215647600263, TPR@1%FPR 0.03564102564102564 |
| 6. Differential Privacy (Local DP) | Optional Gaussian noise + clipping (disabled by default) | Per-client DP when enabled | `python scripts/dp_sweep.py` | `reports/dp_utility_curve.json` (not committed); script at `scripts/dp_sweep.py` |
| 7. DP-Counting (RDP) | Fail-closed RDP accounting with dp-accounting v0.6.0 | Quantifiable epsilon-delta budget | `pytest tests/integration/test_rdp_accounting.py` | No committed epsilon (per-run only); fail-closed test in `tests/integration/test_rdp_accounting.py` |
| 8. Federated Learning with Flower | Multi-task FL app (Kendall + FedBN) with Flower simulation path | Cross-market model without sharing raw data | `python -m insurance_fl.fl.task --prepare && flwr run . --stream --run-config num-server-rounds=2` | **Pre-fix (invalid for 4-partition identity claims, D-ORCH-010):** `docs/evidence/layer1_flwr_run_2rounds.log`, `docs/evidence/layer2_flwr_run_3rounds.log`. **Post-fix participation proof:** `make flwr-guard` (4 distinct partitions/clients in fit+eval, see `scripts/check_flwr_guard_identity.py`). |

## 1. The Problem We Are Solving

European insurers hold highly sensitive customer data. Regulations (GDPR) and competition make pooling raw data impractical.

| # | Challenge | Why it matters |
|---|-----------|----------------|
| **P1** | **Data cannot leave the client** | GDPR + competitive sensitivity. |
| **P2** | **Models must stay private too** | Gradients can leak training data (membership inference). |
| **P3** | **Markets differ (distribution shift)** | A model good on DE/FR may fail on CH/BE. |
| **P4** | **Fairness & auditability** | Regulators require non-discriminatory, explainable, auditable models. |
| **P5** | **Correlation ≠ causation; markets differ** | A model that predicts well in DE/FR may encode confounded associations that fail (or reverse) in CH/BE. Regulators and actuaries need *causal* effects and evidence that they **transport** across countries. |

## 2. How We Solve It

| Layer | Technique | What it solves |
|-------|-----------|----------------|
| **L1** | Flower federated orchestration | P1 — data never leaves the client. |
| **L2** | Multi-task NN + Kendall uncertainty weighting + FedBN | P1/P3 — shared encoder, per-client BatchNorm. |
| **L3** | Secure Aggregation (SecAgg+) + Local Differential Privacy | P2 — gradients never visible in clear. |
| **L4** | RDP privacy accountant | P2 — quantifiable ε-δ budget. |
| **L5** | Fairness metrics + hybrid aggregation | P4 — detect and correct disparate impact. |
| **L6** | **Causal estimation** | P5 — DAG-driven ATE via **backdoor adjustment** and **doubly-robust (AIPW) with cross-fitting**; agreement between estimators is a validity gate. |
| **L7** | **Transportability** | P5 — selection-odds / density-ratio weighting, **SMD** diagnostics, DR transport, overlap checks; emits a **T3 transportability verdict** per task and separates *model* transportability from *causal-effect* transportability. |

## ✅ Results (What the Platform Can Do Today)

- **Federated Flower app executes**: pre-fix `flwr run` logs show completed multi-round simulation runs (`docs/evidence/layer1_flwr_run_2rounds.log`, `docs/evidence/layer2_flwr_run_3rounds.log`), but those runs are **invalid for 4-partition identity/per-client attribution claims** due to a pre-fix `node_id % 4` mapping bug (D-ORCH-010). Post-fix `make flwr-guard` proves 4 distinct partitions/clients participate once each in fit and evaluate (identity-checked).
- **Privacy stack (fail-closed)**: RDP accountant, Local DP, and SecAgg+ modules present; runs halt rather than proceed without an ε guarantee when the backend is unavailable.
- **Fairness & audit**: country-stratified FedProx, disparity metrics, JSONL audit trail (6 event types, secret redaction), MIA evaluation.
- **Hybrid ML**: per-client XGBoost/LightGBM/CatBoost with OOF predictions, NNLS stacking, β-fusion.
- **`is_high_risk` (v2, validated gate artifacts):** federated/pooled/local/GBM test AUCs do not exceed per-test-split Bayes AUC by >2 SE for any client (`reports/is_high_risk_auc.json:test_split_bayes`). Post-hoc calibration (isotonic/temperature/Platt, val-fit/test-eval) showed no uniform improvement; **raw probabilities remain the shipped default** (`reports/is_high_risk_auc.json:calibration_comparison`). Feature surface: 24 identical ordered inputs/client (21 base + 3 missingness indicators).
### ✅ Causal estimation — estimators agree
```
scripts/causal_estimate.py
→ Backdoor ATE: 7.65   (95% CI [7.40, 7.90])
→ DR/AIPW ATE: 7.60   (95% CI [7.27, 7.93])
→ Relative diff: 0.007  → estimators agree ✅
```
Artifacts: `src/insurance_fl/causal/` (`dag_check`, `estimands`, `refute`,
`preflight`, `library_contract`), `docs/causal/dag.yaml`,
notebook `causal_inference_quickstart.ipynb`.

### ✅ Transportability — DE/FR → CH, `claim_risk_score`
```
scripts/transport.py --source-clients de,fr --target-client ch --task claim_risk_score
→ worst SMDs: savings_balance 1.55, renewal_probability 0.84
→ source loss 18.43 · target loss 20.12 · ratio 1.09
→ T3 transportable (model):            True  ✅
→ causal-effect transportable:          False  ⚠️ (country shifts treatment/outcome)
```
Artifacts: `src/insurance_fl/transport/transport.py`, `scripts/transport.py`,
`scripts/transport_all.py`, notebook `transportability_quickstart.ipynb`,
outputs in `runs/causal/transport_*.json`.

> **Reading the verdict correctly:** the *model* transfers to CH with only a
> ~9% loss penalty (ratio 1.09), but the *causal effect* estimated on DE/FR does
> **not** transport — the platform reports this explicitly rather than silently
> reusing the pooled effect. This separation is the whole point of the
> transportability layer.
- **Tests**: 263 unit + 1 slow + 2 integration tests passing.

## Development environment

### Setup (generated data not tracked)

Canonical federated datasets under `data/processed/` are generated artifacts and are **not** tracked in git. Reproduce them locally or in CI before data-dependent tests/notebooks:

```bash
./venv/bin/python -m pip install -e ".[dev,causal]"
make data   # generates via official CLI (see Makefile/data target)
# or: insurance-fl-generate --samples 5000 --seed 42 --output data/processed/european_insurance_federated
```

If the canonical dataset is missing, data-dependent tests skip with a clear message.

> **Canonical generator:** use `insurance-fl-generate` / `insurance_fl.data.generator.EuropeanInsuranceFederatedDataset` only. `src/synthetic/data_generator.py` is legacy/deprecated (non-canonical behavior, `claim_risk` coupling) and must not be used for canonical datasets.


- **Canonical interpreter:** use the project virtual environment at `./venv` (Python 3.11 in this checkout). Run commands as `./venv/bin/python -m ...` and `./venv/bin/pip ...` (or activate the venv). `Makefile` is venv-aware (`PYTHON ?= venv/bin/python` when present).
- **Do not rely on system/Homebrew Python for project validation:** `python3` on this machine resolves to Homebrew Python 3.13 with a different package set (e.g. Flower version/import paths differ) and can produce misleading test/notebook results.
- **Notebook execution kernel:** `scripts/validate_notebooks.py` executes notebooks with the `insurance-fl` Jupyter kernel by default (`NB_KERNEL` overrides). Register it with `make kernel` (or `./venv/bin/python -m ipykernel install --user --name insurance-fl`) before validation; `scripts/run_notebooks.py` currently uses `kernel_name="python3"` and may need the same alignment for non-validator runs.
- **Utility scripts (kept):** `scripts/launch_tls.py` — print/dry-run TLS Flower launch commands (e.g. `./venv/bin/python scripts/launch_tls.py --dry-run --certs-dir certificates`). `scripts/run_e2e.py` — end-to-end hybrid pipeline runner (e.g. `./venv/bin/python scripts/run_e2e.py --rounds 3 --output-dir runs`). `scripts/diagnose_is_high_risk.py` — ad-hoc XGBoost learnability diagnostic for `is_high_risk` (e.g. `./venv/bin/python scripts/diagnose_is_high_risk.py --data-dir data/processed/fl_task0_smoke`).


## ✨ Features

- 🏦 **Realistic Insurance Data**: Synthetic data with real-world statistics
- 🤖 **5 ML Tasks**: 2 Regression, 2 Binary Classification, 1 Multi-class
- 🔒 **Privacy-Preserving**: Federated learning with differential privacy
- 🚀 **Production Ready**: Docker, Kubernetes, FastAPI deployment
- 📊 **Comprehensive Monitoring**: Prometheus, Grafana dashboards
- 🧠 **Causal Inference & Transportability**: DAG-based ATE estimation (backdoor + doubly robust) and cross-country transport analysis

### 4.5 Causal inference & transportability

**Causal inference** — estimate the effect of a treatment (e.g. `has_claims`)
on an outcome (e.g. `claim_risk_score`) from the DAG in `docs/causal/dag.yaml`,
using backdoor adjustment and doubly-robust (AIPW) with cross-fitting:

```bash
# 1) Validate the DAG (nodes, edges, adjustment sets)
python -m insurance_fl.causal.dag_check
#    → 42 nodes, 46 edges, 1 adjustment set ✅

# 2) Estimate ATE (backdoor + DR/AIPW)
python scripts/causal_estimate.py
#    → Backdoor ATE 7.65 (CI [7.40, 7.90]) · DR ATE 7.60 (CI [7.27, 7.93])
#    → relative diff 0.007 → estimators agree

# 3) Refute / sensitivity checks
python scripts/causal_refute.py
```

**Transportability** — test whether the model *and* the causal effect learned
on source countries can be applied to a target country:

```bash
# Single pair
python scripts/transport.py \
  --source-clients de,fr --target-client ch --task claim_risk_score

# All source→target pairs for every task
python scripts/transport_all.py
```

Reports are written to `runs/causal/transport_*.json` and include:
SMD diagnostics, overlap checks, source/target loss and **ratio**, and the
**T3 transportability verdict** — reported separately for the *model* and for
the *causal effect*.

Notebooks for interactive exploration:
`notebooks/causal_inference_quickstart.ipynb`,
`notebooks/transportability_quickstart.ipynb`.

## 📦 Installation (clean environment)

### From Source (recommended)
```bash
git clone https://github.com/GuyKaptue/insurance-federated-learning.git
cd insurance-federated-learning
python3.11 -m venv .venv
source .venv/bin/activate   # Windows: .venv\\Scripts\\activate
pip install --upgrade pip
pip install -e ".[dev,causal]"
```

### Verify the install
```bash
make ci          # lint + typecheck + tests + coverage + 1-round flwr smoke
```

### Run a 1-round federated smoke test
```bash
KMP_DUPLICATE_LIB_OK=True OMP_NUM_THREADS=1 make flwr-guard
```

### Notes
- `dp-accounting` is a declared runtime dependency; do not rely on system Python.
- `mypy==1.10.0` is pinned in the `dev` extra (flwr requires `pathspec<0.13`).
- Verified tip (last `make ci` green): `fb632494d35e89e532208b3dabb2af03dce6dbc4`.
- If you hit an OpenMP error (`OMP: Error #179`), keep `KMP_DUPLICATE_LIB_OK=True OMP_NUM_THREADS=1`.

## 📦 Installation (legacy quick reference)

### From PyPI
```bash
pip install insurance-federated-learning
```

### From Source
```bash
git clone https://github.com/GuyKaptue/insurance-federated-learning.git
cd insurance-federated-learning
pip install -e ".[dev]"
```

## 🚀 Quick Start

### Generate Dataset
```bash
insurance-fl-generate --samples 5000 --output data/processed
```

### Train Federated Model
```bash
insurance-fl-train --config config/federated_config.yaml
```

### Start API Server
```bash
insurance-fl-serve --host 0.0.0.0 --port 8000
```



## Status and evidence

| Capability | Status | Evidence | Decision |
|------------|--------|----------|----------|
| SecAgg+ | Partially implemented (protocol-level) | `tests/integration/test_secagg.py`, `tests/unit/fl/test_server_app.py`, `scripts/check_flwr_guard_identity.py` (not SecAgg runtime) | D-PRIV-003, D-HYBRID-004, D-PRIV-006 |
| LocalDP | Preview (disabled by default) | `src/insurance_fl/fl/dp.py`, `tests/unit/fl/test_dp.py` | D-PRIV-003 |
| RDP accounting | Preview (fail-closed) | `src/insurance_fl/fl/privacy.py`, `tests/integration/test_rdp_accounting.py` | D-PRIV-001, D-PRIV-005 |
| Fairness | Preview | `src/insurance_fl/fl/strategy.py`, `tests/unit/fl/test_fairness.py` | D-FAIR-001 |
| MIA | Preview (empirical) | `src/insurance_fl/evaluation/mia.py`, `tests/unit/evaluation/test_mia.py` | D-MIA-001 |
| TLS (local only) | Preview | `generate_creds.py`, `tests/integration/test_tls.py` | D-HYBRID-004 |
| Causal | Preview | `src/insurance_fl/causal/`, `docs/causal/dag.yaml` | DEP-ORCH-002 |
| Transport | Preview | `src/insurance_fl/transport/transport.py`, `tests/unit/transport/test_transport.py` | DEP-ORCH-002 |
| Hybrid boosters | Preview | `src/insurance_fl/fl/boosters.py`, `src/insurance_fl/fl/stacking.py` | D-HYBRID-002, D-HYBRID-003 |

No capability is marked production-ready.

## Privacy, Fairness, MIA & Audit (Preview)

This release adds four security and governance capabilities (all in
preview/experimental status unless noted):

### Secure Aggregation (SecAgg+)
- Flower 1.31 `SecAggPlusWorkflow` integration and client modifier `secaggplus_mod` exist; protocol-level coverage in `tests/integration/test_secagg.py`.
- **End-to-end Flower simulation orchestration via `server_app.main` is not verified/wired**: `secagg-enabled=true` fails closed with `NotImplementedError` (no hang, no plaintext fallback), covered by `tests/unit/fl/test_server_app.py`.
- Design params (target): num_shares=4, reconstruction_threshold=3, max_weight=1000.0, clipping_range=8.0, quantization_range=2^22.
- **Server visibility (D-PRIV-006):** SecAgg+ protects **parameter updates**; per-client metrics (e.g. `client_id`, train/eval metrics) and `num_examples` can remain visible to the server/strategy in the legacy-compat workflow path.
- **Secure Aggregation alone is NOT a differential privacy guarantee.**

### Local Differential Privacy (Local DP)
- `build_local_dp_mod()` in `src/insurance_fl/fl/dp.py` wraps Flower's `LocalDpMod`.
- Clipping + Gaussian noise applied before SecAgg masking.
- **Disabled by default.** No Local-DP guarantee is provided when disabled.
- Utility sweep: `python scripts/dp_sweep.py` → `reports/dp_utility_curve.json`.

### RDP Accounting
- **RESOLVED**: Uses `dp-accounting` (Google) v0.6.0 as RDP backend.
- `RdpAccountantWrapper.get_epsilon()` returns genuine RDP epsilon values.
- `would_exceed()` projects epsilon without mutating committed state.
- State persistence via JSON with version validation.
- Fail-closed when backend unavailable.

### Fairness (FedProx + Stratified Sampling)
- `StratifiedFedProx` in `strategy.py` guarantees at least one client per
  eligible country (subject to availability and budget).
- Per-country disparity: D = max(m_c) - min(m_c).
- 5% threshold is a configurable hypothesis, not a regulatory standard.
- **v2 `is_high_risk` (fed reference, per-client test AUC):** point disparity 0.0347 with per-seed disparities 0.0338/0.0340/0.0375, but joint bootstrap 95% CI [0.0159, 0.0803] crosses 0.05 (conservative max−min bootstrap) — **status: inconclusive** (not a pass). See `artifacts/fairness_disparity.json`.

### TLS
- Applies to local SuperLink/SuperNode deployments only. SuperGrid transport security is platform-managed.
- Generate certs: `python generate_creds.py --output-dir certificates` (see `tests/integration/test_tls.py`).

### Reproducibility
- Clean-venv install: `python3.11 -m venv .venv && source .venv/bin/activate && pip install -e ".[dev]"`.
- `dp-accounting` is a declared runtime dependency; do not rely on system Python.
- `mypy==1.10.0` is pinned in the `dev` extra (flwr requires `pathspec<0.13`).

### Membership Inference Attack (MIA)
- Black-box MIA via shadow models: `src/insurance_fl/evaluation/mia.py`.
- Reports ROC AUC and TPR@1%FPR. Empirical result, not formal DP guarantee.
- Run: `python scripts/run_mia.py` → `reports/mia_evaluation.json`.
- AUC near 0.5 does not prove zero leakage.

### Audit Trails
- Structured JSONL: `src/insurance_fl/audit.py`. 6 event types.
- Secret redaction (case-insensitive: secret, token, password, key).
- Retention: 180 days (intended policy, not legal compliance).
- No automatic deletion; enforcement is deployment/governance responsibility.

**Causal & transport artifacts at a glance**
- DAG: `docs/causal/dag.yaml`
- Causal code: `src/insurance_fl/causal/` — `dag_check`, `estimands`, `refute`, `preflight`, `library_contract`
- Transport code: `src/insurance_fl/transport/transport.py`
- Scripts: `scripts/causal_estimate.py`, `scripts/causal_refute.py`, `scripts/transport.py`, `scripts/transport_all.py`
- Notebooks: `causal_inference_quickstart.ipynb`, `transportability_quickstart.ipynb`
- Outputs: `runs/causal/transport_*.json`
- Tests: `tests/unit/causal/`, `tests/unit/transport/`, `tests/integration/`
- Decision records: `D-CT-*` in `docs/DECISIONS.md`; evidence in `docs/AUDIT_LOG.md`; open request `CA_TR` in `docs/DEPENDENCY_REQUESTS.md`

## What is NOT claimed

- No formal differential-privacy guarantee from MIA (empirical only).
- No SuperGrid TLS claim (TLS is local SuperLink/SuperNode only).
- No production SLA or production-ready status.
- The 5% fairness disparity threshold is a configurable hypothesis, not a regulatory standard.
- No committed epsilon value (epsilon is computed per run).
- All data is synthetic; no real insurance records are used.

## 📚 Documentation

- [Installation Guide](docs/installation.md)
- [User Guide](docs/usage.md)
- [API Reference](docs/api.md)
- [Contributing Guide](CONTRIBUTING.md)

## 🧪 Testing

```bash
# Run all tests
pytest tests/ -v

# Run with coverage
pytest tests/ --cov=src --cov-report=html

# Run specific test category
pytest tests/unit/ -v
```

## 🤝 Contributing

We welcome contributions! Please see our [Contributing Guidelines](CONTRIBUTING.md).

1. Fork the repository
2. Create your feature branch: `git checkout -b feature/AmazingFeature`
3. Commit your changes: `git commit -m 'Add AmazingFeature'`
4. Push to the branch: `git push origin feature/AmazingFeature`
5. Open a Pull Request

## 📄 License

This project is licensed under the MIT License - see the [LICENSE](LICENSE) file for details.

## 👤 Author

**Guy M. Kaptue T**

- GitHub: [@GuyKaptue](https://github.com/GuyKaptue)
- Email: guykaptue24@gmail.com
- LinkedIn: [Guy Kaptue](https://linkedin.com/in/guy-kaptue)

## 🙏 Acknowledgments

- OECD Statistics
- Eurostat Database
- European Insurance Consortium
- TensorFlow Federated Team
- Open Source Community

## 🔗 Links

- [GitHub Repository](https://github.com/GuyKaptue/insurance-federated-learning)
- [Issue Tracker](https://github.com/GuyKaptue/insurance-federated-learning/issues)
- [Documentation](https://insurance-federated-learning.readthedocs.io)
- [PyPI Package](https://pypi.org/project/insurance-federated-learning/)

## 📊 Citation

If you use this project in your research, please cite:

```bibtex
@software{kaptue_2024_insurance_fl,
  author = {Kaptue T, Guy M.},
  title = {Insurance Federated Learning: A Platform for European Insurance Analytics},
  year = {2024},
  publisher = {GitHub},
  url = {https://github.com/GuyKaptue/insurance-federated-learning}
}
```

## ⭐ Star History

[![Star History Chart](https://api.star-history.com/svg?repos=GuyKaptue/insurance-federated-learning&type=Date)](https://star-history.com/#GuyKaptue/insurance-federated-learning&Date)
```

## Updated setup.py

```python
"""
Setup script for Insurance Federated Learning Project
"""

from setuptools import setup, find_packages
import os

# Read requirements
with open('requirements.txt', 'r') as f:
    requirements = [line.strip() for line in f if line.strip() and not line.startswith('#')]

# Read README
with open('README.md', 'r', encoding='utf-8') as f:
    long_description = f.read()

setup(
    name='insurance-federated-learning',
    version='1.0.0',
    author='Guy M. Kaptue T',
    author_email='guykaptue24@gmail.com',
    description='Federated Learning Platform for European Insurance Analytics',
    long_description=long_description,
    long_description_content_type='text/markdown',
    url='https://github.com/GuyKaptue/insurance-federated-learning',
    project_urls={
        'Bug Tracker': 'https://github.com/GuyKaptue/insurance-federated-learning/issues',
        'Documentation': 'https://insurance-federated-learning.readthedocs.io',
        'Source Code': 'https://github.com/GuyKaptue/insurance-federated-learning',
    },
    packages=find_packages(where='src'),
    package_dir={'': 'src'},
    classifiers=[
        'Development Status :: 4 - Beta',
        'Intended Audience :: Financial and Insurance Industry',
        'Topic :: Scientific/Engineering :: Artificial Intelligence',
        'License :: OSI Approved :: MIT License',
        'Programming Language :: Python :: 3.11',
        'Programming Language :: Python :: 3.12',
        'Operating System :: OS Independent',
    ],
    python_requires='>=3.11,<3.13',
    install_requires=requirements,
    extras_require={
        'dev': [
            'pytest>=7.2.0',
            'pytest-cov>=4.0.0',
            'black>=23.0.0',
            'flake8>=6.0.0',
            'mypy>=1.0.0',
            'isort>=5.12.0',
            'pre-commit>=3.0.0',
            'jupyter>=1.0.0',
        ],
        'docs': [
            'sphinx>=5.3.0',
            'sphinx-rtd-theme>=1.2.0',
            'myst-parser>=1.0.0',
            'mkdocs>=1.4.0',
            'mkdocs-material>=9.0.0',
        ],
        'cloud': [
            'boto3>=1.26.0',
            'google-cloud-storage>=2.8.0',
            'azure-storage-blob>=12.14.0',
        ],
    },
    entry_points={
        'console_scripts': [
            'insurance-fl-generate=src.scripts.generate_dataset:main',
            'insurance-fl-train=src.scripts.run_federated_training:main',
            'insurance-fl-evaluate=src.scripts.evaluate_models:main',
            'insurance-fl-serve=src.scripts.deploy_api:main',
        ],
    },
    include_package_data=True,
    zip_safe=False,
)
```

## GitHub Repository Setup Commands

```bash
# Initialize Git repository
cd insurance_federated_learning
git init

# Add remote repository
git remote add origin https://github.com/GuyKaptue/insurance-federated-learning.git

# Create and switch to main branch
git branch -M main

# Add all files
git add .
git commit -m "Initial commit: European Insurance Federated Learning Platform"

# Push to GitHub
git push -u origin main

# Create and push tags
git tag -a v1.0.0 -m "Release version 1.0.0"
git push origin v1.0.0

# Create development branch
git checkout -b develop
git push -u origin develop
