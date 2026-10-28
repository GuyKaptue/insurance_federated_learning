# Dependency Requests

**Owner:** P1 owns `pyproject.toml`; all other packages request additions here.  
**Process:** P1 applies approved requests in the next P1 commit.  
**ID namespace:** `DEP-<nnn>`

| ID | Date | Requested by | Package | Dependency | Reason | Status |
|---|---|---|---|---|---|---|
| DEP-ORCH-001 | Orchestrator | None required | Discovered that the four root prompt files meant for O0.2/O1 (P1, P2, B2, P3) do not exist in this repository, contradicting the orchestrator prompt and the prior agent's `PROGRAM_INTAKE.md`/`DECISIONS.md` records (D-ORCH-001/002). Resolution: execute the three prompts from `prompts/` copies; do not reconstruct originals. | CLOSED — D-ORCH-003 (Session 1: sha256 FAIL for six files; originals not in repo). |
| DEP-ORCH-002 | Orchestrator | P3 / B2 | Industry data has no known-truth treatment/outcome. Ground-truth ATE spec (P3 Phase 6): target = claim_risk_score (regression); treatment = has_claims (number_of_claims > 0); countries = Germany, France, Belgium, Switzerland. Truth is semi-synthetic: inject a known additive effect of has_claims on claim_risk_score in the generator, kept as a separate truth column (not one of the five ML targets); per-country effect-modifier differences labelled assumption. Adjustment set = age, annual_income, savings_balance, credit_score, debt_to_income_ratio per docs/causal/dag.yaml. | SPEC COMPLETE — awaiting B2 Phase 0 implementation. |

## DEP-ORCH-002 amendments (L6 precondition)

### Injection location
**Injection location:** `src/insurance_fl/causal/truth.py` (new module, B2-owned). It reads the generator's output DataFrame and adds two columns: `claim_risk_score_true` (causal outcome) and `effect_has_claims` (per-row injected additive effect). It does NOT modify `src/insurance_fl/data/generator.py` - B2 statistics ownership preserved. The generator remains the source of the five ML targets; the truth columns are derived downstream.

### DAG re-validation
**DAG re-validation:** The adjustment set in `docs/causal/dag.yaml` (`{age, annual_income, savings_balance, credit_score, debt_to_income_ratio}`) is validated against `claim_risk_score`, not `claim_risk_score_true`. After injection, `tests/unit/causal/test_dag_post_injection.py` asserts, on post-injection data: (a) `has_claims -> claim_risk_score_true` is d-connected unconditionally; (b) it is blocked when conditioning on the adjustment set; (c) no other backdoor path from `has_claims` to `claim_risk_score_true` is open under the adjustment set. If the DAG requires amendment to satisfy (c), the amendment is logged as D-CT-001 in `docs/DECISIONS.md`.

### Leakage test
**Leakage test:** `tests/unit/causal/test_truth_leakage.py` asserts: (a) `claim_risk_score_true` is NOT returned by the data loader for ML training; (b) `effect_has_claims` is NOT among ML features; (c) a model trained on the five ML targets does not predict `claim_risk_score_true` beyond chance. This is DQ-I04's enforcement.

D-CT-001 not yet logged - pending post-injection measurement in B2 Phase 0.

## Approved / Applied

| ID | Date | Applied in commit | Notes |
|---|---|---|---|
| — | — | — | — |

## Open dependency requests (this session)

| ID | Date | Requested by | Package | Dependency | Reason | Status |
|---|---|---|---|---|---|---|
| DEP-ORCH-003 | 2026-10-23 | Orchestrator (Cline) | `xgboost` | `xgboost>=1.7.0,<2.0.0` (pyproject + requirements.txt) vs installed `3.3.0` | xgboost 3.x `fit()` rejects `verbose=False` kwarg; `train_xgb` in `src/insurance_fl/fl/task.py` had `verbose=False` removed to restore compatibility. P1 to decide: pin `<2.0.0` or widen range + audit code for other 3.x incompatibilities. | OPEN |
| DEP-ORCH-004 | 2026-10-23 | Orchestrator (Cline) | `numpy` | `numpy>=1.24.0,<2.0.0` (pyproject + requirements.txt) vs installed `2.4.2` | NumPy 2.x installed despite `<2.0.0` constraint. P1 to reconcile after xgboost decision. | OPEN |
| DEP-ORCH-005 | 2026-10-23 | Orchestrator (Cline) | `scikit-learn` | `scikit-learn>=1.2.0,<1.6.0` (pyproject.toml) vs installed `1.9.0` | scikit-learn 1.9.0 installed despite `<1.6.0` constraint. P1 to reconcile. | OPEN |
| DEP-ORCH-006 | 2026-10-23 | Orchestrator (Cline) | `torch` + `xgboost` | — | Intermittent `OMP: Error #179` / segfaults (pthread_mutex_init) when torch's bundled `libomp` and xgboost's `libxgboost.dylib`→`libomp.dylib` coexist in one process. Workaround: `KMP_DUPLICATE_LIB_OK=True OMP_NUM_THREADS=1`. Documented as known env limitation; verify CI environment is unaffected. | OPEN |
| DEP-ORCH-007 | 2026-10-23 | Orchestrator (Cline) | `CA_TR_PROMPT.md` | Required by Wave 3a (P3) | `CA_TR_PROMPT.md` present in root (verified). No deferral needed at this time. | OPEN |
