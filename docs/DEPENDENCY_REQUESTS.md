# Dependency Requests

**Owner:** P1 owns `pyproject.toml`; all other packages request additions here.  
**Process:** P1 applies approved requests in the next P1 commit.  
**ID namespace:** `DEP-<nnn>`

| ID | Date | Requested by | Package | Dependency | Reason | Status |
|---|---|---|---|---|---|---|
| DEP-ORCH-001 | Orchestrator | None required | Discovered that the four root prompt files meant for O0.2/O1 (P1, P2, B2, P3) do not exist in this repository, contradicting the orchestrator prompt and the prior agent's `PROGRAM_INTAKE.md`/`DECISIONS.md` records (D-ORCH-001/002). Resolution: execute the three prompts from `prompts/` copies; do not reconstruct originals. | CLOSED — D-ORCH-003 (Session 1: sha256 FAIL for six files; originals not in repo). |
| DEP-ORCH-002 | Orchestrator | P3 / B2 | Industry data has no known-truth treatment/outcome (clinical tickets out of scope). Draft DQ-I01–I04 (treatment = logistic score of covariates; outcome with known conditional effect; per-country effect-modifier differences labelled "assumption"; truth columns separate from five ML targets) for batch 3b. | PENDING — assigned to P3/B2 Phase 0/3b. |

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
