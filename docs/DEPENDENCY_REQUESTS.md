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
