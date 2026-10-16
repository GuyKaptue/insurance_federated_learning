# Program Plan — Orchestrator O2

**Date:** 2026-10-16  
**Version:** v1.0  
**Based on:** O0 intake findings  

---

## Environment baseline (verified)

| Item | Value | Status |
|---|---|---|
| Python (system) | 3.13.10 | Available |
| Python (venv) | 3.11.x | Available (`venv/`) |
| `pyproject.toml` range | `>=3.11,<3.14` | Supports both |
| Docker | 29.4.3 | Available |
| Docker Compose | v5.1.3 | Available |
| Free disk | 90 GB | OK (>> 5 GB threshold) |
| OS | macOS Darwin 25.5.0, arm64 | — |

## Scope adjustment from O0

- **Clinical layer (GKV):** OUT OF SCOPE for this repo. No clinical generator files exist.
- **Industry layer:** IN SCOPE — `src/insurance_fl/data/` generator, quality, preprocess.
- **Federated layer:** IN SCOPE — `src/insurance_fl/data/partition.py`, node structures, FL experiments.
- **Causal layer:** Will be built by B2 (DQ-014 to DQ-016, DQ-028, DQ-029) and consumed by P3.

## Wave table (adjusted)

| Wave | Packages | Starts when | Gate (evidence required) |
| --- | --- | --- | --- |
| **0** Intake and reconcile | O0–O2 (this prompt): prompt lint, repo map, patches, program plan, status file | now | `docs/PROGRAM_STATUS.md` and `prompts/PATCHES.md` exist; M1–M14 each CONFIRMED/REFUTED |
| **1** Platform base (P1) | P1 Phase 0, 1, 2 (behaviour-preserving port, packaging, `ClientProtocol`, config loader) | Gate 0 | P1 Gate 0, 1, 2 with output; `python -c "import insurance_fl"`; five console scripts `--help` |
| **2a** Test, Make, Docker, CI base (P1) | P1 Phase 3, 4, 5, 6 with O-4 (baseline coverage) and `make data-manifest` | Gate 1 | `make ci` green; test mirror enforced; Docker/compose per P1 gates (or limitation stated); `actionlint` clean |
| **2b** Data review (P2), parallel with 3a | P2 Phases 0 to 6 on the Wave 1 port (read-only). B2 items become CONFIRMED/REFUTED. | Gate 1 | `docs/DATA_REVIEW.md` complete with dependencies; refuted items deleted from B2 |
| **3a** Causal core on simulators (P3), parallel with 2b | P3 Phases 0 to 3 (and Phases 4 to 5 after `ClientProtocol` exists) using simulators only | Gate 1 (and Gate 2a for CI) | P3 gates; simulator Monte Carlo measured |
| **3b** Data fixes (B2) | P2 Phase 7, fix phases A, B, C, D, E in order, one DQ per commit, tests first | Gate 2b | each DQ closed with before/after numbers; 0 integrity violations; `data_version` updated |
| **4** Node differences and causal data link | B2 fix phase F (DQ-028 to DQ-032) then P3 Phase 6 (project scenarios) | Gate 3b and Gate 3a | difference matrix shows only intended differences; P3 Gate 6 |
| **5** Experiments | P1 Phase 7 (`make fl-all`) and P3 Phase 7 (`make causal-all`), both with the current `data_version` | Gate 4 | all runs complete; summaries generated from saved runs; no STALE reports |
| **6** Final regression | P2 Phase 8, P1 Phase 8 (fresh clone, three green runs, coverage raised to 85), P3 Phase 8 and 9, one README, final reports | Gate 5 | all three Definition-of-Done lists ticked with evidence; PR from `integration/audit` |

## Dependency order (hard constraints)

```
P1 Phase 2  ──► everything else that touches code
P2 review   ──► B2 fixes
B2 fixes (C, F) ──► P3 Phase 6
B2 all      ──► P1 Phase 7
data_version fixed ──► any experiment
```

## Risk register

| Risk | Likelihood | Impact | Mitigation |
|---|---|---|---|
| Prior P1 work on `chore/full-audit` is uncommitted / mixed with legacy code | High | Merge conflicts | Commit or stash cleanly before Wave 1 gate; run `compileall` and `test_structure.py` |
| Clinical tickets in B2 create noise (DQ-007 to DQ-016, 020–022, 024) | Certain | Low | Marked OUT OF SCOPE in O0; P2 skips them |
| `hypothesis` not installed in current env causes test collection failure | High | Low | Install dev deps in the working venv before Wave 2a gate review |
| Console scripts point to `insurance_fl.scripts.*` but modules don't exist yet | Certain | Medium | P1 Phase 2 must create `src/insurance_fl/scripts/` or fix entry points |

## Branch strategy (O-1)

1. `integration/audit` — created from `main`, never force-pushed
2. `chore/full-audit` — P1 work (already exists, has prior work)
3. `chore/data-audit` — P2/B2 work (create from `integration/audit` tip at Wave 2b)
4. `feat/causal-transport` — P3 work (create from `integration/audit` tip at Wave 3a)

Merge order into `integration/audit`:
- Wave 1: `chore/full-audit`
- Wave 2a: `chore/full-audit` (continued)
- Wave 2b: `chore/data-audit`
- Wave 3a: `feat/causal-transport`
- Wave 3b: `chore/data-audit` (continued)
- Wave 4: `feat/causal-transport` (continued)
- Wave 5: `chore/full-audit` (continued) + `feat/causal-transport` (continued)
- Wave 6: all final regression work

## Notes

- **P1 Phase 7 and 8 are deferred** to Waves 5 and 6 per O-3.
- **Coverage threshold** is baseline now, 85 at Wave 6 per O-4.
- **Data lineage** (`make data-manifest`) is implemented in Wave 2a per O-8.
- **Privacy claims** are unified: formal epsilon/delta only with verified accountant per O-11.
