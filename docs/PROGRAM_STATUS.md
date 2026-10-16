# Program status (updated 2026-10-16, commit TBD, data_version TBD)

| Wave | Package | Prompt/phases | Branch | Gate | Status | Evidence | Commit | Stale? |
| --- | --- | --- | --- | --- | --- | --- | --- | --- |
| 0 | Intake | O0 | — | O0 | **PASS** | `docs/PROGRAM_INTAKE.md` exists; M1–M20 confirmed | — | — |
| 0 | Reconcile | O1 | — | O1 | **PASS** | `prompts/` created; `PATCHES.md` written; 5 versioned copies | c738be7 | — |
| 0 | Plan | O2 | — | O2 | **PASS** | `docs/PROGRAM_PLAN.md` created; `integration/audit` branch exists | c738be7 | — |
| 1 | Platform base | P1 0–2 | chore/full-audit | G0,G1,G2 | TODO | — | — | — |
| 2a | Test, Make, Docker, CI | P1 3–6 | chore/full-audit | G3–G6 | TODO | — | — | — |
| 2b | Data review | P2 0–6 | chore/data-audit | G0–G6 | TODO | — | — | — |
| 3a | Causal core (simulators) | P3 0–3 | feat/causal-transport | G0–G3 | TODO | — | — | — |
| 3b | Data fixes | B2 A–E | chore/data-audit | G7 | TODO | — | — | — |
| 4 | Node differences + causal link | B2 F + P3 6 | chore/data-audit / feat/causal-transport | G4 | TODO | — | — | — |
| 5 | Experiments | P1 7 + P3 7 | chore/full-audit / feat/causal-transport | G5 | TODO | — | — | — |
| 6 | Final regression | P1 8 + P2 8 + P3 8–9 | integration/audit | G6 | TODO | — | — | — |

## Open defects (E-ORCH / E-P1 / DQ / CT)

| ID | Description | Opened | Owner | Status |
|---|---|---|---|---|
| E-ORCH-001 | Prior agent left uncommitted changes on `chore/full-audit`; tree not clean | 2026-10-16 | P1 | OPEN — must resolve before Wave 1 gate |

## Open dependency requests

| ID | Requested by | Package | Description | Status |
|---|---|---|---|---|
| — | — | — | — | — |

## Decisions (index)

| ID | Date | Question | Decision |
|---|---|---|---|
| D-ORCH-001 | 2026-10-16 | `PROMPT_1.md` missing | `PROMPT.md` is canonical P1 prompt |
| D-ORCH-002 | 2026-10-16 | `DATA_AUDIT_PROMPT.md` missing | `Data audit prompt.md` (root) treated as canonical P2 |

## Prompt defects (M-numbers) and status

| # | Status | Resolution |
|---|---|---|
| M1 | CONFIRMED | O0.2 + O1 rename |
| M2 | CONFIRMED | O-2 delegation table |
| M3 | CONFIRMED | O-3 deferral |
| M4 | CONFIRMED | O-4 baseline coverage |
| M5 | CONFIRMED | O-5 single owner |
| M6 | CONFIRMED | O-6 ClientProtocol |
| M7 | CONFIRMED | O-1 / O-9 integration branch |
| M8 | CONFIRMED | O-7 PROGRAM_STATUS.md |
| M9 | CONFIRMED | O0.3 scope OUT |
| M10 | CONFIRMED | O-10 shared files |
| M11 | CONFIRMED | O-11 privacy config |
| M12 | CONFIRMED | O-10 dependency requests |
| M13 | CONFIRMED | O-3 post-port review |
| M14 | CONFIRMED | Session rituals |
| M15 | CONFIRMED | D-ORCH-001 |
| M16 | CONFIRMED | D-ORCH-002 |
| M17 | CONFIRMED | E-ORCH-001 |
| M18 | CONFIRMED | No action needed |
| M19 | CONFIRMED | P1 Phase 2 fix entry points |
| M20 | CONFIRMED | Legacy tree exclusion documented |

## Next action (one line)

Prepare and print Wave 1 session brief for P1 Phase 0–2 (Platform base).
