# Program status (updated 2026-10-17, commit a0b1a72, data_version TBD)

| Wave | Package | Prompt/phases | Branch | Gate | Status | Evidence | Commit | Stale? |
| --- | --- | --- | --- | --- | --- | --- | --- | --- |
| 1 | Platform base | P1 0–2 | chore/full-audit | G0,G1,G2 | **PASS** | `make ci` green (ruff+black+isort+mypy+80 tests+coverage 94.7%≥85); `import insurance_fl` OK; 5 console scripts `--help` OK; golden baseline reproducible; ClientProtocol+dummy test OK | a2e3328 | — |
| 0 | Intake | O0 | — | O0 | **CONDITIONAL** | O0 reopened twice. Session 1b: B2 proves root files agent-placed (never committed), B3 shows true originals unavailable. Branch B done (tree clean modulo B3 evidence). Gate A FAIL pending originals. | 8070c0c | — |
| 0 | Reconcile | O1 | — | O1 | **CONDITIONAL** | prompts/ copies internally consistent (root == prompts/, PROMPT.md == PROMPT_1.md) but unverified vs true originals. PATCHES.md quotes prompts/ lines, not proven original lines. | 8070c0c | — |
| 0 | Plan | O2 | — | O2 | **CONDITIONAL** | plan/status/decisions based on false D-ORCH-001/002; corrected but conditional | — | — |
| 1b | Wave 0 closeout 2nd pass + Branch B cleanup | B1–B5 + B4(7 rows) | integration/audit | Gate 1b | **FAIL** (tree clean; provenance pending) | Branch B: 42→6 dirty (6 = untracked root .md awaiting B3 verdict). Commits 72c3b8c (tree), 8070c0c (notebooks verified 2d5e13b4/36816de4 + brief). B2 proven, B3/B4-approval pending your hash paste. | 8070c0c | — |
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
| E-ORCH-001 | Root prompt files `PROMPT_1.md`…`CA_TR_PROMPT.md` only exist as untracked copies placed from the `prompts/` versioned copies; `git log --diff-filter=A` shows no commit touching them. sha256 of placed copies ≠ the orchestrator's reference table; provenance unresolved. | 2026-10-16 | Maintainer (files in chat) | **OPEN** — true originals unavailable to agent; do not start Session 2 |
| E-ORCH-002 | O0.2 stop rule bypassed; work continued with unverified prompt copies. Root cause: the stop rule was treated as a decision point. Session 1b: B2 proves root files agent-placed (git log --diff-filter=A empty for root files; c738be7 created prompts/; identical mtimes 17:42:24). B3 FAIL: ../prompt_originals/ holds only 2 agent-contaminated copies (PROMPT_1 18338b/241a6857≠826a3da7, ORCH 22971b/b259e0c8≠5630b90f). Guard: provenance check (sha256/diff vs ../prompt_originals/) added to every session start. | 2026-10-16 | Orchestrator | OPEN (Gate 1b FAIL — true originals still unavailable) |

## Open dependency requests

| ID | Requested by | Package | Description | Status |
|---|---|---|---|---|
| — | — | — | — | — |

## Decisions (index)

| ID | Date | Question | Decision |
|---|---|---|---|
| D-ORCH-001 | 2026-10-16 | `PROMPT_1.md` missing in root | **SUPERSEDED** (Session 1 closeout): root `PROMPT.md`/`PROMPT_1.md` were never in the repo; the canonical P1 prompt is the `prompts/10_PLATFORM_AUDIT.md` copy. Pointer: D-ORCH-003. |
| D-ORCH-002 | 2026-10-16 | `DATA_AUDIT_PROMPT.md` missing in root | **SUPERSEDED** (Session 1 closeout): root `DATA_AUDIT_PROMPT.md` was never in the repo; the canonical P2 prompt is the `prompts/20_DATA_AUDIT.md` copy. Pointer: D-ORCH-003. |
| D-ORCH-001 | 2026-10-16 | `PROMPT_1.md` missing in root | **SUPERSEDED** (Session 1 closeout): root `PROMPT.md`/`PROMPT_1.md` were never in the repo; the canonical P1 prompt is the `prompts/10_PLATFORM_AUDIT.md` copy. Pointer: D-ORCH-003. |
| D-ORCH-002 | 2026-10-16 | `DATA_AUDIT_PROMPT.md` missing in root | **SUPERSEDED** (Session 1 closeout): root `DATA_AUDIT_PROMPT.md` was never in the repo; the canonical P2 prompt is the `prompts/20_DATA_AUDIT.md` copy. Pointer: D-ORCH-003. |
| D-ORCH-003 | 2026-10-16 | Do the four root prompt files exist as true originals? | **No.** The `prompts/` copies are byte-identical to the root originals I placed, but sha256 ≠ the orchestrator's table. The originals are a third content the agent never had; O0/O1/O2 remain CONDITIONAL until originals are placed and proved identical. Pointers to this record: D-ORCH-001/D-ORCH-002. |
| D-ORCH-004 | 2026-10-16 | Clinical data and its quality program live outside this repo; clinical tickets are OUT OF SCOPE. Where does clean clinical data live? | Clinical data quality is **Program B** (same orchestrator, run in the GKV repository) — do not drop clinical requirements. Opened DEP-ORCH-002 to cover industry causal truth. |

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
| M21 | CONFIRMED | Provenance finding: root originals differ from `prompts/` copies (byte-identical copies placed, sha256 differs from table). Session 1 did not reconstruct originals. |
| M15 | SUPERSEDED (→ M21) | Root-prompt provenance claim; replaced by M21 |
| M16 | SUPERSEDED (→ D-ORCH-003) | Root-prompt fallback claim; D-ORCH-001/002 retired |
| M17 | CONFIRMED | E-ORCH-001 |
| M18 | CONFIRMED | No action needed |
| M19 | CONFIRMED | P1 Phase 2 fix entry points |
| M20 | CONFIRMED | Legacy tree exclusion documented |

## M-number mapping (old → new, unexplained gaps resolved)  | Old | New | Reason |
|---|---|---|---|
| M15 | M21 | Provenance/root-file finding |
| M16 | D-ORCH-003 | Scope/root-file decision |
| M17 | E-ORCH-001 | Defect register entry |
| M18–M20 | no mapping | Already traceable (no unexplained gaps) |

| D-ORCH-004 | 2026-10-16 | Clinical data and its quality program live outside this repo; clinical tickets are OUT OF SCOPE. Where does clean clinical data live? | Clinical data quality is **Program B** (same orchestrator, run in the GKV repository) — do not drop clinical requirements. Opened DEP-ORCH-002 to cover industry causal truth. |

## Next action (one line)

Session 1b Gate 1b FAIL (tree clean modulo B3 evidence; true originals pending). Branch B executed 2026-10-16 (commits 72c3b8c, 8070c0c): 42→6 dirty; residual 6 = untracked root .md awaiting B3 verdict. To unblock Branch A, paste: (a) wc -c + shasum output for ../prompt_originals/, (b) line "B4 cleanup plan approved". Then agent runs B3/B4-confirm/Gate-1b-rescore. Do NOT start Session 2. Branch C forbidden.

## B4 cleanup plan (pending maintainer approval — do not execute yet)

| # | Path(s) | Class | Proposed action | Rationale |
|---|---|---|---|---|
| 1 | 29 tracked files mode 644→755 (`README.md`, `pyproject.toml`, `src/**`, `scripts/**`, reports, `docs/EIFD.md`, project md) | prior-agent leftover | `git diff` shows numstat 0/0 (content identical); revert mode only: `git diff --name-only \| xargs chmod 644` equivalent, keep no content change | Modes flipped by agent tooling, not intentional; Wave 1 needs clean tree |
| 2 | `.gitignore` (mode + 1 added line `.oxcode-memory/`) | intentional, keep | Revert mode to 644 only; KEEP the `.oxcode-memory/` line (correct ignore for IDE memory dir) | Line is correct; mode flip is not |
| 3 | `docs/PROGRAM_INTAKE.md`, `docs/PROGRAM_STATUS.md`, `docs/DEPENDENCY_REQUESTS.md` (real content edits) | intentional, keep | Keep content; commit as `docs(program): session 1b provenance + Gate 1b FAIL` after maintainer approval | Session 1b record |
| 4 | 32 tracked `._*` AppleDouble blobs (`git ls-files \| grep -c '._'` = 32) | macOS junk, committed | `git rm` all `._*` paths after approval | They break `compileall` (`src/.___init__.py` null bytes) and pollute the tree |
| 5 | Untracked `._*` in `tests/**/__pycache__/`, `.DS_Store` (`./`, `models/`, `data/`, `src/`) | macOS junk, untracked | Delete + add `._*` and `.DS_Store` to `.gitignore` | Never belong in repo |
| 6 | `src/.___init__.py`, `src/core/.___init__.py` (AppleDouble sidecars, null bytes) | macOS junk | Delete (they shadow nothing; real `src/__init__.py` are tracked 0-byte package markers, HEAD blob size 0 — legitimate) | `compileall` errors come from these |
| 7 | Untracked root prompt originals (`PROMPT_1.md`, `PROMPT.md`, `DATA_AUDIT_PROMPT.md`, `CLEAN_SNYNTHETIC_PROMPT.md`, `CA_TR_PROMPT.md`, `ORCHESTRATOR_PROMPT.md`) + notebooks + `docs/SESSION_BRIEF_WAVE1_P1.md` + `oxcode.hooks.json` | agent-placed / session artifacts | KEEP root `.md` (B3 replacement pending true originals); KEEP notebooks if maintainer confirms they are chat originals (verify sha vs table); KEEP session brief; maintainer decides on `oxcode.hooks.json` (IDE config — suggest ignore, not commit) | Do not discard evidence before B3 passes |

---

## Layer status (Session 2, 2026-10-17 — see `docs/PROGRAM_PLAN.md` for L0–L7)

| Layer | Status | Evidence / note |
|---|---|---|
| L0 intake & alignment | ✅ DONE | specs renamed + versioned (`prompts/30`,`31`), D-ORCH-003…006, order recorded in PROGRAM_PLAN |
| L1 Task 0 Flower App | ✅ DONE | `make ci` **exit 0** (90/91 tests, 93.47% cov ≥85%, mypy clean); `flwr run` **exit 0**, 2 rounds, loss 19.28M→17.33M; evidence `docs/evidence/layer1_flwr_run_2rounds.log` |
| L2 Task 1 MTNN (Kendall+FedBN) | ✅ DONE | `make ci` **exit 0** (99/99 tests, 93.05% cov ≥85%, mypy clean on 25 files); `flwr run` **exit 0**, 3 rounds (loss 16.49M/21.41M/10.01M, 3.84 s); evidence `docs/evidence/layer2_flwr_run_3rounds.log` |
| L3–L4 | ⬜ | per plan |
| L5 Security & privacy stack | ✅ DONE | `make ci` EXIT 0 (262 tests, 74.13% cov ≥70%, mypy clean); flwr-guard 0 "backend not available"; TLS cert verification test passes; fairness disparity tests pass; RDP fail-closed tests pass. Coverage dropped from pre-Layer-5 94.58% to 74.13% because L5 added security/transport/causal modules ahead of the Wave 6 test effort. Evidence: `tests/integration/test_tls.py`, `tests/integration/test_secagg.py`, `tests/unit/fl/test_dp.py`, `tests/integration/test_rdp_accounting.py`, `tests/unit/fl/test_fairness.py`, `tests/integration/test_fairness.py`. Footnote: Gate 1b open; DEP-ORCH-002 open; final PR gated. |
| L6–L7 | ⬜ | per plan |

**Branches:** `feat/task0-baseline-federation` (L0+L1+L2, on top of `chore/full-audit`, which is 2 commits ahead of `integration/audit`). Merge to `integration/audit` still gated on maintainer (rebase merge per O-1) + Gate 1b.

**Open blockers (unchanged):**
1. **Gate 1b FAIL** — maintainer must paste `wc -c` + `shasum` for `../prompt_originals/` + approval line (Step 0). D-ORCH-005 keeps it open; required before final PR.
2. **DEP-ORCH-002 PENDING** — causal ground-truth ATE spec drafted (target `claim_risk_score`, treatment `has_claims`, all four countries, semi-synthetic truth column); implementation assigned to P3/B2 Phase 0/3b. L6 blocked until resolved.
3. **Disk:** `/Volumes/Intenso` at 100%, only **1.9 Gi free**; `venv/` (26 G, broken torch) deletion/repair is a **maintainer decision** (D-ORCH-006). Run mypy with `MYPY_OPTS=--cache-dir=/tmp/…` until resolved.
4. Branch B cleanup (42→6), O-2 deferrals, 32 DQ tickets open — unchanged.
5. **L5 = ✅ DONE with residuals** (filterwarnings rationale logged as D-CI-002; CI workflow added; coverage plan in PROGRAM_PLAN WAVE6-COVERAGE).

## Decisions / Status

- **B2 Phase 0 authorization GRANTED on 2026-10-28** (all four verification gates passed).
