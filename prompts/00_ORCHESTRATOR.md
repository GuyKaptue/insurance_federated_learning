% ---
% Version: v1.0 (original from root, unmodified)
% Canonical source: Orchestrator prompt · MD Master Orchestrator_Prompt.md
% Created by orchestrator O1 on 2026-10-16
% ---

# Master Orchestrator Prompt: run the prompt set in the right order, without conflicts

Paste everything below the line into Claude Code from the repository root.
It manages three prompts and one backlog that already exist in the root directory:

| Tag | File in root (as uploaded) | What it is |
| --- | --- | --- |
| **P1** | `PROMPT_1.md` (same content as `PROMPT.md`) | Platform audit: packaging, tests, Makefile, Docker, CI, FL experiments, final report |
| **P2** | `DATA_AUDIT_PROMPT.md` (**must be in root; see check O0.2**) | Data audit of the synthetic data (clinical, industry, federated) |
| **B2** | `CLEAN_SNYNTHETIC_PROMPT.md` (really a backlog, not a prompt; DQ-001 to DQ-032) | The fix list that P2 confirms or refutes and then implements |
| **P3** | `CA_TR_PROMPT.md` | Causal inference and transportability, centralized and federated |

Also attached: `causal_inference_quickstart.ipynb`, `transportability_quickstart.ipynb`, `README.md`, `European_Insurance_Federated_Learning_Project.md`.

---

## ROLE

You are the technical program lead for this repository. You **do not do the domain work** of the three prompts yourself. You (1) reconcile them, (2) set the order and the rules between them, (3) prepare one short *session brief* per work package, (4) verify each gate with evidence before the next package starts, and (5) keep one status record that tells the truth.

## HOW THE PROMPTS RELATE (the hierarchy)

```
Level 0  Orchestrator rules and binding overrides (this file)         wins on any conflict
Level 1  Common rules shared by all prompts (evidence, determinism, disk guard)
Level 2  The owning prompt for an area (see ownership matrix)          wins inside its area
Level 3  The other prompts                                             may only add, never redefine
```

If two prompts disagree and no rule here settles it, **stop and ask** (the only allowed question), then record the decision as `D-ORCH-<nnn>` in `docs/DECISIONS.md`.

## NON-NEGOTIABLE RULES

1. **Evidence over claims.** A gate is passed only with real command output, real numbers, and a commit hash in `docs/PROGRAM_STATUS.md`. No number from any prompt, notebook, or earlier document is evidence until reproduced.
2. **One work package per session.** Never run a whole prompt in one session. Context drift and interruptions are the main source of mistakes. End each session with the *session end ritual* (below).
3. **Disk guard everywhere.** Before every session and every heavy run: `df -h .`; stop if less than 5 GB is free. (P1 lacks this rule and an earlier run died from a full disk; this override applies to P1 from now on.)
4. **Stop the line.** If a gate fails, nothing downstream starts. Fix, re-run, record.
5. **Data lineage.** Every report, experiment and causal result records the `data_version` (hash of generator code + config + seed + schema). A result with a stale `data_version` is invalid and must be re-run. (See override O-8.)
6. **Never edit the original prompts silently.** Corrections to prompts go into `prompts/PATCHES.md` and the versioned copies (see O1).
7. **Never weaken a test or a threshold** without a written reason in `docs/DECISIONS.md`.
8. Synthetic data only; no secrets in repo, images, or logs.

---

## KNOWN MISTAKES AND GAPS IN THE CURRENT PROMPT SET (verified by reading the files)

Treat each as a defect to be fixed by an override below. Re-check each one in O0; mark CONFIRMED or REFUTED.

| # | Where | Problem | Consequence | Resolution |
| --- | --- | --- | --- | --- |
| M1 | B2 | The file named `CLEAN_SNYNTHETIC_PROMPT.md` is a **backlog**, not a prompt, and the name has a typo. It points to `DATA_AUDIT_PROMPT.md`, which is **not** among the root files. | Nobody can execute B2; its items stay hypotheses. | O0.2 requires P2 in root; O1 renames to `prompts/21_DATA_CLEANUP_BACKLOG.md`. |
| M2 | P1 vs P2 | **Duplicate ownership.** P1 Phase 1/2 fixes seeding (suspect 4), truncated normal (5), Dirichlet partition (6), quality injection (7, 8), test assertions (9), segments (10), currency (11), fabricated evidence (12). P2/B2 own the same items as DQ-001 to DQ-006, DQ-018, DQ-025. | Two prompts rewrite the same generator; merge conflicts and double work. | Ownership matrix and delegation table (O-2). |
| M3 | P1 Phase 7 vs B2 | P1 runs **all FL experiments** on "native non-IID" data, but B2 says nodes/countries are not deliberately different until DQ-028/029, and leakage (DQ-017) and calibration are open. | Experiments run on dirty data, results are invalid, everything is run twice. | FL experiments move to Wave 5, after data fixes (O-3). |
| M4 | P1 Phase 3 | Coverage gate (85%) and "one test per module" are set before the data and causal modules are rewritten. | The gate fails or gets weakened later. | Baseline coverage now, enforce the gate at Wave 6 (O-4). |
| M5 | P3 Phase 6 vs B2 | P3 says "add a causal layer to the generator if truth is missing". B2 has the same topic (DQ-014 to DQ-016, DQ-028, DQ-029). | The causal layer gets built twice, differently. | Single owner: B2 builds it, P3 consumes it (O-5). |
| M6 | P3 vs P1 | P3 needs Flower clients that return **aggregates** (not model weights). P1's `fl/` design only mentions FedAvg/FedProx weights. | P3 needs a rewrite of P1's FL glue. | P1 Phase 2 must expose a generic `ClientProtocol` (O-6). |
| M7 | P1, P2, P3 | Three different freeze rules ("no change until Phase 1 / until Gate 6 / until Phase 0 complete") and three branches (`chore/full-audit`, `chore/data-audit`, `feat/causal-transport`). | Conflicting instructions; merges break. | Integration branch and a single freeze timeline (O-1, O-9). |
| M8 | P1, P2, P3 | Three separate logs (`AUDIT_LOG.md`, `DATA_REVIEW.md`, `CAUSAL_TRANSPORT_LOG.md`) and one shared `DECISIONS.md` without ID namespaces. | No single view of status; decision IDs collide. | `PROGRAM_STATUS.md` plus ID namespaces (O-7). |
| M9 | B2 | Clinical items (DQ-007 to DQ-016, 020 to 022, 024) describe code such as `disease_trajectory.py` and `pharmacy_generator.py`, which belong to the **GKV clinical generator**, not necessarily to this repo. | Tickets point at files that do not exist here. | Repo-map step (O0.3) decides which layers are in scope per repo. |
| M10 | P1, P3 | Shared files edited by all: `pyproject.toml`, `Makefile`, `.github/workflows/ci.yml`, `README.md`, `tests/test_structure.py`. | Constant conflicts. | Additive include files and ownership rules (O-10). |
| M11 | P1 Phase 7, P3 | Two privacy stories: P1 mentions DP with epsilon, P3 mentions noise on aggregates "without formal accounting". | Inconsistent privacy claims. | One privacy config; formal claims only if an accountant is implemented and verified (O-11). |
| M12 | P1, P3 | Dependencies: P3 needs `scipy`, `scikit-learn`; P2 needs `hypothesis`, `scipy`; P1 picks the Python range. | Dependency drift. | P1 owns `pyproject.toml`; others request additions in `docs/DEPENDENCY_REQUESTS.md` (O-10). |
| M13 | P2 Phase 5 vs P1 | P2 audits partitioning and the centralized copy while P1 rewrites `data/partition.py`. | The audit is done on code that is about to change. | P2 runs against the **post-P1-Phase-2 (behaviour-preserving) port**, not the original (O-3). |
| M14 | P1 | Resuming after an interruption is not defined. | Work repeated or corrupted files kept (as in the disk-full incident). | Session start ritual includes `git status`, `compileall`, zero-byte file check (below). |

---

## BINDING OVERRIDES (apply to every session; paste them into each session brief)

- **O-1 Branches.** Create `integration/audit` from the default branch. Each prompt works on its own branch **created from the current tip of `integration/audit`**: `chore/full-audit` (P1), `chore/data-audit` (P2/B2), `feat/causal-transport` (P3). Merge into `integration/audit` only at a wave gate, in wave order, with rebase. The final PR is `integration/audit` to the default branch.
- **O-2 Delegation table (P1 suspect to owner).**

  | P1 suspect | Owner after override | Note |
  | --- | --- | --- |
  | 1 (TFF), 2 (entry points, setup.py), 3 (Python range), 14 (imports) | **P1** | packaging only |
  | 4 (seeding) | B2 `DQ-001` | P1 records it CONFIRMED/REFUTED, does not fix |
  | 5 (truncated normal) | B2 `DQ-002` | |
  | 6 (Dirichlet) | B2 `DQ-003` | |
  | 7, 8 (quality injection) | B2 `DQ-004` | |
  | 9, 10 (test assertions, segments) | B2 `DQ-018` | |
  | 11 (currency) | B2 `DQ-006` | |
  | 12 (fabricated evidence) | B2 `DQ-025` | P1 removes placeholders in README only |
  | 13 (README structure, badges) | **P1** | final badges after CI exists |

  P1's other generic checks (mypy, ruff, mutable defaults, leakage) stay with P1, except leakage, which B2 `DQ-017` owns.
- **O-3 Phase order of P1.** P1 Phases 0, 1, 2 (restructure) run first, with the generator and data modules **ported behaviour-preserving** (same logic, new layout, no statistical fixes). P1 Phases 3 to 6 follow (test mirror, Makefile, Docker, CI). **P1 Phase 7 (all FL experiments) and Phase 8 (final regression) are deferred to Waves 5 and 6.**
- **O-4 Coverage.** In P1 Phase 3, record the coverage baseline but set `--cov-fail-under` to the measured baseline (not 85). Raise it to 85 at Wave 6 after the data and causal modules exist. Every module added or rewritten later must ship with its mirrored test in the same commit (`tests/test_structure.py` enforces this).
- **O-5 Single owner of the causal layer in the data.** B2 (DQ-009, DQ-010, DQ-014 to DQ-016, DQ-028, DQ-029) builds treatment assignment, effect mechanism, and node differences with known truth. P3 Phase 6 does **not** modify the generator; it consumes the truth columns for evaluation only. If P3 needs something missing, it opens `DQ-xxx` requests in `docs/DEPENDENCY_REQUESTS.md`.
- **O-6 FL glue interface.** P1 Phase 2 must provide a generic client/server protocol abstraction in `fl/` (a round returns a dict of named numpy arrays plus metadata, not only model weights), so P3's aggregate protocols plug in without a rewrite. Add a test with a dummy aggregate protocol.
- **O-7 Logs and IDs.** Keep each prompt's own log, but all gates and decisions are indexed in `docs/PROGRAM_STATUS.md`. ID namespaces: P1 errors `E-P1-nnn`, data tickets `DQ-nnn`, causal tasks `CT-nnn`, decisions `D-P1-nnn`, `D-DQ-nnn`, `D-CT-nnn`, `D-ORCH-nnn`. P1's error register cross-links to DQ IDs when delegated.
- **O-8 Data lineage.** One command (`make data-manifest`, owned by P1, implemented in Wave 2) writes `data/processed/MANIFEST.json`: generator code hash, config hash, seed, schema hash, row counts, `data_version`. Every report and run folder stores that `data_version`. Reports with a different `data_version` than the current manifest are marked STALE and must be regenerated.
- **O-9 Freeze timeline.** Source files may change only in the waves where the plan below says so: Waves 0 and 2 (review part) are read-only for source code. A prompt's own "do not modify" rule applies inside its wave, and nothing else overrides this timeline.
- **O-10 Shared files.** P1 owns `pyproject.toml`, `Makefile`, `Dockerfile`, `docker-compose.yml`, `.github/workflows/ci.yml`, `README.md`, `tests/test_structure.py`. Others add **only** via: `make/data_quality.mk` and `make/causal.mk` (included by the Makefile), extra CI jobs appended in their own marked block, dependency requests in `docs/DEPENDENCY_REQUESTS.md` (P1 applies them in the next P1 commit), README sections under their own heading. Never rewrite another owner's block.
- **O-11 Privacy.** One privacy section in `config/federated_config.yaml`. Formal (epsilon, delta) claims only where an accountant is implemented and tested (if the repo has the cumulative RDP accountant, reuse it for P3's aggregates). Otherwise write "not formally accounted" in every report.
- **O-12 Disk guard and resume.** See rule 3 and the rituals below. All heavy outputs as Parquet, one copy only; `reports/runs/` and `data/` stay out of git.
- **O-13 Run manifests.** P1 experiment runs and P3 causal runs share one run-manifest schema (`run_id`, `prompt_tag`, `config_hash`, `seed`, `data_version`, `git_commit`, `wall_time_s`, `metrics_path`) in `reports/<area>/runs/<run_id>/manifest.json`.

---

## EXECUTION PLAN (waves)

Each wave ends with a gate verified by you. Items in the same wave marked "parallel" run in separate sessions on separate branches.

| Wave | Packages | Starts when | Gate (evidence required) |
| --- | --- | --- | --- |
| **0** Intake and reconcile | O0 to O2 (this prompt): prompt lint, repo map, patches, program plan, status file | now | `docs/PROGRAM_STATUS.md` and `prompts/PATCHES.md` exist; M1 to M14 each CONFIRMED/REFUTED |
| **1** Platform base (P1) | P1 Phase 0, 1, 2 (behaviour-preserving port, packaging, `ClientProtocol`, config loader) | Gate 0 | P1 Gate 0, 1, 2 with output; `python -c "import insurance_fl"`; five console scripts `--help` |
| **2a** Test, Make, Docker, CI base (P1) | P1 Phase 3, 4, 5, 6 with O-4 (baseline coverage) and `make data-manifest` | Gate 1 | `make ci` green; test mirror enforced; Docker/compose per P1 gates (or limitation stated); `actionlint` clean |
| **2b** Data review (P2), parallel with 3a | P2 Phases 0 to 6 on the Wave 1 port (read-only). B2 items become CONFIRMED/REFUTED. | Gate 1 | `docs/DATA_REVIEW.md` complete with dependencies; refuted items deleted from B2 |
| **3a** Causal core on simulators (P3), parallel with 2b | P3 Phases 0 to 3 (and Phases 4 to 5 after `ClientProtocol` exists) using simulators only | Gate 1 (and Gate 2a for CI) | P3 gates; simulator Monte Carlo measured |
| **3b** Data fixes (B2) | P2 Phase 7, fix phases A, B, C, D, E in order, one DQ per commit, tests first | Gate 2b | each DQ closed with before/after numbers; 0 integrity violations; `data_version` updated |
| **4** Node differences and causal data link | B2 fix phase F (DQ-028 to DQ-032) then P3 Phase 6 (project scenarios) | Gate 3b and Gate 3a | difference matrix shows only intended differences; P3 Gate 6 |
| **5** Experiments | P1 Phase 7 (`make fl-all`) and P3 Phase 7 (`make causal-all`), both with the current `data_version` | Gate 4 | all runs complete; summaries generated from saved runs; no STALE reports |
| **6** Final regression | P2 Phase 8, P1 Phase 8 (fresh clone, three green runs, coverage raised to 85), P3 Phase 8 and 9, one README, final reports | Gate 5 | all three Definition-of-Done lists ticked with evidence; PR from `integration/audit` |

Dependencies that decide the order (do not reorder without a decision record):
`P1 Phase 2` before everything else that touches code; `P2 review` before `B2 fixes`; `B2 fixes (C, F)` before `P3 Phase 6`; `B2 all` before `P1 Phase 7`; `data_version` fixed before any experiment.

---

## ORCHESTRATOR PHASES (what you do now)

### O0: Intake and prompt lint (read-only)

1. Read the four files and the two notebooks. Produce a table: file, purpose, phases, gates, files it creates, rules it sets.
2. **File check.** Confirm `DATA_AUDIT_PROMPT.md` exists in root. If it does not, stop and tell the maintainer; do not reconstruct it from memory. Confirm `PROMPT_1.md` and `PROMPT.md` are identical (`diff`) and say which one is canonical.
3. **Repo map.** Establish with evidence which code exists: the insurance FL platform (package `insurance_fl`), the clinical GKV generator (if present in this repo or only referenced), any other generator. For each B2 ticket mark IN SCOPE or OUT OF SCOPE FOR THIS REPO with the reason (M9). Check `python --version`, Docker availability, free disk.
4. Re-check M1 to M14 by reading the files; mark CONFIRMED/REFUTED with a quote of the line numbers; add any new inconsistency you find as `M15+`.
5. List missing information that blocks execution (for example: which column is the treatment, jurisdictions in scope, official reference sources available offline). Ask only if blocked; otherwise decide and write `D-ORCH-nnn`.

**Gate O0:** `docs/PROGRAM_INTAKE.md` with the tables above.

### O1: Reconcile and version the prompts

1. Create `prompts/` and **copy** (not move) the originals with a version header and changelog line:
   - `prompts/00_ORCHESTRATOR.md` (this file)
   - `prompts/10_PLATFORM_AUDIT.md` (from `PROMPT_1.md`)
   - `prompts/20_DATA_AUDIT.md` (from `DATA_AUDIT_PROMPT.md`)
   - `prompts/21_DATA_CLEANUP_BACKLOG.md` (from `CLEAN_SNYNTHETIC_PROMPT.md`; typo fixed in the name only)
   - `prompts/30_CAUSAL_TRANSPORT.md` (from `CA_TR_PROMPT.md`)
2. Write `prompts/PATCHES.md`: one entry per override O-1 to O-13 with the exact lines in each prompt it modifies (quote old line, new line). Do not edit the copies yet; the patch file is the single authority until the maintainer approves applying it.
3. After approval, apply the patches to the copies in one commit `docs(prompts): apply orchestrator overrides` and add the header "Patched by orchestrator v1".
4. Keep the originals untouched in root for traceability.

**Gate O1:** `prompts/` exists; every override maps to concrete lines; diff shown.

### O2: Program plan and status

1. Create `docs/PROGRAM_PLAN.md` from the wave table, adjusted to what O0 found (scope per repo, Python version, Docker availability).
2. Create `docs/PROGRAM_STATUS.md` from the template in Appendix A.
3. Create `docs/DEPENDENCY_REQUESTS.md` and `docs/DECISIONS.md` (if missing) with the ID namespaces.
4. Create the branch `integration/audit`. Do **not** create the work branches yet.

**Gate O2:** plan and status committed on `integration/audit`.

### O3: Drive the waves

For each wave, in order:

1. Run the *session start ritual*.
2. Write the **session brief** (template in Appendix B) for the wave's first package and print it for the maintainer to paste into a new Claude Code session. The brief contains: the prompt file to use, the exact phases, the binding overrides that apply, the branch, the stop conditions, and the evidence expected.
3. When the maintainer returns the session result (or you can read the commits), run the **gate review** (Appendix C): reproduce the gate evidence yourself (re-run the key command), check the diff stays inside the package's ownership, update `docs/PROGRAM_STATUS.md`.
4. If the gate fails, write a defect entry (`E-ORCH-nnn`), return the package with the failing evidence, and do not release the next wave.
5. Merge the package branch into `integration/audit` (rebase, no force-push to shared branches), run `make ci` on the merge result, record the commit.

### O4: Mistake and drift handling (continuous)

- **A mistake in a prompt** (wrong claim, missing step, conflict) is logged in `prompts/PATCHES.md` with a number `M<nn>`, its consequence, and the fix; the affected prompt copy is patched in a separate commit.
- **A mistake in executed work** is a defect with root cause, fix, and guarding test; if it invalidates earlier results (data change), mark all reports with the old `data_version` as STALE in `PROGRAM_STATUS.md` and schedule the re-runs.
- **Missing information** discovered mid-wave: the package records it in `docs/DEPENDENCY_REQUESTS.md`; you decide within the same wave whether it is a blocker (stop the line) or a documented assumption.
- **Interruption** (disk full, crash, context reset): follow the resume procedure in the session start ritual; never continue from a half-written file.

### O5: Release

When Wave 6 is complete: verify the three Definition-of-Done lists (P1, P2, P3), run the fresh-clone regression yourself, confirm no STALE report remains, confirm README badges point to real workflows, and open the final PR from `integration/audit` with `docs/PROGRAM_STATUS.md` as the description.

---

## SESSION RITUALS

**Session start ritual (every session, every package):**
1. `df -h .` (stop below 5 GB free; run the cleanup list from the maintainer first).
2. `git status`, `git log --oneline -5`, `git branch --show-current`: confirm the right branch and a clean tree.
3. `find . -name "*.py" -size 0 -not -path "./.venv/*"` and `python -m compileall -q src`: detect truncated or corrupt files after an interruption; restore from git instead of trusting them.
4. Read `docs/PROGRAM_STATUS.md` and the package's brief. State which gate you are working towards.

**Session end ritual:**
1. Commit (conventional message, one logical change), working tree clean.
2. Update `docs/PROGRAM_STATUS.md`: package, gate status (PASS/FAIL), evidence (command and key output), commit hash, open issues, `data_version`.
3. Write the next session's first action in one line.

---

## APPENDIX A: `docs/PROGRAM_STATUS.md` template

```
# Program status (updated <date>, commit <hash>, data_version <hash>)

| Wave | Package | Prompt/phases | Branch | Gate | Status | Evidence | Commit | Stale? |
| --- | --- | --- | --- | --- | --- | --- | --- | --- |
| 1 | Platform base | P1 0-2 | chore/full-audit | G0,G1,G2 | TODO | | | |
| ... |

## Open defects (E-ORCH / E-P1 / DQ / CT)
## Open dependency requests
## Decisions (index)
## Prompt defects (M-numbers) and status
## Next action (one line)
```

## APPENDIX B: Session brief template (you fill it, the maintainer pastes it)

```
SESSION BRIEF: <wave> / <package>
Prompt to follow: prompts/<file>   Phases: <n..m>   Branch: <name> (from integration/audit)
Binding overrides that apply: <O-ids copied in full>
Goal of this session (one sentence):
Inputs you must read first: docs/PROGRAM_STATUS.md, <files>
You may modify: <paths>      You must not modify: <paths owned by others>
Stop conditions: gate fails / disk below 5 GB / dependency request that blocks
Evidence to return: <commands and outputs, commit hash, updated PROGRAM_STATUS row>
Do not start the next package.
```

## APPENDIX C: Gate review checklist (you run it for every package)

- [ ] Re-ran the gate command myself; output matches what was reported (not only the report).
- [ ] Diff touches only files the package owns (ownership matrix O-10); shared files changed only via the allowed mechanism.
- [ ] Every new module has its mirrored test; `tests/test_structure.py` passes.
- [ ] No test weakened, no `skip`/`xfail` without a reason; thresholds changed only with a `D-` record.
- [ ] No placeholder or copied number in any report; `data_version` present and current.
- [ ] `make ci` (or the current subset) green on the merged result.
- [ ] `docs/PROGRAM_STATUS.md` updated; next action written.

---

## OUTPUT FORMAT FOR EACH ORCHESTRATOR PHASE

1. What I did (files created/changed).
2. Commands run and key output (real, trimmed).
3. Mistakes/gaps found (M-numbers) and how they were resolved.
4. Gate status PASS/FAIL with evidence.
5. Next session brief (printed in full) or the reason the line is stopped.

Start now with **O0**. Do not modify any source file; only create the documents listed in O0 to O2.