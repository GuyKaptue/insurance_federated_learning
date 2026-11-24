# Prompt Patches — Orchestrator Overrides

**Authority:** This file is the single source of truth for overrides O-1 to O-13.  
**Status:** Pending maintainer approval before applying to the versioned copies in `prompts/`.  
**Version:** v1.0  
**Date:** 2026-10-16

---

## How to read this file

Each override has:
- **Scope:** which prompt copies are affected
- **Old line:** the text as it appears in the v1.0 copy
- **New line:** the text after the patch is applied
- **Or** a narrative patch when the change is too large for a single line swap

---

## O-1 Branches

**Scope:** All prompt copies (preamble injection)  
**Type:** Additive — insert after the title/header of each copy

**Narrative patch:**
Insert the following block immediately after the version header in each copy:

```markdown
## Orchestrator Override (O-1)

**Branch rule:** Create `integration/audit` from the default branch.  
This prompt's work branch must be created from the **current tip of `integration/audit`**:
- P1 → `chore/full-audit`
- P2/B2 → `chore/data-audit`
- P3 → `feat/causal-transport`

Merge into `integration/audit` only at a wave gate, in wave order, with rebase.  
The final PR is `integration/audit` to the default branch.
```

---

## O-2 Delegation table (P1 suspect to owner)

**Scope:** `prompts/10_PLATFORM_AUDIT.md` — Phase 1 suspect list and Phase 2 fix list  
**Type:** Line-level replacements

### P1 suspect 4 (seeding)

**Old line (copy v1.0, line ~57):**
```markdown
| 4 | **Non-deterministic seeding**: `np.random.default_rng(cfg.seed + hash(country) % 10_000)` uses Python's `hash()` on a string, which is salted per process (PYTHONHASHSEED) | Run generation twice in separate processes and compare checksums. Fix with a stable hash (for example `zlib.crc32`) and add a determinism test. |
```

**New line:**
```markdown
| 4 | **Non-deterministic seeding**: `np.random.default_rng(cfg.seed + hash(country) % 10_000)` uses Python's `hash()` on a string, which is salted per process (PYTHONHASHSEED) | Run generation twice in separate processes and compare checksums. Record CONFIRMED or REFUTED in `docs/AUDIT_LOG.md`. **Do not fix here — delegated to B2 DQ-001.** |
```

### P1 suspect 5 (truncated normal)

**Old line (copy v1.0, line ~58):**
```markdown
| 5 | `_truncated_normal` computes `a, b` but never uses them and just clips (this piles mass at the bounds and is not truncated-normal) | Check boundary mass. Use `scipy.stats.truncnorm` or rejection sampling. Add a distribution test. |
```

**New line:**
```markdown
| 5 | `_truncated_normal` computes `a, b` but never uses them and just clips (this piles mass at the bounds and is not truncated-normal) | Check boundary mass. Record CONFIRMED or REFUTED. **Do not fix here — delegated to B2 DQ-002.** |
```

### P1 suspect 6 (Dirichlet)

**Old line (copy v1.0, line ~59):**
```markdown
| 6 | `dirichlet_partition`: `splits[-1] = len(idx) - splits[:-1].sum()` can go negative; floor rounding drops samples; comment about `LinAlgError` for small alpha is dubious | Fuzz with many seeds/alphas; assert every index is assigned exactly once and no split is negative. Use the cumulative-proportion split method. |
```

**New line:**
```markdown
| 6 | `dirichlet_partition`: `splits[-1] = len(idx) - splits[:-1].sum()` can go negative; floor rounding drops samples; comment about `LinAlgError` for small alpha is dubious | Fuzz with many seeds/alphas; assert every index is assigned exactly once and no split is negative. Record CONFIRMED or REFUTED. **Do not fix here — delegated to B2 DQ-003.** |
```

### P1 suspect 7, 8 (quality injection)

**Old line (copy v1.0, line ~60):**
```markdown
| 7 | `inject_quality_issues`: assigning NaN into integer columns, outlier injection based on a max that may itself be NaN or an earlier outlier, tenure forced to `age + 5` | Test dtypes after injection, parquet round-trip, and that the injected rates match configuration. |
```

**New line:**
```markdown
| 7 | `inject_quality_issues`: assigning NaN into integer columns, outlier injection based on a max that may itself be NaN or an earlier outlier, tenure forced to `age + 5` | Test dtypes after injection, parquet round-trip, and that the injected rates match configuration. Record CONFIRMED or REFUTED. **Do not fix here — delegated to B2 DQ-004.** |
```

**Old line (copy v1.0, line ~61):**
```markdown
| 8 | `GeneratorConfig.inject_quality_issues` flag exists, but quality injection is done in the CLI rather than the generator | Make behavior consistent and tested. |
```

**New line:**
```markdown
| 8 | `GeneratorConfig.inject_quality_issues` flag exists, but quality injection is done in the CLI rather than the generator | Record CONFIRMED or REFUTED. **Do not fix here — delegated to B2 DQ-004.** |
```

### P1 suspect 9, 10 (test assertions, segments)

**Old line (copy v1.0, line ~62):**
```markdown
| 9 | Tests assert relationships the generator never creates (for example `age` vs `tenure_years` correlation `> 0.1`, but tenure is drawn independently; Swiss income `> 1.3×` German mean is plausible but threshold unverified; income–credit-score `> 0.2`) | Run them, observe actual values, then either improve the generative process (preferred, documented) or correct the assertion with justification. |
```

**New line:**
```markdown
| 9 | Tests assert relationships the generator never creates (for example `age` vs `tenure_years` correlation `> 0.1`, but tenure is drawn independently; Swiss income `> 1.3×` German mean is plausible but threshold unverified; income–credit-score `> 0.2`) | Run them, observe actual values, record CONFIRMED or REFUTED. **Do not fix here — delegated to B2 DQ-018.** |
```

**Old line (copy v1.0, line ~63):**
```markdown
| 10 | Segment probabilities: `seg_logits = segment_probs * (1 + income/mean)` is not a meaningful income-dependent mapping, and a per-row Python loop with `rng.choice` is slow | Vectorize and make income genuinely shift segment odds. Test the monotonic effect. |
```

**New line:**
```markdown
| 10 | Segment probabilities: `seg_logits = segment_probs * (1 + income/mean)` is not a meaningful income-dependent mapping, and a per-row Python loop with `rng.choice` is slow | Record CONFIRMED or REFUTED. **Do not fix here — delegated to B2 DQ-018.** |
```

### P1 suspect 11 (currency)

**Old line (copy v1.0, line ~64):**
```markdown
| 11 | Mixed currencies (EUR vs CHF) mixed into combined data without normalization | Decide: keep local currency with an explicit column and document, or add a normalized column. Test it. |
```

**New line:**
```markdown
| 11 | Mixed currencies (EUR vs CHF) mixed into combined data without normalization | Record CONFIRMED or REFUTED. **Do not fix here — delegated to B2 DQ-006.** |
```

### P1 suspect 12 (fabricated evidence)

**Old line (copy v1.0, line ~65):**
```markdown
| 12 | The attached analysis contains fabricated-looking evidence (KL values, p-values, timings, "7 passed") | Replace with real measured outputs stored under `reports/`. |
```

**New line:**
```markdown
| 12 | The attached analysis contains fabricated-looking evidence (KL values, p-values, timings, "7 passed") | Replace with real measured outputs stored under `reports/`. **P1 removes placeholders in README only; full fix delegated to B2 DQ-025.** |
```

### P1 leakage check

**Narrative patch:** Add a note after the suspect table in P1 Phase 1:
```markdown
> **Override O-2:** Leakage checks (target-derived features, label leakage between tasks) stay with P1 **except** leakage-specific fixes, which are owned by B2 `DQ-017`.
```

---

## O-3 Phase order of P1

**Scope:** `prompts/10_PLATFORM_AUDIT.md` — Phase 7 and Phase 8  
**Type:** Line-level replacements + narrative note

### Phase 7 deferral

**Old line (copy v1.0, line ~143):**
```markdown
## PHASE 7 – Run all federated learning
```

**New line:**
```markdown
## PHASE 7 – Run all federated learning (DEFERRED to Wave 5 per O-3)
```

**Narrative patch:** Insert immediately after the Phase 7 heading:
```markdown
> **Override O-3:** This phase is deferred until Wave 5, after the data fixes (B2) are complete.  
> P1 Phases 0–2 run first as a behaviour-preserving port (same logic, new layout, no statistical fixes).  
> P1 Phases 3–6 follow (test mirror, Makefile, Docker, CI).  
> P2 runs its review against the **post-P1-Phase-2 port**, not the original code.
```

### Phase 8 deferral

**Old line (copy v1.0, line ~160):**
```markdown
## PHASE 8 – Full regression and final report
```

**New line:**
```markdown
## PHASE 8 – Full regression and final report (DEFERRED to Wave 6 per O-3)
```

---

## O-4 Coverage

**Scope:** `prompts/10_PLATFORM_AUDIT.md` — Phase 3, line ~97  
**Type:** Line-level replacement

**Old line:**
```markdown
6. Coverage target: at least 85% line coverage with branch coverage on, enforced in config (`--cov-fail-under=85`). Report honestly if you cannot reach it, and why.
```

**New line:**
```markdown
6. Coverage target: record the **measured baseline** in this phase and set `--cov-fail-under` to that baseline (not 85). Raise the threshold to 85 at Wave 6 after the data and causal modules exist. Every module added or rewritten later must ship with its mirrored test in the same commit (`tests/test_structure.py` enforces this).
```

---

## O-5 Single owner of the causal layer in the data

**Scope:** `prompts/30_CAUSAL_TRANSPORT.md` — Phase 6, Step 6.1  
**Type:** Line-level replacement

**Old lines (copy v1.0, lines ~216–220):**
```markdown
Step 6.1 Treatment, outcome, covariates
- Use the Phase 0 findings. If the data lacks a treatment/outcome with known truth, add a documented **causal layer** to the generator (config-driven): treatment assignment through a logistic score of covariates, an outcome with a known conditional effect that depends on effect modifiers, per-country differences in the effect-modifier distribution. Every parameter needs a documented reason; sourced values where possible, otherwise labelled "assumption". This links to the data cleanup items on assignment probability, effect attribution, and node differences.
- Keep this layer separate from the five existing ML targets so nothing changes their labels (leakage check test).
```

**New lines:**
```markdown
Step 6.1 Treatment, outcome, covariates
- Use the Phase 0 findings. **Do not modify the generator.** The causal layer (treatment assignment, effect mechanism, node differences with known truth) is built by B2 (DQ-009, DQ-010, DQ-014 to DQ-016, DQ-028, DQ-029). P3 consumes the truth columns for evaluation only.
- If the data lacks a treatment/outcome with known truth, open a `DQ-xxx` request in `docs/DEPENDENCY_REQUESTS.md` rather than modifying the generator.
- Keep the causal layer separate from the five existing ML targets so nothing changes their labels (leakage check test).
```

---

## O-6 FL glue interface

**Scope:** `prompts/10_PLATFORM_AUDIT.md` — Phase 2, item 2 (fl/ design)  
**Type:** Line-level replacement

**Old line (copy v1.0, line ~81):**
```markdown
- `fl/client.py`, `fl/server.py`, `fl/strategies.py` using **Flower** (`flwr`). Support FedAvg and FedProx at minimum, optionally differential privacy (clipping + noise) behind a config flag. Provide both in-process simulation and real gRPC server/client modes.
```

**New line:**
```markdown
- `fl/client.py`, `fl/server.py`, `fl/strategies.py` using **Flower** (`flwr`). Support FedAvg and FedProx at minimum, optionally differential privacy (clipping + noise) behind a config flag. Provide both in-process simulation and real gRPC server/client modes. **Additionally, expose a generic `ClientProtocol` abstraction so a round can return a dict of named numpy arrays plus metadata (not only model weights). This allows P3's aggregate protocols to plug in without a rewrite. Add a test with a dummy aggregate protocol.**
```

---

## O-7 Logs and IDs

**Scope:** All prompt copies (preamble injection)  
**Type:** Additive

**Narrative patch:** Insert after O-1 block in each copy:

```markdown
## Orchestrator Override (O-7)

**Unified status:** All gates and decisions are indexed in `docs/PROGRAM_STATUS.md`.  
**ID namespaces:**
- P1 errors: `E-P1-nnn`
- Data tickets: `DQ-nnn`
- Causal tasks: `CT-nnn`
- Decisions: `D-P1-nnn`, `D-DQ-nnn`, `D-CT-nnn`, `D-ORCH-nnn`

P1's error register must cross-link to DQ IDs when delegated per O-2.
```

---

## O-8 Data lineage

**Scope:** `prompts/10_PLATFORM_AUDIT.md` — Phase 4 (Makefile)  
**Type:** Additive

**Narrative patch:** Add to the Makefile target table in P1 Phase 4:

```markdown
| `data-manifest` | Write `data/processed/MANIFEST.json` with generator code hash, config hash, seed, schema hash, row counts, `data_version` |
```

And add to Gate 4 evidence requirements:
```markdown
**Gate 4 (addition):** `make data-manifest` produces a valid `data/processed/MANIFEST.json`. Every report and run folder stores the current `data_version`. Reports with a different `data_version` than the manifest are marked STALE and must be regenerated.
```

---

## O-9 Freeze timeline

**Scope:** All prompt copies (preamble injection)  
**Type:** Additive

**Narrative patch:** Insert after O-7 block in each copy:

```markdown
## Orchestrator Override (O-9)

**Freeze timeline:** Source files may change only in the waves where the execution plan says so.  
- Waves 0 and 2b (P2 review part) are **read-only** for source code.  
- A prompt's own "do not modify" rule applies inside its wave, and nothing else overrides this timeline.
```

---

## O-10 Shared files

**Scope:** All prompt copies (preamble injection)  
**Type:** Additive

**Narrative patch:** Insert after O-9 block in each copy:

```markdown
## Orchestrator Override (O-10)

**Shared file ownership:** P1 owns `pyproject.toml`, `Makefile`, `Dockerfile`, `docker-compose.yml`, `.github/workflows/ci.yml`, `README.md`, `tests/test_structure.py`.

Other prompts add **only** via:
- `make/data_quality.mk` and `make/causal.mk` (included by the Makefile)
- Extra CI jobs appended in their own marked block
- Dependency requests in `docs/DEPENDENCY_REQUESTS.md` (P1 applies them in the next P1 commit)
- README sections under their own heading

**Never rewrite another owner's block.**
```

---

## O-11 Privacy

**Scope:** `prompts/10_PLATFORM_AUDIT.md` (Phase 7) and `prompts/30_CAUSAL_TRANSPORT.md` (Phase 4)  
**Type:** Line-level replacements

### P1 Phase 7 privacy line

**Old line (copy v1.0, line ~150):**
```markdown
- **Privacy:** DP off and on (report epsilon/delta if DP is implemented and accounted correctly; otherwise say it is not formally accounted).
```

**New line:**
```markdown
- **Privacy:** DP off and on. Formal (epsilon, delta) claims are made **only** where a cumulative RDP accountant is implemented and verified. Otherwise every report must state "not formally accounted". One privacy section lives in `config/federated_config.yaml`.
```

### P3 Phase 4 privacy line

**Old line (copy v1.0, line ~175):**
```markdown
- Optional Gaussian noise on aggregates behind a flag. Document that formal (epsilon, delta) is **not** claimed unless an accountant is implemented and verified.
```

**New line:**
```markdown
- Optional Gaussian noise on aggregates behind a flag. Document that formal (epsilon, delta) is **not** claimed unless an accountant is implemented and verified. Reuse the repo's cumulative RDP accountant if it exists; otherwise state "not formally accounted" in every report.
```

---

## O-12 Disk guard and resume

**Scope:** `prompts/10_PLATFORM_AUDIT.md` (Phase 1 and general rules)  
**Type:** Additive

**Narrative patch:** P1 currently lacks a disk-guard rule. Insert after P1 rule 7 (line ~32):

```markdown
8. **Disk guard.** Before every session and every heavy run: `df -h .`; stop if less than 5 GB is free. All heavy outputs as Parquet, one copy only; `reports/runs/` and `data/` stay out of git.
9. **Resume ritual.** If interrupted, run the session start ritual (`git status`, `compileall`, zero-byte file check) before continuing. Never continue from a half-written file.
```

(Existing P1 rules 8 and 9 become 10 and 11.)

---

## O-13 Run manifests

**Scope:** `prompts/10_PLATFORM_AUDIT.md` (Phase 7) and `prompts/30_CAUSAL_TRANSPORT.md` (Phase 7)  
**Type:** Additive

**Narrative patch:** Add to P1 Phase 7 "For each run save" list:
```markdown
- `manifest.json` with schema: `run_id`, `prompt_tag`, `config_hash`, `seed`, `data_version`, `git_commit`, `wall_time_s`, `metrics_path`
```

Add to P3 Phase 7 Step 7.2:
```markdown
- Save `manifest.json` per run using the shared schema defined in O-13 (`run_id`, `prompt_tag`, `config_hash`, `seed`, `data_version`, `git_commit`, `wall_time_s`, `metrics_path`).
```

---

## Summary of patches per copy

| Copy | Overrides affecting it | Patch count |
|---|---|---|
| `00_ORCHESTRATOR.md` | O-1, O-7, O-9, O-10 | 4 additive |
| `10_PLATFORM_AUDIT.md` | O-1, O-2, O-3, O-4, O-6, O-7, O-8, O-9, O-10, O-11, O-12, O-13 | 12 line-level + 6 additive |
| `20_DATA_AUDIT.md` | O-1, O-7, O-9, O-10 | 4 additive |
| `21_DATA_CLEANUP_BACKLOG.md` | O-1, O-7, O-9, O-10 | 4 additive |
| `30_CAUSAL_TRANSPORT.md` | O-1, O-5, O-7, O-9, O-10, O-11, O-13 | 3 line-level + 5 additive |

---

## Approval block

```
[ ] Maintainer has reviewed PATCHES.md
[ ] Maintainer approves applying patches to the copies
[ ] Patches applied in commit: ___________________
```
