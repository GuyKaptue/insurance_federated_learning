% ---
% Version: v1.0 (original from root, unmodified)
% Canonical source: CLEAN_SNYNTHETIC_PROMPT.md
% Created by orchestrator O1 on 2026-10-16
% Note: typo in original filename (SNYNTHETIC) preserved in source; copy uses corrected name
% ---

# TODO: Make the synthetic data clean (Clinical, Industry, Federated)

**Status of every item: HYPOTHESIS until the audit confirms it.** These items come from (a) the synthetic_gkv review pattern and (b) the suspect list in `PROMPT.md`. Phase 1 of `DATA_AUDIT_PROMPT.md` marks each one CONFIRMED or REFUTED with evidence. Delete refuted items; add new ones the audit finds.

Layers: **[C]** clinical, **[I]** industry, **[F]** federated.
Order follows fix dependencies, not only severity.

## Phase A: Reproducibility and quick correctness

- [ ] **DQ-001 [C][I][F] Deterministic generation across processes.** Remove `hash()` on strings, iteration over `set`, global `np.random` state; use stable hashes (`zlib.crc32`) and per-entity `SeedSequence`. *Accept:* all tables identical across 2 processes with different `PYTHONHASHSEED`, CSV and Parquet. *Test:* cross-process checksum test.
- [ ] **DQ-002 [I] Truncated-normal is really truncated.** Replace clip-based sampling with `scipy.stats.truncnorm` or rejection sampling. *Accept:* no probability mass piled at bounds. *Test:* boundary-mass test.
- [ ] **DQ-003 [F] Partition correctness.** Cumulative-proportion split; fuzz seeds and alphas. *Accept:* each index assigned exactly once, no negative or empty split. *Test:* hypothesis property test.
- [ ] **DQ-004 [C][I] Quality injection correctness.** Nullable integer dtypes, outliers not based on a NaN/earlier-outlier max, no forced arithmetic (`tenure = age + 5`), one place (generator or CLI) owns the flag. *Accept:* injected rates equal configured rates; Parquet round-trip keeps dtypes.
- [ ] **DQ-005 [C][I][F] Dead configuration removed or wired.** Every config key (node age/region settings, flags) is read by the generator or deleted. *Accept:* changing a key changes the output (diff test).
- [ ] **DQ-006 [I] Currency handling.** Explicit currency column and a normalised column; never mix EUR and CHF in one unflagged column. *Test:* schema + value test.
- [ ] **DQ-007 [C] Complete code mappings.** Every diagnosis with a prescription rule has an ICD-to-ATC/EBM mapping entry. *Accept:* no rule that can never fire (coverage test over all rules).
- [ ] **DQ-008 [C] Sex/age plausibility of specialties and codes.** Referral logic checks sex and age; one shared specialty list across all modules; remove unused practices/names. *Accept:* 0 violations in a rule-based check.

## Phase B: Foundation (state and order)

- [ ] **DQ-009 [C] Year-to-year disease memory.** Chronic status persists unless a recorded remission/cure event occurs. *Accept:* 0 "disappear then reappear" patterns without an event.
- [ ] **DQ-010 [C] Pipeline order.** Prescriptions, costs and billing generated after treatment effects/remission are known (depends on DQ-009). *Accept:* no prescription for a cured condition.
- [ ] **DQ-011 [C] Prescription realism.** Dosing frequency, supply days computed from dose, refills relative to the previous fill. *Accept:* no two fills of the same drug overlapping beyond supply.
- [ ] **DQ-012 [C][I] Switch/churn/death consistency.** Records stop at the exit date; decide whether new members join over time. *Accept:* 0 events after exit in all tables.
- [ ] **DQ-013 [C][I][F] Referential and temporal integrity suite.** PK unique, FK resolves, dates in order, nothing after death. *Accept:* 0 violations on all tables and nodes.

## Phase C: Core mechanism (targets, causal engine, leakage)

- [ ] **DQ-014 [C] Treatment-effect attribution.** Measure each treatment against the state immediately before it, not a shared snapshot.
- [ ] **DQ-015 [C] No hidden effect scaling; harm and no-effect possible; all outcome categories reachable.** Declared effect equals realised effect (within sampling error).
- [ ] **DQ-016 [C] Assignment probability by sigmoid, no cap.** Combine bonuses into a risk score, convert with a logistic function; use risk ratios (e.g. sex) as ratios. *Accept:* no probability needs clipping; share at any ceiling is 0.
- [ ] **DQ-017 [I] Leakage-free targets.** No target computed from model input features (e.g. `high_risk` from `claim_risk_score` and `credit_score`); no leakage between the 5 tasks. *Test:* feature-vs-target derivation check.
- [ ] **DQ-018 [I] Generative process matches test assertions.** Make the relationships (age-tenure, income-credit-score, income-segment) real in the generator, vectorised, monotonic effect tested; otherwise correct the assertion with justification.
- [ ] **DQ-019 [F] Preprocessing fit on train only.** Imputer/scaler/encoder fitted on train; no person across splits. *Test:* split-leak test.

## Phase D: Calibrate the numbers (each value needs a source)

- [ ] **DQ-020 [C][I] Age x sex joint distribution** sums to 100%, drawn jointly from official counts, no random fallback.
- [ ] **DQ-021 [C] Disease prevalence and incidence by age and sex** (diabetes, hypertension, CVD, COPD, depression, ...) within agreed tolerance of sourced tables; men/women ratio per band checked.
- [ ] **DQ-022 [C] Utilisation, cost, length of stay and mortality** against sourced reference.
- [ ] **DQ-023 [I] Premium, claim frequency/severity, churn, segment mix per country** against sourced references, or marked UNSOURCED and documented as assumptions.
- [ ] **DQ-024 [C] Source every treatment-effect and disease-to-disease risk number.** Add the missing diseases to the risk table. Values with no source are labelled "assumption".

## Phase E: Documentation and full re-validation

- [ ] **DQ-025 [C][I][F] Docs match code.** Rewrite METHODOLOGICAL_FRAMEWORK, README and data dictionary to describe only what exists. Replace placeholder numbers (KL values, p-values, timings, "N passed") with measured values in `reports/`.
- [ ] **DQ-026 [C][I][F] Full re-validation** of DQ-001 to DQ-025 on freshly generated data; FL smoke run on all tasks; tests green 3 times.
- [ ] **DQ-027 [C][I][F] Data contract and data-quality gate in CI.** `docs/DATA_CONTRACT.md` + `tests/data_quality/` run in `make ci` and GitHub Actions.

## Phase F: Federated node differentiation (only after A to E)

- [ ] **DQ-028 [F] Generator reads per-node settings** (age table, regions, prevalence, assignment and effect parameters), with the national/country reference as default.
- [ ] **DQ-029 [F] Deliberate, sourced differences per node/country** (age structure, regional mix, socioeconomic profile, treatment mechanism) while the union stays close to the national reference. *Accept:* per-feature KS and label-divergence matrix shows the intended differences and nothing else.
- [ ] **DQ-030 [F] Centralized oracle integrity.** Rows equal the sum of nodes, IDs unique after re-indexing, original node/ID tracked, same schema and dtypes, no duplicates.
- [ ] **DQ-031 [F] Client feature alignment.** Same columns, order, encodings and category vocabularies in every client; handling of categories missing in a client; data locality (each client holds only its own rows).
- [ ] **DQ-032 [F] Non-IID measures reported from data** (KS per feature, label divergence per task, per node pair) in `reports/summary.md`, replacing any placeholder numbers.

## Definition of done

- [ ] All findings CONFIRMED/REFUTED with evidence in `docs/DATA_REVIEW.md`
- [ ] 0 integrity violations (keys, time, domain rules) on all layers
- [ ] Reference-vs-generated tables inside agreed tolerance, or the gap documented
- [ ] Cross-process determinism proven
- [ ] No leakage in any of the 5 tasks; splits clean
- [ ] Nodes differ only in documented, sourced ways; centralized copy verified
- [ ] Docs describe the code as it is; data-quality tests run in CI