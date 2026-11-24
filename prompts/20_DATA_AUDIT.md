% ---
% Version: v1.0 (original from root, unmodified)
% Canonical source: Data audit prompt.md
% Created by orchestrator O1 on 2026-10-16
% ---

# Master Prompt: Synthetic Data Audit (Clinical, Industry, Federated)

Paste everything below the line into Claude Code from the repository root.
Attach `docs/synthetic_dataset_review_TEMPLATE.md` (the synthetic_gkv review by Maria Petralia) as the **format and depth template**. It is a template for *how* to review, not a source of facts about this data.

---

## ROLE

You are a senior data-quality and synthetic-data engineer (statistics, healthcare/insurance data, causal inference, federated learning). You review the synthetic data produced by this repository the way a skeptical external reviewer would: by running the generator, looking at real output rows, and comparing them with external reference numbers.

## DATA LAYERS IN SCOPE

| Layer | What it is | Typical contents |
| --- | --- | --- |
| **Clinical** | Patient-level health records | persons, diagnoses (ICD-10-GM), ambulatory acts (EBM), prescriptions (PZN/ATC), hospital episodes (G-DRG/OPS), mortality, interventions and any causal ground truth (ITE/ATE/CATE, counterfactuals) |
| **Industry** | Insurer/business data | policies, claims, premiums, costs, billing, customers, per-country profiles (DE/FR/BE/CH), currencies, 5 ML task targets |
| **Federated** | The node/client structure on top | per-node/per-country datasets, non-IID partitions, centralized oracle copy, train/val/test splits, ID re-indexing, FL experiment inputs |

Locate the real code and data for each layer first (`git ls-files`, `data/`, `configs/`, generator modules). If a layer does not exist in this repo, say so and skip it. Do not invent it.

## MISSION

Produce a review in the exact style of the template: numbered problems, each with **Ticket, File, What it means, Example, Fix**, grouped into phases in the order they should be fixed (not only by severity). Then implement the fixes phase by phase with tests.

## NON-NEGOTIABLE RULES

1. **Evidence over claims.** Never say something is right or wrong unless you ran code and saw the output. Quote the command and the real numbers. Never copy numbers from the template or from existing docs; those are unverified until you reproduce them.
2. **"The check passed" is not evidence.** For every existing validation (tests, reports, "PASSED" labels) state what it actually verifies. A check that only confirms the code runs and files export says nothing about correctness. Say this explicitly where it applies (Phase 0).
3. **Trace real example records.** Generate a tiny sample (about 5 persons or policies), pick them, and use the same 5 as running examples for the bugs they show. Bugs that only appear at scale are marked "confirmed on larger run (N rows)".
4. **Every external reference needs a source.** For each distribution you compare against, record source, table, year, population, and licence. If you cannot find a source, mark the item UNSOURCED, do not guess a number.
5. **Root cause, not patches.** Per problem: symptom, root cause, fix, guarding test. Never weaken a test or loosen a threshold without writing why.
6. **Do not change code during the review (Phases 0 to 6).** Only add read-only analysis scripts under `tests/data_quality/` and outputs under `reports/data_audit/`.
7. **Ask only when blocked.** Otherwise decide, and log the decision in `docs/DECISIONS.md`.
8. **Disk safety.** Before each phase and before any full generation run, run `df -h .` and stop if less than 5 GB is free. Write large outputs as Parquet, never keep multiple full copies.
9. **Synthetic only.** No real personal data, no credentials in outputs or logs.
10. Small commits on branch `chore/data-audit`, conventional-commit messages.

## OUTPUT FILES

- `docs/DATA_REVIEW.md`: the review, in the template's format (this is the main deliverable).
- `docs/DATA_CONTRACT.md`: the data contract (schemas, keys, ranges, units) found and the one that should hold.
- `tests/data_quality/`: one executable check per finding (so the finding can be re-run).
- `reports/data_audit/`: measured outputs (tables, plots, checksums).

Each finding in `docs/DATA_REVIEW.md` uses this block:

```
### <n>. <short title>
Ticket: DQ-<nnn> (depends on DQ-<mmm> if any)
Layer: Clinical | Industry | Federated
File: <real file paths>
Status: CONFIRMED | REFUTED | UNSOURCED   (with the command that shows it)
What it means: <plain language, one paragraph>
Example: <real numbers from a real run, or a traced example record>
Fix: <proposed fix, with the reference source if distributional>
Guarding test: <test file::test name>
```

---

## PHASE 0: What the existing validation really covers (read-only, no code)

1. List every existing validation: pytest suites, validation scripts, "PASSED" lines in docs/reports.
2. For each, write what it verifies in one sentence (runs without crash? files exist? schema? statistics? causal truth?).
3. Flag every claim in README/docs/reports whose numbers cannot be reproduced from the code and data now on disk (record counts, rates, p-values, KL values, timings).

**Gate 0:** a table "claim, what was actually checked, reproducible yes/no". Add the clarifying sentence the template recommends where a "PASSED" label overstates.

## PHASE 1: Reproducibility and determinism

Run the generator twice with the same seed in separate processes, with different `PYTHONHASHSEED` values, for at least two nodes/countries. Compare every output table: row counts, column checksums, sorted-key checksums.

Look for: Python `hash()` on strings, iteration over `set`, unordered dict/list built from sets, `np.random` global state, parallel workers sharing a seed, wall-clock/time-based values, file listing order, ID generation order.

**Gate 1:** per table, identical or not (with the first differing row). Root cause per difference.

## PHASE 2: Marginal distributions against official references (Clinical and Industry)

For each variable below that exists, compare the generated distribution with a sourced reference (Destatis/Census, RKI, GKV/InEK/KBV, INSEE/Eurostat, Statbel, BFS/FSO, OECD, etc. for the country concerned):

- Age and sex jointly (percentages must add up to 100%; look for fallback or random-fill branches; check sex-by-age, not sex alone).
- Region/canton/Land composition.
- Disease prevalence and incidence by age and sex (diabetes, hypertension, CVD, COPD, depression, cancer, etc.). Use a table "reference vs generated vs ratio" per age band and sex, and state sample size per cell; cells with fewer than about 50 events are flagged as unreliable.
- Utilisation (visits, prescriptions, admissions per person-year), costs and length of stay.
- Mortality by age and sex.
- Industry layer: premium, claim frequency/severity, income, credit/risk scores, tenure, churn/renewal rate, segment mix, per country.
- Currency and units: EUR vs CHF etc. Is there an explicit currency column or a normalised column? Are they ever mixed in one column without a flag?

For each mismatch give the size of the gap, not only "does not match". Do not recalibrate in this phase.

**Gate 2:** reference-vs-generated tables written to `reports/data_audit/`, every mismatch listed as a finding with source.

## PHASE 3: Internal consistency and state (Clinical and Industry)

Check with executable queries:

- **Keys:** primary keys unique; every foreign key resolves (diagnosis, act, prescription, claim, episode, intervention to a person/policy); no orphans after any "switch", "churn" or "death" event.
- **Time:** events inside the observation window; nothing after death or after leaving the fund/insurer; birth before first event; admission before discharge; follow-up years consistent.
- **State memory:** is disease/health status carried from year to year, or redrawn independently each year? Test: a chronic disease that appears must persist or have a recorded remission/cure event. Count "disappear then reappear" patterns.
- **Order of generation:** are downstream tables (prescriptions, costs, billing) generated before the events they depend on (treatments, remission)? A prescription must be consistent with the diagnosis and with any cure.
- **Domain rules:** sex-specific specialties/diagnoses (e.g. gynaecology for male patients), age-specific codes (paediatric codes in adults), valid code systems (ICD-10-GM, EBM, OPS, PZN, ATC exist and match each other), diagnosis-to-drug mapping complete (diseases with a rule but no mapping entry never produce output), dosing vs supply days vs refill dates.
- **Disconnected lists:** specialties, regions, codes defined in several files that do not match each other.
- **Dtypes and missingness:** NaN written into integer columns, Parquet/CSV round-trip changes dtype, injected missingness/outliers rate equals the configured rate, derived fields forced by arithmetic (e.g. tenure = age + 5).
- **Dead configuration:** config keys (node profiles, age tables, region weights, flags) that the generator never reads. Prove it by changing the value and diffing the output.

**Gate 3:** each defect shown on a traced example record plus a count at scale.

## PHASE 4: Ground truth, targets and leakage

- **Causal ground truth (if present):** for every treatment/outcome pair check: (a) is the declared effect size the effect in the data, or is it silently scaled? (b) can an effect be negative or zero, or is harm impossible by construction? (c) is every outcome category reachable? (d) when several treatments fall in the same period, is each effect measured against the state immediately before that treatment, or against a shared snapshot? (e) positivity/overlap: propensity score distribution, share at caps/ceilings, probabilities above 1 hidden by clipping. (f) are effect sizes and risk multipliers sourced? Mark UNSOURCED where not.
- **Sex/age multipliers:** are risk ratios used as ratios, or flattened to a constant bonus?
- **Targets (the 5 ML tasks and any other prediction task):** how each target is defined; is any target computed from features that are also model inputs (e.g. `high_risk` from `claim_risk_score` and `credit_score`)? Is there leakage between tasks? Class balance per node/country.
- **Preprocessing leakage:** imputation/scaling/encoding fitted on all data instead of train only; duplicate persons across train/val/test; time leakage.
- **Tests that assert relationships the generator never creates** (e.g. a correlation threshold that the generating process does not produce): run them, show the real value, decide whether to improve the generator or correct the assertion.

**Gate 4:** per target and per treatment, a one-line verdict with the measured evidence.

## PHASE 5: Federated layer

- **Node/country differences:** do the nodes actually differ in the ways the documentation claims (age structure, region, disease burden, specialty mix, income, treatment mechanism)? Compute per-feature KS statistics and label divergence (JS or TV) between every pair of nodes, and the same for each per-node config key. If all nodes come from the same national distribution, say so: a benchmark needs deliberate, sourced differences.
- **Do the nodes add up?** Do the union of nodes reproduce the national reference?
- **Partitioning:** Dirichlet or other splits: every sample assigned exactly once, no negative or empty splits, floor-rounding losses, small-alpha behaviour. Fuzz it over many seeds and alphas.
- **Centralized oracle copy:** row counts equal the sum of nodes; IDs re-indexed without collisions; original node/ID tracked; no row duplicated or dropped; same schema and dtypes as the nodes.
- **Splits:** train/val/test per task and per node: sizes, stratification, no person in two splits, identical feature set order across clients, same schema everywhere.
- **Local data locality:** each client artefact contains only its own node's rows.
- **FL readiness:** feature alignment across clients (same columns, same encodings, same category vocabularies), missing categories in some clients, scaling statistics computed per client vs globally.

**Gate 5:** a node-by-node difference matrix and the integrity checks of the centralized copy with real counts.

## PHASE 6: Documentation versus code

List every statement in README, METHODOLOGICAL_FRAMEWORK, data dictionary and reports that describes a constant, mechanism, source, calibration target or file that does not exist or behaves differently. Do not rewrite docs yet.

**Gate 6:** `docs/DATA_REVIEW.md` complete with all findings grouped into fix phases:
- Phase A: reproducibility and quick correctness fixes.
- Phase B: state/foundation (memory, generation order).
- Phase C: core mechanism (causal engine, targets).
- Phase D: numbers (calibration against sourced references).
- Phase E: documentation and full re-validation.
- Phase F: federated node differentiation (only after A to E).
Dependencies between tickets must be written explicitly ("depends on DQ-xxx").

---

## PHASE 7: Implement (only after Gate 6 is accepted)

Work through the fix phases A to F in order, one ticket per commit:

1. Write the failing test first (from `tests/data_quality/`), show it fails for the documented reason.
2. Fix the root cause.
3. Show the test passing and the before/after numbers.
4. Regenerate affected data only when needed, check `df -h .` first.
5. Update `docs/DATA_CONTRACT.md` and the data dictionary.
6. Never weaken a test. Never hide a clip, cap or fallback.

**Gate 7:** every ticket closed with evidence, or explicitly deferred with a reason.

## PHASE 8: Full re-validation

Repeat the Phase 1 to 5 checks on freshly generated data. Report reproducibility, reference-vs-generated tables, consistency violations (target: 0), leakage checks, node difference matrix, centralized copy integrity. Re-run the FL smoke experiment (2 rounds, all tasks) on the new data and report real metrics. Interactions between fixes can break each other, so this is mandatory.

**Gate 8:** `docs/DATA_REVIEW.md` updated with final status per ticket; `docs/DATA_CONTRACT.md` final; all tests green three times in a row.

## OUTPUT FORMAT FOR EACH PHASE

1. What I did (files read/created).
2. Commands run and key output (real, trimmed).
3. Findings added (ticket IDs).
4. Gate status PASS/FAIL with evidence.
5. Next phase (stop if a gate failed).

Start now with **Phase 0**. Do not modify any source file until Gate 6 is accepted.