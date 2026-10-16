% ---
% Version: v1.0 (original from root, unmodified)
% Canonical source: CA_TR_PROMPT.md
% Created by orchestrator O1 on 2026-10-16
% ---

# Master Prompt: Add Causal Inference and Transportability (centralized and federated)

Paste everything below the line into Claude Code from the repository root.
Attach as context: `causal_inference_quickstart.ipynb`, `transportability_quickstart.ipynb`, `README.md`, and (if present) `docs/DATA_REVIEW.md` and `docs/DATA_CLEANUP_TODO.md`.

---

## ROLE

You are a senior engineer in causal inference, federated learning and MLOps. You turn two research notebooks into tested, production-quality package code, step by step, and you prove each step with measured output.

## GOAL

Add two capabilities to the project, each usable **centrally and federated**:

1. **Causal inference:** naive baseline, doubly robust AIPW, cross-fitting, overlap/positivity diagnostics, target-trial-emulation (TTE) protocol checks.
2. **Transportability:** estimate the effect in a **target** population (or site/country) from a **source** population, where only target **covariates** are available, with doubly robust and weighted forms, and overlap diagnostics that tell you when not to trust the result.

The federated parts must work on Flower clients (`insurance_fl.fl`), so raw rows never leave a client; only the aggregates listed below do.

## WHAT THE NOTEBOOKS ESTABLISH (hypotheses to re-verify, not facts to quote)

| Source | Idea | Core formula / design |
| --- | --- | --- |
| causal notebook §1 | Confounding bias of the naive difference in means | `naive = mean(Y\|A=1) - mean(Y\|A=0)` |
| §2 | AIPW | `psi = (m1-m0) + A(Y-m1)/e - (1-A)(Y-m0)/(1-e)`; ATE = mean(psi); SE = sd(psi)/sqrt(n); propensity clipped to [0.01, 0.99] |
| §3 | Cross-fitting | K folds; fit nuisances on K-1 folds, predict on the held-out fold; `clone()` every model per fold and per arm |
| §4 | Monte Carlo | repeat 200 times: bias and 95% CI coverage |
| §5 | Federated exact DR | OLS from aggregates `(X'X, X'y)` per arm; logistic propensity by federated Newton-Raphson (clients send gradient `D'(A-p)` and Hessian `D'WD`); final step clients send only `(n, sum s, sum s^2)`; also local-only + inverse-variance combination |
| §5 notes | Limits | no exact federated equivalent for LightGBM/CatBoost; cross-fitting multiplies communication; positivity only visible per client unless summaries are shared; aggregates of tiny clients are not private |
| §6 | TTE checklist | eligibility, treatment strategies, assignment, time zero, outcome, contrast (ITT vs per-protocol), analysis plan; pitfalls: immortal time, prevalent-user, confounding, censoring |
| transport notebook §2-4 | Transport | `selection_odds = P(target\|X)/P(source\|X)` from a stacked logistic model (quadratic term for Gaussian covariates with different variances). **DR form:** `mean_target(m1(X)-m0(X)) + sum(w*resid)/n_target`. **Weighted-AIPW form:** `average(source psi, weights=w)` |
| §5 | Overlap diagnostic | ESS = `(sum w)^2 / sum w^2`, max weight share, central-98% covariate ranges of source vs target |
| §6 | Monte Carlo | DR form is less sensitive to a wrong selection model than weighted-AIPW; wrong-outcome and wrong-selection rows test this |

The notebooks use one scalar covariate `X`. Real data has many covariates, categorical variables and tree-based learners, so **generalisation is part of the work** (Phase 5).

## NON-NEGOTIABLE RULES

1. **Evidence over claims.** Never state a bias, coverage, ESS or difference unless you ran it. Do not copy numbers from the notebooks or from any document; re-measure them. Quote real commands and real output.
2. **Faithful port first, generalisation second.** First reproduce each notebook function exactly (scalar `X`) and show it matches the notebook's behaviour. Only then generalise.
3. **Seeds and determinism.** Every random operation takes an explicit seed; no `hash()` on strings, no global `np.random`; fold assignment uses a stable hash of the unit ID so all federated clients agree.
4. **State the assumptions in code and docs.** Consistency, exchangeability, positivity, and for transport: exchangeability over the population indicator given X and positivity of selection. The code must expose diagnostics for what can be checked (overlap, ESS, clipping share) and say plainly what cannot be checked (unmeasured confounding).
5. **Refuse unsafe output.** When diagnostics fail configured thresholds, the API returns the estimate with a `reliable=False` flag and reasons. It never silently returns a number.
6. **Privacy.** Federated aggregates are not automatically private. Enforce a minimum cell size, make DP noise optional behind a config flag, and document that formal privacy accounting is not provided unless implemented and verified.
7. **Do not weaken tests** or loosen tolerances without writing why in `docs/DECISIONS.md`.
8. **Disk safety:** run `df -h .` before each phase and before Monte Carlo runs; stop if less than 5 GB is free.
9. **Mirror rule:** every `src/insurance_fl/<pkg>/<module>.py` has `tests/unit/<pkg>/test_<module>.py`; `tests/test_structure.py` must still pass.
10. **Small commits** on branch `feat/causal-transport`; one commit per phase or logical step; conventional commits. Keep a running log in `docs/CAUSAL_TRANSPORT_LOG.md`.
11. All data is synthetic; no secrets.

---

## PHASE 0: Orientation (read-only)

1. Read both notebooks cell by cell. Write a function inventory table: notebook, cell, function, inputs, outputs, formula, what it demonstrates.
2. Read the repo: package layout, `fl/` (client, server, strategies), `data/` (generator, partition, preprocess), config loader, scripts, Makefile, CI workflows, existing tests.
3. Answer from the repo, with evidence, and list gaps:
   - Which columns can serve as **treatment A**, **outcome Y**, **covariates X**, and **unit ID**? (For the insurance data: a business intervention such as a retention offer/policy action vs renewal, claim cost, etc. For the clinical/GKV data: interventions and ITE/ATE ground truth if present.)
   - Does the generator contain a **known true effect** (ground truth)? If not, note it: the notebooks work because the simulation knows the truth.
   - Do the nodes/countries differ in effect modifiers? (Transport needs that.)
4. Run the notebooks' code as scripts (extract cells) and record their real output.

**Gate 0:** inventory table, repo findings and the list of missing ingredients (treatment/outcome/truth/effect modifiers) in the log. Nothing modified.

## PHASE 1: Design and layout

Create the skeleton, with tests mirrored, and decide interfaces **before** coding. Write `docs/CAUSAL_TRANSPORT_DESIGN.md` with the table below (adjust to the repo, but keep the mirror rule):

| Source module | Responsibility | Test module |
| --- | --- | --- |
| `causal/simulate.py` | simulators with known truth (confounded, effect modifier, multi-client) | `tests/unit/causal/test_simulate.py` |
| `causal/nuisance.py` | outcome model, propensity model, clipping, design matrix | `.../test_nuisance.py` |
| `causal/estimators.py` | naive, AIPW, cross-fit AIPW | `.../test_estimators.py` |
| `causal/diagnostics.py` | positivity/overlap, clipping share, covariate balance (SMD), ESS | `.../test_diagnostics.py` |
| `causal/tte.py` | target-trial protocol spec, validator, time-zero alignment | `.../test_tte.py` |
| `causal/federated.py` | federated OLS, federated Newton logistic, aggregate scores, local-only + inverse variance | `.../test_federated.py` |
| `causal/montecarlo.py` | replication harness (bias, sd, RMSE, coverage) | `.../test_montecarlo.py` |
| `transport/selection.py` | selection model and odds (linear/quadratic/flexible) | `tests/unit/transport/test_selection.py` |
| `transport/estimators.py` | `source_ate`, `transported_ate(form="dr"/"weighted_aipw")` | `.../test_estimators.py` |
| `transport/diagnostics.py` | ESS, max weight share, range check, reliability verdict | `.../test_diagnostics.py` |
| `transport/federated.py` | federated selection model, federated transport DR | `.../test_federated.py` |
| `fl/causal_client.py`, `fl/causal_server.py` | Flower glue for the aggregate protocols | `tests/unit/fl/test_causal_client.py`, `.../test_causal_server.py` |
| `scripts/run_causal_experiments.py` | CLI for the experiment grid | `tests/unit/scripts/test_run_causal_experiments.py` |

Interfaces (typed, documented):

```python
@dataclass(frozen=True)
class CausalData:            # one population
    X: np.ndarray            # (n, p) covariates
    A: np.ndarray            # (n,) binary treatment
    Y: np.ndarray            # (n,) outcome
    unit_id: np.ndarray      # (n,) stable IDs for fold assignment

@dataclass(frozen=True)
class EffectEstimate:
    ate: float
    se: float
    ci: tuple[float, float]
    n: int
    diagnostics: dict[str, float]
    reliable: bool
    reasons: list[str]
```

**Gate 1:** design doc committed; empty modules and failing-by-design test stubs exist; `tests/test_structure.py` passes.

## PHASE 2: Causal core (centralized), faithful port

Step 2.1 `simulate.py`
- Port `simulate(n, rng)` and `make_client(...)` exactly. Add `make_population` from the transport notebook.
- Tests: with a fixed seed, shapes, dtypes, same output across two processes; known true ATE constants; empirical effect close to truth for large n (state the tolerance and why).

Step 2.2 `nuisance.py`
- Port clipping (`PS_CLIP = 0.01`) as a config constant, `design(X)` (intercept + covariates), outcome model per arm, propensity model. Always `sklearn.base.clone` before fit.
- Tests: clipping bounds, arm-specific fits do not share state (fit twice, same numbers), no leakage between arms.

Step 2.3 `estimators.py`
- `naive_diff(data)`, `aipw_ate(data, outcome_model, propensity_model, clip)`, `crossfit_aipw(data, outcome_model, propensity_model, n_folds, clip, seed)`.
- Fold assignment: deterministic from `unit_id` (stable hash), not from row order.
- Return `EffectEstimate` with SE = sd(psi, ddof=1)/sqrt(n).
- Tests: (a) algebra test: if `m1 = Y` and `m0 = Y` of the right arm the formula degenerates as expected; (b) recovers the known ATE in a single draw within `k*SE` (state k); (c) naive is biased in the confounded simulator (assert sign and magnitude measured, not copied); (d) cross-fit vs plain AIPW with GradientBoosting runs and returns finite numbers; (e) determinism.

Step 2.4 `diagnostics.py`
- Propensity range, share of clipped scores, ESS of IPW weights, standardised mean differences before/after weighting, `reliable` verdict from thresholds in config.
- Tests: construct a near-positivity-violation dataset and assert `reliable=False` with the right reason.

Step 2.5 `montecarlo.py` and slow tests
- Harness: R replications, returns bias, sd, RMSE, 95% coverage. Mark `@pytest.mark.slow`.
- Run R=200 (n=2000) for naive and AIPW. **Report the real measured bias and coverage.** If coverage is far from 0.95, investigate and document (do not tune until it looks nice).

**Gate 2:** unit tests green; slow Monte Carlo table pasted with real numbers; coverage report for the new modules.

## PHASE 3: Target trial emulation (design layer)

Step 3.1 `tte.py`: a `TargetTrialProtocol` dataclass with the seven protocol elements (eligibility, treatment strategies, assignment, time zero, outcome, causal contrast ITT or per-protocol, analysis plan) plus a `validate()` that returns a list of problems.

Step 3.2 Pitfall checks that can run on data (implement what is checkable, document what is not):
- Immortal time: time zero must equal eligibility and assignment time; flag rows where treatment start is after time zero without alignment.
- Prevalent user: flag treated units with prior treatment before time zero (new-user design).
- Censoring/loss to follow-up: report the share and whether it differs by arm.
- Time-varying confounding: only flagged as "not handled by static AIPW".

Step 3.3 Fill one real protocol for this project in `docs/tte_protocol_<question>.md` (PICO, all seven elements) for the chosen treatment/outcome from Phase 0.

Tests: protocol validation (missing element fails loudly), each pitfall check on a hand-built dataset where the answer is known.

**Gate 3:** protocol file filled; tests green.

## PHASE 4: Federated causal estimation

Implement exactly the aggregate-only protocol of the notebook, generalised to `p` covariates.

Step 4.1 Federated outcome model (one round per arm)
- Client sends `(D_arm' D_arm, D_arm' y_arm)`; server solves the linear system. Use a ridge-stabilised solve and raise a clear error on singular matrices.

Step 4.2 Federated propensity model (Newton-Raphson, up to 50 rounds)
- Client at coefficients `beta` sends gradient `D'(A-p)` and Hessian `D'diag(p(1-p))D`; server sums, solves, updates; stop at `max|step| < tol`.
- Log the number of rounds and communication volume (floats per round).

Step 4.3 Aggregate scores
- Client computes AIPW scores locally and sends only `(n, sum s, sum s^2)`; server gets ATE and SE. Add `sum_clipped` (share of clipped propensity) so positivity can be monitored **per client without sharing rows**.

Step 4.4 Local-only baseline
- Each client runs AIPW locally; server combines by inverse-variance weights. Document that this answers a different question if effects differ across clients.

Step 4.5 Cross-fitting in the federated setting
- Coordinated fold assignment from a stable hash of `unit_id` (all clients use the same function and `K`). Each fold needs its own federated nuisance fit; log the communication multiplication. Mark this as approximate for non-parametric learners and say so in docs.

Step 4.6 Flower glue (`fl/causal_client.py`, `fl/causal_server.py`)
- Map each protocol round to a Flower round (`fit` returns aggregates, not model weights). Support in-process simulation first, then gRPC mode.

Step 4.7 Privacy guards
- Minimum cell size: refuse to send aggregates when a client has fewer than `min_n` rows (config). Optional Gaussian noise on aggregates behind a flag. Document that formal (epsilon, delta) is **not** claimed unless an accountant is implemented and verified.

Tests (the key ones):
- **Federated-exact equals centralized:** on the multi-client simulator, `|ate_fed - ate_central| < tol`. Measure the real difference and set `tol` from it with justification (the notebook reports machine-precision level; verify that yourself).
- Newton converges, rounds logged; singular design raises.
- Aggregates contain only the allowed fields (assert on the message schema, so no raw rows can leak).
- Local-only inverse-variance combination matches a hand computation.
- Hypothesis property test: the result is invariant to the order/partition of rows across clients.

**Gate 4:** the three-method table (centralized, federated-exact, local-only) with measured ATE/SE and the per-client propensity ranges, produced by the code.

## PHASE 5: Transportability

Step 5.1 `transport/selection.py`
- `selection_odds(Xs, Xt, features="linear"|"quadratic"|"spline")`: stacked logistic model of `S` (target=1), returning `p/(1-p)` for source rows. Generalise from scalar `X` to `p` covariates: `quadratic` = linear + squares (+ optional interactions, configurable).
- Standardise covariates before the polynomial expansion; guard against separation (clip, ridge) and report it.

Step 5.2 `transport/estimators.py`
- `source_ate` (AIPW in source, port of the notebook).
- `transported_ate(source, target_X, form, selection_features, outcome)` with the two forms:
  - `dr`: `mean_target(m1-m0) + sum(w*resid)/n_target`.
  - `weighted_aipw`: `average(psi_source, weights=w)`.
- Return `EffectEstimate` plus diagnostics. Bootstrap SE option (resample source and target independently) with a fixed seed; document that the simple SE ignores selection-model uncertainty.

Step 5.3 `transport/diagnostics.py`
- ESS, ESS ratio, max weight share, central-98% covariate ranges, share of target covariate values outside the source range, per covariate. Verdict `reliable` from config thresholds (state the default thresholds as decisions, and justify them in `docs/DECISIONS.md`).

Step 5.4 Failure demonstrations (tests, not prose)
- Poor-overlap simulation (target wider than source): assert diagnostics flag it and the estimate is flagged `reliable=False`.
- "Wrong model" simulations: DR with wrong selection (linear) and DR with wrong outcome vs weighted-AIPW with wrong selection. Assert only what you measured, with Monte Carlo margins.

Step 5.5 Federated transport (`transport/federated.py`)
- The selection model needs covariates from **both** source and target sites. Implement federated logistic regression by Newton rounds where each site contributes gradient and Hessian on the stacked problem with the site indicator as label (rows never move).
- Source sites also contribute DR residual aggregates `sum(w*resid)` and `n`; target sites contribute `sum(m1-m0)` over their covariates and `n_target`. Server assembles the DR estimator.
- Overlap diagnostics federated: each client sends `sum w`, `sum w^2`, `max w`, covariate min/max or quantile sketches (no raw rows).
- Tests: federated transport equals the centralized transport on the same pooled data (measure the gap, set tolerance from it).

**Gate 5:** Monte Carlo table (R=100, as in the notebook) for the six estimator configurations with measured mean, bias, sd, plus the overlap diagnostic printed for one example.

## PHASE 6: Connect to the project data

Step 6.1 Treatment, outcome, covariates
- Use the Phase 0 findings. If the data lacks a treatment/outcome with known truth, add a documented **causal layer** to the generator (config-driven): treatment assignment through a logistic score of covariates, an outcome with a known conditional effect that depends on effect modifiers, per-country differences in the effect-modifier distribution. Every parameter needs a documented reason; sourced values where possible, otherwise labelled "assumption". This links to the data cleanup items on assignment probability, effect attribution, and node differences.
- Keep this layer separate from the five existing ML targets so nothing changes their labels (leakage check test).

Step 6.2 Leakage-safe preprocessing
- Reuse `data/preprocess.py` fitted on training/source data only; the nuisance models and selection model never see the outcome of the target.

Step 6.3 Country scenarios
- Source/target scenarios in `config/causal_experiments.yaml`: for example source = DE+FR+BE, target = CH (covariates only); and each country as target in turn (leave-one-country-out).
- Known truth per country from the generator is used only to **evaluate**, never as an input.

Tests: truth columns never enter the nuisance fit; the transported estimate for a country uses only that country's covariates.

**Gate 6:** one real scenario produces centralized, federated-exact, local-only, source-only and transported estimates, each compared with the known truth, with the overlap verdict.

## PHASE 7: Experiments and reports

Step 7.1 `config/causal_experiments.yaml` grid:
- Estimators: naive, AIPW, cross-fit AIPW (linear and tree-based learner, labelled approximate when federated), federated-exact, local-only inverse-variance, centralized.
- Transport: none, DR, weighted-AIPW; selection features linear/quadratic; outcome model right/wrong (simulation only).
- Scenarios: simulator (scalar), simulator (multivariate), project data scenarios from Phase 6.
- Seeds: at least 3 for project data, 100 or more replications for simulators. Report mean and standard deviation.

Step 7.2 `scripts/run_causal_experiments.py`
- Saves per run: config, seed, estimates, SE, diagnostics, wall time, communication volume under `reports/causal/runs/<run_id>/`.

Step 7.3 `reports/causal/summary.md` (generated from the saved runs only)
- Tables: bias, sd, RMSE, coverage; ESS and overlap verdicts; federated vs centralized gap; communication rounds.
- Plots: estimate vs truth with CI per method; overlap (weights and covariate density); convergence of the Newton rounds.
- Honest interpretation: where transport helps, where overlap is poor, where federated equals centralized and where it cannot (tree learners).

**Gate 7:** `make causal-all` completes; summary generated from real outputs.

## PHASE 8: Tooling, CI, docs

1. **Makefile** targets (self-documenting `##`): `causal-test`, `causal-slow`, `causal-sim`, `causal-all`, `causal-report`.
2. **CI** (GitHub Actions): add a job `causal-smoke` (small simulators, 2 to 3 federated rounds, tiny Monte Carlo); keep the slow Monte Carlo out of the default run.
3. **Docs:** `docs/CAUSAL_TRANSPORT.md` with the assumptions, formulas, the federated protocols (what each message contains), the limits (listed below), and how to read the diagnostics. Update the README with the new capability, the Makefile targets and the results table copied from `reports/causal/summary.md`.
4. **Known limits to state in docs** (verify each still holds in your implementation):
   - No exact federated equivalent for tree-based learners; approximate only.
   - Cross-fitting multiplies communication rounds and needs coordinated folds.
   - Positivity is only visible per client unless summaries are shared.
   - Local-only combination answers a different question when effects differ by client.
   - Aggregates from small clients can reveal information; secure aggregation/DP is optional and not a formal guarantee unless accounted.
   - No sensitivity analysis for unmeasured confounding yet (optional next step: E-value or similar).
   - Transport assumptions (exchangeability given X, selection positivity) are assumed, not tested.

**Gate 8:** `make ci` includes the new tests and passes; docs consistent with the code.

## PHASE 9: Full regression and PR

1. Fresh clone, then: `make install-dev && make ci && make causal-all`.
2. Run the full test suite three times; fix any flakiness by seed/tolerance/ordering, not by deleting tests.
3. Write `docs/CAUSAL_TRANSPORT_REPORT.md`: what was added, measured evidence per gate, deviations from the notebooks and why, remaining risks.
4. Open a PR from `feat/causal-transport` with the report as the description.

---

## DEFINITION OF DONE (each with evidence)

- [ ] Notebook functions ported faithfully and verified against the notebooks' behaviour (real re-run numbers).
- [ ] Multivariate generalisation works; scalar case unchanged.
- [ ] Federated-exact equals centralized within a measured tolerance; aggregates-only schema enforced.
- [ ] Transport DR and weighted forms implemented; overlap diagnostics set `reliable` correctly; poor-overlap case demonstrated.
- [ ] Monte Carlo tables (bias, sd, coverage) measured, not copied.
- [ ] TTE protocol module and a filled protocol for this project's question.
- [ ] Project-data scenarios evaluated against known truth; truth never used as input.
- [ ] Every new module has its mirrored test module; `tests/test_structure.py` green; coverage at or above the repo threshold.
- [ ] `make causal-all`, CI job, docs and README updated; limits documented.

## OUTPUT FORMAT FOR EACH PHASE

1. What I did (files created/changed).
2. Commands run and key output (real, trimmed).
3. Errors found and how they were fixed.
4. Gate status PASS/FAIL with evidence.
5. Next phase (stop if a gate failed).

Start now with **Phase 0**. Do not modify any file until Phase 0 is complete.