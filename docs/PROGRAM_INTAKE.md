# Program Intake — Orchestrator Phase O0

**Date:** 2026-10-16  
**Orchestrator:** OxCode  
**Repository:** `insurance_federated_learning`  
**Branch at intake:** `chore/full-audit` (uncommitted changes from prior P1 work)  
**Default branch:** `main`  

---

## 1. Prompt file inventory

| File (root) | Canonical name | Purpose | Phases | Gates | Files it creates | Rules it sets |
|---|---|---|---|---|---|---|
| `PROMPT.md` | **P1** (`PROMPT_1.md` missing) | Platform audit: packaging, tests, Makefile, Docker, CI, FL experiments, final report | 0–8 | G0–G8 | `docs/AUDIT_LOG.md`, `docs/TEST_MATRIX.md`, `docs/FINAL_REPORT.md`, `Makefile`, `Dockerfile`, `docker-compose.yml`, `.github/workflows/ci.yml`, `config/experiments.yaml` | Evidence over claims; one phase at a time; small commits on `chore/full-audit`; reproducibility; no secrets |
| `Data audit prompt.md` | **P2** (`DATA_AUDIT_PROMPT.md` missing) | Data audit of synthetic data (clinical, industry, federated) | 0–8 | G0–G8 | `docs/DATA_REVIEW.md`, `docs/DATA_CONTRACT.md`, `tests/data_quality/`, `reports/data_audit/` | Evidence over claims; read-only review until Gate 6; trace real examples; every reference needs a source; branch `chore/data-audit` |
| `CLEAN_SNYNTHETIC_PROMPT.md` | **B2** (backlog, not prompt) | Fix list DQ-001 to DQ-032 for synthetic data | A–F | G7–G8 | Same as P2 plus fixed generator code | One ticket per commit; tests first; never weaken tests; branch `chore/data-audit` |
| `CA_TR_PROMPT.md` | **P3** | Causal inference and transportability, centralized and federated | 0–9 | G0–G8 | `docs/CAUSAL_TRANSPORT_LOG.md`, `docs/CAUSAL_TRANSPORT_DESIGN.md`, `docs/CAUSAL_TRANSPORT.md`, `docs/CAUSAL_TRANSPORT_REPORT.md`, `docs/tte_protocol_*.md`, `reports/causal/summary.md`, `src/insurance_fl/causal/`, `src/insurance_fl/transport/`, `src/insurance_fl/fl/causal_client.py`, `src/insurance_fl/fl/causal_server.py` | Evidence over claims; faithful port first; seeds/determinism; state assumptions; refuse unsafe output; privacy not automatic; mirror rule; branch `feat/causal-transport` |
| `causal_inference_quickstart.ipynb` | — | Notebook: AIPW, cross-fitting, federated exact DR, TTE checklist | 5 sections | — | — | Hypotheses to re-verify, not facts to quote |
| `transportability_quickstart.ipynb` | — | Notebook: selection odds, DR transport, weighted-AIPW, overlap diagnostics, Monte Carlo | 6 sections | — | — | Hypotheses to re-verify, not facts to quote |

---

## 2. File check (O0.2)

| Check | Result | Evidence |
|---|---|---|
| `DATA_AUDIT_PROMPT.md` in root | **MISSING** | `ls *.md` shows `Data audit prompt.md` (spaces, lowercase) instead |
| `PROMPT_1.md` in root | **MISSING** | `ls *.md` shows only `PROMPT.md` |
| `PROMPT_1.md` identical to `PROMPT.md` | **N/A** (file missing) | `PROMPT.md` is the canonical P1 prompt |
| `CLEAN_SNYNTHETIC_PROMPT.md` in root | **PRESENT** | Confirmed by `ls` |
| `CA_TR_PROMPT.md` in root | **PRESENT** | Confirmed by `ls` |

**Decision:** `PROMPT.md` is canonical for P1. `Data audit prompt.md` (with spaces) is the de-facto P2 file and will be treated as such. The name mismatch is recorded as **M16** below.

---

## 3. Repo map (O0.3)

### Environment

| Tool | Version / Status |
|---|---|
| OS | macOS Darwin 25.5.0, arm64 |
| Python (system) | 3.13.10 |
| Python (venv) | 3.11.x (`venv/bin/python`) |
| Docker | 29.4.3, build 055a478 |
| Docker Compose | v5.1.3 |
| Free disk | 90 GB (`/Volumes/Intenso`) |

### Code layers in scope

| Layer | Package / Path | Status | Evidence |
|---|---|---|---|
| **Insurance FL platform** | `src/insurance_fl/` | **IN SCOPE** — exists, partially built | `git ls-files` shows `data/`, `models/`, `config/` subpackages; `tests/unit/` mirrored |
| **Legacy core** | `src/core/`, `src/synthetic/` | **IN SCOPE** — excluded from lint, pending migration | `pyproject.toml` `[tool.ruff] extend-exclude` lists them; still tracked in git |
| **Clinical GKV generator** | `disease_trajectory.py`, `pharmacy_generator.py`, etc. | **OUT OF SCOPE FOR THIS REPO** | `find` returns no matches; no clinical files in `git ls-files` |

### B2 ticket scope verdict (M9)

| Ticket | Layer | In scope? | Reason |
|---|---|---|---|
| DQ-001 | C, I, F | **IN** | Determinism — applies to `src/insurance_fl/data/generator.py` |
| DQ-002 | I | **IN** | Truncated normal — applies to industry generator |
| DQ-003 | F | **IN** | Partition correctness — `src/insurance_fl/data/partition.py` exists |
| DQ-004 | C, I | **IN** (I only) | Quality injection — `src/insurance_fl/data/quality.py` exists; clinical layer absent |
| DQ-005 | C, I, F | **IN** (I, F only) | Dead config — applies to existing configs; no clinical config |
| DQ-006 | I | **IN** | Currency — industry data has `currency` column |
| DQ-007 | C | **OUT** | Clinical code mappings — no clinical generator in repo |
| DQ-008 | C | **OUT** | Sex/age plausibility of specialties — no clinical generator |
| DQ-009 | C | **OUT** | Disease memory — no clinical generator |
| DQ-010 | C | **OUT** | Pipeline order — no clinical generator |
| DQ-011 | C | **OUT** | Prescription realism — no clinical generator |
| DQ-012 | C, I | **IN** (I only) | Switch/churn/death consistency — industry layer only |
| DQ-013 | C, I, F | **IN** (I, F only) | Referential/temporal integrity — industry + federated only |
| DQ-014 | C | **OUT** | Treatment-effect attribution — no clinical generator |
| DQ-015 | C | **OUT** | Hidden effect scaling — no clinical generator |
| DQ-016 | C | **OUT** | Assignment probability — no clinical generator |
| DQ-017 | I | **IN** | Leakage-free targets — `src/insurance_fl/data/generator.py` |
| DQ-018 | I | **IN** | Generative process matches assertions — industry generator |
| DQ-019 | F | **IN** | Preprocessing fit on train only — `src/insurance_fl/data/preprocess.py` |
| DQ-020 | C, I | **IN** (I only) | Age × sex joint distribution — industry only |
| DQ-021 | C | **OUT** | Disease prevalence — no clinical generator |
| DQ-022 | C | **OUT** | Utilisation/cost/LOS — no clinical generator |
| DQ-023 | I | **IN** | Premium/claim/churn/segment per country — industry layer |
| DQ-024 | C | **OUT** | Source treatment-effect numbers — no clinical generator |
| DQ-025 | C, I, F | **IN** (I, F only) | Docs match code — applies to existing docs |
| DQ-026 | C, I, F | **IN** (I, F only) | Full re-validation — industry + federated only |
| DQ-027 | C, I, F | **IN** (I, F only) | Data contract + CI gate — industry + federated only |
| DQ-028 | F | **IN** | Per-node settings — federated layer |
| DQ-029 | F | **IN** | Deliberate node differences — federated layer |
| DQ-030 | F | **IN** | Centralized oracle integrity — federated layer |
| DQ-031 | F | **IN** | Client feature alignment — federated layer |
| DQ-032 | F | **IN** | Non-IID measures — federated layer |

---

## 4. Mistake re-check (M1–M14) + new findings

| # | Status | Evidence (line numbers / command output) | Resolution |
|---|---|---|---|
| M1 | **CONFIRMED** | B2 line 3: "Phase 1 of `DATA_AUDIT_PROMPT.md` marks each one…" — file `DATA_AUDIT_PROMPT.md` does not exist in root. B2 filename `CLEAN_SNYNTHETIC_PROMPT.md` contains typo "SNYNTHETIC". | O0.2 treats `Data audit prompt.md` as P2; O1 renames B2 copy to `prompts/21_DATA_CLEANUP_BACKLOG.md` |
| M2 | **CONFIRMED** | P1 Phase 1 suspects 4–12 (lines 57–67) list seeding, truncated normal, Dirichlet, quality injection, test assertions, segments, currency, fabricated evidence. B2 DQ-001 to DQ-006, DQ-018, DQ-025 cover the same topics. | O-2 delegation table assigns suspects 4–12 to B2; P1 records only |
| M3 | **CONFIRMED** | P1 Phase 7 (lines 143–158) runs "all federated learning" on "native country non-IID" data. B2 DQ-028/029 (lines 52–54) says deliberate differences come only in Phase F, after fixes A–E. | O-3 defers P1 Phase 7 to Wave 5 |
| M4 | **CONFIRMED** | P1 Phase 3 (lines 88–99) sets `--cov-fail-under=85` and enforces test mirror before data/causal modules are rewritten. | O-4 sets baseline coverage now, raises to 85 at Wave 6 |
| M5 | **CONFIRMED** | P3 Phase 6 (lines 216–220) says "add a documented causal layer to the generator if truth is missing". B2 DQ-014 to DQ-016, DQ-028, DQ-029 (lines 29–54) already own treatment assignment, effect mechanism, node differences. | O-5: B2 builds causal layer; P3 consumes it only |
| M6 | **CONFIRMED** | P1 Phase 2 (lines 75–84) designs `fl/client.py`, `fl/server.py` for FedAvg/FedProx **model weights**. P3 Phase 4 (lines 151–176) needs Flower clients returning **aggregates** (numpy arrays), not weights. | O-6: P1 Phase 2 must expose generic `ClientProtocol` |
| M7 | **CONFIRMED** | P1 line 29: branch `chore/full-audit`, no freeze rule. P2 line 37: "do not change code during review (Phases 0 to 6)", branch `chore/data-audit`. P3 line 50: branch `feat/causal-transport`, no explicit freeze. | O-1, O-9: single integration branch `integration/audit`; freeze timeline per wave |
| M8 | **CONFIRMED** | P1 line 33: `docs/AUDIT_LOG.md`. P2 line 41: `docs/DATA_REVIEW.md`. P3 line 50: `docs/CAUSAL_TRANSPORT_LOG.md`. All share `docs/DECISIONS.md` with no ID namespace. | O-7: `docs/PROGRAM_STATUS.md` plus ID namespaces |
| M9 | **CONFIRMED** | B2 DQ-007 (line 10), DQ-008 (line 11), DQ-009–012, DQ-014–016, DQ-020–022, DQ-024 reference clinical files. `find` / `git ls-files` show **no** `disease_trajectory.py`, `pharmacy_generator.py`, or any clinical module. | O0.3 marks clinical tickets OUT OF SCOPE |
| M10 | **CONFIRMED** | P1 Phase 2 edits `pyproject.toml`, `README.md`. P3 Phase 8 edits `Makefile`, CI, README. P2 Phase 7 touches config and docs. | O-10: additive include files; P1 owns shared files |
| M11 | **CONFIRMED** | P1 Phase 7 (line 150) mentions "DP off and on (report epsilon/delta if DP is implemented and accounted correctly)". P3 Phase 4 (line 175) says "Optional Gaussian noise … document that formal (epsilon, delta) is **not** claimed unless an accountant is implemented". | O-11: one privacy config; formal claims only with verified accountant |
| M12 | **CONFIRMED** | P3 needs `scipy`, `scikit-learn` (notebook imports). P2 needs `hypothesis`, `scipy` (Phase 3). P1 `pyproject.toml` already lists `scipy`, `scikit-learn`, `hypothesis` in dev deps. | O-10: P1 owns `pyproject.toml`; others request via `docs/DEPENDENCY_REQUESTS.md` |
| M13 | **CONFIRMED** | P2 Phase 5 (lines 119–128) audits partitioning and centralized copy. P1 Phase 2 rewrites `data/partition.py`. | O-3: P2 runs against post-P1-Phase-2 port |
| M14 | **CONFIRMED** | P1 has no resume procedure. `docs/AUDIT_LOG.md` lines 283–288 document a real disk-full interruption with corrupted files. | Session start ritual in this orchestrator (O-12) |
| **M15** | **CONFIRMED** | `PROMPT_1.md` does not exist in root. Only `PROMPT.md` exists. | `PROMPT.md` is canonical |
| **M16** | **CONFIRMED** | `Data audit prompt.md` (spaces, lowercase) exists instead of `DATA_AUDIT_PROMPT.md` (underscores, uppercase). | Treat `Data audit prompt.md` as canonical P2 for this repo |
| **M17** | **CONFIRMED** | `git status` shows staged + unstaged changes on `chore/full-audit` from prior agent; tree is not clean. Commit hash `b01d23c` is only "first commit". | Prior P1 work must be committed cleanly before Wave 1 gate review |
| **M18** | **CONFIRMED** | `venv/bin/python` is Python 3.11; system Python is 3.13.10. `pyproject.toml` line 21 says `>=3.11,<3.14` (allows 3.13). | No conflict, but venv is 3.11; system 3.13 can also be used |
| **M19** | **CONFIRMED** | `pyproject.toml` lines 91–95 define console scripts pointing to `insurance_fl.scripts.*`, but `scripts/generate_dataset.py` lives at repo root `scripts/`, not `src/insurance_fl/scripts/`. | P1 Phase 2 must create `src/insurance_fl/scripts/` or fix entry points |
| **M20** | **CONFIRMED** | `src/insurance_fl/` already exists (from prior agent) alongside legacy `src/core/` and `src/synthetic/`. `pyproject.toml` line 178 finds `insurance_fl*` under `src/`. | Legacy trees are excluded from lint; migration is partial |

---

## 5. Missing information and decisions

| Item | Blocks execution? | Decision / Action |
|---|---|---|
| Which column is the **treatment** for causal inference? | No — P3 Phase 0 will determine this | Documented as open question for P3 Phase 0 |
| Which column is the **outcome** with known truth? | No — B2 will build causal layer (O-5) | Deferred to B2 DQ-014 to DQ-016 |
| Jurisdictions in scope | No — README and code agree on DE, FR, BE, CH | Confirmed: 4 countries |
| Official reference sources available offline | No — P2 will source or mark UNSOURCED | No blocker |
| `PROMPT_1.md` missing | No — `PROMPT.md` is canonical | **D-ORCH-001**: `PROMPT.md` is the canonical P1 prompt. `PROMPT_1.md` reference in all docs is aliased to `PROMPT.md`. |
| `DATA_AUDIT_PROMPT.md` missing | No — `Data audit prompt.md` is canonical | **D-ORCH-002**: `Data audit prompt.md` (root) is treated as `DATA_AUDIT_PROMPT.md` for all orchestrator purposes. Copy in `prompts/` will use canonical name `20_DATA_AUDIT.md`. |
| Prior agent left uncommitted P1 work | No — but must be resolved before Wave 1 gate | Recorded in `PROGRAM_STATUS.md` as risk; commit or stash required before `integration/audit` merge |

---

## Gate O0 status: PASS

**Evidence:**
- `docs/PROGRAM_INTAKE.md` exists (this file)
- All four prompt files + two notebooks read and inventoried
- M1–M14 each marked CONFIRMED or REFUTED with line-number quotes
- M15–M20 added with evidence
- Scope per repo established: clinical layer OUT OF SCOPE; industry + federated IN SCOPE
- Environment verified: Python 3.13.10 / 3.11 venv, Docker 29.4.3, 90 GB free
- Two decisions recorded: D-ORCH-001, D-ORCH-002

**Next:** O1 — reconcile and version the prompts.
