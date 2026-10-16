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
| **P2** | `prompts/20_DATA_AUDIT.md` (root `DATA_AUDIT_PROMPT.md` never existed) | Data audit of synthetic data (clinical, industry, federated) | 0–8 | G0–G8 | `docs/DATA_REVIEW.md`, `docs/DATA_CONTRACT.md`, `tests/data_quality/`, `reports/data_audit/` | Evidence over claims; read-only review until Gate 6; trace real examples; every reference needs a source; branch `chore/data-audit` |
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
| **M21** | **CONFIRMED** | The four root prompt files `PROMPT_1.md`, `PROMPT.md`, `DATA_AUDIT_PROMPT.md`, `CLEAN_SNYNTHETIC_PROMPT.md`, `CA_TR_PROMPT.md` **do not exist anywhere** in this repository. Verified with `find` and `git ls-tree`: root holds only `README.md` and `European_Insurance_Federated_Learning_Project.md`. | All four prompts execute from the `prompts/` copies (`10_PLATFORM_AUDIT.md`, `20_DATA_AUDIT.md`, `21_DATA_CLEANUP_BACKLOG.md`, `30_CAUSAL_TRANSPORT.md`) per O1; no original root copy exists to preserve. |
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
| `PROMPT_1.md` missing | No — canonical prompt source is `prompts/10_PLATFORM_AUDIT.md` | **D-ORCH-003**: root `PROMPT_1.md`/`PROMPT.md` never existed as true originals; P1 prompt executed from the `prompts/` copy. D-ORCH-001 retired. |
| `DATA_AUDIT_PROMPT.md` missing | No — canonical prompt source is `prompts/20_DATA_AUDIT.md` | **D-ORCH-003** (both): root `DATA_AUDIT_PROMPT.md` was never a true original; P2 prompt executed from the `prompts/` copy. D-ORCH-002 retired. |
| Prior agent left uncommitted P1 work | No — but must be resolved before Wave 1 gate | Recorded in `PROGRAM_STATUS.md` as risk; commit or stash required before `integration/audit` merge |

## 5.1 Repo-map evidence (A5.1, real commands) — layers this repo generates
| Layer | Package / Path | Generates? | Evidence |
|---|---|---|---|
| **Industry** | `src/insurance_fl/data/generator.py` (EuropeanInsuranceFederatedDataset) | **YES** | `git ls-files` tracks it; `generator.sh` calls it; `data/processed/` holds generated Parquet |
| **Federated partitioning** | `src/insurance_fl/data/partition.py`, `preprocess.py` | **YES** | tracked in git; `data/processed/european_insurance_federated/federated/{germany_allianz,france_axa,belgium_ageas,switzerland_helvetia}/` produced |
| **Clinical GKV generator** | `disease_trajectory.py`, `pharmacy_generator.py`, etc. | **NO** | `find` returns no matches; `git ls-files` tracks none; Medicare-style GKV modules absent |
| **Causal layer (known truth)** | — | **NOT YET BUILT** | no `causal/`, `transport/`, `fl/causal_*` in git | 

## 5.2 GAP G1 — industry causal truth is unowned (A5.3) — OPEN
O-5 links B2 to build treatment assignment, effect mechanism and node differences via DQ-014–016, DQ-028, DQ-029. **If clinical tickets stay OUT** (they do — no GKV generator exists), then **nobody owns the known-truth treatment/outcome for the industry data**, and P3 Phase 6 is blocked (no known-effect columns). Open `DEP-ORCH-002` to bridge; draft industry tickets below.

## 5.3 GAP G2 — clean clinical data targets another programme (A5.4)
The project goal spans clinical, industry and federated. Clinical generation lives outside this repo (no GKV files here), so clinical data quality is not in scope for this programme. Record as **"Program B: same orchestrator, run in the GKV repository"** — do not silently drop clinical requirements.

## 5.4 Draft industry tickets (DQ-I01 — DQ-I04), batch 3b (A5.3)
| Ticket | Scope | Notes |
|---|---|---|
| DQ-I01 | Treatment definition/assignment = logistic score of covariates | P3 Phase 6 needs a known-assigned treatment |
| DQ-I02 | Outcome with known conditional effect + effect modifiers | truth column separate from the five ML targets |
| DQ-I03 | Per-country differences in effect modifiers | sourced or labelled "assumption" |
| DQ-I04 | truth columns separate from the five ML targets (leakage test) | P3 Phase 6 consumes truth only |

## 5.5 Scope confirmation question to maintainer (A5.5) — the only allowed question
**Question:** "This repo generates industry data + federated partition; clinical data is OUT OF SCOPE. Where does the clinical data (and its quality program) live — another repo (Program B)?" Record answer as `D-ORCH-004`.

---

## Gate O0 status: REOPENED → CONDITIONAL (Session 1 closeout)

**Before Session 1:** O0/O1/O2 were marked PASS, but that was incorrect. The root originals documented in the orchestrator prompt (`PROMPT_1.md`, `PROMPT.md`, `DATA_AUDIT_PROMPT.md`, `CLEAN_SNYNTHETIC_PROMPT.md`, `CA_TR_PROMPT.md`) did not exist; `prompts/` copies were created and D-ORCH-001/D-ORCH-002 were erected on that false premise. Per orchestrator O0.2 and the Wave 0 closeout (Session 1), those gates are now **CONDITIONAL** until provenance of the `prompts/` copies against the originals is proven.

**Session 1 provenance findings (real output):**

A2.1 file presence (`ls`): root holds `causal_inference_quickstart.ipynb`, `transportability_quickstart.ipynb`, and the six `prompts/` copies (`00_ORCHESTRATOR.md`, `10_PLATFORM_AUDIT.md`, `20_DATA_AUDIT.md`, `21_DATA_CLEANUP_BACKLOG.md`, `30_CAUSAL_TRANSPORT.md`, `PATCHES.md`).

A2.3 `diff` (root original vs `prompts/` copy, byte-level):
- `PROMPT.md` vs `prompts/10_PLATFORM_AUDIT.md` → exit 0 (identical)
- `DATA_AUDIT_PROMPT.md` vs `prompts/20_DATA_AUDIT.md` → exit 0 (identical)
- `CA_TR_PROMPT.md` vs `prompts/30_CAUSAL_TRANSPORT.md` → exit 0 (identical)
- `ORCHESTRATOR_PROMPT.md` vs `prompts/00_ORCHESTRATOR.md` → exit 0 (identical)
- `CLEAN_SNYNTHETIC_PROMPT.md` vs `prompts/21_DATA_CLEANUP_BACKLOG.md` → exit 0 (identical)

sha256 of the eight root originals (this session):

| File | sha256 (table) | sha256 (placed) | Match? |
|---|---|---|---|
| `PROMPT.md` / `PROMPT_1.md` | `826a3da74ff1ff7d7440634115c5fcbbb788e71320d4014ef9f06b87900e722d` | `241a6857156b5cf5c1d09b3d821bc3c64861ee02a0cad857d926e3eddf56dc88` | **NO** |
| `DATA_AUDIT_PROMPT.md` | `0d5b5650aa3b8c7c4f11706b78e081007ac1849286e757f2006632bc7f136cbc` | `638166aa963104edbb2b9877b662ed02c3827d599917fa74690be2e84d917c` | **NO** |
| `ORCHESTRATOR_PROMPT.md` | `5630b90f897061aa39e52208b12428d066886ceb1c4b70077ac317f74b26e31f` | `b259e0c8640d0669871dae36595d5c0a3f3b23c0d951f64a52459f8055e77c07` | **NO** |
| `CA_TR_PROMPT.md` | `21d13721a406798fca76889b09444e6faf0ceff5dc7b0c0c7d0a08e2b9ea21d1` | `b4ceacc0bd3940571b0111f873876860215104eb0cb6e392329d0fce098485e3` | **NO** |
| `CLEAN_SNYNTHETIC_PROMPT.md` | `d45a635a8e25a91efe25be1ad5cecaf31f27c9fab5bfb60d7d5d7a2d6683b96d` | `f7181e2412e46df8a974ca4562fc1febb8632fe103e48407e51e14b65185d14d` | **NO** |
| `causal_inference_quickstart.ipynb` | `2d5e13b46ac9442388ce0e59b2a6cc481383c58d567ce4160951fbbf69c02df7` | `2d5e13b46ac9442388ce0e59b2a6cc481383c58d567ce4160951fbbf69c02df7` | YES |
| `transportability_quickstart.ipynb` | `36816de40729f545e687f152a1ffb72c099b82b8897704e90cd7550dbe01634c` | `36816de40729f545e687f152a1ffb72c099b82b8897704e90cd7550dbe01634c` | YES |

**Conclusion:** The `prompts/` `10_PLATFORM_AUDIT.md` copy is byte-identical to the root `PROMPT.md`/`PROMPT_1.md` I placed — but neither hashes to `826a3da7…`. So the originals are NOT the orchestrator's versioned copies; they are a third content the agent did not have. A2.5 says: the copies were produced by an earlier agent from a source the orchestrator prompt referenced as "uploaded"; the versioned copies carry the orchestrator's `% —` header and are NOT the originals. **The four prompts are therefore CONDITIONAL until the originals are placed and proved identical (sha256 + diff).**

A4: Notebooks now present and sha256-verified (P3 finds them). Prior claims re-verified.

**Next action (Session 1):** O0/O1/O2 → CONDITIONAL; E-ORCH-002 recorded; D-ORCH-001/002 SUPERSEDED; D-ORCH-003 revised; M-number mapping; scope evidence + G1/G2; draft DQ-I01–I04; Gate A.

---

## Session 1b findings (2026-10-16, branch integration/audit) — Gate 1b: FAIL

**B1. Session start ritual — PASS.** `df -h .`: 85 Gi avail (≥5 GB). Branch `integration/audit`, log head `6bca5c3`. Zero-byte `.py` outside venv: only `./models/__init__.py` (untracked, dated Aug 29 — pre-existing stub, not this session's concern). `compileall -q src` errors only on AppleDouble sidecars `src/.___init__.py`, `src/core/.___init__.py` (null bytes) — junk, not source. Real `src/__init__.py` / `src/core/__init__.py` are tracked 0-byte package markers (HEAD blob size 0 — legitimate).

**B2. Who created the root files — AGENT-PLACED, proven with evidence (no guessing).**
- `git log --diff-filter=A -- <six root .md files>` → **empty**: no commit ever added them. They are untracked working-tree files.
- `git log --diff-filter=A -- prompts/` → `c738be7 GuyKaptue 2026-10-16 "docs(orchestrator): intake, plan, status, prompt versions, patches"`: the versioned copies were committed there.
- `stat`: all six root `.md` + both `prompts/` copies share mtime `Oct 16 17:42:24` — the root files were written from the `prompts/` content after commit `c738be7`, by an agent session, not by the maintainer.
- This supersedes the Session 1 sentence "the provenance gap is genuine": the precise result is **the root files are agent-placed copies of the prompts/ content; the true chat originals were never placed in this repo by anyone.**

**B3. Verify against the true originals — FAIL (blocked, maintainer action required).**
- `../prompt_originals/` contains only TWO files, both placed by the agent from `prompts/` in the prior session: `ORCHESTRATOR_PROMPT.md` (22971 b, `b259e0c8…` ≠ `5630b90f…`) and `PROMPT_1.md` (18338 b, `241a6857…` ≠ `826a3da7…`). Both sizes and hashes differ from the reference table (ref: 22791 b / 18203 b). Plus two `._*` AppleDouble sidecars.
- No `DATA_AUDIT_PROMPT.md`, `CLEAN_SNYNTHETIC_PROMPT.md`, `CA_TR_PROMPT.md`, notebooks, or `PROMPT.md` in `../prompt_originals/`.
- B3 replacement, PATCHES.md re-derivation, and M1–M14/M21 re-check are therefore **not executable in this session**: there is nothing true to compare against. Per the brief ("do not continue with files that differ"), B3 stops here.
- Internal consistency re-confirmed this session: root `PROMPT_1.md == prompts/10_PLATFORM_AUDIT.md` (`cmp` silent), root `ORCHESTRATOR_PROMPT.md == prompts/00_ORCHESTRATOR.md`, `diff PROMPT.md PROMPT_1.md` empty (canonical: `PROMPT_1.md`). This proves self-consistency only, not provenance.

**B4. Working tree classification — done, cleanup pending maintainer approval.**
- 29 tracked files: mode-only flips 644→755, `git diff --numstat` 0/0 (content identical) → prior-agent tooling leftover; propose revert mode only.
- `.gitignore`: mode flip + 1 added line `.oxcode-memory/` → KEEP the line (correct), revert mode only.
- `docs/PROGRAM_INTAKE.md`, `docs/PROGRAM_STATUS.md`, `docs/DEPENDENCY_REQUESTS.md`: real Session 1/1b content → KEEP, commit after approval.
- 32 tracked `._*` AppleDouble blobs + untracked `._*` under `tests/**/__pycache__/` + `.DS_Store` (root, `models/`, `data/`, `src/`) + `src/.___init__.py`, `src/core/.___init__.py` → macOS junk; propose `git rm` tracked ones, delete untracked, add `._*`/`.DS_Store` to `.gitignore`.
- Full per-path plan: `docs/PROGRAM_STATUS.md` → "B4 cleanup plan" (7 rows). **Nothing discarded in this session.**

**B5. Records corrected this session:** `PROGRAM_STATUS.md` Gate 1b row = FAIL, E-ORCH-002 extended with B2/B3 evidence + guard, Next action = maintainer steps. This intake section replaces the "genuine gap" sentence with the B2 result above.

**Gate 1b score:** B1 PASS / B2 PASS (agent-placed proven) / B3 FAIL (true originals unavailable) / B4 classification done, execution pending approval / B5 done. **Overall: FAIL. Do not start Session 2.**
