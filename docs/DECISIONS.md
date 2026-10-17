# Decisions

**ID namespaces:**
- `D-P1-nnn` — Platform audit decisions
- `D-DQ-nnn` — Data quality / cleanup decisions
- `D-CT-nnn` — Causal transport decisions
- `D-ORCH-nnn` — Orchestrator decisions

| ID | Date | Question | Decision | Approved by |
|---|---|---|---|---|
| D-ORCH-001 | 2026-10-16 | `PROMPT_1.md` missing in root | `PROMPT.md` is the canonical P1 prompt. All references to `PROMPT_1.md` are aliased to `PROMPT.md`. | Orchestrator |
| D-ORCH-002 | 2026-10-16 | `DATA_AUDIT_PROMPT.md` missing in root | `Data audit prompt.md` (spaces, lowercase) in root is treated as the canonical P2 prompt. Copy in `prompts/` uses normalized name `20_DATA_AUDIT.md`. | Orchestrator |
| D-ORCH-003 | 2026-10-17 | Two new root specs (`Task_Specification.md`, `Hybrid_Multi-Task_Federated_Learning.md`): canon status? | Both adopted as the **build-program canon** alongside the 4 audit prompts; versioned byte-identically as `prompts/30_HYBRID_PLATFORM_SPEC.md` + `prompts/31_TASK_SPECIFICATION.md`. `Task_Specification.md` is the canonical **build prompt** (tasks 0–12). Root originals stay untracked until Gate 1b passes. Trailing-space filename fixed by rename (content hash unchanged). | Maintainer (via directive "begin with the first layer") |
| D-ORCH-004 | 2026-10-17 | Spec requires `flwr>=1.30`; installed baseline is 1.29.0 | Pin raised `flwr>=1.4.0` → `flwr>=1.29.0,<2.0.0` (pyproject + requirements.txt). **Deviation recorded**: `>=1.30` deferred until a validated upgrade (App API used here — `ServerAppComponents` — exists in 1.29). | Orchestrator |
| D-ORCH-005 | 2026-10-17 | Order of work: audit waves vs build tasks; Gate 1b still FAIL | Adopted **layered order L0–L7** (see `docs/PROGRAM_PLAN.md`). Maintainer directed Layer 1 start → build proceeds in parallel; **Gate 1b provenance remains OPEN** (not waived) — Step 0 paste still required before final PR. | Maintainer |
| D-ORCH-006 | 2026-10-17 | Environment incidents during Layer 1 | (a) `~/.flwr/local-superlink/state.db` (541 MB) held a foreign alembic revision `dfdbe2a8f936` → SuperLink crash; backed up as `state.db.pre-1.29-backup-20261017`, fresh DB created (run OK). (b) exFAT volume hit 100% mid-CI → cleared gitignored tool caches (`.mypy_cache` 1.9 G etc., per `make clean`); `.mypy_cache` rebuild = 1.8 G → Makefile `typecheck` now accepts `MYPY_OPTS` (use `--cache-dir=/tmp/…`). **`venv/` = 26 G broken env is the remaining space lever — maintainer decision pending.** | Orchestrator |

## Pending

| ID | Date | Question | Blocked package |
|---|---|---|---|
| — | — | — | — |
