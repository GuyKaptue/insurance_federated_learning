SESSION BRIEF: Wave 1 / P1 Platform base
Prompt to follow: prompts/10_PLATFORM_AUDIT.md   Phases: 0–2   Branch: chore/full-audit (from integration/audit)
Binding overrides that apply:
- O-1 Branches: Create chore/full-audit from the current tip of integration/audit. Merge into integration/audit only at wave gate, in wave order, with rebase.
- O-2 Delegation: P1 records suspects 4–12 CONFIRMED/REFUTED but does NOT fix them (delegated to B2 DQ-001, DQ-002, DQ-003, DQ-004, DQ-006, DQ-018, DQ-025). P1 owns packaging, imports, README structure.
- O-3 Phase order: P1 Phases 0–2 run first as behaviour-preserving port (same logic, new layout, no statistical fixes). Phases 7–8 deferred to Waves 5–6.
- O-6 FL glue: P1 Phase 2 must expose a generic ClientProtocol in fl/ (round returns dict of named numpy arrays + metadata, not only model weights). Add a test with a dummy aggregate protocol.
- O-7 Logs/IDs: Use ID namespaces E-P1-nnn. Cross-link to DQ IDs when delegated.
- O-9 Freeze: Source files change only in assigned waves.
- O-10 Shared files: P1 owns pyproject.toml, Makefile, Dockerfile, docker-compose.yml, .github/workflows/ci.yml, README.md, tests/test_structure.py.
- O-12 Disk guard: df -h . before every session; stop below 5 GB.

Goal of this session (one sentence):
Port the existing generator and data modules into src/insurance_fl/ behaviour-preserving, fix packaging so python -c "import insurance_fl" works, add ClientProtocol abstraction in fl/, and commit cleanly.

Inputs you must read first:
- docs/PROGRAM_STATUS.md
- prompts/10_PLATFORM_AUDIT.md (Phases 0–2 only)
- docs/PROGRAM_INTAKE.md (repo map, M-numbers, scope)
- Prior work exists in stashes on this repo; run `git stash list` and inspect

You may modify:
- src/insurance_fl/ (create, rewrite, port)
- pyproject.toml (packaging fixes, entry points, deps)
- scripts/ (move into src/insurance_fl/scripts/ if entry points require it)
- tests/ (mirror rule enforcement)
- Makefile (if already present, refine)
- README.md (structure fixes only, not final badges)
- config/ (create federated_config.yaml)

You must not modify:
- prompts/ (orchestrator territory)
- docs/PROGRAM_STATUS.md (update only at session end per ritual)
- data/ generator logic statistics (behaviour-preserving port only; no statistical fixes)
- Any delegated suspect (4–12) — record only

Stop conditions:
- Gate fails (evidence does not match claim)
- Disk below 5 GB
- Dependency request that blocks (file in docs/DEPENDENCY_REQUESTS.md)
- Any conflict with another prompt's ownership

Evidence to return:
1. P1 Gate 0: inventory and gap list in docs/AUDIT_LOG.md
2. P1 Gate 1: error register with every suspect CONFIRMED/REFUTED and evidence quoted
3. P1 Gate 2: `python -c "import insurance_fl"` output
4. P1 Gate 2: all five console scripts respond to `--help` (or list which are missing and why)
5. P1 Gate 2: `tests/test_structure.py` passes
6. Commit hash of the clean commit on chore/full-audit
7. Updated docs/PROGRAM_STATUS.md row for Wave 1

Do not start the next package.
