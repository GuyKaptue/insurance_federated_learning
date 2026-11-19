"""Structure enforcement: every source module must have a mirrored test.

Mirror rule (docs/TEST_MATRIX.md):
``src/insurance_fl/<pkg>/<module>.py``  ->  ``tests/unit/<pkg>/test_<module>.py``

``__init__.py`` files are excluded (package plumbing, not testable units).
This test makes the one-test-per-file rule self-enforcing: adding a source
module without a test FAILS the suite.
"""

from __future__ import annotations

from pathlib import Path

SRC_ROOT = Path(__file__).resolve().parents[1] / "src" / "insurance_fl"
TESTS_ROOT = Path(__file__).resolve().parents[1] / "tests" / "unit"


def _source_modules() -> list[Path]:
    """All non-``__init__`` .py files under src/insurance_fl."""
    return sorted(
        p for p in SRC_ROOT.rglob("*.py") if p.name != "__init__.py" and not p.name.startswith("._")
    )


def _mirrored_test(src_file: Path) -> Path:
    rel = src_file.relative_to(SRC_ROOT)  # e.g. data/generator.py
    pkg = rel.parts[0] if len(rel.parts) > 1 else ""
    stem = src_file.stem
    if pkg:
        return TESTS_ROOT / pkg / f"test_{stem}.py"
    return TESTS_ROOT / f"test_{stem}.py"


def test_source_tree_not_empty() -> None:
    assert _source_modules(), f"No source modules found under {SRC_ROOT}"


def test_every_module_has_mirrored_test() -> None:
    missing = [str(s) for s in _source_modules() if not _mirrored_test(s).exists()]
    assert not missing, (
        "Source modules without mirrored test files "
        "(create tests/unit/<pkg>/test_<module>.py):\n  " + "\n  ".join(missing)
    )


def test_no_orphan_test_modules() -> None:
    """Every unit test file must mirror an existing source module.

    Explicit allowlist covers tests that validate scripts, notebook helpers,
    or cross-module integration points not represented as 1:1 src/insurance_fl
    modules.
    """
    allowed = {
        Path("data/test_signal_and_generator.py"),
        Path("scripts/test_mia_script.py"),
        Path("test_artifact_provenance.py"),
    }
    orphans = []
    for t in sorted(TESTS_ROOT.rglob("test_*.py")):
        if t.name == "__init__.py" or t.name.startswith("._"):
            continue
        pkg_dir = t.parent.relative_to(TESTS_ROOT)
        rel = t.relative_to(TESTS_ROOT)
        if rel in allowed:
            continue
        stem = t.stem[len("test_") :]  # test_loader -> loader
        if len(pkg_dir.parts) == 0:
            src = SRC_ROOT / f"{stem}.py"
        else:
            src = SRC_ROOT / pkg_dir / f"{stem}.py"
        if not src.exists():
            orphans.append(str(t))
    assert not orphans, "Test files without a source module:\n  " + "\n  ".join(orphans)
