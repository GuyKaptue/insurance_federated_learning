from __future__ import annotations

from pathlib import Path

from scripts.validate_notebooks import _notebook_hashes, _notebooks_unchanged


def test_notebook_hash_guard_passes_when_sources_are_unchanged(tmp_path: Path) -> None:
    notebook = tmp_path / "example.ipynb"
    notebook.write_text('{"cells": []}', encoding="utf-8")
    before = _notebook_hashes([notebook])

    assert _notebooks_unchanged(before, [notebook])


def test_notebook_hash_guard_fails_when_validation_changes_source(tmp_path: Path) -> None:
    notebook = tmp_path / "example.ipynb"
    notebook.write_text('{"cells": []}', encoding="utf-8")
    before = _notebook_hashes([notebook])

    notebook.write_text('{"cells": [], "metadata": {"changed": true}}', encoding="utf-8")

    assert not _notebooks_unchanged(before, [notebook])
