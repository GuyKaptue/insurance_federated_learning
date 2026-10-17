"""``src/insurance_fl/fl/privacy.py`` is EMPTY BY DESIGN (Task 0 structure).

Guards against a silent regression: the module must keep exporting
nothing until Tasks 6-9 actually implement the privacy stack.
"""

from __future__ import annotations

import pytest

pytest.importorskip("flwr")

from insurance_fl.fl import privacy


def test_privacy_is_empty_by_design() -> None:
    assert privacy.__all__ == []
    assert "Task 6" in (privacy.__doc__ or "") or "Task" in (privacy.__doc__ or "")
