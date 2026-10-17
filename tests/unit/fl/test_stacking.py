"""``src/insurance_fl/fl/stacking.py`` is EMPTY BY DESIGN (Task 0 structure).

Guards against a silent regression: the module must keep exporting
nothing until Task 3 actually implements NNLS stacking.
"""

from __future__ import annotations

import pytest

pytest.importorskip("flwr")

from insurance_fl.fl import stacking


def test_stacking_is_empty_by_design() -> None:
    assert stacking.__all__ == []
    assert "Task" in (stacking.__doc__ or "")
