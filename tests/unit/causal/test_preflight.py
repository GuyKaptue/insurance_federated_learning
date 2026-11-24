"""Preflight tests for the causal inference and transportability layer.

Verifies that the optional ``causal`` dependencies (statsmodels, pgmpy)
are installed, importable, and expose the APIs the layer depends on.
These tests fail loudly if ``make install-causal`` has not been run.
"""

from __future__ import annotations

import importlib


def test_statsmodels_importable() -> None:
    """statsmodels must be importable (installed via [causal] extra)."""
    statsmodels = importlib.import_module("statsmodels")
    assert statsmodels is not None


def test_pgmpy_importable() -> None:
    """pgmpy must be importable (installed via [causal] extra)."""
    pgmpy = importlib.import_module("pgmpy")
    assert pgmpy is not None


def test_statsmodels_glm_inference_available() -> None:
    """statsmodels.api must expose Logit and OLS for GLM estimation with inference."""
    from statsmodels.api import OLS, Logit

    assert Logit is not None
    assert OLS is not None


def test_pgmpy_dag_available() -> None:
    """pgmpy.base.DAG must be available for d-separation checks."""
    from pgmpy.base import DAG

    assert DAG is not None
