"""Library contract tests for the causal inference layer.

Proves that statsmodels provides real GLM inference (conf_int present,
not just point estimates) and that pgmpy's d-separation correctly
handles the collider case (conditioning on a common effect opens a path).
These are the two operations that hand-rolled implementations get wrong.
"""

from __future__ import annotations

import numpy as np
from pgmpy.base import DAG
from statsmodels.api import Logit


def test_statsmodels_glm_returns_conf_int() -> None:
    """Logit.fit() must return conf_int() — real inference, not just params."""
    rng = np.random.RandomState(42)
    n = 500
    z = rng.normal(size=n)
    p = 1.0 / (1.0 + np.exp(-(0.5 + 1.0 * z)))
    y = (rng.random(n) < p).astype(int)

    import statsmodels.api as sm

    X = sm.add_constant(z)
    model = Logit(y, X)
    result = model.fit(disp=0)

    ci = result.conf_int()
    assert ci is not None
    ci = np.asarray(ci)
    assert ci.shape == (2, 2)
    # The z coefficient's CI should not be degenerate (lower < upper)
    assert float(ci[1, 0]) < float(ci[1, 1])
    # The coefficient on z should be positive (data-generating process)
    assert float(result.params[1]) > 0


def test_pgmpy_dsep_chain() -> None:
    """Chain A -> B -> C: A and C are d-connected given [], d-separated given B."""
    g = DAG([("A", "B"), ("B", "C")])
    assert g.is_dconnected("A", "C", [])  # connected unconditionally (active trail)
    assert not g.is_dconnected("A", "C", ["B"])  # d-separated given B (non-collider conditioned)


def test_pgmpy_dsep_collider() -> None:
    """Collider A -> C <- B: A and B are d-separated given [], d-connected given [C].

    Conditioning on a collider OPENS the path — this is the case
    hand-rolled d-separation checks most commonly get wrong.
    """
    g = DAG([("A", "C"), ("B", "C")])
    assert not g.is_dconnected("A", "B", [])  # d-separated given []
    assert g.is_dconnected("A", "B", ["C"])  # d-connected given [C] (collider opened)
