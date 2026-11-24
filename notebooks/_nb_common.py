"""Shared notebook utilities. Keep this small; reuse insurance_fl wherever possible."""
from __future__ import annotations

import importlib
import json
import os
import platform
import random
import sys
import time
from dataclasses import asdict, dataclass, field
from pathlib import Path
from typing import Any, Callable, Optional

import numpy as np
from IPython.display import Markdown, display


def story(md: str) -> None:
    """Render a reading of the result above. Every number must come from an f-string placeholder."""
    display(Markdown(md.strip()))


def pct(x: float, d: int = 1) -> str:
    return f"{100 * x:.{d}f}%"


def pts(x: float, d: int = 1) -> str:
    """Signed difference of two rates or AUCs, in percentage points (0.01 = 1 point)."""
    return f"{100 * x:+.{d}f} points"


def rel(new: float, old: float, d: int = 1) -> str:
    """Signed relative change of new versus old."""
    return f"{100 * (new - old) / old:+.{d}f}%"


def overlaps(lo1: float, hi1: float, lo2: float, hi2: float) -> bool:
    return not (hi1 < lo2 or hi2 < lo1)


SEED = 42
# Light mode by default: set NB_MODE=full for the expensive version.
MODE = os.environ.get("NB_MODE", "light").lower()
assert MODE in {"light", "full"}, "NB_MODE must be 'light' or 'full'"


def repo_root(start: Optional[Path] = None) -> Path:
    """Walk upward until pyproject.toml is found; works from any cwd."""
    p = (start or Path.cwd()).resolve()
    for cand in [p, *p.parents]:
        if (cand / "pyproject.toml").exists():
            return cand
    raise FileNotFoundError("pyproject.toml not found above " + str(p))


ROOT = repo_root()
for extra in (ROOT / "src", ROOT / "notebooks"):
    if extra.exists() and str(extra) not in sys.path:
        sys.path.insert(0, str(extra))

DATA_DIR = ROOT / "data" / "processed" / "european_insurance_federated"
REPORTS = ROOT / "reports"
ARTIFACTS = ROOT / "artifacts"
FIG_DIR = ROOT / "reports" / "figures"  # created lazily
CLIENTS = ["germany_allianz", "france_axa", "belgium_ageas", "switzerland_helvetia"]
CLIENT_LABELS = {
    "germany_allianz": "Germany",
    "france_axa": "France",
    "belgium_ageas": "Belgium",
    "switzerland_helvetia": "Switzerland",
}


def set_seed(seed: int = SEED) -> np.random.Generator:
    random.seed(seed)
    np.random.seed(seed)
    try:
        import torch

        torch.manual_seed(seed)
        if torch.cuda.is_available():
            torch.cuda.manual_seed_all(seed)
    except Exception:
        pass
    return np.random.default_rng(seed)


def optional(module: str) -> Optional[Any]:
    """Import an optional dependency or return None (caller must handle None visibly)."""
    try:
        return importlib.import_module(module)
    except Exception as e:
        print(f"[optional dependency missing] {module} -> {e}; related cells are skipped")
        return None


def env_report() -> dict:
    info = {
        "python": sys.version.split()[0],
        "executable": sys.executable,
        "platform": platform.platform(),
        "mode": MODE,
        "seed": SEED,
        "cwd": str(Path.cwd()),
        "root": str(ROOT),
    }
    for m in [
        "numpy",
        "pandas",
        "scipy",
        "sklearn",
        "matplotlib",
        "plotly",
        "seaborn",
        "torch",
        "flwr",
        "dp_accounting",
        "networkx",
        "statsmodels",
        "pgmpy",
        "cryptography",
    ]:
        try:
            info[m] = importlib.import_module(m).__version__
        except Exception:
            info[m] = None
    # Machine-readable marker for validators (notebook cells may also display return value)
    try:
        print("NB_ENV_JSON:" + json.dumps(info, sort_keys=True))
    except Exception:
        pass
    return info


def require_data() -> Path:
    """Clear failure with the exact fix instead of a bare AssertionError."""
    if not DATA_DIR.exists():
        raise FileNotFoundError(
            f"{DATA_DIR} missing. Generate it with:\n"
            "  insurance-fl-generate --samples 5000 --output data/processed"
        )
    return DATA_DIR


def load_clients(
    split: str = "train",
    n_max: Optional[int] = None,
    light_default: int = 2000,
):
    import pandas as pd

    root = require_data() / "federated"
    cap = n_max
    if cap is None and MODE == "light":
        cap = light_default
    out = {}
    for c in CLIENTS:
        path = root / c / f"{split}.parquet"
        if not path.exists():
            raise FileNotFoundError(f"Missing partition: {path}")
        df = pd.read_parquet(path)
        if cap is not None and len(df) > cap:
            df = df.sample(cap, random_state=SEED).reset_index(drop=True)
        else:
            df = df.reset_index(drop=True)
        out[c] = df
    return out


def style():
    import matplotlib as mpl

    mpl.rcParams.update(
        {
            "figure.dpi": 110,
            "savefig.dpi": 150,
            "figure.figsize": (7.5, 4.5),
            "axes.spines.top": False,
            "axes.spines.right": False,
            "axes.grid": True,
            "grid.alpha": 0.25,
            "axes.titlesize": 12,
            "axes.labelsize": 10,
            "legend.frameon": False,
            "axes.prop_cycle": mpl.cycler(
                color=["#2a6f97", "#d1495b", "#edae49", "#00798c", "#6a4c93", "#66a182"]
            ),
        }
    )


def save_fig(fig, name: str):
    import matplotlib.pyplot as plt

    FIG_DIR.mkdir(parents=True, exist_ok=True)
    path = FIG_DIR / f"{name}.png"
    fig.savefig(path, bbox_inches="tight", dpi=150)
    print(f"[saved figure] {path}")
    return path


def read_json_if_exists(path: Path):
    path = Path(path)
    if not path.exists():
        print(f"[artifact missing] {path}")
        return None
    with open(path) as f:
        return json.load(f)


class Timer:
    def __enter__(self):
        self.t = time.perf_counter()
        return self

    def __exit__(self, *a):
        self.s = time.perf_counter() - self.t


def short_df(df, n=5):
    return df.head(n)


MANIFEST_PATH = ROOT / "docs" / "evidence" / "dataset_manifest.json"


class StaleArtifactError(RuntimeError):
    """Artifact was computed on a different dataset version or config."""


def provenance_fields(manifest_path: Optional[Path] = None) -> dict:
    mp = Path(manifest_path) if manifest_path else MANIFEST_PATH
    if not mp.is_absolute():
        mp = ROOT / mp
    m = json.loads(mp.read_text())
    return {
        "dataset_version": m.get("dataset_version"),
        "config_hash": m.get("config_hash"),
    }


def load_artifact(path, manifest_path: Optional[Path] = None) -> dict:
    """Load a JSON artifact and reject it if dataset provenance does not match."""
    p = Path(path)
    if not p.is_absolute():
        p = ROOT / p
    d = json.loads(p.read_text())
    mp = Path(manifest_path) if manifest_path else MANIFEST_PATH
    if not mp.is_absolute():
        mp = ROOT / mp
    if not mp.exists():
        raise FileNotFoundError(f"Dataset manifest missing: {mp}")
    m = json.loads(mp.read_text())
    for k in ("dataset_version", "config_hash"):
        if d.get(k) != m.get(k):
            raise StaleArtifactError(
                f"{p}: {k}={d.get(k)!r} but dataset has {m.get(k)!r}"
            )
    return d  # type: ignore


__all__ = [
    "SEED",
    "MODE",
    "ROOT",
    "DATA_DIR",
    "REPORTS",
    "ARTIFACTS",
    "FIG_DIR",
    "MANIFEST_PATH",
    "CLIENTS",
    "CLIENT_LABELS",
    "repo_root",
    "set_seed",
    "story",
    "pct",
    "pts",
    "rel",
    "overlaps",
]

# Unit tests for the helper functions


import unittest


class TestPct(unittest.TestCase):
    def test_basic(self):
        self.assertEqual(pct(0.05), "5.0%")
        self.assertEqual(pct(0.1234, 2), "12.34%")
        self.assertEqual(pct(1.0), "100.0%")
        self.assertEqual(pct(0.0), "0.0%")

    def test_rounding(self):
        self.assertEqual(pct(1 / 3, 2), "33.33%")


class TestPts(unittest.TestCase):
    def test_positive(self):
        self.assertEqual(pts(0.05), "+5.0 points")
        self.assertEqual(pts(-0.03), "-3.0 points")

    def test_zero(self):
        self.assertEqual(pts(0), "+0.0 points")


class TestRel(unittest.TestCase):
    def test_positive(self):
        self.assertEqual(rel(110, 100), "+10.0%")
        self.assertEqual(rel(105, 100), "+5.0%")


class TestOverlaps(unittest.TestCase):
    def test_overlap(self):
        self.assertTrue(overlaps(0.0, 0.5, 0.2, 0.8))
        self.assertTrue(overlaps(0.0, 1.0, 0.5, 0.5))

    def test_no_overlap(self):
        self.assertFalse(overlaps(0.0, 0.5, 0.6, 1.0))
        self.assertFalse(overlaps(0.0, 0.3, 0.5, 1.0))

    def test_adjacent(self):
        # intervals that just touch: [0, 0.5] and [0.5, 1.0] - overlap at boundary
        # Our function: not (0.5 < 0.5 or 1.0 < 0.0) = not (False or False) = True
        self.assertTrue(overlaps(0.0, 0.5, 0.5, 1.0))


if __name__ == "__main__":
    unittest.main()
