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


__all__ = [
    "SEED",
    "MODE",
    "ROOT",
    "DATA_DIR",
    "REPORTS",
    "ARTIFACTS",
    "FIG_DIR",
    "CLIENTS",
    "CLIENT_LABELS",
    "repo_root",
    "set_seed",
    "optional",
    "env_report",
    "require_data",
    "load_clients",
    "style",
    "save_fig",
    "read_json_if_exists",
    "Timer",
    "short_df",
]
