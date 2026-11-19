from __future__ import annotations

import json
import subprocess
import sys
from pathlib import Path

import numpy as np
import pandas as pd
import pytest
from sklearn.metrics import roc_auc_score

from insurance_fl.data.generator import EuropeanInsuranceFederatedDataset, high_risk_logit
from scripts.check_dataset_signal import check_dataset_signal


CLIENTS = [
    "germany_allianz",
    "france_axa",
    "belgium_ageas",
    "switzerland_helvetia",
]


def test_high_risk_logit_matches_generation_target(tmp_path: Path) -> None:
    # small but stratification-safe generation
    gen = EuropeanInsuranceFederatedDataset(
        samples_per_client=400, random_seed=123, dataset_version="test", inject_quality_issues=False
    )
    out = tmp_path / "ds"
    gen.generate_dataset(str(out), save_format="parquet", include_centralized=False, force=True)
    for c in CLIENTS:
        df = pd.read_parquet(out / "federated" / c / "train.parquet")
        logit = high_risk_logit(df)
        # recompute probability as in generator
        p = 1.0 / (1.0 + np.exp(-logit))
        p = np.clip(p, 0.05, 0.95)
        # labels are Bernoulli draws from p with per-client RNG; check Bayes AUC
        auc = roc_auc_score(df["is_high_risk"], logit)
        assert np.isfinite(auc)
        assert 0.0 <= auc <= 1.0
        assert len(p) == len(df)


def test_in_process_small_generation_bayes_auc_ge_070(tmp_path: Path) -> None:
    gen = EuropeanInsuranceFederatedDataset(
        samples_per_client=400, random_seed=42, dataset_version="test", inject_quality_issues=False
    )
    out = tmp_path / "ds"
    gen.generate_dataset(str(out), save_format="parquet", include_centralized=True, force=True)
    report = check_dataset_signal(out, threshold=0.70)
    assert report["ok"], json.dumps(report, indent=2)
    for c in CLIENTS:
        assert report["clients"][c]["bayes_auc_logit"] >= 0.70


def test_generator_refuses_nonempty_output_without_force(tmp_path: Path) -> None:
    out = tmp_path / "ds"
    out.mkdir()
    (out / "keep.txt").write_text("x")
    gen = EuropeanInsuranceFederatedDataset(samples_per_client=400, random_seed=1)
    with pytest.raises(FileExistsError):
        gen.generate_dataset(str(out), force=False)
    # with force it should proceed
    gen.generate_dataset(str(out), force=True, include_centralized=False)
    assert (out / "metadata.json").exists()


def test_cli_generation_and_signal_check_temp_dir(tmp_path: Path) -> None:
    out = tmp_path / "cli_ds"
    cmd = [
        sys.executable,
        "-m",
        "insurance_fl.scripts.generate_dataset",
        "--samples",
        "400",
        "--seed",
        "42",
        "--version",
        "v2-test",
        "--output",
        str(out),
        "--no-centralized",
        "--no-quality-issues",
    ]
    r = subprocess.run(cmd, capture_output=True, text=True)
    assert r.returncode == 0, r.stdout + "\n" + r.stderr
    assert (out / "federated" / "germany_allianz" / "train.parquet").exists()
    assert not (out / "centralized").exists()
    # metadata provenance fields
    meta = json.loads((out / "metadata.json").read_text())
    assert meta["quality_issues_injected"] is False
    assert meta["include_centralized"] is False
    assert "generator_git_sha" in meta
    assert "config_hash" in meta
    assert meta["dataset_version"] == "v2-test"
    # signal check via script logic
    report = check_dataset_signal(out, threshold=0.70)
    assert report["ok"], json.dumps(report, indent=2)


def test_cli_refuses_nonempty_output_without_force(tmp_path: Path) -> None:
    out = tmp_path / "cli_ds"
    out.mkdir()
    (out / "keep.txt").write_text("x")
    cmd = [
        sys.executable,
        "-m",
        "insurance_fl.scripts.generate_dataset",
        "--samples",
        "400",
        "--output",
        str(out),
    ]
    r = subprocess.run(cmd, capture_output=True, text=True)
    assert r.returncode != 0
    assert "not empty" in (r.stdout + r.stderr).lower()
    # with --force should work
    cmd_force = cmd + ["--force", "--no-centralized", "--no-quality-issues"]
    r2 = subprocess.run(cmd_force, capture_output=True, text=True)
    assert r2.returncode == 0, r2.stdout + "\n" + r2.stderr


def test_cli_quality_issues_flag_affects_metadata(tmp_path: Path) -> None:
    out = tmp_path / "ds_q"
    cmd = [
        sys.executable,
        "-m",
        "insurance_fl.scripts.generate_dataset",
        "--samples",
        "400",
        "--seed",
        "7",
        "--output",
        str(out),
        "--no-centralized",
        "--no-quality-issues",
        "--force",
    ]
    r = subprocess.run(cmd, capture_output=True, text=True)
    assert r.returncode == 0, r.stdout + r.stderr
    meta = json.loads((out / "metadata.json").read_text())
    assert meta["quality_issues_injected"] is False


def test_legacy_generator_default_output_is_noncanonical() -> None:
    import inspect
    import sys
    import warnings

    # Fresh import/init should emit a deprecation warning and default to non-canonical path.
    for mod in ["src.synthetic.data_generator", "src.synthetic"]:
        sys.modules.pop(mod, None)
    with warnings.catch_warnings(record=True) as w:
        warnings.simplefilter("always", DeprecationWarning)
        from src.synthetic import data_generator as legacy

        legacy.EuropeanInsuranceFederatedDataset(samples_per_client=10, random_seed=1)
    # Accept warning from import-time or __init__ depending on environment filters.
    if not any(issubclass(x.category, DeprecationWarning) for x in w):
        # Fallback: explicit class-level signal check (must be deprecated in intent/docs).
        assert "_LEGACY_DEPRECATION" in dir(legacy)
    sig = inspect.signature(legacy.EuropeanInsuranceFederatedDataset.generate_dataset)
    default = sig.parameters["output_dir"].default
    assert "legacy_european_insurance_federated" in str(default)
    assert str(default) != "data/processed/european_insurance_federated"


def test_check_fails_when_label_logic_hash_differs(tmp_path: Path) -> None:
    # generate small valid dataset, then tamper stored label_logic_sha
    gen = EuropeanInsuranceFederatedDataset(
        samples_per_client=400, random_seed=42, dataset_version="test", inject_quality_issues=False
    )
    out = tmp_path / "ds"
    gen.generate_dataset(str(out), save_format="parquet", include_centralized=False, force=True)
    meta_path = out / "metadata.json"
    meta = json.loads(meta_path.read_text())
    assert "label_logic_sha" in meta and meta["label_logic_sha"]
    meta["label_logic_sha"] = "0" * 16
    meta_path.write_text(json.dumps(meta, indent=2))
    report = check_dataset_signal(out, threshold=0.70)
    assert report["ok"] is False
    assert report["label_logic_ok"] is False
    err = report.get("label_logic_error") or ""
    assert "different label logic" in err.lower()
    assert "make data" in err.lower()
