from __future__ import annotations

import importlib.util
import json
from pathlib import Path

import pandas as pd


def _load_check_dataset_signal():
    root = Path(__file__).resolve().parents[2]
    mod_path = root / "scripts" / "check_dataset_signal.py"
    spec = importlib.util.spec_from_file_location("check_dataset_signal", mod_path)
    assert spec and spec.loader
    mod = importlib.util.module_from_spec(spec)
    spec.loader.exec_module(mod)
    return mod


def test_logical_hash_changes_when_parquet_value_changes(tmp_path: Path) -> None:
    mod = _load_check_dataset_signal()
    data_dir = tmp_path / "data"
    data_dir.mkdir()
    rel = "federated/client_a/train.parquet"
    p = data_dir / rel
    p.parent.mkdir(parents=True)
    df = pd.DataFrame(
        {
            "age": [30, 40, 50],
            "score": [0.1, 0.2, 0.3],
            "is_high_risk": [0, 1, 0],
        }
    )
    df.to_parquet(p, index=False)
    files, overall = mod.compute_logical_content_hashes(data_dir, [rel])

    manifest = {
        "files_content_sha256": {
            "overall": overall,
            "files": files,
        }
    }
    manifest_path = tmp_path / "manifest.json"
    manifest_path.write_text(json.dumps(manifest))
    ok = mod.check_logical_content_hashes(data_dir, manifest_path=manifest_path, enforce=True)
    assert ok["ok"] is True

    df2 = df.copy()
    df2.loc[1, "score"] = 999.0
    df2.to_parquet(p, index=False)
    bad = mod.check_logical_content_hashes(data_dir, manifest_path=manifest_path, enforce=True)
    assert bad["ok"] is False
    assert bad["mismatches"]
    assert bad["mismatches"][0]["file"] == rel


def test_logical_hash_json_stable_and_value_sensitive(tmp_path: Path) -> None:
    mod = _load_check_dataset_signal()
    p = tmp_path / "a.json"
    p.write_text(json.dumps({"b": 1, "a": 2}))
    p2 = tmp_path / "b.json"
    p2.write_text(json.dumps({"a": 2, "b": 1}))
    assert mod.logical_hash_file(p) == mod.logical_hash_file(p2)
    p.write_text(json.dumps({"a": 2, "b": 3}))
    assert mod.logical_hash_file(p) != mod.logical_hash_file(p2)
