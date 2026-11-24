from __future__ import annotations

import json
import sys
from pathlib import Path

import pytest

ROOT = Path(__file__).resolve().parents[2]
NB_DIR = ROOT / "notebooks"
if str(NB_DIR) not in sys.path:
    sys.path.insert(0, str(NB_DIR))

import _nb_common as nb  # noqa: E402


def _write_manifest(path: Path, version: str = "v2", config: str = "abc123") -> Path:
    mp = path / "manifest.json"
    mp.write_text(json.dumps({"dataset_version": version, "config_hash": config}))
    return mp


def test_load_artifact_rejects_stale_dataset_version(tmp_path: Path) -> None:
    mp = _write_manifest(tmp_path, "v2", "cfg")
    art = tmp_path / "a.json"
    art.write_text(json.dumps({"dataset_version": "v1", "config_hash": "cfg", "x": 1}))
    with pytest.raises(nb.StaleArtifactError) as ei:
        nb.load_artifact(art, manifest_path=mp)
    assert "dataset_version" in str(ei.value)


def test_load_artifact_rejects_stale_config_hash(tmp_path: Path) -> None:
    mp = _write_manifest(tmp_path, "v2", "cfg-new")
    art = tmp_path / "a.json"
    art.write_text(json.dumps({"dataset_version": "v2", "config_hash": "cfg-old"}))
    with pytest.raises(nb.StaleArtifactError) as ei:
        nb.load_artifact(art, manifest_path=mp)
    assert "config_hash" in str(ei.value)


def test_load_artifact_accepts_current_provenance(tmp_path: Path) -> None:
    mp = _write_manifest(tmp_path, "v2", "cfg")
    art = tmp_path / "a.json"
    payload = {"dataset_version": "v2", "config_hash": "cfg", "value": 42}
    art.write_text(json.dumps(payload))
    out = nb.load_artifact(art, manifest_path=mp)
    assert out["value"] == 42


def test_provenance_fields_reads_manifest(tmp_path: Path) -> None:
    mp = _write_manifest(tmp_path, "v9", "h9")
    p = nb.provenance_fields(manifest_path=mp)
    assert p == {"dataset_version": "v9", "config_hash": "h9"}
