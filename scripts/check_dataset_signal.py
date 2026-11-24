#!/usr/bin/env python3
"""Dataset label-signal and provenance/content-hash sanity check.

Computes Bayes-optimal AUC for the canonical `is_high_risk` logit per client,
validates label logic, and verifies platform-independent logical content hashes
against ``docs/evidence/dataset_manifest.json`` (``files_content_sha256``).
``files_sha256`` in the manifest is informational/writer-dependent (parquet
bytes can vary by writer/library version); logical hashes are the integrity gate.
"""

from __future__ import annotations

import argparse
import hashlib
import json
import sys
from pathlib import Path

import numpy as np
import pandas as pd
from sklearn.metrics import roc_auc_score

# Ensure local src is importable when run as a script.
ROOT = Path(__file__).resolve().parents[1]
SRC = ROOT / "src"
if str(SRC) not in sys.path:
    sys.path.insert(0, str(SRC))

from insurance_fl.data.generator import high_risk_logit, label_logic_sha  # noqa: E402

CLIENTS = [
    "germany_allianz",
    "france_axa",
    "belgium_ageas",
    "switzerland_helvetia",
]
SPLITS = ("train", "val", "test", "site")
MANIFEST_PATH = ROOT / "docs" / "evidence" / "dataset_manifest.json"
CANONICAL_DATA_DIR = ROOT / "data" / "processed" / "european_insurance_federated"


def _load_client_all(data_dir: Path, client: str) -> pd.DataFrame:
    base = data_dir / "federated" / client
    dfs = []
    for split in SPLITS:
        p = base / f"{split}.parquet"
        if p.exists():
            dfs.append(pd.read_parquet(p))
    if not dfs:
        raise FileNotFoundError(f"No parquet splits found for {client} under {base}")
    return pd.concat(dfs, ignore_index=True)


def logical_hash_parquet_df(df: pd.DataFrame) -> str:
    """Platform-independent logical hash for tabular content.

    Hashes ordered column names+dtypes and pandas' stable row/value hashing
    (includes index) so data changes are detected regardless of parquet writer.
    """
    h = hashlib.sha256()
    h.update("|".join(f"{c}:{df[c].dtype}" for c in df.columns).encode("utf-8"))
    h.update(pd.util.hash_pandas_object(df, index=True).to_numpy().tobytes())
    return h.hexdigest()


def logical_hash_json_obj(obj: object) -> str:
    b = json.dumps(obj, sort_keys=True, separators=(",", ":")).encode("utf-8")
    return hashlib.sha256(b).hexdigest()


def logical_hash_file(path: Path) -> str:
    if path.suffix == ".parquet":
        return logical_hash_parquet_df(pd.read_parquet(path))
    if path.suffix == ".json":
        return logical_hash_json_obj(json.loads(path.read_text(encoding="utf-8")))
    return hashlib.sha256(path.read_bytes()).hexdigest()


def compute_logical_content_hashes(
    data_dir: Path, rel_paths: list[str]
) -> tuple[dict[str, str], str]:
    files: dict[str, str] = {}
    for rel in sorted(rel_paths):
        p = data_dir / rel
        if not p.exists():
            # Skip files not present in this data dir (e.g. temp-generated test
            # data is not the canonical dataset; content-hash verification only
            # applies to files that exist in both manifest and data dir).
            continue
        files[rel] = logical_hash_file(p)
    overall = hashlib.sha256()
    for rel in sorted(files):
        overall.update(rel.encode("utf-8"))
        overall.update(b"\0")
        overall.update(files[rel].encode("utf-8"))
    return files, overall.hexdigest()


def check_logical_content_hashes(
    data_dir: Path, manifest_path: Path = MANIFEST_PATH, enforce: bool | None = None
) -> dict:
    """Verify logical content hashes of files in ``data_dir`` against manifest.

    ``enforce=True`` forces verification (used by unit tests on temp data).
    ``enforce=False`` always skips. ``enforce=None`` (default) verifies only
    when ``data_dir`` is the canonical dataset dir.
    """
    result: dict = {
        "ok": True,
        "manifest_path": str(manifest_path),
        "error": None,
        "checked_files": 0,
        "mismatches": [],
        "missing_files": [],
        "overall_match": None,
        "expected_overall": None,
        "computed_overall": None,
    }
    # Decide whether to enforce content-hash verification.
    if enforce is None:
        enforce = data_dir.resolve() == CANONICAL_DATA_DIR.resolve()
    if not enforce:
        result.update(
            {
                "ok": True,
                "checked_files": 0,
                "error": "content-hash verification skipped (non-canonical data dir)",
            }
        )
        return result
    if not manifest_path.exists():
        result.update({"ok": False, "error": f"Manifest missing: {manifest_path}"})
        return result
    try:
        manifest = json.loads(manifest_path.read_text(encoding="utf-8"))
    except Exception as e:  # noqa: BLE001
        result.update({"ok": False, "error": f"Could not read manifest {manifest_path}: {e}"})
        return result

    expected_block = manifest.get("files_content_sha256")
    if not isinstance(expected_block, dict):
        result.update(
            {
                "ok": False,
                "error": "Manifest missing files_content_sha256 logical hash block",
            }
        )
        return result
    expected_files = expected_block.get("files")
    if not isinstance(expected_files, dict):
        result.update({"ok": False, "error": "Manifest files_content_sha256.files missing/invalid"})
        return result

    rel_paths = sorted(expected_files.keys())
    try:
        computed_files, computed_overall = compute_logical_content_hashes(data_dir, rel_paths)
    except FileNotFoundError as e:
        result.update({"ok": False, "error": str(e)})
        return result
    except Exception as e:  # noqa: BLE001
        result.update({"ok": False, "error": f"Failed computing logical hashes: {e}"})
        return result

    mismatches = []
    for rel in rel_paths:
        if computed_files.get(rel) != expected_files.get(rel):
            mismatches.append(
                {
                    "file": rel,
                    "expected": expected_files.get(rel),
                    "computed": computed_files.get(rel),
                }
            )
    expected_overall = expected_block.get("overall")
    if computed_files:
        overall_match = (computed_overall == expected_overall) if expected_overall else None
    else:
        overall_match = None
    ok = not mismatches and overall_match is not False
    result.update(
        {
            "ok": ok,
            "checked_files": len(computed_files),
            "mismatches": mismatches,
            "overall_match": overall_match,
            "expected_overall": expected_overall,
            "computed_overall": computed_overall,
            "files_sha256_note": manifest.get("files_sha256_note")
            or "files_sha256 is informational/writer-dependent; files_content_sha256 is the integrity gate",
        }
    )
    if not ok:
        if mismatches:
            result["error"] = f"Logical content hash mismatch for {len(mismatches)} file(s)"
        elif overall_match is False:
            result["error"] = "Logical content overall hash mismatch"
    return result


def check_dataset_signal(
    data_dir: str | Path, threshold: float = 0.70, manifest_path: Path = MANIFEST_PATH
) -> dict:
    data_dir = Path(data_dir)
    # Label-logic staleness guard (must run before trusting Bayes/signal results)
    expected_label_sha = label_logic_sha()
    meta_path = data_dir / "metadata.json"
    stored_label_sha = None
    label_logic_ok = True
    label_logic_error = None
    if not meta_path.exists():
        label_logic_ok = False
        label_logic_error = f"Dataset metadata missing at {meta_path}. Regenerate with: make data"
    else:
        try:
            meta = json.loads(meta_path.read_text())
        except Exception as e:
            label_logic_ok = False
            label_logic_error = f"Could not read dataset metadata {meta_path}: {e}. Regenerate with: make data"
        else:
            stored_label_sha = meta.get("label_logic_sha")
            if stored_label_sha != expected_label_sha:
                label_logic_ok = False
                label_logic_error = (
                    f"Dataset was generated with different label logic "
                    f"(stored={stored_label_sha!r}, expected={expected_label_sha!r}). "
                    "Regenerate with: make data"
                )

    content_hash = check_logical_content_hashes(data_dir, manifest_path=manifest_path)

    results = {}
    ok = label_logic_ok and bool(content_hash.get("ok", False))
    for client in CLIENTS:
        try:
            df = _load_client_all(data_dir, client)
        except FileNotFoundError as e:
            results[client] = {"status": "missing", "error": str(e)}
            ok = False
            continue
        y = df["is_high_risk"].to_numpy()
        s = high_risk_logit(df)
        mask = np.isfinite(s) & np.isfinite(y.astype(float))
        auc = float(roc_auc_score(y[mask], s[mask])) if mask.sum() > 0 else float("nan")
        pos = float(np.mean(y))
        n = int(len(df))
        status = "pass" if np.isfinite(auc) and auc >= threshold else "fail"
        if status != "pass":
            ok = False
        results[client] = {"n": n, "positive_rate": pos, "bayes_auc_logit": auc, "status": status}
    return {
        "data_dir": str(data_dir),
        "threshold": threshold,
        "ok": ok,
        "label_logic_sha_expected": expected_label_sha,
        "label_logic_sha_stored": stored_label_sha,
        "label_logic_ok": label_logic_ok,
        "label_logic_error": label_logic_error,
        "content_hash": content_hash,
        "clients": results,
    }


def main(argv: list[str] | None = None) -> int:
    p = argparse.ArgumentParser(description=__doc__)
    p.add_argument(
        "--data-dir",
        default="data/processed/european_insurance_federated",
        help="Path to federated dataset root",
    )
    p.add_argument("--threshold", type=float, default=0.70)
    p.add_argument(
        "--manifest",
        default=str(MANIFEST_PATH),
        help="Path to dataset manifest with files_content_sha256",
    )
    p.add_argument("--json", action="store_true", help="Print JSON only")
    args = p.parse_args(argv)
    try:
        report = check_dataset_signal(args.data_dir, args.threshold, manifest_path=Path(args.manifest))
    except Exception as e:  # noqa: BLE001
        print(f"ERROR: {e}", file=sys.stderr)
        return 2
    if args.json:
        print(json.dumps(report, indent=2))
    else:
        print(f"Dataset signal check: {report['data_dir']} (threshold AUC >= {report['threshold']})")
        if not report.get("label_logic_ok", True):
            print(f"LABEL_LOGIC: FAIL ({report.get('label_logic_error')})")
        else:
            print(f"LABEL_LOGIC: ok (sha={report.get('label_logic_sha_stored')})")
        ch = report.get("content_hash", {})
        if ch.get("ok"):
            print(
                f"CONTENT_HASH: ok (files={ch.get('checked_files')} overall={ch.get('computed_overall')})"
            )
        else:
            print(f"CONTENT_HASH: FAIL ({ch.get('error')})")
            for mm in ch.get("mismatches", [])[:10]:
                print(f"  - mismatch: {mm.get('file')}")
        for c, r in report["clients"].items():
            if r.get("status") == "missing":
                print(f"  - {c}: MISSING ({r.get('error')})")
            else:
                print(
                    f"  - {c}: bayes_auc={r['bayes_auc_logit']:.4f} pos={r['positive_rate']:.4f} "
                    f"n={r['n']} [{r['status']}]"
                )
        print("RESULT:", "PASS" if report["ok"] else "FAIL")
    return 0 if report["ok"] else 1


if __name__ == "__main__":
    raise SystemExit(main())
