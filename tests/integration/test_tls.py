"""TLS certificate generation and verification (CI-enforced)."""

from __future__ import annotations

import subprocess
import sys
from pathlib import Path

import pytest


@pytest.mark.integration
def test_tls_cert_chain_validates(tmp_path: Path) -> None:
    root = Path(__file__).resolve().parents[2]
    out = tmp_path / "certs"
    subprocess.run(
        [sys.executable, str(root / "generate_creds.py"), "--output-dir", str(out)],
        check=True,
        capture_output=True,
        text=True,
        cwd=str(root),
    )
    assert (out / "ca.crt").exists()
    assert (out / "server.pem").exists()
    assert (out / "client.pem").exists()

    subprocess.run(
        ["openssl", "verify", "-CAfile", str(out / "ca.crt"), str(out / "server.pem")],
        check=True,
        capture_output=True,
        text=True,
    )
    subprocess.run(
        ["openssl", "verify", "-CAfile", str(out / "ca.crt"), str(out / "client.pem")],
        check=True,
        capture_output=True,
        text=True,
    )
