"""TLS certificate generation and verification (CI-enforced)."""

from __future__ import annotations

import subprocess
from pathlib import Path

import pytest


@pytest.mark.integration
def test_tls_cert_chain_validates(tmp_path: Path) -> None:
    out = tmp_path / "certs"
    subprocess.run(
        ["python", "generate_creds.py", "--output-dir", str(out)],
        check=True,
        capture_output=True,
        text=True,
    )
    assert (out / "ca.crt").exists()
    assert (out / "server.pem").exists()
    assert (out / "client.pem").exists()

    # Verify server cert against CA
    subprocess.run(
        ["openssl", "verify", "-CAfile", str(out / "ca.crt"), str(out / "server.pem")],
        check=True,
        capture_output=True,
        text=True,
    )
    # Verify client cert against CA
    subprocess.run(
        ["openssl", "verify", "-CAfile", str(out / "ca.crt"), str(out / "client.pem")],
        check=True,
        capture_output=True,
        text=True,
    )
