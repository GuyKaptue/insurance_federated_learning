"""TLS certificate generation and verification (CI-enforced)."""

from __future__ import annotations

import subprocess
import sys
from datetime import UTC, datetime
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


@pytest.mark.integration
def test_tls_rejects_wrong_ca_expiry_and_hostname(tmp_path: Path) -> None:
    root = Path(__file__).resolve().parents[2]
    out = tmp_path / "certs"
    subprocess.run(
        [sys.executable, str(root / "generate_creds.py"), "--output-dir", str(out)],
        check=True,
        capture_output=True,
        text=True,
        cwd=str(root),
    )
    server_cert = out / "server.pem"
    ca_cert = out / "ca.crt"

    subprocess.run(
        [
            "openssl",
            "verify",
            "-verify_hostname",
            "localhost",
            "-CAfile",
            str(ca_cert),
            str(server_cert),
        ],
        check=True,
        capture_output=True,
        text=True,
    )

    rogue_ca = tmp_path / "rogue-ca.crt"
    rogue_key = tmp_path / "rogue-ca.key"
    subprocess.run(
        [
            "openssl",
            "req",
            "-new",
            "-x509",
            "-days",
            "2",
            "-newkey",
            "rsa:2048",
            "-nodes",
            "-keyout",
            str(rogue_key),
            "-out",
            str(rogue_ca),
            "-subj",
            "/CN=Untrusted-Test-CA",
        ],
        check=True,
        capture_output=True,
        text=True,
    )
    wrong_ca = subprocess.run(
        ["openssl", "verify", "-CAfile", str(rogue_ca), str(server_cert)],
        capture_output=True,
        text=True,
    )
    assert wrong_ca.returncode != 0

    wrong_hostname = subprocess.run(
        [
            "openssl",
            "verify",
            "-verify_hostname",
            "wrong.example",
            "-CAfile",
            str(ca_cert),
            str(server_cert),
        ],
        capture_output=True,
        text=True,
    )
    assert wrong_hostname.returncode != 0

    end_date = (
        subprocess.run(
            ["openssl", "x509", "-noout", "-enddate", "-in", str(server_cert)],
            check=True,
            capture_output=True,
            text=True,
        )
        .stdout.strip()
        .split("=", 1)[1]
    )
    expires = datetime.strptime(end_date, "%b %d %H:%M:%S %Y %Z").replace(tzinfo=UTC)
    expired_at = int(expires.timestamp()) + 2
    expired = subprocess.run(
        [
            "openssl",
            "verify",
            "-attime",
            str(expired_at),
            "-CAfile",
            str(ca_cert),
            str(server_cert),
        ],
        capture_output=True,
        text=True,
    )
    assert expired.returncode != 0
    assert "expired" in (expired.stdout + expired.stderr).lower()
