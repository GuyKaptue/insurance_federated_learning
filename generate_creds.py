"""Generate TLS certificates for Flower SuperLink/SuperNode (§2.2).

Usage:
    python generate_creds.py [--output-dir certificates]

Generates: ca.crt, server.pem, server.key, client.pem, client.key
"""

from __future__ import annotations

import argparse
import subprocess
import sys
from pathlib import Path


def generate(output_dir: Path) -> None:
    output_dir.mkdir(parents=True, exist_ok=True)

    # Generate CA key and cert
    subprocess.run(
        ["openssl", "genrsa", "-out", str(output_dir / "ca.key"), "2048"],
        check=True, capture_output=True,
    )
    subprocess.run(
        ["openssl", "req", "-new", "-x509", "-days", "365",
         "-key", str(output_dir / "ca.key"),
         "-out", str(output_dir / "ca.crt"),
         "-subj", "/CN=InsuranceFL-CA"],
        check=True, capture_output=True,
    )

    # Generate server key and CSR
    subprocess.run(
        ["openssl", "genrsa", "-out", str(output_dir / "server.key"), "2048"],
        check=True, capture_output=True,
    )
    subprocess.run(
        ["openssl", "req", "-new",
         "-key", str(output_dir / "server.key"),
         "-out", str(output_dir / "server.csr"),
         "-subj", "/CN=InsuranceFL-Server"],
        check=True, capture_output=True,
    )
    # Create ext file with SAN for localhost
    ext_file = output_dir / "server.ext"
    ext_file.write_text(
        "subjectAltName = IP:127.0.0.1, DNS:localhost\n"
        "extendedKeyUsage = serverAuth\n"
    )
    subprocess.run(
        ["openssl", "x509", "-req", "-days", "365",
         "-in", str(output_dir / "server.csr"),
         "-CA", str(output_dir / "ca.crt"),
         "-CAkey", str(output_dir / "ca.key"),
         "-CAcreateserial",
         "-out", str(output_dir / "server.pem"),
         "-extfile", str(ext_file)],
        check=True, capture_output=True,
    )
    ext_file.unlink(missing_ok=True)

    # Generate client key and CSR
    subprocess.run(
        ["openssl", "genrsa", "-out", str(output_dir / "client.key"), "2048"],
        check=True, capture_output=True,
    )
    subprocess.run(
        ["openssl", "req", "-new",
         "-key", str(output_dir / "client.key"),
         "-out", str(output_dir / "client.csr"),
         "-subj", "/CN=InsuranceFL-Client"],
        check=True, capture_output=True,
    )
    ext_file = output_dir / "client.ext"
    ext_file.write_text(
        "subjectAltName = IP:127.0.0.1, DNS:localhost\n"
        "extendedKeyUsage = clientAuth\n"
    )
    subprocess.run(
        ["openssl", "x509", "-req", "-days", "365",
         "-in", str(output_dir / "client.csr"),
         "-CA", str(output_dir / "ca.crt"),
         "-CAkey", str(output_dir / "ca.key"),
         "-CAcreateserial",
         "-out", str(output_dir / "client.pem"),
         "-extfile", str(ext_file)],
        check=True, capture_output=True,
    )
    ext_file.unlink(missing_ok=True)

    # Clean up CSR and serial files
    for f in ["server.csr", "client.csr", "ca.srl", "server.ext", "client.ext"]:
        p = output_dir / f
        if p.exists():
            p.unlink()

    print(f"Certificates generated in {output_dir}/")
    print("  ca.crt, server.pem, server.key, client.pem, client.key")


def main() -> int:
    parser = argparse.ArgumentParser(description="Generate TLS certificates")
    parser.add_argument("--output-dir", default="certificates", help="Output directory")
    args = parser.parse_args()

    output_dir = Path(args.output_dir)
    try:
        generate(output_dir)
    except FileNotFoundError:
        print("Error: openssl not found. Install OpenSSL first.", file=sys.stderr)
        return 1
    except subprocess.CalledProcessError as e:
        print(f"Error generating certificates: {e}", file=sys.stderr)
        return 1
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
