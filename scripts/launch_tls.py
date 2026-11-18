"""TLS-secured Flower launch helper (§2.2).

Generates the appropriate flwr run/launch commands with TLS flags.
Usage:
    python scripts/launch_tls.py --certs-dir certificates --num-supernodes 4
"""

from __future__ import annotations

import argparse
import sys
from pathlib import Path


def build_superlink_cmd(certs_dir: Path) -> list[str]:
    return [
        "flower-superlink",
        "--ssl-ca-certfile",
        str(certs_dir / "ca.crt"),
        "--ssl-certfile",
        str(certs_dir / "server.pem"),
        "--ssl-keyfile",
        str(certs_dir / "server.key"),
    ]


def build_supernode_cmd(certs_dir: Path, superlink: str, partition_id: int) -> list[str]:
    return [
        "flower-supernode",
        "--root-certificates",
        str(certs_dir / "ca.crt"),
        "--superlink",
        superlink,
        f"--node-config=partition-id={partition_id} num-partitions=4",
    ]


def build_run_cmd(certs_dir: Path, superlink: str) -> list[str]:
    return [
        "flwr",
        "run",
        ".",
        superlink,
        "--stream",
        "--root-certificates",
        str(certs_dir / "ca.crt"),
    ]


def validate_certs(certs_dir: Path) -> list[str]:
    errors: list[str] = []
    required = ["ca.crt", "server.pem", "server.key", "client.pem", "client.key"]
    for f in required:
        if not (certs_dir / f).exists():
            errors.append(f"Missing certificate: {certs_dir / f}")
    return errors


def main() -> int:
    parser = argparse.ArgumentParser(description="TLS launch helper")
    parser.add_argument("--certs-dir", default="certificates")
    parser.add_argument("--superlink", default="127.0.0.1:8000")
    parser.add_argument("--num-supernodes", type=int, default=4)
    parser.add_argument("--dry-run", action="store_true")
    args = parser.parse_args()

    certs_dir = Path(args.certs_dir)
    errors = validate_certs(certs_dir)
    if errors:
        for e in errors:
            print(f"ERROR: {e}", file=sys.stderr)
        print("Run `python generate_creds.py` first.", file=sys.stderr)
        return 1

    if args.dry_run:
        print("SuperLink:")
        print("  " + " ".join(build_superlink_cmd(certs_dir)))
        for i in range(args.num_supernodes):
            print(f"SuperNode {i}:")
            print("  " + " ".join(build_supernode_cmd(certs_dir, args.superlink, i)))
        print("Run:")
        print("  " + " ".join(build_run_cmd(certs_dir, args.superlink)))
        return 0

    print("Use the commands above to launch TLS-secured Flower.")
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
