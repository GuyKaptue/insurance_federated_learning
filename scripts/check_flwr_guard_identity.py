#!/usr/bin/env python3
"""Parse flwr-guard log and assert 4-client identity participation in fit/eval."""

from __future__ import annotations

import re
import sys
from pathlib import Path

EXPECTED_CLIENT_IDS = {
    "germany_allianz",
    "france_axa",
    "belgium_ageas",
    "switzerland_helvetia",
}


def main(path: str = "flwr-guard.log") -> int:
    text = Path(path).read_text(errors="ignore")

    maps: dict[int, tuple[int, str]] = {}
    for m in re.finditer(
        r"CLIENT_MAPPING node_id=(\d+) partition_id=(\d+) num_partitions=(\d+) client_id=([A-Za-z0-9_]+)",
        text,
    ):
        nid = int(m.group(1))
        pid = int(m.group(2))
        npart = int(m.group(3))
        cid = m.group(4)
        if npart != 4:
            print(
                f"ERROR: expected num_partitions=4, got {npart} for node_id={nid}",
                file=sys.stderr,
            )
            return 1
        if nid in maps and maps[nid] != (pid, cid):
            print(
                f"ERROR: conflicting CLIENT_MAPPING for node {nid}: {maps[nid]} vs {(pid, cid)}",
                file=sys.stderr,
            )
            return 1
        maps[nid] = (pid, cid)

    if len(maps) != 4:
        print(f"ERROR: expected 4 distinct node mappings, got {len(maps)}: {maps}", file=sys.stderr)
        return 1
    if {cid for _, cid in maps.values()} != EXPECTED_CLIENT_IDS:
        print(
            f"ERROR: expected client IDs {sorted(EXPECTED_CLIENT_IDS)}, got "
            f"{sorted({cid for _, cid in maps.values()})}",
            file=sys.stderr,
        )
        return 1
    if {pid for pid, _ in maps.values()} != {0, 1, 2, 3}:
        print(
            f"ERROR: expected partition IDs 0-3, got {sorted({pid for pid, _ in maps.values()})}",
            file=sys.stderr,
        )
        return 1

    def phase_ids(phase: str) -> list[str]:
        return re.findall(rf"CLIENT_PHASE_{phase} client_id=([A-Za-z0-9_]+)", text)

    for phase in ("FIT", "EVALUATE"):
        ids = phase_ids(phase)
        if len(ids) != 4 or set(ids) != EXPECTED_CLIENT_IDS:
            print(
                f"ERROR: expected exactly four client IDs once each in {phase.lower()}, got {ids}",
                file=sys.stderr,
            )
            return 1

    print("flwr-guard identity checks: OK (4 mappings, 4 fit, 4 evaluate)")
    return 0


if __name__ == "__main__":
    raise SystemExit(main(sys.argv[1] if len(sys.argv) > 1 else "flwr-guard.log"))
