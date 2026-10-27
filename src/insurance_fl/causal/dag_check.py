"""DAG validation for the causal inference layer.

Parses docs/causal/dag.yaml, checks every edge node exists, and checks
every adjustment set is a d-separator using pgmpy.base.DAG.is_dconnected.
"""

from __future__ import annotations

from pathlib import Path
from typing import Any

import yaml
from pgmpy.base import DAG


def load_dag(dag_path: Path | str = Path("docs/causal/dag.yaml")) -> dict[str, Any]:
    """Load and validate the DAG YAML file."""
    dag_path = Path(dag_path)
    if not dag_path.exists():
        raise FileNotFoundError(f"DAG file not found: {dag_path}")
    with dag_path.open() as f:
        spec = yaml.safe_load(f)
    return spec  # type: ignore[no-any-return]


def build_pgmpy_dag(spec: dict[str, Any]) -> DAG:
    """Build a pgmpy DAG from the spec's edges."""
    nodes = [n["name"] for n in spec["nodes"]]
    edges = [(e["source"], e["target"]) for e in spec["edges"]]
    g = DAG()
    g.add_nodes_from(nodes)
    g.add_edges_from(edges)
    return g


def validate_edge_nodes(spec: dict[str, Any]) -> list[str]:
    """Return a list of error messages for edges referencing unknown nodes."""
    node_names = {n["name"] for n in spec["nodes"]}
    errors: list[str] = []
    for e in spec["edges"]:
        src, tgt = e["source"], e["target"]
        if src not in node_names:
            errors.append(f"edge ({src} -> {tgt}): source node '{src}' not in nodes")
        if tgt not in node_names:
            errors.append(f"edge ({src} -> {tgt}): target node '{tgt}' not in nodes")
    return errors


def validate_adjustment_sets(spec: dict[str, Any], g: DAG) -> list[str]:
    """Return a list of error messages for adjustment sets that are not d-separators.

    For each estimand (treatment → outcome), every backdoor path must be blocked
    by the adjustment set. A backdoor path starts with an edge INTO the treatment.
    We check d-separation: treatment and outcome are d-separated given Z when all
    backdoor paths are blocked. Equivalently, there is no active trail from
    treatment to outcome through a backdoor edge after conditioning on Z.
    """
    errors: list[str] = []
    adjustment_sets = spec.get("adjustment_sets", {})

    # Parse estimand name from key: "treatment__outcome"
    for estimand_name, z_list in adjustment_sets.items():
        if "__" not in estimand_name:
            errors.append(
                f"adjustment set key '{estimand_name}' not in 'treatment__outcome' format"
            )
            continue
        treatment, outcome = estimand_name.split("__", 1)
        z = set(z_list)

        # Validate that all adjustment nodes exist
        node_names = {n["name"] for n in spec["nodes"]}
        for zn in z:
            if zn not in node_names:
                errors.append(f"estimand '{estimand_name}': adjustment node '{zn}' not in nodes")

        # Check d-separation: treatment and outcome should be d-separated given Z
        # via all backdoor paths. We use the full d-separation check: if treatment
        # and outcome are d-connected given Z through any path that starts with
        # an edge into treatment, the adjustment set is insufficient.
        #
        # Simpler and sound: treatment and outcome must be d-separated given Z
        # when we remove all direct outgoing edges from treatment (front-door).
        # But the pragmatic check used here: is_dconnected(treatment, outcome, Z)
        # should be False for backdoor paths only. Since pgmpy checks ALL paths
        # including the causal one, we instead verify that every backdoor path
        # (starting with <- into treatment) is blocked.
        #
        # We implement this by checking: for every neighbor U of treatment where
        # the edge is U -> treatment (backdoor start), U and outcome should be
        # d-separated given Z (conditioning on Z blocks the backdoor).
        # Check d-separation: remove all outgoing edges from treatment (these are
        # the causal/front-door paths). Any remaining active trail from T to Y
        # given Z is a backdoor path that must be blocked.
        import copy

        g_no_out = copy.deepcopy(g)
        outgoing = [e for e in spec["edges"] if e["source"] == treatment]
        for src, tgt in [(e["source"], e["target"]) for e in outgoing]:
            if g_no_out.has_edge(src, tgt):
                g_no_out.remove_edge(src, tgt)

        if g_no_out.is_dconnected(treatment, outcome, list(z)):
            errors.append(
                f"estimand '{estimand_name}': backdoor path from {treatment} to "
                f"{outcome} not blocked by Z={sorted(z)} after removing "
                f"front-door edges from {treatment}"
            )

    return errors


def validate_dag(spec: dict[str, Any] | None = None) -> list[str]:
    """Run all DAG validations. Returns list of error messages (empty = valid)."""
    if spec is None:
        spec = load_dag()
    errors: list[str] = []
    errors += validate_edge_nodes(spec)
    g = build_pgmpy_dag(spec)
    errors += validate_adjustment_sets(spec, g)
    return errors


def main() -> int:
    spec = load_dag()
    errors = validate_dag(spec)
    if errors:
        print("DAG validation FAILED:")
        for err in errors:
            print(f"  - {err}")
        return 1
    print(
        f"DAG validation PASSED: {len(spec['nodes'])} nodes, "
        f"{len(spec['edges'])} edges, "
        f"{len(spec.get('adjustment_sets', {}))} adjustment set(s)"
    )
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
