"""DAG re-validation after semi-synthetic truth injection (DQ-I04 / DEP-ORCH-002)."""

from __future__ import annotations

import copy

from pgmpy.base import DAG

from insurance_fl.causal.dag_check import build_pgmpy_dag, load_dag
from insurance_fl.causal.truth import OUTCOME_TRUE, TREATMENT

ADJUSTMENT_SET = ["age", "annual_income", "savings_balance", "credit_score", "debt_to_income_ratio"]


def _post_injection_spec() -> tuple[dict, list[str]]:
    spec = load_dag()
    for node in spec["nodes"]:
        if node["name"] == "claim_risk_score":
            node["name"] = OUTCOME_TRUE
    for edge in spec["edges"]:
        if edge.get("target") == "claim_risk_score":
            edge["target"] = OUTCOME_TRUE
    z = list(spec["adjustment_sets"]["has_claims__claim_risk_score"])
    spec["adjustment_sets"] = {"has_claims__" + OUTCOME_TRUE: list(z)}
    return spec, z


def test_post_injection_dag_unconditionally_connected() -> None:
    spec, _ = _post_injection_spec()
    graph = build_pgmpy_dag(spec)
    assert graph.is_dconnected(TREATMENT, OUTCOME_TRUE)


def test_post_injection_dag_blocked_by_adjustment_set() -> None:
    spec, z = _post_injection_spec()
    graph = build_pgmpy_dag(spec)
    backdoor_graph = copy.deepcopy(graph)
    for source, target in list(backdoor_graph.edges()):
        if source == TREATMENT:
            backdoor_graph.remove_edge(source, target)
    assert not backdoor_graph.is_dconnected(TREATMENT, OUTCOME_TRUE, observed=z)


def test_post_injection_no_open_backdoor_path() -> None:
    spec, z = _post_injection_spec()
    graph = build_pgmpy_dag(spec)
    backdoor_graph = copy.deepcopy(graph)
    for source, target in list(backdoor_graph.edges()):
        if source == TREATMENT:
            backdoor_graph.remove_edge(source, target)
    assert not backdoor_graph.is_dconnected(TREATMENT, OUTCOME_TRUE, observed=z)
    assert isinstance(graph, DAG)
