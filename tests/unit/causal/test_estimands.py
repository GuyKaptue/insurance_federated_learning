# Test for causal estimands module

from insurance_fl.causal.estimands import CausalEstimand


def test_estimand_creation():
    estimand = CausalEstimand.from_dag(
        name="test__outcome",
        treatment="treatment",
        outcome="outcome",
        adjustment_set=(),
        dag_path="docs/causal/dag.yaml",
    )
    assert estimand.name == "test__outcome"
