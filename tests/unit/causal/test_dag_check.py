# Test for causal dag_check module

from insurance_fl.causal.dag_check import load_dag, validate_dag


def test_load_dag():
    spec = load_dag()
    assert spec is not None


def test_validate_dag():
    spec = load_dag()
    errors = validate_dag(spec)
    assert errors == []
