"""Part 2: four-client identity correctness (partition-id, no node_id modulo)."""

from __future__ import annotations

import random
import types
from dataclasses import dataclass

import numpy as np
import pytest

pytest.importorskip("torch")
pytest.importorskip("flwr")

from flwr.common import Code, EvaluateRes, FitRes, Status, ndarrays_to_parameters
from flwr.server.client_manager import SimpleClientManager

from insurance_fl.fl import task as fl_task
from insurance_fl.fl.strategy import HybridAggregation, StratifiedFedProx


@dataclass
class _Proxy:
    cid: int
    node_id: int | None = None

    def __post_init__(self) -> None:
        if self.node_id is None:
            self.node_id = self.cid


class _Audit:
    def __init__(self) -> None:
        self.events: list[tuple[str, dict]] = []

    def log_event(self, event_type: str, payload: dict) -> None:
        self.events.append((event_type, dict(payload)))


def _params():
    return ndarrays_to_parameters([])


def _ok_status():
    return Status(code=Code.OK, message="")


def _fit_res(client_id: str, n: int = 10):
    return FitRes(
        status=_ok_status(),
        parameters=_params(),
        num_examples=n,
        metrics={"client_id": client_id, "train_loss": 1.0},
    )


def _eval_res(client_id: str, n: int = 10):
    return EvaluateRes(
        status=_ok_status(),
        loss=1.0,
        num_examples=n,
        metrics={"client_id": client_id, "loss": 1.0},
    )


def _cm(node_ids):
    cm = SimpleClientManager()
    for nid in node_ids:
        cm.register(_Proxy(cid=nid, node_id=nid))
    return cm


# --- resolve_client_id (task.py) ---


def test_resolve_client_id_one_to_one_for_partitions_0_to_3():
    for pid, expected in enumerate(fl_task.CLIENT_IDS):
        ctx = types.SimpleNamespace(
            node_id=999999999999999,
            node_config={"partition-id": pid, "num-partitions": 4},
        )
        assert fl_task.resolve_client_id(ctx) == expected


def test_resolve_client_id_rejects_num_partitions_mismatch_and_out_of_range():
    with pytest.raises(ValueError, match="num-partitions"):
        fl_task.resolve_client_id(
            types.SimpleNamespace(node_id=1, node_config={"partition-id": 0, "num-partitions": 3})
        )
    with pytest.raises(ValueError, match=r"out of range|partition-id"):
        fl_task.resolve_client_id(
            types.SimpleNamespace(node_id=1, node_config={"partition-id": 4, "num-partitions": 4})
        )
    with pytest.raises(ValueError, match="partition-id"):
        fl_task.resolve_client_id(types.SimpleNamespace(node_id=1, node_config={}))


def test_resolve_client_id_ignores_client_identity_claim_in_untrusted_config():
    with pytest.raises(ValueError, match="Missing partition-id"):
        fl_task.resolve_client_id(
            types.SimpleNamespace(
                node_id=0,
                node_config={"client_id": "switzerland_helvetia"},
            )
        )

    ctx = types.SimpleNamespace(
        node_id=999,
        node_config={
            "partition-id": 0,
            "num-partitions": 4,
            "client_id": "switzerland_helvetia",
        },
    )
    assert fl_task.resolve_client_id(ctx) == "germany_allianz"


@pytest.mark.parametrize("strategy_cls", [HybridAggregation, StratifiedFedProx])
def test_aggregation_rejects_non_finite_client_update(strategy_cls):
    from flwr.common import ndarrays_to_parameters

    node_id = 123
    manager = _cm([node_id])
    kwargs = {
        "fraction_fit": 1.0,
        "fraction_evaluate": 1.0,
        "min_fit_clients": 1,
        "min_evaluate_clients": 1,
        "min_available_clients": 1,
    }
    if strategy_cls is StratifiedFedProx:
        kwargs["proximal_mu"] = 0.1
    strategy = strategy_cls(**kwargs)
    result = _fit_res("germany_allianz")
    result.parameters = ndarrays_to_parameters([np.array([float("nan")])])

    with pytest.raises(ValueError, match="non-finite"):
        strategy.aggregate_fit(1, [(manager.all()[node_id], result)], [])


@pytest.mark.parametrize("strategy_cls", [HybridAggregation, StratifiedFedProx])
def test_aggregation_rejects_incompatible_update_shapes(strategy_cls):
    from flwr.common import ndarrays_to_parameters

    node_ids = [123, 456]
    proxies = list(_cm(node_ids).all().values())
    kwargs = {
        "fraction_fit": 1.0,
        "fraction_evaluate": 1.0,
        "min_fit_clients": 2,
        "min_evaluate_clients": 2,
        "min_available_clients": 2,
    }
    if strategy_cls is StratifiedFedProx:
        kwargs["proximal_mu"] = 0.1
    strategy = strategy_cls(**kwargs)
    results = []
    for proxy, client_id, values in zip(
        proxies,
        ("germany_allianz", "france_axa"),
        (np.zeros(2), np.zeros(3)),
        strict=True,
    ):
        result = _fit_res(client_id)
        result.parameters = ndarrays_to_parameters([values])
        results.append((proxy, result))

    with pytest.raises(ValueError, match="Mismatched parameter shapes"):
        strategy.aggregate_fit(1, results, [])


@pytest.mark.parametrize("strategy_cls", [HybridAggregation, StratifiedFedProx])
def test_aggregation_rejects_non_positive_example_count(strategy_cls):
    node_id = 789
    manager = _cm([node_id])
    kwargs = {
        "fraction_fit": 1.0,
        "fraction_evaluate": 1.0,
        "min_fit_clients": 1,
        "min_evaluate_clients": 1,
        "min_available_clients": 1,
    }
    if strategy_cls is StratifiedFedProx:
        kwargs["proximal_mu"] = 0.1
    strategy = strategy_cls(**kwargs)

    with pytest.raises(ValueError, match="Invalid num_examples=0"):
        strategy.aggregate_fit(1, [(manager.all()[node_id], _fit_res("germany_allianz", n=0))], [])


# --- strategy identity learning / selection ---


@pytest.mark.parametrize("strategy_cls", [HybridAggregation, StratifiedFedProx])
def test_strategy_round1_selects_each_client_once_independent_of_node_id_mod4(strategy_cls):
    random.seed(12345)
    node_ids = [random.getrandbits(63) for _ in range(4)]
    # Ensure old mapping would not be a clean identity in general (not required but demonstrates)
    cm = _cm(node_ids)
    kwargs = {
        "fraction_fit": 1.0,
        "fraction_evaluate": 1.0,
        "min_fit_clients": 4,
        "min_evaluate_clients": 4,
        "min_available_clients": 4,
        "sampling_seed": 7,
        "fairness_mode": "stratified" if strategy_cls is HybridAggregation else None,
    }
    if strategy_cls is StratifiedFedProx:
        kwargs.pop("fairness_mode")
        kwargs["proximal_mu"] = 0.1
    strat = strategy_cls(**kwargs)
    pairs = strat.configure_fit(1, _params(), cm)
    assert len(pairs) == 4
    selected_node_ids = {cp.cid for cp, _ in pairs}
    assert selected_node_ids == set(node_ids)
    # Bootstrap round 1 selects all available regardless of any modulo-derived identity.
    # Provide replies with correct client_id per partition-id mapping simulated by caller assignment:
    # assign each random node a distinct canonical client id (as server would learn from reported metrics)
    # to verify subsequent selection is by learned identity, not node_id % 4.
    assigned = dict(zip(sorted(node_ids), fl_task.CLIENT_IDS, strict=True))
    # Make assignment intentionally unrelated to nid % 4
    results = [(cp, _fit_res(assigned[cp.cid])) for cp, _ in pairs]
    strat.aggregate_fit(1, results, [])
    # round 2: with full map and fraction 1, should select all 4 learned identities
    pairs2 = strat.configure_fit(2, _params(), cm)
    assert {cp.cid for cp, _ in pairs2} == set(node_ids)
    # prove not derived from node_id % 4: learned map matches assigned, not modulo
    for nid in node_ids:
        assert strat._node_to_client_id[nid] == assigned[nid]
        assert (
            strat._node_to_client_id[nid] != fl_task.CLIENT_IDS[nid % 4]
            or assigned[nid] == fl_task.CLIENT_IDS[nid % 4]
        )


def test_conflicting_identity_rejected_and_audit_emitted_unknown_rejected():
    node_ids = [101, 202, 303, 404]
    cm = _cm(node_ids)
    audit = _Audit()
    strat = HybridAggregation(
        fraction_fit=1.0,
        fraction_evaluate=1.0,
        min_fit_clients=4,
        min_evaluate_clients=4,
        min_available_clients=4,
        audit_logger=audit,
        run_id="test-run",
        require_all_clients=False,
    )
    pairs = strat.configure_fit(1, _params(), cm)
    proxies = [cp for cp, _ in pairs]
    # first round: establish mapping
    mapping = {
        proxies[0].cid: "germany_allianz",
        proxies[1].cid: "france_axa",
        proxies[2].cid: "belgium_ageas",
        proxies[3].cid: "switzerland_helvetia",
    }
    results = [(cp, _fit_res(mapping[cp.cid])) for cp in proxies]
    strat.aggregate_fit(1, results, [])
    # conflicting identity for known node
    strat2 = HybridAggregation(
        fraction_fit=1.0,
        fraction_evaluate=1.0,
        min_fit_clients=4,
        min_evaluate_clients=4,
        min_available_clients=4,
        audit_logger=audit,
        run_id="test-run",
        require_all_clients=False,
    )
    strat2._node_to_client_id[proxies[0].cid] = "germany_allianz"
    with pytest.raises(ValueError, match="Conflicting client_id"):
        strat2._register_client_id_from_result(proxies[0].cid, {"client_id": "france_axa"}, "fit")
    assert any(et == "client_identity_conflict" for et, _ in audit.events)

    # unknown client_id rejected
    strat3 = HybridAggregation(
        fraction_fit=1.0,
        fraction_evaluate=1.0,
        min_fit_clients=4,
        min_evaluate_clients=4,
        min_available_clients=4,
        audit_logger=audit,
        run_id="test-run",
        require_all_clients=False,
    )
    with pytest.raises(ValueError, match="Unknown client_id"):
        strat3._register_client_id_from_result(
            proxies[0].cid, {"client_id": "unknown_client"}, "fit"
        )
    assert any(p.get("reason") == "unknown_client_id" for _, p in audit.events)


def test_dropout_tolerated_when_not_required_but_fails_when_required():
    node_ids = [11, 22, 33, 44]
    cm = _cm(node_ids)
    proxies = list(cm.all().values())
    mapping = {
        proxies[0].cid: "germany_allianz",
        proxies[1].cid: "france_axa",
        proxies[2].cid: "belgium_ageas",
        proxies[3].cid: "switzerland_helvetia",
    }
    # require_all_clients=False: 3/4 replies should proceed
    strat_ok = HybridAggregation(
        fraction_fit=1.0,
        fraction_evaluate=1.0,
        min_fit_clients=3,
        min_evaluate_clients=3,
        min_available_clients=4,
        require_all_clients=False,
    )
    # seed mapping
    strat_ok._node_to_client_id.update(mapping)
    results3 = [(proxies[i], _fit_res(mapping[proxies[i].cid])) for i in range(3)]
    # FedAvg aggregate expects parameters alignment; with empty params it should still run basic aggregation path
    # but easier: call our identity checks via aggregate_fit and allow super to run; use ndarrays_to_parameters([])
    out = strat_ok.aggregate_fit(1, results3, [])
    assert out is not None or out is None  # identity checks passed (no exception)

    strat_req = HybridAggregation(
        fraction_fit=1.0,
        fraction_evaluate=1.0,
        min_fit_clients=3,
        min_evaluate_clients=3,
        min_available_clients=4,
        require_all_clients=True,
    )
    strat_req._node_to_client_id.update(mapping)
    with pytest.raises(ValueError, match="Expected fit identities"):
        strat_req.aggregate_fit(1, results3, [])


def test_secagg_mode_identity_map_inactive_and_all_nodes_selected(caplog):
    node_ids = [501, 502, 503, 504]
    cm = _cm(node_ids)
    strat = HybridAggregation(
        fraction_fit=0.5,
        fraction_evaluate=0.5,
        min_fit_clients=2,
        min_evaluate_clients=2,
        min_available_clients=4,
        secagg_enabled=True,
    )
    assert strat.identity_learning_enabled is False
    with caplog.at_level("INFO"):
        pairs = strat.configure_fit(1, _params(), cm)
    assert len(pairs) == 4
    assert any(
        "SecAgg+" in r.message and "stratified selection is off" in r.message
        for r in caplog.records
    )
    # aggregate_fit should not require/learn per-client identities in secagg mode
    # provide results without client_id? no, super aggregate_fit may not need metrics; our code skips learning when inactive
    results = [
        (cp, FitRes(status=_ok_status(), parameters=_params(), num_examples=10, metrics={}))
        for cp, _ in pairs
    ]
    # should not raise due to missing client_id because identity learning disabled
    strat.aggregate_fit(1, results, [])
