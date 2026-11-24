from __future__ import annotations

import pandas as pd

from insurance_fl.evaluation.aggregation_comparison import (
    SUPPORTED_MODES,
    ComparisonConfig,
    ComparisonResult,
    rounds_allowed,
    run_comparison,
)
from insurance_fl.fl.strategy import build_strategy


def test_supported_modes_match_build_strategy():
    # build_strategy accepts the canonical 4 modes
    for m in SUPPORTED_MODES:
        s = build_strategy(m, min_fit_clients=1, min_evaluate_clients=1, min_available_clients=4)
        assert s is not None
    assert set(SUPPORTED_MODES) == {"fedavg", "fedprox", "hybrid", "qfedavg"}


def test_rounds_allowed_privacy_binds():
    cfg = ComparisonConfig(
        modes=SUPPORTED_MODES,
        seeds=(0,),
        rounds=5,
        n_rows_per_client=50,
        privacy_enabled=True,
        clip_norm=1.0,
        noise_std=1.1,
        delta=1e-5,
        max_epsilon=8.0,
        executor="inprocess",
    )
    assert rounds_allowed(cfg) < cfg.rounds


def test_failed_mode_not_dropped(monkeypatch, tmp_path):
    from insurance_fl.evaluation import aggregation_comparison as ac

    real = ac._run_inprocess_mode

    def fake(cfg, mode, seed, data_dir):
        if mode == "fedprox":
            return {
                "run": {
                    "mode": mode,
                    "seed": seed,
                    "status": "FAILED",
                    "error": "boom",
                    "rounds_trained": 0,
                    "rounds_skipped": 0,
                    "epsilon_final": None,
                    "stopped_for_budget": False,
                    "wall_s": 0.0,
                },
                "per_client": [],
                "history": [],
            }
        return real(cfg, mode, seed, data_dir)

    monkeypatch.setattr(ac, "_run_inprocess_mode", fake)
    cfg = ComparisonConfig(
        modes=("fedavg", "fedprox"),
        seeds=(0,),
        rounds=1,
        n_rows_per_client=32,
        privacy_enabled=False,
        executor="inprocess",
    )
    res = run_comparison(cfg)
    assert set(res.runs["mode"]) == {"fedavg", "fedprox"}
    row = res.runs[res.runs["mode"] == "fedprox"].iloc[0]
    assert row["status"] == "FAILED"
    assert "boom" in row["error"]


def test_privacy_gate_refuses_all_modes(tmp_path):
    cfg = ComparisonConfig(
        modes=SUPPORTED_MODES,
        seeds=(0,),
        rounds=5,
        n_rows_per_client=64,
        privacy_enabled=True,
        clip_norm=1.0,
        noise_std=1.1,
        delta=1e-5,
        max_epsilon=8.0,
        executor="inprocess",
    )
    expected = rounds_allowed(cfg)
    res = run_comparison(cfg)
    ok = res.runs[res.runs["status"] == "OK"]
    assert set(res.runs["mode"]) == set(SUPPORTED_MODES)
    assert (ok["rounds_trained"] == expected).all()
    assert (ok["rounds_skipped"] == cfg.rounds - expected).all()
    assert ok["stopped_for_budget"].all()


def test_json_round_trip(tmp_path):
    res = ComparisonResult(
        runs=pd.DataFrame([{"mode": "fedavg", "seed": 0, "status": "OK"}]),
        per_client=pd.DataFrame(
            [{"mode": "fedavg", "seed": 0, "client": "c", "auc": 0.5, "loss": 1.0}]
        ),
        history=pd.DataFrame(
            [
                {
                    "mode": "fedavg",
                    "seed": 0,
                    "round": 1,
                    "dispatched": True,
                    "epsilon": None,
                    "global_auc": 0.5,
                }
            ]
        ),
        provenance={"executor": "inprocess"},
    )
    p = tmp_path / "r.json"
    res.to_json(p)
    loaded = ComparisonResult.from_json(p)
    assert loaded.provenance["executor"] == "inprocess"
    assert loaded.runs.iloc[0]["mode"] == "fedavg"
    assert len(loaded.history) == 1
