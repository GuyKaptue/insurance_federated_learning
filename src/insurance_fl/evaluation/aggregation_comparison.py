"""In-process comparison of federated aggregation modes on the synthetic 4-client federation.

The in-process executor reuses the real ``build_strategy``/``GatedStrategy`` path and
real client training/evaluation code (``insurance_fl.fl.task`` + ``MultiTaskNN``), but
it is NOT a Flower/Ray simulation. Flower simulation runs go through
``scripts/flwr_sim.sh`` via the CLI.
"""

from __future__ import annotations

import hashlib
import json
import subprocess
import time
from dataclasses import asdict, dataclass
from pathlib import Path
from typing import Any

import numpy as np
import pandas as pd
from flwr.common import (
    Code,
    EvaluateIns,
    EvaluateRes,
    FitRes,
    Status,
    ndarrays_to_parameters,
    parameters_to_ndarrays,
)
from flwr.server.client_manager import SimpleClientManager
from flwr.server.client_proxy import ClientProxy

from insurance_fl.fl.privacy import PrivacyBackendUnavailable, PrivacyConfig, RdpAccountantWrapper
from insurance_fl.fl.strategy import _ALLOWED, build_strategy
from insurance_fl.fl.task import CLIENT_IDS, build_model, evaluate, load_partition, train

SUPPORTED_MODES: tuple[str, ...] = tuple(sorted(_ALLOWED))
_OK = Status(code=Code.OK, message="")


def _repo_root(start: Path | None = None) -> Path:
    p = (start or Path.cwd()).resolve()
    for cand in [p, *p.parents]:
        if (cand / "pyproject.toml").exists():
            return cand
    return p


@dataclass(frozen=True)
class ComparisonConfig:
    modes: tuple[str, ...]
    seeds: tuple[int, ...]
    rounds: int
    n_rows_per_client: int
    privacy_enabled: bool
    clip_norm: float = 1.0
    noise_std: float = 1.1  # = noise_multiplier * clip_norm
    delta: float = 1e-5
    max_epsilon: float = 8.0
    executor: str = "inprocess"  # "inprocess" | "flower"
    data_dir: str | Path | None = None


@dataclass
class ComparisonResult:
    runs: pd.DataFrame
    per_client: pd.DataFrame
    history: pd.DataFrame
    provenance: dict

    def to_json(self, path: str | Path) -> None:
        p = Path(path)
        p.parent.mkdir(parents=True, exist_ok=True)
        payload = {
            "runs": self.runs.to_dict(orient="records"),
            "per_client": self.per_client.to_dict(orient="records"),
            "history": self.history.to_dict(orient="records"),
            "provenance": self.provenance,
        }
        p.write_text(json.dumps(payload, indent=2, sort_keys=True) + "\n", encoding="utf-8")

    @classmethod
    def from_json(cls, path: str | Path) -> ComparisonResult:
        payload = json.loads(Path(path).read_text(encoding="utf-8"))
        return cls(
            runs=pd.DataFrame(payload.get("runs", [])),
            per_client=pd.DataFrame(payload.get("per_client", [])),
            history=pd.DataFrame(payload.get("history", [])),
            provenance=dict(payload.get("provenance", {})),
        )


class _DummyProxy(ClientProxy):
    def __init__(self, cid: str):
        self.cid = str(cid)

    def get_properties(self, ins, timeout=None, group_id=None):
        raise NotImplementedError

    def get_parameters(self, ins, timeout=None, group_id=None):
        raise NotImplementedError

    def fit(self, ins, timeout=None, group_id=None):
        raise NotImplementedError

    def evaluate(self, ins, timeout=None, group_id=None):
        raise NotImplementedError

    def reconnect(self, ins, timeout=None, group_id=None):
        raise NotImplementedError


def _git_sha() -> str | None:
    try:
        out = subprocess.check_output(
            ["git", "rev-parse", "HEAD"], stderr=subprocess.DEVNULL, text=True
        )
        return out.strip() or None
    except Exception:
        return None


def _dataset_hash(data_dir: Path) -> str | None:
    manifest = data_dir / "dataset_info.json"
    if not manifest.exists():
        return None
    try:
        return hashlib.sha256(manifest.read_bytes()).hexdigest()
    except Exception:
        return None


def _make_accountant(cfg: ComparisonConfig) -> RdpAccountantWrapper | None:
    if not cfg.privacy_enabled:
        return None
    nm = cfg.noise_std / max(cfg.clip_norm, 1e-12)
    return RdpAccountantWrapper(
        PrivacyConfig(
            max_epsilon=cfg.max_epsilon,
            target_delta=cfg.delta,
            delta=cfg.delta,
            noise_multiplier=nm,
            sampling_rate=1.0,
        )
    )


def rounds_allowed(cfg: ComparisonConfig) -> int:
    """Dry-run accountant: number of rounds before max_epsilon would be exceeded."""
    acc = _make_accountant(cfg)
    if acc is None:
        return int(cfg.rounds)
    allowed = 0
    for _ in range(int(cfg.rounds)):
        if acc.project_epsilon(additional_releases=1) > cfg.max_epsilon:
            break
        acc.step()
        allowed += 1
    return allowed


def _cap_rows(x: np.ndarray, y: dict[str, np.ndarray], n: int, rng: np.random.Generator):
    n = min(int(n), len(x))
    idx = rng.choice(len(x), size=n, replace=False)
    return x[idx], {k: v[idx] for k, v in y.items()}


def _auc_from_logits(logits: np.ndarray, y_true: np.ndarray) -> float:
    from sklearn.metrics import roc_auc_score

    y = np.asarray(y_true).reshape(-1)
    if len(np.unique(y)) < 2:
        return float("nan")
    s = np.asarray(logits).reshape(-1)
    try:
        return float(roc_auc_score(y, s))
    except Exception:
        return float("nan")


def _eval_client(model, x, y) -> dict[str, float]:
    import torch

    model.eval()
    with torch.no_grad():
        xt = torch.as_tensor(x)
        logits = model.forward(xt)
        # binary classification AUC on is_high_risk; loss from evaluate
        metrics = evaluate(model, x, y)
        auc = _auc_from_logits(logits["is_high_risk"].numpy(), y["is_high_risk"])
    return {"auc": auc, "loss": float(metrics.get("loss", float("nan")))}


def _train_client(model, x, y, *, seed: int) -> float:
    torch = __import__("torch")
    torch.manual_seed(int(seed))
    np.random.seed(int(seed))
    return float(train(model, x, y, epochs=1, batch_size=32, lr=1e-3))


def _initial_parameters(data_dir: Path) -> tuple[Any, int]:
    cid = CLIENT_IDS[0]
    x, _ = load_partition(data_dir, cid, "train")
    model = build_model(x.shape[1])
    params = ndarrays_to_parameters(model.get_federated_weights())
    return params, x.shape[1]


def _run_inprocess_mode(
    cfg: ComparisonConfig, mode: str, seed: int, data_dir: Path
) -> dict[str, Any]:
    t0 = time.perf_counter()
    np.random.default_rng(seed)
    run_row: dict[str, Any] = {
        "mode": mode,
        "seed": seed,
        "status": "OK",
        "error": "",
        "rounds_trained": 0,
        "rounds_skipped": 0,
        "epsilon_final": None,
        "stopped_for_budget": False,
        "wall_s": 0.0,
    }
    per_client_rows: list[dict[str, Any]] = []
    history_rows: list[dict[str, Any]] = []

    try:
        client_data = {}
        for cid in CLIENT_IDS:
            xtr, ytr = load_partition(data_dir, cid, "train")
            xte, yte = load_partition(data_dir, cid, "test")
            xtr, ytr = _cap_rows(xtr, ytr, cfg.n_rows_per_client, np.random.default_rng(seed))
            xte, yte = _cap_rows(
                xte, yte, max(200, cfg.n_rows_per_client // 2), np.random.default_rng(seed + 1)
            )
            client_data[cid] = (xtr, ytr, xte, yte)

        initial_params, input_dim = _initial_parameters(data_dir)
        accountant = _make_accountant(cfg)
        strategy = build_strategy(
            mode,
            fraction_fit=1.0,
            fraction_evaluate=1.0,
            min_fit_clients=len(CLIENT_IDS),
            min_evaluate_clients=len(CLIENT_IDS),
            min_available_clients=len(CLIENT_IDS),
            accountant=accountant,
            max_epsilon=cfg.max_epsilon if cfg.privacy_enabled else None,
            privacy_state_path=None,
            fit_metrics_aggregation_fn=lambda metrics: {},
            evaluate_metrics_aggregation_fn=lambda metrics: {},
        )
        # initialize strategy parameters
        init = getattr(strategy, "initialize_parameters", None)
        if callable(init):
            p = init(None)
            if p is not None:
                initial_params = p

        # QFedAvg in Flower 1.31 uses self.evaluate(...) loss to weight client deltas.
        # In-process comparison has no server-side eval fn by default; provide one so
        # qfedavg can aggregate instead of hitting a free-variable loss bug.
        eval_target = getattr(strategy, "inner", strategy)
        if getattr(eval_target, "evaluate_fn", "missing") is None:
            def _qfed_eval(server_round: int, parameters, _config=None):
                losses = []
                ndarrays = (
                    parameters_to_ndarrays(parameters)
                    if hasattr(parameters, "tensors")
                    else parameters
                )
                for cid in CLIENT_IDS:
                    _, _, xte, yte = client_data[cid]
                    model = build_model(input_dim)
                    model.set_federated_weights(ndarrays)
                    m = _eval_client(model, xte, yte)
                    losses.append(float(m["loss"]))
                return float(np.nanmean(losses)) if losses else 0.0, {}

            eval_target.evaluate_fn = _qfed_eval

        proxies = [_DummyProxy(str(i)) for i in range(len(CLIENT_IDS))]
        cm = SimpleClientManager()
        for p in proxies:
            cm.register(p)
        # Map proxies to client ids deterministically by order
        proxy_to_cid = {proxies[i].cid: CLIENT_IDS[i] for i in range(len(CLIENT_IDS))}
        current_params = initial_params

        for rnd in range(1, int(cfg.rounds) + 1):
            fit_ins_list = strategy.configure_fit(rnd, current_params, cm)
            dispatched = bool(fit_ins_list)
            eps = None
            acc_obj = getattr(strategy, "accountant", None) or accountant
            if acc_obj is not None:
                try:
                    eps_raw = acc_obj.epsilon().get("epsilon")
                    if eps_raw is None:
                        raise PrivacyBackendUnavailable("accountant returned no epsilon")
                    eps = float(eps_raw)
                except Exception:
                    eps = None
            if not dispatched:
                run_row["rounds_skipped"] += 1
                # no dispatch this round; still record history using last eval if any
                last_auc = history_rows[-1]["global_auc"] if history_rows else float("nan")
                history_rows.append(
                    {
                        "mode": mode,
                        "seed": seed,
                        "round": rnd,
                        "dispatched": False,
                        "epsilon": eps,
                        "global_auc": last_auc,
                    }
                )
                continue

            # train selected clients
            fit_results = []
            for proxy, fit_ins in fit_ins_list:
                cid = proxy_to_cid.get(proxy.cid, CLIENT_IDS[int(proxy.cid) % len(CLIENT_IDS)])
                xtr, ytr, _, _ = client_data[cid]
                model = build_model(input_dim)
                model.set_federated_weights(parameters_to_ndarrays(fit_ins.parameters))
                loss = _train_client(model, xtr, ytr, seed=seed + rnd)
                nds = model.get_federated_weights()
                fit_results.append(
                    (
                        proxy,
                        FitRes(
                            status=_OK,
                            parameters=ndarrays_to_parameters(nds),
                            num_examples=len(xtr),
                            metrics={"train_loss": loss, "client_id": cid},
                        ),
                    )
                )
            agg = strategy.aggregate_fit(rnd, fit_results, [])
            if agg is not None and agg[0] is not None:
                current_params = agg[0]
            run_row["rounds_trained"] += 1

            # evaluate all clients
            eval_ins_list = strategy.configure_evaluate(rnd, current_params, cm)
            if not eval_ins_list:
                eval_ins_list = [(p, EvaluateIns(current_params, {})) for p in proxies]
            eval_results = []
            aucs = []
            for proxy, ev_ins in eval_ins_list:
                cid = proxy_to_cid.get(proxy.cid, CLIENT_IDS[int(proxy.cid) % len(CLIENT_IDS)])
                _, _, xte, yte = client_data[cid]
                model = build_model(input_dim)
                model.set_federated_weights(parameters_to_ndarrays(ev_ins.parameters))
                m = _eval_client(model, xte, yte)
                aucs.append(m["auc"])
                eval_results.append(
                    (
                        proxy,
                        EvaluateRes(
                            status=_OK,
                            loss=float(m["loss"]),
                            num_examples=len(xte),
                            metrics={
                                "loss": float(m["loss"]),
                                "auc": float(m["auc"]),
                                "client_id": cid,
                            },
                        ),
                    )
                )
            strategy.aggregate_evaluate(rnd, eval_results, [])
            history_rows.append(
                {
                    "mode": mode,
                    "seed": seed,
                    "round": rnd,
                    "dispatched": True,
                    "epsilon": eps,
                    "global_auc": float(np.nanmean(aucs)) if aucs else float("nan"),
                }
            )

        # final per-client metrics on test split with current params
        for cid in CLIENT_IDS:
            _, _, xte, yte = client_data[cid]
            model = build_model(input_dim)
            model.set_federated_weights(parameters_to_ndarrays(current_params))
            m = _eval_client(model, xte, yte)
            per_client_rows.append(
                {"mode": mode, "seed": seed, "client": cid, "auc": m["auc"], "loss": m["loss"]}
            )

        acc_obj = getattr(strategy, "accountant", None) or accountant
        if acc_obj is not None:
            try:
                eps_raw = acc_obj.epsilon().get("epsilon")
                if eps_raw is None:
                    raise PrivacyBackendUnavailable("accountant returned no epsilon")
                run_row["epsilon_final"] = float(eps_raw)
            except Exception:
                run_row["epsilon_final"] = None
        run_row["stopped_for_budget"] = bool(getattr(strategy, "stopped_for_budget", False))
        if run_row["rounds_trained"] == 0 and cfg.privacy_enabled:
            # if never trained due to immediate budget refusal, mark stopped
            run_row["stopped_for_budget"] = bool(getattr(strategy, "budget_exhausted", True))
    except Exception as exc:
        run_row["status"] = "FAILED"
        run_row["error"] = f"{type(exc).__name__}: {exc}"

    run_row["wall_s"] = time.perf_counter() - t0
    return {"run": run_row, "per_client": per_client_rows, "history": history_rows}


def run_comparison(cfg: ComparisonConfig) -> ComparisonResult:
    if cfg.executor not in {"inprocess", "flower"}:
        raise ValueError(f"unsupported executor: {cfg.executor}")
    unknown = [m for m in cfg.modes if m not in SUPPORTED_MODES]
    if unknown:
        raise ValueError(f"unknown modes: {unknown}; supported={SUPPORTED_MODES}")
    if cfg.data_dir is not None:
        data_dir = Path(cfg.data_dir)
        if not data_dir.is_absolute():
            data_dir = (_repo_root() / data_dir).resolve()
    else:
        data_dir = (_repo_root() / "data/processed/european_insurance_federated").resolve()
    runs: list[dict[str, Any]] = []
    per_client: list[dict[str, Any]] = []
    history: list[dict[str, Any]] = []
    for mode in cfg.modes:
        for seed in cfg.seeds:
            if cfg.executor == "flower":
                # Flower executor is intentionally thin; in-process is default.
                runs.append(
                    {
                        "mode": mode,
                        "seed": seed,
                        "status": "BLOCKED",
                        "error": "flower executor requires scripts/run_aggregation_comparison.py --executor flower",
                        "rounds_trained": 0,
                        "rounds_skipped": 0,
                        "epsilon_final": None,
                        "stopped_for_budget": False,
                        "wall_s": 0.0,
                    }
                )
                continue
            out = _run_inprocess_mode(cfg, mode, int(seed), data_dir)
            runs.append(out["run"])
            per_client.extend(out["per_client"])
            history.extend(out["history"])

    provenance = {
        "config": asdict(cfg),
        "config_hash": hashlib.sha256(
            json.dumps(asdict(cfg), sort_keys=True).encode("utf-8")
        ).hexdigest(),
        "dataset_hash_files_content_sha256": _dataset_hash(data_dir),
        "git_sha": _git_sha(),
        "versions": {
            "python": subprocess.check_output(["python3", "--version"], text=True).strip(),
        },
        "executor": cfg.executor,
    }
    return ComparisonResult(
        runs=pd.DataFrame(runs),
        per_client=pd.DataFrame(per_client),
        history=pd.DataFrame(history),
        provenance=provenance,
    )
