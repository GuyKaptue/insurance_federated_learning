"""Per-client booster pre-training and stacking (§1.4, §1.2).

Trains XGBoost, LightGBM, and CatBoost per client per task with
validation-based early stopping, generates out-of-fold (OOF) predictions,
and solves NNLS stacking coefficients via ``fl/stacking.py``.

Boosters remain local to each client (never federated).
"""

from __future__ import annotations

from pathlib import Path
from typing import Any

import numpy as np

from insurance_fl.fl.stacking import fit_stacking_meta, save_stacking_meta
from insurance_fl.fl.task import DEFAULT_TASK_TYPES


def _task_type(task_name: str) -> str:
    return DEFAULT_TASK_TYPES.get(task_name, "regression")


def train_boosters_for_client(
    client_id: str,
    x_train: np.ndarray,
    y_train: dict[str, np.ndarray],
    x_val: np.ndarray,
    y_val: dict[str, np.ndarray],
    output_dir: Path,
    *,
    tasks: list[str] | None = None,
    n_folds: int = 5,
    seed: int = 42,
) -> dict[str, Any]:
    """Train boosters per task for one client.

    Returns a dict of {task: {alpha_xgb, alpha_lgbm, alpha_cat, oof_rmse, ...}}.
    Persists artifacts to output_dir.
    """
    if tasks is None:
        tasks = list(DEFAULT_TASK_TYPES.keys())

    output_dir = Path(output_dir)
    output_dir.mkdir(parents=True, exist_ok=True)

    results: dict[str, Any] = {}

    for task in tasks:
        task_type = _task_type(task)
        y_tr = np.asarray(y_train[task])
        y_v = np.asarray(y_val[task])

        # Train individual boosters
        xgb_model = _train_xgboost(x_train, y_tr, x_val, y_v, task_type, seed)
        lgbm_model = _train_lightgbm(x_train, y_tr, x_val, y_v, task_type, seed)
        cat_model = _train_catboost(x_train, y_tr, x_val, y_v, task_type, seed)

        # Generate OOF predictions
        oof_preds = _generate_oof(x_train, y_tr, task_type, n_folds, seed)

        # Fit stacking meta
        if oof_preds is not None:
            meta = fit_stacking_meta(
                [oof_preds["xgb"], oof_preds["lgbm"], oof_preds["cat"]],
                y_tr,
            )
            raw = list(meta) if not hasattr(meta, "tolist") else meta.tolist()
            alpha = np.asarray(raw, dtype=float).ravel()
            fallback = False
        else:
            alpha = np.asarray([1.0, 0.0, 0.0], dtype=float)
            fallback = True

        # FIX: retrain final boosters on ALL training rows, not a fold booster.
        xgb_model = _train_xgboost(
            x_train,
            y_tr,
            x_val,
            y_v,
            task_type,
            seed,
            early_stopping=20,
            max_depth=3,
            n_estimators=200,
        )
        lgbm_model = _train_lightgbm(
            x_train,
            y_tr,
            x_val,
            y_v,
            task_type,
            seed,
            early_stopping=20,
            max_depth=3,
            n_estimators=200,
        )
        cat_model = _train_catboost(
            x_train, y_tr, x_val, y_v, task_type, seed, early_stopping=20, depth=3, iterations=200
        )

        # Save artifacts (after retrain)
        xgb_path = output_dir / f"xgboost_{task}.json"
        lgbm_path = output_dir / f"lightgbm_{task}.txt"
        cat_path = output_dir / f"catboost_{task}.cbm"

        _save_xgboost(xgb_model, xgb_path)
        _save_lightgbm(lgbm_model, lgbm_path)
        _save_catboost(cat_model, cat_path)

        # Save stacking meta
        if oof_preds is not None:
            if task_type == "regression":
                oof_loss = float(np.mean((oof_preds["xgb"] - y_tr) ** 2))
            elif task_type == "binary":
                p = np.clip(oof_preds["xgb"], 1e-12, 1.0 - 1e-12)
                oof_loss = float(-np.mean(y_tr * np.log(p) + (1 - y_tr) * np.log(1 - p)))
            else:
                oof_loss = float("nan")
        else:
            oof_loss = 0.0
        save_stacking_meta(output_dir / f"stacking_meta_{task}.json", alpha, oof_loss, fallback)

        results[task] = {
            "alpha_xgb": float(alpha[0]),
            "alpha_lgbm": float(alpha[1]),
            "alpha_cat": float(alpha[2]),
            "task": task,
            "task_type": task_type,
            "client_id": client_id,
            "fallback": fallback,
        }

    return results


def _train_xgboost(
    x_train,
    y_train,
    x_val,
    y_val,
    task_type,
    seed,
    early_stopping=20,
    max_depth=3,
    n_estimators=200,
):
    import xgboost as xgb

    kwargs: dict[str, Any] = {
        "n_estimators": n_estimators,
        "max_depth": max_depth,
        "learning_rate": 0.1,
        "random_state": seed,
    }
    # xgboost>=2.0 removed use_label_encoder and supports early_stopping_rounds
    # in the sklearn estimator constructor.
    if early_stopping is not None:
        kwargs["early_stopping_rounds"] = early_stopping
    if task_type == "regression":
        model = xgb.XGBRegressor(**kwargs)
    elif task_type == "binary":
        model = xgb.XGBClassifier(eval_metric="logloss", **kwargs)
    else:
        model = xgb.XGBClassifier(eval_metric="mlogloss", **kwargs)
    if early_stopping is not None:
        model.fit(x_train, y_train, eval_set=[(x_val, y_val)], verbose=False)
    else:
        model.fit(x_train, y_train, verbose=False)
    return model


def _train_lightgbm(
    x_train,
    y_train,
    x_val,
    y_val,
    task_type,
    seed,
    early_stopping=20,
    max_depth=3,
    n_estimators=200,
):
    import lightgbm as lgb

    if task_type == "regression":
        model = lgb.LGBMRegressor(
            n_estimators=n_estimators,
            max_depth=max_depth,
            learning_rate=0.1,
            random_state=seed,
            verbose=-1,
        )
    else:
        model = lgb.LGBMClassifier(
            n_estimators=n_estimators,
            max_depth=max_depth,
            learning_rate=0.1,
            random_state=seed,
            verbose=-1,
        )

    if early_stopping is not None:
        try:
            model.fit(
                x_train,
                y_train,
                eval_set=[(x_val, y_val)],
                callbacks=[lgb.early_stopping(early_stopping, verbose=False)],
            )
        except Exception:
            model.fit(x_train, y_train)
    else:
        model.fit(x_train, y_train)
    return model


def _train_catboost(
    x_train,
    y_train,
    x_val,
    y_val,
    task_type,
    seed,
    early_stopping=20,
    depth=3,
    iterations=200,
):
    from catboost import CatBoostClassifier, CatBoostRegressor

    cls = CatBoostRegressor if task_type == "regression" else CatBoostClassifier
    model = cls(
        iterations=iterations,
        depth=depth,
        learning_rate=0.1,
        random_seed=seed,
        verbose=0,
    )

    if early_stopping is not None:
        try:
            model.fit(
                x_train, y_train, eval_set=(x_val, y_val), early_stopping_rounds=early_stopping
            )
        except Exception:
            model.fit(x_train, y_train)
    else:
        model.fit(x_train, y_train)
    return model


def _save_xgboost(model, path):
    model.save_model(str(path))


def _save_lightgbm(model, path):
    model.booster_.save_model(str(path))


def _save_catboost(model, path):
    model.save_model(str(path))


def _generate_oof(x_train, y_train, task_type, n_folds, seed):
    """Generate OOF predictions for all three boosters.

    FIX: clamp n_folds so each fold has >=200 training rows. Disable
    early stopping when per-fold train size < 500.
    """
    from sklearn.model_selection import KFold, StratifiedKFold

    n = len(x_train)
    if n < 2 * 100:
        return None

    # FIX: require >=200 train rows per fold
    n_folds = min(n_folds, max(2, n // 200))

    oof_xgb = np.zeros(n)
    oof_lgbm = np.zeros(n)
    oof_cat = np.zeros(n)

    if task_type == "regression":
        kf = KFold(n_splits=n_folds, shuffle=True, random_state=seed)
    else:
        try:
            kf = StratifiedKFold(n_splits=n_folds, shuffle=True, random_state=seed)
        except ValueError:
            kf = KFold(n_splits=n_folds, shuffle=True, random_state=seed)

    per_fold_train = int(n * (n_folds - 1) / n_folds)
    es_rounds = 10 if per_fold_train < 500 else 20

    for train_idx, val_idx in kf.split(x_train, y_train if task_type != "regression" else None):
        x_tr, x_vl = x_train[train_idx], x_train[val_idx]
        y_tr, y_vl = y_train[train_idx], y_train[val_idx]

        m_xgb = _train_xgboost(
            x_tr,
            y_tr,
            x_vl,
            y_vl,
            task_type,
            seed,
            early_stopping=es_rounds,
            max_depth=3,
            n_estimators=200,
        )
        m_lgbm = _train_lightgbm(
            x_tr,
            y_tr,
            x_vl,
            y_vl,
            task_type,
            seed,
            early_stopping=es_rounds,
            max_depth=3,
            n_estimators=200,
        )
        m_cat = _train_catboost(
            x_tr,
            y_tr,
            x_vl,
            y_vl,
            task_type,
            seed,
            early_stopping=es_rounds,
            depth=3,
            iterations=200,
        )

        oof_xgb[val_idx] = _predict(m_xgb, x_vl, task_type)
        oof_lgbm[val_idx] = _predict(m_lgbm, x_vl, task_type)
        oof_cat[val_idx] = _predict(m_cat, x_vl, task_type)

    return {"xgb": oof_xgb, "lgbm": oof_lgbm, "cat": oof_cat}


def _predict(model, x, task_type):
    if task_type == "regression":
        return model.predict(x)
    elif task_type == "binary":
        proba = model.predict_proba(x)
        return proba[:, 1] if proba.ndim == 2 else proba
    else:
        proba = model.predict_proba(x)
        return proba[:, 1] if proba.ndim == 2 and proba.shape[1] > 1 else proba.ravel()


def artifact_dir_for(client_id: str, base_dir: str | Path = "artifacts") -> Path:
    """Return the deterministic artifact directory for a client."""
    return Path(base_dir) / "boosters" / client_id


def load_stacking_meta_for_task(artifact_dir: Path, task: str) -> dict[str, Any] | None:
    """Load stacking meta for a task, or None if unavailable."""
    from insurance_fl.fl.stacking import load_stacking_meta

    path = artifact_dir / f"stacking_meta_{task}.json"
    if not path.exists():
        return None
    try:
        return load_stacking_meta(path)
    except Exception:
        return None


def load_booster_artifacts(artifact_dir: Path, task: str) -> dict[str, Path]:
    """Return paths to booster artifacts for a task."""
    return {
        "xgb": artifact_dir / f"xgboost_{task}.json",
        "lgbm": artifact_dir / f"lightgbm_{task}.txt",
        "cat": artifact_dir / f"catboost_{task}.cbm",
        "stacking_meta": artifact_dir / f"stacking_meta_{task}.json",
    }


def artifacts_available(artifact_dir: Path, task: str) -> bool:
    """Check if all required artifacts exist for a task."""
    arts = load_booster_artifacts(artifact_dir, task)
    return all(p.exists() for p in arts.values())


__all__ = [
    "artifact_dir_for",
    "artifacts_available",
    "load_booster_artifacts",
    "load_stacking_meta_for_task",
    "train_boosters_for_client",
]
