"""LightGBM training over id-level folds, with OOF and test prediction.

Class imbalance (~3.55% positive) is handled with `scale_pos_weight`; ROC-AUC is
rank-based so we do not threshold. Early stopping uses each fold's own validation
split only (no leakage of validation info into the final fit)."""
from __future__ import annotations

import numpy as np
import lightgbm as lgb

DEFAULT_PARAMS = {
    "objective": "binary",
    "metric": "auc",
    "boosting_type": "gbdt",
    "learning_rate": 0.03,
    "num_leaves": 127,
    "feature_fraction": 0.7,
    "bagging_fraction": 0.8,
    "bagging_freq": 1,
    "min_child_samples": 200,
    "lambda_l2": 1.0,
    "max_depth": -1,
    "verbosity": -1,
}


def train_cv(
    X: np.ndarray,
    y: np.ndarray,
    folds: np.ndarray,
    X_test: np.ndarray,
    feature_names: list[str],
    params: dict | None = None,
    num_boost_round: int = 5000,
    early_stopping_rounds: int = 200,
    seed: int = 42,
):
    """Train one model per fold. Returns (oof, test_pred, models, best_iters)."""
    p = dict(DEFAULT_PARAMS)
    if params:
        p.update(params)
    p.setdefault("seed", seed)
    # scale_pos_weight from the global positive rate unless overridden.
    if "scale_pos_weight" not in p:
        pos = y.mean()
        p["scale_pos_weight"] = float((1 - pos) / max(pos, 1e-9))

    oof = np.zeros(len(y), dtype="float64")
    test_pred = np.zeros(X_test.shape[0], dtype="float64")
    models = []
    best_iters = []
    n_folds = int(folds.max()) + 1

    for k in range(n_folds):
        tr = folds != k
        va = folds == k
        dtrain = lgb.Dataset(X[tr], label=y[tr], feature_name=feature_names)
        dvalid = lgb.Dataset(X[va], label=y[va], feature_name=feature_names)
        booster = lgb.train(
            p,
            dtrain,
            num_boost_round=num_boost_round,
            valid_sets=[dvalid],
            callbacks=[
                lgb.early_stopping(early_stopping_rounds, verbose=False),
                lgb.log_evaluation(0),
            ],
        )
        oof[va] = booster.predict(X[va], num_iteration=booster.best_iteration)
        test_pred += booster.predict(X_test, num_iteration=booster.best_iteration) / n_folds
        models.append(booster)
        best_iters.append(booster.best_iteration)
        print(f"[lgbm] fold {k}: best_iter={booster.best_iteration} "
              f"auc={booster.best_score['valid_0']['auc']:.5f}")

    return oof, test_pred, models, best_iters
