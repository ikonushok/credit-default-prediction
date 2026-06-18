"""CatBoost training over id-level folds (phase 2 — ensemble diversity).

Same aggregated features and folds as the LightGBM runs, so OOF/test predictions
align by id for fair comparison and blending. Diversity comes from the different
algorithm (ordered boosting, symmetric trees), not from different data."""
from __future__ import annotations

import numpy as np
from catboost import CatBoostClassifier, Pool

DEFAULT_PARAMS = {
    "loss_function": "Logloss",
    "eval_metric": "AUC",
    "learning_rate": 0.03,
    "depth": 8,
    "l2_leaf_reg": 3.0,
    "random_strength": 1.0,
    "auto_class_weights": "Balanced",  # handles the ~3.55% positive rate
    "bootstrap_type": "Bernoulli",
    "subsample": 0.8,
    "allow_writing_files": False,
    "verbose": False,
}


def train_cv(X, y, folds, X_test, feature_names, params=None,
             num_boost_round=5000, early_stopping_rounds=200, seed=42):
    """Train one CatBoost per fold. Returns (oof, test_pred, models, best_iters)."""
    p = dict(DEFAULT_PARAMS)
    if params:
        p.update(params)
    p["iterations"] = num_boost_round
    p["random_seed"] = seed

    oof = np.zeros(len(y), dtype="float64")
    test_pred = np.zeros(X_test.shape[0], dtype="float64")
    models, best_iters = [], []
    n_folds = int(folds.max()) + 1
    test_pool = Pool(X_test, feature_names=list(feature_names))

    for k in range(n_folds):
        tr, va = folds != k, folds == k
        dtr = Pool(X[tr], label=y[tr], feature_names=list(feature_names))
        dva = Pool(X[va], label=y[va], feature_names=list(feature_names))
        model = CatBoostClassifier(**p)
        model.fit(dtr, eval_set=dva, early_stopping_rounds=early_stopping_rounds, verbose=False)
        oof[va] = model.predict_proba(dva)[:, 1]
        test_pred += model.predict_proba(test_pool)[:, 1] / n_folds
        models.append(model)
        best_iters.append(model.get_best_iteration())
        print(f"[catboost] fold {k}: best_iter={model.get_best_iteration()} "
              f"auc={model.get_best_score()['validation']['AUC']:.5f}")

    return oof, test_pred, models, best_iters
