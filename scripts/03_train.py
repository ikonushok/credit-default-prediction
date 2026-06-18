"""CLI: train a model over id-level folds, write OOF / test predictions / metrics.

Usage:
    python scripts/03_train.py --config configs/lgbm_baseline.yaml

Reads data/processed/{train,test}_features.parquet (run 02_aggregate first).
Saves to artifacts/<run_id>/: folds.npy, oof.npy, test_pred.npy, test_ids.npy,
metrics.json, and the resolved config. Appends a row to experiments/experiment_log.csv.
"""
import argparse
import json
import shutil
from pathlib import Path

import numpy as np
import pandas as pd

import _bootstrap  # noqa: F401

from credit_scoring import config as C
from credit_scoring import cv, metrics, tracking
from credit_scoring.config import RunConfig
from credit_scoring.models import lgbm
from credit_scoring.models import catboost_model


def _load_features(feature_set: str):
    train_path = C.features_path("train", feature_set)
    test_path = C.features_path("test", feature_set)
    # Fallback to legacy unversioned files for the baseline set.
    if not train_path.exists() and feature_set == "baseline":
        train_path = C.DATA_PROCESSED / "train_features.parquet"
        test_path = C.DATA_PROCESSED / "test_features.parquet"
    if not train_path.exists():
        raise SystemExit(
            f"missing {train_path.name}; run: python scripts/02_aggregate.py "
            f"--feature-set {feature_set}")
    train = pd.read_parquet(train_path)
    test = pd.read_parquet(test_path)
    feat_cols = [c for c in train.columns if c not in (C.ID_COL, C.TARGET_COL)]
    test = test.reindex(columns=[C.ID_COL] + feat_cols, fill_value=0.0)
    return train, test, feat_cols


def main() -> None:
    ap = argparse.ArgumentParser()
    ap.add_argument("--config", required=True)
    args = ap.parse_args()

    cfg = RunConfig.from_yaml(args.config)
    run_id = tracking.now_run_id(cfg.name)
    run_dir = C.ARTIFACTS / run_id
    run_dir.mkdir(parents=True, exist_ok=True)
    shutil.copy(args.config, run_dir / "config.yaml")

    train, test, feat_cols = _load_features(cfg.feature_set)
    X = train[feat_cols].to_numpy("float32")
    y = train[C.TARGET_COL].to_numpy("int8")
    ids = train[C.ID_COL]
    X_test = test[feat_cols].to_numpy("float32")
    print(f"[train] {X.shape[0]:,} ids x {len(feat_cols)} features; pos rate {y.mean():.4f}")

    # id-level stratified folds (one row per id -> no row leakage).
    folds = cv.make_folds(ids, pd.Series(y), n_folds=cfg.n_folds, seed=cfg.seed)

    trainer = {"lgbm": lgbm, "catboost": catboost_model}.get(cfg.model)
    if trainer is None:
        raise SystemExit(f"model '{cfg.model}' not supported by 03_train.py")
    oof, test_pred, _, best_iters = trainer.train_cv(
        X, y, folds, X_test, feat_cols,
        params=cfg.params, num_boost_round=cfg.num_boost_round,
        early_stopping_rounds=cfg.early_stopping_rounds, seed=cfg.seed,
    )

    rep = metrics.fold_report(y, oof, folds)
    print(f"[train] OOF ROC-AUC={rep['oof_roc_auc']:.5f}  PR-AUC={rep['oof_pr_auc']:.5f}  "
          f"fold mean±std={rep['fold_roc_mean']:.5f}±{rep['fold_roc_std']:.5f}")

    # Temporal cross-check: train on earliest 1-frac ids, evaluate on latest frac.
    th = cv.time_holdout_mask(ids, frac=cfg.time_holdout_frac)
    th_fold = np.where(th, 0, 1).astype("int8")  # fold 0 = holdout (validation)
    oof_th, _, _, _ = trainer.train_cv(
        X, y, th_fold, X_test[:1], feat_cols,
        params=cfg.params, num_boost_round=cfg.num_boost_round,
        early_stopping_rounds=cfg.early_stopping_rounds, seed=cfg.seed,
    )
    time_auc = metrics.roc_auc(y[th], oof_th[th])
    print(f"[train] time-holdout ROC-AUC (latest {cfg.time_holdout_frac:.0%} ids)={time_auc:.5f} "
          f"(random-CV OOF={rep['oof_roc_auc']:.5f}; gap={rep['oof_roc_auc']-time_auc:+.5f})")

    rep["time_holdout_roc_auc"] = time_auc
    rep["best_iters"] = best_iters
    np.save(run_dir / "folds.npy", folds)
    np.save(run_dir / "oof.npy", oof)
    np.save(run_dir / "test_pred.npy", test_pred)
    np.save(run_dir / "test_ids.npy", test[C.ID_COL].to_numpy("int32"))
    tracking.write_metrics(run_id, rep)

    tracking.append_log({
        "run_id": run_id, "timestamp": run_id[:15], "data_hash": "",
        "split": f"skf{cfg.n_folds}+timeholdout", "seed": cfg.seed,
        "feature_set": cfg.feature_set, "model": cfg.model,
        "params_ref": str(run_dir / "config.yaml"),
        "fold_auc": f"{rep['fold_roc_mean']:.5f}±{rep['fold_roc_std']:.5f}",
        "oof_auc": f"{rep['oof_roc_auc']:.5f}", "test_pred": str(run_dir / "test_pred.npy"),
        "submission": "", "notes": cfg.notes,
    })
    print(f"[train] run_id={run_id}  artifacts in {run_dir}")


if __name__ == "__main__":
    main()
