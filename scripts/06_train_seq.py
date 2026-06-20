"""CLI: train the GRU sequence model over raw credit-product history.

Usage:
    python scripts/06_train_seq.py --config configs/sequence.yaml [--nrows N]

Uses the SAME id-level folds (make_folds, seed) as the GBDT runs, so OOF/test
predictions align by id and are directly comparable (05_compare) and
ensemble-able (07_ensemble). Saves to artifacts/<run_id>/ like 03_train.py.
"""
import argparse
import shutil

import numpy as np
import pandas as pd

import _bootstrap  # noqa: F401

from credit_scoring import config as C
from credit_scoring import cv, data_io, metrics, tracking
from credit_scoring.config import RunConfig
from credit_scoring.models import sequence as seq


def main() -> None:
    ap = argparse.ArgumentParser()
    ap.add_argument("--config", required=True)
    ap.add_argument("--nrows", type=int, default=None, help="smoke: read first N rows")
    ap.add_argument("--model-seed", type=int, default=None,
                    help="override model-init seed only; folds stay on cfg.seed "
                         "(for multi-seed averaging on identical folds)")
    args = ap.parse_args()

    cfg = RunConfig.from_yaml(args.config)
    model_seed = args.model_seed if args.model_seed is not None else cfg.seed
    if args.model_seed is not None:
        cfg.name = f"{cfg.name}_s{args.model_seed}"
    run_id = tracking.now_run_id(cfg.name)
    run_dir = C.ARTIFACTS / run_id
    run_dir.mkdir(parents=True, exist_ok=True)
    shutil.copy(args.config, run_dir / "config.yaml")

    feature_cols = [c for c in
                    data_io.read_history(C.TRAIN_DATA, nrows=1).columns
                    if c not in (C.ID_COL, C.RN_COL)]

    print("[seq] loading raw history…")
    train_df = data_io.read_history(C.TRAIN_DATA, nrows=args.nrows)
    test_df = data_io.read_history(C.TEST_DATA, nrows=args.nrows)
    train_store = seq.SeqStore(train_df, feature_cols)
    test_store = seq.SeqStore(test_df, feature_cols)
    del train_df, test_df

    target = data_io.read_target().set_index(C.ID_COL)[C.TARGET_COL]
    y = target.reindex(train_store.uniq_ids).to_numpy("int8")
    assert not np.isnan(y).any(), "some train id has no target"
    print(f"[seq] {len(train_store.uniq_ids):,} train ids, {len(test_store.uniq_ids):,} test ids; "
          f"device={seq.get_device()}; pos rate {y.mean():.4f}")

    folds = cv.make_folds(pd.Series(train_store.uniq_ids), pd.Series(y),
                          n_folds=cfg.n_folds, seed=cfg.seed)

    oof, test_pred, per_fold = seq.train_cv(
        train_store, y, folds, test_store, params=cfg.params, seed=model_seed)

    rep = metrics.fold_report(y, oof, folds)
    rep["per_fold_best_val_auc"] = per_fold
    print(f"[seq] OOF ROC-AUC={rep['oof_roc_auc']:.5f}  PR-AUC={rep['oof_pr_auc']:.5f}  "
          f"fold mean±std={rep['fold_roc_mean']:.5f}±{rep['fold_roc_std']:.5f}")

    np.save(run_dir / "folds.npy", folds)
    np.save(run_dir / "oof.npy", oof)
    np.save(run_dir / "test_pred.npy", test_pred)
    np.save(run_dir / "test_ids.npy", test_store.uniq_ids)
    tracking.write_metrics(run_id, rep)
    tracking.append_log({
        "run_id": run_id, "timestamp": run_id[:15], "data_hash": "",
        "split": f"skf{cfg.n_folds}", "seed": cfg.seed, "feature_set": "raw_sequence",
        "model": cfg.model, "params_ref": str(run_dir / "config.yaml"),
        "fold_auc": f"{rep['fold_roc_mean']:.5f}±{rep['fold_roc_std']:.5f}",
        "oof_auc": f"{rep['oof_roc_auc']:.5f}", "test_pred": str(run_dir / "test_pred.npy"),
        "submission": "", "notes": cfg.notes,
    })
    print(f"[seq] run_id={run_id}  artifacts in {run_dir}")


if __name__ == "__main__":
    main()
