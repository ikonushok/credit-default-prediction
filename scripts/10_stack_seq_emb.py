"""CLI: embedding-stacking — train the bi-GRU, export its leak-safe penultimate
representation per id (OOF + averaged test), then fit a LightGBM meta-learner on
those embeddings over the SAME id-level folds.

Hypothesis: the 128-d learned representation carries non-linear signal that the
scalar logit (and hence the linear rank-blend) discards. The meta-prediction is a
new base to rank-blend into the existing 5-way ensemble (07_ensemble).

Leakage: oof_emb[i] comes from the seq model that did NOT train on fold(i); the
LightGBM meta then trains per fold on other folds' embeddings only — standard
OOF-stacking, no fold-k label ever touches a fold-k prediction.

Usage:
    python scripts/10_stack_seq_emb.py --config configs/sequence_emb.yaml [--nrows N]
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
from credit_scoring.models import lgbm


def main() -> None:
    ap = argparse.ArgumentParser()
    ap.add_argument("--config", default="configs/sequence_emb.yaml")
    ap.add_argument("--nrows", type=int, default=None, help="smoke: read first N rows")
    ap.add_argument("--model-seed", type=int, default=None)
    args = ap.parse_args()

    cfg = RunConfig.from_yaml(args.config)
    model_seed = args.model_seed if args.model_seed is not None else cfg.seed
    run_id = tracking.now_run_id("stack_seq_emb")
    run_dir = C.ARTIFACTS / run_id
    run_dir.mkdir(parents=True, exist_ok=True)
    shutil.copy(args.config, run_dir / "config.yaml")

    feature_cols = [c for c in
                    data_io.read_history(C.TRAIN_DATA, nrows=1).columns
                    if c not in (C.ID_COL, C.RN_COL)]

    print("[stack] loading raw history…")
    train_df = data_io.read_history(C.TRAIN_DATA, nrows=args.nrows)
    test_df = data_io.read_history(C.TEST_DATA, nrows=args.nrows)
    train_store = seq.SeqStore(train_df, feature_cols)
    test_store = seq.SeqStore(test_df, feature_cols)
    del train_df, test_df

    target = data_io.read_target().set_index(C.ID_COL)[C.TARGET_COL]
    y = target.reindex(train_store.uniq_ids).to_numpy("int8")
    assert not np.isnan(y).any(), "some train id has no target"
    test_ids = test_store.uniq_ids.copy()
    print(f"[stack] {len(train_store.uniq_ids):,} train ids, {len(test_ids):,} test ids; "
          f"device={seq.get_device()}; pos rate {y.mean():.4f}")

    folds = cv.make_folds(pd.Series(train_store.uniq_ids), pd.Series(y),
                          n_folds=cfg.n_folds, seed=cfg.seed)

    # --- 1. Train bi-GRU and export leak-safe embeddings ----------------------
    oof, test_pred, per_fold, oof_emb, test_emb = seq.train_cv(
        train_store, y, folds, test_store, params=cfg.params, seed=model_seed,
        collect_repr=True)

    seq_auc = metrics.roc_auc(y, oof)
    print(f"[stack] bi-GRU scalar OOF ROC-AUC={seq_auc:.5f} (sanity vs ~0.77982); "
          f"emb dim={oof_emb.shape[1]}")
    np.save(run_dir / "folds.npy", folds)
    np.save(run_dir / "test_ids.npy", test_ids)
    np.save(run_dir / "oof_emb.npy", oof_emb)
    np.save(run_dir / "test_emb.npy", test_emb.astype("float16"))
    np.save(run_dir / "seq_oof.npy", oof)
    np.save(run_dir / "seq_test_pred.npy", test_pred)

    del train_store, test_store  # free SeqStore (~3GB) before the GBDT meta

    # --- 2. LightGBM meta-learner on the embeddings (same folds) --------------
    X = oof_emb.astype("float32")
    X_test = test_emb.astype("float32")
    feat_names = [f"e{i}" for i in range(X.shape[1])]
    print(f"[stack] training LightGBM meta on {X.shape} embeddings…")
    meta_oof, meta_test, _, best_iters = lgbm.train_cv(
        X, y.astype("int32"), folds, X_test, feat_names, seed=cfg.seed)

    rep = metrics.fold_report(y, meta_oof, folds)
    rep["seq_scalar_oof_auc"] = float(seq_auc)
    rep["meta_best_iters"] = [int(b) for b in best_iters]
    print(f"[stack] META OOF ROC-AUC={rep['oof_roc_auc']:.5f}  PR-AUC={rep['oof_pr_auc']:.5f}  "
          f"fold mean±std={rep['fold_roc_mean']:.5f}±{rep['fold_roc_std']:.5f}")
    print(f"[stack] meta vs seq scalar: {rep['oof_roc_auc']:.5f} vs {seq_auc:.5f} "
          f"(Δ={rep['oof_roc_auc'] - seq_auc:+.5f})")

    # Save meta predictions in the standard base-model layout so 07_ensemble can
    # rank-blend this as a new base.
    np.save(run_dir / "oof.npy", meta_oof.astype("float32"))
    np.save(run_dir / "test_pred.npy", meta_test.astype("float32"))
    tracking.write_metrics(run_id, rep)
    tracking.append_log({
        "run_id": run_id, "timestamp": run_id[:15], "data_hash": "",
        "split": f"skf{cfg.n_folds}", "seed": cfg.seed, "feature_set": "seq_embeddings",
        "model": "lgbm_meta_on_seq_emb", "params_ref": str(run_dir / "config.yaml"),
        "fold_auc": f"{rep['fold_roc_mean']:.5f}±{rep['fold_roc_std']:.5f}",
        "oof_auc": f"{rep['oof_roc_auc']:.5f}", "test_pred": str(run_dir / "test_pred.npy"),
        "submission": "", "notes": f"embedding-stacking: LightGBM on bi-GRU {X.shape[1]}-d repr",
    })
    print(f"[stack] run_id={run_id}  artifacts in {run_dir}")


if __name__ == "__main__":
    main()
