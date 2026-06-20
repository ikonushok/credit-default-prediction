"""CLI: average several same-architecture seed runs into one lower-variance base.

Usage:
    python scripts/09_avg_seeds.py --runs <run_seed42> <run_seed101> <run_seed202> \
        [--name sequence_emb_bigru_avg] [--rank]

All runs must share the SAME folds (identical make_folds seed) so each OOF is a
valid out-of-fold prediction on the same partition; averaging then reduces model
variance without leakage. test predictions average by id. Writes a standard
artifacts run (folds/oof/test_pred/test_ids/metrics) so it drops into 07_ensemble
and 04_predict_submit unchanged.

Default averaging is plain probability mean (same model → same scale). Use --rank
to average rank-normalized predictions instead (scale-robust, matches the blend).
"""
import argparse

import numpy as np
import pandas as pd
from scipy.stats import rankdata

import _bootstrap  # noqa: F401

from credit_scoring import config as C
from credit_scoring import metrics, tracking


def _rank01(s: np.ndarray) -> np.ndarray:
    return rankdata(s) / len(s)


def main() -> None:
    ap = argparse.ArgumentParser()
    ap.add_argument("--runs", nargs="+", required=True, help="seed run_ids to average")
    ap.add_argument("--name", default="sequence_emb_bigru_avg")
    ap.add_argument("--rank", action="store_true", help="average rank-normalized preds")
    args = ap.parse_args()

    y = pd.read_parquet(C.DATA_PROCESSED / "train_features.parquet")[C.TARGET_COL].to_numpy("int8")

    folds0 = test_ids0 = None
    oof_sum = test_sum = None
    print("individual OOF ROC-AUC:")
    for r in args.runs:
        d = C.ARTIFACTS / r
        f = np.load(d / "folds.npy")
        ti = np.load(d / "test_ids.npy")
        oof = np.load(d / "oof.npy")
        tp = np.load(d / "test_pred.npy")
        if folds0 is None:
            folds0, test_ids0 = f, ti
            oof_sum = np.zeros_like(oof, dtype="float64")
            test_sum = np.zeros_like(tp, dtype="float64")
        else:
            if not np.array_equal(f, folds0):
                raise SystemExit(f"folds of {r} differ — seeds must share folds to average OOF")
            if not np.array_equal(ti, test_ids0):
                raise SystemExit(f"test_ids of {r} differ — cannot align")
        print(f"  {r}: {metrics.roc_auc(y, oof):.5f}")
        oof_sum += _rank01(oof) if args.rank else oof
        test_sum += _rank01(tp) if args.rank else tp

    n = len(args.runs)
    oof_avg = (oof_sum / n).astype("float32")
    test_avg = (test_sum / n).astype("float32")

    auc = metrics.roc_auc(y, oof_avg)
    best_single = max(metrics.roc_auc(y, np.load(C.ARTIFACTS / r / "oof.npy")) for r in args.runs)
    print(f"\naveraged ({n} seeds, {'rank' if args.rank else 'proba'}) OOF ROC-AUC = {auc:.5f}"
          f"  PR-AUC = {metrics.pr_auc(y, oof_avg):.5f}")
    print(f"best single = {best_single:.5f}  gain = {auc - best_single:+.5f}")

    run_id = tracking.now_run_id(args.name)
    d = C.ARTIFACTS / run_id
    d.mkdir(parents=True, exist_ok=True)
    np.save(d / "folds.npy", folds0)
    np.save(d / "oof.npy", oof_avg)
    np.save(d / "test_pred.npy", test_avg)
    np.save(d / "test_ids.npy", test_ids0)
    rep = metrics.fold_report(y, oof_avg, folds0)
    rep["seeds"] = args.runs
    rep["averaging"] = "rank" if args.rank else "proba"
    tracking.write_metrics(run_id, rep)
    tracking.append_log({
        "run_id": run_id, "timestamp": run_id[:15], "split": "skf5",
        "feature_set": "raw_sequence", "model": f"avg_seeds({'+'.join(args.runs)})",
        "fold_auc": f"{rep['fold_roc_mean']:.5f}±{rep['fold_roc_std']:.5f}",
        "oof_auc": f"{auc:.5f}", "test_pred": str(d / "test_pred.npy"),
        "notes": f"mean of {n} seeds ({'rank' if args.rank else 'proba'}); same folds",
    })
    print(f"\navg run_id={run_id}  ->  use in 07_ensemble / 04_predict_submit")


if __name__ == "__main__":
    main()
