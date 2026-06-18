"""CLI: blend several runs' OOF/test predictions by rank-weighted averaging.

Usage:
    python scripts/07_ensemble.py --runs <run_a> <run_b> [<run_c> ...] [--name ens]

Verifies folds are identical across runs (fair blend), rank-normalizes each
model's predictions (scales differ: GBDT proba vs NN sigmoid), and searches
non-negative weights summing to 1 that maximize OOF ROC-AUC. Writes an artifacts
run dir so 04_predict_submit.py can build the submission from it.
"""
import argparse
import json

import numpy as np
import pandas as pd
from scipy.optimize import minimize
from scipy.stats import rankdata

import _bootstrap  # noqa: F401

from credit_scoring import config as C
from credit_scoring import metrics, tracking


def _rank01(s: np.ndarray) -> np.ndarray:
    return rankdata(s) / len(s)


def main() -> None:
    ap = argparse.ArgumentParser()
    ap.add_argument("--runs", nargs="+", required=True)
    ap.add_argument("--name", default="ensemble")
    args = ap.parse_args()

    y = pd.read_parquet(C.DATA_PROCESSED / "train_features.parquet")[C.TARGET_COL].to_numpy("int8")

    # All runs store predictions in ascending-id order, so OOF/test align by
    # position. Blending OOF only requires that each model's predictions are valid
    # out-of-fold over the same id set — identical fold *partitions* are not needed
    # (we warn if they differ, since weight-fitting is then mildly optimistic).
    oofs, tests, folds0, test_ids0, n_train = [], [], None, None, None
    for r in args.runs:
        d = C.ARTIFACTS / r
        f = np.load(d / "folds.npy")
        ti = np.load(d / "test_ids.npy")
        oof = np.load(d / "oof.npy")
        if folds0 is None:
            folds0, test_ids0, n_train = f, ti, len(oof)
        else:
            if not np.array_equal(ti, test_ids0):
                raise SystemExit(f"test_ids of {r} differ — cannot align")
            if len(oof) != n_train:
                raise SystemExit(f"oof length of {r} differs — cannot align")
            if not np.array_equal(f, folds0):
                print(f"WARNING: folds of {r} differ from base — OOF weights may be mildly optimistic")
        oofs.append(_rank01(oof))
        tests.append(_rank01(np.load(d / "test_pred.npy")))

    O = np.vstack(oofs)            # (m, n_train)
    T = np.vstack(tests)           # (m, n_test)
    m = O.shape[0]

    print("individual OOF ROC-AUC:")
    for r, o in zip(args.runs, oofs):
        print(f"  {r}: {metrics.roc_auc(y, o):.5f}")

    def neg_auc(theta):
        w = np.exp(theta - theta.max()); w /= w.sum()
        return -metrics.roc_auc(y, w @ O)

    res = minimize(neg_auc, np.zeros(m), method="Nelder-Mead",
                   options={"xatol": 1e-4, "fatol": 1e-6, "maxiter": 2000})
    w = np.exp(res.x - res.x.max()); w /= w.sum()
    ens_oof = w @ O
    ens_test = w @ T
    auc = metrics.roc_auc(y, ens_oof)
    print("\nweights:", {r: round(float(wi), 3) for r, wi in zip(args.runs, w)})
    print(f"ensemble OOF ROC-AUC = {auc:.5f}  PR-AUC = {metrics.pr_auc(y, ens_oof):.5f}")
    best_single = max(metrics.roc_auc(y, o) for o in oofs)
    print(f"best single = {best_single:.5f}  gain = {auc - best_single:+.5f}")

    run_id = tracking.now_run_id(args.name)
    d = C.ARTIFACTS / run_id
    d.mkdir(parents=True, exist_ok=True)
    np.save(d / "folds.npy", folds0)
    np.save(d / "oof.npy", ens_oof)
    np.save(d / "test_pred.npy", ens_test)
    np.save(d / "test_ids.npy", test_ids0)
    rep = metrics.fold_report(y, ens_oof, folds0)
    rep["members"] = args.runs
    rep["weights"] = {r: float(wi) for r, wi in zip(args.runs, w)}
    tracking.write_metrics(run_id, rep)
    tracking.append_log({
        "run_id": run_id, "timestamp": run_id[:15], "split": "skf5",
        "model": f"ensemble({'+'.join(args.runs)})", "feature_set": "rank_blend",
        "fold_auc": f"{rep['fold_roc_mean']:.5f}±{rep['fold_roc_std']:.5f}",
        "oof_auc": f"{auc:.5f}", "test_pred": str(d / "test_pred.npy"),
        "notes": f"weights={rep['weights']}",
    })
    print(f"\nensemble run_id={run_id}  ->  python scripts/04_predict_submit.py --run {run_id}")


if __name__ == "__main__":
    main()
