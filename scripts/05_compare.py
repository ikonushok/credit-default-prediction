"""CLI: compare two training runs' OOF on identical folds.

Usage:
    python scripts/05_compare.py --base <run_id_a> --cand <run_id_b>

Verifies the fold assignments are identical (fair comparison), then reports
per-fold and overall OOF ROC-AUC / PR-AUC deltas. Use to decide whether a new
feature set / model is a real improvement vs fold noise.
"""
import argparse
import json

import numpy as np
import pandas as pd

import _bootstrap  # noqa: F401

from credit_scoring import config as C
from credit_scoring import metrics


def _load(run_id):
    d = C.ARTIFACTS / run_id
    return (np.load(d / "folds.npy"), np.load(d / "oof.npy"),
            json.loads((d / "metrics.json").read_text()))


def main() -> None:
    ap = argparse.ArgumentParser()
    ap.add_argument("--base", required=True)
    ap.add_argument("--cand", required=True)
    args = ap.parse_args()

    # Target order matches the sorted-id feature frame both runs trained on.
    y = pd.read_parquet(C.DATA_PROCESSED / "train_features.parquet")[C.TARGET_COL].to_numpy("int8")

    fb, ob, mb = _load(args.base)
    fc, oc, mc = _load(args.cand)
    if not np.array_equal(fb, fc):
        raise SystemExit("fold assignments differ — comparison would be unfair")

    rb = metrics.fold_report(y, ob, fb)
    rc = metrics.fold_report(y, oc, fc)
    print(f"{'':<10}{'base':>12}{'cand':>12}{'delta':>12}")
    print(f"{'OOF AUC':<10}{rb['oof_roc_auc']:>12.5f}{rc['oof_roc_auc']:>12.5f}"
          f"{rc['oof_roc_auc'] - rb['oof_roc_auc']:>+12.5f}")
    print(f"{'OOF PR':<10}{rb['oof_pr_auc']:>12.5f}{rc['oof_pr_auc']:>12.5f}"
          f"{rc['oof_pr_auc'] - rb['oof_pr_auc']:>+12.5f}")
    print(f"{'fold std':<10}{rb['fold_roc_std']:>12.5f}{rc['fold_roc_std']:>12.5f}")
    print("\nper-fold AUC delta:")
    for pb, pc in zip(rb["per_fold"], rc["per_fold"]):
        print(f"  fold {pb['fold']}: {pc['roc_auc'] - pb['roc_auc']:+.5f}")
    gain = rc["oof_roc_auc"] - rb["oof_roc_auc"]
    noise = max(rb["fold_roc_std"], rc["fold_roc_std"])
    verdict = "REAL improvement" if gain > noise else "within fold noise — RETEST"
    print(f"\nverdict: gain {gain:+.5f} vs fold-std {noise:.5f} -> {verdict}")


if __name__ == "__main__":
    main()
