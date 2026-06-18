"""CLI: build a validated submission CSV from a trained run's test predictions.

Usage:
    python scripts/04_predict_submit.py --run <run_id>

Reads artifacts/<run_id>/{test_pred.npy,test_ids.npy,metrics.json}, builds and
validates data/processed/submissions/<run_id>.csv, and writes a submission card.
"""
import argparse
import json
from pathlib import Path

import numpy as np
import pandas as pd

import _bootstrap  # noqa: F401

from credit_scoring import config as C
from credit_scoring import data_io, submission, tracking


def main() -> None:
    ap = argparse.ArgumentParser()
    ap.add_argument("--run", required=True, help="run_id under artifacts/")
    args = ap.parse_args()

    run_dir = C.ARTIFACTS / args.run
    test_pred = np.load(run_dir / "test_pred.npy")
    test_ids = np.load(run_dir / "test_ids.npy")
    metrics = json.loads((run_dir / "metrics.json").read_text(encoding="utf-8"))

    out_path = C.SUBMISSIONS / f"{args.run}.csv"
    info = submission.build_submission(pd.Series(test_ids), test_pred, out_path)

    card = {
        "run_id": args.run,
        "submission_path": info["path"],
        "sha256": info["sha256"],
        "rows": info["rows"],
        "oof_roc_auc": metrics.get("oof_roc_auc"),
        "oof_pr_auc": metrics.get("oof_pr_auc"),
        "fold_roc_mean_std": f"{metrics.get('fold_roc_mean'):.5f}±{metrics.get('fold_roc_std'):.5f}",
        "time_holdout_roc_auc": metrics.get("time_holdout_roc_auc"),
        "leakage_checks": "aggregation within id; split on id; target joined by id (train only)",
        "format_checks": "columns [id,flag]; 900000 rows; ids==sample; flag in [0,1]; no NaN/inf",
        "upload_recommendation": "review vs daily 3-upload limit before uploading",
    }
    card_path = tracking.write_submission_card(args.run, card)
    print(f"[submit] submission card -> {card_path}")


if __name__ == "__main__":
    main()
