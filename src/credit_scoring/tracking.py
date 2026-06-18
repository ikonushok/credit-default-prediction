"""Experiment log and submission-card writers (project memory / evidence layer)."""
from __future__ import annotations

import csv
import datetime as dt
import json
from pathlib import Path

from . import config as C

LOG_COLUMNS = [
    "run_id", "timestamp", "data_hash", "split", "seed", "feature_set",
    "model", "params_ref", "fold_auc", "oof_auc", "test_pred", "submission", "notes",
]


def now_run_id(name: str) -> str:
    stamp = dt.datetime.now().strftime("%Y%m%d_%H%M%S")
    return f"{stamp}_{name}"


def append_log(row: dict) -> None:
    C.ensure_dirs()
    path = C.EXPERIMENTS / "experiment_log.csv"
    exists = path.exists()
    with open(path, "a", newline="", encoding="utf-8") as fh:
        w = csv.DictWriter(fh, fieldnames=LOG_COLUMNS)
        if not exists:
            w.writeheader()
        w.writerow({k: row.get(k, "") for k in LOG_COLUMNS})


def write_metrics(run_id: str, metrics: dict) -> Path:
    run_dir = C.ARTIFACTS / run_id
    run_dir.mkdir(parents=True, exist_ok=True)
    path = run_dir / "metrics.json"
    with open(path, "w", encoding="utf-8") as fh:
        json.dump(metrics, fh, indent=2)
    return path


def write_submission_card(run_id: str, card: dict) -> Path:
    C.ensure_dirs()
    path = C.CARDS / f"{run_id}.md"
    lines = [f"# Submission card — {run_id}", ""]
    for k, v in card.items():
        lines.append(f"- **{k}**: {v}")
    path.write_text("\n".join(lines) + "\n", encoding="utf-8")
    return path
