"""CLI: lightweight EDA / data-quality report -> experiments/eda_report.md.

Usage:
    python scripts/01_eda.py [--nrows N]

Checks: target distribution, rows-per-id, id-set alignment, train/test id-range
drift, enc_loans_* category frequencies, payment-status badness. No plots (text
report) to keep it dependency-light and fast.
"""
import argparse

import numpy as np
import pandas as pd

import _bootstrap  # noqa: F401

from credit_scoring import config as C
from credit_scoring import data_io


def main() -> None:
    ap = argparse.ArgumentParser()
    ap.add_argument("--nrows", type=int, default=None)
    args = ap.parse_args()
    C.ensure_dirs()

    target = data_io.read_target()
    sample = data_io.read_sample_submission()
    train = data_io.read_history(C.TRAIN_DATA, nrows=args.nrows)
    test = data_io.read_history(C.TEST_DATA, nrows=args.nrows)

    tr_ids = set(train[C.ID_COL].unique())
    te_ids = set(test[C.ID_COL].unique())
    rows_per_id = train.groupby(C.ID_COL).size()

    lines = ["# EDA / data-quality report", ""]
    lines.append(f"- train_data: {len(train):,} rows, {len(tr_ids):,} ids")
    lines.append(f"- test_data : {len(test):,} rows, {len(te_ids):,} ids")
    lines.append(f"- positive rate (flag=1): {target[C.TARGET_COL].mean():.4f} "
                 f"({int(target[C.TARGET_COL].sum()):,}/{len(target):,})")
    lines.append(f"- rows per id (train): mean {rows_per_id.mean():.2f}, "
                 f"min {rows_per_id.min()}, max {rows_per_id.max()}")
    if args.nrows is None:
        lines.append(f"- train ids == target ids: {tr_ids == set(target[C.ID_COL])}")
        lines.append(f"- test ids == submission ids: {te_ids == set(sample[C.ID_COL])}")
    lines.append(f"- train/test id overlap: {len(tr_ids & te_ids)}")
    lines.append(f"- train id range: [{train[C.ID_COL].min()}, {train[C.ID_COL].max()}]")
    lines.append(f"- test  id range: [{test[C.ID_COL].min()}, {test[C.ID_COL].max()}] "
                 f"(temporal split expected: test later)")

    lines.append("\n## enc_loans_* category frequencies (train)")
    for col in C.CATEGORICAL_COLS:
        vc = train[col].value_counts(normalize=True).sort_index()
        lines.append(f"- {col}: " + ", ".join(f"{int(k)}:{v:.3f}" for k, v in vc.items()))

    lines.append("\n## payment status (enc_paym_*) overall mean per slot (train)")
    paym_means = train[C.ENC_PAYM_COLS].mean()
    lines.append("- " + ", ".join(f"{c.split('_')[-1]}:{v:.2f}" for c, v in paym_means.items()))

    out = C.EXPERIMENTS / "eda_report.md"
    out.write_text("\n".join(lines) + "\n", encoding="utf-8")
    print("\n".join(lines))
    print(f"\n[eda] wrote {out}")


if __name__ == "__main__":
    main()
