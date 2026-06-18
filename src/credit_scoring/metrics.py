"""Metric helpers. ROC-AUC is the competition metric; PR-AUC is a sanity check
given the ~3.55% positive rate."""
from __future__ import annotations

import numpy as np
from sklearn.metrics import average_precision_score, roc_auc_score


def roc_auc(y_true, y_score) -> float:
    return float(roc_auc_score(y_true, y_score))


def pr_auc(y_true, y_score) -> float:
    return float(average_precision_score(y_true, y_score))


def fold_report(y_true: np.ndarray, oof: np.ndarray, folds: np.ndarray) -> dict:
    """Per-fold and aggregate OOF ROC-AUC / PR-AUC."""
    per_fold = []
    for k in sorted(np.unique(folds)):
        m = folds == k
        per_fold.append({
            "fold": int(k),
            "n": int(m.sum()),
            "roc_auc": roc_auc(y_true[m], oof[m]),
            "pr_auc": pr_auc(y_true[m], oof[m]),
        })
    return {
        "per_fold": per_fold,
        "oof_roc_auc": roc_auc(y_true, oof),
        "oof_pr_auc": pr_auc(y_true, oof),
        "fold_roc_mean": float(np.mean([f["roc_auc"] for f in per_fold])),
        "fold_roc_std": float(np.std([f["roc_auc"] for f in per_fold])),
    }
