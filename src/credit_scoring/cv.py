"""Validation splits, defined at the `id` level.

After aggregation there is exactly one row per `id`, so a StratifiedKFold over
the aggregated frame cannot leak history rows across folds. A separate
time-ordered holdout (later `id`s = later applications) mirrors the platform's
temporal train/test split and is used as a cross-check against random CV.
"""
from __future__ import annotations

import numpy as np
import pandas as pd
from sklearn.model_selection import StratifiedKFold

from . import config as C


def make_folds(ids: pd.Series, y: pd.Series, n_folds: int = 5, seed: int = C.SEED) -> np.ndarray:
    """Return an int array of fold assignments (0..n_folds-1), aligned to `ids`."""
    skf = StratifiedKFold(n_splits=n_folds, shuffle=True, random_state=seed)
    fold = np.full(len(ids), -1, dtype="int8")
    for k, (_, val_idx) in enumerate(skf.split(ids, y)):
        fold[val_idx] = k
    assert (fold >= 0).all(), "every id must be assigned to a fold"
    return fold


def time_holdout_mask(ids: pd.Series, frac: float = 0.2) -> np.ndarray:
    """Boolean mask: True for the latest `frac` of ids (by id order = by time)."""
    order = ids.rank(method="first")
    threshold = order.quantile(1.0 - frac)
    return (order > threshold).to_numpy()
