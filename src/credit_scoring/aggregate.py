"""Build and persist per-`id` feature tables for train and test.

Output (data/processed/):
  - train_features.parquet : index id, feature columns + `flag` (joined from target)
  - test_features.parquet  : index id, feature columns (no target)

Aggregation is strictly within `id`; the target is joined only for train, by `id`.
"""
from __future__ import annotations

from pathlib import Path

import pandas as pd

from . import config as C
from . import data_io, features


def _align_columns(train_feats: pd.DataFrame, test_feats: pd.DataFrame) -> tuple[pd.DataFrame, pd.DataFrame]:
    """Ensure train/test share the same feature columns (category-count columns
    may differ if a category is absent in one split). Missing -> 0."""
    cols = sorted(set(train_feats.columns) | set(test_feats.columns))
    train_feats = train_feats.reindex(columns=cols, fill_value=0.0).copy()
    test_feats = test_feats.reindex(columns=cols, fill_value=0.0).copy()
    return train_feats, test_feats


def build_split(history_path: Path, feature_set: str, nrows: int | None) -> pd.DataFrame:
    hist = data_io.read_history(history_path, nrows=nrows)
    feats = features.build_id_features(hist, feature_set=feature_set)
    return feats


def run(feature_set: str = "baseline", nrows: int | None = None,
        out_dir: Path | None = None) -> dict[str, Path]:
    C.ensure_dirs()
    out_dir = out_dir or C.DATA_PROCESSED

    train_feats = build_split(C.TRAIN_DATA, feature_set, nrows)
    test_feats = build_split(C.TEST_DATA, feature_set, nrows)
    train_feats, test_feats = _align_columns(train_feats, test_feats)

    # Join target (train only), by id.
    target = data_io.read_target().set_index(C.ID_COL)[C.TARGET_COL]
    train_feats = train_feats.join(target, how="left")
    if train_feats[C.TARGET_COL].isna().any():
        missing = int(train_feats[C.TARGET_COL].isna().sum())
        raise ValueError(f"{missing} train ids have no target label — check id alignment")
    train_feats[C.TARGET_COL] = train_feats[C.TARGET_COL].astype("int8")

    train_path = C.features_path("train", feature_set)
    test_path = C.features_path("test", feature_set)
    train_feats.reset_index().to_parquet(train_path, index=False)
    test_feats.reset_index().to_parquet(test_path, index=False)

    n_feat = len([c for c in train_feats.columns if c != C.TARGET_COL])
    print(f"[aggregate] feature_set='{feature_set}'")
    print(f"[aggregate] train_features: {train_feats.shape[0]:,} ids x {n_feat} features "
          f"(+target), positive rate {train_feats[C.TARGET_COL].mean():.4f}")
    print(f"[aggregate] test_features : {test_feats.shape[0]:,} ids x {n_feat} features")
    print(f"[aggregate] wrote {train_path}")
    print(f"[aggregate] wrote {test_path}")
    return {"train": train_path, "test": test_path}
