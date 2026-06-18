"""IO helpers: read long-format parquet with memory-saving dtypes, read target/sample.

All 61 feature columns are small non-negative integers, so we downcast them to
int8/int16; `id` needs int32 (max ~3.0M), `rn` is <=55 -> int8. This shrinks the
train frame from ~9 GB to ~1.5 GB in memory.
"""
from __future__ import annotations

import hashlib
from pathlib import Path

import pandas as pd
import pyarrow as pa
import pyarrow.parquet as pq

from . import config as C


def _cast_arrow(table: pa.Table) -> pa.Table:
    """Cast columns to small int types at the Arrow level, before materializing to
    pandas. This avoids the transient int64 memory spike on the 18.3M-row train
    frame (id->int32, everything else fits in int16: rn<=55, features 0..19)."""
    fields = []
    arrays = []
    for name in table.column_names:
        col = table.column(name)
        if name == C.ID_COL:
            target = pa.int32()
        elif name == C.TARGET_COL:
            target = pa.int8()
        else:
            target = pa.int16()
        arrays.append(col.cast(target))
        fields.append(pa.field(name, target))
    return pa.table(arrays, schema=pa.schema(fields))


def read_history(path: str | Path, nrows: int | None = None) -> pd.DataFrame:
    """Read a long-format credit-history parquet (train_data/test_data), cast to
    small int dtypes at the Arrow level, and return a pandas frame.

    If `nrows` is given, only the leading `nrows` rows are kept (smoke testing).
    """
    table = pq.read_table(str(path))
    if nrows is not None:
        table = table.slice(0, nrows)
    table = _cast_arrow(table)
    return table.to_pandas()


def read_history_for_ids(path: str | Path, ids: set[int]) -> pd.DataFrame:
    """Read history rows only for a subset of `id`s (used for smoke runs)."""
    table = pq.read_table(str(path), filters=[(C.ID_COL, "in", list(ids))])
    return _cast_arrow(table).to_pandas()


def read_target() -> pd.DataFrame:
    df = pd.read_csv(C.TRAIN_TARGET)
    df[C.ID_COL] = df[C.ID_COL].astype("int32")
    df[C.TARGET_COL] = df[C.TARGET_COL].astype("int8")
    return df


def read_sample_submission() -> pd.DataFrame:
    df = pd.read_csv(C.SAMPLE_SUBMISSION)
    df[C.ID_COL] = df[C.ID_COL].astype("int32")
    return df


def sha256(path: str | Path) -> str:
    h = hashlib.sha256()
    with open(path, "rb") as fh:
        for chunk in iter(lambda: fh.read(1 << 20), b""):
            h.update(chunk)
    return h.hexdigest()
