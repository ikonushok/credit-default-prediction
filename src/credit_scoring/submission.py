"""Build and validate the submission CSV against the sample_submission contract.

Contract: columns exactly [id, flag]; one row per sample_submission id (900 000);
ids match the sample exactly; flag is a float probability in [0, 1]; no NaN/inf.
Row order follows sample_submission.
"""
from __future__ import annotations

from pathlib import Path

import numpy as np
import pandas as pd

from . import config as C
from . import data_io


def build_submission(test_ids: pd.Series, test_pred: np.ndarray, out_path: Path) -> dict:
    """Map predictions to test ids, reorder to sample_submission, validate, write."""
    sample = data_io.read_sample_submission()
    pred = pd.DataFrame({C.ID_COL: pd.Series(test_ids).astype("int32").to_numpy(),
                         C.TARGET_COL: np.asarray(test_pred, dtype="float64")})

    # Reorder/align to the sample submission ids.
    sub = sample[[C.ID_COL]].merge(pred, on=C.ID_COL, how="left")

    _validate(sub, sample)

    out_path.parent.mkdir(parents=True, exist_ok=True)
    sub.to_csv(out_path, index=False)
    digest = data_io.sha256(out_path)
    print(f"[submission] wrote {out_path}  rows={len(sub):,}  sha256={digest[:12]}…")
    return {"path": str(out_path), "sha256": digest, "rows": int(len(sub))}


def _validate(sub: pd.DataFrame, sample: pd.DataFrame) -> None:
    assert list(sub.columns) == [C.ID_COL, C.TARGET_COL], f"bad columns: {list(sub.columns)}"
    assert len(sub) == len(sample), f"row count {len(sub)} != sample {len(sample)}"
    assert sub[C.ID_COL].tolist() == sample[C.ID_COL].tolist(), "id mismatch/order vs sample"
    vals = sub[C.TARGET_COL].to_numpy()
    assert not np.isnan(vals).any(), "NaN in flag — some test id has no prediction"
    assert np.isfinite(vals).all(), "inf in flag"
    assert vals.min() >= 0.0 and vals.max() <= 1.0, f"flag out of [0,1]: [{vals.min()}, {vals.max()}]"
