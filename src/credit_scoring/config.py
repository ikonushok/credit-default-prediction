"""Project paths and config loading.

Paths are resolved relative to the repository root so the pipeline is
machine-independent (no absolute paths baked into configs).
"""
from __future__ import annotations

import os
from dataclasses import dataclass, field
from pathlib import Path
from typing import Any

import yaml

# Repo root = two levels up from this file (src/credit_scoring/config.py).
ROOT = Path(__file__).resolve().parents[2]

DATA_RAW = ROOT / "data" / "raw"
DATA_INTERIM = ROOT / "data" / "interim"
DATA_PROCESSED = ROOT / "data" / "processed"
SUBMISSIONS = ROOT / "submissions"  # always at project root
EXPERIMENTS = ROOT / "experiments"
CARDS = EXPERIMENTS / "cards"
ARTIFACTS = ROOT / "artifacts"  # trained models / OOF / fold ids per run

TRAIN_DATA = DATA_RAW / "train_data.parquet"
TEST_DATA = DATA_RAW / "test_data.parquet"
TRAIN_TARGET = DATA_RAW / "train_target.csv"
SAMPLE_SUBMISSION = ROOT / "data" / "Кредитный скорринг" / "sample_submission.csv"

ID_COL = "id"
RN_COL = "rn"
TARGET_COL = "flag"
SEED = 42

# Column groups (the 61 feature columns are all integer-encoded).
ENC_PAYM_COLS = [f"enc_paym_{i}" for i in range(25)]
DELINQ_COLS = ["pre_loans5", "pre_loans530", "pre_loans3060", "pre_loans6090", "pre_loans90"]
IS_ZERO_LOANS_COLS = [
    "is_zero_loans5", "is_zero_loans530", "is_zero_loans3060",
    "is_zero_loans6090", "is_zero_loans90",
]
CATEGORICAL_COLS = [
    "enc_loans_account_holder_type", "enc_loans_credit_status",
    "enc_loans_credit_type", "enc_loans_account_cur",
]
FLAG_COLS = [
    "is_zero_util", "is_zero_over2limit", "is_zero_maxover2limit",
    "pclose_flag", "fclose_flag",
] + IS_ZERO_LOANS_COLS
# Ordinal binned amount / term / utilization columns (means/max/last useful).
ORDINAL_COLS = [
    "pre_since_opened", "pre_since_confirmed", "pre_pterm", "pre_fterm",
    "pre_till_pclose", "pre_till_fclose", "pre_loans_credit_limit",
    "pre_loans_next_pay_summ", "pre_loans_outstanding", "pre_loans_total_overdue",
    "pre_loans_max_overdue_sum", "pre_loans_credit_cost_rate",
    "pre_util", "pre_over2limit", "pre_maxover2limit",
]


def ensure_dirs() -> None:
    for d in (DATA_INTERIM, DATA_PROCESSED, SUBMISSIONS, EXPERIMENTS, CARDS, ARTIFACTS):
        d.mkdir(parents=True, exist_ok=True)


def features_path(split: str, feature_set: str) -> Path:
    """Versioned per-id feature table, e.g. data/processed/train_features__v2.parquet.

    Versioning lets different feature sets coexist on disk for fair ablation on
    identical folds. `split` is "train" or "test".
    """
    return DATA_PROCESSED / f"{split}_features__{feature_set}.parquet"


@dataclass
class RunConfig:
    """Resolved config for a training run, loaded from a YAML file."""

    model: str = "lgbm"
    feature_set: str = "baseline"
    seed: int = SEED
    n_folds: int = 5
    time_holdout_frac: float = 0.2
    params: dict[str, Any] = field(default_factory=dict)
    early_stopping_rounds: int = 200
    num_boost_round: int = 5000
    notes: str = ""
    name: str = "lgbm_baseline"

    @classmethod
    def from_yaml(cls, path: str | os.PathLike) -> "RunConfig":
        with open(path, "r", encoding="utf-8") as fh:
            raw = yaml.safe_load(fh) or {}
        known = {f for f in cls.__dataclass_fields__}  # type: ignore[attr-defined]
        kwargs = {k: v for k, v in raw.items() if k in known}
        cfg = cls(**kwargs)
        if "name" not in raw:
            cfg.name = Path(path).stem
        return cfg
