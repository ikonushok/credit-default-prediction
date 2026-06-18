"""Aggregate long-format credit history to one feature vector per `id`.

Feature engineering is organized as a **registry of named blocks**. A *feature
set* is an ordered list of blocks. To add features (the dedicated phase-2 stage):

  1. write a new ``@block("my_block")`` function returning a frame indexed by id;
  2. register a new feature set in ``FEATURE_SETS`` that reuses the baseline
     blocks plus your new ones — the baseline stays byte-for-byte reproducible;
  3. run ``scripts/02_aggregate.py --feature-set <name>`` then train with a
     config whose ``feature_set`` matches, and compare on identical folds.

All aggregation is computed strictly within each `id` (no cross-id mixing) and is
label-free, so it is identical for train and test.
"""
from __future__ import annotations

from dataclasses import dataclass
from typing import Callable

import pandas as pd

from . import config as C

# Categoricals whose per-category counts are informative enough to one-hot-count.
_COUNT_CATS = ["enc_loans_credit_status", "enc_loans_credit_type", "enc_loans_account_cur"]


@dataclass
class Ctx:
    """Shared inputs handed to every feature block."""
    df: pd.DataFrame                 # raw long history (downcast dtypes)
    g: "pd.core.groupby.DataFrameGroupBy"  # df.groupby(id, sort=True)
    index: pd.Index                  # sorted unique id index


BlockFn = Callable[[Ctx], pd.DataFrame]
BLOCKS: dict[str, BlockFn] = {}


def block(name: str) -> Callable[[BlockFn], BlockFn]:
    def deco(fn: BlockFn) -> BlockFn:
        BLOCKS[name] = fn
        return fn
    return deco


def _flatten(cols) -> list[str]:
    return [f"{c}_{f}" for c, f in cols]


# --------------------------------------------------------------------------- #
# Baseline blocks
# --------------------------------------------------------------------------- #
@block("volume")
def _volume(ctx: Ctx) -> pd.DataFrame:
    return ctx.g.size().rename("n_products").astype("int32").to_frame()


@block("ordinal_stats")
def _ordinal_stats(ctx: Ctx) -> pd.DataFrame:
    out = ctx.g[C.ORDINAL_COLS].agg(["mean", "max", "min", "std"])
    out.columns = _flatten(out.columns)
    return out


@block("delinquency")
def _delinquency(ctx: Ctx) -> pd.DataFrame:
    out = ctx.g[C.DELINQ_COLS].agg(["sum", "mean", "max"])
    out.columns = _flatten(out.columns)
    return out


@block("flags")
def _flags(ctx: Ctx) -> pd.DataFrame:
    out = ctx.g[C.FLAG_COLS].mean()
    out.columns = [f"{c}_rate" for c in out.columns]
    return out


@block("payment_status")
def _payment_status(ctx: Ctx) -> pd.DataFrame:
    out = ctx.g[C.ENC_PAYM_COLS].agg(["mean", "max"])
    out.columns = _flatten(out.columns)
    overall = pd.DataFrame(index=ctx.index)
    overall["enc_paym_overall_mean"] = out[[f"{c}_mean" for c in C.ENC_PAYM_COLS]].mean(axis=1)
    overall["enc_paym_overall_max"] = out[[f"{c}_max" for c in C.ENC_PAYM_COLS]].max(axis=1)
    return pd.concat([out, overall], axis=1)


@block("categorical_diversity")
def _cat_diversity(ctx: Ctx) -> pd.DataFrame:
    out = ctx.g[C.CATEGORICAL_COLS].nunique()
    out.columns = [f"{c}_nunique" for c in out.columns]
    return out


@block("categorical_counts")
def _cat_counts(ctx: Ctx) -> pd.DataFrame:
    parts = []
    for col in _COUNT_CATS:
        ct = pd.crosstab(ctx.df[C.ID_COL], ctx.df[col])
        ct.columns = [f"{col}_cnt_{int(v)}" for v in ct.columns]
        parts.append(ct)
    return pd.concat(parts, axis=1)


@block("last_product")
def _last_product(ctx: Ctx) -> pd.DataFrame:
    last = ctx.df.sort_values([C.ID_COL, C.RN_COL]).groupby(C.ID_COL, sort=True).tail(1)
    last = last.set_index(C.ID_COL)
    cols = C.ORDINAL_COLS + C.CATEGORICAL_COLS + ["pclose_flag", "fclose_flag"]
    return last[cols].add_suffix("_last")


# --------------------------------------------------------------------------- #
# Extension blocks (phase 2 — feature-addition stage). Leak-safe, within-id.
# --------------------------------------------------------------------------- #
@block("term_gaps")
def _term_gaps(ctx: Ctx) -> pd.DataFrame:
    """Planned-vs-actual term / closure gaps, aggregated per id."""
    df = ctx.df
    gap_pf = (df["pre_pterm"] - df["pre_fterm"]).groupby(df[C.ID_COL], sort=True)
    gap_close = (df["pre_till_pclose"] - df["pre_till_fclose"]).groupby(df[C.ID_COL], sort=True)
    out = pd.DataFrame(index=ctx.index)
    out["term_gap_pf_mean"] = gap_pf.mean()
    out["term_gap_pf_max"] = gap_pf.max()
    out["till_close_gap_mean"] = gap_close.mean()
    out["till_close_gap_max"] = gap_close.max()
    return out


@block("delinq_share")
def _delinq_share(ctx: Ctx) -> pd.DataFrame:
    """Share of products that ever had any delinquency, per id."""
    df = ctx.df
    any_delinq = (df[C.DELINQ_COLS].sum(axis=1) > 0).astype("int8")
    out = any_delinq.groupby(df[C.ID_COL], sort=True).mean().rename("any_delinq_share").to_frame()
    out["delinq_total"] = df[C.DELINQ_COLS].sum(axis=1).groupby(df[C.ID_COL], sort=True).sum()
    return out


@block("term_clean")
def _term_clean(ctx: Ctx) -> pd.DataFrame:
    """Sentinel-aware term features.

    `pre_pterm`/`pre_fterm` carry a fixed sentinel bin (4 / 8) on rows where the
    planned/actual closure date is undefined (`pclose_flag`/`fclose_flag` == 1).
    Here we aggregate the term **only over rows where it is actually defined**, so
    the mean/max are not polluted by the sentinel, plus the share of products with
    a defined term. ids with no defined term at all get -1 (distinct from bins 0..17).
    """
    df = ctx.df
    gid = df[C.ID_COL]
    planned = df["pre_pterm"].where(df["pclose_flag"] == 0)
    actual = df["pre_fterm"].where(df["fclose_flag"] == 0)
    out = pd.DataFrame(index=ctx.index)
    out["pre_pterm_clean_mean"] = planned.groupby(gid, sort=True).mean()
    out["pre_pterm_clean_max"] = planned.groupby(gid, sort=True).max()
    out["pre_fterm_clean_mean"] = actual.groupby(gid, sort=True).mean()
    out["pre_fterm_clean_max"] = actual.groupby(gid, sort=True).max()
    out["pterm_defined_share"] = (df["pclose_flag"] == 0).groupby(gid, sort=True).mean()
    out["fterm_defined_share"] = (df["fclose_flag"] == 0).groupby(gid, sort=True).mean()
    fill = ["pre_pterm_clean_mean", "pre_pterm_clean_max",
            "pre_fterm_clean_mean", "pre_fterm_clean_max"]
    out[fill] = out[fill].fillna(-1.0)
    return out


@block("paym_trend")
def _paym_trend(ctx: Ctx) -> pd.DataFrame:
    """Trend in payment status: recent month-slots vs early ones (per id)."""
    early = [f"enc_paym_{i}" for i in range(5)]
    recent = [f"enc_paym_{i}" for i in range(20, 25)]
    df = ctx.df
    e = df[early].mean(axis=1).groupby(df[C.ID_COL], sort=True).mean()
    r = df[recent].mean(axis=1).groupby(df[C.ID_COL], sort=True).mean()
    out = pd.DataFrame(index=ctx.index)
    out["enc_paym_early_mean"] = e
    out["enc_paym_recent_mean"] = r
    out["enc_paym_trend"] = r - e
    return out


# --------------------------------------------------------------------------- #
# Feature sets
# --------------------------------------------------------------------------- #
_BASELINE = [
    "volume", "ordinal_stats", "delinquency", "flags", "payment_status",
    "categorical_diversity", "categorical_counts", "last_product",
]

_V2 = _BASELINE + ["term_gaps", "delinq_share", "paym_trend"]

FEATURE_SETS: dict[str, list[str]] = {
    "baseline": _BASELINE,
    # Phase-2 set: baseline + new leak-safe blocks.
    "v2": _V2,
    # v2 + sentinel-aware "clean" term features (isolates the term_clean block).
    "v3": _V2 + ["term_clean"],
}


def build_id_features(df: pd.DataFrame, feature_set: str = "baseline") -> pd.DataFrame:
    """Return a DataFrame indexed by `id` for the requested feature set."""
    if feature_set not in FEATURE_SETS:
        raise KeyError(f"unknown feature_set '{feature_set}'; known: {list(FEATURE_SETS)}")
    g = df.groupby(C.ID_COL, sort=True)
    index = g.size().index
    ctx = Ctx(df=df, g=g, index=index)

    parts = [BLOCKS[name](ctx) for name in FEATURE_SETS[feature_set]]
    feats = pd.concat(parts, axis=1)

    std_cols = [c for c in feats.columns if c.endswith("_std")]
    if std_cols:
        feats[std_cols] = feats[std_cols].fillna(0.0)
    feats = feats.astype("float32")
    feats.index = feats.index.astype("int32")
    return feats
