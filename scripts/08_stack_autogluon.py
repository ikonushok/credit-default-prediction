"""CLI: stack several runs' OOF/test predictions with AutoGluon Tabular.

Usage:
    python scripts/08_stack_autogluon.py --runs <run_a> <run_b> [...] \
        [--name stack_ag] [--feature-set baseline] [--preset medium_quality] \
        [--time-limit 1800] [--bag-folds 5] [--stack-levels 0]

Builds a meta-feature matrix on train rows — rank-normalized OOF predictions of
each base run, optionally concatenated with an id-level aggregate table — and
trains a bagged AutoGluon TabularPredictor as the meta-learner (replacing the
linear rank-blend in 07_ensemble.py). The bagged predictor yields leak-safe
out-of-fold meta-predictions via predict_proba_oof(); test predictions come from
the same fitted ensemble. Writes an artifacts run dir so 04_predict_submit.py can
build the submission from it.

Alignment: all runs store OOF/test in ascending-id order (same as 07_ensemble),
so meta-features and the aggregate table concat positionally. AutoGluon does its
own internal bagged CV, so its meta-OOF folds differ from the project skf5 — we
reuse the base run's folds.npy only for the per-fold metric breakdown and for
downstream blending in 07_ensemble.py (which warns if folds differ).
"""
import argparse
import json

import numpy as np
import pandas as pd
from scipy.stats import rankdata

import _bootstrap  # noqa: F401

from credit_scoring import config as C
from credit_scoring import metrics, tracking


def _rank01(s: np.ndarray) -> np.ndarray:
    return rankdata(s) / len(s)


def main() -> None:
    ap = argparse.ArgumentParser()
    ap.add_argument("--runs", nargs="+", required=True)
    ap.add_argument("--name", default="stack_ag")
    ap.add_argument("--feature-set", default=None,
                    help="if set, concat data/processed/{split}_features__{fs}.parquet "
                         "feature columns onto the meta-matrix (omit to stack OOF only)")
    ap.add_argument("--preset", default="medium_quality",
                    help="AutoGluon preset: medium_quality (fast) .. best_quality (slow)")
    ap.add_argument("--time-limit", type=int, default=1800, help="fit budget in seconds")
    ap.add_argument("--bag-folds", type=int, default=5)
    ap.add_argument("--stack-levels", type=int, default=0)
    ap.add_argument("--mem-ratio", type=float, default=1.0,
                    help="ag.max_memory_usage_ratio; raise >1 to push through OOM skips")
    ap.add_argument("--subsample", type=int, default=0,
                    help="if >0, stratified-sample this many train rows for fit "
                         "(DIAGNOSTIC: bagged OOF covers only the sample, no submission artifact)")
    args = ap.parse_args()

    try:
        from autogluon.tabular import TabularPredictor
    except ImportError:
        raise SystemExit("AutoGluon not installed — `pip install autogluon.tabular`")

    y = pd.read_parquet(C.DATA_PROCESSED / "train_features.parquet")[C.TARGET_COL].to_numpy("int8")

    # --- assemble base meta-features (rank-normalized OOF / test per run) ---
    oof_cols, test_cols, folds0, test_ids0, n_train = {}, {}, None, None, None
    for r in args.runs:
        d = C.ARTIFACTS / r
        f = np.load(d / "folds.npy")
        ti = np.load(d / "test_ids.npy")
        oof = np.load(d / "oof.npy")
        if folds0 is None:
            folds0, test_ids0, n_train = f, ti, len(oof)
        else:
            if not np.array_equal(ti, test_ids0):
                raise SystemExit(f"test_ids of {r} differ — cannot align")
            if len(oof) != n_train:
                raise SystemExit(f"oof length of {r} differs — cannot align")
            if not np.array_equal(f, folds0):
                print(f"WARNING: folds of {r} differ from base — meta-OOF mildly optimistic")
        oof_cols[f"oof__{r}"] = _rank01(oof)
        test_cols[f"oof__{r}"] = _rank01(np.load(d / "test_pred.npy"))

    train_df = pd.DataFrame(oof_cols)
    test_df = pd.DataFrame(test_cols)

    # --- optional: enrich with id-level aggregates (positional concat) ---
    if args.feature_set:
        # "baseline" lives in the unversioned table; v2/v3/... are versioned.
        def _ftr_path(split):
            return (C.DATA_PROCESSED / f"{split}_features.parquet"
                    if args.feature_set == "baseline"
                    else C.features_path(split, args.feature_set))
        ftr = pd.read_parquet(_ftr_path("train"))
        fte = pd.read_parquet(_ftr_path("test"))
        drop = [c for c in (C.TARGET_COL, C.ID_COL) if c in ftr.columns]
        ftr = ftr.drop(columns=drop)
        fte = fte.drop(columns=[c for c in drop if c in fte.columns])
        if len(ftr) != n_train or len(fte) != len(test_ids0):
            raise SystemExit("aggregate table length mismatch — cannot align positionally")
        train_df = pd.concat([train_df, ftr.reset_index(drop=True)], axis=1)
        test_df = pd.concat([test_df, fte.reset_index(drop=True)], axis=1)
        print(f"enriched with {ftr.shape[1]} aggregate columns from feature_set={args.feature_set}")

    train_df[C.TARGET_COL] = y
    print(f"meta-matrix: {train_df.shape[1] - 1} features, {len(train_df)} train rows")

    run_id = tracking.now_run_id(args.name)
    d = C.ARTIFACTS / run_id
    d.mkdir(parents=True, exist_ok=True)

    # DIAGNOSTIC: subsample (stratified) to fit within memory. The bagged OOF then
    # covers only the sample, so we report score_val and skip submission artifacts.
    fit_df, y_fit = train_df, y
    if args.subsample and args.subsample < len(train_df):
        rng = np.random.RandomState(C.SEED)
        pos = np.where(y == 1)[0]
        neg = np.where(y == 0)[0]
        frac = args.subsample / len(train_df)
        keep = np.concatenate([
            rng.choice(pos, int(round(len(pos) * frac)), replace=False),
            rng.choice(neg, int(round(len(neg) * frac)), replace=False),
        ])
        keep.sort()
        fit_df = train_df.iloc[keep].reset_index(drop=True)
        y_fit = y[keep]
        print(f"DIAGNOSTIC subsample: {len(fit_df)} rows ({frac:.1%}), pos rate {y_fit.mean():.4f}")

    predictor = TabularPredictor(
        label=C.TARGET_COL, problem_type="binary", eval_metric="roc_auc",
        path=str(d / "ag_models"),
    ).fit(
        fit_df, presets=args.preset, time_limit=args.time_limit,
        num_bag_folds=args.bag_folds, num_stack_levels=args.stack_levels,
        ag_args_fit={"ag.max_memory_usage_ratio": args.mem_ratio},
    )

    # Leak-safe OOF from the bagged ensemble; positive-class proba.
    oof_meta = predictor.predict_proba_oof()[1].to_numpy()
    auc = metrics.roc_auc(y_fit, oof_meta)
    best_base = max(metrics.roc_auc(y, v) for v in oof_cols.values())
    print(f"\nstack OOF ROC-AUC = {auc:.5f}  PR-AUC = {metrics.pr_auc(y_fit, oof_meta):.5f}")
    print(f"best base (full OOF) = {best_base:.5f}  gain = {auc - best_base:+.5f}")
    print("\nleaderboard (top):")
    print(predictor.leaderboard(silent=True).head(8).to_string(index=False))

    if args.subsample and args.subsample < len(train_df):
        print("\nDIAGNOSTIC run — partial OOF, no submission artifact written.")
        tracking.write_metrics(run_id, {
            "diagnostic_subsample": int(len(fit_df)), "subsample_oof_roc_auc": auc,
            "best_base_full_oof": best_base, "preset": args.preset,
            "feature_set": args.feature_set, "members": args.runs,
        })
        return

    test_meta = predictor.predict_proba(test_df)[1].to_numpy()
    np.save(d / "folds.npy", folds0)
    np.save(d / "oof.npy", oof_meta)
    np.save(d / "test_pred.npy", test_meta)
    np.save(d / "test_ids.npy", test_ids0)
    rep = metrics.fold_report(y, oof_meta, folds0)
    rep["members"] = args.runs
    rep["preset"] = args.preset
    rep["feature_set"] = args.feature_set
    tracking.write_metrics(run_id, rep)
    tracking.append_log({
        "run_id": run_id, "timestamp": run_id[:15], "split": "skf5",
        "model": f"stack_ag({'+'.join(args.runs)})", "feature_set": args.feature_set or "oof_only",
        "fold_auc": f"{rep['fold_roc_mean']:.5f}±{rep['fold_roc_std']:.5f}",
        "oof_auc": f"{auc:.5f}", "test_pred": str(d / "test_pred.npy"),
        "notes": f"AutoGluon {args.preset}, bag={args.bag_folds}, stack_levels={args.stack_levels}",
    })
    print(f"\nstack run_id={run_id}  ->  python scripts/04_predict_submit.py --run {run_id}")


if __name__ == "__main__":
    main()
