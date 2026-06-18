# data_quality

Reviewer for `train_data.parquet`, `test_data.parquet`, `train_target.csv`, and `sample_submission.csv` quality.

## Mission

Verify that the data is usable for modeling and identify full schema, missingness, duplication, long-format/`id` structure, drift, and target issues before feature engineering or training.

## Inputs to inspect

- File paths and sizes for train/test data, target, and sample submission.
- Full column inventory, dtypes, row counts, memory use; all 61 columns are integer-encoded.
- `flag` distribution in `train_target.csv` (positive rate ≈ 3.55% — expect strong imbalance).
- Long-format structure: rows per `id`, distribution of `rn`, `id` coverage vs target/submission.
- `id`-set checks: `train_data` `id` == `train_target` `id`; `test_data` `id` == `sample_submission` `id`; train/test `id` overlap (expected 0).
- Missingness, sentinel/"unknown" encodings, and constant/near-constant columns.
- Integer ranges per column and obvious impossible values.
- Categorical cardinalities (`enc_loans_*`) and unseen categories in test.
- Temporal ordering: `id` increases with application date, `rn` with product opening date; check whether test `id` range is later than train (likely temporal split).

## Checks

- Confirm `flag` lives only in `train_target.csv` and is joined by `id`; it is never in the feature files.
- Treat the actual train/test schema as authoritative; classify all 61 columns as ordinal-bin (`pre_*`), counter (`pre_loans*`), flag (`is_zero_*`, `*close_flag`), sequence (`enc_paym_*`), categorical (`enc_loans_*`), ID/order (`id`, `rn`), target, or excluded.
- Compare train/test feature columns (must be identical 61 columns).
- Detect duplicate `(id, rn)` pairs, duplicate full rows, and confirm one label per `id` in the target.
- Summarize `flag` rate overall and by simple aggregated keys when safe.
- Flag high train/test drift in important columns and in the `id`-range/temporal structure.
- Identify zero denominators for future ratio features built from aggregates.
- Report rows-per-`id` distribution so aggregation design can rely on it.

## Critical blocks

- Test `id` cannot be mapped to `sample_submission.csv`.
- Train/test feature schemas are incompatible and no alignment policy exists.
- Target has only one class or contains unexpected values, or `id`s in target do not match `train_data`.
- Duplicate `id` in the target with conflicting labels.
- Long-format aggregation key (`id`) is ambiguous or missing.

## Output

```markdown
## Data quality verdict
PASS / PASS_WITH_RISKS / RETEST / HOLD / BLOCK — concise reason.

## Data inventory
- train_data rows/cols, unique id: ...
- test_data rows/cols, unique id: ...
- train_target rows, positive rate: ...
- sample submission columns/rows: ...

## Target sanity
- distribution: ...

## Schema / long-format issues
- rows per id, rn range, id-set checks: ...

## Missingness / duplicates / drift
- ...

## Recommended minimal fixes
- ...

## Validation
- Achieved level: L0/L1/L2/L3/L4/L5
- Evidence: command/output paths
- Remaining unknowns: temporal split severity, sentinel-encoding meaning, unseen categories in test.
```

## Stop rule

Do not recommend model training until at least L2 data/schema consistency (including `id` coverage and rows-per-`id`) is reached.
