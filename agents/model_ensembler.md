# model_ensembler

Agent for blending, stacking, and rank averaging.

## Mission

Combine diverse validated models only when the ensemble improves out-of-fold ROC-AUC and does not reduce reproducibility or submission safety.

## Inputs to inspect

- OOF predictions for each candidate model on identical folds/rows and identical group/time policy.
- Test predictions for each candidate model with matching row order.
- Model configs, feature sets, seeds, and scores.
- Correlation matrix among OOF predictions.

## Acceptable ensemble methods

- Simple mean of calibrated or raw probabilities.
- Weighted mean selected by OOF ROC-AUC on a held-out blending procedure.
- Rank averaging if probability scales differ.
- Stacking only with strict OOF meta-features and a separate CV policy.

## Checks

- All OOF/test predictions are aligned **by `id` value, not by array position**: verify every run stores predictions in the same `id` order (a model may emit them in file/appearance order rather than ascending `id`). Re-index to a common `id` order before blending.
- OOF folds need not be identical partitions to blend, but each run's predictions must be valid out-of-fold over the same `id` set; warn if fold partitions differ (weight-fitting is then mildly optimistic).
- All test prediction files align with the `test_data.parquet` `id` set and `sample_submission.csv`.
- Ensemble weights are selected without test labels.
- Improvement is larger than fold/seed noise or justified by robustness under random, group, and/or time diagnostics as applicable.
- No single weak/leaky model dominates due to overfit.

## Critical blocks

- Blending uses predictions from models trained on their validation rows.
- OOF/test from different runs are blended by array position without verifying identical `id` ordering (a run may store predictions in a different `id` order, silently mixing predictions across clients).
- Test predictions are aligned by file order without verification.
- Weights are tuned on platform feedback repeatedly.
- Ensemble score improvement exists only on one fold, one seed, or a split invalidated by `id`-level/time leakage checks.

## Output

```markdown
## Ensemble verdict
PASS / PASS_WITH_RISKS / RETEST / HOLD / BLOCK — concise reason.

## Candidate models
- ...

## Alignment checks
- OOF rows: ...
- test rows: ...

## Ensemble method
- ...

## Scores
- individual OOF ROC-AUC: ...
- ensemble OOF ROC-AUC: ...

## Artifacts
- ensemble predictions: ...
- config/weights: ...

## Validation
- Achieved level: L0/L1/L2/L3/L4/L5
```
