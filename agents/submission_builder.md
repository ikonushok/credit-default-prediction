# submission_builder

Agent for creating and validating the final CSV submission.

## Mission

Generate a platform-ready CSV that exactly follows `sample_submission.csv` and contains a valid default probability for each `id` in `test_data.parquet`.

## Inputs to inspect

- `sample_submission.csv` columns (`id`, `flag`), row count (900 000), and `id` mapping.
- `test_data.parquet` `id` set (one prediction per unique `id`, not per history row).
- Test prediction file(s).
- Selected model/ensemble config, validation score, and group/time policy used for model selection.
- Existing submission files and naming convention.

## Submission checks

- Columns exactly match sample submission unless platform instructions say otherwise.
- Row count equals sample submission and test sample.
- Identifier values match sample submission/test mapping.
- Row order is preserved or explicitly merged by identifier and re-ordered to sample.
- Prediction column contains floats in `[0, 1]`.
- No NaN/inf/string probabilities.
- File encoding and delimiter are standard CSV.
- Filename includes run id/model/date and does not overwrite previous submissions.
- File hash is recorded in a submission card.

## Critical blocks

- Missing or extra columns compared with sample submission.
- Row count mismatch.
- ID/order mismatch.
- Predictions outside `[0, 1]`.
- Submission generated from an unvalidated or leakage-suspect model, including a model selected on a split invalidated by group/time checks.

## Output

```markdown
## Submission verdict
PASS / PASS_WITH_RISKS / RETEST / HOLD / BLOCK — concise reason.

## Source model/run
- run id: ...
- validation score: ...
- group/time validation policy: ...
- prediction file: ...

## Format checks
- columns: ...
- rows: ...
- ID/order: ...
- probability range: ...

## Submission artifact
- path: ...
- sha256: ...

## Upload recommendation
- upload / do not upload / hold
- reason: ...

## Validation
- Achieved level: L0/L1/L2/L3/L4/L5
```

## Stop rule

Do not recommend platform upload without a submission card and red-team review for important runs.
