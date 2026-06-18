# AGENTS.md — Alfa Bank / MIPT credit-scoring (default prediction) project

This project uses an agent-native workflow for tabular ML development, validation, and submission building for the Alfa Bank x MIPT case "Кредитный скоринг". The task is to predict the probability that a client goes into default on a loan, using the client's credit-product history.

Primary objective: build a reproducible model that predicts `P(flag = 1)` for each `id` in `test_data.parquet`, where `flag = 1` means the client defaulted (non-payment for more than 3 months within a year). The main quality criterion is ROC-AUC. Do not optimize for leaderboard score at the expense of leakage, irreproducibility, or invalid submission format.

## Operating model: Codex + ChatGPT

Use this split by default:

- **Codex**: repository inspection, data/schema checks, local patches, notebooks/scripts, experiments, tests, artifact generation, submission CSV checks.
- **ChatGPT**: task framing, architecture reasoning, experiment design, leakage review, red-team critique, prompt preparation, result interpretation.
- **README / experiment logs / submission cards**: project memory and living specification, not proof of model quality.
- **Scripts, notebooks, validation output, metrics, and generated submissions**: evidence layer.

Default context:

```text
AGENTS.md + agents/context_router.md + one primary agent + zero or one reviewer
```

Do not load all agents for a single task.

## Case facts to preserve

Data files (in `data/`):

- `data/raw/train_data.parquet` — training credit history, long format, no target;
- `data/raw/test_data.parquet` — test credit history, long format, no target;
- `data/raw/train_target.csv` — `id, flag`; the supervised label, one row per train `id`;
- `data/Кредитный скорринг/sample_submission.csv` — required submission schema (`id, flag`).

Verified structural facts (re-check, do not assume):

- **Long format.** One `id` (application) has many rows, one per credit product, ordered by `rn` (1..55, mean ≈ 8.7). The label is per `id`. History must be aggregated to one feature vector per `id` before joining the target and training.
- `train_data` rows ≈ 18.32M over 2.10M `id`; `test_data` rows ≈ 7.85M over 0.90M `id`.
- `train_data` `id` set == `train_target` `id` set; `test_data` `id` set == `sample_submission` `id` set; train/test `id` sets do **not** overlap (0 shared).
- `id` is ordered by application date (larger = later); `rn` is ordered by product opening date. The train/test split is temporal: train is N months, test is the subsequent K months.

Core fields (61 columns, all integer-encoded; classify and use or explicitly exclude every one):

- `id` — application identifier (index/ordering, not a raw predictor); `rn` — credit-product order within `id`;
- binarized `pre_*` (bins 0–19): `pre_since_opened`, `pre_since_confirmed`, `pre_pterm`, `pre_fterm`, `pre_till_pclose`, `pre_till_fclose`, `pre_loans_credit_limit`, `pre_loans_next_pay_summ`, `pre_loans_outstanding`, `pre_loans_total_overdue`, `pre_loans_max_overdue_sum`, `pre_loans_credit_cost_rate`, `pre_util`, `pre_over2limit`, `pre_maxover2limit`;
- delinquency counters: `pre_loans5`, `pre_loans530`, `pre_loans3060`, `pre_loans6090`, `pre_loans90`;
- zero-flags: `is_zero_loans5/530/3060/6090/90`, `is_zero_util`, `is_zero_over2limit`, `is_zero_maxover2limit`;
- payment-status sequence: `enc_paym_0..24` (monthly statuses, encoded 0–4);
- encoded categoricals: `enc_loans_account_holder_type`, `enc_loans_credit_status`, `enc_loans_credit_type`, `enc_loans_account_cur`;
- closure flags: `pclose_flag`, `fclose_flag`;
- `flag` (in `train_target.csv` only): `1` default, `0` repaid; default rate ≈ 3.55% (strong imbalance).

Modeling objective: rank clients by default probability; metric: ROC-AUC; final artifact: valid CSV submission with probabilities. Because the data is long and the label is per `id`, all aggregation must be computed strictly within each `id`, and every CV split must be made at the `id` level (never at the raw-row level before aggregation).

Allowed tools: Python 3.10+ and open-source ML/preprocessing libraries. Do not use closed libraries, private APIs, paid external datasets, or borrowed code without a compatible open license. Use only the provided train/test/target data unless the user explicitly gives additional allowed files from the platform.

Important platform constraint: do not burn daily submissions. The case allows no more than 3 platform uploads per day. Treat each exported submission as a scarce artifact and label it clearly.

## Task modes

Declare one mode before non-trivial work:

- `inspect_only` — read/analyze, no edits.
- `plan_only` — propose plan, no edits.
- `patch_small` — one local code/docs/config change.
- `data_quality_review` — schema, missingness, duplicates, long-format/aggregation sanity, train/test drift, target sanity.
- `eda_review` — relationships between features and default, distribution checks.
- `leakage_review` — target leakage, test leakage, aggregation/ID/time leakage.
- `feature_engineering` — create/validate aggregated per-`id` predictors.
- `cv_design` — validation strategy and split design at the `id` level.
- `baseline_build` — minimal reproducible baseline model.
- `model_training` — train/tune models and save artifacts.
- `ensemble_review` — blending/stacking and diversity checks.
- `metric_validation` — ROC-AUC and probability-output validation.
- `submission_build` — generate and validate CSV submission.
- `experiment_tracking` — update experiment log and artifact registry.
- `docs_sync` — README/decision-log update after real behavior/contract change.
- `red_team` — adversarial review before important submission/use.

## Global working rules

1. Spec first for non-trivial work: goal, inputs, affected files, protected contracts, validation.
2. Inspect before patching. Do not change files by memory.
3. Prefer minimal localized changes. No broad refactor unless explicitly requested.
4. Separate EDA, aggregation/feature building, training, validation, inference, and submission generation.
5. Never use `flag` outside the training rows. Never infer test labels from the sample submission.
6. Run full schema discovery on all 61 columns; every train/test feature must be classified, aggregated/used safely, or explicitly excluded with a reason.
7. The data is long: aggregate to one row per `id` first; compute aggregates strictly within `id`; split at the `id` level. Check temporal drift (train `id` earlier, test `id` later) before trusting random CV.
8. Validate schema, `id` coverage, and row order before writing a submission.
9. Report what was checked, what was not checked, and residual risk.
10. Explanations to the project owner should be in Russian unless code/comments/file content require English.

## Protected contracts

These are mandatory across all agents:

- Every accepted result must be traceable to input files, script/notebook version, config, random seed, split definition, model version, feature list, metric output, and artifact paths.
- `flag` must be used only as the supervised label in training/validation, never as a feature, and only via a join on `id` from `train_target.csv`.
- Test-set features may be used only for schema/drift-aware preprocessing that does not inspect labels and does not create target-derived encodings from test outcomes.
- Submission must match `sample_submission.csv` columns (`id`, `flag`), row count (900 000), `id` mapping, and probability range `[0, 1]`.
- ROC-AUC must be computed on held-out validation folds with probability scores, not hard labels.
- Aggregation from long to per-`id` rows must be computed within each `id`; cross-validation must split at the `id` level and avoid leakage from temporal ordering or target encodings.
- Categorical encodings, imputers, scalers, target encoders, and feature selectors must be fit inside each training fold/pipeline where applicable.
- Do not compare experiments if splits, features, preprocessing, target definition, aggregation policy, or metric implementation differ without explicit labels.
- Do not select a model only by one lucky fold or one public submission. Inspect variance, calibration shape, and robustness.
- Any leaderboard upload recommendation must include a submission card with source commit/script, validation score, generated file hash, and known risks.
- README/decision-log updates must distinguish what is implemented, what is planned, and what is confirmed by checks.

## Feature engineering principles

Prioritize interpretable, leak-safe per-`id` aggregates after full schema discovery. All features are integer-encoded ordinals/categoricals/flags — there are no raw monetary fields.

- **Volume/recency**: number of credit products (`max(rn)` / row count per `id`), recency via `pre_since_opened`/`pre_since_confirmed` bins (min/last), share of recently opened products.
- **Delinquency aggregates**: sums/means of `pre_loans5/530/3060/6090/90` across products, share of products with any delinquency, worst delinquency bucket reached, aggregates of `is_zero_loans*` flags.
- **Payment-status sequence**: from `enc_paym_0..24`, derive counts/means per status value, worst/last status, number of "bad" months, run-length features; this sequence is the strongest signal candidate.
- **Amounts/utilization (binarized)**: mean/max/last of `pre_loans_credit_limit`, `pre_loans_outstanding`, `pre_loans_next_pay_summ`, `pre_util`, `pre_over2limit`, `pre_maxover2limit`, and overdue bins; treat as ordinal bins, not raw rubles.
- **Closure/status**: rates of `pclose_flag`/`fclose_flag`, distribution of `enc_loans_credit_status`, planned-vs-fact term bins (`pre_pterm` vs `pre_fterm`, `pre_till_pclose` vs `pre_till_fclose`).
- **Categorical handling**: frequency/one-hot/CatBoost-native or fold-safe target encoding for `enc_loans_credit_type`, `enc_loans_account_cur`, `enc_loans_account_holder_type`, `enc_loans_credit_status`, aggregated to `id` (e.g. counts per category, dominant category).
- **Class imbalance**: with a ~3.55% positive rate, prefer `scale_pos_weight`/class weights over resampling; ROC-AUC is the target metric, but keep an eye on PR-AUC.
- **Missingness/encoding sentinels**: some bins/encodings may act as "unknown"; add indicators where informative.

Handle division by zero explicitly. Never use `id` or `rn` as a raw predictive feature (`id` is a time-ordered index and is leak-prone); engineered counts derived from `rn` are fine. Never create a feature that encodes the target or future outcomes.

## Validation levels

| Level | Meaning | Example |
|---|---|---|
| L0 | Static/document check | file tree, Markdown consistency, obvious contradictions |
| L1 | Local syntax/smoke | script imports, one small run, CLI `--help`, notebook executes first cells |
| L2 | Data/schema consistency | train/test columns, dtypes, long-format/`id` coverage, missingness, duplicates, target distribution, submission shape |
| L3 | Reproducible CV validation | deterministic `id`-level folds, per-fold ROC-AUC, out-of-fold predictions, saved config/artifacts |
| L4 | Robustness/regression | alternative splits, seed variance, feature ablation, drift checks, leakage checks, ensemble sanity |
| L5 | Submission readiness | red-team + submission card + hash + exact sample format + residual risk accepted |

Always state the achieved validation level. Do not use `PASS` when only README/agent files were inspected.

## Active agents

Use these first:

- `agents/context_router.md` — selects task mode, minimal context, primary agent, reviewer, validation level.
- `agents/architect.md` — task decomposition, scope control, protected contracts.
- `agents/task_spec_short.md` — compact task spec for non-trivial work.
- `agents/test_validation.md` — validation plan and evidence sufficiency.
- `agents/red_team.md` — adversarial review before important decisions/submission.
- `agents/readme_consistency_reviewer.md` — README/spec/decision-log drift control.
- `agents/decision_log_handoff.md` — reproducibility and handoff records.

Domain agents:

- `agents/data_quality.md` — schema, missingness, duplicates, long-format/`id` coverage, drift, target sanity.
- `agents/eda_analyst.md` — feature/default relationships and business interpretation.
- `agents/leakage_guard.md` — target/test/time/ID/aggregation leakage review.
- `agents/feature_engineer.md` — safe per-`id` aggregate design and pipeline implementation review.
- `agents/cv_validator.md` — `id`-level split design and out-of-fold validation.
- `agents/baseline_builder.md` — minimal reproducible baseline.
- `agents/model_trainer.md` — model training, tuning, and artifact saving.
- `agents/model_ensembler.md` — blending/stacking and ensemble validation.
- `agents/metric_validator.md` — ROC-AUC/probability-output checks.
- `agents/submission_builder.md` — submission CSV generation and final checks.
- `agents/experiment_manager.md` — experiment registry, seeds, configs, run comparison.
- `agents/interpretability_reviewer.md` — feature importance/SHAP/PDP and business sanity.
- `agents/reproducibility_reviewer.md` — environment, deterministic rerun, dependency and artifact checks.

## Routing examples

- New repository/data intake -> `context_router.md` -> `data_quality.md` + `test_validation.md`; include full schema discovery and long-format/`id`-coverage checks.
- Baseline script request -> `baseline_builder.md` + `metric_validator.md`.
- Feature engineering patch -> `feature_engineer.md` + `leakage_guard.md`.
- Validation split design -> `cv_validator.md` + `leakage_guard.md`.
- CatBoost/LightGBM/XGBoost tuning -> `model_trainer.md` + `metric_validator.md`.
- Blending several models -> `model_ensembler.md` + `red_team.md`.
- Generate final CSV -> `submission_builder.md` + `metric_validator.md`.
- Before platform upload -> `red_team.md` + `submission_builder.md` + `decision_log_handoff.md`.
- README/experiment log update -> `readme_consistency_reviewer.md` + `decision_log_handoff.md`.

## Decision states

Use these consistently:

- `BLOCK` — must not proceed; critical evidence, leakage, data, metric, or submission-format gap.
- `HOLD` — insufficient evidence; get more checks/data/output first.
- `RETEST` — plausible but requires rerun or stronger validation.
- `PASS_WITH_RISKS` — acceptable for the stated scope with explicit caveats.
- `PASS` — acceptable only for the reviewed scope; never means leaderboard success is guaranteed.

## Default review output

```markdown
## Verdict
<state> — one sentence.

## Critical
- ...

## Medium
- ...

## Minimal patch / action
- ...

## Validation
- Achieved level: L0/L1/L2/L3/L4/L5
- What was checked
- What remains unchecked

## Decision log
- Inputs, assumptions, artifacts, and next owner
```

## Embedded record templates

Keep operational records in repository files created by Codex:

- Experiment log columns: `run_id`, `timestamp`, `data_hash`, `split`, `seed`, `feature_set`, `model`, `params_ref`, `fold_auc`, `oof_auc`, `test_pred`, `submission`, `notes`.
- Submission card fields: file path, SHA256, generation script/config, source run id, fold/OOF ROC-AUC, leakage/metric/submission/red-team verdicts, sample-format checks, upload recommendation, daily upload count.
- Validation matrix fields: check, validation level, owner agent, evidence path, status, notes.

## Strict final rule

If required evidence is missing, say exactly what cannot be concluded. Do not fill missing data, metric, leakage, or submission facts by assumption.
