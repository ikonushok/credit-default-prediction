# Credit Default Prediction

Проект для задачи Альфа-Банка x МФТИ «Кредитный скоринг».

## Цель проекта

Построить воспроизводимую ML-модель, которая прогнозирует вероятность выхода клиента в дефолт по кредиту (неплатёж более 3 месяцев в течение года) на основе истории его кредитных продуктов.

Целевая переменная (`flag` в `train_target.csv`):

- `flag = 1` — клиент вышел в дефолт;
- `flag = 0` — клиент выплатил кредит;
- доля дефолтов ≈ 3.55% — сильный дисбаланс классов.

Основная метрика качества:

- ROC-AUC (PR-AUC — как вспомогательная проверка).

Итоговый артефакт:

- CSV-файл с вероятностью дефолта на каждый `id`, совместимый с форматом `sample_submission.csv`.

## Данные

Ожидаемые локальные файлы:

    data/raw/train_data.parquet      # обучающая история кредитных продуктов (long)
    data/raw/test_data.parquet       # тестовая история (long)
    data/raw/train_target.csv        # метки id, flag для train
    data/Кредитный скорринг/sample_submission.csv   # формат сабмита id, flag

Формат **long**: один `id` (заявка) = несколько строк, по одной на кредитный продукт, упорядоченных по `rn`. Метка — одна на `id`, поэтому историю агрегируем в один вектор признаков на `id`. Размеры: train ≈ 18.3M строк / 2.10M `id`, test ≈ 7.85M / 0.90M `id`. Файлы с данными не коммитятся в Git.

## Ограничения

- Python 3.10+.
- Только open-source библиотеки.
- Используются только данные, предоставленные в рамках задания.
- Нельзя использовать `flag` как признак; он джойнится по `id` только в train.
- Нельзя использовать `id`/`rn` как сырые предикторы (это порядковый временной индекс).
- Нельзя использовать test labels или информацию из public leaderboard для подгонки модели.
- Сабмиты на платформу — ограниченный ресурс: не более 3 загрузок в день.

## Структура проекта

    data/raw/              # исходные parquet/csv, не коммитятся
    data/interim/          # промежуточные артефакты
    data/processed/        # агрегированные признаки на id (train/test_features__<set>.parquet)

    src/credit_scoring/    # основной Python-пакет проекта
      config.py            # пути, seed, группы колонок, RunConfig (YAML)
      data_io.py           # чтение parquet с downcast dtypes, target, sample
      features.py          # реестр блоков фич + feature sets (baseline/v2/v3)
      aggregate.py         # long -> один вектор признаков на id
      cv.py                # id-level StratifiedKFold + time-holdout по порядку id
      metrics.py           # ROC-AUC / PR-AUC и per-fold отчёт
      submission.py        # сборка и валидация submission по контракту
      tracking.py          # experiment log и submission cards
      models/
        lgbm.py            # LightGBM
        catboost_model.py  # CatBoost
        sequence.py        # GRU над последовательностью продуктов (torch/MPS)

    scripts/               # CLI-скрипты для запусков (01..07)
    configs/               # YAML-конфиги прогонов
    experiments/           # experiment_log.csv, cards/, отчёты и логи
    artifacts/             # per-run: folds, OOF/test predictions, metrics, config
    submissions/           # финальные CSV-сабмиты (всегда в корне проекта)

## Рекомендуемый порядок работы

### 1. Data quality / EDA

Проверяем размеры train/test/target/sample, наличие `flag` только в target, совместимость схем, распределение rows-per-`id`, совпадение множеств `id`, дрейф train/test по диапазону `id` (сплит временной), частоты категорий и платёжных статусов.

    python scripts/01_eda.py            # -> experiments/eda_report.md

### 2. Агрегация long -> id

Каждый `id` сворачивается в один вектор признаков (агрегация строго внутри `id`). Feature sets заданы реестром блоков в `features.py`; результат версионируется на диске.

    python scripts/02_aggregate.py --feature-set baseline
    # данные пишутся в data/processed/{train,test}_features__baseline.parquet

Чтобы **добавить признаки**: написать новый блок `@block("...")`, зарегистрировать набор в `FEATURE_SETS`, переагрегировать с `--feature-set <name>` и сравнить с baseline на тех же фолдах.

### 3. Validation design

После агрегации на `id` ровно одна строка, поэтому `StratifiedKFold` по `id` (seed фиксирован) не допускает утечки строк между фолдами. Дополнительно считается **time-holdout** (поздние 20% `id`), чтобы оценить временной дрейф — платформенный тест временной.

### 4. Baseline model

    python scripts/03_train.py --config configs/lgbm_baseline.yaml

LightGBM со `scale_pos_weight` под дисбаланс; печатает per-fold и OOF ROC-AUC, time-holdout, сохраняет артефакты прогона и строку в experiment log.

### 5. Feature engineering

Приоритетные группы агрегатов: число продуктов, агрегаты просрочек, платёжная история (`enc_paym`), утилизация/суммы (как ordinal-бины), сроки и закрытие, категориальные. Все supervised-преобразования (target-encoding) — внутри CV-folds. Эффект каждого набора проверяется ablation'ом:

    python scripts/03_train.py --config configs/lgbm_v2.yaml
    python scripts/05_compare.py --base <baseline_run> --cand <v2_run>

### 6. Sequence model

GRU читает последовательность продуктов клиента (порядок `rn`), 59 признаков на шаг — альтернативное представление, не агрегаты. Те же `id`-фолды для ансамблируемости.

    python scripts/06_train_seq.py --config configs/sequence.yaml

### 7. Model training (CatBoost)

    python scripts/03_train.py --config configs/catboost.yaml

Сравнивать можно только эксперименты с одинаковой CV-схемой, target definition и понятной feature policy.

### 8. Ensemble

Rank-блендинг нескольких прогонов с подбором весов по OOF; принимается только при приросте OOF ROC-AUC выше fold/seed-шума.

    python scripts/07_ensemble.py --runs <run_a> <run_b> <run_c> --name ens

### 9. Submission build

Перед отправкой проверяется: колонки `id,flag`; 900 000 строк; `id` совпадают с `sample_submission`; вероятности в `[0, 1]`; нет NaN/inf; SHA256 в submission card.

    python scripts/04_predict_submit.py --run <run_id>
    # -> submissions/<run_id>.csv + experiments/cards/<run_id>.md

## Валидационные уровни

- L0 — статическая проверка файлов и документации.
- L1 — smoke/syntax checks.
- L2 — проверка данных, схем, long-формата и покрытия `id`.
- L3 — воспроизводимая CV-валидация с OOF ROC-AUC (id-level фолды).
- L4 — robustness checks, alternative splits, leakage review, ансамбль.
- L5 — submission readiness: sample-format check, hash, submission card, red-team review.

## Стартовая команда для Codex

    Use AGENTS.md + agents/context_router.md + agents/data_quality.md + agents/test_validation.md.

    Mode: data_quality_review.

    Task: Inspect train_data.parquet, test_data.parquet, train_target.csv, and sample_submission.csv for the Alfa Bank credit-scoring (default prediction) task.

    Inputs:
    - data/raw/train_data.parquet
    - data/raw/test_data.parquet
    - data/raw/train_target.csv
    - data/Кредитный скорринг/sample_submission.csv

    Constraints:
    - Do not train a model yet.
    - Do not use flag outside train labels; join it by id only.
    - Inspect the full 61-column schema.
    - Check target distribution, train/test schema compatibility, long-format rows-per-id, id-set alignment, id-range temporal drift, and sample_submission compatibility.
    - Report achieved validation level.

    Stop if data files are missing, train/test schema cannot be aligned, target has unexpected values, or test ids cannot be mapped to sample_submission.

## Текущий статус

Пайплайн реализован и прогнан end-to-end. Результаты (5-fold OOF ROC-AUC, выравнивание по `id`):

| Модель | Представление | OOF ROC-AUC |
|---|---|---|
| LightGBM (baseline) | агрегаты на id | 0.7548 |
| CatBoost | агрегаты на id | 0.7548 |
| Sequence GRU | последовательность продуктов | 0.7547 |
| **Ансамбль (rank-blend)** | GRU 0.48 / CatBoost 0.28 / LGBM 0.25 | **0.7645** |

Ablation показал, что простые производные агрегаты (`v2`/`v3`) дают прирост в пределах шума; основной прирост (+0.0097) — от ансамбля разнородных представлений. Временной дрейф (random-CV vs time-holdout) ≈ −0.025, поэтому ожидаемый скор на платформенном (временном) тесте ниже OOF. Финальный сабмит ансамбля — в `submissions/`.
