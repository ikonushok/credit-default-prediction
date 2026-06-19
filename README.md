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
        sequence.py        # Embedding bi-GRU над последовательностью продуктов (torch/MPS)

    scripts/               # CLI-скрипты для запусков (01..07)
    configs/               # YAML-конфиги прогонов
    experiments/           # experiment_log.csv, cards/, отчёты и логи
    artifacts/             # per-run: folds, OOF/test predictions, metrics, config
    submissions/           # финальные CSV-сабмиты (всегда в корне проекта)
    agents/                # 21 специализированный агент (см. agents/README.md)

## Рекомендуемый порядок работы

### 1. Data quality / EDA

Перед моделированием необходимо проверить:

- размеры train/test/target/sample;
- наличие `flag` только в `train_target.csv`;
- совместимость train/test schema (61 целочисленная колонка);
- распределение rows-per-`id` (long-формат, ~8.7 строк/id);
- совпадение множеств `id` (train_data == train_target, test_data == sample_submission, пересечение = 0);
- дрейф train/test по диапазону `id` (сплит временной: train — ранние `id`, test — поздние);
- частоты категорий (`enc_loans_*`) и платёжных статусов (`enc_paym_*`);
- sentinel-кодирование пропусков (`pclose_flag=1` → `pre_pterm=4`, `fclose_flag=1` → `pre_fterm=8`).

Ожидаемые артефакты:

    experiments/eda_report.md

Команда:

    python scripts/01_eda.py

### 2. Агрегация long → id

Каждый `id` сворачивается в один вектор признаков (агрегация строго внутри `id`). Feature sets заданы реестром блоков в `features.py`; результат версионируется на диске.

    python scripts/02_aggregate.py --feature-set baseline
    # -> data/processed/{train,test}_features__baseline.parquet

Чтобы **добавить признаки**: написать новый блок `@block("...")`, зарегистрировать набор в `FEATURE_SETS`, переагрегировать с `--feature-set <name>` и сравнить с baseline на тех же фолдах.

### 3. Validation design

Используется `StratifiedKFold(5, seed=42)` по `id` после агрегации (одна строка = один `id`, утечка строк невозможна). Дополнительно — **time-holdout** (поздние 20% `id`): оценка временного дрейфа, поскольку платформенный тест временной.

Fold assignments детерминированы по seed (не по случайному порядку).

### 4. Baseline model

Первый baseline — LightGBM на агрегатах:

- обработка дисбаланса через `scale_pos_weight`;
- per-fold и OOF ROC-AUC + PR-AUC;
- сравнение random-CV vs time-holdout;
- сохранённые folds, OOF/test predictions, config, metrics;
- feature manifest (182 агрегата из baseline-блоков).

Команда:

    python scripts/03_train.py --config configs/lgbm_baseline.yaml

### 5. Feature engineering

Рабочие признаки (используются в baseline, 182 агрегата):

- **volume/recency** — число продуктов (`n_products`), min/last `pre_since_opened/confirmed`;
- **delinquency** — sum/mean/max `pre_loans5/530/3060/6090/90`, агрегаты `is_zero_loans*`;
- **payment-status** — mean/max `enc_paym_0..24`, cross-slot `enc_paym_overall_mean/max`;
- **amounts/utilization** — mean/max/min/std `pre_loans_credit_limit`, `pre_util`, `pre_over2limit` и др. (как ordinal-бины, не сырые суммы);
- **closure/status** — rates `pclose_flag`/`fclose_flag`, распределение `enc_loans_credit_status`;
- **categorical diversity** — nunique + per-category counts для `enc_loans_*`;
- **last product** — значения ключевых колонок последнего (по `rn`) продукта.

Проверены и **закрыты** (экспериментально нейтральные, ablation на тех же фолдах):

- `term_gaps` (pre_pterm − pre_fterm, till_pclose − till_fclose) — δ OOF = −0.0001 (шум);
- `delinq_share` (доля продуктов с просрочкой, суммарная) — выводится из baseline;
- `paym_trend` (тренд enc_paym recent vs early) — δ OOF = 0 (шум);
- `term_clean` (sentinel-aware агрегаты сроков) — δ OOF = +0.0002 (шум, информация уже в флагах).

Вывод: **GBDT-ветка вышла на плато ~0.755 OOF** — ручные производные агрегаты не добавляют сигнала. Основной рычаг — в представлении данных (sequence-модель).

Все supervised-преобразования выполняются внутри CV-folds. Сравнение:

    python scripts/05_compare.py --base <baseline_run> --cand <v2_run>

### 6. Sequence model (Embedding bi-GRU)

Каждый `id` — последовательность кредитных продуктов (порядок `rn`), каждый продукт — вектор из 59 целочисленных колонок. Все 59 колонок — категориальные коды (бины 0–19, статусы 0–4, флаги), поэтому каждая получает собственный обучаемый эмбеддинг (общая таблица `nn.Embedding` + per-column offsets, размер = `max(train, test) + 1`). Эмбеддинги конкатенируются по таймстепу и подаются в **двунаправленный GRU**; финальные скрытые состояния обоих направлений → MLP-голова → логит дефолта. Паддинг замаскирован через `pack_padded_sequence`.

Это альтернативное представление данных — модель учится на сырых последовательностях, а не на агрегатах. Те же `id`-фолды (seed=42) для ансамблируемости с GBDT.

Embedding bi-GRU — **главный рычаг качества**: одиночная модель дала OOF **0.7798** (+0.025 к GBDT). Эмбеддинги позволяют модели выучить представление каждого кода, а bi-GRU — паттерны в последовательности.

Команда:

    python scripts/06_train_seq.py --config configs/sequence_emb.yaml

### 7. Model training (CatBoost)

    python scripts/03_train.py --config configs/catboost.yaml

Проверенные модели и итоги:

| модель | OOF ROC-AUC | time-holdout | роль в финальном решении |
|---|---|---|---|
| LightGBM (baseline, 182 агрегата) | 0.7548 | 0.7295 | компонента ансамбля (6%) |
| CatBoost (те же агрегаты) | 0.7548 | 0.7295 | компонента ансамбля (10%) |
| Sequence GRU (scaled-numeric) | 0.7547 | — | закрыт (заменён Embedding bi-GRU) |
| **Embedding bi-GRU** | **0.7798** | — | **champion-компонента** ансамбля (84%) |

Сравнивать можно только эксперименты с одинаковой CV-схемой, одинаковым target definition и понятной feature policy.

### 8. Ensemble

Ансамбль допустим только при наличии aligned OOF/test predictions (по **значению `id`**, не по позиции) и подтверждённого прироста OOF ROC-AUC выше fold/seed-шума.

Рабочий метод — **rank-percentile blend**:

    blend = emb_gru_rank × w1 + catboost_rank × w2 + lgbm_rank × w3

Rank-нормализация выполняется для выравнивания масштабов (GBDT-вероятности vs NN-сигмоида). Веса подбираются Nelder-Mead по OOF ROC-AUC (неотрицательные, сумма = 1). Разные модели могут использовать разные OOF-фолды (с предупреждением). Прирост ансамбля: **+0.001 к лучшей одиночной** (0.7798 → 0.7808).

    python scripts/07_ensemble.py --runs <run_a> <run_b> <run_c> --name ens

### 9. Submission build

Перед отправкой на платформу проверяется:

- колонки совпадают с `sample_submission.csv` (`id`, `flag`);
- число строк совпадает (900 000);
- порядок `id` и mapping проверены;
- вероятности находятся в диапазоне `[0, 1]`;
- нет NaN/inf;
- файл имеет уникальное имя;
- SHA256 записан в submission card;
- проведён red-team review.

Команда:

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

    Task: Inspect train_data.parquet, test_data.parquet, train_target.csv, and
    sample_submission.csv for the Alfa Bank credit-scoring (default prediction) task.

    Inputs:
    - data/raw/train_data.parquet
    - data/raw/test_data.parquet
    - data/raw/train_target.csv
    - data/Кредитный скорринг/sample_submission.csv

    Constraints:
    - Do not train a model yet.
    - Do not use flag outside train labels; join it by id only.
    - Inspect the full 61-column schema.
    - Check target distribution, train/test schema compatibility, long-format
      rows-per-id, id-set alignment, id-range temporal drift, and
      sample_submission compatibility.
    - Report achieved validation level.

    Stop if data files are missing, train/test schema cannot be aligned, target
    has unexpected values, or test ids cannot be mapped to sample_submission.

## Текущий статус

**Public best: 78.261** (ROC-AUC × 100). Пайплайн реализован и прогнан end-to-end. Стадия: ансамбль Embedding bi-GRU + GBDT.

### Результаты (5-fold OOF ROC-AUC)

| модель | OOF ROC-AUC | роль |
|---|---|---|
| LightGBM (baseline) | 0.7548 | компонента ансамбля |
| CatBoost | 0.7548 | компонента ансамбля |
| Sequence GRU (scaled-numeric) | 0.7547 | закрыт |
| **Embedding bi-GRU** | **0.7798** | **champion** |
| Ансамбль v1 (LGBM+Cat+GRU) | 0.7645 | закрыт (заменён v2) |
| **Ансамбль v2 (LGBM+Cat+EmbGRU)** | **0.7808** | **финальный** |

### Загрузки и реальные результаты

| дата | сабмит | OOF | Public AUC | разрыв |
|---|---|---|---|---|
| 06-18 | `lgbm_baseline` | 0.7548 | 75.591 | +0.001 |
| 06-18 | `ens_lgb_cat_seq` (v1) | 0.7645 | 76.456 | −0.000 |
| 06-19 | `ens_lgb_cat_seqemb` (v2) | 0.7808 | **78.261** | **+0.002** |

**Ключевой вывод по валидации:** Public почти точно совпадает с OOF (даже чуть выше), а time-holdout (~0.73) оказался переоценкой дрейфа. Значит **random-CV OOF — надёжный, слегка консервативный прокси лидерборда**: улучшая OOF, мы прямо улучшаем Public. Отбор моделей ведём по OOF, time-holdout не используем как критерий.

### Ключевые находки

- GBDT-ветка (агрегаты) вышла на плато ~0.755 — ручные производные фичи (`v2`/`v3`) дали прирост в пределах шума.
- **Embedding bi-GRU — главный рычаг:** +0.025 OOF к одиночной модели и +1.8 пункта Public (76.456 → 78.261).
- Ансамбль v2 (Emb-GRU 84% + CatBoost 10% + LGBM 6%) даёт **0.7808** OOF / **78.261** Public.
- Опасения по временному дрейфу не подтвердились на реальном тесте — Public ≥ OOF.
- Sentinel-кодирование пропусков (флаги + фиксированные бины) обрабатывается нативно; `NaN` в данных нет.

### Закрытые направления

- Агрегатные feature sets `v2`/`v3` (term_gaps, delinq_share, paym_trend, term_clean) — δ OOF ≤ 0.0002, шум.
- Scaled-numeric GRU (без эмбеддингов) — 0.7547, заменён Embedding bi-GRU.

### Открытые направления

Подробный анализ с экспериментами — в [SOLUTION.md](SOLUTION.md) §6. Кратко по приоритету:

1. Архитектура sequence-модели (attention/Transformer, capacity, max_len=55) — наибольший headroom, т.к. это доминирующая модель.
2. Multi-seed averaging Embedding bi-GRU — дёшево, надёжно +.
3. Stacking: OOF-эмбеддинги bi-GRU как фичи для GBDT.
4. XGBoost и доп. seed-модели для диверсификации ансамбля.

Финальный сабмит ансамбля — в `submissions/`.
