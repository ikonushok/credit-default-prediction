# Credit Default Prediction

Проект для задачи Альфа-Банка × МФТИ «Кредитный скоринг» (прогноз дефолта).

**Финальный результат: OOF ROC-AUC = 0.78391 → Public LB = 0.7836.**

Подробное описание решения — в [SOLUTION.md](SOLUTION.md). Разведочный анализ данных — в [EDA-ноутбуке](notebooks/eda_final.ipynb).

## Quick Start — воспроизведение результата

```bash
# 1. Установить зависимости (нужен torch с MPS/CUDA для seq-моделей)
pip install -r requirements.txt

# 2. Положить данные
#    data/raw/train_data.parquet
#    data/raw/test_data.parquet
#    data/raw/train_target.csv

# 3. Запустить воспроизведение
bash reproduce.sh

# 4. Результат
#    submissions/final/submission.csv
```

Скрипт `reproduce.sh` последовательно строит агрегаты на `id`, обучает LightGBM + CatBoost, три сида bi-GRU (+ усреднение), attention-GRU и Transformer-энкодер, затем собирает 5-way rank-percentile blend. GBDT-ветка — минуты на CPU; seq-ветка — 5 прогонов по ~3-4ч на Apple MPS (итого ~15-20ч).

## Цель проекта

Построить воспроизводимую ML-модель, которая прогнозирует вероятность выхода клиента в дефолт по кредиту (неплатёж более 3 месяцев в течение года) по истории его кредитных продуктов.

- Целевая переменная: `flag` (1 — дефолт, 0 — выплатил), доля дефолтов ≈ 3.55%.
- Метрика: ROC-AUC (PR-AUC — вспомогательная проверка).
- Итоговый артефакт: CSV с вероятностью дефолта на каждый `id`, совместимый с `sample_submission.csv`.

## Данные

    data/raw/train_data.parquet      # обучающая история кредитных продуктов (long)
    data/raw/test_data.parquet       # тестовая история (long)
    data/raw/train_target.csv        # метки id, flag для train
    data/Кредитный скорринг/sample_submission.csv   # формат сабмита

Формат **long**: один `id` = несколько строк, по одной на кредитный продукт, упорядоченных по `rn`. Метка — одна на `id`. Размеры: train ≈ 18.3M строк / 2.10M `id`, test ≈ 7.85M / 0.90M `id`. Все 59 признаков — целочисленные категориальные коды. Файлы с данными не коммитятся в Git.

## Ограничения

- Python 3.10+, только open-source библиотеки.
- Используются только данные, предоставленные в рамках задания.
- Нельзя использовать `flag` как признак; он джойнится по `id` только в train.
- Нельзя использовать `id`/`rn` как сырые предикторы (это порядковый временной индекс).
- Нельзя использовать test labels или public leaderboard для подгонки модели.
- Сабмиты на платформу — ограниченный ресурс.

## Структура проекта

    data/raw/              # исходные parquet/csv, не коммитятся
    data/processed/        # агрегированные признаки на id (train/test_features*.parquet)

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
        sequence.py        # Embedding bi-GRU / attention / Transformer (torch/MPS)

    scripts/               # CLI-скрипты для запусков (01..09)
    configs/               # YAML-конфиги прогонов
    experiments/           # experiment_log.csv, cards/, отчёты и логи
    artifacts/             # per-run: folds, OOF/test predictions, metrics, config
    submissions/           # финальные CSV-сабмиты и папки upload
    notebooks/             # eda_final.ipynb
    agents/                # 21 специализированный агент (см. agents/README.md)

## Текущая реализация

Рабочий контур — набор воспроизводимых CLI-скриптов и per-run артефактов (а не один entrypoint):

- `scripts/01_eda.py` — разведочный анализ, `experiments/eda_report.md`;
- `scripts/02_aggregate.py` — long → вектор признаков на `id`;
- `scripts/03_train.py` — GBDT (LightGBM / CatBoost) по `id`-фолдам;
- `scripts/06_train_seq.py` — seq-модели (bi-GRU / attention / Transformer), флаг `--model-seed` для мульти-сида;
- `scripts/09_avg_seeds.py` — усреднение сидов одной архитектуры в низкодисперсную базу;
- `scripts/07_ensemble.py` — rank-percentile blend (Nelder-Mead по OOF);
- `scripts/04_predict_submit.py` — сборка и валидация submission;
- `scripts/05_compare.py` — сравнение прогонов на одних фолдах;
- `scripts/08_stack_autogluon.py` — AutoGluon-мета-стекер (проверен, повторяет rank-blend — закрыт).

Ключевые подтверждённые артефакты:

- финальный 5-way бленд: `artifacts/20260620_153734_ens_5way_seqavg_attn_transfavg3`;
- усреднённые базы (3 сида): bi-GRU `artifacts/20260620_064150_sequence_emb_bigru_avg`, transformer `artifacts/20260620_153505_sequence_emb_transformer_avg3`;
- карточки сабмитов: `experiments/cards/`; журнал: `experiments/experiment_log.csv`;
- кандидаты на загрузку: `submissions/upload_20260620/`.

## Рекомендуемый порядок работы

### 1. Data quality / EDA

Перед моделированием проверяется: размеры train/test/target/sample; наличие `flag` только в `train_target.csv`; совместимость схем (61 целочисленная колонка); распределение rows-per-`id` (~8.7); совпадение множеств `id` (пересечение train/test = 0); дрейф по диапазону `id` (сплит временной); частоты `enc_loans_*` / `enc_paym_*`; sentinel-кодирование (`pclose_flag=1` → `pre_pterm=4`).

    python scripts/01_eda.py        # -> experiments/eda_report.md

Визуальный разбор — в [notebooks/eda_final.ipynb](notebooks/eda_final.ipynb).

### 2. Агрегация long → id

Каждый `id` сворачивается в один вектор (агрегация строго внутри `id`). Feature sets заданы реестром блоков в `features.py`, результат версионируется.

    python scripts/02_aggregate.py --feature-set baseline

### 3. Validation design

`StratifiedKFold(5, seed=42)` по `id` после агрегации (одна строка = один `id`, утечка строк невозможна). Те же фолды используют и seq-модели для ансамблируемости. Дополнительно — **time-holdout** (поздние 20% `id`) как диагностика дрейфа. Fold assignments детерминированы по seed.

**Ключевой вывод валидации:** реальный Public совпал с random-CV OOF (гэп ≈0 на финале), а time-holdout оказался переоценкой дрейфа. Поэтому **отбор моделей ведём по OOF** — приросты переносятся на LB без траты загрузок.

### 4. Baseline model

LightGBM / CatBoost на 182 агрегатах (`scale_pos_weight≈27` / `auto_class_weights=Balanced`), per-fold + OOF ROC-AUC, time-holdout cross-check, сохранённые folds/OOF/test/config/metrics.

    python scripts/03_train.py --config configs/lgbm_baseline.yaml
    python scripts/03_train.py --config configs/catboost.yaml

### 5. Feature engineering

Рабочие признаки (baseline, 182 агрегата): volume/recency, delinquency, payment-status (`enc_paym_*`), amounts/utilization (ordinal-бины), closure/status rates, categorical diversity, last-product.

Проверены и **закрыты** (ablation на тех же фолдах, δ OOF ≤ 0.0002): `term_gaps`, `delinq_share`, `paym_trend`, `term_clean`. Вывод: **GBDT на плато ~0.755** — ручные агрегаты не добавляют сигнала; основной рычаг — в порядке продуктов (sequence-модель), который агрегаты уничтожают.

    python scripts/05_compare.py --base <baseline_run> --cand <cand_run>

### 6. Sequence models (главный рычаг)

Каждый `id` — последовательность продуктов (порядок `rn`). Все 59 колонок — категориальные коды, поэтому каждая получает собственный обучаемый эмбеддинг (общая `nn.Embedding` + per-column offsets). Три архитектуры над общим фронт-эндом:

- **Embedding bi-GRU** — last-hidden pooling. OOF **0.77982** (+0.025 к scaled-numeric GRU — это и есть главный рычаг качества).
- **Attention bi-GRU** — пуллинг вниманием. OOF 0.77878 (диверсификатор).
- **Transformer-энкодер** — self-attention, learned positions. OOF 0.77447 (слабее, но декоррелирован 0.943).

```bash
python scripts/06_train_seq.py --config configs/sequence_emb.yaml              # bi-GRU
python scripts/06_train_seq.py --config configs/sequence_emb_v2.yaml           # attention
python scripts/06_train_seq.py --config configs/sequence_emb_transformer.yaml  # transformer
```

**Мульти-сид:** bi-GRU и transformer обучаются на сидах 42/101/202 при фиксированных фолдах (варьируется только init через `--model-seed`), затем усредняются. bi-GRU: OOF **0.78248** (+0.0019). Transformer: OOF **0.77955** (+0.0051 к одиночному).

```bash
python scripts/06_train_seq.py --config configs/sequence_emb.yaml --model-seed 101
python scripts/06_train_seq.py --config configs/sequence_emb_transformer.yaml --model-seed 101
python scripts/09_avg_seeds.py --runs <seed_42> <seed_101> <seed_202> --name sequence_emb_bigru_avg
```

### 7. Ensemble

Rank-percentile blend: предсказания каждой базы ранг-нормализуются (0..1) для выравнивания масштабов (GBDT-вероятности vs NN-сигмоида); веса — Nelder-Mead по OOF ROC-AUC; OOF/test выравниваются по **значению `id`**.

    python scripts/07_ensemble.py --runs <lgbm> <cat> <bigru_avg> <attn> <transf> --name ens

Финальный 5-way бленд: **avg_bigru 0.535 + avg_transformer 0.318 + attn 0.120 + lgbm 0.027 + cat 0.0** → OOF **0.78391**. Мульти-сид transformer усилился с 0.21 до 0.32 веса — мульти-сид снизил дисперсию и повысил полезность декоррелированной базы. AutoGluon-мета и stacking через агрегаты проверены — не бьют линейный rank-blend.

### 8. Submission build

Проверки перед отправкой: колонки совпадают с `sample_submission.csv` (`id`, `flag`); 900 000 строк; `id`-mapping; вероятности в `[0, 1]`; нет NaN/inf; уникальное имя; SHA256 в submission card.

    python scripts/04_predict_submit.py --run <run_id>
    # -> submissions/<run_id>.csv + experiments/cards/<run_id>.md

## Валидационные уровни

- L0 — статическая проверка файлов и документации.
- L1 — smoke/syntax checks.
- L2 — проверка данных, схем, long-формата и покрытия `id`.
- L3 — воспроизводимая CV-валидация с OOF ROC-AUC (id-level фолды).
- L4 — robustness checks, alternative splits, leakage review, ансамбль.
- L5 — submission readiness: sample-format check, hash, submission card, red-team review.

## Текущий статус

**Финальный Public ROC-AUC: 0.7836 (78.36).** Финал — 5-way rank-blend seq-моделей (обе seq-базы мульти-сид); GBDT-ветка на плато и весит ≈0.

### Результаты (5-fold OOF ROC-AUC)

| модель | OOF ROC-AUC | роль |
|---|---|---|
| LightGBM / CatBoost | 0.7548 | страховка (вес ≈0) |
| Embedding bi-GRU (seed 42) | 0.77982 | база мульти-сида |
| **avg bi-GRU (3 сида)** | **0.78248** | **champion** |
| attention bi-GRU | 0.77878 | диверсификатор |
| Transformer encoder (seed 42) | 0.77447 | база мульти-сида |
| **avg Transformer (3 сида)** | **0.77955** | **декоррелир. (w=0.318)** |
| **5-way бленд (финал)** | **0.78391** | **финальный** |

### Загрузки и реальные результаты

| дата | сабмит | OOF | Public AUC | разрыв |
|---|---|---|---|---|
| 06-18 | `lgbm_baseline` | 0.7548 | 75.591 | +0.001 |
| 06-18 | `ens_lgb_cat_seq` (v1) | 0.7645 | 76.456 | −0.000 |
| 06-19 | `ens_lgb_cat_seqemb` (v2) | 0.7808 | 78.261 | +0.002 |
| 06-20 | `ens_5way_seqavg_attn_transf` | 0.78340 | 78.35 | +0.000 |
| **06-20** | **`ens_5way_seqavg_attn_transfavg3`** | **0.78391** | **78.36** | **−0.000** |

### Ключевые находки

- **Эмбеддинги seq — главный рычаг:** +0.025 OOF (scaled-numeric → emb bi-GRU); порядок продуктов несёт сигнал, который агрегаты теряют.
- **Мульти-сид bi-GRU:** +0.0019; усреднённая база бьёт прежний бленд в одиночку.
- **Мульти-сид transformer:** +0.0051 к одиночному; вес в бленде вырос 0.21→0.32, бленд +0.0005.
- **OOF — надёжный прокси Public** (гэп ≈0): приросты принимаются по OOF без траты загрузок.
- **Шумовой пол LB:** +0.0005 OOF → всего +0.0001 LB; суб-0.001 улучшения OOF не конвертируются.

### Закрытые направления

- Агрегатные feature sets `v2`/`v3` — δ OOF ≤ 0.0002, шум.
- Scaled-numeric GRU (без эмбеддингов) — 0.7547, заменён эмбеддингами.
- **AutoGluon как блендер** — OOF-стек повторяет rank-blend (0.78201), обогащение агрегатами вредит (≈−0.0005). Горлышко не в блендере, а в базах.
- **Мульти-сид transformer** — реализован, дал +0.0005 OOF / +0.0001 LB; мы на шумовом полу.

### Открытые направления

Подробный анализ — в [SOLUTION.md](SOLUTION.md) §6. Кратко по приоритету:

1. TCN — третья декоррелированная архитектура для диверсификации.
2. Stacking: OOF-эмбеддинги seq как фичи для GBDT.

Финальный сабмит — `submissions/upload_20260620/20260620_153734_ens_5way_seqavg_attn_transfavg3.csv`.
