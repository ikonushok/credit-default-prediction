#!/usr/bin/env bash
set -euo pipefail

# ============================================================================
# reproduce.sh — воспроизведение финального решения от data/raw/ до submission
#
# Результат: submissions/final/submission.csv
#            OOF ROC-AUC = 0.78340, public LB = 0.7835
#            (5-way rank-blend: avg-bigru + transformer + attn + lgbm + cat)
#
# Требования:
#   - Python 3.11+ с пакетами из requirements.txt (torch с MPS/CUDA для seq-моделей)
#   - data/raw/train_data.parquet, data/raw/test_data.parquet, data/raw/train_target.csv
#
# Время: GBDT-ветка ~10 мин (CPU); seq-ветка — 5 прогонов × ~3-4ч на Apple MPS
#        (3 сида bi-GRU + attn + transformer). Итого ~15-20ч на M3 Pro.
#        seq-прогоны независимы — можно гонять параллельно при наличии памяти.
# ============================================================================

REPO=$(cd "$(dirname "$0")" && pwd)
cd "$REPO"

PY="${PY:-python3}"
echo "Python: $($PY --version)"
export PYTHONPATH="$REPO/scripts:${PYTHONPATH:-}"

# --- проверка данных ---
for f in data/raw/train_data.parquet data/raw/test_data.parquet data/raw/train_target.csv; do
    [ -f "$f" ] || { echo "ОШИБКА: не найден $f"; exit 1; }
done
echo "Данные на месте."

latest() { basename "$(ls -dt artifacts/$1 | head -1)"; }

# ============================================================================
# Шаг 1. Агрегаты на id (baseline feature set) → train/test_features
# ============================================================================
echo ""
echo "=== Шаг 1/8: агрегация истории в признаки на id ==="
$PY scripts/02_aggregate.py --feature-set baseline
# 07/09 читают неверсионированный train_features.parquet; гарантируем его наличие
[ -f data/processed/train_features.parquet ] || \
    cp data/processed/train_features__baseline.parquet data/processed/train_features.parquet
[ -f data/processed/test_features.parquet ] || \
    cp data/processed/test_features__baseline.parquet data/processed/test_features.parquet

# ============================================================================
# Шаг 2. GBDT-ветка (диверсификаторы, вес в финале ~0): LightGBM + CatBoost
# ============================================================================
echo ""
echo "=== Шаг 2/8: LightGBM + CatBoost на агрегатах ==="
$PY scripts/03_train.py --config configs/lgbm_baseline.yaml
$PY scripts/03_train.py --config configs/catboost.yaml
LGBM=$(latest "*_lgbm_baseline")
CAT=$(latest "*_catboost_baseline")

# ============================================================================
# Шаг 3. Sequence bi-GRU, seed 42 (база) — per-column embeddings + biGRU
# ============================================================================
echo ""
echo "=== Шаг 3/8: Embedding bi-GRU (seed 42) ==="
$PY scripts/06_train_seq.py --config configs/sequence_emb.yaml
BIGRU42=$(latest "*_sequence_emb_bigru")

# ============================================================================
# Шаг 4. bi-GRU сиды 101 и 202 (те же фолды, варьируется только init модели)
# ============================================================================
echo ""
echo "=== Шаг 4/8: bi-GRU сиды 101, 202 (мульти-сид) ==="
$PY scripts/06_train_seq.py --config configs/sequence_emb.yaml --model-seed 101
$PY scripts/06_train_seq.py --config configs/sequence_emb.yaml --model-seed 202
BIGRU101=$(latest "*_sequence_emb_bigru_s101")
BIGRU202=$(latest "*_sequence_emb_bigru_s202")

# ============================================================================
# Шаг 5. Усреднение 3 сидов bi-GRU → доминирующая seq-база (OOF 0.78248)
# ============================================================================
echo ""
echo "=== Шаг 5/8: усреднение 3 сидов bi-GRU ==="
$PY scripts/09_avg_seeds.py --runs "$BIGRU42" "$BIGRU101" "$BIGRU202" \
    --name sequence_emb_bigru_avg
BIGRU_AVG=$(latest "*_sequence_emb_bigru_avg")

# ============================================================================
# Шаг 6. attention-pooling bi-GRU (диверсификатор, OOF 0.77878)
# ============================================================================
echo ""
echo "=== Шаг 6/8: attention bi-GRU ==="
$PY scripts/06_train_seq.py --config configs/sequence_emb_v2.yaml
ATTN=$(latest "*_sequence_emb_attn")

# ============================================================================
# Шаг 7. Transformer-энкодер (декоррелированный диверсификатор, OOF 0.77447)
# ============================================================================
echo ""
echo "=== Шаг 7/8: Transformer encoder (seed 42) ==="
$PY scripts/06_train_seq.py --config configs/sequence_emb_transformer.yaml
TRANSF=$(latest "*_sequence_emb_transformer")

# ============================================================================
# Шаг 8. Rank-blend 5 баз (Nelder-Mead по OOF) → submission
# ============================================================================
echo ""
echo "=== Шаг 8/8: rank-blend → submission ==="
$PY scripts/07_ensemble.py --runs \
    "$LGBM" "$CAT" "$BIGRU_AVG" "$ATTN" "$TRANSF" \
    --name ens_5way_seqavg_attn_transf
ENS=$(latest "*_ens_5way_seqavg_attn_transf")

$PY scripts/04_predict_submit.py --run "$ENS"

mkdir -p submissions/final
cp "submissions/$ENS.csv" submissions/final/submission.csv
SHA=$($PY -c "import hashlib;print(hashlib.sha256(open('submissions/final/submission.csv','rb').read()).hexdigest())")

echo ""
echo "============================================================================"
echo "ГОТОВО. Финальный submission: submissions/final/submission.csv"
echo "  run_id : $ENS"
echo "  SHA256 : $SHA"
echo "  Ожидаемый OOF ROC-AUC = 0.78340, public LB ≈ 0.7835"
echo "============================================================================"
