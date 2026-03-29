#!/bin/bash
# =============================================================
# DIDI Dataset Experiments (Research Plan Phase 4: 4a-4d)
#
# Runs on SZ_DIDI and CD_DIDI:
#   - Zero-shot baseline (test mode)
#   - ICT aggregator training (3 epochs, K=3, KNN, attention)
#
# Usage (from OpenCity/):
#   bash run_didi_experiments.sh
# =============================================================
set -e

PROJ_DIR="$(cd "$(dirname "$0")" && pwd)"
cd "$PROJ_DIR"

PYTHON=".venv/bin/python"
CONF="conf/general_conf/pretrain.conf"
WEIGHTS_DIR="model_weights/OpenCity"
MAIN="main.py"

# Common model flags (OpenCity-Plus: 512 embed, 6 layers)
MODEL_FLAGS="-model OpenCity -load_pretrain_path OpenCity-plus.pth --embed_dim 512 --skip_dim 512 --enc_depth 6"

# ICT flags (matching PEMS07M Exp 8b: 3 epochs, K=3, KNN retrieval, attention aggregator)
ICT_FLAGS="-num_demonstrations 3 -demo_selection similar -aggregator_type attention -aggregator_epochs 3 -aggregator_lr 1e-4"

swap_dataset() {
    # Replace dataset_use line in pretrain.conf
    local ds="$1"
    sed -i "s/^dataset_use = .*/dataset_use = ['${ds}']/" "$CONF"
    echo "[CONFIG] dataset_use set to '${ds}'"
    grep '^dataset_use' "$CONF"
}

separator() {
    echo ""
    echo "============================================================"
    echo "  $1"
    echo "============================================================"
    echo ""
}

# ==========================
#  SZ_DIDI Experiments
# ==========================
swap_dataset "SZ_DIDI"

# 4a: SZ_DIDI zero-shot baseline
separator "4a: SZ_DIDI Zero-Shot Baseline"
$PYTHON $MAIN -mode test $MODEL_FLAGS -batch_size 2

# 4b: SZ_DIDI aggregator training + test
separator "4b: SZ_DIDI ICT Aggregator Training (3ep, K=3, KNN, Attention)"
$PYTHON $MAIN -mode ict_train_aggregator $MODEL_FLAGS -batch_size 16 $ICT_FLAGS

# Backup SZ_DIDI aggregator weights
if [ -f "$WEIGHTS_DIR/aggregator_best.pth" ]; then
    cp "$WEIGHTS_DIR/aggregator_best.pth" "$WEIGHTS_DIR/aggregator_best_SZ_DIDI.pth"
    echo "[BACKUP] aggregator_best.pth -> aggregator_best_SZ_DIDI.pth"
fi
# Clean checkpoint so CD_DIDI starts fresh
rm -f "$WEIGHTS_DIR/aggregator_ckpt.pth"
rm -f "$WEIGHTS_DIR/aggregator_best.pth"

# ==========================
#  CD_DIDI Experiments
# ==========================
swap_dataset "CD_DIDI"

# 4c: CD_DIDI zero-shot baseline
separator "4c: CD_DIDI Zero-Shot Baseline"
$PYTHON $MAIN -mode test $MODEL_FLAGS -batch_size 2

# 4d: CD_DIDI aggregator training + test
separator "4d: CD_DIDI ICT Aggregator Training (3ep, K=3, KNN, Attention)"
$PYTHON $MAIN -mode ict_train_aggregator $MODEL_FLAGS -batch_size 16 $ICT_FLAGS

# Backup CD_DIDI aggregator weights
if [ -f "$WEIGHTS_DIR/aggregator_best.pth" ]; then
    cp "$WEIGHTS_DIR/aggregator_best.pth" "$WEIGHTS_DIR/aggregator_best_CD_DIDI.pth"
    echo "[BACKUP] aggregator_best.pth -> aggregator_best_CD_DIDI.pth"
fi
rm -f "$WEIGHTS_DIR/aggregator_ckpt.pth"
rm -f "$WEIGHTS_DIR/aggregator_best.pth"

# ==========================
#  Restore config to PEMS07M
# ==========================
swap_dataset "PEMS07M"

separator "ALL DIDI EXPERIMENTS COMPLETE"
echo "Results logged to: $WEIGHTS_DIR/run.log"
echo "Saved weights:"
ls -lh "$WEIGHTS_DIR"/aggregator_best_*.pth 2>/dev/null || echo "  (none)"
