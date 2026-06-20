#!/usr/bin/env bash
# run_inference_sweep_wlista_ken_grasso.sh
# Inferenza su TUTTE le epoche disponibili e TUTTE le posizioni del dataset
# Ken_grasso_nowalls, per:
#   - W-LISTA baseline (pesi per-asse)
#   - LR-W-LISTA joint training (rank=8, default di run_lowrank_ken_grasso.py)
#
# Uso:
#   bash run_inference_sweep_wlista_ken_grasso.sh
#   nohup bash run_inference_sweep_wlista_ken_grasso.sh > inference_sweep_wlista.log 2>&1 &

set -e
SCRIPT_DIR="$(cd "$(dirname "${BASH_SOURCE[0]}")" && pwd)"
cd "$SCRIPT_DIR"

PYTHON="python3"
SCRIPT="loop_inference_ken_grasso.py"

echo "========================================================"
echo "  W-LISTA baseline — inference sweep Ken_grasso"
echo "========================================================"
$PYTHON $SCRIPT \
    --prefix wlista_ken_grasso \
    --ckpt-dir checkpoints_lista_ken_grasso \
    --start-epoch 1

echo ""
echo "========================================================"
echo "  LR-W-LISTA joint (rank=8) — inference sweep Ken_grasso"
echo "========================================================"
$PYTHON $SCRIPT \
    --prefix wlista_lowrank_ken_grasso_r8 \
    --ckpt-dir checkpoints_lista_lowrank_ken_grasso \
    --start-epoch 1

echo ""
echo "=== Sweep W-LISTA / LR-W-LISTA joint completato ==="
