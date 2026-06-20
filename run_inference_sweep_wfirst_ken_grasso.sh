#!/usr/bin/env bash
# run_inference_sweep_wfirst_ken_grasso.sh
# Inferenza su TUTTE le epoche disponibili e TUTTE le posizioni del dataset
# Ken_grasso_nowalls, per LR-W-LISTA W-FIRST (rank=8, default di
# run_wlista_lowrank_wfirst_ken_grasso.py).
#
# Uso:
#   bash run_inference_sweep_wfirst_ken_grasso.sh
#   nohup bash run_inference_sweep_wfirst_ken_grasso.sh > inference_sweep_wfirst.log 2>&1 &

set -e
SCRIPT_DIR="$(cd "$(dirname "${BASH_SOURCE[0]}")" && pwd)"
cd "$SCRIPT_DIR"

PYTHON="python3"
SCRIPT="loop_inference_ken_grasso.py"

echo "========================================================"
echo "  LR-W-LISTA W-FIRST (rank=8) — inference sweep Ken_grasso"
echo "========================================================"
$PYTHON $SCRIPT \
    --prefix wlista_lowrank_wfirst_ken_grasso_r8 \
    --ckpt-dir checkpoints_lista_lowrank_wfirst_ken_grasso \
    --start-epoch 1

echo ""
echo "=== Sweep W-FIRST completato ==="
