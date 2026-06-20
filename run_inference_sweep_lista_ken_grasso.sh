#!/usr/bin/env bash
# run_inference_sweep_lista_ken_grasso.sh
# Inferenza su TUTTE le epoche disponibili e TUTTE le posizioni del dataset
# Ken_grasso_nowalls, per il modello LISTA (caso base).
#
# Uso:
#   bash run_inference_sweep_lista_ken_grasso.sh
#   nohup bash run_inference_sweep_lista_ken_grasso.sh > inference_sweep_lista.log 2>&1 &

set -e
SCRIPT_DIR="$(cd "$(dirname "${BASH_SOURCE[0]}")" && pwd)"
cd "$SCRIPT_DIR"

PYTHON="python3"
SCRIPT="loop_inference_ken_grasso.py"

echo "========================================================"
echo "  LISTA (caso base) — inference sweep Ken_grasso"
echo "========================================================"
$PYTHON $SCRIPT \
    --prefix lista_ken_grasso \
    --ckpt-dir checkpoints_lista_plain_ken_grasso \
    --start-epoch 1

echo ""
echo "=== Sweep LISTA completato ==="
