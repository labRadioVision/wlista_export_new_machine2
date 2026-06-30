#!/usr/bin/env bash
# run_inference_sweep_ken_muscle_newpos.sh
# Inferenza su TUTTE le epoche disponibili, posizioni 15..21 (hz=0.847),
# dataset Ken_muscle_newpos.
#
# Uso:
#   bash run_inference_sweep_ken_muscle_newpos.sh
#   bash run_inference_sweep_ken_muscle_newpos.sh 2>&1 | tee inference_sweep_ken_muscle_newpos.log

set -e
PYTHON="D:/holography_scripts/.conda/python.exe"
SCRIPT="loop_inference_ken_muscle_newpos.py"

echo "========================================================"
echo "  W-LISTA Ken_muscle_newpos - inference sweep pos 15..21"
echo "========================================================"
"$PYTHON" "$SCRIPT" \
    --prefix wlista_ken_muscle_newpos \
    --ckpt-dir checkpoints_lista_ken_muscle_newpos \
    --start-epoch 1

echo ""
echo "=== Sweep Ken_muscle_newpos completato ==="
