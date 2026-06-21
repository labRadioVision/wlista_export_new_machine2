#!/usr/bin/env bash
# run_inference_sweep_phased_ken_grasso.sh
# Inferenza su TUTTE le epoche disponibili e TUTTE le posizioni del dataset
# Ken_grasso_nowalls, per LR-W-LISTA PHASED (rank=8), separatamente per
# ciascuna fase (A, B, C) — i checkpoint hanno il tag di fase nel nome
# (es. wlista_lowrank_phased_ken_grasso_r8_B_ep005.pt).
#
# Uso:
#   bash run_inference_sweep_phased_ken_grasso.sh
#   nohup bash run_inference_sweep_phased_ken_grasso.sh > inference_sweep_phased.log 2>&1 &

set -e
SCRIPT_DIR="$(cd "$(dirname "${BASH_SOURCE[0]}")" && pwd)"
cd "$SCRIPT_DIR"

PYTHON="python3"
SCRIPT="loop_inference_ken_grasso.py"
CKPT_DIR="checkpoints_lista_lowrank_phased_ken_grasso"
BASE_PREFIX="wlista_lowrank_phased_ken_grasso_r8"

for PHASE in A B C; do
    PREFIX="${BASE_PREFIX}_${PHASE}"
    # salta la fase se non ci sono checkpoint (es. fase C non lanciata)
    if ! ls "${CKPT_DIR}/${PREFIX}_ep"*.pt >/dev/null 2>&1; then
        echo "Nessun checkpoint per fase ${PHASE} (${PREFIX}) — salto."
        continue
    fi
    echo "========================================================"
    echo "  PHASED fase ${PHASE} — inference sweep Ken_grasso"
    echo "========================================================"
    $PYTHON $SCRIPT \
        --prefix "$PREFIX" \
        --ckpt-dir "$CKPT_DIR" \
        --start-epoch 1
done

echo ""
echo "=== Sweep PHASED completato ==="
