#!/usr/bin/env bash
# run_inference_sweep_phased_synthetic_nowalls.sh
# Inferenza su TUTTE le epoche disponibili e TUTTE le posizioni del caso
# sintetico nowalls (PEC/freespace), per LR-W-LISTA PHASED (rank=8),
# separatamente per ciascuna fase (A, B, C) — i checkpoint hanno il tag di
# fase nel nome (es. wlista_lowrank_phased_nowalls_r8_B_ep005.pt).
#
# Uso:
#   bash run_inference_sweep_phased_synthetic_nowalls.sh
#   nohup bash run_inference_sweep_phased_synthetic_nowalls.sh > inference_sweep_phased_synthetic_nowalls.log 2>&1 &

set -e
SCRIPT_DIR="$(cd "$(dirname "${BASH_SOURCE[0]}")" && pwd)"
cd "$SCRIPT_DIR"

PYTHON="python3"
SCRIPT="loop_inference_synthetic_nowalls.py"
CKPT_DIR="checkpoints_lista_lowrank_phased_synthetic_nowalls"
BASE_PREFIX="wlista_lowrank_phased_nowalls_r8"

for PHASE in A B C; do
    PREFIX="${BASE_PREFIX}_${PHASE}"
    # salta la fase se non ci sono checkpoint (es. fase C non lanciata)
    if ! ls "${CKPT_DIR}/${PREFIX}_ep"*.pt >/dev/null 2>&1; then
        echo "Nessun checkpoint per fase ${PHASE} (${PREFIX}) — salto."
        continue
    fi
    echo "========================================================"
    echo "  PHASED fase ${PHASE} — inference sweep synthetic_nowalls"
    echo "========================================================"
    $PYTHON $SCRIPT \
        --prefix "$PREFIX" \
        --ckpt-dir "$CKPT_DIR" \
        --start-epoch 1
done

echo ""
echo "=== Sweep PHASED (synthetic_nowalls) completato ==="
