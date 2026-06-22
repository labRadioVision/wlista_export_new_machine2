#!/usr/bin/env bash
# run_inference_sweep_synthetic_nowalls_gpu.sh
# Inferenza su TUTTE le epoche disponibili e TUTTE le 11 posizioni del caso
# sintetico nowalls (PEC/freespace), per i checkpoint prodotti da
# run_wlista_synthetic_nowalls_gpu.py:
#   - LISTA plain   (--model lista)
#   - W-LISTA       (--model wlista)
# Entrambi salvano in checkpoints_lista/ con prefisso <model>_synthetic_nowalls_gpu.
#
# Uso:
#   bash run_inference_sweep_synthetic_nowalls_gpu.sh
#   nohup bash run_inference_sweep_synthetic_nowalls_gpu.sh > inference_sweep_synthetic_nowalls_gpu.log 2>&1 &

set -e
SCRIPT_DIR="$(cd "$(dirname "${BASH_SOURCE[0]}")" && pwd)"
cd "$SCRIPT_DIR"

PYTHON="python3"
SCRIPT="loop_inference_synthetic_nowalls.py"
CKPT_DIR="checkpoints_lista"

for MODEL in lista wlista; do
    PREFIX="${MODEL}_synthetic_nowalls_gpu"
    if ! ls "${CKPT_DIR}/${PREFIX}_ep"*.pt >/dev/null 2>&1; then
        echo "Nessun checkpoint per ${MODEL} (${PREFIX}) — salto."
        continue
    fi
    echo "========================================================"
    echo "  ${MODEL^^} — inference sweep synthetic_nowalls"
    echo "========================================================"
    $PYTHON $SCRIPT \
        --prefix "$PREFIX" \
        --ckpt-dir "$CKPT_DIR" \
        --start-epoch 1
done

echo ""
echo "=== Sweep LISTA/W-LISTA (synthetic_nowalls) completato ==="
