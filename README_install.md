# W-LISTA nowalls — installation on a new machine

Minimal package to train / run **W-LISTA / LISTA** on the synthetic *nowalls*
data (PEC, free-space) using the file pair:

- `E_total_freespace_nowalls.mat` (incident / free-space field)
- `E_total_Ken_PEC_nowalls.mat` (total field, 11 positions)

Entry-point script: **`run_wlista_synthetic_nowalls_gpu.py`** (full-GPU).

This version is **self-contained**: it does NOT require the real measurement
file `empty_20_11_2024.mat`. The RX geometry (162x80 grid on the z=0 plane)
and the frequency (2.45 GHz) are generated internally.

---

## Step-by-step quick start

Requires an **NVIDIA GPU** (driver >= 560, CUDA 12.x) and **Miniconda/Anaconda**.

**Step 1 — Get the files onto the machine.**
Copy the whole `wlista_export_new_machine/` folder. It already contains the
scripts, the dependencies, and the two `.mat` files under
`Dataset TUM/sinthetic_data/`. Open a terminal in that folder:

```bat
cd path\to\wlista_export_new_machine
```

**Step 2 — Create and activate a Python 3.11 environment.**

```bat
conda create -n holo_wlista python=3.11 -y
conda activate holo_wlista
```

**Step 3 — Install PyTorch for the machine's CUDA** (CUDA 12.x example):

```bat
pip install torch --index-url https://download.pytorch.org/whl/cu121
```

**Step 4 — Install the remaining dependencies.**

```bat
pip install -r requirements.txt
```

**Step 5 — Verify the GPU stack is visible.**

```bat
python -c "import torch, cupy; print('torch CUDA:', torch.cuda.is_available()); print('cupy dev:', cupy.cuda.runtime.getDeviceCount())"
```

Expected: `torch CUDA: True` and at least one cupy device.

**Step 6 — (only if data is elsewhere) point to the dataset.**
The data is already bundled, so normally you can skip this. To use a different
location:

```bat
set HOLO_SYNTH_DIR=D:\path\to\data        :: Windows
export HOLO_SYNTH_DIR=/path/to/data       # Linux/Mac
```

**Step 7 — Launch the training.**

```bat
python run_wlista_synthetic_nowalls_gpu.py > wlista_synthetic_nowalls_gpu.log 2>&1
```

Training **auto-resumes by default**: on every launch it continues from the
highest-epoch checkpoint in `checkpoints_lista/` (or starts from scratch if
none exists). To force a clean run from scratch, add `--fresh`. To resume from
a specific checkpoint, pass `--resume <path>`.

**Step 8 — Collect the results.**
Output goes to `results_synthetic_nowalls_gpu/` (PNG/npz/mat + per-epoch
reconstructions in `epoch_recon/`). Checkpoints go to `checkpoints_lista/`
(`<model>_synthetic_nowalls_gpu_best.pt`). Progress is in the `.log` file.

---

## Dataset resolution order

The dataset folder is resolved in this order: `HOLO_SYNTH_DIR` > local
(`<this folder>/Dataset TUM/sinthetic_data`) > parent
(`../Dataset TUM/sinthetic_data`). So the bundled local data is used first,
with a fallback to the parent folder for backward compatibility.

## Included files

| File | Role |
|------|------|
| `run_wlista_synthetic_nowalls_gpu.py` | **entry point** (training/inference, GPU) |
| `run_wlista_synthetic_nowalls.py` | base module: nowalls data loader + plotting |
| `holography_operator_fast.py` | A/AH operator, GPU (torch + cupy, DLPack zero-copy) |
| `holography_operator.py` | A/AH operator (CuPy, used by base if imported) |
| `holography_operator_numpy.py` | A/AH operator, pure NumPy (CPU fallback) |
| `lista_holography.py` | LISTA network |
| `lista_holography_weighted.py` | W-LISTA network |
| `lista_holography_lowrank.py` | LR-W-LISTA network (required by inference_common) |
| `inference_common.py` | MF/ISTA baselines, metrics, saving |
| `generate_z_true.py` | z_true from a parametric body model |
| `requirements.txt` | pip dependencies |

## About z_true

The `z_true` (reference permittivity) for each position is **generated** by the
parametric body model (`make_z_true_body_model`) translated to the mannequin
position, and cached in `results_z_true_nowalls/`. No external z_true file is
required.

## Coordinate mapping (FEKO -> holographic grid)

```
holo_x = 2.510 + FEKO_y
holo_z = 3.317 - FEKO_x
```

Imaging grid: X[0-5] (161), Y[0-2.5] (81), Z[0.3-2.3] (65).

## Implementation status per case — what's missing

Two datasets/cases in the repo: **Ken_grasso** (`E_total_Ken_grasso_nowalls.mat`)
and **Ken_nowalls** (PEC/freespace — `E_total_Ken_PEC_nowalls.mat` +
`E_total_freespace_nowalls.mat`, the one bundled by default in this
folder). Ken_grasso already has every variant implemented; Ken_nowalls is
behind.

| Variant | Ken_grasso | Ken_nowalls (PEC/freespace) |
|---|---|---|
| LISTA plain | `run_lista_ken_grasso.py` | `run_wlista_synthetic_nowalls_gpu.py --model lista` ✓ |
| W-LISTA | `run_wlista_ken_grasso.py` | `run_wlista_synthetic_nowalls_gpu.py --model wlista` ✓ |
| LR-W-LISTA PHASED | `run_wlista_lowrank_phased_ken_grasso.py` | `run_wlista_lowrank_phased_synthetic_nowalls.py` ✓ |
| Inference sweep — LISTA/W-LISTA plain | `run_inference_sweep_lista_ken_grasso.sh`, `run_inference_sweep_wlista_ken_grasso.sh` | `run_inference_sweep_synthetic_nowalls_gpu.sh` ✓ |
| Inference sweep — PHASED | `run_inference_sweep_phased_ken_grasso.sh` | `run_inference_sweep_phased_synthetic_nowalls.sh` ✓ |

### Run examples (Ken_nowalls)

```bash
# LISTA plain
nohup python3 run_wlista_synthetic_nowalls_gpu.py --model lista > lista_synthetic_nowalls_gpu.log 2>&1 &

# W-LISTA (--model wlista è il default, può anche essere omesso)
nohup python3 run_wlista_synthetic_nowalls_gpu.py --model wlista > wlista_synthetic_nowalls_gpu.log 2>&1 &

# LR-W-LISTA PHASED (rank=8 di default)
nohup python3 run_wlista_lowrank_phased_synthetic_nowalls.py > phased_synthetic_nowalls.log 2>&1 &
```

Per i dettagli (resume, --fresh, --skip-a/--skip-b, ecc.) vedi `README_RUN.md`.

Tutte le varianti previste per Ken_nowalls (LISTA, W-LISTA, LR-W-LISTA
PHASED + relativi inference sweep) sono coperte. LR-W-LISTA joint e
W-FIRST non sono previsti per questo caso.

