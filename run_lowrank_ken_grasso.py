# -*- coding: utf-8 -*-
"""
run_lowrank_ken_grasso.py
=========================
LR-W-LISTA su E_total_Ken_grasso_nowalls.mat (dati sintetici FEKO, PEC senza muri).

Copia questo file nella stessa cartella degli altri moduli ed esegui:

    python run_lowrank_ken_grasso.py > lowrank_ken_grasso.log 2>&1
    python run_lowrank_ken_grasso.py --rank 8 --resume checkpoints_lista_lowrank_ken_grasso/wlista_lowrank_ken_grasso_r8_ep005.pt

Il file E_total_Ken_grasso_nowalls.mat deve essere in Dataset TUM/sinthetic_data/
(sottocartella della cartella in cui si trova questo script).
"""

import os, sys, time, argparse
sys.stdout.reconfigure(line_buffering=True)
import numpy as np
import scipy.io as sio
import torch
import cupy as cp

SCRIPT_DIR = os.path.dirname(os.path.abspath(__file__))
sys.path.insert(0, SCRIPT_DIR)

import run_wlista_synthetic_nowalls as base
from holography_operator_fast import HolographyOperatorFast
from lista_holography_lowrank  import LRWLISTAHolography
import inference_common as ic

# ---------------------------------------------------------------------------
# Override: file dati (Dataset TUM è DENTRO questa cartella)
# ---------------------------------------------------------------------------
SYNTH_DIR   = os.path.join(SCRIPT_DIR, "Dataset TUM", "sinthetic_data")
ETOTAL_FILE = os.path.join(SYNTH_DIR, "E_total_Ken_grasso_nowalls.mat")
base.ETOTAL_FILE = ETOTAL_FILE
base.SYNTH_DIR   = SYNTH_DIR

if not os.path.exists(ETOTAL_FILE):
    raise FileNotFoundError(
        f"File non trovato: {ETOTAL_FILE}\n"
        f"Assicurati che sia in {SYNTH_DIR}")

# --- auto-detect numero di posizioni dal file ---
_mat  = sio.loadmat(ETOTAL_FILE)
_keys = [k for k in _mat if not k.startswith("_")]
_n_pos = None
for _k in _keys:
    if _mat[_k].ndim == 3 and _mat[_k].shape[:2] == (162, 80):
        _n_pos = _mat[_k].shape[2]
        break
if _n_pos is None:
    _n_pos = _mat[_keys[0]].shape[-1]
print(f"[Ken_grasso] N_pos rilevato = {_n_pos}  (chiave usata: '{_k}')")

# --- split: tutte in train, ultima come val/plot ---
base.TRAIN_IDX  = list(range(_n_pos))
base.VAL_IDX    = [_n_pos - 1]
base.POS_LABELS = [f"ken_grasso_pos{i:02d}" for i in range(_n_pos)]

# ---------------------------------------------------------------------------
# Cartelle output dedicate
# ---------------------------------------------------------------------------
OUT_DIR  = os.path.join(SCRIPT_DIR, "results_synthetic_lowrank_ken_grasso")
CKPT_DIR = os.path.join(SCRIPT_DIR, "checkpoints_lista_lowrank_ken_grasso")
os.makedirs(OUT_DIR,  exist_ok=True)
os.makedirs(CKPT_DIR, exist_ok=True)
base.OUT_DIR  = OUT_DIR
base.CKPT_DIR = CKPT_DIR

# ---------------------------------------------------------------------------
# Iperparametri
# ---------------------------------------------------------------------------
NX, NY, NZ  = base.NX, base.NY, base.NZ
K           = base.K
N_EPOCHS    = base.N_EPOCHS
LR          = base.LR
LR_W        = base.LR_W
LAMBDA_INIT = base.LAMBDA_INIT
L_EST       = base.L_EST

RANK      = 8
LR_LR     = 1e-2
ALPHA_Z   = 1.0
BETA_DATA = 1e-4
GAMMA_REG = 1e-1

DEVICE = torch.device("cuda" if torch.cuda.is_available() else "cpu")


# ---------------------------------------------------------------------------
# Operator + loss + training
# ---------------------------------------------------------------------------
def build_operator_fast(rx_ref, k, omega):
    print("\nBuilding FAST operator (DLPack) ...")
    Nx_rx, Ny_rx = rx_ref["S21"].shape[:2]
    x_rx = rx_ref["X"][:, 0]; y_rx = rx_ref["Y"][0, :]
    XX, YY = np.meshgrid(x_rx, y_rx, indexing="ij")
    r_rx  = np.column_stack([XX.ravel(), YY.ravel(), np.zeros(Nx_rx * Ny_rx)])
    XXv, YYv, ZZv = np.meshgrid(base.X_IMG, base.Y_IMG, base.Z_IMG, indexing="ij")
    r_vox = np.column_stack([XXv.ravel(), YYv.ravel(), ZZv.ravel()])
    dV = (base.X_IMG[1]-base.X_IMG[0])*(base.Y_IMG[1]-base.Y_IMG[0])*(base.Z_IMG[1]-base.Z_IMG[0])
    print(f"  Receivers : {r_rx.shape[0]}  ({Nx_rx}x{Ny_rx})")
    print(f"  Voxels    : {r_vox.shape[0]}  ({NX}x{NY}x{NZ})")
    return HolographyOperatorFast(cp.asarray(r_rx), cp.asarray(r_vox),
                                  k=k, omega=omega, mu0=base.MU0, dV=dV,
                                  batch_rx=base.BATCH_RX)


def loss_terms(model, op, b, z_true):
    z_pred = model(b, op, warm_start=True)
    loss_z = torch.mean((z_pred.abs() - z_true.abs()) ** 2)
    y_hat  = model.measure(z_pred, op)
    normb2 = torch.mean(b.abs() ** 2) + 1e-12
    loss_d = torch.mean((y_hat - b).abs() ** 2) / normb2
    reg    = model.lowrank_frob_sq()
    return ALPHA_Z * loss_z + BETA_DATA * loss_d + GAMMA_REG * reg, loss_z, loss_d, reg


def train(op, b_tr, z_tr, b_va, z_va, ckpt_name, resume=None):
    M     = op.N_rx
    model = LRWLISTAHolography(K=K, L_est=L_EST, Nx=NX, Ny=NY, Nz=NZ,
                               M=M, rank=RANK, lambda_init=LAMBDA_INIT).to(DEVICE)
    optim = torch.optim.Adam([
        {"params": [model.log_mu, model.log_lambda],          "lr": LR},
        {"params": [model.log_wx, model.log_wy, model.log_wz],"lr": LR_W},
        {"params": [model.U_re, model.U_im, model.V_re, model.V_im], "lr": LR_LR},
    ])
    start_epoch, loss_history, val_history, best_val = 1, [], [], float("inf")
    if resume:
        ck = torch.load(resume, map_location=DEVICE, weights_only=False)
        model.load_state_dict(ck["model_state"])
        if "optim_state" in ck: optim.load_state_dict(ck["optim_state"])
        start_epoch  = ck["epoch"] + 1
        loss_history = list(ck.get("loss_history", []))
        val_history  = list(ck.get("val_history", []))
        best_val     = ck.get("best_val", float("inf"))
        print(f"  Resumed from epoch {start_epoch} (best_val={best_val:.4e})")

    b_tr = [x.to(DEVICE) for x in b_tr]; z_tr = [x.to(DEVICE) for x in z_tr]
    b_va = [x.to(DEVICE) for x in b_va]; z_va = [x.to(DEVICE) for x in z_va]
    N = len(b_tr)
    print(f"\n[LR-W-LISTA Ken_grasso] K={K} rank={RANK} epochs={start_epoch}->{N_EPOCHS} "
          f"N_train={N} N_val={len(b_va)} device={DEVICE}")

    for epoch in range(start_epoch, N_EPOCHS + 1):
        t0 = time.time(); model.train(); optim.zero_grad()
        agg = torch.zeros((), device=DEVICE); lz = ld = lr_ = 0.0
        for idx in np.random.permutation(N):
            tot, a, b_, c = loss_terms(model, op, b_tr[idx], z_tr[idx])
            agg = agg + tot; lz += float(a); ld += float(b_); lr_ += float(c)
        (agg / N).backward()
        torch.nn.utils.clip_grad_norm_(model.parameters(), max_norm=1.0)
        optim.step()
        with torch.no_grad():
            model.log_wx.clamp_(-base.W_LOG_CLAMP, base.W_LOG_CLAMP)
            model.log_wy.clamp_(-base.W_LOG_CLAMP, base.W_LOG_CLAMP)
            model.log_wz.clamp_(-base.W_LOG_CLAMP, base.W_LOG_CLAMP)

        model.eval(); vloss = 0.0
        with torch.no_grad():
            for j in range(len(b_va)):
                vt, _, _, _ = loss_terms(model, op, b_va[j], z_va[j])
                vloss += float(vt)
        vloss /= max(len(b_va), 1)

        tr_loss = float(agg) / N
        loss_history.append(tr_loss); val_history.append(vloss)
        st = model.lowrank_stats()
        print(f"  Ep {epoch:3d}/{N_EPOCHS}  train={tr_loss:.4e}  val={vloss:.4e} "
              f"  time={time.time()-t0:.0f}s  |U|={st['U_norm']:.2e}  |V|={st['V_norm']:.2e}")

        ck = {"epoch": epoch, "model_state": model.state_dict(),
              "optim_state": optim.state_dict(), "loss": tr_loss,
              "loss_history": loss_history, "val_history": val_history,
              "best_val": best_val, "K": K, "Nx": NX, "Ny": NY, "Nz": NZ,
              "M": M, "rank": RANK}
        torch.save(ck, os.path.join(CKPT_DIR, f"{ckpt_name}_ep{epoch:03d}.pt"))
        if vloss < best_val:
            best_val = vloss
            torch.save(ck, os.path.join(CKPT_DIR, f"{ckpt_name}_best.pt"))
            print(f"    [best] val={best_val:.4e}")

    print(f"\n  Best ckpt: {CKPT_DIR}/{ckpt_name}_best.pt  best_val={best_val:.4e}")
    return model, loss_history


# ---------------------------------------------------------------------------
if __name__ == "__main__":
    ap = argparse.ArgumentParser()
    ap.add_argument("--rank",       type=int, default=RANK)
    ap.add_argument("--resume",     default=None)
    ap.add_argument("--infer-only", default=None)
    args = ap.parse_args()
    RANK      = args.rank
    ckpt_name = f"wlista_lowrank_ken_grasso_r{RANK}"

    t_start = time.time()
    print("=" * 68)
    print(f"LR-W-LISTA Ken_grasso_nowalls — rank={RANK}  device={DEVICE}")
    print(f"  Train idx {base.TRAIN_IDX}   Val idx {base.VAL_IDX}")
    print("=" * 68)

    b_tr_np, z_tr_np, k, omega, rx_ref = base.load_synthetic_dataset("wlista")
    op = build_operator_fast(rx_ref, k, omega)
    b_va_np, z_va_np = base.load_validation_data(rx_ref, k)

    if args.infer_only:
        ck    = torch.load(args.infer_only, map_location="cpu", weights_only=False)
        model = LRWLISTAHolography(K=int(ck["K"]), L_est=L_EST,
                                   Nx=int(ck["Nx"]), Ny=int(ck["Ny"]), Nz=int(ck["Nz"]),
                                   M=int(ck["M"]), rank=int(ck["rank"]),
                                   lambda_init=LAMBDA_INIT)
        model.load_state_dict(ck["model_state"]); model = model.cpu()
        loss_history = list(ck.get("loss_history", [ck["loss"]]))
        print(f"  Loaded epoch={ck['epoch']} val={ck.get('val', float('nan')):.4e}")
    else:
        b_tr = [torch.as_tensor(x) for x in b_tr_np]
        z_tr = [torch.as_tensor(x) for x in z_tr_np]
        b_va = [torch.as_tensor(x) for x in b_va_np]
        z_va = [torch.as_tensor(x) for x in z_va_np]
        model, loss_history = train(op, b_tr, z_tr, b_va, z_va, ckpt_name, resume=args.resume)
        model = model.cpu()

    train_labels = [base.POS_LABELS[i] for i in base.TRAIN_IDX]
    base.plot_results(b_tr_np, z_tr_np, train_labels, model, op, loss_history,
                      "wlista", ckpt_name, split_tag="train")
    val_labels = [base.POS_LABELS[i] for i in base.VAL_IDX]
    base.plot_results(b_va_np, z_va_np, val_labels, model, op, loss_history,
                      "wlista", ckpt_name, split_tag="val")

    print(f"\nTotal elapsed: {(time.time()-t_start)/60:.1f} min\nDone.")
