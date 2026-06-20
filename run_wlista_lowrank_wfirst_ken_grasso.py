# -*- coding: utf-8 -*-
"""
run_wlista_lowrank_wfirst_ken_grasso.py
========================================
LR-W-LISTA con strategia W-FIRST su E_total_Ken_grasso_nowalls.mat.

Warmup (epoche 1-WARMUP_EPOCHS):
  - solo W + mu + lambda vengono addestrati  (= W-LISTA puro)
  - UV congelato (U=0 => nessuna correzione low-rank)
  => il modello apprende prima i pesi W stabili, poi introduce Delta T

Dopo warmup (epoca WARMUP_EPOCHS+1 in poi):
  - UV viene aggiunto all'ottimizzatore
  - tutti i parametri vengono addestrati insieme

Run
---
    # ⚠️  USA SEMPRE nohup E python3
    nohup python3 run_wlista_lowrank_wfirst_ken_grasso.py > wfirst_ken_grasso.log 2>&1 &

    # resume da checkpoint esistente
    python3 run_wlista_lowrank_wfirst_ken_grasso.py --resume checkpoints_lista_lowrank_wfirst_ken_grasso/wlista_lowrank_wfirst_ken_grasso_r16_ep005.pt

    # oppure parti da un checkpoint W-LISTA convertito
    python3 convert_wlista_to_wfirst_ken_grasso.py
    python3 run_wlista_lowrank_wfirst_ken_grasso.py --resume <ckpt_convertito>

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
# Override dataset
# ---------------------------------------------------------------------------
SYNTH_DIR   = os.path.join(SCRIPT_DIR, "Dataset TUM", "sinthetic_data")
ETOTAL_FILE = os.path.join(SYNTH_DIR, "E_total_Ken_grasso_nowalls.mat")
base.ETOTAL_FILE = ETOTAL_FILE
base.SYNTH_DIR   = SYNTH_DIR

if not os.path.exists(ETOTAL_FILE):
    raise FileNotFoundError(
        f"File non trovato: {ETOTAL_FILE}\n"
        f"Assicurati che sia in {SYNTH_DIR}")

# --- auto-detect numero di posizioni ---
_mat  = sio.loadmat(ETOTAL_FILE)
_keys = [k for k in _mat if not k.startswith("_")]
_n_pos, _k = None, _keys[0]
for _k in _keys:
    if _mat[_k].ndim == 3 and _mat[_k].shape[:2] == (162, 80):
        _n_pos = _mat[_k].shape[2]
        break
if _n_pos is None:
    _n_pos = _mat[_keys[0]].shape[-1]
print(f"[Ken_grasso wfirst] N_pos rilevato = {_n_pos}  (chiave: '{_k}')")

base.TRAIN_IDX  = list(range(_n_pos))
base.VAL_IDX    = [_n_pos - 1]
base.POS_LABELS = [f"ken_grasso_pos{i:02d}" for i in range(_n_pos)]

# ---------------------------------------------------------------------------
# Cartelle output
# ---------------------------------------------------------------------------
OUT_DIR  = os.path.join(SCRIPT_DIR, "results_wlista_lowrank_wfirst_ken_grasso")
CKPT_DIR = os.path.join(SCRIPT_DIR, "checkpoints_lista_lowrank_wfirst_ken_grasso")
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
LR          = 1e-2   # era base.LR=5e-2: ridotto, causava collasso di z a epoca 7 (fase FULL)
LR_W        = 1e-1   # era base.LR_W=5e-1: ridotto in proporzione
LAMBDA_INIT = base.LAMBDA_INIT
L_EST       = base.L_EST

RANK            = 8
LR_LR           = 1e-5       # UV: lr bassa, si attiva dopo warmup
WARMUP_EPOCHS   = 6          # epoche con solo W+mu+lambda (UV congelato)
ALPHA_Z         = 1.0
BETA_DATA       = 1e-4
GAMMA_REG       = 1e-1
REF_EPOCH_START = 2          # da quale epoca salvare snapshot

DEVICE = torch.device("cuda" if torch.cuda.is_available() else "cpu")


# ---------------------------------------------------------------------------
# Operator
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


# ---------------------------------------------------------------------------
# Loss
# ---------------------------------------------------------------------------
def loss_terms(model, op, b, z_true):
    z_pred = model(b, op, warm_start=True)
    loss_z = torch.mean((z_pred.abs() - z_true.abs()) ** 2)
    y_hat  = model.measure(z_pred, op)
    normb2 = torch.mean(b.abs() ** 2) + 1e-12
    loss_d = torch.mean((y_hat - b).abs() ** 2) / normb2
    reg    = model.lowrank_frob_sq()
    return ALPHA_Z*loss_z + BETA_DATA*loss_d + GAMMA_REG*reg, loss_z, loss_d, reg


def _build_optim_warmup(model):
    """Solo W + mu + lambda. UV congelato."""
    return torch.optim.Adam([
        {"params": [model.log_mu, model.log_lambda], "lr": LR},
        {"params": [model.log_wx, model.log_wy, model.log_wz], "lr": LR_W},
    ])


def _build_optim_full(model):
    """Aggiunge UV."""
    return torch.optim.Adam([
        {"params": [model.log_mu, model.log_lambda], "lr": LR},
        {"params": [model.log_wx, model.log_wy, model.log_wz], "lr": LR_W},
        {"params": [model.U_re, model.U_im, model.V_re, model.V_im], "lr": LR_LR},
    ])


# ---------------------------------------------------------------------------
# Training
# ---------------------------------------------------------------------------
def train(op, b_tr_np, z_tr_np, b_va_np, z_va_np, ckpt_name, resume=None):
    M     = op.N_rx
    model = LRWLISTAHolography(K=K, L_est=L_EST, Nx=NX, Ny=NY, Nz=NZ,
                               M=M, rank=RANK, lambda_init=LAMBDA_INIT).to(DEVICE)

    in_warmup = True
    optim = _build_optim_warmup(model)

    start_epoch, loss_history, val_history, best = 1, [], [], float("inf")
    if resume is not None:
        ck = torch.load(resume, map_location=DEVICE, weights_only=False)
        model.load_state_dict(ck["model_state"])
        start_epoch  = ck["epoch"] + 1
        loss_history = list(ck.get("loss_history", []))
        val_history  = list(ck.get("val_history", []))
        best         = ck.get("best_val", float("inf"))
        if start_epoch > WARMUP_EPOCHS:
            in_warmup = False
            optim = _build_optim_full(model)
        if "optim_state" in ck:
            try:
                optim.load_state_dict(ck["optim_state"])
            except Exception:
                pass
        print(f"  Resumed from epoch {start_epoch} (best={best:.4e})")

    b_tr = [torch.as_tensor(x).to(DEVICE) for x in b_tr_np]
    z_tr = [torch.as_tensor(x).to(DEVICE) for x in z_tr_np]
    b_va = [torch.as_tensor(x).to(DEVICE) for x in b_va_np]
    z_va = [torch.as_tensor(x).to(DEVICE) for x in z_va_np]
    N = len(b_tr)

    # snapshot di riferimento (val, pos10)
    REF_DIR = os.path.join(OUT_DIR, "epoch_recon")
    os.makedirs(REF_DIR, exist_ok=True)
    b_ref    = b_va[0].clone() if b_va else b_tr[0].clone()
    z_ref    = (z_va[0] if b_va else z_tr[0]).detach().cpu().numpy()
    b_ref_np = b_ref.detach().cpu().numpy()
    z_mf_ref   = ic.run_matched_filter(op, b_ref_np)
    z_ista_ref = ic.run_ista(op, b_ref_np, K, LAMBDA_INIT, L_EST)

    print(f"\n[W-FIRST LR-W-LISTA Ken_grasso] K={K} rank={RANK} "
          f"epochs={start_epoch}->{N_EPOCHS} warmup={WARMUP_EPOCHS} device={DEVICE}")
    print(f"  N_train={N} N_val={len(b_va)}  "
          f"ALPHA_Z={ALPHA_Z} BETA_DATA={BETA_DATA} GAMMA_REG={GAMMA_REG}")

    for epoch in range(start_epoch, N_EPOCHS + 1):
        # transizione warmup -> full
        if in_warmup and epoch > WARMUP_EPOCHS:
            print(f"\n  [W-first] Warmup completato a ep{epoch-1}. "
                  f"Aggiungo UV (lr={LR_LR:.1e}).\n")
            optim = _build_optim_full(model)
            in_warmup = False

        phase = "WARMUP" if in_warmup else "FULL"

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

        # validation
        model.eval(); vloss = 0.0
        with torch.no_grad():
            for j in range(len(b_va)):
                vt, _, _, _ = loss_terms(model, op, b_va[j], z_va[j])
                vloss += float(vt)
        vloss /= max(len(b_va), 1)

        tr_loss = float(agg) / N
        loss_history.append(tr_loss); val_history.append(vloss)
        st = model.lowrank_stats()
        print(f"  Ep {epoch:3d}/{N_EPOCHS} [{phase}]  train={tr_loss:.4e} "
              f"(z={lz/N:.3e} data={ld/N:.3e} reg={lr_/N:.2e})  "
              f"val={vloss:.4e}  t={time.time()-t0:.0f}s  "
              f"|U|={st['U_fro']:.2e} |V|={st['V_fro']:.2e}")

        if epoch >= REF_EPOCH_START:
            model.eval()
            with torch.no_grad():
                z_snap = model(b_ref, op, warm_start=True).detach().cpu().numpy()
            ic.save_epoch_snapshot(REF_DIR, f"{ckpt_name}_val0", epoch,
                                   base.X_IMG, base.Y_IMG, base.Z_IMG,
                                   z_snap, z_true=z_ref,
                                   z_mf=z_mf_ref, z_ista=z_ista_ref)

        ckpt = dict(epoch=epoch, K=K, Nx=NX, Ny=NY, Nz=NZ, M=M, rank=RANK,
                    model_state=model.state_dict(), optim_state=optim.state_dict(),
                    loss=tr_loss, val=vloss, best_val=best,
                    loss_history=loss_history, val_history=val_history,
                    train_idx=base.TRAIN_IDX, val_idx=base.VAL_IDX,
                    warmup_epochs=WARMUP_EPOCHS)
        torch.save(ckpt, os.path.join(CKPT_DIR, f"{ckpt_name}_ep{epoch:03d}.pt"))
        if vloss < best:
            best = vloss; ckpt["best_val"] = best
            torch.save(ckpt, os.path.join(CKPT_DIR, f"{ckpt_name}_best.pt"))
            print(f"    [best] val={best:.4e}")

    print(f"\n  Best ckpt: {CKPT_DIR}/{ckpt_name}_best.pt  best={best:.4e}")
    return model, loss_history


# ---------------------------------------------------------------------------
if __name__ == "__main__":
    ap = argparse.ArgumentParser()
    ap.add_argument("--rank",        type=int, default=RANK)
    ap.add_argument("--warmup",      type=int, default=WARMUP_EPOCHS)
    ap.add_argument("--resume",      default=None)
    ap.add_argument("--infer-only",  default=None)
    args = ap.parse_args()
    RANK          = args.rank
    WARMUP_EPOCHS = args.warmup
    ckpt_name     = f"wlista_lowrank_wfirst_ken_grasso_r{RANK}"

    t_start = time.time()
    print("=" * 68)
    print(f"LR-W-LISTA W-FIRST Ken_grasso — rank={RANK} warmup={WARMUP_EPOCHS} device={DEVICE}")
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
        model, loss_history = train(op, b_tr_np, z_tr_np, b_va_np, z_va_np,
                                    ckpt_name, resume=args.resume)
        model = model.cpu()

    train_labels = [base.POS_LABELS[i] for i in base.TRAIN_IDX]
    base.plot_results(b_tr_np, z_tr_np, train_labels, model, op, loss_history,
                      "wlista", ckpt_name, split_tag="train")
    val_labels = [base.POS_LABELS[i] for i in base.VAL_IDX]
    base.plot_results(b_va_np, z_va_np, val_labels, model, op, loss_history,
                      "wlista", ckpt_name, split_tag="val")

    print(f"\nTotal elapsed: {(time.time()-t_start)/60:.1f} min\nDone.")
