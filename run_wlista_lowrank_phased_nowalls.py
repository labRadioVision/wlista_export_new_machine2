# -*- coding: utf-8 -*-
"""
run_wlista_lowrank_phased_nowalls.py
=====================================
LR-W-LISTA "phased" sul dataset sintetico nowalls originale (8 posizioni
train + 3 posizioni val, gli stessi di run_wlista_synthetic_nowalls_gpu.py).
Stessa logica di run_wlista_lowrank_phased_ken_grasso.py, ma senza override
del dataset: usa SYNTH_DIR/TRAIN_IDX/VAL_IDX/POS_LABELS di default definiti
in run_wlista_synthetic_nowalls.py.

Differenza chiave rispetto a wfirst: dopo la fase di training di W, W
(log_wx/wy/wz) e log_lambda vengono CONGELATI — si allenano solo U, V (+
log_mu). In wfirst invece tutti i parametri restano allenabili insieme dopo
il warmup, il che puo' causare instabilita'/collasso di z quando UV si
attiva.

Non richiede un checkpoint W-LISTA pre-esistente: la fase A allena W **da
zero**, dentro lo stesso script, con U/V congelati a zero (matematicamente
equivalente a allenare W-LISTA puro).

Fasi
----
  A) Allena SOLO log_mu, log_lambda, log_wx/wy/wz da zero (U,V congelati a
     zero -> equivalente a W-LISTA puro). PHASE_A_EPOCHS epoche.
  B) Congelati log_wx/wy/wz e log_lambda: allena solo U, V, log_mu.
  C) (opzionale) Fine-tune congiunto di tutti i parametri a LR ridotte.

Run
---
    nohup python3 run_wlista_lowrank_phased_nowalls.py > phased_nowalls.log 2>&1 &

    # riprendi una fase interrotta a META' (epoca esatta, non la ripete da zero)
    python3 run_wlista_lowrank_phased_nowalls.py \\
        --resume-a checkpoints_lista_lowrank_phased_nowalls/wlista_lowrank_phased_nowalls_r8_A_ep006.pt

    # salta fase A+B (riparti da un checkpoint di fase B GIA' COMPLETA)
    python3 run_wlista_lowrank_phased_nowalls.py \\
        --skip-b checkpoints_lista_lowrank_phased_nowalls/wlista_lowrank_phased_nowalls_r8_B_best.pt

`--skip-a/--skip-b` = la fase e' GIA' FINITA, non rifarla, vai alla successiva.
`--resume-a/--resume-b/--resume-c` = la fase si era FERMATA A META', continua
dall'epoca esatta (poi procede normalmente alle fasi successive).
"""

import os, sys, time, argparse
sys.stdout.reconfigure(line_buffering=True)
import numpy as np
import torch
import cupy as cp

SCRIPT_DIR = os.path.dirname(os.path.abspath(__file__))
sys.path.insert(0, SCRIPT_DIR)

import run_wlista_synthetic_nowalls as base
from holography_operator_fast import HolographyOperatorFast
from lista_holography_lowrank  import LRWLISTAHolography
import inference_common as ic

# ---------------------------------------------------------------------------
# Nessun override dataset: usa SYNTH_DIR/TRAIN_IDX/VAL_IDX/POS_LABELS di
# default definiti in run_wlista_synthetic_nowalls.py (8 train + 3 val)
# ---------------------------------------------------------------------------

# ---------------------------------------------------------------------------
# Cartelle output dedicate (non sovrascrivono checkpoints_lista/ di W-LISTA)
# ---------------------------------------------------------------------------
OUT_DIR  = os.path.join(SCRIPT_DIR, "results_wlista_lowrank_phased_nowalls")
CKPT_DIR = os.path.join(SCRIPT_DIR, "checkpoints_lista_lowrank_phased_nowalls")
os.makedirs(OUT_DIR,  exist_ok=True)
os.makedirs(CKPT_DIR, exist_ok=True)

# ---------------------------------------------------------------------------
# Iperparametri
# ---------------------------------------------------------------------------
NX, NY, NZ  = base.NX, base.NY, base.NZ
K           = base.K
LAMBDA_INIT = base.LAMBDA_INIT
L_EST       = base.L_EST

# LR ridotti (stesso fix applicato alla variante Ken_grasso: base.LR=5e-2 /
# base.LR_W=5e-1 erano troppo alti, rischio di collasso di z)
LR_BASE   = 1e-2
LR_W_BASE = 1e-1

RANK            = 8
PHASE_A_EPOCHS  = 10              # training W da zero (UV congelati a zero)
PHASE_B_EPOCHS  = 20
PHASE_C_EPOCHS  = 10
LR_LR           = 1e-3       # U, V  (fase B) — basso: UV e' una correzione fine, non va spinto forte
LR_MU_B         = 1e-2       # log_mu in fase B
LR_C_SCALE      = 0.1        # scala LR fase C rispetto a LR_BASE/LR_W_BASE
FREEZE_LAMBDA_B = True       # log_lambda congelato anche in fase B (oltre a W)

ALPHA_Z   = 1.0
BETA_DATA = 1e-4
GAMMA_REG = 1e-1

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
# Loss + helpers
# ---------------------------------------------------------------------------
def loss_terms(model, op, b, z_true):
    z_pred = model(b, op, warm_start=True)
    loss_z = torch.mean((z_pred.abs() - z_true.abs()) ** 2)
    y_hat  = model.measure(z_pred, op)
    normb2 = torch.mean(b.abs() ** 2) + 1e-12
    loss_d = torch.mean((y_hat - b).abs() ** 2) / normb2
    reg    = model.lowrank_frob_sq()
    tot    = ALPHA_Z * loss_z + BETA_DATA * loss_d + GAMMA_REG * reg
    return tot, loss_z, loss_d, reg


def evaluate(model, op, b_va, z_va):
    model.eval(); vloss = 0.0; vloss_z = 0.0
    with torch.no_grad():
        for j in range(len(b_va)):
            vt, vz, _, _ = loss_terms(model, op, b_va[j], z_va[j])
            vloss += float(vt); vloss_z += float(vz)
    n = max(len(b_va), 1)
    return vloss / n, vloss_z / n


def set_requires_grad(model, names, flag):
    for n, p in model.named_parameters():
        if n in names:
            p.requires_grad_(flag)


# ---------------------------------------------------------------------------
# Loop di training per una fase (con resume per-epoca)
# ---------------------------------------------------------------------------
def train_phase(model, op, optim, b_tr, z_tr, b_va, z_va,
                n_epochs, phase_tag, ckpt_name, resume=None):
    N = len(b_tr)
    loss_hist, val_hist, best_val, start_epoch = [], [], float("inf"), 1
    best_path = os.path.join(CKPT_DIR, f"{ckpt_name}_{phase_tag}_best.pt")

    if resume is not None:
        ck_r = torch.load(resume, map_location=DEVICE, weights_only=False)
        model.load_state_dict(ck_r["model_state"])
        if "optim_state" in ck_r:
            try:
                optim.load_state_dict(ck_r["optim_state"])
            except Exception as e:
                print(f"  [FASE {phase_tag}] optim_state non ripristinato ({e})")
        start_epoch = ck_r["epoch"] + 1
        loss_hist   = list(ck_r.get("loss_history", []))
        val_hist    = list(ck_r.get("val_history", []))
        best_val    = ck_r.get("best_val", float("inf"))
        print(f"\n[FASE {phase_tag}] resume da {resume}: riparto da epoca "
              f"{start_epoch} (best={best_val:.4e})")
        if start_epoch > n_epochs:
            print(f"  [FASE {phase_tag}] gia' completa ({start_epoch-1}/{n_epochs}), nessuna epoca da fare.")
            return best_path, loss_hist, best_val

    REF_DIR = os.path.join(OUT_DIR, f"epoch_recon_{phase_tag}")
    os.makedirs(REF_DIR, exist_ok=True)
    b_ref = b_va[0] if b_va else b_tr[0]
    if z_va:
        z_ref = z_va[0].detach().cpu().numpy()
    else:
        z_ref = z_tr[0].detach().cpu().numpy()

    # baseline MF + ISTA per il caso di riferimento: una sola volta per fase
    b_ref_np   = b_ref.detach().cpu().numpy()
    z_mf_ref   = ic.run_matched_filter(op, b_ref_np)
    z_ista_ref = ic.run_ista(op, b_ref_np, K, LAMBDA_INIT, L_EST)

    trainable = [n for n, p in model.named_parameters() if p.requires_grad]
    print(f"\n[FASE {phase_tag}] epochs={start_epoch}->{n_epochs}  trainable={trainable}")

    for epoch in range(start_epoch, n_epochs + 1):
        t0 = time.time(); model.train(); optim.zero_grad()
        agg = torch.zeros((), device=DEVICE)
        lz = ld = lr_ = 0.0
        for idx in np.random.permutation(N):
            tot, a, b_, c = loss_terms(model, op, b_tr[idx], z_tr[idx])
            agg = agg + tot
            lz += float(a); ld += float(b_); lr_ += float(c)
        (agg / N).backward()
        torch.nn.utils.clip_grad_norm_(
            [p for p in model.parameters() if p.requires_grad], max_norm=1.0)
        optim.step()
        with torch.no_grad():
            model.log_wx.clamp_(-base.W_LOG_CLAMP, base.W_LOG_CLAMP)
            model.log_wy.clamp_(-base.W_LOG_CLAMP, base.W_LOG_CLAMP)
            model.log_wz.clamp_(-base.W_LOG_CLAMP, base.W_LOG_CLAMP)

        vloss, vloss_z = evaluate(model, op, b_va, z_va)
        tr_loss = float(agg) / N
        loss_hist.append(tr_loss); val_hist.append(vloss)
        st = model.lowrank_stats()
        mu_max = float(torch.exp(model.log_mu).max())
        print(f"  [{phase_tag}] Ep {epoch:3d}/{n_epochs}  train={tr_loss:.4e} "
              f"(z={lz/N:.3e} data={ld/N:.3e} reg={lr_/N:.2e})  "
              f"val={vloss:.4e} val_z={vloss_z:.3e}  t={time.time()-t0:.0f}s  "
              f"|U|={st['U_fro']:.2e} |V|={st['V_fro']:.2e} mu_max={mu_max:.2e}")

        model.eval()
        with torch.no_grad():
            z_snap = model(b_ref, op, warm_start=True).detach().cpu().numpy()
        ic.save_epoch_snapshot(REF_DIR, f"{ckpt_name}_{phase_tag}", epoch,
                               base.X_IMG, base.Y_IMG, base.Z_IMG,
                               z_snap, z_true=z_ref,
                               z_mf=z_mf_ref, z_ista=z_ista_ref)

        ckpt = dict(epoch=epoch, phase=phase_tag, K=K,
                    Nx=NX, Ny=NY, Nz=NZ, M=model.M, rank=model.rank,
                    model_state=model.state_dict(), optim_state=optim.state_dict(),
                    loss=tr_loss, val=vloss, best_val=best_val,
                    loss_history=loss_hist, val_history=val_hist,
                    train_idx=base.TRAIN_IDX, val_idx=base.VAL_IDX)
        torch.save(ckpt, os.path.join(CKPT_DIR, f"{ckpt_name}_{phase_tag}_ep{epoch:03d}.pt"))
        if vloss < best_val:
            best_val = vloss; ckpt["best_val"] = best_val
            torch.save(ckpt, best_path)
            print(f"    [best {phase_tag}] val={best_val:.4e}")

    print(f"  [FASE {phase_tag}] best_val={best_val:.4e}  -> {best_path}")
    return best_path, loss_hist, best_val


# ---------------------------------------------------------------------------
if __name__ == "__main__":
    ap = argparse.ArgumentParser()
    ap.add_argument("--rank",        type=int, default=RANK)
    ap.add_argument("--skip-a",      default=None,
                    help="checkpoint da cui partire saltando la fase A (gia' allenato W)")
    ap.add_argument("--skip-b",      default=None,
                    help="checkpoint da cui partire saltando le fasi A+B")
    ap.add_argument("--resume-a",    default=None,
                    help="continua la fase A da questo checkpoint (epoca esatta, non la salta)")
    ap.add_argument("--resume-b",    default=None,
                    help="continua la fase B da questo checkpoint (epoca esatta, non la salta)")
    ap.add_argument("--resume-c",    default=None,
                    help="continua la fase C da questo checkpoint (epoca esatta, non la salta)")
    ap.add_argument("--infer-only",  default=None)
    args = ap.parse_args()
    RANK      = args.rank
    ckpt_name = f"wlista_lowrank_phased_nowalls_r{RANK}"

    t_start = time.time()
    print("=" * 68)
    print(f"LR-W-LISTA PHASED (W allenato da zero, poi congelato) nowalls — rank={RANK}  device={DEVICE}")
    print(f"  Fase A: {PHASE_A_EPOCHS} ep (solo mu,lambda,W — U,V congelati a zero)")
    print(f"  Fase B: {PHASE_B_EPOCHS} ep (solo U,V,mu — W e lambda congelati)")
    print(f"  Fase C: {PHASE_C_EPOCHS} ep (joint, LR x{LR_C_SCALE})")
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
        print(f"  Loaded phase={ck.get('phase','?')} epoch={ck['epoch']} "
              f"val={ck.get('val', float('nan')):.4e}")
    else:
        b_tr = [torch.as_tensor(x).to(DEVICE) for x in b_tr_np]
        z_tr = [torch.as_tensor(x).to(DEVICE) for x in z_tr_np]
        b_va = [torch.as_tensor(x).to(DEVICE) for x in b_va_np]
        z_va = [torch.as_tensor(x).to(DEVICE) for x in z_va_np]

        model = LRWLISTAHolography(K=K, L_est=L_EST, Nx=NX, Ny=NY, Nz=NZ,
                                   M=op.N_rx, rank=RANK,
                                   lambda_init=LAMBDA_INIT).to(DEVICE)

        if args.skip_b is not None:
            # ---- salta A+B: carica direttamente un checkpoint di fase B ----
            ck_b = torch.load(args.skip_b, map_location=DEVICE, weights_only=False)
            model.load_state_dict(ck_b["model_state"])
            loss_history = list(ck_b.get("loss_history", []))
            val_b = ck_b.get("best_val", float("inf"))
            print(f"\n[FASE A+B] saltate, caricato {args.skip_b}  (val={val_b:.4e})")
        else:
            if args.skip_a is not None:
                # ---- salta A: carica un checkpoint di fase A gia' allenato ----
                ck_a = torch.load(args.skip_a, map_location=DEVICE, weights_only=False)
                model.load_state_dict(ck_a["model_state"])
                hist_a = list(ck_a.get("loss_history", []))
                print(f"\n[FASE A] saltata, caricato {args.skip_a}")
            else:
                # ---- FASE A: allena SOLO mu, lambda, W da zero (o resume) ----
                # U,V congelati a zero (init di default) -> equivalente a W-LISTA puro
                set_requires_grad(model, ["U_re", "U_im", "V_re", "V_im"], False)
                optim_a = torch.optim.Adam([
                    {"params": [model.log_mu, model.log_lambda], "lr": LR_BASE},
                    {"params": [model.log_wx, model.log_wy, model.log_wz], "lr": LR_W_BASE},
                ])
                best_a, hist_a, val_a = train_phase(
                    model, op, optim_a, b_tr, z_tr, b_va, z_va,
                    PHASE_A_EPOCHS, "A", ckpt_name, resume=args.resume_a)
                ck_a = torch.load(best_a, map_location=DEVICE, weights_only=False)
                model.load_state_dict(ck_a["model_state"])
                print(f"\n[FASE A] completata (W allenato da zero, U=0).  best_val={val_a:.4e}")

            # ---- FASE B: solo U, V (+ mu); W e lambda CONGELATI ----
            frozen = ["log_wx", "log_wy", "log_wz"]
            if FREEZE_LAMBDA_B:
                frozen.append("log_lambda")
            set_requires_grad(model, frozen, False)
            set_requires_grad(model, ["U_re", "U_im", "V_re", "V_im"], True)
            optim_b = torch.optim.Adam([
                {"params": [model.log_mu], "lr": LR_MU_B},
                {"params": [model.U_re, model.U_im,
                            model.V_re, model.V_im], "lr": LR_LR},
            ])
            best_b, hist_b, val_b = train_phase(
                model, op, optim_b, b_tr, z_tr, b_va, z_va,
                PHASE_B_EPOCHS, "B", ckpt_name, resume=args.resume_b)
            ck_b = torch.load(best_b, map_location=DEVICE, weights_only=False)
            model.load_state_dict(ck_b["model_state"])
            loss_history = hist_a + hist_b

        # ---- FASE C: fine-tune congiunto a LR ridotte (opzionale) ----
        if PHASE_C_EPOCHS > 0:
            set_requires_grad(model, ["log_wx", "log_wy", "log_wz",
                                      "log_lambda", "log_mu"], True)
            optim_c = torch.optim.Adam([
                {"params": [model.log_mu, model.log_lambda],
                 "lr": LR_BASE * LR_C_SCALE},
                {"params": [model.log_wx, model.log_wy, model.log_wz],
                 "lr": LR_W_BASE * LR_C_SCALE},
                {"params": [model.U_re, model.U_im, model.V_re, model.V_im],
                 "lr": LR_LR * LR_C_SCALE},
            ])
            best_c, hist_c, val_c = train_phase(
                model, op, optim_c, b_tr, z_tr, b_va, z_va,
                PHASE_C_EPOCHS, "C", ckpt_name, resume=args.resume_c)
            if val_c < val_b:
                ck_c = torch.load(best_c, map_location=DEVICE, weights_only=False)
                model.load_state_dict(ck_c["model_state"])
                print(f"\n  Fase C migliora: {val_b:.4e} -> {val_c:.4e}")
            else:
                print(f"\n  Fase C NON migliora ({val_c:.4e} >= {val_b:.4e}): "
                      f"tengo best di fase B")
            loss_history = loss_history + hist_c

        model = model.cpu()

    train_labels = [base.POS_LABELS[i] for i in base.TRAIN_IDX]
    base.plot_results(b_tr_np, z_tr_np, train_labels, model, op, loss_history,
                      "wlista", ckpt_name, split_tag="train")
    val_labels = [base.POS_LABELS[i] for i in base.VAL_IDX]
    base.plot_results(b_va_np, z_va_np, val_labels, model, op, loss_history,
                      "wlista", ckpt_name, split_tag="val")

    print(f"\nTotal elapsed: {(time.time()-t_start)/60:.1f} min\nDone.")
