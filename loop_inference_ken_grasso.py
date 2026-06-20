# -*- coding: utf-8 -*-
"""
loop_inference_ken_grasso.py
=============================
SOLO INFERENZA su dataset sintetico Ken_grasso_nowalls: dato un prefisso di
checkpoint, cicla su TUTTE le epoche disponibili (`<prefix>_epNNN.pt`,
escluso `<prefix>_best.pt`) e su TUTTE le posizioni del dataset (z_true e'
disponibile per ognuna, generato dal body model), eseguendo MF + ISTA +
modello (LISTA / W-LISTA / LR-W-LISTA, auto-detect).

Per ogni (epoca, posizione) salva 3 file:
  - <tag>.png   pannelli MIP-xy: GT, MF, ISTA, modello
  - <tag>.mat   volumi + MIP + metriche
  - <tag>.npz   idem, formato numpy

Inoltre salva metrics.csv con, per ogni riga: epoch, pos_label, holo_x, holo_z,
data_consistency, signal_clutter, mse_full, mse_occ, mse_bg, checkpoint.

Uso
---
  python3 loop_inference_ken_grasso.py --prefix lista_ken_grasso \\
      --ckpt-dir checkpoints_lista_plain_ken_grasso

  python3 loop_inference_ken_grasso.py --prefix wlista_ken_grasso \\
      --ckpt-dir checkpoints_lista_ken_grasso

  python3 loop_inference_ken_grasso.py --prefix wlista_lowrank_ken_grasso_r16 \\
      --ckpt-dir checkpoints_lista_lowrank_ken_grasso

  python3 loop_inference_ken_grasso.py --prefix wlista_lowrank_wfirst_ken_grasso_r8 \\
      --ckpt-dir checkpoints_lista_lowrank_wfirst_ken_grasso

Se --ckpt-dir e' omesso, viene dedotto dal nome del prefisso.
"""

import os, sys, glob, re, csv, argparse, time
sys.stdout.reconfigure(line_buffering=True)
import numpy as np
import scipy.io as sio
import torch

SCRIPT_DIR = os.path.dirname(os.path.abspath(__file__))
sys.path.insert(0, SCRIPT_DIR)

import run_wlista_synthetic_nowalls as base
import inference_common as ic

# ---------------------------------------------------------------------------
# Dataset Ken_grasso (stesso override usato dagli script di training)
# ---------------------------------------------------------------------------
SYNTH_DIR   = os.path.join(SCRIPT_DIR, "Dataset TUM", "sinthetic_data")
ETOTAL_FILE = os.path.join(SYNTH_DIR, "E_total_Ken_grasso_nowalls.mat")
base.ETOTAL_FILE = ETOTAL_FILE
base.SYNTH_DIR   = SYNTH_DIR

if not os.path.exists(ETOTAL_FILE):
    raise FileNotFoundError(
        f"File non trovato: {ETOTAL_FILE}\nAssicurati che sia in {SYNTH_DIR}")

_mat  = sio.loadmat(ETOTAL_FILE)
_keys = [k for k in _mat if not k.startswith("_")]
_n_pos, _k = None, _keys[0]
for _k in _keys:
    if _mat[_k].ndim == 3 and _mat[_k].shape[:2] == (162, 80):
        _n_pos = _mat[_k].shape[2]
        break
if _n_pos is None:
    _n_pos = _mat[_keys[0]].shape[-1]
print(f"[Ken_grasso inference] N_pos rilevato = {_n_pos}  (chiave: '{_k}')")

base.TRAIN_IDX  = list(range(_n_pos))   # tutte le posizioni
base.VAL_IDX    = []                     # non serve qui: prendiamo tutto da TRAIN_IDX
base.POS_LABELS = [f"ken_grasso_pos{i:02d}" for i in range(_n_pos)]

OUT_ROOT = os.path.join(SCRIPT_DIR, "results_inference_ken_grasso")


def guess_ckpt_dir(prefix):
    if "wfirst" in prefix:
        return os.path.join(SCRIPT_DIR, "checkpoints_lista_lowrank_wfirst_ken_grasso")
    if "lowrank" in prefix:
        return os.path.join(SCRIPT_DIR, "checkpoints_lista_lowrank_ken_grasso")
    if prefix.startswith("lista_"):
        return os.path.join(SCRIPT_DIR, "checkpoints_lista_plain_ken_grasso")
    return os.path.join(SCRIPT_DIR, "checkpoints_lista_ken_grasso")


def find_ckpts(ckpt_dir, prefix, start_epoch):
    """Tutti i <prefix>_epNNN.pt ordinati (>= start_epoch) + il finale <prefix>.pt
    se presente. <prefix>_best.pt escluso (e' un duplicato di una delle epoche)."""
    eps = []
    for p in glob.glob(os.path.join(ckpt_dir, f"{prefix}_ep*.pt")):
        m = re.search(r"_ep(\d+)\.pt$", os.path.basename(p))
        if m:
            e = int(m.group(1))
            if e >= start_epoch:
                eps.append((e, p))
    eps.sort()
    out = list(eps)
    final = os.path.join(ckpt_dir, f"{prefix}.pt")
    if os.path.exists(final) and (not eps or eps[-1][1] != final):
        out.append(("final", final))
    return out


def build_operator_fast(rx_ref, k, omega, device):
    Nx_rx, Ny_rx = rx_ref["S21"].shape[:2]
    x_rx = rx_ref["X"][:, 0]; y_rx = rx_ref["Y"][0, :]
    XX, YY = np.meshgrid(x_rx, y_rx, indexing="ij")
    r_rx  = np.column_stack([XX.ravel(), YY.ravel(), np.zeros(Nx_rx * Ny_rx)])
    XXv, YYv, ZZv = np.meshgrid(base.X_IMG, base.Y_IMG, base.Z_IMG, indexing="ij")
    r_vox = np.column_stack([XXv.ravel(), YYv.ravel(), ZZv.ravel()])
    dV = (base.X_IMG[1]-base.X_IMG[0])*(base.Y_IMG[1]-base.Y_IMG[0])*(base.Z_IMG[1]-base.Z_IMG[0])
    print(f"  Receivers : {r_rx.shape[0]}  ({Nx_rx}x{Ny_rx})")
    print(f"  Voxels    : {r_vox.shape[0]}")
    if "cuda" in str(device):
        import cupy as cp
        from holography_operator_fast import HolographyOperatorFast
        return HolographyOperatorFast(cp.asarray(r_rx), cp.asarray(r_vox),
                                      k=k, omega=omega, mu0=base.MU0, dV=dV,
                                      batch_rx=base.BATCH_RX)
    from holography_operator_numpy import HolographyOperatorNumpy
    return HolographyOperatorNumpy(r_rx, r_vox, k=k, omega=omega, mu0=base.MU0,
                                   dV=dV, batch_rx=base.BATCH_RX)


def save_full_snapshot(out_dir, tag, x_img, y_img, z_img,
                       z_model, z_true, z_mf, z_ista, extra_meta):
    """PNG (4 pannelli con GT) + .mat/.npz (volumi+MIP+metriche)."""
    png = os.path.join(out_dir, tag + ".png")
    ic.plot_panels(png, x_img, y_img, z_img,
                  extra_meta.get("suptitle", tag),
                  z_model, extra_meta.get("model_label", "model"),
                  z_true=z_true, z_mf=z_mf, z_ista=z_ista)

    payload = dict(z_model=z_model, z_true=z_true, z_mf=z_mf, z_ista=z_ista,
                   X_IMG=x_img, Y_IMG=y_img, Z_IMG=z_img)
    for k_, v_ in extra_meta.items():
        if k_ in ("suptitle", "model_label"):
            continue
        payload[k_] = v_

    ic.save_outputs(os.path.join(out_dir, tag + ".npz"),
                    os.path.join(out_dir, tag + ".mat"),
                    x_img, y_img, z_img, payload)
    return png


def main():
    ap = argparse.ArgumentParser()
    ap.add_argument("--prefix", required=True,
                    help="prefisso checkpoint, es. lista_ken_grasso / wlista_ken_grasso / "
                         "wlista_lowrank_ken_grasso_r16 / wlista_lowrank_wfirst_ken_grasso_r8")
    ap.add_argument("--ckpt-dir", default=None,
                    help="cartella checkpoint (default: dedotta dal prefisso)")
    ap.add_argument("--model", choices=ic.MODEL_CHOICES, default="auto")
    ap.add_argument("--start-epoch", type=int, default=1,
                    help="prima epoca da includere (default 1: tutte le epoche)")
    ap.add_argument("--device", default=("cuda" if torch.cuda.is_available() else "cpu"))
    args = ap.parse_args()

    ckpt_dir = args.ckpt_dir or guess_ckpt_dir(args.prefix)
    ckpts = find_ckpts(ckpt_dir, args.prefix, args.start_epoch)
    if not ckpts:
        sys.exit(f"Nessun checkpoint per '{args.prefix}' in {ckpt_dir} (start_epoch={args.start_epoch})")

    out_dir = os.path.join(OUT_ROOT, args.prefix)
    os.makedirs(out_dir, exist_ok=True)

    class _Tee:
        def __init__(self, *streams): self.streams = streams
        def write(self, d):
            for st in self.streams: st.write(d); st.flush()
        def flush(self):
            for st in self.streams: st.flush()
    _logf = open(os.path.join(out_dir, "sweep.log"), "a", encoding="utf-8")
    sys.stdout = _Tee(sys.__stdout__, _logf)

    t_start = time.time()
    print("=" * 70)
    print(f"INFERENCE SWEEP Ken_grasso  prefix={args.prefix}  device={args.device}")
    print(f"  ckpt_dir = {ckpt_dir}")
    print(f"  #checkpoint = {len(ckpts)}  (start_epoch={args.start_epoch}, best escluso)")
    print(f"  #posizioni  = {_n_pos}")
    print("=" * 70)

    # --- dati per TUTTE le posizioni: una sola volta ---
    b_list, z_list, k, omega, rx_ref = base.load_synthetic_dataset("wlista")
    op = build_operator_fast(rx_ref, k, omega, args.device)

    # --- baseline MF + ISTA per ogni posizione: una sola volta ---
    print("\nBaseline MF + ISTA per ciascuna posizione ...")
    z_mf_list, z_ista_list = [], []
    for i, b_np in enumerate(b_list):
        z_mf_list.append(ic.run_matched_filter(op, b_np))
        z_ista_list.append(ic.run_ista(op, b_np, base.K, base.LAMBDA_INIT, base.L_EST))
        print(f"  [{i:02d}] {base.POS_LABELS[i]}  done")

    rows = []
    for ep, path in ckpts:
        ck = torch.load(path, map_location="cpu", weights_only=False)
        L_est = float(ck.get("L_est", base.L_EST))
        lam   = float(ck.get("lambda_init", base.LAMBDA_INIT))
        model, kind = ic.build_model(ck, L_est, lam, model_sel=args.model)
        model = model.to(args.device)
        epn = ck.get("epoch", 0) if ep == "final" else ep

        for i, (b_np, z_true) in enumerate(zip(b_list, z_list)):
            lbl = base.POS_LABELS[i]
            z_model, dc = ic.run_model(model, op, b_np, args.device)
            m = ic.metrics(z_model, z_true)

            tag = f"recon_{args.prefix}_{lbl}_ep{int(epn):03d}"
            save_full_snapshot(
                out_dir, tag, base.X_IMG, base.Y_IMG, base.Z_IMG,
                z_model, z_true, z_mf_list[i], z_ista_list[i],
                extra_meta=dict(
                    suptitle=f"{args.prefix}  {lbl}  ep{epn}  "
                             f"({ic.pretty(kind)})",
                    model_label=f"{ic.pretty(kind)} ep{epn}",
                    epoch=int(epn) if epn != "final" else -1,
                    pos_label=lbl,
                    data_consistency=dc,
                    **m,
                ))

            rows.append((epn, lbl, dc, m["SC"], m["mse_full"], m["mse_occ"], m["mse_bg"],
                        os.path.basename(path)))
            print(f"  ep {str(ep):>5}  {lbl}  DC={dc:.4f}  S/C={m['SC']:.3f}  "
                  f"MSE_occ={m['mse_occ']:.3e}")

    csv_path = os.path.join(out_dir, "metrics.csv")
    with open(csv_path, "w", newline="") as f:
        w = csv.writer(f)
        w.writerow(["epoch", "pos_label", "data_consistency", "signal_clutter",
                    "mse_full", "mse_occ", "mse_bg", "checkpoint"])
        for r in rows:
            w.writerow(r)

    print(f"\nSalvato {csv_path}  ({len(rows)} righe)")
    print(f"Output in {out_dir}")
    print(f"Total elapsed: {(time.time()-t_start)/60:.1f} min\nDone.")


if __name__ == "__main__":
    main()
