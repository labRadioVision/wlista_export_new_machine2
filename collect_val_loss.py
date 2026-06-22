# -*- coding: utf-8 -*-
"""
collect_val_loss.py
====================
Cicla su tutti i checkpoint di una cartella (un prefisso alla volta) e
raccoglie epoch / train loss / val loss / best_val in un singolo file .mat
(+ .npz + .csv), per visualizzare/plottare l'andamento del training senza
dover ricaricare ogni .pt singolarmente.

Generico: funziona con qualsiasi cartella di checkpoint .pt che salvi
`epoch`, `loss`, `val`, `best_val` (tutti gli script di training del
progetto lo fanno: lista, wlista, lowrank, wfirst, phased) — riconosce
anche i checkpoint con tag di fase (es.
wlista_lowrank_phased_ken_grasso_r8_A_ep005.pt).

Uso
---
    python3 collect_val_loss.py \\
        --ckpt-dir checkpoints_lista_lowrank_phased_ken_grasso \\
        --prefix wlista_lowrank_phased_ken_grasso_r8_A \\
        --out val_loss_phaseA

    python3 collect_val_loss.py \\
        --ckpt-dir checkpoints_lista_ken_grasso \\
        --prefix wlista_ken_grasso \\
        --out val_loss_wlista

Output (nella cartella corrente, o --out-dir):
    <out>.mat   (epoch, train_loss, val_loss, best_val  -- array numpy)
    <out>.npz   (idem)
    <out>.csv   (idem, leggibile a mano)
"""

import os, sys, glob, re, argparse, csv
import numpy as np
import scipy.io as sio
import torch


def find_ckpts(ckpt_dir, prefix):
    """Tutti i <prefix>_epNNN.pt ordinati + il finale <prefix>.pt (se esiste).
    <prefix>_best.pt escluso (duplica una delle epoche)."""
    eps = []
    for p in glob.glob(os.path.join(ckpt_dir, f"{prefix}_ep*.pt")):
        m = re.search(r"_ep(\d+)\.pt$", os.path.basename(p))
        if m:
            eps.append((int(m.group(1)), p))
    eps.sort()
    out = list(eps)
    final = os.path.join(ckpt_dir, f"{prefix}.pt")
    if os.path.exists(final) and (not eps or eps[-1][1] != final):
        out.append((None, final))   # epoca "finale": la leggiamo dal contenuto
    return out


def main():
    ap = argparse.ArgumentParser()
    ap.add_argument("--ckpt-dir", required=True, help="cartella con i checkpoint .pt")
    ap.add_argument("--prefix",   required=True, help="prefisso checkpoint (incluso tag di fase se presente)")
    ap.add_argument("--out",      default=None, help="nome file output (senza estensione, default = prefix)")
    ap.add_argument("--out-dir",  default=".", help="cartella di output (default: corrente)")
    args = ap.parse_args()

    ckpts = find_ckpts(args.ckpt_dir, args.prefix)
    if not ckpts:
        sys.exit(f"Nessun checkpoint trovato per prefisso '{args.prefix}' in {args.ckpt_dir}")

    out_name = args.out or args.prefix
    os.makedirs(args.out_dir, exist_ok=True)

    rows = []
    for ep, path in ckpts:
        ck = torch.load(path, map_location="cpu", weights_only=False)
        epoch    = int(ck.get("epoch", ep if ep is not None else -1))
        phase    = ck.get("phase", "")
        loss     = float(ck.get("loss", float("nan")))
        val      = float(ck.get("val", float("nan")))
        best_val = float(ck.get("best_val", float("nan")))
        rows.append((epoch, phase, loss, val, best_val, os.path.basename(path)))
        print(f"  ep {epoch:4d}  phase={phase!s:>3}  loss={loss:.4e}  "
              f"val={val:.4e}  best_val={best_val:.4e}  ({os.path.basename(path)})")

    rows.sort(key=lambda r: r[0])
    epochs    = np.array([r[0] for r in rows], dtype=np.int32)
    train_l   = np.array([r[2] for r in rows], dtype=np.float64)
    val_l     = np.array([r[3] for r in rows], dtype=np.float64)
    best_l    = np.array([r[4] for r in rows], dtype=np.float64)

    mat_path = os.path.join(args.out_dir, out_name + ".mat")
    npz_path = os.path.join(args.out_dir, out_name + ".npz")
    csv_path = os.path.join(args.out_dir, out_name + ".csv")

    sio.savemat(mat_path, dict(epoch=epochs, train_loss=train_l,
                               val_loss=val_l, best_val=best_l,
                               prefix=args.prefix))
    np.savez(npz_path, epoch=epochs, train_loss=train_l,
            val_loss=val_l, best_val=best_l, prefix=args.prefix)
    with open(csv_path, "w", newline="") as f:
        w = csv.writer(f)
        w.writerow(["epoch", "phase", "train_loss", "val_loss", "best_val", "checkpoint"])
        for r in rows:
            w.writerow(r)

    print(f"\nSalvati:\n  {mat_path}\n  {npz_path}\n  {csv_path}")
    print(f"({len(rows)} epoche)\nDone.")


if __name__ == "__main__":
    main()
