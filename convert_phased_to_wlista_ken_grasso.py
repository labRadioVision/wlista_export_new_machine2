"""
convert_phased_to_wlista_ken_grasso.py
=======================================
Estrae i pesi W da un checkpoint LRWLISTAHolography (fase A phased)
e crea un checkpoint WLISTAHolography compatibile con run_wlista_ken_grasso.py.

Parametri copiati: log_mu, log_lambda, log_wx, log_wy, log_wz
Parametri UV ignorati (non presenti in WLISTAHolography).

Uso:
  python convert_phased_to_wlista_ken_grasso.py
  python convert_phased_to_wlista_ken_grasso.py --src checkpoints_wlista_lowrank/wlista_lowrank_phased_ken_grasso_r8_A_ep010.pt
  python convert_phased_to_wlista_ken_grasso.py --src ... --dst checkpoints_lista_ken_grasso/wlista_ken_grasso_from_phased_ep010.pt
"""

import os, sys, argparse, torch
sys.path.insert(0, os.path.dirname(os.path.abspath(__file__)))

import run_wlista_synthetic_nowalls as base
from lista_holography_weighted import WLISTAHolography

_ap = argparse.ArgumentParser()
_ap.add_argument("--src", default=None,
                 help="checkpoint LRWLISTAHolography sorgente (phased fase A)")
_ap.add_argument("--dst", default=None,
                 help="path output (default: auto)")
_args = _ap.parse_args()

# default src: ultimo checkpoint fase A disponibile
if _args.src is None:
    import glob, re
    ckpts = sorted(
        glob.glob(os.path.join("checkpoints_wlista_lowrank",
                               "wlista_lowrank_phased_ken_grasso_r8_A_ep*.pt")) +
        glob.glob(os.path.join("checkpoints_lista_lowrank_phased_ken_grasso",
                               "wlista_lowrank_phased_ken_grasso_r8_A_ep*.pt"))
    )
    if not ckpts:
        sys.exit("Nessun checkpoint fase A trovato in checkpoints_wlista_lowrank/")
    _args.src = ckpts[-1]
    print(f"Src auto-selezionato: {_args.src}")

SRC_CKPT = _args.src

if _args.dst is None:
    _base = os.path.splitext(os.path.basename(SRC_CKPT))[0]
    _ep   = _base.split("_ep")[-1] if "_ep" in _base else "converted"
    os.makedirs("checkpoints_lista_ken_grasso", exist_ok=True)
    DST_CKPT = os.path.join("checkpoints_lista_ken_grasso",
                             f"wlista_ken_grasso_from_phased_ep{_ep}.pt")
else:
    DST_CKPT = _args.dst

K           = base.K
NX, NY, NZ  = base.NX, base.NY, base.NZ
LAMBDA_INIT = base.LAMBDA_INIT
L_EST       = base.L_EST

# --- carica sorgente ---
src    = torch.load(SRC_CKPT, map_location="cpu", weights_only=False)
src_sd = src["model_state"]
ep_src = int(src.get("epoch", 0))

print(f"Sorgente: {SRC_CKPT}  (epoch={ep_src})")
print("Chiavi disponibili:")
for k, v in src_sd.items():
    print(f"  {k}: {tuple(v.shape)}")

# --- costruisci WLISTAHolography ---
model  = WLISTAHolography(K=K, L_est=L_EST, Nx=NX, Ny=NY, Nz=NZ,
                           lambda_init=LAMBDA_INIT)
new_sd = model.state_dict()

for key in ["log_mu", "log_lambda", "log_wx", "log_wy", "log_wz"]:
    if key in src_sd:
        new_sd[key] = src_sd[key].clone()
        print(f"  copiato: {key}")
    else:
        print(f"  [WARN] chiave mancante: {key}")

model.load_state_dict(new_sd)

# --- salva checkpoint compatibile con run_wlista_ken_grasso.py ---
ckpt = dict(
    epoch        = ep_src,
    K            = K,
    Nx           = NX,
    Ny           = NY,
    Nz           = NZ,
    model_type   = "wlista",
    model_state  = model.state_dict(),
    loss         = float("nan"),
    val          = float("nan"),
    best_val     = float("inf"),
    loss_history = [],
    val_history  = [],
    train_idx    = src.get("train_idx", list(range(11))),
    val_idx      = src.get("val_idx", [10]),
)

os.makedirs(os.path.dirname(os.path.abspath(DST_CKPT)), exist_ok=True)
torch.save(ckpt, DST_CKPT)
print(f"\nSalvato: {DST_CKPT}")
print("Puoi ora lanciare:")
print(f'  D:\\holography_scripts\\.conda\\python.exe run_wlista_ken_grasso.py --resume "{DST_CKPT}"')
