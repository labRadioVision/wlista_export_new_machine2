"""
convert_wlista_to_wfirst_ken_grasso.py
=======================================
Converte un checkpoint WLISTAHolography (Ken_grasso) in un checkpoint
compatibile con run_wlista_lowrank_wfirst_ken_grasso.py (LRWLISTAHolography).

Parametri copiati: log_mu, log_lambda, log_wx, log_wy, log_wz
Parametri nuovi:   U_re=0, U_im=0, V_re=randn*1e-2, V_im=randn*1e-2

Uso:
  python3 convert_wlista_to_wfirst_ken_grasso.py
  python3 convert_wlista_to_wfirst_ken_grasso.py --src checkpoints_lista_ken_grasso/wlista_ken_grasso_ep010.pt
  python3 convert_wlista_to_wfirst_ken_grasso.py --rank 16 --src checkpoints_lista_ken_grasso/wlista_ken_grasso_best.pt
"""

import os, sys, argparse, torch

SCRIPT_DIR = os.path.dirname(os.path.abspath(__file__))
sys.path.insert(0, SCRIPT_DIR)

import run_wlista_synthetic_nowalls as base
from lista_holography_lowrank import LRWLISTAHolography

_ap = argparse.ArgumentParser()
_ap.add_argument("--src", default=os.path.join(SCRIPT_DIR,
                 "checkpoints_lista_ken_grasso", "wlista_ken_grasso_best.pt"),
                 help="checkpoint WLISTAHolography sorgente")
_ap.add_argument("--dst", default=None,
                 help="path output (default: auto)")
_ap.add_argument("--rank", type=int, default=8)
_args = _ap.parse_args()

SRC_CKPT = os.path.abspath(_args.src)
RANK     = _args.rank

if _args.dst is None:
    _base = os.path.splitext(os.path.basename(SRC_CKPT))[0]
    _tag  = _base.split("_ep")[-1] if "_ep" in _base else "converted"
    DST_DIR = os.path.join(SCRIPT_DIR, "checkpoints_lista_lowrank_wfirst_ken_grasso")
    os.makedirs(DST_DIR, exist_ok=True)
    DST_CKPT = os.path.join(DST_DIR,
                             f"wlista_lowrank_wfirst_ken_grasso_r{RANK}_{_tag}.pt")
else:
    DST_CKPT = os.path.abspath(_args.dst)

print("=" * 68)
print("convert_wlista_to_wfirst_ken_grasso")
print(f"  src  : {SRC_CKPT}")
print(f"  dst  : {DST_CKPT}")
print(f"  rank : {RANK}")
print(f"  src esiste: {os.path.exists(SRC_CKPT)}")
print("=" * 68)

K           = base.K
NX, NY, NZ  = base.NX, base.NY, base.NZ
LAMBDA_INIT = base.LAMBDA_INIT
L_EST       = base.L_EST

# --- carica sorgente ---
if not os.path.exists(SRC_CKPT):
    raise FileNotFoundError(f"Checkpoint sorgente non trovato: {SRC_CKPT}\n"
                             f"Esegui prima run_wlista_ken_grasso.py")

src    = torch.load(SRC_CKPT, map_location="cpu", weights_only=False)
src_sd = src["model_state"]

print("Chiavi W-LISTA source:")
for k, v in src_sd.items():
    print(f"  {k}: {tuple(v.shape)}")

assert src_sd["log_wx"].shape == (K, NX), \
    f"shape mismatch log_wx: {src_sd['log_wx'].shape} vs atteso ({K},{NX})"

M = int(src.get("M", 12960))   # 162 * 80 = 12960
print(f"\nM (ricevitori) = {M}  rank={RANK}")

model  = LRWLISTAHolography(K=K, L_est=L_EST, Nx=NX, Ny=NY, Nz=NZ,
                              M=M, rank=RANK, lambda_init=LAMBDA_INIT)
new_sd = model.state_dict()

# --- copia pesi W-LISTA ---
for key in ["log_mu", "log_lambda", "log_wx", "log_wy", "log_wz"]:
    if key in src_sd:
        new_sd[key] = src_sd[key].clone()
        print(f"  copiato: {key}")
    else:
        print(f"  [WARN] chiave mancante: {key}")

print(f"  U_re inizializzato a zero: {tuple(new_sd['U_re'].shape)}")
print(f"  V_re inizializzato randn*1e-2: {tuple(new_sd['V_re'].shape)}")

model.load_state_dict(new_sd)

# --- salva checkpoint compatibile con wfirst ---
ep_src = int(src.get("epoch", 0))
ckpt = dict(
    epoch          = ep_src,
    K              = K,
    Nx             = NX, Ny=NY, Nz=NZ,
    M              = M,
    rank           = RANK,
    model_state    = model.state_dict(),
    loss           = float("nan"),
    val            = float("nan"),
    best_val       = float("inf"),
    loss_history   = [],
    val_history    = [],
    warmup_epochs  = 6,
    train_idx      = list(range(11)),
    val_idx        = [10],
)

torch.save(ckpt, DST_CKPT)
print(f"\nSalvato: {DST_CKPT}")
print("Puoi ora lanciare:")
print(f'  nohup python3 run_wlista_lowrank_wfirst_ken_grasso.py --rank {RANK} --resume "{DST_CKPT}" > wfirst_ken_grasso.log 2>&1 &')
