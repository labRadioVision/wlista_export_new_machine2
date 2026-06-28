# SESSION_LOG — W-LISTA / LR-W-LISTA su dati FEKO sintetici

Ultimo aggiornamento: 2026-06-29

---

## 1. Repository e struttura cartelle

```
wlista_export/                        ← cartella git (push → GitHub)
├── Dataset TUM/sinthetic_data/       ← dati .mat (in .gitignore, non committati)
│   ├── E_total_Ken_PEC_nowalls.mat   # 11 posizioni, fantasma PEC
│   ├── E_total_freespace_nowalls.mat # campo incidente (per nowalls)
│   ├── E_total_Ken_grasso_nowalls.mat# n posizioni, fantasma "grasso"
│   ├── E_total_Ken_muscle_newpos.mat # 22 posizioni, fantasma muscolo
│   └── E_inc.mat                     # campo incidente (per muscle_newpos)
├── run_wlista_synthetic_nowalls.py   ← modulo base (loader, plot, train LISTA/W-LISTA)
├── run_wlista_ken_grasso.py          ← W-LISTA su Ken_grasso_nowalls
├── run_lowrank_ken_grasso.py         ← LR-W-LISTA joint su Ken_grasso
├── run_wlista_lowrank_wfirst_ken_grasso.py  ← LR-W-LISTA W-FIRST
├── run_wlista_lowrank_phased_ken_grasso.py  ← LR-W-LISTA PHASED (A/B/C)
├── run_wlista_lowrank_phased_nowalls.py     ← LR-W-LISTA PHASED su nowalls
├── run_wlista_ken_muscle_newpos.py   ← W-LISTA su Ken_muscle_newpos (NUOVO)
├── loop_inference_ken_grasso.py      ← inferenza sweep su tutti i checkpoint
├── collect_val_loss.py               ← estrae loss/val da checkpoint → .mat/.csv
├── convert_wlista_to_wfirst_ken_grasso.py  ← converte checkpoint W-LISTA → W-FIRST
├── inference_common.py               ← utility condivise (MF, ISTA, plot, metriche)
├── holography_operator_fast.py       ← operatore holografico GPU (DLPack/CuPy)
├── lista_holography_weighted.py      ← WLISTAHolography
├── lista_holography_lowrank.py       ← LRWLISTAHolography
├── generate_z_true.py                ← modello corpo parametrico (z_true)
├── README_RUN.md                     ← comandi rapidi per vast.ai
└── SESSION_LOG.md                    ← questo file
```

Repository GitHub: https://github.com/labRadioVision/wlista_export_new_machine2

---

## 2. Modelli

### W-LISTA
LISTA con pesi spaziali per-asse in scala logaritmica: `log_wx`, `log_wy`, `log_wz` (shape K×N_x/y/z), `log_mu` (step size), `log_lambda` (soglia).

### LR-W-LISTA
W-LISTA + correzione low-rank ΔT = U·V^H (U_re, U_im, V_re, V_im, rank R). Aggiunge flessibilità senza esplodere il numero di parametri.

### Varianti di training LR-W-LISTA

| Variante | Descrizione |
|---|---|
| **joint** | Tutti i parametri (W + UV) allenati insieme dall'inizio |
| **W-FIRST** | Warmup su W, poi W continua ad allenarsi con UV → rischio collasso z |
| **PHASED** | Fase A: solo W (UV=0); Fase B: solo UV+mu (W congelato); Fase C: joint fine-tune |

Il collasso di z (output zeroed) fu osservato in W-FIRST a epoca 7 con LR_W alto → PHASED è più stabile.

---

## 3. Iperparametri

### Base (`run_wlista_synthetic_nowalls.py`)

| Parametro | Valore | Note |
|---|---|---|
| K | 10 | layer LISTA |
| N_EPOCHS | 30 | |
| LR | 5e-2 | sovrascritta localmente in tutti gli script figli |
| LR_W | 2.5e-2 | sovrascritta localmente |
| LAMBDA_INIT | 1e-4 | |
| L_EST | 1.141e4 | stima lipschitz |
| W_LOG_CLAMP | 4.0 | clamp su log_w (era 5.0, ridotto) |
| DELTA_Z | 1.5297+0j | contrasto dielettrico muscolo a 2.45 GHz |

### Override per Ken_grasso e muscle_newpos

| Parametro | Valore | Motivazione |
|---|---|---|
| LR | 1e-2 | base 5e-2 troppo alto |
| LR_W | 2.5e-1 | base 5e-1 troppo alto (z-collapse osservato) |

### PHASED (entrambi i dataset)

| Parametro | Valore |
|---|---|
| RANK | 8 |
| LR_BASE | 1e-2 |
| LR_W_BASE | 1e-1 |
| LR_LR (UV, fase B) | 1e-3 |
| LR_MU_B | 1e-2 |
| LR_C_SCALE | 0.1 |
| FREEZE_LAMBDA_B | True |
| ALPHA_Z / BETA_DATA / GAMMA_REG | 1.0 / 1e-4 / 1e-1 |
| PHASE_A_EPOCHS | 10 |
| PHASE_B_EPOCHS (ken_grasso) | 20 |
| PHASE_B_EPOCHS (nowalls) | 10 |
| PHASE_C_EPOCHS | 10 |

**Nessuno scheduler LR**: Adam con LR fisso per tutta la durata di ogni fase. La stabilizzazione visiva delle MIP nelle prime epoche è comportamento naturale di Adam (accumulo momenti del secondo ordine + gradient clipping `max_norm=1.0`), non un decay esplicito.

---

## 4. Dataset

### nowalls (E_total_Ken_PEC_nowalls)
- 11 posizioni, fantasma PEC
- TRAIN_IDX=[0..7], VAL_IDX=[8,9,10]
- Campo incidente: `E_total_freespace_nowalls.mat`

### Ken_grasso_nowalls (E_total_Ken_grasso_nowalls)
- N posizioni (auto-detect: chiave `E_mag_total_mann` shape (162,80,N))
- TRAIN_IDX=list(range(N)), VAL_IDX=[N-1]
- Campo incidente: `E_total_freespace_nowalls.mat`

### Ken_muscle_newpos (E_total_Ken_muscle_newpos) — NUOVO
- 22 posizioni, fantasma muscolo (DELTA_Z=1.5297, invariato)
- Campo incidente: `E_inc.mat` (chiavi `E_total_freespace`, `E_phase_freespace`)
- Griglia 4 profondità × 7 laterali + 1 centrale

```
Coordinate olografiche: holo_x = feko_y + 2.510,  holo_z = 3.317 - feko_x

  hz=1.717 (feko_x=1.60): idx 0              → 1 pos (centro)
  hz=1.427 (feko_x=1.89): idx 1..7           → 7 pos laterali
  hz=1.137 (feko_x=2.18): idx 8..14          → 7 pos laterali
  hz=0.847 (feko_x=2.47): idx 15..21         → 7 pos laterali
```

Script corrente usa solo profondità hz=0.847 (caso semplificato/veloce):
- TRAIN_IDX = [15..21] — target che trasla da hx=0.91 a hx=3.71
- VAL_IDX   = [15]     — posizione più interna (hx=2.110, feko_y=-0.4)

Nota: idx 15 è sia in train che in val (convenzione ken_grasso).

---

## 5. Bug corretti

| Bug | Fix |
|---|---|
| `import run_wlista_synthetic as base` | → `import run_wlista_synthetic_nowalls as base` |
| Workaround su vast.ai senza git pull | `ln -s run_wlista_synthetic_nowalls.py run_wlista_synthetic.py` |
| `python: command not found` | usare sempre `python3` |
| `ModuleNotFoundError: scipy` | `pip3 install -r requirements.txt` |
| z-collapse a epoca 7 in W-FIRST | Ridotto LR_W + passato a PHASED |
| LR_LR=1e-2 per UV troppo alto | → 1e-3 |
| W_LOG_CLAMP=5.0 | → 4.0 |
| RANK=16 default | → 8 |
| PHASE_A_EPOCHS=30 | → 10 |
| PHASE_B_EPOCHS=20 nowalls | → 10 (switch ogni 10 epoche per uniformità) |
| Snapshot PHASED senza MF/ISTA | Calcolati una volta per fase, passati a `save_epoch_snapshot` |
| `base.EINC_FILE` non sovrascritta in muscle_newpos | Aggiunto `base.EINC_FILE = EINC_FILE` |

---

## 6. Comandi run (vast.ai)

> ⚠️ Usare sempre `python3` e `nohup ... &`. Mai `tmux`.

### Setup iniziale (una volta per istanza)

```bash
cd /workspace
git clone https://github.com/labRadioVision/wlista_export_new_machine2.git
cd wlista_export_new_machine2
pip3 install -r requirements.txt
# Upload via Jupyter in Dataset TUM/sinthetic_data/:
#   E_total_Ken_grasso_nowalls.mat, E_total_freespace_nowalls.mat
#   E_total_Ken_muscle_newpos.mat, E_inc.mat
```

### W-LISTA Ken_grasso

```bash
nohup python3 run_wlista_ken_grasso.py > wlista_ken_grasso.log 2>&1 &
tail -f wlista_ken_grasso.log

# Resume:
nohup python3 run_wlista_ken_grasso.py \
    --resume checkpoints_lista_ken_grasso/wlista_ken_grasso_ep005.pt \
    > wlista_ken_grasso.log 2>&1 &
```

### W-LISTA Ken_muscle_newpos

```bash
nohup python3 run_wlista_ken_muscle_newpos.py > wlista_ken_muscle_newpos.log 2>&1 &
tail -f wlista_ken_muscle_newpos.log

# Resume:
nohup python3 run_wlista_ken_muscle_newpos.py \
    --resume checkpoints_lista_ken_muscle_newpos/wlista_ken_muscle_newpos_ep005.pt \
    > wlista_ken_muscle_newpos.log 2>&1 &
```

### LR-W-LISTA joint Ken_grasso

```bash
nohup python3 run_lowrank_ken_grasso.py --rank 8 > lowrank_ken_grasso.log 2>&1 &
tail -f lowrank_ken_grasso.log
```

### LR-W-LISTA W-FIRST

```bash
python3 convert_wlista_to_wfirst_ken_grasso.py --rank 8
nohup python3 run_wlista_lowrank_wfirst_ken_grasso.py \
    --resume checkpoints_lista_lowrank_wfirst_ken_grasso/wlista_lowrank_wfirst_ken_grasso_r8_converted.pt \
    --warmup 0 > wfirst_ken_grasso.log 2>&1 &
tail -f wfirst_ken_grasso.log
```

### LR-W-LISTA PHASED Ken_grasso

```bash
nohup python3 run_wlista_lowrank_phased_ken_grasso.py > phased_ken_grasso.log 2>&1 &
tail -f phased_ken_grasso.log

# Resume fase interrotta a metà:
python3 run_wlista_lowrank_phased_ken_grasso.py \
    --resume-a checkpoints_lista_lowrank_phased_ken_grasso/wlista_lowrank_phased_ken_grasso_r8_A_ep006.pt
python3 run_wlista_lowrank_phased_ken_grasso.py \
    --resume-b checkpoints_lista_lowrank_phased_ken_grasso/wlista_lowrank_phased_ken_grasso_r8_B_ep012.pt
python3 run_wlista_lowrank_phased_ken_grasso.py \
    --resume-c checkpoints_lista_lowrank_phased_ken_grasso/wlista_lowrank_phased_ken_grasso_r8_C_ep003.pt

# Salta fase A già completata:
python3 run_wlista_lowrank_phased_ken_grasso.py \
    --skip-a checkpoints_lista_lowrank_phased_ken_grasso/wlista_lowrank_phased_ken_grasso_r8_A_best.pt
```

### LR-W-LISTA PHASED nowalls

```bash
nohup python3 run_wlista_lowrank_phased_nowalls.py > phased_nowalls.log 2>&1 &
tail -f phased_nowalls.log
# Resume/skip: stessa sintassi di ken_grasso (sostituire "ken_grasso" con "nowalls")
```

---

## 7. Output e risultati

### Checkpoint

```
checkpoints_lista_ken_grasso/               wlista_ken_grasso_ep001.pt .. _best.pt
checkpoints_lista_ken_muscle_newpos/        wlista_ken_muscle_newpos_ep001.pt ..
checkpoints_lista_lowrank_ken_grasso/       wlista_lowrank_ken_grasso_r8_ep001.pt ..
checkpoints_lista_lowrank_phased_ken_grasso/ .._A_ep001.pt .. _B_ .. _C_ .. _best.pt
checkpoints_lista_lowrank_phased_nowalls/    idem con "nowalls"
```

Campi salvati: `epoch`, `model_state`, `optim_state`, `loss`, `val`, `best_val`, `loss_history`, `val_history`, `train_idx`, `val_idx`.

### Snapshot per-epoca (da `REF_EPOCH_START=5`)

```
results_*/epoch_recon[_A|_B|_C]/   PNG per epoca: modello vs MF vs ISTA vs GT (val ref)
```

### Plot finali (a fine training)

```
results_*/
  {name}_train_imgs.png    MIP di tutti i campioni train
  {name}_train_stats.png   curva loss, mu/lambda, S/C ratio
  {name}_train_weights.png heatmap pesi wx/wy/wz per layer (W-LISTA only)
  {name}_val_*.png         stessi per val set
  {name}_{label}_z.mat     z_net + z_mf reshape 3D per campione
```

### Export loss/val

```bash
python3 collect_val_loss.py \
    --ckpt-dir checkpoints_lista_lowrank_phased_ken_grasso \
    --prefix wlista_lowrank_phased_ken_grasso_r8_A \
    --out val_loss_phased_A
# → val_loss_phased_A.mat + .npz + .csv  (colonne: epoch, loss, val, best_val)
```

### Inference sweep

```bash
nohup bash run_inference_sweep_phased_ken_grasso.sh > inference_sweep_phased.log 2>&1 &
# → results_inference_ken_grasso/{prefix}/ PNG + .mat + .npz per (epoca, posizione)
# → results_inference_ken_grasso/{prefix}/metrics.csv
```

---

## 8. Gestione processi su vast.ai

```bash
ps aux | grep python3                          # processi attivi
pkill -f run_wlista_ken_grasso.py              # ferma script specifico
ls -v checkpoints_lista_ken_grasso/*.pt | tail -3   # ultimo checkpoint

# Terminale bloccato (Ctrl+C non risponde):
# Jupyter → File menu → Terminals → Shut Down All → nuovo terminale

git pull                                       # sincronizza script (non tocca dati/checkpoint)

# Scaricare risultati:
zip -r results_muscle.zip results_synthetic_ken_muscle_newpos/
# poi scaricare via Jupyter file browser
```

---

## 9. Note metodologiche

**LR scheduling**: nessuno scheduler esplicito. La stabilizzazione delle MIP dopo epoch 1-2 è effetto Adam (accumulo momenti) + gradient clipping, non decay artificioso.

**LR_W inconsistenza (nota aperta)**: plain W-LISTA nowalls usa `base.LR_W=2.5e-2`; Fase A del PHASED nowalls usa `LR_W_BASE=1e-1` (4×). Se si vuole Fase A metodologicamente equivalente al W-LISTA plain, allineare i valori.

**Auto-detect tipo modello**: `ic.detect_kind()` controlla chiavi `state_dict` (`U_re` → lrwlista, `log_wx` → wlista, altrimenti lista). Rank auto-rilevato da `sd["U_re"].shape`.

**Resume**: ripristina `model_state`, `optim_state` (momenti Adam), `loss_history`, `val_history`, `best_val`. `start_epoch = ck['epoch'] + 1`.
