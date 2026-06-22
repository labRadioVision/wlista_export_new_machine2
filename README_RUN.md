# Come lanciare gli script — Ken_grasso

Riferimento rapido. Per dettagli/troubleshooting vedi `SESSION_LOG.md`.

Apri questo file nel terminale Jupyter con: `nano README_RUN.md` (Ctrl+X per uscire).

---

## 0. Setup (una volta per istanza)

```bash
cd /workspace
git clone https://github.com/labRadioVision/wlista_export_new_machine2.git
cd wlista_export_new_machine2
pip3 install -r requirements.txt
```

Carica via Jupyter Upload in `Dataset TUM/sinthetic_data/`:
- `E_total_Ken_grasso_nowalls.mat`
- `E_total_freespace_nowalls.mat`

> ⚠️ Usa sempre `python3` (non `python`) e `nohup ... &` (non tmux).

---

## 1. LISTA (caso base)

```bash
nohup python3 run_lista_ken_grasso.py > lista_ken_grasso.log 2>&1 &
tail -f lista_ken_grasso.log
# da un'altra cartella / dopo aver chiuso e riaperto il terminale:
tail -f /workspace/wlista_export_new_machine2/lista_ken_grasso.log
```

## 2. W-LISTA (baseline con pesi per-asse)

```bash
nohup python3 run_wlista_ken_grasso.py > wlista_ken_grasso.log 2>&1 &
tail -f wlista_ken_grasso.log
# da un'altra cartella / dopo aver chiuso e riaperto il terminale:
tail -f /workspace/wlista_export_new_machine2/wlista_ken_grasso.log
```

## 3. LR-W-LISTA — joint training

```bash
nohup python3 run_lowrank_ken_grasso.py --rank 8 > lowrank_ken_grasso.log 2>&1 &
tail -f lowrank_ken_grasso.log
# da un'altra cartella / dopo aver chiuso e riaperto il terminale:
tail -f /workspace/wlista_export_new_machine2/lowrank_ken_grasso.log
```

## 4. LR-W-LISTA — W-FIRST (warm start da W-LISTA)

```bash
# 4a. converti il checkpoint W-LISTA in formato LR
python3 convert_wlista_to_wfirst_ken_grasso.py --rank 8

# 4b. lancia wfirst dal checkpoint convertito
nohup python3 run_wlista_lowrank_wfirst_ken_grasso.py \
    --resume checkpoints_lista_lowrank_wfirst_ken_grasso/wlista_lowrank_wfirst_ken_grasso_r8_converted.pt \
    --warmup 0 \
    > wfirst_ken_grasso.log 2>&1 &
tail -f wfirst_ken_grasso.log
# da un'altra cartella / dopo aver chiuso e riaperto il terminale:
tail -f /workspace/wlista_export_new_machine2/wfirst_ken_grasso.log
```

> ⚠️ In wfirst, dopo il warmup, W continua ad allenarsi insieme a UV — può
> essere instabile (collasso di z osservato a epoca 7). Se succede, usa il
> phased qui sotto (più stabile: congela W dopo il warm start).

## 4bis. LR-W-LISTA — PHASED (autonomo, W congelato dopo fase A — più stabile)

Non richiede un checkpoint W-LISTA pre-esistente: la fase A allena W **da
zero**, dentro lo stesso script.

```bash
nohup python3 run_wlista_lowrank_phased_ken_grasso.py > phased_ken_grasso.log 2>&1 &
tail -f phased_ken_grasso.log
# da un'altra cartella / dopo aver chiuso e riaperto il terminale:
tail -f /workspace/wlista_export_new_machine2/phased_ken_grasso.log
```

Fasi automatiche:
- **A** (30 epoche, solo mu/lambda/W — U,V congelati a zero: equivalente a W-LISTA puro)
- **B** (20 epoche, solo U/V/mu — W e lambda congelati, `LR_LR=1e-3`)
- **C** (10 epoche, fine-tune congiunto a LR ulteriormente ridotte, opzionale — tenuto solo se migliora il val di fase B)

Per saltare la fase A (se hai già un checkpoint di fase A):
```bash
python3 run_wlista_lowrank_phased_ken_grasso.py --skip-a checkpoints_lista_lowrank_phased_ken_grasso/wlista_lowrank_phased_ken_grasso_r8_A_best.pt
```

---

## 4ter. LR-W-LISTA — PHASED su caso synthetic nowalls (PEC/freespace)

Stesso schema di 4bis ma sul caso base `E_total_Ken_PEC_nowalls.mat` +
`E_total_freespace_nowalls.mat` (non Ken_grasso) — già bundled, nessun
upload aggiuntivo richiesto. Train=tutte le 11 posizioni, val=pos 7
(stesso split usato da `run_wlista_synthetic_nowalls_gpu.py`, per confronto
diretto LISTA/W-LISTA vs LR-W-LISTA su questo caso).

```bash
nohup python3 run_wlista_lowrank_phased_synthetic_nowalls.py > phased_synthetic_nowalls.log 2>&1 &
tail -f phased_synthetic_nowalls.log
```

Per saltare la fase A (se hai già un checkpoint di fase A):
```bash
python3 run_wlista_lowrank_phased_synthetic_nowalls.py --skip-a checkpoints_lista_lowrank_phased_synthetic_nowalls/wlista_lowrank_phased_nowalls_r8_A_best.pt
```

---

## 5. Inferenza su tutte le epoche e tutte le posizioni

Per ciascun modello, cicla su TUTTI i checkpoint d'epoca disponibili e su TUTTE
le 11 posizioni del dataset Ken_grasso. Per ogni coppia (epoca, posizione)
salva 3 file (`.png`, `.mat`, `.npz`) + un `metrics.csv` aggregato
(data-consistency, signal/clutter, MSE) in `results_inference_ken_grasso/<prefix>/`.

```bash
# LISTA (caso base)
nohup bash run_inference_sweep_lista_ken_grasso.sh > inference_sweep_lista.log 2>&1 &

# W-LISTA + LR-W-LISTA joint (rank=8)
nohup bash run_inference_sweep_wlista_ken_grasso.sh > inference_sweep_wlista.log 2>&1 &

# LR-W-LISTA W-FIRST (rank=8)
nohup bash run_inference_sweep_wfirst_ken_grasso.sh > inference_sweep_wfirst.log 2>&1 &

# LR-W-LISTA PHASED (rank=8) — separato per fase A/B/C
nohup bash run_inference_sweep_phased_ken_grasso.sh > inference_sweep_phased.log 2>&1 &

# LISTA + W-LISTA plain su caso synthetic nowalls (PEC/freespace)
nohup bash run_inference_sweep_synthetic_nowalls_gpu.sh > inference_sweep_synthetic_nowalls_gpu.log 2>&1 &

# LR-W-LISTA PHASED su caso synthetic nowalls (PEC/freespace, rank=8) — fase A/B/C
nohup bash run_inference_sweep_phased_synthetic_nowalls.sh > inference_sweep_phased_synthetic_nowalls.log 2>&1 &
```

> ⚠️ Genera molti file (epoche × 11 posizioni × 3 file). Lancia uno sweep alla
> volta se la GPU/disco sono condivisi con un training in corso.

Per un singolo modello/prefisso, con controllo granulare:
```bash
python3 loop_inference_ken_grasso.py --prefix wlista_lowrank_ken_grasso_r8 \
    --ckpt-dir checkpoints_lista_lowrank_ken_grasso --start-epoch 1
```

---

## Comandi utili

**Fermare uno script in background:**
```bash
pkill -f run_lista_ken_grasso.py        # sostituisci col nome script
```

**Riprendere training interrotto:**
```bash
nohup python3 run_wlista_ken_grasso.py --resume checkpoints_lista_ken_grasso/wlista_ken_grasso_ep005.pt > wlista_ken_grasso.log 2>&1 &
```

**Vedere ultimo checkpoint salvato:**
```bash
ls -v checkpoints_lista_ken_grasso/*.pt | tail -3
```

**Controllare se un processo è ancora attivo:**
```bash
ps aux | grep python3
```

**Sincronizzare modifiche dal repo (script aggiornati):**
```bash
git pull
```
(non tocca checkpoint/risultati/dati — sono in `.gitignore`)
