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
nohup python3 run_lowrank_ken_grasso.py --rank 16 > lowrank_ken_grasso.log 2>&1 &
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
