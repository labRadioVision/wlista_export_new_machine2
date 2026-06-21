# Session log — Ken_grasso scripts + vast.ai setup
**Data:** 2026-06-19

---

## Script creati

Script in `wlista_export_new_machine2/` per dataset sintetico Ken_grasso:

| Script | Modello | Note |
|--------|---------|------|
| `run_lista_ken_grasso.py` | LISTA plain (caso base) | nessun peso per-asse, nessuna correzione low-rank |
| `run_wlista_ken_grasso.py` | W-LISTA baseline | da lanciare per primo |
| `run_lowrank_ken_grasso.py` | LR-W-LISTA (rank=8) | joint training |
| `run_wlista_lowrank_wfirst_ken_grasso.py` | LR-W-LISTA W-FIRST | warmup W-only, poi tutto insieme (W NON si congela) |
| `run_wlista_lowrank_phased_ken_grasso.py` | LR-W-LISTA PHASED | Fase A allena W da zero, poi congelato — solo U,V,mu |
| `convert_wlista_to_wfirst_ken_grasso.py` | — | converte ckpt W-LISTA → wfirst (solo per wfirst, non serve a phased) |
| `loop_inference_ken_grasso.py` | — | inferenza su tutte le epoche/posizioni (auto-detect modello) |
| `run_inference_sweep_lista_ken_grasso.sh` | — | wrapper sweep per LISTA |
| `run_inference_sweep_wlista_ken_grasso.sh` | — | wrapper sweep per W-LISTA + LR-W-LISTA joint |
| `run_inference_sweep_wfirst_ken_grasso.sh` | — | wrapper sweep per LR-W-LISTA W-FIRST |
| `run_inference_sweep_phased_ken_grasso.sh` | — | wrapper sweep per LR-W-LISTA PHASED (fasi A/B/C separate) |

**wfirst vs phased:** wfirst richiede un checkpoint W-LISTA pre-esistente
(via `convert_wlista_to_wfirst_ken_grasso.py`) e dopo il warmup continua ad
allenare W insieme a UV (puo' essere instabile: osservato collasso di z a
epoca 7 con LR_W alto). **phased e' autonomo**: la Fase A allena W **da
zero, dentro lo stesso script** (U,V congelati a zero durante la Fase A —
matematicamente equivalente a W-LISTA puro, nessun checkpoint esterno
richiesto), poi la Fase B CONGELA W e allena solo `U,V,log_mu` con
`LR_LR=1e-3` (basso: la correzione low-rank e' un raffinamento fine, non va
spinto forte), infine una Fase C opzionale di fine-tune congiunto a LR
ulteriormente ridotte. **phased e' la scelta consigliata** dopo il collasso
osservato in wfirst.

**Inferenza (`loop_inference_ken_grasso.py`):** ispirato a `loop_inference_synthetic.py`
in `holography_scripts/`. Riusa `inference_common.py` (`build_model` con
auto-detect LISTA/W-LISTA/LR-W-LISTA, `build_operator`, `run_model`,
baseline MF/ISTA, `metrics`, `plot_panels`, `save_outputs`). Differenza
principale: cicla su TUTTE le 11 posizioni (non una sola di riferimento),
perche' per Ken_grasso lo z_true e' disponibile per ognuna (body model).
Output per ogni (epoca, posizione): `.png` (pannelli GT/MF/ISTA/modello),
`.mat` e `.npz` (volumi + MIP + metriche), in
`results_inference_ken_grasso/<prefix>/`, con `metrics.csv` aggregato
(data_consistency, signal_clutter, mse_full, mse_occ, mse_bg).

**Differenze rispetto agli script in `holography_scripts/`:**
- `SCRIPT_DIR` punta alla cartella dello script stesso (non a holography_scripts)
- `Dataset TUM/` è una sottocartella di `wlista_export_new_machine2/` (non cartella sorella)
- Output e checkpoint vanno in `wlista_export_new_machine2/results_*` e `checkpoints_*`
- Import: `run_wlista_synthetic_nowalls` (non `run_wlista_synthetic`)

**Fix learning rate (2026-06-20):** `base.LR=5e-2` e `base.LR_W=5e-1` (da
`run_wlista_synthetic_nowalls.py`) erano troppo alti per Ken_grasso — causa
probabile del collasso di z osservato in wfirst a epoca 7 (soglia di
soft-threshold esplosa). Ridotti localmente (override, NON tocca `base.py`)
in `run_wlista_ken_grasso.py`, `run_lowrank_ken_grasso.py`,
`run_wlista_lowrank_wfirst_ken_grasso.py`: `LR=1e-2`, `LR_W=1e-1` (fattore 5x).
`run_lista_ken_grasso.py` non modificato (nessun peso per-asse).

**Bug corretto:** `run_wlista_ken_grasso.py` e `run_lowrank_ken_grasso.py` importavano
`run_wlista_synthetic` (inesistente) → corretto in `run_wlista_synthetic_nowalls`.

---

## Come eseguire su vast.ai via Jupyter browser

SSH non funziona affidabilmente su vast.ai (publickey denied). Usare sempre il terminale Jupyter dal browser.

### Sequenza completa (testata)

**1. Crea istanza** con template PyTorch+CUDA su vast.ai.

**2. Apri Jupyter** → vast.ai → Instances → Open

**3. Nuovo terminale** → File → New → Terminal

**4. Clona repo e installa dipendenze**
```bash
cd /workspace
git clone https://github.com/labRadioVision/wlista_export_new_machine2.git
cd wlista_export_new_machine2
pip3 install -r requirements.txt
# NOTA: usare pip3 (non pip) — testato su istanza RTX 4090
```

**5. Carica i file `.mat`** via Jupyter file browser (Upload) in `Dataset TUM/sinthetic_data/`:
- `E_total_Ken_grasso_nowalls.mat`
- `E_total_freespace_nowalls.mat`

**6. Lancia il training — USA SEMPRE `nohup` E `python3`**

> ⚠️ **IMPORTANTE:** usare obbligatoriamente `nohup` (non tmux) e `python3` (non python).
> - `nohup` garantisce che il processo continui anche dopo la chiusura del tab browser
> - `python3` perché su questi container `python` non è nel PATH

```bash
nohup python3 run_wlista_ken_grasso.py > wlista_ken_grasso.log 2>&1 &
echo "PID: $!"
```

**7. Monitora**
```bash
tail -f wlista_ken_grasso.log
```

**8. Esci dal monitor** → chiudi il tab del browser. Il training continua su vast.ai.

**9. Se il terminale si blocca** → Jupyter menu → Terminals → Shut Down All Terminals → poi apri un nuovo terminale.

**10. Controlla lo stato** da nuovo terminale:
```bash
tail -f /workspace/wlista_export_new_machine2/wlista_ken_grasso.log
```

> **Nota:** `tmux` non funziona bene dal terminale Jupyter browser perché Ctrl+B viene intercettato dal browser. Usare `nohup ... &` è più affidabile.

Sequenza training: prima W-LISTA, poi LR-W-LISTA con `--resume` dal checkpoint W-LISTA:
```bash
nohup python3 run_lowrank_ken_grasso.py --resume checkpoints_lista_ken_grasso/wlista_ken_grasso_best.pt > lowrank_ken_grasso.log 2>&1 &
```

---

## Problemi vast.ai incontrati

### Errore 1 — CDI device error
```
failed to inject CDI devices: unresolvable CDI devices
```
**Causa:** host GPU mal configurato sul lato vast.ai.  
**Fix:** Destroy istanza, scegliere un host diverso.

### Errore 2 — SSH Permission denied (publickey)
```
root@ssh3.vast.ai: Permission denied (publickey).
root@ssh7.vast.ai: Permission denied (publickey).
```
**Causa probabile:** vast.ai non inietta correttamente le chiavi SSH nei container.  
**Tentativi fatti:**
- Verifica che la chiave pubblica su Account → SSH Keys corrisponda alla locale ✓
- Correzione permessi Windows: `icacls id_ed25519 /inheritance:r /grant:r` ✓
- Generazione nuova chiave `id_vastai` e reinserimento su vast.ai
- Provate 3 istanze diverse — stesso errore

**Workaround:** usare il terminale Jupyter direttamente dal browser (vast.ai → Instances → Open), senza SSH.

---

## Chiave SSH attiva

```
C:\Users\STEFANOSAVAZZI\.ssh\id_vastai
```
Pubblica:
```
ssh-ed25519 AAAAC3NzaC1lZDI1NTE5AAAAIOTlRRJ0ZiULQ1Sp6i5sim4+qR8w1PxRq5b6gN5iQ9EJ
```

---

## Note

- Checkpoint per ogni epoca: `checkpoints_lista_ken_grasso/wlista_ken_grasso_ep001.pt` ecc.
- Resume manuale: `python run_wlista_ken_grasso.py --resume checkpoints_lista_ken_grasso/wlista_ken_grasso_ep005.pt`
- 10 epoche sufficienti (loss si appiattisce prima di 30)
- Costo stimato su RTX 3090: ~5.5h/epoca × 10 epoche ≈ 55h ≈ €10-15
