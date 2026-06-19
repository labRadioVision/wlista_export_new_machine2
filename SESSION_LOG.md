# Session log — Ken_grasso scripts + vast.ai setup
**Data:** 2026-06-19

---

## Script creati

Script in `wlista_export_new_machine2/` per dataset sintetico Ken_grasso:

| Script | Modello | Note |
|--------|---------|------|
| `run_wlista_ken_grasso.py` | W-LISTA baseline | da lanciare per primo |
| `run_lowrank_ken_grasso.py` | LR-W-LISTA (rank=16) | joint training |
| `run_wlista_lowrank_wfirst_ken_grasso.py` | LR-W-LISTA W-FIRST | warmup W-only, poi UV |
| `convert_wlista_to_wfirst_ken_grasso.py` | — | converte ckpt W-LISTA → wfirst |

> ⚠️ Il file `run_wlista_lowrank_phased_ken_grasso.py` è un esperimento non testato (approccio alternativo, fasi separate). Non includerlo nei run principali.

**Differenze rispetto agli script in `holography_scripts/`:**
- `SCRIPT_DIR` punta alla cartella dello script stesso (non a holography_scripts)
- `Dataset TUM/` è una sottocartella di `wlista_export_new_machine2/` (non cartella sorella)
- Output e checkpoint vanno in `wlista_export_new_machine2/results_*` e `checkpoints_*`
- Import: `run_wlista_synthetic_nowalls` (non `run_wlista_synthetic`)

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
