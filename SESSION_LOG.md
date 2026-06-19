# Session log — Ken_grasso scripts + vast.ai setup
**Data:** 2026-06-19

---

## Script creati

Due nuovi script in `wlista_export_new_machine2/`:

- `run_wlista_ken_grasso.py` — W-LISTA baseline su `E_total_Ken_grasso_nowalls.mat`
- `run_lowrank_ken_grasso.py` — LR-W-LISTA (rank=16) sullo stesso dataset

**Differenze rispetto agli script in `holography_scripts/`:**
- `SCRIPT_DIR` punta alla cartella dello script stesso (non a holography_scripts)
- `Dataset TUM/` è una sottocartella di `wlista_export_new_machine2/` (non cartella sorella)
- Output e checkpoint vanno in `wlista_export_new_machine2/results_*` e `checkpoints_*`

**Bug corretto in `run_lowrank_ken_grasso.py`:** rimossa la riga `SCRIPT_DIR = base.SCRIPT_DIR`
che sovrascriveva il path corretto con quello di holography_scripts.

---

## Come eseguire su macchina remota

```powershell
# 1. Copia la cartella completa (PowerShell locale)
scp -P <PORTA> -r "C:\Users\STEFANOSAVAZZI\Desktop\OneDrive - CNR\HOLDEN_files\Holography_export\wlista_export_new_machine2" root@<HOST>:~/

# 2. Connettiti SSH
ssh -i C:\Users\STEFANOSAVAZZI\.ssh\id_vastai -p <PORTA> root@<HOST>

# 3. Avvia training in tmux
cd ~/wlista_export_new_machine2
tmux new -s ken
python run_wlista_ken_grasso.py > wlista_ken_grasso.log 2>&1

# Detach: Ctrl+B poi D
# Rientra: tmux attach -t ken
# Controlla: tail -f wlista_ken_grasso.log
```

Sequenza: prima W-LISTA, poi LR-W-LISTA con `--resume` dal checkpoint W-LISTA.

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
