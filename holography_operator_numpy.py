# -*- coding: utf-8 -*-
"""
holography_operator_numpy.py
============================
Operatore A / A^H in NumPy PURO (nessun cupy/CUDA) per inferenza su CPU.

Kernel di Green identico a holography_operator(_fast).py (complex128), quindi
numericamente equivalente: su macchine SENZA GPU permette di eseguire SOLO
l'inferenza (lento, ma indipendente da CUDA).

API compatibile con HolographyOperatorFast per la parte usata dall'inferenza:
  - A(z) / AH(b)            : numpy in/out (complex128)
  - A_np(z) / AH_np(b)      : alias numpy-in/out (per baseline MF/ISTA unificati)
  - A_torch(z) / AH_torch(b): autograd torch su CPU (forward A -> backward A^H)
  - attributi N_rx, N_vox, batch_rx
"""

import sys, time
import numpy as np

# torch è OPZIONALE: serve solo per A_torch/AH_torch (autograd CPU).
# Il path NumPy puro (A/AH/A_np/AH_np), usato p.es. da run_mf_nowalls_cpu.py,
# funziona senza torch -> ambiente CPU minimale (numpy/scipy/matplotlib).
try:
    import torch
    _HAS_TORCH = True
except ImportError:
    torch = None
    _HAS_TORCH = False


def _green_yy(r_obs_batch, r_src, k, omega, mu0):
    dr   = r_obs_batch[:, None, :] - r_src[None, :, :]      # (B, N_vox, 3)
    R    = np.linalg.norm(dr, axis=-1)                       # (B, N_vox)
    dy   = dr[:, :, 1]
    G_sc = np.exp(-1j * k * R) / (4.0 * np.pi * R)
    alpha = -1j * k - 1.0 / R
    d2G   = G_sc * (alpha**2 * (dy / R)**2
                    + (dy / R)**2 / R**2
                    + alpha * (1.0 - (dy / R)**2) / R)
    return -1j * omega * mu0 * (G_sc + d2G / k**2)


class HolographyOperatorNumpy:
    def __init__(self, r_rx, r_vox, k, omega, mu0, dV, batch_rx=100):
        self.r_rx  = np.asarray(r_rx,  dtype=np.float64)
        self.r_vox = np.asarray(r_vox, dtype=np.float64)
        self.k     = float(k); self.omega = float(omega)
        self.mu0   = float(mu0); self.dV = float(dV)
        self.batch_rx = int(batch_rx)
        self.N_rx  = int(self.r_rx.shape[0])
        self.N_vox = int(self.r_vox.shape[0])
        self.verbose = False   # set True per progress CPU

    def _progress(self, label, i0, t0):
        n_batches = -(-self.N_rx // self.batch_rx)   # ceil
        done = i0 // self.batch_rx + 1
        pct  = done / n_batches
        bar  = int(pct * 20)
        elapsed = time.time() - t0
        eta = (elapsed / pct * (1 - pct)) if pct > 0 else 0
        sys.stdout.write(
            f"\r    {label}  [{'=' * bar}{' ' * (20 - bar)}]"
            f"  {done}/{n_batches} batch  {pct*100:4.0f}%"
            f"  {elapsed:5.1f}s  ETA {eta:5.1f}s"
        )
        sys.stdout.flush()

    def A(self, z):
        z   = np.asarray(z, dtype=np.complex128)
        out = np.zeros(self.N_rx, dtype=np.complex128)
        t0  = time.time() if self.verbose else None
        for i0 in range(0, self.N_rx, self.batch_rx):
            i1 = min(i0 + self.batch_rx, self.N_rx)
            G  = _green_yy(self.r_rx[i0:i1], self.r_vox, self.k, self.omega, self.mu0)
            out[i0:i1] = (G * self.dV) @ z
            if self.verbose: self._progress("A ", i0, t0)
        if self.verbose: sys.stdout.write("\n"); sys.stdout.flush()
        return out

    def AH(self, b):
        b   = np.asarray(b, dtype=np.complex128)
        out = np.zeros(self.N_vox, dtype=np.complex128)
        t0  = time.time() if self.verbose else None
        for i0 in range(0, self.N_rx, self.batch_rx):
            i1 = min(i0 + self.batch_rx, self.N_rx)
            G  = _green_yy(self.r_rx[i0:i1], self.r_vox, self.k, self.omega, self.mu0)
            out += (np.conj(G * self.dV) * b[i0:i1, None]).sum(axis=0)
            if self.verbose: self._progress("A^H", i0, t0)
        if self.verbose: sys.stdout.write("\n"); sys.stdout.flush()
        return out

    # alias numpy-in/out (baseline unificati)
    def A_np(self, z):  return self.A(z)
    def AH_np(self, b): return self.AH(b)

    # interfaccia torch (CPU, gradient-aware) -- richiede torch installato
    def A_torch(self, z):
        if not _HAS_TORCH:
            raise RuntimeError("A_torch richiede torch: pip install torch "
                               "(non necessario per il path NumPy A/AH).")
        return _ApplyA.apply(z, self)
    def AH_torch(self, b):
        if not _HAS_TORCH:
            raise RuntimeError("AH_torch richiede torch: pip install torch "
                               "(non necessario per il path NumPy A/AH).")
        return _ApplyAH.apply(b, self)


if _HAS_TORCH:
    class _ApplyA(torch.autograd.Function):
        @staticmethod
        def forward(ctx, z, op):
            ctx.op = op; ctx.device = z.device
            b = op.A(z.detach().cpu().numpy())
            return torch.as_tensor(b.astype(np.complex64), dtype=torch.cfloat, device=z.device)
        @staticmethod
        def backward(ctx, grad_b):
            gz = ctx.op.AH(grad_b.detach().cpu().numpy())
            return torch.as_tensor(gz.astype(np.complex64), dtype=torch.cfloat, device=ctx.device), None


    class _ApplyAH(torch.autograd.Function):
        @staticmethod
        def forward(ctx, b, op):
            ctx.op = op; ctx.device = b.device
            z = op.AH(b.detach().cpu().numpy())
            return torch.as_tensor(z.astype(np.complex64), dtype=torch.cfloat, device=b.device)
        @staticmethod
        def backward(ctx, grad_z):
            gb = ctx.op.A(grad_z.detach().cpu().numpy())
            return torch.as_tensor(gb.astype(np.complex64), dtype=torch.cfloat, device=ctx.device), None
