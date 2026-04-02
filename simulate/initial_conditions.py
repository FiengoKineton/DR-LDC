#!/usr/bin/env python3
import numpy as np


def _initial_condition_from_eigenvalues(M, scale=0.05):
    eigvals, eigvecs = np.linalg.eig(M)
    idx = np.argsort(np.real(eigvals))   # sorted by real part
    critical_idx = idx[-1]               # least stable / most critical
    v = eigvecs[:, critical_idx]         # shape (n,)

    v_real = np.real(v)
    norm = np.linalg.norm(v_real)
    if norm == 0:
        raise ValueError("Eigenvector has zero real part; pick another mode.")

    v_real = v_real / norm               # normalize
    x0 = scale * v_real                  # shape (n,)
    return x0.reshape(-1, 1)             # shape (n, 1)

