# recover_controller.py
import json
import numpy as np
from numpy.linalg import cholesky, inv, lstsq, eigvals, norm
from systems import Plant, Controller

def load_dro_json(path: str):
    with open(path, "r", encoding="utf-8") as f:
        d = json.load(f)
    # Required pieces
    Pbar = np.array(d["Pbar"], dtype=float)
    Abar = np.array(d["Abar"], dtype=float)
    Bbar = np.array(d["Bbar"], dtype=float)
    Cbar = np.array(d["Cbar"], dtype=float)
    Dbar = np.array(d["Dbar"], dtype=float)
    return Pbar, Abar, Bbar, Cbar, Dbar, d

def _nearest_pd(A, jitter=1e-9):
    """Higham-like projection to the nearest PD matrix."""
    A = 0.5 * (A + A.T)
    w, V = np.linalg.eigh(A)
    w_clipped = np.maximum(w, jitter)
    return (V * w_clipped) @ V.T

def closed_loop_from_bar(Pbar, Abar, Bbar, Cbar, Dbar, jitter=1e-9):
    """
    Direct gauge: set T = I, P = Pbar. Then:
      A_cl = Pbar^{-1} Abar
      B_cl = Pbar^{-1} Bbar
      C_cl = Cbar
      D_cl = Dbar
    This damps explosive scalings when Pbar has huge entries.
    """
    Pbar = np.array(Pbar, dtype=float)
    Abar = np.array(Abar, dtype=float)
    Bbar = np.array(Bbar, dtype=float)
    Cbar = np.array(Cbar, dtype=float)
    Dbar = np.array(Dbar, dtype=float)

    # Symmetrize and ensure invertibility (nearest PD if needed)
    Pbar = 0.5 * (Pbar + Pbar.T)
    try:
        Pinv = np.linalg.inv(Pbar)
    except np.linalg.LinAlgError:
        # project to nearest PD then invert
        w, V = np.linalg.eigh(Pbar)
        w_clip = np.maximum(w, jitter)
        Pbar_pd = (V * w_clip) @ V.T
        Pinv = np.linalg.inv(Pbar_pd)

    A_cl = Pinv @ Abar
    B_cl = Pinv @ Bbar
    C_cl = Cbar
    D_cl = Dbar
    return A_cl, B_cl, C_cl, D_cl



def recover_controller_from_closed_loop(plant: Plant, A_cl, B_cl, C_cl, D_cl, rcond=1e-9):
    """
    Solve for Dc, Cc, Bc, Ac using least-squares when needed.
    Returns Controller and a residual report.
    """
    A, Bw, Bu, Cz, Dzw, Dzu, Cy, Dyw = plant.A, plant.Bw, plant.Bu, plant.Cz, plant.Dzw, plant.Dzu, plant.Cy, plant.Dyw
    nx = A.shape[0]
    nxc = A_cl.shape[0] - nx
    if nxc <= 0:
        raise ValueError("Composite A_cl has invalid size relative to plant nx.")

    # Partition composite matrices
    A11 = A_cl[:nx, :nx]
    A12 = A_cl[:nx, nx:]
    A21 = A_cl[nx:, :nx]
    A22 = A_cl[nx:, nx:]

    B1 = B_cl[:nx, :]
    B2 = B_cl[nx:, :]

    C1 = C_cl[:, :nx]
    C2 = C_cl[:, nx:]

    # 1) Recover Dc from D_cl = Dzw + Dzu Dc Dyw  ->  Dzu Dc Dyw = D_cl - Dzw
    RHS = D_cl - Dzw
    def _tikhonov_left(A, B, alpha=1e-8):
        # Solve A X ≈ B: (A^T A + alpha I) X = A^T B
        _, n = A.shape
        return np.linalg.solve(A.T @ A + alpha*np.eye(n), A.T @ B)

    # Dc from Dzu Dc Dyw = RHS using two-sided Tikhonov
    Dc_mid = _tikhonov_left(Dyw.T, RHS.T, alpha=1e-8).T
    Dc = _tikhonov_left(Dzu, Dc_mid, alpha=1e-8)

    # Cc = Dzu^\dagger C2   and  Bc = B2 Dyw^\dagger  with Tikhonov
    Cc = _tikhonov_left(Dzu, C2, alpha=1e-8)
    Bc = (_tikhonov_left(Dyw.T, B2.T, alpha=1e-8)).T

    # 4) Recover Ac directly
    Ac = A22

    # Residual checks (sanity)
    res = {}
    res["A12"] = norm(A12 - Bu @ Cc) / (1 + norm(A12))
    res["A21"] = norm(A21 - Bc @ Cy) / (1 + norm(A21))
    res["A11"] = norm(A11 - (A + Bu @ Dc @ Cy)) / (1 + norm(A11))
    res["B1"]  = norm(B1  - (Bw + Bu @ Dc @ Dyw)) / (1 + norm(B1))
    res["C1"]  = norm(C1  - (Cz + Dzu @ Dc @ Cy)) / (1 + norm(C1))
    res["D"]   = norm(D_cl - (Dzw + Dzu @ Dc @ Dyw)) / (1 + norm(D_cl))

    return Controller(Ac=Ac, Bc=Bc, Cc=Cc, Dc=Dc), res

def recover_controller_from_dro_json(json_path: str, plant: Plant):
    Pbar, Abar, Bbar, Cbar, Dbar, meta = load_dro_json(json_path)
    A_cl, B_cl, C_cl, D_cl = closed_loop_from_bar(Pbar, Abar, Bbar, Cbar, Dbar)
    ctrl, residuals = recover_controller_from_closed_loop(plant, A_cl, B_cl, C_cl, D_cl)
    # Quick stability peek on composite A
    rho = max(abs(eigvals(A_cl)))
    return ctrl, residuals, float(rho)
