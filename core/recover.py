import json
import numpy as np
from numpy.linalg import norm
from scipy.linalg import solve_sylvester

from core.systems import Plant, Controller
from config import cfg


inp = bool(cfg.get("params", {}).get("inp", 0))



# ------------------------- RECOVER ERR MAT (YOUNG) --------------------------

def recover_deltas(P, X, Y, M, N, Cy, 
                   Ahat, Buhat,
                   beta_AA, beta_AB,
                   eps=1e-12, rcond=1e-10):
    
    def _as2d_float(a):
        a = np.asarray(a)
        if a.ndim != 2:
            raise ValueError(f"Expected 2D array, got shape {a.shape}")
        return np.asarray(a, dtype=float, order='C')

    def _solve_sylvester_safe(A, B, C, rcond=1e-12):
        """Try scipy sylvester; on failure, fall back to Kronecker least-squares."""
        A = _as2d_float(A); B = _as2d_float(B); C = _as2d_float(C)
        # Try standard solver first
        try:
            X = solve_sylvester(A, B, C)
            X = np.real_if_close(X)
            return X
        except Exception as e:
            # Kronecker fallback: (I ⊗ A + B^T ⊗ I) vec(X) = vec(C)
            n, n2 = A.shape; m, m2 = B.shape
            if n != n2 or m != m2 or C.shape != (n, m):
                raise ValueError(f"Sylvester dims incompatible: A{A.shape}, B{B.shape}, C{C.shape}") from e
            K = np.kron(np.eye(m), A) + np.kron(B.T, np.eye(n))
            rhs = C.reshape(-1,  order='F')
            X_vec, *_ = np.linalg.lstsq(K, rhs, rcond=rcond)
            X = X_vec.reshape(n, m, order='F')
            X = np.real_if_close(X)
            return X

    def build_A_lift(Ax, Bu, Cy, X, Y, K, L, M, N):
        n = Ax.shape[0]
        # coerce to float arrays to avoid object dtypes from cvxpy
        Ax = _as2d_float(Ax); Bu=_as2d_float(Bu); Cy=_as2d_float(Cy)
        X=_as2d_float(X); Y=_as2d_float(Y); M=_as2d_float(M); N=_as2d_float(N)
        K=_as2d_float(K); L=_as2d_float(L)
        A11 = Ax @ Y + Bu @ M
        A12 = Ax + Bu @ (N @ Cy)
        A21 = K
        A22 = X @ Ax + L @ Cy
        return np.block([[A11, A12],
                        [A21, A22]])

    def EAA_of(DeltaA, X, Y):
        n = DeltaA.shape[0]
        return np.block([[DeltaA @ Y, DeltaA],
                        [np.zeros((n,n)), X @ DeltaA]])

    def EAB_of(DeltaB, M, N, Cy):
        n = DeltaB.shape[0]
        return np.block([[DeltaB @ M, DeltaB @ (N @ Cy)],
                        [np.zeros((n,n)), np.zeros((n,n))]])


    n, m = Buhat.shape
    # Coerce everything to plain float arrays early
    P=_as2d_float(P); X=_as2d_float(X); Y=_as2d_float(Y)
    M=_as2d_float(M); N=_as2d_float(N); Cy=_as2d_float(Cy)
    Ahat=_as2d_float(Ahat); Buhat=_as2d_float(Buhat)

    # Use zeros for K,L if you didn’t store the optimal ones; else pass the real K*,L*
    K0 = np.zeros((n, n), dtype=float)
    L0 = np.zeros((n, Cy.shape[0]), dtype=float)

    A_lift = build_A_lift(Ahat, Buhat, Cy, X, Y, K0, L0, M, N)
    Z = P @ A_lift
    Z = _as2d_float(Z)
    Z_norm = float(np.linalg.norm(Z, 'fro'))

    if not np.isfinite(Z_norm) or Z_norm < eps:
        # Degenerate: return zeros
        return (np.zeros_like(Ahat),
                np.zeros_like(Buhat))

    # Young-saturating directions
    E_AA_star = (float(beta_AA) / Z_norm) * Z
    E_AB_star = (float(beta_AB) / Z_norm) * Z

    # Blocks for ΔA
    A1 = E_AA_star[0:n,    0:n   ]
    A2 = E_AA_star[0:n,    n:2*n ]
    A3 = E_AA_star[n:2*n,  n:2*n ]

    LHS_left  = X.T @ X
    LHS_right = Y @ Y.T + np.eye(n)
    RHS = A2 + A1 @ Y.T + X.T @ A3

    # Defensive coercions
    LHS_left  = _as2d_float(LHS_left)
    LHS_right = _as2d_float(LHS_right)
    RHS       = _as2d_float(RHS)

    DeltaA = _solve_sylvester_safe(LHS_left, LHS_right, RHS, rcond=rcond)

    # Blocks for ΔB
    # Top row of E_AB_star: [ ΔB M | ΔB N Cy ]
    T1 = E_AB_star[0:n, 0:n]
    T2 = E_AB_star[0:n, n:2*n]

    NCy = N @ Cy
    GB  = (M @ M.T) + (NCy @ NCy.T)           # (m x m)
    RHS_B = T1 @ M.T + T2 @ (Cy.T @ N.T)      # (n x m)

    # Pseudoinverse with SVD
    U, s, Vt = np.linalg.svd(_as2d_float(GB), full_matrices=False)
    s_inv = np.where(s > rcond, 1.0/s, 0.0)
    GB_pinv = (Vt.T * s_inv) @ U.T
    DeltaB = _as2d_float(RHS_B) @ GB_pinv

    # Clean tiny imaginary residues if any
    DeltaA = np.real_if_close(DeltaA)
    DeltaB = np.real_if_close(DeltaB)
    return DeltaA, DeltaB, EAB_of(DeltaB, M, N, Cy), EAA_of(DeltaA, X, Y)

# ------------------------- RECOVER MATRICES FROM CLOSED-LOOP ------------------

class Recover():
    def __init__(self):
        pass

    def load_dro_json(self, path: str):
        with open(path, "r", encoding="utf-8") as f:
            d = json.load(f)
        # Required pieces
        Pbar = np.array(d["Pbar"], dtype=float)
        Abar = np.array(d["Abar"], dtype=float)
        Bbar = np.array(d["Bbar"], dtype=float)
        Cbar = np.array(d["Cbar"], dtype=float)
        Dbar = np.array(d["Dbar"], dtype=float)
        return Pbar, Abar, Bbar, Cbar, Dbar, d

    def Mc_from_bar(self, res, plant: Plant):
        X_val = res.X
        Y_val = res.Y
        K_val = res.K
        L_val = res.L
        M_val = res.M
        N_val = res.N

        A, Bu, Cy = plant.A, plant.Bu, plant.Cy
        nx = X_val[0].size

        try:
            U, S, Vt = np.linalg.svd(np.eye(nx) - X_val @ Y_val)
            V = Vt.T
            epsilon = 1e-10
            S_sqrt = np.diag(np.sqrt(np.maximum(S, epsilon)))
            S_sqrt_inv = np.diag(1.0 / np.sqrt(np.maximum(S, epsilon)))
            U_new = U @ S_sqrt
            V_new = V @ S_sqrt
            U_new_inv = S_sqrt_inv @ U.T
            V_new_inv_T = V @ S_sqrt_inv
            A_c = U_new_inv @ (K_val - X_val @ A @ Y_val - L_val @ Cy @ Y_val - X_val @ Bu @ (M_val - N_val @ Cy @ Y_val)) @ V_new_inv_T
            B_c = U_new_inv @ (L_val - X_val @ Bu @ N_val)
            C_c = (M_val - N_val @ Cy @ Y_val) @ V_new_inv_T
            D_c = N_val
        except np.linalg.LinAlgError:
            print("Error: Singular matrix in controller reconstruction. Using fallback.")
            if inp: input("seriously?")
            A_c = K_val - X_val @ A @ Y_val - L_val @ Y_val - X_val @ Bu @ M_val
            B_c = L_val
            C_c = M_val
            D_c = N_val
        
        return A_c, B_c, C_c, D_c

    def recover_controller_from_closed_loop(self, plant: Plant, M_cl):
        """
        Solve for Dc, Cc, Bc, Ac using least-squares when needed.
        Returns Controller and a residual report.
        """
        A_cl, B_cl, C_cl, D_cl = M_cl
        A, Bw, Bu, Cz, Dzw, Dzu, Cy, Dyw = plant.A, plant.Bw, plant.Bu, plant.Cz, plant.Dzw, plant.Dzu, plant.Cy, plant.Dyw
        nx = A.shape[0]

        if A_cl.shape[0] - nx <= 0: raise ValueError("Composite A_cl has invalid size relative to plant nx.")

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


