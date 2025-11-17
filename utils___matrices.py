import __main__
import re, json, yaml, sys
import numpy as np
from scipy.linalg import expm
from utils___systems import Plant, Controller, Plant_cl, Data, Plant_k
from typing import Tuple, List
from numpy.linalg import norm
from utils___simulate import Open_Loop
from pathlib import Path
from scipy.linalg import solve_sylvester


yaml_path="problem___parameters.yaml"
if yaml is None: raise ImportError("PyYAML not available. Install with `pip install pyyaml`.")
with open(yaml_path, "r", encoding="utf-8") as f:
    cfg = yaml.safe_load(f)


# ------------------------- COMPOSE MATRICES FROM LMI --------------------------

def compose_closed_loop(plant: Plant, ctrl: Controller):
    """
    Build the composite matrices (Abar|Bbar; Cbar|Dbar) for
      [ X_{t+1} ]   [ Abar  Bbar ] [ X_t ]
      [   z_t   ] = [ Cbar  Dbar ] [ w_t ]
    with X = [x; x_c].
    Formula matches the screenshot: blue terms are controller blocks.
    """
    A, Bw, Bu, Cz, Dzw, Dzu, Cy, Dyw = \
        plant.A, plant.Bw, plant.Bu, plant.Cz, plant.Dzw, plant.Dzu, plant.Cy, plant.Dyw
    Ac, Bc, Cc, Dc = ctrl.Ac, ctrl.Bc, ctrl.Cc, ctrl.Dc

    # Top-left block 𝒜:
    A11 = A + Bu @ Dc @ Cy
    A12 = Bu @ Cc
    A21 = Bc @ Cy
    A22 = Ac
    A_cl = np.block([[A11, A12],
                     [A21, A22]])

    # Top-right block 𝓑:
    B1 = Bw + Bu @ Dc @ Dyw
    B2 = Bc @ Dyw
    B_cl = np.vstack([B1, B2])

    # Bottom-left block 𝒞:
    C1 = Cz + Dzu @ Dc @ Cy
    C2 = Dzu @ Cc
    C_cl = np.hstack([C1, C2])

    # Bottom-right block 𝒟:
    D_cl = Dzw + Dzu @ Dc @ Dyw

    return A_cl, B_cl, C_cl, D_cl


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


# ------------------------- PUBLIC API -----------------------------------------

class MatricesAPI():
    def __init__(self):
        self.p = cfg.get("params", {})
        out = self.p.get("directories", {}).get("data", "./out/data/session_")
        m = self.p.get("ambiguity", {}).get("model", "W2")
        runID = self.p.get("directories", {}).get("runID", "temp")
        _type = self.p.get("plant", {}).get("type", "explicit")
        _model = self.p.get("model", "independent") if m == "W2" else m
        #_data = "DDD" if bool(self.p.get("FROM_DATA", False)) else "MBD"

        self.csv_path = out + f"{runID}___{_type}_{_model}.csv"    # _{_data}
        self.use_set_out_mats = bool(self.p.get("use_set_out_mats", False))


    def get_system(self, FROM_DATA: bool = None, Generating_data: bool = False, gamma: float = None, upd: bool = False, **kwargs):
        """
        If FROM_DATA=True, pass data_csv="path/to/file.csv" (and optional settings).
        Example:
        get_system(FROM_DATA=True,
                    data_csv="out/data/run01.csv",
                    delimiter=",",
                    nw=None, ny=None, nz=None,
                    ridge=1e-6)
        """
        FROM_DATA = FROM_DATA if FROM_DATA is not None else self.p.get("FROM_DATA", False)
        
        if FROM_DATA and not Generating_data:                    
            print("\nBuilding system from data...\n\n")
            return self.make_matrices_from_data(gamma=gamma, upd=upd, **kwargs)
        else:
            if self.p.get("plant", {}).get("type", None) == "PaperLike":
                print("\nBuilding paper-like system...\n\n")
                return self.make_paper_like_system()
            else:
                print("\nBuilding example system from YAML...\n\n")
                return self.make_example_system()


    # ------------------------- EXAMPLE SYSTEM CONSTRUCTION -------------------------

    def build_out_matrices(self, nw: int = None) -> tuple[np.ndarray, np.ndarray, np.ndarray, np.ndarray, np.ndarray]:

        # ======================================================================
        # Output-construction helpers
        # ======================================================================

        def make_Cy_Dyw(nx: int, ny: int, nw: int, select: str = "first", dyw_zero: bool = True) -> tuple[np.ndarray, np.ndarray]:
            """
            Measured output model y = Cy x + Dyw w.
            - select="first": Cy picks the first ny states (identity rows).
            - select="random": random orthonormal rows (stable numerics).
            - dyw_zero=True -> Dyw = 0 (typical unless sensor sees w).
            """
            ny = int(min(ny, nx))
            if select == "random":
                Q, _ = np.linalg.qr(np.random.randn(nx, nx))
                Cy = Q[:ny, :]
            else:
                Cy = np.zeros((ny, nx))
                if self.p.get("plant", {}).get("type", None) == "PaperLike":
                    idx = np.array([0, 2, 4])
                    Cy[np.arange(ny), idx] = 1.0
                else:
                    Cy[np.arange(ny), np.arange(ny)] = 1.0

            Dyw = np.zeros((ny, nw)) if dyw_zero else 0.05 * np.random.randn(ny, nw)
            return Cy, Dyw

        def make_performance_A(nx: int, nu: int,
                            Qx_diag: np.ndarray | None = None,
                            Ru_diag: np.ndarray | None = None) -> tuple[np.ndarray, np.ndarray, np.ndarray]:
            """
            Option A (state-input H2-style):
                z = [ Qx^{1/2} x ; Ru^{1/2} u ]
            Returns (Cz, Dzw, Dzu) with Dzw = 0, nz = nx + nu.
            """
            if Qx_diag is None or len(Qx_diag) != nx:
                Qx_diag = np.ones(nx)
            if Ru_diag is None or len(Ru_diag) != nu:
                Ru_diag = 1 * np.ones(nu)

            Qx_sqrt = np.sqrt(np.maximum(Qx_diag, 0.0))
            Ru_sqrt = np.sqrt(np.maximum(Ru_diag, 1e-12))

            Cz_top = np.diag(Qx_sqrt)              # (nx x nx)
            Cz_bot = np.zeros((nu, nx))            # pad for input rows
            Cz = np.vstack([Cz_top, Cz_bot])       # (nx+nu) x nx

            Dzu_top = np.zeros((nx, nu))
            Dzu_bot = np.diag(Ru_sqrt)
            Dzu = np.vstack([Dzu_top, Dzu_bot])    # (nx+nu) x nu

            Dzw = np.zeros((nx + nu, 0))           # placeholder (use correct nw when instantiating)
            return Cz, Dzw, Dzu

        def make_performance_B(Cy: np.ndarray, Dyw: np.ndarray,
                            Wy_diag: np.ndarray | None,
                            Ru_diag: np.ndarray | None) -> tuple[np.ndarray, np.ndarray, np.ndarray]:
            """
            Option B (output-input H2-style):
                z = [ Wy y ; Ru^{1/2} u ]  with y = Cy x + Dyw w
            Returns (Cz, Dzw, Dzu) with nz = ny + nu.
            """
            ny, nx = Cy.shape
            nu = int(Ru_diag.shape[0]) if Ru_diag is not None else None

            if Wy_diag is None:
                Wy_diag = np.ones(ny)
            if Ru_diag is None:
                raise ValueError("Ru_diag must be provided for performance Option B.")

            Wy = np.diag(np.sqrt(np.maximum(Wy_diag, 0.0)))
            Ru_sqrt = np.sqrt(np.maximum(Ru_diag, 1e-12))

            Cz_top = Wy @ Cy                         # (ny x nx)
            Cz_bot = np.zeros((nu, nx))              # (nu x nx)
            Cz = np.vstack([Cz_top, Cz_bot])         # (ny+nu) x nx

            Dzw_top = Wy @ Dyw                       # (ny x nw)
            # Dzw_bot is zero (inputs don't map w directly here)
            # Allocate at runtime if nw known; we return empty placeholder for shape deferral.
            Dzw = np.vstack([Dzw_top, np.zeros((nu, Dzw_top.shape[1]))]) if Dzw_top.size else np.zeros((ny + nu, 0))

            Dzu_top = np.zeros((ny, nu))
            Dzu_bot = np.diag(Ru_sqrt)
            Dzu = np.vstack([Dzu_top, Dzu_bot])      # (ny+nu) x nu

            return Cz, Dzw, Dzu


        # ======================================================================
        # YAML-driven builder (optional)
        # problem___parameters.yaml schema:
        # params:
        #   dimensions: {nx: 4, nw: 2, nu: 2, nz: 3, ny: 2}
        #   outputs:
        #     mode: "A"        # "A" -> state-input, "B" -> output-input
        #     Qx_diag: [1,1,0.3,0.1]
        #     Ru_diag: [0.1,0.1]
        #     Wy_diag: [1,1]   # only for mode "B"
        #     measured: {select: "first", dyw_zero: true}
        # ======================================================================

        """
        Load dimensions + output specs from YAML and produce (Cz, Dzw, Dzu, Cy, Dyw).
        If PyYAML is unavailable, raise a helpful error.
        """


        dims = self.p.get("dimensions", {})
        nx = int(dims.get("nx"))
        nw = int(dims.get("nw")) if nw is None else nw
        nu = int(dims.get("nu"))
        ny = int(dims.get("ny"))

        outspec = self.p.get("outputs", {})
        mode = str(outspec.get("mode", "A")).upper().strip()

        meas = outspec.get("measured", {}) or {}
        Cy, Dyw = make_Cy_Dyw(
            nx=nx, ny=ny, nw=nw,
            select=str(meas.get("select", "first")).lower(),
            dyw_zero=bool(meas.get("dyw_zero", True))
        )

        if mode == "A":
            Qx_diag = np.array(outspec.get("Qx_diag", [1.0] * nx), dtype=float)
            Ru_diag = np.array(outspec.get("Ru_diag", [0.1] * nu), dtype=float)
            Cz, _, Dzu = make_performance_A(nx, nu, Qx_diag=Qx_diag, Ru_diag=Ru_diag)
            # Fill Dzw with the correct width (nw), zeros by construction in Option A
            Dzw = np.zeros((Cz.shape[0], nw))
        elif mode == "B":
            Wy_diag = np.array(outspec.get("Wy_diag", [1.0] * ny), dtype=float)
            Ru_diag = np.array(outspec.get("Ru_diag", [0.1] * nu), dtype=float)
            Cz, Dzw, Dzu = make_performance_B(Cy, Dyw, Wy_diag=Wy_diag, Ru_diag=Ru_diag)
            # Ensure Dzw has correct width
            if Dzw.shape[1] == 0:
                Dzw = np.zeros((Cz.shape[0], nw))
            elif Dzw.shape[1] != nw:
                # pad or trim to nw just in case
                if Dzw.shape[1] > nw:
                    Dzw = Dzw[:, :nw]
                else:
                    Dzw = np.hstack([Dzw, np.zeros((Dzw.shape[0], nw - Dzw.shape[1]))])
        else:
            raise ValueError("outputs.mode must be 'A' or 'B'")

        return Cz, Dzw, Dzu, Cy, Dyw

    def build_AB_from_yaml(self) -> tuple[np.ndarray, np.ndarray, np.ndarray]:
        """
        Build (A, Bu, Bw) from YAML.
        YAML schema example:
        params:
            dimensions: {nx: 4, nw: 2, nu: 2}
            plant:
            type: "random_stable"   # or "explicit"
            seed: 42
            A:
                eig_min: 0.7
                eig_max: 0.95
            Bu:
                scale: 0.5
            Bw:
                rank: 2
                scale: 1.0
            # If type == "explicit", provide numeric arrays:
            # A_mat: [[...],[...],...]
            # Bu_mat: [[...],[...],...]
            # Bw_mat: [[...],[...],...]
        """

        def _stable_A_random(nx: int, eig_min: float, eig_max: float, rng: np.random.Generator) -> np.ndarray:
            """Random orthogonal similarity of diagonal eigenvalues in (eig_min, eig_max)."""
            M = rng.normal(size=(nx, nx))
            Q, _ = np.linalg.qr(M)
            eigvals = eig_min + (eig_max - eig_min) * rng.random(nx)
            return Q @ np.diag(eigvals) @ Q.T


        def _random_full(shape: tuple[int, int], scale: float, rng: np.random.Generator) -> np.ndarray:
            return scale * rng.normal(size=shape)


        def _random_orthonormal_columns(nx: int, rank: int, scale: float, rng: np.random.Generator) -> np.ndarray:
            Q, _ = np.linalg.qr(rng.normal(size=(nx, nx)))
            return Q[:, :rank] * scale

        dims = self.p.get("dimensions", {})
        nx = int(dims["nx"]); nw = int(dims["nw"]); nu = int(dims["nu"])
        plant_cfg = self.p.get("plant", {}) or {}
        ptype = str(plant_cfg.get("type", "random_stable")).lower()
        seed = int(plant_cfg.get("seed", 0))
        rng = np.random.default_rng(seed)

        if ptype == "explicit":
            A = np.array(plant_cfg["A_mat"], dtype=float)
            Bu = np.array(plant_cfg["Bu_mat"], dtype=float)
            Bw = np.array(plant_cfg["Bw_mat"], dtype=float)
            if A.shape != (nx, nx) or Bu.shape != (nx, nu) or Bw.shape != (nx, nw):
                raise ValueError("Explicit matrices do not match (nx,nu,nw) in YAML.")
            return A, Bu, Bw

        # random_stable (default)
        Aconf = plant_cfg.get("A", {}) or {}
        eig_min = float(Aconf.get("eig_min", 0.7))
        eig_max = float(Aconf.get("eig_max", 0.95))
        if not (0.0 < eig_min < eig_max < 1.0):
            raise ValueError("Require 0 < eig_min < eig_max < 1 for stability.")

        Buconf = plant_cfg.get("Bu", {}) or {}
        Bu_scale = float(Buconf.get("scale", 0.5))

        Bwconf = plant_cfg.get("Bw", {}) or {}
        Bw_rank = int(Bwconf.get("rank", nw))
        Bw_scale = float(Bwconf.get("scale", 1.0))
        Bw_rank = max(1, min(Bw_rank, min(nx, nw)))

        A = _stable_A_random(nx, eig_min, eig_max, rng)
        Bu = _random_full((nx, nu), Bu_scale, rng)
        Bw = _random_orthonormal_columns(nx, Bw_rank, Bw_scale, rng)

        # If requested Bw has fewer columns than nw, pad with zeros to match declared nw
        if Bw.shape[1] < nw:
            Bw = np.hstack([Bw, np.zeros((nx, nw - Bw.shape[1]))])

        return A, Bu, Bw

    def get_dimensions_from_yaml(self) -> tuple[int, int, int, int, int]:
        """
        Extract (nx, nw, nu, ny, nz) from YAML file.
        """

        dims = self.p.get("dimensions", {})
        nx = int(dims.get("nx", 0))
        nw = int(dims.get("nw", 0))
        nu = int(dims.get("nu", 0))
        ny = int(dims.get("ny", 0))

        out = self.p.get("outputs", {})
        mode = str(out.get("mode", "A")).upper().strip()
        nz = nx + nu if mode == "A" else ny + nu
        return nx, nw, nu, ny, nz

    def build_initial_Mc(self, nxc: int, ny: int, nu: int) -> tuple[np.ndarray, np.ndarray, np.ndarray, np.ndarray]:
        Ac0 = 0.0 * np.eye(nxc)
        Bc0 = 0.1 * np.eye(nxc, ny)
        Cc0 = 0.1 * np.eye(nu, nxc)
        Dc0 = 0.0 * np.eye(nu, ny)

        return Ac0, Bc0, Cc0, Dc0


    # ------------------------- DATA-DRIVEN DDD CONSTRUCTION ------------------------

    def make_matrices_from_data(
        self,
        gamma: bool = None,
        delimiter: str = ",",
        ridge: float = 1e-6,
        eval: bool = False,
        upd: bool = False
    ):
        """
        Data-driven DT identification with optional Bias-Correction (BC) or Instrumental Variables (IV).

        CSV must have headers x1..x{nx}, u1..u{nu}; optional y1.., z1...
        Builds aligned (X, U, X_next) and estimates:
        - [A  Bu] by ridge regression on centered data (stable numerically)
        - Bw from top energy directions of residual covariance (SVD)
        - Optional outputs by ridge on [X;U;R] / [X;R]
        Options for BC/IV are pulled from self.p["ident"]:
        - use_bc: bool
        - use_iv: bool
        - sigma_v2: float | None          # BC only; if None, estimated by first-differences
        - stable_project: bool            # optional radial shrink of eig(A) for simulation
        - nw_energy: float in (0,1]       # fraction of residual energy for Bw columns
        - X_iv_path: path to CSV containing a second x* sequence (IV instruments)
        Returns (plant, ctrl0); prints diagnostics.
        """

        if not eval:
            if not Path(self.csv_path).exists(): 
                Open_Loop(gamma=gamma)


        data_csv = str(self.csv_path)
        if not data_csv:
            raise("Provide data_csv='path/to/file.csv' to read data.")
            

        # ------------------------- CSV LOADING HELPERS -------------------------

        def _read_csv_with_headers(path: str, delimiter: str = ",") -> Tuple[List[str], np.ndarray]:
            with open(path, "r", encoding="utf-8") as f:
                header = f.readline().strip()
            headers = [h.strip() for h in header.split(delimiter)]
            data = np.loadtxt(path, delimiter=delimiter, skiprows=1)
            if data.ndim == 1:
                data = data[None, :]
            if data.shape[1] != len(headers):
                raise ValueError(f"Column count mismatch: headers={len(headers)} vs data={data.shape[1]}")
            return headers, data

        def _pick_columns(headers: List[str], data: np.ndarray, prefix: str) -> np.ndarray:
            pat = re.compile(rf"^{re.escape(prefix)}(\d+)$", re.IGNORECASE)
            indices = [(i, int(m.group(1))) for i, h in enumerate(headers) if (m := pat.match(h)) is not None]
            if not indices:
                return np.empty((0, data.shape[0]))
            indices.sort(key=lambda t: t[1])
            cols = [i for i, _ in indices]
            block = data[:, cols].T  # (count x T)
            return block

        def _build_blocks_from_csv(path: str, delimiter: str = ","):
            headers, data = _read_csv_with_headers(path, delimiter=delimiter)
            X = _pick_columns(headers, data, "x")
            U = _pick_columns(headers, data, "u")
            Y = _pick_columns(headers, data, "y")
            Z = _pick_columns(headers, data, "z")
            if X.size == 0 or U.size == 0:
                raise ValueError("CSV must contain at least x* and u* columns.")

            Tx = X.shape[1]; Tu = U.shape[1]
            Ty = Y.shape[1] if Y.size else np.inf
            Tz = Z.shape[1] if Z.size else np.inf
            Tpair = int(min(Tx, Tu, Ty, Tz)) - 1
            if Tpair < 1:
                raise ValueError(f"Not enough samples to form (X, X_next): got Tx={Tx}, Tu={Tu}, Ty={Ty}, Tz={Tz}")

            X_reg  = X[:, :Tpair]
            U_reg  = U[:, :Tpair]
            X_next = X[:, 1:Tpair+1]
            Y_reg  = Y[:, :Tpair] if Y.size else None
            Z_reg  = Z[:, :Tpair] if Z.size else None
            return dict(X=X_reg, U=U_reg, X_next=X_next, Y=Y_reg, Z=Z_reg)

        # ------------------------- UTILS -------------------------

        def demean(M): 
            return M - M.mean(axis=1, keepdims=True)

        def _bw_from_residual_svd(R: np.ndarray, energy: float = 0.95) -> Tuple[np.ndarray, int, np.ndarray]:
            nx, T = R.shape
            S = (R @ R.T) / max(T, 1)
            S = 0.5 * (S + S.T) + 1e-12 * np.eye(nx)
            U_s, s_s, _ = np.linalg.svd(S, full_matrices=False)
            cum = np.cumsum(s_s) / max(np.sum(s_s), 1e-18)
            k = int(np.clip(np.searchsorted(cum, energy) + 1, 1, nx))
            Bw = U_s[:, :k] @ np.diag(np.sqrt(s_s[:k]))
            return Bw, k, S

        def _estimate_sigma_v2_diff(X):
            D = np.diff(X, axis=1)
            v2_i = 0.5 * np.var(D, axis=1, ddof=1)
            return float(np.mean(v2_i)), v2_i

        def _project_stable_dt(A, margin=0.995):
            w, V = np.linalg.eig(A)
            w2 = np.where(np.abs(w) >= margin, margin * w / np.abs(w), w)
            return np.real_if_close(V @ np.diag(w2) @ np.linalg.inv(V), tol=1e-10)

        # ------------------------- LOAD DATA -------------------------

        # Identification options
        ident          = self.p.get("ident", {})
        use_bc         = bool(ident.get("use_bc", False))
        zero_mean      = bool(ident.get("zero_mean", False))
        use_iv         = bool(ident.get("use_iv", False))
        plant_iv_ols   = bool(ident.get("plant_iv_ols", False))
        sigma_v2       = None                                              # 2.5e-4
        stable_project = bool(ident.get("stabilise", True))
        nw_energy      = 0.95
        X_iv_path      = ident.get("X_iv_path", None)

        blocks = _build_blocks_from_csv(data_csv, delimiter=delimiter)

        if upd: 
            X = blocks["X"]
            X_next = blocks["X_next"]
            U = blocks["U"]
            Y = blocks["Y"]
            Z = blocks["Z"]
            return Data(X=X, X_next=X_next, U=U, Y=Y, Z=Z, W=None, rx=None, ry=None, rz=None)
        
        X_raw, U_raw, X1_raw = blocks["X"], blocks["U"], blocks["X_next"]
        if zero_mean:
            X, U, X_next = demean(X_raw), demean(U_raw), demean(X1_raw)
        else:
            X, U, X_next = X_raw, U_raw, X1_raw
        T = X_raw.shape[1]

        nx, _, nu, ny, nz = self.get_dimensions_from_yaml()

        # ---------- Raw “moments” for the SDP (no demeaning) ----------
        Phi_raw = np.vstack([U_raw, X_raw])   # (nu+nx) x T
        Z0 = (X_raw @ Phi_raw.T) / T           # nx x (nu+nx)
        Z1 = (X1_raw @ Phi_raw.T) / T           # nx x (nu+nx)

        # ---------- Apply BC/IV substitutions for the SDP ----------
        Z0_use, Z1_use = Z0, Z1
        method_note = "Naive (biased)"

        if use_iv:
            if X_iv_path is None:
                raise ValueError("use_iv=True but no 'X_iv_path' provided.")
            headers_iv, data_iv = _read_csv_with_headers(X_iv_path, delimiter=delimiter)
            X0_iv_raw = _pick_columns(headers_iv, data_iv, "x")
            if X0_iv_raw.shape != X_raw.shape:
                raise ValueError(f"[IV] state shape mismatch: {X0_iv_raw.shape} vs {X_raw.shape}")
            Phi_iv = np.vstack([U_raw, X0_iv_raw])
            Z0_use = (X_raw @ Phi_iv.T) / T
            Z1_use = (X1_raw @ Phi_iv.T) / T
            method_note = "IV moments used in SDP"

        elif use_bc:
            if sigma_v2 is None:
                sigma_v2, _ = _estimate_sigma_v2_diff(X_raw)  # single-run fallback
            Psi = np.hstack([np.zeros((nx, nu)), (T * sigma_v2) * np.eye(nx)])  # nx x (nu+nx)
            Z0_use = Z0 - (1.0 / T) * Psi
            Z1_use = Z1
            method_note = f"BC moments used in SDP (σ_v^2={sigma_v2:.3e})"

        print(f"[DDD] identification mode: {method_note}")

        # ------------------------- A, Bu estimation (ridge on centered time-domain) -------------------------

        D = np.vstack([X, U])                          # (nx+nu) x T
        rank_D = np.linalg.matrix_rank(D)
        if rank_D < (nx + nu):
            raise RuntimeError(f"[PE fail] rank([X;U])={rank_D} < {nx+nu}. Collect richer data.")

        G = D @ D.T + ridge * np.eye(nx + nu)
        W = X_next @ D.T @ np.linalg.inv(G)            # nx x (nx+nu)
        A = W[:, :nx]
        Bu = W[:, nx:]

        if stable_project:
            A = _project_stable_dt(A, margin=0.995)

        # ------------------------- Residuals and Bw (SVD energy cut) -------------------------
        R = X_next - (A @ X + Bu @ U)                  # nx x T
        Bw, nw_eff, _ = _bw_from_residual_svd(R, energy=nw_energy)

        # ------------------------- Outputs (same as before) -------------------------

        if zero_mean:
            Y, Z = demean(blocks["Y"]), demean(blocks["Z"])  
        else: 
            Y, Z = blocks["Y"], blocks["Z"]

        if self.use_set_out_mats or ((Y is None or Y.size == 0) or (Z is None or Z.size == 0)):
            Cz, Dzw, Dzu, Cy, Dyw = self.build_out_matrices(nw=nw_eff)
        else:
            # Y regression on [X; R]
            if ny is None:
                ny = Y.shape[0]
            ThetaY = np.vstack([X, R])
            GY = ThetaY @ ThetaY.T + ridge * np.eye(ThetaY.shape[0])
            WY = Y @ ThetaY.T @ np.linalg.inv(GY)
            Cy = WY[:, :nx]
            Dyw = WY[:, nx:nx+nw_eff]
            if Cy.shape[0] != ny:
                if Cy.shape[0] > ny:
                    Cy = Cy[:ny, :]
                    Dyw = Dyw[:ny, :]
                else:
                    pad = ny - Cy.shape[0]
                    Cy = np.vstack([Cy, np.zeros((pad, nx))])
                    Dyw = np.vstack([Dyw, np.zeros((pad, Dyw.shape[1]))])

            # Z regression on [X; U; R]
            if nz is None:
                nz = Z.shape[0]
            ThetaZ = np.vstack([X, U, R])
            GZ = ThetaZ @ ThetaZ.T + ridge * np.eye(ThetaZ.shape[0])
            WZ = Z @ ThetaZ.T @ np.linalg.inv(GZ)
            Cz  = WZ[:, :nx]
            Dzu = WZ[:, nx:nx+nu]
            Dzw = WZ[:, nx+nu:nx+nu+nw_eff]
            if Cz.shape[0] != nz:
                if Cz.shape[0] > nz:
                    Cz  = Cz[:nz, :]
                    Dzu = Dzu[:nz, :]
                    Dzw = Dzw[:nz, :]
                else:
                    pad = nz - Cz.shape[0]
                    Cz  = np.vstack([Cz,  np.zeros((pad, nx))])
                    Dzu = np.vstack([Dzu, np.zeros((pad, nu))])
                    Dzw = np.vstack([Dzw, np.zeros((pad, Dzw.shape[1]))])

        # ---------- Diagnostics ----------
        rho_A = float(np.max(np.abs(np.linalg.eigvals(A))))
        rel_resid = float(np.linalg.norm(R) / (np.linalg.norm(X_next) + 1e-12))
        print(f"[DDD] rho(A)={rho_A:.6f}, rel_residual={rel_resid:.3e}, Bw_cols={Bw.shape[1]} (energy={nw_energy:.2f})")

        # Build plant
        plant = Plant(A=A, Bw=Bw, Bu=Bu, Cz=Cz, Dzw=Dzw, Dzu=Dzu, Cy=Cy, Dyw=Dyw)

        # Neutral controller seed (full-order, tiny static gains)
        Ac0, Bc0, Cc0, Dc0 = self.build_initial_Mc(nxc=nx, ny=Cy.shape[0], nu=Bu.shape[1])
        ctrl0 = Controller(Ac=Ac0, Bc=Bc0, Cc=Cc0, Dc=Dc0)

        return plant, ctrl0

    def change_of_coordinates(self, plant: Plant, T: np.ndarray) -> Plant:
        """
        Apply state-space change of coordinates x_new = T x.
        Returns new Plant with transformed A, Bu, Bw, Cz.
        """

        T_inv = np.linalg.inv(T)
        A_new  = T @ plant.A @ T_inv
        Bu_new = T @ plant.Bu
        Bw_new = T @ plant.Bw
        Cz_new = plant.Cz @ T_inv

        return Plant(A=A_new, Bu=Bu_new, Bw=Bw_new,
                     Cz=Cz_new, Dzw=plant.Dzw,
                     Dzu=plant.Dzu, Cy=plant.Cy,
                     Dyw=plant.Dyw)
    
    def K_rapresentation_change(self, plant: Plant, ctrl: Controller) -> Plant_k:
        # Dzw, Dyw are zero by definition in our setups
        A, Bu, Bw = plant.A, plant.Bu, plant.Bw
        Cy, Cz, Dzu = plant.Cy, plant.Cz, plant.Dzu
        Ac, Bc, Cc, Dc = ctrl.Ac, ctrl.Bc, ctrl.Cc, ctrl.Dc

        Abar = np.block([[A, 0], [Bc @ Cy, Ac]])
        Bbar = np.block([[Bu], [np.zeros((Ac.shape[0], Bu.shape[1]))]])
        K = np.block([[Dc @ Cy, Cc]])
        V = np.block([[Bw], [np.zeros((Ac.shape[0], Bw.shape[1]))]])
        Cbar = np.block([[Cz, np.zeros((Cz.shape[0], Ac.shape[1]))]])
        Dbar = Dzu

        return Plant_k(A=Abar, B=Bbar, C=Cbar, D=Dbar, K=K, V=V)

    # ------------------------- LEGACY EXAMPLE (unchanged) --------------------------

    def make_example_system(self):
        """
        Replace this with your real matrices.
        Discrete-time example with modest dimensions.
        """


        dims = self.p.get("dimensions", {})
        nx = int(dims["nx"]); ny = int(dims["ny"]); nu = int(dims["nu"])

        A, Bu, Bw = self.build_AB_from_yaml()
        Cz, Dzw, Dzu, Cy, Dyw = self.build_out_matrices()

        plant = Plant(A=A, Bw=Bw, Bu=Bu, Cz=Cz, Dzw=Dzw, Dzu=Dzu, Cy=Cy, Dyw=Dyw)

        # Full-order controller as a starting point; zero dynamics + small static gain
        nxc = nx
        Ac0, Bc0, Cc0, Dc0 = self.build_initial_Mc(nxc=nxc, ny=ny, nu=nu)

        ctrl0 = Controller(Ac=Ac0, Bc=Bc0, Cc=Cc0, Dc=Dc0)
        return plant, ctrl0


    # -------------------------- PAPER_LIKE MBD EXAMPLE ---------------------------------

    def make_paper_like_system(self):
        # Turbine parameters
        J = 4e7  # Rotor inertia (kg·m^2)
        k_omega = -1e5  # Torque sensitivity to rotor speed (N·m·s/rad)
        k_h = 1e4  # Torque sensitivity to flapwise displacement (N·m/m)
        k_phi = -1e4  # Torque sensitivity to torsional angle (N·m/rad)
        k_beta = -1e6  # Torque sensitivity to pitch angle (N·m/rad)
        k_v = 5e5  # Torque sensitivity to wind speed (N·m·s/m)
        m = 1e4  # Effective blade mass (kg)
        omega_f = 6.28  # Flapwise natural frequency (rad/s)
        zeta_f = 0.05  # Flapwise damping ratio
        f_omega = 1e3  # Force sensitivity to rotor speed (N·s/rad)
        f_beta = -1e4  # Force sensitivity to pitch angle (N/rad)
        f_v = 2e4  # Force sensitivity to wind speed (N·s/m)
        I_t = 1e5  # Torsional inertia (kg·m^2)
        omega_t = 31.4  # Torsional natural frequency (rad/s)
        zeta_t = 0.02  # Torsional damping ratio
        m_omega = 1e2  # Moment sensitivity to rotor speed (N·m·s/rad)
        m_beta = -1e3  # Moment sensitivity to pitch angle (N·m/rad)
        m_v = 1e4  # Moment sensitivity to wind speed (N·m·s/m)
        omega_p = 10  # Pitch actuator natural frequency (rad/s)
        zeta_p = 0.7  # Pitch actuator damping ratio

        # Turbine state-space matrices
        A_continuous = np.array([
            [k_omega/J, k_h/J, 0, k_phi/J, 0, k_beta/J, 0],  # omega_dot
            [0, 0, 1, 0, 0, 0, 0],  # h_dot
            [f_omega/m, -omega_f**2, -2*zeta_f*omega_f, 0, 0, f_beta/m, 0],  # h_ddot
            [0, 0, 0, 0, 1, 0, 0],  # phi_dot
            [m_omega/I_t, 0, 0, -omega_t**2, -2*zeta_t*omega_t, m_beta/I_t, 0],  # phi_ddot
            [0, 0, 0, 0, 0, 0, 1],  # beta_dot
            [0, 0, 0, 0, 0, -omega_p**2, -2*zeta_p*omega_p]  # beta_ddot
        ])
        B_continuous = np.array([[0], [0], [0], [0], [0], [0], [omega_p**2]])  # Input: beta_dot_c
        E_continuous = np.array([
            [k_v/J, 0],  # omega (v_z only)
            [0, 0],      # h
            [f_v/m, f_v/m],  # h_dot (v_x, v_z)
            [0, 0],      # phi
            [m_v/I_t, m_v/I_t],  # phi_dot (v_x, v_z)
            [0, 0],      # beta
            [0, 0]       # beta_dot
        ])  # 7x2: [v_x, v_z]

        dt = self.p.get("simulation", {}).get("ts", 0.5)
        nx, _, nu, ny, _ = self.get_dimensions_from_yaml()


        A = expm(A_continuous * dt)
        def discretize_input(A_c, B_c, dt):
            n = A_c.shape[0]
            m = B_c.shape[1]
            Phi = expm(np.block([[A_c, B_c], [np.zeros((m, n)), np.zeros((m, m))]]) * dt)
            A_d = Phi[:n, :n]
            B_d = Phi[:n, n:]
            return A_d, B_d

        _, Bu = discretize_input(A_continuous, B_continuous, dt)
        _, Bw = discretize_input(A_continuous, E_continuous, dt)

        # Consider scaling of random noise
        Bw = Bw/np.sqrt(dt)

        Cz, Dzw, Dzu, Cy, Dyw = self.build_out_matrices()
        Ac0, Bc0, Cc0, Dc0 = self.build_initial_Mc(nxc=nx, ny=ny, nu=nu)

        plant = Plant(A=A, Bw=Bw, Bu=Bu, Cz=Cz, Dzw=Dzw, Dzu=Dzu, Cy=Cy, Dyw=Dyw)
        ctrl0 = Controller(Ac=Ac0, Bc=Bc0, Cc=Cc0, Dc=Dc0)
        return plant, ctrl0

    def _augment_matrices(self, B_w, D_vw, D_yw, var: float = 0, Sigma_nom: np.ndarray = None, N: tuple = None) -> tuple[np.ndarray, np.ndarray, np.ndarray, int, np.ndarray]:
        nx, _, _, ny, nz = self.get_dimensions_from_yaml()

        if N is None:
            n1, n2 = nx, ny
        else:
            n1, n2 = N

        B_w = np.block([[B_w, (1e-4)*np.eye(nx, n1), np.zeros((nx, n2))]])
        D_vw = np.block([[D_vw, np.zeros((nz, n1 + n2))]])
        D_yw = np.block([[D_yw,np.zeros((ny, n1)),(1e-4)*np.eye(ny, n2)]])
        n_w = B_w.shape[1]
        
        if Sigma_nom is None:
            Sigma_nom = np.eye(n_w) if var==0 else var * np.eye(n_w)
        else:
            extra_dim = n_w - Sigma_nom.shape[0]
            if extra_dim > 0:
                Sigma_nom = np.block([[Sigma_nom, np.zeros((Sigma_nom.shape[0], extra_dim))],
                                      [np.zeros((extra_dim, Sigma_nom.shape[1])), (var if var>0 else 1e-6)*np.eye(extra_dim)]])
        return B_w, D_vw, D_yw, n_w, Sigma_nom


    # ------------------------- PRINTING HELPERS -------------------------------------

    def print_plant(self, plant: Plant):
        print("\nPlant Matrices:")
        print(f"A [{plant.A.shape}]:\n", plant.A)
        print(f"Bw [{plant.Bw.shape}]:\n", plant.Bw)
        print(f"Bu [{plant.Bu.shape}]:\n", plant.Bu)
        print(f"Cz [{plant.Cz.shape}]:\n", plant.Cz)
        print(f"Dzw [{plant.Dzw.shape}]:\n", plant.Dzw)
        print(f"Dzu [{plant.Dzu.shape}]:\n", plant.Dzu)
        print(f"Cy [{plant.Cy.shape}]:\n", plant.Cy)
        print(f"Dyw [{plant.Dyw.shape}]:\n", plant.Dyw)
        print("\n\n")

    def print_plant_cl(self, plant_cl: Plant_cl):
        print("\nComposite Plant Matrices:")
        print(f"Acl [{plant_cl.Acl.shape}]:\n", plant_cl.Acl)
        print(f"Bcl [{plant_cl.Bcl.shape}]:\n", plant_cl.Bcl)
        print(f"Ccl [{plant_cl.Ccl.shape}]:\n", plant_cl.Ccl)
        print(f"Dcl [{plant_cl.Dcl.shape}]:\n", plant_cl.Dcl)
        print("\n\n")

    def print_controller(self, ctrl: Controller):
        print("\n\nController Matrices:")
        print(f"Ac [{ctrl.Ac.shape}]:\n", ctrl.Ac)
        print(f"Bc [{ctrl.Bc.shape}]:\n", ctrl.Bc)
        print(f"Cc [{ctrl.Cc.shape}]:\n", ctrl.Cc)
        print(f"Dc [{ctrl.Dc.shape}]:\n", ctrl.Dc)
        print("\n\n")


# ------------------------- Main execution -------------------------------------

if __name__== "__main__":

    mat = MatricesAPI()
    plant, ctrl0 = mat.get_system()
    mat.print_plant(plant)

