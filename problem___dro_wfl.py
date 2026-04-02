import sys
import numpy as np, casadi as ca
from scipy.signal import savgol_filter

from core import (
    DROLMIResult, Noise,            # systems.py
    MatricesAPI,                    # matrices.py
)
from disturbances import Disturbances
from simulate import Open_Loop


# =============================================================================================== #

class WFL:
    def __new__(cls, *args, **kwargs):
        return WFL_nonConvex(*args, **kwargs)

# =============================================================================================== #

class Utils: 
    def __init__(self, eps=1e-6):
        self.eps = eps
        self.aux_vars = []

    # ------------------------------------------------------------------
    @staticmethod
    def _I(n, m=None):
        m = n if m is None else m
        return np.eye(n, m)

    @staticmethod
    def _Z(n, m=None):
        m = n if m is None else m
        return np.zeros((n, m))
    
    @staticmethod
    def _val(m):
        if m is None:
            return None
        return float(m) if np.isscalar(m) else m

    @staticmethod
    def select_representative_run(datasets, keys=("X", "U", "Y", "Z", "X_next"),
                                  weights=None):
        if not datasets:
            raise ValueError("datasets is empty")

        T_min = min(d["meta"]["T"] for d in datasets)
        Teff = T_min
        Teff_next = T_min - 1
        if Teff_next <= 0:
            raise ValueError("Need T >= 2 to align X_next")

        stacks = {}
        for k in keys:
            if k == "X_next":
                stacks[k] = np.stack([d[k][..., :Teff_next] for d in datasets], axis=0)
            else:
                stacks[k] = np.stack([d[k][..., :Teff] for d in datasets], axis=0)

        N = next(iter(stacks.values())).shape[0]
        if weights is None:
            weights = {k: 1.0 for k in keys}

        def fro_scale(S):
            return np.mean([np.linalg.norm(S[i]) for i in range(S.shape[0])]) + 1e-12

        scales = {k: fro_scale(S) for k, S in stacks.items()}

        dists = np.zeros(N)
        for i in range(N):
            tot = 0.0
            for k, S in stacks.items():
                Sk = S / scales[k]
                diff = Sk - Sk[i]
                tot += float(weights[k]) * np.sum(diff ** 2)
            dists[i] = tot
        i_star = int(np.argmin(dists))

        out = {
            "X": stacks["X"][i_star],
            "U": stacks["U"][i_star],
            "Y": stacks["Y"][i_star],
            "Z": stacks["Z"][i_star],
            "X_next": np.hstack([
                stacks["X_next"][i_star],
                stacks["X_next"][i_star][:, -1][:, None]
            ]),
            "meta": {
                **datasets[0]["meta"],
                "T": T_min,
                "N": len(datasets),
                "selected_seed": i_star,
                "selection": "medoid"
            }
        }
        return out

    @staticmethod
    def _block_hankel(D: np.ndarray, L: int) -> np.ndarray:
        """
        Build block Hankel H_L(D) from data D of shape (d, T).
        Returns shape (d*L, T-L+1).
        """
        if D.ndim != 2:
            raise ValueError("D must be 2D with shape (d, T).")
        d, T = D.shape
        if L < 1:
            raise ValueError("L must be >= 1.")
        if T < L:
            raise ValueError(f"Need T >= L. Got T={T}, L={L}.")
        N = T - L + 1
        H = np.zeros((d * L, N))
        for i in range(L):
            H[i*d:(i+1)*d, :] = D[:, i:i+N]
        return H

    # ------------------------------------------------------------------
    def _sym(self, M: np.ndarray):
        return 0.5 * (M + M.T) + self.eps * self._I(M.shape[0])
    
    def _pseudo_inv(self, M: np.ndarray):
        return M.T @ np.linalg.inv(M @ M.T + self.eps * self._I(M.shape[0]))

    def _negdef(self, matrix, name):        # (Cholesky Lifting)
        """
        Enforces matrix < -eps*I via Cholesky decomposition.
        Constraint: matrix + L*L.T + eps*I = 0
        """
        n = matrix.shape[0]
        n_L = n * (n + 1) // 2
        L_sym = ca.SX.sym(f"L_{name}", n_L)
        
        if not hasattr(self, "aux_vars"): self.aux_vars = []
        self.aux_vars.append(L_sym)
        
        # Reconstruct L
        L = ca.SX.zeros(n, n)
        idx = 0
        for i in range(n):
            for j in range(i + 1):
                L[i, j] = L_sym[idx]
                idx += 1
                
        return ca.vec(matrix + L @ L.T + self.eps * ca.DM.eye(n))

    def _posdef(self, matrix, name):        # (Cholesky Lifting)
        """
        Enforces matrix > eps*I via Cholesky decomposition.
        Constraint: matrix - L*L.T - eps*I = 0
        """
        n = matrix.shape[0]
        n_L = n * (n + 1) // 2
        L_sym = ca.SX.sym(f"L_{name}", n_L)
        
        if not hasattr(self, "aux_vars"): self.aux_vars = []
        self.aux_vars.append(L_sym)
        
        L = ca.SX.zeros(n, n)
        idx = 0
        for i in range(n):
            for j in range(i + 1):
                L[i, j] = L_sym[idx]
                idx += 1
                
        return ca.vec(matrix - (L @ L.T + self.eps * ca.DM.eye(n)))

class Initializer(Utils): 
    def __init__(self, **kwargs):
        
        super().__init__(eps=kwargs.get("eps", 1e-6))

        self.eval_from_ol = kwargs.get("eval_from_ol", True)
        self.gamma = kwargs.get("gamma", 1.0)
        self.Sigma_nom = kwargs.get("Sigma_nom", None)
        self.var = kwargs.get("var", 1.0)
        self.vals = kwargs.get("vals", None)
        self.api = kwargs.get("api", None)
        self.N_sims = kwargs.get("N_sims", 1)
        self.estm_noise = kwargs.get("estm_noise", False)
        self.Bw_mode = kwargs.get("Bw_mode", "known_cov")
        self.Bw_type = kwargs.get("Bw_type", "ident")
        self.aug_mode = kwargs.get("aug_mode", "std")
        self.augmented = kwargs.get("augmented", False)
        self.real_perf_mats = kwargs.get("real_perf_mats", True)
        self.noiseless = kwargs.get("noiseless", True)


    # ------------------------------------------------------------------
    def get_data(self):
        return (self.data["X_"],
                self.data["U_"],
                self.data["X"],
                self.data["Y_"],
                self.data["Z_"])

    def get_dims(self):
        return (self.dims["nx"],
                self.dims["nu"],
                self.dims["nw"],
                self.dims["ny"],
                self.dims["nz"])
    
    def get_mats(self):
        return (self.mats["Bw"],
                self.mats["Cz"],
                self.mats["Dzw"],
                self.mats["Dzu"])

    # ------------------------------------------------------------------
    def simulate_(self):
        if not self.eval_from_ol:
            upd, FROM_DATA, *_ = self.vals
            data = self.api.get_system(FROM_DATA=FROM_DATA,
                                       gamma=self.gamma,
                                       upd=upd)
            X_, X, U_, Y_, Z_ = data.get_data()
        else:
            op = Open_Loop(MAKE_DATA=False,
                           EVAL_FROM_PATH=False,
                           DATASETS=True,
                           N=self.N_sims)
            datasets = op.datasets
            avg = (self.select_representative_run(datasets)
                   if self.N_sims != 1 else datasets)
            X_, U_, Y_, Z_, X = (avg["X"], avg["U"],
                                 avg["Y"], avg["Z"],
                                 avg["X_next"])
        
        if self.noiseless:
            X  = savgol_filter(X,  window_length=11, polyorder=3, axis=1)
            X_ = savgol_filter(X_, window_length=11, polyorder=3, axis=1)
            Y_ = savgol_filter(Y_, window_length=11, polyorder=3, axis=1)
            U_ = savgol_filter(U_, window_length=11, polyorder=3, axis=1)
            Z_ = savgol_filter(Z_, window_length=11, polyorder=3, axis=1)

        self.data = {"X": X, "X_": X_, "U_": U_, "Y_": Y_, "Z_": Z_}
        self.dims = {
            "T": X_.shape[1],
            "nx": X_.shape[0],
            "nu": U_.shape[0],
            "ny": Y_.shape[0],
            "nz": Z_.shape[0],
        }

    # ------------------------------------------------------------------
    def estm_mats(self):
        X_, U_, X, Y_, Z_ = self.get_data()
        nx, nu = self.dims["nx"], self.dims["nu"]

        Dx = np.vstack([X_, U_])
        Ox = X @ self._pseudo_inv(Dx)
        Ax, Bu = Ox[:, :nx], Ox[:, nx:nx + nu]

        R = X - (Ax @ X_ + Bu @ U_)

        if self.Bw_type == "ident":
            *_, Bw = self.api.build_AB_from_yaml()
            nw = Bw.shape[1]
        else: 
            Bw, nw, _ = self.estm_Bw(R)
        W_ = self._pseudo_inv(Bw) @ R if not self.noiseless else np.zeros_like(R)

        if self.estm_noise:
            d = Disturbances(n=nw)
            self.Sigma_nom = d.estm_Sigma_nom(W_.T)
            self.gamma, *_ = d._estimate_gamma_with_ci(W_.T)

        Dy = np.vstack([X_, W_])
        Oy = Y_ @ self._pseudo_inv(Dy)
        Cy, Dyw = Oy[:, :nx], Oy[:, nx:nx + nw]

        if self.real_perf_mats:
            Cz, Dzw, Dzu, *_ = self.api.build_out_matrices(nw=nw)
        else:
            Dz = np.vstack([X_, U_, W_])
            Oz = Z_ @ self._pseudo_inv(Dz)
            Cz, Dzu, Dzw = (Oz[:, :nx],
                            Oz[:, nx:nx + nu],
                            Oz[:, nx + nu:nx + nu + nw])

        # niente augment complicato, lo lascio come nel tuo codice
        if self.augmented:
            N = None if self.aug_mode == "std" else (1, 1)
            Bw, Dzw, Dyw, nw, self.Sigma_nom = self.api._augment_matrices(
                B_w=Bw, D_vw=Dzw, D_yw=Dyw, var=self.var,
                Sigma_nom=self.Sigma_nom, N=N)

        """
        # bound beta su [A Bu] come in appendice B (punto 3)
        ss = np.linalg.svd(Dx, compute_uv=False)
        smin = float(ss[-1]) if ss.size else 0.0
        beta = np.linalg.norm(R, "fro") / max(smin, 1e-12)
        self.beta = float(beta)
        self.beta_A = beta * np.sqrt(nx / (nx + nu))
        self.beta_B = beta * np.sqrt(nu / (nx + nu))
        #"""

        # salviamo tutto
        self.dims["nw"] = nw
        self.data["W_"], self.data["R"] = W_, R
        self.mats = {
            "Ax": Ax,
            "Bu": Bu,
            "Bw": Bw,
            "Cy": Cy,
            "Dyw": Dyw,
            "Cz": Cz,
            "Dzu": Dzu,
            "Dzw": Dzw,
        }

    def estm_Bw(self, R: np.ndarray, eta: float = 0.95):
        nx, T = R.shape
        S = (R @ R.T) / max(T, 1)
        S = self._sym(S)

        s_vals, U = np.linalg.eigh(S)
        s_vals = np.clip(s_vals, 0.0, None)
        w = np.sqrt(s_vals)
        order = np.argsort(s_vals)[::-1]
        s = s_vals[order]
        U = U[:, order]

        if self.Bw_mode == "known_cov":
            Sigma_nom = self.Sigma_nom
            nw = Sigma_nom.shape[0]
            if nw > nx:
                raise ValueError(f"nw = {nw} > nx = {nx}")

            Up = U[:, :nw]
            sp = s[:nw]
            sp = np.clip(sp, self.eps, None)
            sp_sqrt = np.sqrt(sp)

            lam, Q = np.linalg.eigh(Sigma_nom)
            lam = np.clip(lam, self.eps, None)
            Sigma_inv_sqrt = Q @ np.diag(lam ** -0.5) @ Q.T

            Bw = Up @ np.diag(sp_sqrt) @ Sigma_inv_sqrt
            return Bw, nw, (U, s, w)

        total = max(float(np.sum(s)), 1e-18)
        cum = np.cumsum(s) / total
        nw = int(np.clip(np.searchsorted(cum, eta) + 1, 1, nx))
        Up = U[:, :nw]
        sp = s[:nw]
        sp_sqrt = np.sqrt(sp)

        if self.Bw_mode in {"factor", "white"}:
            Bw = Up @ np.diag(sp_sqrt)
        elif self.Bw_mode == "proj":
            Bw = Up
        else:
            raise ValueError("Bw_mode invalid")

        return Bw, nw, (U, s, w)

    # ------------------------------------------------------------------
    def build_Phi(self):
        nx, nu, nw, ny, nz = self.get_dims()

        Ix, Iu, Iy, Iw, I2x = self._I(nx), self._I(nu), self._I(ny), self._I(nw), self._I(2*nx)
        Oxy, Oux, O2xw, Oz2x = self._Z(nx, ny), self._Z(nu, nx), self._Z(2*nx, nw), self._I(nz, 2*nx)
        Oxu = Oux.T; Oyx = Oxy.T; Ow2x = O2xw.T
        
        self.Psi_w = ca.DM(np.block([[O2xw], [Iw]]))
        self.Psi_2x = ca.DM(np.block([[I2x], [Ow2x]]))
        self.Psi_z = ca.DM(np.block([[I2x], [Oz2x]]))

        self.psi_1 = ca.DM(np.block([[Ix, Oxy]]))
        self.psi_2 = ca.DM(np.block([[Ix], [Oux]]))
        self.psi_3 = ca.DM(np.block([[Oxu], [Iu]]))
        self.psi_4 = ca.DM(np.block([[Oyx, Iy]]))

class Constructor(Initializer): 
    def __init__(self, vals, api, noise, Bw_mode, **kwargs):
        self.eps = eps = kwargs.get("eps", 1e-6)
        super().__init__(eps=eps,
                         eval_from_ol=kwargs.get("eval_from_ol", True), gamma=noise.gamma, N_sims=kwargs.get("N_sims", 1), vals=vals, api=api,
                         estm_noise=kwargs.get("estm_noise", False), Bw_mode=Bw_mode, Bw_type=kwargs.get("Bw_type", "ident"), Sigma_nom=noise.Sigma_nom, var=noise.var,
                         aug_mode=kwargs.get("aug_mode", "std"), augmented=vals[3], real_perf_mats = kwargs.get("real_perf_mats", True))

    def build_Mc_from_XYKLmn(self, Ax, Bu, Cy, X, Y, K, Lvar, Mvar, Nvar):
        """
        Ax: (nx,nx), Bu: (nx,nu), Cy: (ny,nx)  (all casadi SX)
        X,Y: (nx,nx) symmetric decision variables (SX expressions)
        K: (nx,nx), Lvar: (nx,ny), Mvar: (nu,nx), Nvar: (nu,ny) decision variables (SX)
        Returns: Ac,Bc,Cc,Dc,Mc as SX expressions
        """

        def symmetrize(S, eps=1e-9):
            n = S.size1()
            return 0.5*(S + S.T) + eps*ca.SX.eye(n)

        def chol_lower_from_casadi(S):
            """
            CasADi's chol() convention can differ by backend.
            Many builds return an upper-triangular R s.t. S = R.T @ R.
            We convert to a lower L s.t. S = L @ L.T by setting L = R.T.
            If your chol already returns lower, then L = chol(S) is fine.
            """
            R = ca.chol(S)     # typically upper
            L = R.T            # lower candidate
            return L
        
        nx = X.size1()
        # S = I - X Y must be PD
        S = ca.SX.eye(nx) - X @ Y
        S = symmetrize(S, eps=self.eps)

        # Cholesky factorization S = L L^T
        L = chol_lower_from_casadi(S)  # (nx,nx) lower triangular (intended)

        # Inverses via linear solves (more stable than explicit inv)
        I = ca.SX.eye(nx)
        Linv = ca.solve(L, I)          # L^{-1}
        LinvT = Linv.T                 # L^{-\top}

        # Core affine pieces
        T1 = Mvar - Nvar @ Cy @ Y      # (nu,nx)
        T2 = Lvar - X @ Bu @ Nvar      # (nx,ny)

        Dc = Nvar
        Cc = T1 @ LinvT                # (nu,nx)
        Bc = Linv @ T2                 # (nx,ny)

        # Ac = L^{-1} [K - XAxY - LC_yY - XBu(M - NC_yY)] L^{-\top}
        mid = (K
            - X @ Ax @ Y
            - Lvar @ Cy @ Y
            - X @ Bu @ T1)          # (nx,nx)
        Ac = Linv @ mid @ LinvT

        Mc = ca.blockcat([[Ac, Bc],
                        [Cc, Dc]])
        return Ac, Bc, Cc, Dc, Mc, S, L

    def build_wfl_hankels(self, L: int = 1, use_z: bool = False):
        """
        Builds true block-Hankel matrices of depth L from self.data.

        Uses:
          self.data["X_"]: (nx, T)  = x_0..x_{T-1}
          self.data["U_"]: (nu, T)  = u_0..u_{T-1}
          self.data["Y_"]: (ny, T)  = y_0..y_{T-1}
          self.data["X"] : (nx, T)  = x_1..x_T  (aligned with X_)
          (optional) self.data["Z_"]: (nz, T)

        For L=1, this reduces to simple stacking.
        For L>1, it builds:
          Hx  = H_L(X_)             shape (nx*L, N)
          Hu  = H_L(U_)             shape (nu*L, N)
          Hy  = H_L(Y_)             shape (ny*L, N)
          Hx+ = H_L(X)              shape (nx*L, N)
        where N = T - L + 1.

        Stores to self.wfl:
          self.wfl["L"], ["Hx"], ["Hu"], ["Hy"], ["Hx_plus"], and stacked Xp, Xf.

        Returns:
          Xp = [Hx; Hu]             shape ((nx+nu)*L, N)
          Xf = [Hx_plus; Hy]        shape ((nx+ny)*L, N)
          (optional) Hz = H_L(Z_)   shape (nz*L, N)
        """
        X_ = self.data["X_"]
        U_ = self.data["U_"]
        Y_ = self.data["Y_"]
        Xp1 = self.data["X"]     # x^+ aligned (x_1..x_T)

        # Basic checks
        if not (X_.ndim == U_.ndim == Y_.ndim == Xp1.ndim == 2):
            raise ValueError("X_,U_,Y_,X must be 2D arrays (dim, T).")
        nx, T = X_.shape
        _, Tu = U_.shape
        _, Ty = Y_.shape
        nx2, Tx = Xp1.shape
        if not (Tu == Ty == Tx == T):
            T = min(T, Tu, Ty, Tx)
            X_, U_, Y_, Xp1 = X_[:, :T], U_[:, :T], Y_[:, :T], Xp1[:, :T]
        if nx2 != nx:
            raise ValueError("X has inconsistent state dimension vs X_.")

        # True block Hankels
        Hx      = self._block_hankel(X_,  L)
        Hu      = self._block_hankel(U_,  L)
        Hy      = self._block_hankel(Y_,  L)
        Hx_plus = self._block_hankel(Xp1, L)

        Xp = np.vstack([Hx, Hu])
        Xf = np.vstack([Hx_plus, Hy])

        self.wfl = getattr(self, "wfl", {})
        self.wfl.update({
            "L": L,
            "Hx": Hx,
            "Hu": Hu,
            "Hy": Hy,
            "Hx_plus": Hx_plus,
            "Xp": Xp,
            "Xf": Xf,
            "T": T,
            "N": Xp.shape[1],
        })

        if use_z:
            Z_ = self.data.get("Z_")
            if Z_ is None:
                raise ValueError("use_z=True but self.data['Z_'] is missing.")
            Hz = self._block_hankel(Z_, L)
            self.wfl["Hz"] = Hz
            return Xp, Xf, Hz

        return Xp, Xf

# =============================================================================================== #

class WFL_nonConvex(Constructor):
    
    def __init__(self, vals, model, api, noise, **kwargs):
        
        Bw_mode = "proj" if kwargs.get("estm_noise", False) else kwargs.get("Bw_mode", "known_cov")

        self.api = api
        self.eps, self.rho = kwargs.get("eps", 1e-6), kwargs.get("rho", 1e-2)
        self.L = kwargs.get("L", 1)
        self.model = model
        self.inp = vals[4]
        self.reg_fro = kwargs.get("reg_fro", False)

        super().__init__(vals, api, noise, Bw_mode, **kwargs)

    
    # ------------------------------------------------------------------
    # PUBLIC ENTRY
    # ------------------------------------------------------------------
    def run(self):
        # 1. Prepare Data & Hankel Matrices
        self.simulate_()
        self.estm_mats()
        self.build_wfl_hankels(L=self.L, use_z=False)
        
        # 2. Build Optimization Problem
        self.build_Phi() # Structural matrices (psi_1, etc.)
        self.build_var() # Variables (now includes Mp, g, w)
        self.build_con() # Constraints (LMI + WFL consistency)
        self.build_obj() # Objective
        self.build_reg() # Regularization (if any)
        
        # 3. Solve
        self.solve_prb()
        self.pack_outs()
        
        return (self.outs,
                self.get_plant(),
                self.Sigma_nom,
                self.others)
                #(self.violations, self.total_constraints))

    # ------------------------------------------------------------------
    def get_plant(self):
        Bw, Cz, Dzw, Dzu = self.get_mats()
        Ax = self.others["Ax_opt"]
        Bu = self.others["Bu_opt"]
        Cy = self.others["Cy_opt"]
        Dyw = self.others["Dyw"]
        return (Ax, Bw, Bu, Cy, Dyw, Cz, Dzw, Dzu)

    def get_vars(self, which="main"):
        if which == "main":
            return (self.vars["lam"], self.vars["Q"], self.vars["P"])
        elif which == "inner":
            return (self.vars["K"], self.vars["L"],
                    self.vars["Y"], self.vars["X"],
                    self.vars["M"], self.vars["N"])
        elif which == "mats":
            return (self.vars["A_true"], self.vars["B"],
                    self.vars["C"], self.vars["D"])
        elif which == "t":
            return (self.vars.get("tK", 0.0),
                    self.vars.get("tL", 0.0),
                    self.vars.get("tY", 0.0),
                    self.vars.get("tX", 0.0),
                    self.vars.get("tM", 0.0),
                    self.vars.get("tN", 0.0))

    # ------------------------------------------------------------------
    def build_var(self):
        nx, nu, nw, ny, nz = self.get_dims()
        Bw_init, Cz_init, _, Dzu_init = self.get_mats()
        
        # Initial guesses from LS for Mp priors
        Ax_init = self.mats["Ax"]
        Bu_init = self.mats["Bu"]
        Cy_init = self.mats["Cy"]

        cas = ca.SX 

        # --- Decision Variables ---
        def symm_var(name, n): return cas.sym(name, n * (n + 1) // 2)
        def full_sym(v, n):
            M = cas.zeros(n, n)
            idx = 0
            for i in range(n):
                for j in range(i, n):
                    M[i, j] = v[idx]; M[j, i] = v[idx]; idx += 1
            return M

        vX, vY, vQ = symm_var("X", nx), symm_var("Y", nx), symm_var("Q", nw)
        K, L = cas.sym("K", nx, nx), cas.sym("L", nx, ny)
        M_var, N_var = cas.sym("M", nu, nx), cas.sym("N", nu, ny)

        X, Y, Q = full_sym(vX, nx), full_sym(vY, nx), full_sym(vQ, nw)
        lam = cas.sym("lam", 1)

        # WFL Variables
        Mp = cas.sym("Mp", nx + ny, nx + nu)
        N_samples = self.wfl["N"]
        g_wfl = cas.sym("g_wfl", N_samples)
        w_wfl = cas.sym("w_wfl", nx + ny, N_samples)

        # --- 1. Extract Variable Plant Matrices from Mp ---
        # Ax = psi_1 * Mp * psi_2, etc. (Linear slicing)
        Ax = self.psi_1 @ Mp @ self.psi_2
        Bu = self.psi_1 @ Mp @ self.psi_3
        Cy = self.psi_4 @ Mp @ self.psi_2
        
        # Constants / Fixed matrices
        Bw = ca.DM(Bw_init)
        Cz = ca.DM(Cz_init)
        Dzu = ca.DM(Dzu_init)
        Dzw = ca.DM(self._Z(nz, nw))
        Dyw = ca.DM(self._Z(ny, nw)) 

        # --- 2. Construct CoV Closed-Loop Matrices (NO INVERSES) ---
        # A_cov = [ A Y + Bu M,       A + Bu N Cy ]
        #         [ K,                X A + L Cy  ]
        r1 = ca.horzcat( Ax @ Y + Bu @ M_var,    Ax + Bu @ N_var @ Cy )
        r2 = ca.horzcat( K,                      X @ Ax + L @ Cy      )
        A_cov = ca.vertcat(r1, r2) 

        # B_cov = [ Bw + Bu N Dyw ]  
        #         [ X Bw + L Dyw  ]
        b1 = Bw + Bu @ N_var @ Dyw
        b2 = X @ Bw + L @ Dyw
        B_cov = ca.vertcat(b1, b2) 

        # C_cov = [ Cz Y + Dzu M,     Cz + Dzu N Cy ]
        c1 = Cz @ Y + Dzu @ M_var
        c2 = Cz + Dzu @ N_var @ Cy
        C_cov = ca.horzcat(c1, c2) 

        # D_cov = Dzw + Dzu N Dyw
        D_cov = Dzw + Dzu @ N_var @ Dyw 

        # Combine into the single matrix PMcl used in LMIs
        PMcl = ca.blockcat([
            [A_cov, B_cov],
            [C_cov, D_cov]
        ])

        # Lyapunov Matrix P
        Ix = ca.DM.eye(nx)
        Iz = ca.DM.eye(nz)
        P = ca.blockcat([[Y, Ix], [Ix, X]])
        
        # P_aug for the LMI blocks 
        P_aug = ca.blockcat([
            [P,                         ca.DM(self._Z(2*nx, nz))],
            [ca.DM(self._Z(nz, 2*nx)),  Iz]
        ])

        self.cas_vars = {
            "vX": vX, "vY": vY, "vQ": vQ, "X": X, "Y": Y, "Q": Q, "lam": lam,
            "K": K, "L": L, "M": M_var, "N": N_var,
            "Mp": Mp, "g_wfl": g_wfl, "w_wfl": w_wfl,
            "PMcl": PMcl, "P_aug": P_aug,
            "Ax_init": Ax_init, "Bu_init": Bu_init, "Cy_init": Cy_init
        }

    # ------------------------------------------------------------------
    def build_con(self):
        nx, nu, nw, ny, nz = self.get_dims()
        cv = self.cas_vars
        
        P_aug = cv["P_aug"]
        PMcl  = cv["PMcl"] # Using CoV version (Safe)
        Q     = cv["Q"]
        lam   = cv["lam"]
        Mp    = cv["Mp"]

        # Constants
        eps = self.eps
        I_w = ca.DM(np.eye(nw))
        Z_wxz = ca.DM(np.zeros((nw, 2*nx + nz)))

        g_list = []

        # --- 1. Trust Region Constraint ---
        Mp_prior = ca.DM(np.block([
            [cv["Ax_init"], cv["Bu_init"]],
            [cv["Cy_init"], np.zeros((ny, nu))]
        ]))
        # ||Mp - Prior|| - delta <= 0
        delta = 0.05 * ca.norm_fro(Mp_prior)
        g_trust = ca.norm_fro(Mp - Mp_prior) - delta
        g_list.append(g_trust) 

        # --- 2. Willems' Lemma (Sliced L=1) ---
        Xp_full = ca.DM(self.wfl["Xp"])
        Xf_full = ca.DM(self.wfl["Xf"])
        
        # Slice to L=1 to match Mp dimensions
        n_in, n_out = nx + nu, nx + ny
        Xp_1step = Xp_full[0:n_in, :]
        Xf_1step = Xf_full[0:n_out, :]
        
        g_vec = cv["g_wfl"]
        
        # Residual: Xf*g - Mp*Xp*g
        # We enforce this equals the noise variable 'w' in the objective implicitly,
        # OR we can enforce it explicitly here. 
        # Standard: Xf*g - Mp*Xp*g - w = 0
        resid = ca.vec((Xf_1step @ g_vec) - (Mp @ Xp_1step @ g_vec))
        
        g_list.append(resid) # == 0


        # --- 3. LMI Constraints (Lifted) ---
        
        if self.model == "independent":
            # Xi1
            Xi1 = ca.blockcat([
                [-self.Psi_z.T @ P_aug @ self.Psi_z,    (PMcl @ self.Psi_2x).T],
                [ PMcl @ self.Psi_2x,                   -P_aug]
            ])
            g_list.append(self._negdef(Xi1, "Xi1"))

            # Xi2
            Xi2 = ca.blockcat([
                [-lam * I_w,                lam * I_w,          (PMcl @ self.Psi_w).T],
                [ lam * I_w,                -Q - lam * I_w,     Z_wxz],
                [ PMcl @ self.Psi_w,        Z_wxz.T,            -P_aug]
            ])
            g_list.append(self._negdef(Xi2, "Xi2"))
            
        elif self.model == "correlated":
             # Use the correlated kernel structure here if needed
             pass

        # Positive Definite constraints
        g_list.append(self._posdef(P_aug, "P_aug"))
        g_list.append(self._posdef(Q, "Q"))
        
        # Lambda >= 0  => -lam <= 0
        g_list.append(-lam) 

        self.cas_con = {"g": ca.vertcat(*g_list), "g_list": g_list}

    # ------------------------------------------------------------------
    def build_obj(self):
        """
        Build the scalar objective in CasADi:
            f = trace(Q Sigma_nom) + lam * gamma^2
        """
        cv = self.cas_vars
        Q = cv["Q"]
        lam = cv["lam"]
        Mp = cv["Mp"]
        
        # 1. Control Performance Cost
        Sigma_nom = ca.DM(self.Sigma_nom)
        J_ctrl = ca.trace(Q @ Sigma_nom) + lam * (self.gamma ** 2)
        
        # 2. Model Selection Regularization (Eq 2.18 / 362)
        J_model = ca.norm_fro(Mp)**2 
        
        # Total Objective
        self.cas_obj = J_ctrl #+ 1e-2 * J_model

    # ------------------------------------------------------------------
    def build_reg(self):
        """
        Optional regularizer (non-structural, just to tame the search).
        """
            
        if self.reg_fro:
            cv = self.cas_vars
            K = cv["K"]; L = cv["L"]; M = cv["M"]; N = cv["N"]
            X = cv["X"]; Y = cv["Y"]

            reg = (ca.sumsqr(K) + ca.sumsqr(L) +
                ca.sumsqr(M) + ca.sumsqr(N) +
                ca.sumsqr(X) + ca.sumsqr(Y))

            self.cas_reg = self.rho * reg
        else: 
            self.cas_reg = 0.0

    # ------------------------------------------------------------------
    def solve_prb(self):
        cv = self.cas_vars
        nx, nu, nw, ny, nz = self.get_dims()

        # 1. Vectorize Variables
        n_vX = nx * (nx + 1) // 2
        n_vY = nx * (nx + 1) // 2
        n_vQ = nw * (nw + 1) // 2
        n_lam = 1
        n_K, n_L = nx * nx, nx * ny
        n_M, n_N = nu * nx, nu * ny
        n_Mp = (nx + ny) * (nx + nu)
        n_g = self.wfl["N"]
        n_w = (nx + ny) * self.wfl["N"]

        def vec(mat): return np.array(mat).flatten('F')
        def vecsym(mat):
            vals = []
            m = np.triu(mat)
            for i in range(len(mat)):
                for j in range(i, len(mat)):
                    vals.append(m[i,j])
            return np.array(vals)

        # 2. Initial Guesses
        # FIX: Init X, Y small so I - XY is Positive Definite
        x0_X = vecsym(np.eye(nx) * 0.1) 
        x0_Y = vecsym(np.eye(nx) * 0.1)
        x0_Q = vecsym(np.eye(nw))
        x0_lam = np.array([1.0]) 
        
        x0_K = vec(np.eye(nx))
        x0_L = vec(np.zeros((nx, ny)))
        x0_M = vec(np.zeros((nu, nx)))
        x0_N = vec(np.zeros((nu, ny)))

        Mp_est = np.block([
            [cv["Ax_init"], cv["Bu_init"]],
            [cv["Cy_init"], np.zeros((ny, nu))]
        ])
        x0_Mp = vec(Mp_est) + 1e-6 * np.random.randn(len(vec(Mp_est)))
        x0_g = np.zeros(n_g)
        x0_w = np.zeros(n_w)

        x0_main = np.concatenate([
            x0_X, x0_Y, x0_Q, x0_lam, 
            x0_K, x0_L, x0_M, x0_N, 
            x0_Mp, x0_g, x0_w
        ])

        # 3. Aux Variables Init (L matrices)
        x0_aux_list = []
        if hasattr(self, "aux_vars"):
            for aux in self.aux_vars:
                N_len = aux.shape[0]
                n = int((-1 + np.sqrt(1 + 8 * N_len)) / 2)
                # Identity guess is safe for Cholesky factors
                L_init = np.eye(n)
                vals = []
                for i in range(n):
                    for j in range(i + 1):
                        vals.append(L_init[i, j])
                x0_aux_list.append(np.array(vals))

        x0_aux = np.concatenate(x0_aux_list) if x0_aux_list else np.array([])
        x0 = np.concatenate([x0_main, x0_aux])

        # 4. Build NLP
        vars_main = ca.vertcat(
            cv["vX"], cv["vY"], cv["vQ"], cv["lam"],
            ca.vec(cv["K"]), ca.vec(cv["L"]), ca.vec(cv["M"]), ca.vec(cv["N"]),
            ca.vec(cv["Mp"]), cv["g_wfl"], ca.vec(cv["w_wfl"])
        )
        
        # Include aux vars in optimization vector
        if hasattr(self, "aux_vars"):
            vars_cas = ca.vertcat(vars_main, *self.aux_vars)
        else:
            vars_cas = vars_main
        
        nlp = {'x': vars_cas, 'f': self.cas_obj, 'g': self.cas_con["g"]}

        opts = {
            "ipopt.max_iter": 5000,
            "ipopt.print_level": 5, 
            "ipopt.tol": 1e-4,
            "ipopt.acceptable_tol": 1e-3,
            "ipopt.mu_init": 0.1, 
        }
        
        self.solver_obj = ca.nlpsol("solver", "ipopt", nlp, opts)
        print("Solver initialized successfully.")

        # 5. Solve & Bounds
        lbg = np.zeros(self.cas_con["g"].shape[0])
        ubg = np.zeros(self.cas_con["g"].shape[0])
        
        # Relax specific inequalities
        # Index 0 is Trust Region (Inequality)
        lbg[0] = -np.inf  
        
        # Last index is Lambda (Inequality -lam <= 0)
        lbg[-1] = -np.inf 

        # Note: WFL (Index 1) and Lifted LMIs are Equalities, so bounds remain 0.

        try:
            res = self.solver_obj(x0=x0, lbg=lbg, ubg=ubg)
        except RuntimeError as e:
            print(f"Solver Failure: {e}")
            self.success = False
            return

        self.status = self.solver_obj.stats()["return_status"]
        self.success = self.status == "Solve_Succeeded"
        self.obj_val = float(res["f"])
        self.sol_x = res["x"].full().flatten()

        # 6. Unpack
        idx = 0
        def eat(n): nonlocal idx; out = self.sol_x[idx:idx+n]; idx += n; return out
        
        self.res_vals = {}
        self.res_vals["vX"] = eat(n_vX)
        self.res_vals["vY"] = eat(n_vY)
        self.res_vals["vQ"] = eat(n_vQ)
        self.res_vals["lam"] = eat(n_lam)
        self.res_vals["K"] = eat(n_K).reshape((nx, nx), order='F')
        self.res_vals["L"] = eat(n_L).reshape((nx, ny), order='F')
        self.res_vals["M"] = eat(n_M).reshape((nu, nx), order='F')
        self.res_vals["N"] = eat(n_N).reshape((nu, ny), order='F')
        self.res_vals["Mp"] = eat(n_Mp).reshape((nx + ny, nx + nu), order='F')
        self.res_vals["g_wfl"] = eat(n_g)
        self.res_vals["w_wfl"] = eat(n_w).reshape((nx + ny, self.wfl["N"]), order='F')

    # ------------------------------------------------------------------
    def pack_outs(self):
        # 1. Reconstruct Symmetric Matrices from Vectors
        def reconstruct_sym(v, n):
            M = np.zeros((n, n))
            idx = 0
            for i in range(n):
                for j in range(i, n):
                    M[i, j] = v[idx]
                    M[j, i] = v[idx]
                    idx += 1
            return M

        nx, nu, nw, ny, _ = self.get_dims()

        if not hasattr(self, "res_vals"):
            self.outs = None
            return

        rv = self.res_vals   

        # Primary Variables
        lam_val = float(rv["lam"][0])
        Q_val = reconstruct_sym(rv["vQ"], nw)
        X_val = reconstruct_sym(rv["vX"], nx)
        Y_val = reconstruct_sym(rv["vY"], nx)
        
        K_val, L_val = rv["K"], rv["L"]
        M_val, N_val = rv["M"], rv["N"]
        
        # Optimized Plant (Mp)
        Mp_opt = rv["Mp"]

        Bw, Cz, Dzw, Dzu = self.get_mats()
        Dyw = np.zeros((ny, nw))  # Assuming no direct feedthrough for disturbance to output
        
        # 2. Extract Optimized System Matrices from Mp
        # Recall: psi_1=[I 0], psi_2=[I; 0], etc. (Simple slicing)
        # Mp = [Ax  Bu]
        #      [Cy  * ]
        Ax_opt = Mp_opt[0:nx, 0:nx]
        Bu_opt = Mp_opt[0:nx, nx:nx+nu]
        Cy_opt = Mp_opt[nx:nx+ny, 0:nx]
        # Note: We discard the bottom-right block of Mp as it is usually 0 or irrelevant
        

        A_bar = np.block([
            [Ax_opt @ Y_val + Bu_opt @ M_val,   Ax_opt + Bu_opt @ N_val @ Cy_opt ], 
            [K_val,                             X_val @ Ax_opt + L_val @ Cy_opt]
        ])
        B_bar = np.block([
            [Bw], # + Bu_opt @ N_val @ Dyw], 
            [X_val @ Bw] # + L_val @ Dyw]
        ])
        C_bar = np.block([ 
            [Cz @ Y_val + Dzu @ M_val,      Cz + Dzu @ N_val @ Cy_opt]
        ])
        D_bar = Dzw # + Dzu @ N_val @ Dyw

        # 5. Pack into DROLMIResult
        dro = DROLMIResult(
            solver="ipopt", # Hardcoded as we used CasADi/Ipopt
            status=self.status,
            obj_value=self.obj_val,
            gamma=self.gamma,
            lambda_opt=lam_val,
            Q=Q_val, X=X_val, Y=Y_val,
            K=K_val, L=L_val, M=M_val, N=N_val,
            # We pass the reconstructed CLOSED LOOP matrices here
            Abar=A_bar, 
            # Placeholders for B/C/D bar if not strictly required by your downstream analysis
            Bbar=B_bar, 
            Cbar=C_bar, 
            Dbar=D_bar,
            Pbar=np.block([[Y_val, self._I(nx)], [self._I(nx), X_val]]),
            Tp=None, P=None,
        )
        self.outs = dro

        # 6. Pack "Others" (WFL specifics)
        # Instead of DeltaA/DeltaB, we provide the Optimized Plant and WFL residuals
        self.others = {
            "Mp_opt": Mp_opt,       # The Plant (Ax, Bu, Cy) consistent with data
            "g_wfl": rv["g_wfl"],   # The behavioral coefficients
            "w_wfl": rv["w_wfl"],   # The noise realization
            "Ax_opt": Ax_opt,       # Convenient access
            "Bu_opt": Bu_opt, 
            "Cy_opt": Cy_opt,
            "Dyw": Dyw,
        }

# =============================================================================================== #

if __name__ == "__main__":

    # ------------------------------------------------------------
    # 1) Instantiate required objects (placeholders explained)
    # ------------------------------------------------------------

    # vals is whatever your pipeline already uses
    # e.g. vals = (upd, FROM_DATA, ..., augmented, inp)
    vals = (
        True,       # upd (example)
        True,       # FROM_DATA
        None,       # whatever else your code expects
        False,      # augmented
        False       # inp
    )
    model = "independent"  # or "correlated"


    # api must be your MatricesAPI / system interface
    api = MatricesAPI()

    # noise must be an instance with attributes gamma, var, Sigma_nom
    noise = Noise(
        n=2,
        avrg=0,
        gamma=0.5,
        var=1.0,
        Sigma_nom=np.eye(2)   # adapt dimension if needed
    )

    # ------------------------------------------------------------
    # 2) Create WFL object
    # ------------------------------------------------------------

    wfl = WFL_nonConvex(
        vals=vals,
        api=api,
        model=model,
        noise=noise,
        N_sims=1,
        eval_from_ol=True   # important: populate self.data via simulate_()
    )

    wfl.simulate_()
    wfl.estm_mats() # Initial guess for Mp comes from here
    wfl.build_wfl_hankels(L=wfl.L, use_z=False) # Build Hx, Hu, Hy matrices

    # 2. Build Optimization Problem
    wfl.build_Phi() # Structural matrices (psi_1, etc.)
    wfl.build_var() # Variables (now includes Mp, g, w)
    wfl.build_reg() # Regularization (if any)
    wfl.build_con() # Constraints (LMI + WFL consistency)
    wfl.build_obj() # Objective
    wfl.solve_prb() # Solve with IPOPT

    """
    # ------------------------------------------------------------
    # 3) Run the data generation
    # ------------------------------------------------------------

    wfl.simulate_()   # <-- THIS is mandatory before Hankels

    # Sanity check
    print("Available data keys:", wfl.data.keys())
    # must include: X_, U_, X, Y_

    # ------------------------------------------------------------
    # 4) Build TRUE block Hankels (depth L)
    # ------------------------------------------------------------

    L = 10
    Xp, Xf = wfl.build_wfl_hankels(L=L)

    print("Xp shape:", Xp.shape)
    print("Xf shape:", Xf.shape)

    # ------------------------------------------------------------
    # 5) Define the WFL set as constraints
    # ------------------------------------------------------------

    Mp, W, cons = wfl.define_M_wfl_cvxpy(
        L=L,
        w_set="l2_ball",
        eps_w=1e-2
    )

    print(f"Defined M_WFL with {len(cons)} constraints")

    # ------------------------------------------------------------
    # 6) Select one representative Mp ∈ M_WFL
    # ------------------------------------------------------------

    sol = wfl.select_one_Mp_in_Mwfl(
        L=L,
        w_set="l2_ball",
        eps_w=1e-2,
        objective="min_fro",
        solver="MOSEK",    # or "SCS" if MOSEK not available
        verbose=True
    )

    # ------------------------------------------------------------
    # 7) Extract result
    # ------------------------------------------------------------

    if sol["status"] not in ("optimal", "optimal_inaccurate"):
        raise RuntimeError(f"WFL selection failed: {sol['status']}")

    Mp_hat = sol["Mp"]

    print("Mp_hat shape:", Mp_hat.shape)
    #"""


    """
    def wfl_rank_info(self, rcond: float = 1e-10):
        # Simple rank + conditioning report for Xp (useful to sanity-check data richness).
        # Requires build_wfl_hankels() called first, or will build it.
        if not hasattr(self, "wfl") or "Xp" not in self.wfl:
            self.build_wfl_hankels(use_z=False)

        Xp = self.wfl["Xp"]
        U, s, Vt = np.linalg.svd(Xp, full_matrices=False)
        rank = int(np.sum(s > rcond * s[0])) if s.size else 0
        smin = float(s[-1]) if s.size else 0.0
        smax = float(s[0]) if s.size else 0.0
        return {"rank": rank, "smax": smax, "smin": smin, "cond_est": (smax / max(smin, 1e-30))}

    def define_M_wfl_cvxpy(self,
                           L: int = 1,
                           w_set: str = "l2_ball",
                           eps_w: float = 1e-2,
                           Q: np.ndarray = None,
                           per_column: bool = True):
        # Builds WFL set constraints using block Hankels of depth L:
        #     Xf = Mp Xp + W, with columns w_i in W-set.
        # 
        # Returns CVXPY variables Mp, W and constraint list.
        # Stores them under self.wfl["cvxpy"].
        # Ensure Hankels exist
        if not hasattr(self, "wfl") or self.wfl.get("L", None) != L:
            self.build_wfl_hankels(L=L, use_z=False)

        Xp = self.wfl["Xp"]
        Xf = self.wfl["Xf"]
        n_in, N = Xp.shape
        n_out, N2 = Xf.shape
        if N2 != N:
            raise ValueError("Xp and Xf must have same number of columns.")

        Mp = cp.Variable((n_out, n_in))
        W  = cp.Variable((n_out, N))
        constraints = [Xf == Mp @ Xp + W]

        if w_set == "l2_ball":
            if per_column:
                for i in range(N):
                    constraints.append(cp.norm(W[:, i], 2) <= eps_w)
            else:
                constraints.append(cp.norm(W, "fro") <= eps_w)

        elif w_set == "fro_ball":
            constraints.append(cp.norm(W, "fro") <= eps_w)

        elif w_set == "ellipsoid":
            if Q is None:
                raise ValueError("w_set='ellipsoid' requires Q with shape (n_out, n_out).")
            Q = np.asarray(Q)
            if Q.shape != (n_out, n_out):
                raise ValueError(f"Q must be {(n_out, n_out)}, got {Q.shape}.")
            Qinv = np.linalg.inv(Q)
            if not per_column:
                raise ValueError("Ellipsoid should be imposed per-column. Set per_column=True.")
            for i in range(N):
                constraints.append(cp.quad_form(W[:, i], Qinv) <= 1.0)
        else:
            raise ValueError(f"Unknown w_set='{w_set}'")

        self.wfl.setdefault("cvxpy", {})
        self.wfl["cvxpy"].update({
            "L": L, "Mp": Mp, "W": W, "constraints": constraints,
            "w_set": w_set, "eps_w": eps_w
        })
        return Mp, W, constraints

    def select_one_Mp_in_Mwfl(self,
                              L: int = 1,
                              w_set: str = "l2_ball",
                              eps_w: float = 1e-2,
                              Q: np.ndarray = None,
                              objective: str = "min_fro",
                              Mp_prior: np.ndarray = None,
                              lam_prior: float = 1e-2,
                              solver: str = None,
                              verbose: bool = False):
        # Solve a convex program to pick one Mp from the WFL set.
        Mp, W, cons = self.define_M_wfl_cvxpy(L=L, w_set=w_set, eps_w=eps_w, Q=Q, per_column=True)

        if objective == "min_fro":
            obj = cp.Minimize(cp.norm(Mp, "fro"))
        elif objective == "prior":
            if Mp_prior is None:
                raise ValueError("objective='prior' requires Mp_prior.")
            Mp_prior = np.asarray(Mp_prior)
            if Mp_prior.shape != Mp.shape:
                raise ValueError(f"Mp_prior shape {Mp_prior.shape} must match Mp shape {Mp.shape}.")
            obj = cp.Minimize(cp.sum_squares(Mp - Mp_prior) + lam_prior * cp.sum_squares(W))
        else:
            raise ValueError(f"Unknown objective='{objective}'")

        prob = cp.Problem(obj, cons)
        prob.solve(solver=solver, verbose=verbose)

        sol = {
            "status": prob.status,
            "objective_value": prob.value,
            "Mp": None if Mp.value is None else Mp.value,
            "W": None if W.value is None else W.value,
        }
        self.wfl["cvxpy"]["solution"] = sol
        return sol
    #"""

