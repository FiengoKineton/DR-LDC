import sys
import numpy as np, casadi as ca
from scipy.signal import savgol_filter

from disturbances import Disturbances
from simulate import Open_Loop


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
