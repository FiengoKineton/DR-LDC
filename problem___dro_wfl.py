import json, sys, time, psutil, os, numpy as np, cvxpy as cp, casadi as ca
import matplotlib.pyplot as plt

from pathlib import Path
from typing import Dict, Any, List
from utils___systems import Plant, Plant_cl, Controller, DROLMIResult, Noise
from utils___matrices import MatricesAPI, recover_deltas, compose_closed_loop, Recover
from utils___ambiguity import Disturbances
from utils___simulate import Open_Loop, Closed_Loop
from utils___Nsims_mats import NsimsMatricesAnalyzer, mean_dict, select_representative_run, plot_first3_and_mean



class WFL:
    def __init__(self, *args, **kwargs):
        return WFL_nonConvex(*args, **kwargs)


# =============================================================================================== #

class WFL_nonConvex:
    """
    Direct data-driven DRO-LMI *senza* la Young–Schur relaxation.

    - Identificazione A_hat, Bu_hat, Bw_hat, Cy_hat, Dyw_hat, Cz_hat, Dzu_hat, Dzw_hat
      via i tuoi metodi.
    - Introduce esplicitamente gli errori DeltaA, DeltaB con bound Frobenius
      (beta_A, beta_B) e costruisce il blocco E come in eq. (B.1).
    - Usa A_true = A_nom + E nel kernel Xi(A_true, B, C, D) del paper.
    - Nessun P_bar, Ps, G, H, S, sigma_s, sigma_p.
    """

    # ------------------------------------------------------------------
    # INIT & helpers
    # ------------------------------------------------------------------
    def __init__(self,
                 vals: tuple, model: str,
                 api, noise,
                 rho: float = 1e-2, eps: float = 1e-6, 
                 N_sims: int = 1, L: int = 10,
                 Bw_mode: str = "known_cov", Bw_type: str = "ident",
                 real_Z_mats: bool = True,
                 aug_mode: str = "std",
                 eval_from_ol: bool = True,
                 estm_noise: bool = False,
                 reg_fro: bool = False):
        
        Bw_mode = "proj" #if estm_noise else Bw_mode

        self.api = api
        self.eps, self.rho = eps, rho
        self.N_sims, self.L = N_sims, L
        self.model, self.Bw_mode, self.vals, self.Bw_type = model, Bw_mode, vals, Bw_type
        self.real_perf_mats = real_Z_mats
        self.augmented = vals[3]
        self.aug_mode = aug_mode
        self.eval_from_ol = eval_from_ol
        self.estm_noise = estm_noise

        self.gamma = noise.gamma
        self.var = noise.var
        self.Sigma_nom = noise.Sigma_nom

        self.inp = vals[4]
        self.reg_fro = reg_fro

    # basic helpers ----------------------------------------------------
    def _I(self, n, m=None):
        m = n if m is None else m
        return np.eye(n, m)

    def _Z(self, n, m=None):
        m = n if m is None else m
        return np.zeros((n, m))

    def _pseudo_inv(self, M: np.ndarray):
        return M.T @ np.linalg.inv(M @ M.T + self.eps * self._I(M.shape[0]))

    def _sym(self, M: np.ndarray):
        return 0.5 * (M + M.T) + self.eps * self._I(M.shape[0])

    def _negdef(self, M: cp.Expression, strict: bool = True):
        if strict:
            return M << -self.eps * self._I(M.shape[0], M.shape[1])
        return M << 0

    def _posdef(self, M: cp.Expression, strict: bool = True):
        if strict:
            return M >> self.eps * self._I(M.shape[0], M.shape[1])
        return M >> 0

    def _val(self, m):
        if m is None:
            return None
        return float(m) if np.isscalar(m) else m

    # ------------------------------------------------------------------
    # PUBLIC ENTRY
    # ------------------------------------------------------------------
    def run(self):
        self.simulate_()
        self.estm_mats()
        self.build_Phi()
        self.build_var()
        self.build_con()
        self.build_obj()
        self.build_reg()
        self.solve_prb()
        self.pack_outs()
        return (self.outs,
                self.get_mats(),
                self.Sigma_nom,
                self.others,
                (self.violations, self.total_constraints))

    # ------------------------------------------------------------------
    # DATA ACCESSORS
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
            return (self.vars.get("tK"),
                    self.vars.get("tL"),
                    self.vars.get("tY"),
                    self.vars.get("tX"),
                    self.vars.get("tM"),
                    self.vars.get("tN"))

    # ------------------------------------------------------------------
    # OPEN-LOOP SIMULATION + MEDOID (identico al tuo, accorciato)
    # ------------------------------------------------------------------
    def select_representative_run(self, datasets, keys=("X", "U", "Y", "Z", "X_next"),
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

        self.data = {"X": X, "X_": X_, "U_": U_, "Y_": Y_, "Z_": Z_}
        self.dims = {
            "T": X_.shape[1],
            "nx": X_.shape[0],
            "nu": U_.shape[0],
            "ny": Y_.shape[0],
            "nz": Z_.shape[0],
        }

    # ------------------------------------------------------------------
    # IDENTIFICAZIONE + B_w (uguale al tuo, ma calcolo betaA/B/E dal testo)
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
            Bw, nw, self._residual_anisotropy_weights = self.estm_Bw(R)
        W_ = self._pseudo_inv(Bw) @ R

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

        # bound beta su [A Bu] come in appendice B (punto 3)
        ss = np.linalg.svd(Dx, compute_uv=False)
        smin = float(ss[-1]) if ss.size else 0.0
        beta = np.linalg.norm(R, "fro") / max(smin, 1e-12)
        self.beta = float(beta)
        self.beta_A = beta * np.sqrt(nx / (nx + nu))
        self.beta_B = beta * np.sqrt(nu / (nx + nu))

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
    # VARIABILI
    # ------------------------------------------------------------------
    def build_var(self):
        nx, nu, nw, ny, nz = self.get_dims()
        Bw, Cz, _, Dzu = self.get_mats()
        Bw, Cz, Dzu = ca.DM(Bw), ca.DM(Cz), ca.DM(Dzu)

        cas = ca.SX  # <--- use SX, not MX

        # symmetric matrices packed as vectors
        def symm_var(name, n):
            return cas.sym(name, n * (n + 1) // 2)

        def full_sym(v, n):
            M = cas.zeros(n, n)
            idx = 0
            for i in range(n):
                for j in range(i, n):
                    M[i, j] = v[idx]
                    M[j, i] = v[idx]
                    idx += 1
            return M        


        # Decision variables
        vX = symm_var("X", nx)
        vY = symm_var("Y", nx)
        vQ = symm_var("Q", nw)

        K  = cas.sym("K", nx, nx)
        L  = cas.sym("L", nx, ny)
        M  = cas.sym("M", nu, nx)
        N  = cas.sym("N", nu, ny)


        X = full_sym(vX, nx)
        Y = full_sym(vY, nx)
        Q = full_sym(vQ, nw)
        lam = cas.sym("lam", 1)

        Ix  = ca.DM(np.eye(nx))
        Iw  = ca.DM(np.eye(nw))
        Iz  = ca.DM(np.eye(nz))

        P = ca.blockcat([[Y,  Ix],
                        [Ix,  X ]])
        P_aug = ca.blockcat([
            [P,                         ca.DM(self._Z(2*nx, nz))],
            [ca.DM(self._Z(nz, 2*nx)),  Iz]
        ])

        # Plant
        Mp = cas.sym("Mp", nx+ny, nx+nu)

        Ax = self.psi_1 @ Mp @ self.psi_2
        Bu = self.psi_1 @ Mp @ self.psi_3
        Cy = self.psi_4 @ Mp @ self.psi_2

        # Controller
        Ac, Bc, Cc, Dc, Mc, *_ = self.build_Mc_from_XYKLmn(Ax, Bu, Cy, X, Y, K, L, M, N)

        # Controller - relaxed
        """
        Ac = cas.sym("Ac", nx, nx)
        Bc = cas.sym("Bc", nx, ny)
        Cc = cas.sym("Cc", nu, nx)
        Dc = cas.sym("Dc", nu, ny)

        Mc = cas.blockcat([
            [Ac, Bc],
            [Cc, Dc]
        ])
        #"""

        # Closed-loop matrices
        A = ca.blockcat([
            [Ax + Bu @ Dc @ Cy,     Bu @ Cc],
            [Bc @ Cy,               Ac]
            ])
        B = ca.blockcat([
            [Bw],
            [cas.DM(self._Z(ny, nw))]
            ])
        C = ca.blockcat([
            [Cz + Dzu @ Dc @ Cy,    Dzu @ Cc]
            ])
        D = cas.DM(self._Z(nz, nw))
        
        Mcl = ca.blockcat([
            [A, B],
            [C, D]
            ])
        M = P_aug @ Mcl


        self.cas_vars = {
            # decision variables
            "vX": vX, "vY": vY, "vQ": vQ,
            "X": X, "Y": Y, 
            "K": K, "L": L, "M": M, "N": N,
            "lam": lam, "Q": Q,
            "P": P, "P_aug": P_aug,
            # closed-loop mats
            "Ax": Ax, "Bu": Bu, "Cy": Cy,
            "A": A, "B": B, "C": C, "D": D,
            "M": M, "Mcl": Mcl, "Mp": Mp, "Mc": Mc,
            # controller
            "Ac": Ac, "Bc": Bc, "Cc": Cc, "Dc": Dc,
        }

    def build_Phi(self):
        nx, nu, nw, ny, nz = self.get_dims()

        Ix, Iu, Iy, Iw, Iz, I2x = self._I(nx), self._I(nu), self._I(ny), self._I(nw), self._I(nz), self._I(2*nx)
        Oxy, Oux, O2xw, Oz2x = self._Z(nx, ny), self._Z(nu, nx), self._Z(2*nx, nw), self._I(nz, 2*nx)
        Oxu = Oux.T; Oyx = Oxy.T; Ow2x = O2xw.T; O2xz = Oz2x.T
        
        self.Psi_w = ca.DM(np.block([[O2xw], [Iw]]))
        self.Psi_2x = ca.DM(np.block([[I2x], [Ow2x]]))
        self.Psi_z = ca.DM(np.block([[I2x], [Oz2x]]))

        self.psi_1 = ca.DM(np.block([[Ix, Oxy]]))
        self.psi_2 = ca.DM(np.block([[Ix], [Oux]]))
        self.psi_3 = ca.DM(np.block([[Oxu], [Iu]]))
        self.psi_4 = ca.DM(np.block([[Oyx, Iy]]))

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



    # ============================================================
    # WFL: build one-step "Hankel" stacks from self.data
    # ============================================================

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
        nu, Tu = U_.shape
        ny, Ty = Y_.shape
        nx2, Tx = Xp1.shape
        if not (Tu == Ty == Tx == T):
            raise ValueError("Time length mismatch between X_,U_,Y_,X.")
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

    def wfl_rank_info(self, rcond: float = 1e-10):
        """
        Simple rank + conditioning report for Xp (useful to sanity-check data richness).
        Requires build_wfl_hankels() called first, or will build it.
        """
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
        """
        Builds WFL set constraints using block Hankels of depth L:
            Xf = Mp Xp + W, with columns w_i in W-set.

        Returns CVXPY variables Mp, W and constraint list.
        Stores them under self.wfl["cvxpy"].
        """
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
        """
        Solve a convex program to pick one Mp from the WFL set.
        """
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


    # ------------------------------------------------------------------
    # CONSTRAINTS: kernel del paper + bound su DeltaA/DeltaB
    # ------------------------------------------------------------------
    def build_con(self):
        nx, _, nw, _, nz = self.get_dims()

        cv = self.cas_vars
        P_aug  = cv["P_aug"]
        Q      = cv["Q"]
        lam    = cv["lam"]
        Mcl    = cv["Mcl"]

        eps = self.eps
        I_w = ca.DM(self._I(nw))
        I_xw = ca.DM(self._I(2*nx + nw))
        Z_zw = ca.DM(self._Z(nz, nw))
        Z_wxz = ca.DM(self._Z(nw, 2*nx+nz))

        g_list = []

        # ---------- kernel(s) ----------
        if self.model == "correlated":
            Xi = ca.blockcat([
                [-P_aug @ (I_xw + (lam-1)*self.Psi_w@self.Psi_w.T), lam*self.Psi_w, (P_aug @ Mcl).T ],
                [lam*self.Psi_w.T,                                  -Q -lam*I_w,    Z_wxz           ],
                [P_aug @ Mcl,                                       Z_zw,           -P_aug          ]
            ])
            eig_Xi = ca.eig_symbolic(Xi)          # Xi is SX → OK
            min_eig_Xi = ca.mmin(eig_Xi)
            g_Xi = min_eig_Xi + eps               # <= 0
            g_list.append(g_Xi)

        elif self.model == "independent":
            # Xi1
            Xi1 = ca.blockcat([
                [-self.Psi_z.T @ P_aug @ self.Psi_z,    (P_aug @ Mcl @ self.Psi_2x).T   ],
                [P_aug @ Mcl @ self.Psi_2x,             -P_aug                          ]
            ])
            eig_Xi1 = ca.eig_symbolic(Xi1)
            min_eig_Xi1 = ca.mmin(eig_Xi1)
            g_Xi1 = min_eig_Xi1 + eps
            g_list.append(g_Xi1)

            # Xi2
            Xi2 = ca.blockcat([
                [-lam * I_w,                lam * I_w,          (P_aug @ Mcl @ self.Psi_w).T    ],
                [ lam * I_w,                -Q - lam * I_w,     Z_wxz                           ],
                [ P_aug @ Mcl @ self.Psi_w, Z_wxz.T,            -P_aug                          ]
            ])
            eig_Xi2 = ca.eig_symbolic(Xi2)
            min_eig_Xi2 = ca.mmin(eig_Xi2)
            g_Xi2 = min_eig_Xi2 + eps
            g_list.append(g_Xi2)

        else:
            raise ValueError("model must be 'correlated' or 'independent'")


        # PSD-ish on P, Q: lambda_min >= eps
        eig_P = ca.eig_symbolic(P_aug)
        eig_Q = ca.eig_symbolic(Q)
        min_eig_P = ca.mmin(eig_P)
        min_eig_Q = ca.mmin(eig_Q)
        g_P = eps - min_eig_P
        g_Q = eps - min_eig_Q
        g_list.extend([g_P, g_Q])

        # lambda >= 0 (bounded in solve_prb)
        g_lam = lam
        g_list.append(g_lam)

        g = ca.vertcat(*g_list)
        self.cas_con = {"g": g, "g_list": g_list}

    # ------------------------------------------------------------------
    # OBJECTIVE & REG
    # ------------------------------------------------------------------
    def build_obj(self):
        """
        Build the scalar objective in CasADi:
            f = trace(Q Sigma_nom) + lam * gamma^2
        """
        cv = self.cas_vars
        Q   = cv["Q"]
        lam = cv["lam"]

        Sigma_nom = ca.DM(self.Sigma_nom)
        self.cas_obj = ca.trace(Q @ Sigma_nom) + lam * (self.gamma ** 2)

    def build_reg(self):
        """
        Optional regularizer (non-structural, just to tame the search).
        """
        if not self.reg_fro:
            self.cas_reg = 0.0
            return

        cv = self.cas_vars
        K = cv["K"]; L = cv["L"]; M = cv["M"]; N = cv["N"]
        X = cv["X"]; Y = cv["Y"]

        reg = (ca.sumsqr(K) + ca.sumsqr(L) +
            ca.sumsqr(M) + ca.sumsqr(N) +
            ca.sumsqr(X) + ca.sumsqr(Y))

        self.cas_reg = self.rho * reg

    # ------------------------------------------------------------------
    # SOLVE
    # ------------------------------------------------------------------
    def solve_prb(self):
        """
        Build the NLP (non-convex) in CasADi and solve it with IPOPT.
        Uses:
            x  = stacked decision variables
            f  = obj + reg
            g  = constraint vector (<= 0, except last one for lam >= 0)
        """
        cv  = self.cas_vars
        con = self.cas_con

        g = con["g"]
        f = self.cas_obj + self.cas_reg

        # --- pack all decision variables into x -------------------------
        vX = cv["vX"]; vY = cv["vY"]; vQ = cv["vQ"]
        K  = cv["K"];  L  = cv["L"]
        M  = cv["M"];  N  = cv["N"]
        DA = cv["DeltaA"]; DB = cv["DeltaB"]
        lam = cv["lam"]

        x = ca.vertcat(
            vX,
            vY,
            vQ,
            ca.reshape(K, -1, 1),
            ca.reshape(L, -1, 1),
            ca.reshape(M, -1, 1),
            ca.reshape(N, -1, 1),
            ca.reshape(DA, -1, 1),
            ca.reshape(DB, -1, 1),
            lam
        )

        n_x = x.numel()
        n_g = g.numel()

        nlp = {"x": x, "f": f, "g": g}

        opts = {
            "ipopt.print_level": 5,
            "ipopt.max_iter": 500,
            "print_time": True,
        }
        solver = ca.nlpsol("solver", "ipopt", nlp, opts)

        # Bounds on x (very loose)
        lbx = -1e3 * np.ones(n_x)
        ubx =  1e3 * np.ones(n_x)

        # Constraints:
        #   g[0]..g[4] <= 0
        #   g[5] = lam >= 0 -> lower bound 0, upper large
        lbg = -1e9 * np.ones(n_g)
        ubg = np.zeros(n_g)
        # lam >= 0
        lbg[-1] = 0.0
        ubg[-1] = 1e9

        x0 = np.zeros(n_x)  # you can seed from convex solution if you want

        sol = solver(x0=x0, lbx=lbx, ubx=ubx, lbg=lbg, ubg=ubg)

        x_opt = np.array(sol["x"]).flatten()
        f_opt = float(sol["f"])

        self.solver = "ipopt"
        self.status = solver.stats().get("return_status", "unknown")
        self.value  = f_opt

        # ---- unpack solution back to matrices --------------------------
        # same order as in x stacking
        nx, nu, nw, ny, nz = self.get_dims()

        def unstack_sym(v, n):
            M = np.zeros((n, n))
            idx = 0
            for i in range(n):
                for j in range(i, n):
                    M[i, j] = v[idx]
                    M[j, i] = v[idx]
                    idx += 1
            return M, idx

        idx = 0
        vX_opt = x_opt[idx : idx + nx*(nx+1)//2]
        idx += nx*(nx+1)//2
        vY_opt = x_opt[idx : idx + nx*(nx+1)//2]
        idx += nx*(nx+1)//2
        vQ_opt = x_opt[idx : idx + nw*(nw+1)//2]
        idx += nw*(nw+1)//2

        X_opt, _ = unstack_sym(vX_opt, nx)
        Y_opt, _ = unstack_sym(vY_opt, nx)
        Q_opt, _ = unstack_sym(vQ_opt, nw)

        # K
        K_size = nx*nx
        K_opt = x_opt[idx : idx + K_size].reshape(nx, nx); idx += K_size
        # L
        L_size = nx*ny
        L_opt = x_opt[idx : idx + L_size].reshape(nx, ny); idx += L_size
        # M
        M_size = nu*nx
        M_opt = x_opt[idx : idx + M_size].reshape(nu, nx); idx += M_size
        # N
        N_size = nu*ny
        N_opt = x_opt[idx : idx + N_size].reshape(nu, ny); idx += N_size
        # DeltaA
        DA_size = nx*nx
        DeltaA_opt = x_opt[idx : idx + DA_size].reshape(nx, nx); idx += DA_size
        # DeltaB
        DB_size = nx*nu
        DeltaB_opt = x_opt[idx : idx + DB_size].reshape(nx, nu); idx += DB_size
        # lam
        lam_opt = float(x_opt[idx])

        # rebuild A_true, B, C, D, P numerically
        Ax, Bw, Bu, Cy, Dyw, Cz, Dzw, Dzu = self.get_mats()
        Ix = np.eye(nx)

        P_opt = np.block([[Y_opt, Ix],
                        [Ix,    X_opt]])

        # same formulas as in build_var
        A_nom_opt = np.block([
            [Ax @ Y_opt + Bu @ M_opt,       Ax + Bu @ N_opt @ Cy],
            [K_opt,                         X_opt @ Ax + L_opt @ Cy]
        ])

        B_opt = np.block([
            [Bw + Bu @ N_opt @ Dyw],
            [X_opt @ Bw + L_opt @ Dyw]
        ])

        C_opt = np.block([
            [Cz @ Y_opt + Dzu @ M_opt, Cz + Dzu @ N_opt @ Cy]
        ])

        D_opt = Dzw + Dzu @ N_opt @ Dyw

        # structured E
        Zxx = np.zeros((nx, nx))
        Zxn = np.zeros((nx, nu))
        E_A_opt = np.block([
            [DeltaA_opt @ Y_opt, DeltaA_opt],
            [Zxx,                X_opt @ DeltaA_opt]
        ])
        E_B_opt = np.block([
            [DeltaB_opt @ M_opt, DeltaB_opt @ N_opt @ Cy],
            [Zxn,                Zxx]
        ])
        E_opt = E_A_opt + E_B_opt
        A_true_opt = A_nom_opt + E_opt

        # package result (you can adapt to your DROLMIResult)
        dro = DROLMIResult(
            solver=self.solver,
            status=self.status,
            obj_value=self.value,
            gamma=self.gamma,
            lambda_opt=lam_opt,
            Q=Q_opt, X=X_opt, Y=Y_opt,
            K=K_opt, L=L_opt, M=M_opt, N=N_opt,
            Pbar=P_opt,
            Abar=A_true_opt, Bbar=B_opt, Cbar=C_opt, Dbar=D_opt,
            Tp=None, P=None,
        )

        self.outs = dro
        self.total_constraints = int(n_g)
        # violations here puoi stimarle ricalcolando g(x_opt), se vuoi

    # ------------------------------------------------------------------
    # PACK OUTPUTS
    # ------------------------------------------------------------------
    def pack_outs(self):
        lam, Q, P = self.get_vars("main")
        K, L, Y, X, M, N = self.get_vars("inner")
        A_true, B, C, D = self.get_vars("mats")

        P_val, Q_val, lam_val = map(self._val, [P.value, Q.value, lam.value])
        K_val, L_val, Y_val, X_val, M_val, N_val = map(
            self._val, [K.value, L.value, Y.value, X.value, M.value, N.value]
        )
        A_val, B_val, C_val, D_val = map(
            self._val, [A_true.value, B.value, C.value, D.value]
        )

        dro = DROLMIResult(
            solver=self.solver,
            status=self.status,
            obj_value=float(self.value),
            gamma=self.gamma,
            lambda_opt=lam_val,
            Q=Q_val, X=X_val, Y=Y_val,
            K=K_val, L=L_val, M=M_val, N=N_val,
            Pbar=P_val, Abar=A_val, Bbar=B_val,
            Cbar=C_val, Dbar=D_val,
            Tp=None, P=None,
        )
        self.outs = dro

        # per ora salvo solo cose basilari sugli errori
        DeltaA = self.vars["DeltaA"].value
        DeltaB = self.vars["DeltaB"].value
        self.others = {
            "DeltaA": DeltaA,
            "DeltaB": DeltaB,
            "beta": self.beta,
            "beta_A": self.beta_A,
            "beta_B": self.beta_B,
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
