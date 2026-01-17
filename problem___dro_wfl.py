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
                 rho: float = 1e-2, eps: float = 1e-6, N_sims: int = 1,
                 Bw_mode: str = "known_cov",
                 real_Z_mats: bool = True,
                 aug_mode: str = "std",
                 eval_from_ol: bool = True,
                 estm_noise: bool = False,
                 reg_fro: bool = False):
        
        Bw_mode = "proj" if estm_noise else Bw_mode

        self.api = api
        self.eps, self.rho, self.N_sims = eps, rho, N_sims
        self.model, self.Bw_mode, self.vals = model, vals[5], vals
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
        return (self.mats["Ax"],
                self.mats["Bw"],
                self.mats["Bu"],
                self.mats["Cy"],
                self.mats["Dyw"],
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
        Ax, Bw, Bu, Cy, Dyw, Cz, Dzw, Dzu = self.get_mats()

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

        vX = symm_var("X", nx)
        vY = symm_var("Y", nx)
        vQ = symm_var("Q", nw)

        Ac = cas.sym("Ac", nx, nx)
        Bc = cas.sym("Bc", nx, ny)
        Cc = cas.sym("Cc", nu, nx)
        Dc = cas.sym("Dc", nu, ny)

        K  = cas.sym("K", nx, nx)
        L  = cas.sym("L", nx, ny)
        M  = cas.sym("M", nu, nx)
        N  = cas.sym("N", nu, ny)

        lam = cas.sym("lam", 1)
        Mp = cas.sym("Mp", nx+ny, nx+nu)
        Mc = cas.blockcat([
            [Ac, Bc],
            [Cc, Dc]
        ])

        X = full_sym(vX, nx)
        Y = full_sym(vY, nx)
        Q = full_sym(vQ, nw)

        Ix  = ca.DM(np.eye(nx))
        Iw  = ca.DM(np.eye(nw))
        Iz  = ca.DM(np.eye(nz))

        Ax = self.psi_1 @ Mp @ self.psi_2
        Bu = self.psi_1 @ Mp @ self.psi_3
        Cy = self.psi_4 @ Mp @ self.psi_2

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


        AxDM  = ca.DM(Ax)
        BuDM  = ca.DM(Bu)
        BwDM  = ca.DM(Bw)
        CyDM  = ca.DM(Cy)
        DywDM = ca.DM(Dyw)
        CzDM  = ca.DM(Cz)
        DzuDM = ca.DM(Dzu)
        DzwDM = ca.DM(Dzw)

        P = ca.blockcat([[Y,  Ix],
                        [Ix,  X ]])
        P_aug = ca.blockcat([
            [P,                         ca.DM(self._Z(2*nx, nz))],
            [ca.DM(self._Z(nz, 2*nx)),  Iz]
        ])

        A_nom = ca.blockcat([
            [AxDM @ Y + BuDM @ M,       AxDM + BuDM @ N @ CyDM],
            [K,                        X @ AxDM + L @ CyDM]
        ])

        B = ca.blockcat([
            [BwDM + BuDM @ N @ DywDM],
            [X @ BwDM + L @ DywDM]
        ])

        C = ca.blockcat([
            [CzDM @ Y + DzuDM @ M, CzDM + DzuDM @ N @ CyDM]
        ])

        D = DzwDM + DzuDM @ N @ DywDM

        Zxx = ca.DM(np.zeros((nx, nx)))

        E_A = ca.blockcat([
            [DeltaA @ Y, DeltaA     ],
            [Zxx,        X @ DeltaA ]
        ])
        E_B = ca.blockcat([
            [DeltaB @ M, DeltaB @ N @ CyDM],
            [Zxx,        Zxx             ]
        ])

        E      = E_A + E_B
        A_true = A_nom + E

        self.cas_vars = {
            "vX": vX, "vY": vY, "vQ": vQ,
            "X": X, "Y": Y, "Q": Q,
            "K": K, "L": L, "M": M, "N": N,
            "DeltaA": DeltaA, "DeltaB": DeltaB,
            "lam": lam,
            "P": P,
            "A_nom": A_nom,
            "B": B, "C": C, "D": D,
            "E": E, "A_true": A_true,
            "Iw": Iw, "Iz": Iz,
        }

    def build_selectors(self):
        nx, nu, nw, ny, nz = self.get_dims()

        Ix, Iu, Iy, Iw, Iz, I2x = self._I(nx), self._I(nu), self._I(ny), self._I(nw), self._I(nz), self._I(2*nx)
        Oxy, Oux, O2xw, Oz2x = self._Z(nx, ny), self._Z(nu, nx), self._Z(2*nx, nw), self._I(nz, 2*nx)
        Oxu = Oux.T; Oyx = Oxy.T; Ow2x = O2xw.T; O2xz = Oz2x.T
        
        self.Psi_w = np.block([[O2xw], [Iw]])
        self.Psi_2x = np.block([[I2x], [Ow2x]])
        self.Psi_z = np.block([[I2x], [Oz2x]])

        self.psi_1 = np.block([[Ix, Oxy]])
        self.psi_2 = np.block([[Ix], [Oux]])
        self.psi_3 = np.block([[Oxu], [Iu]])
        self.psi_4 = np.block([[Oyx, Iy]])



    # ------------------------------------------------------------------
    # CONSTRAINTS: kernel del paper + bound su DeltaA/DeltaB
    # ------------------------------------------------------------------
    def build_con(self):
        nx, _, nw, _, nz = self.get_dims()
        Nchi = 2 * nx

        cv = self.cas_vars
        P      = cv["P"]
        Q      = cv["Q"]
        lam    = cv["lam"]
        A_true = cv["A_true"]
        B      = cv["B"]
        C      = cv["C"]
        D      = cv["D"]
        Iw     = cv["Iw"]
        Iz     = cv["Iz"]
        DeltaA = cv["DeltaA"]
        DeltaB = cv["DeltaB"]

        eps = self.eps

        Z_chi_w = ca.DM(np.zeros((Nchi, nw)))
        Z_w_chi = ca.DM(np.zeros((nw, Nchi)))
        Z_chi_z = ca.DM(np.zeros((Nchi, nz)))
        Z_z_chi = ca.DM(np.zeros((nz, Nchi)))
        Z_w_z   = ca.DM(np.zeros((nw, nz)))
        Z_z_w   = ca.DM(np.zeros((nz, nw)))

        g_list = []

        # ---------- kernel(s) ----------
        if self.model == "correlated":
            Xi = ca.blockcat([
                [-P,          Z_chi_w,         Z_chi_w,         A_true.T,   C.T],
                [Z_w_chi,    -lam * Iw,        lam * Iw,        B.T,        D.T],
                [Z_w_chi,     lam * Iw,       -Q - lam * Iw,    Z_w_chi,    Z_w_z],
                [A_true,      B,               Z_chi_w,        -P,          Z_chi_z],
                [C,           D,               Z_z_w,           Z_z_chi,   -Iz]
            ])
            eig_Xi = ca.eig_symbolic(Xi)          # Xi is SX → OK
            min_eig_Xi = ca.mmin(eig_Xi)
            g_Xi = min_eig_Xi + eps               # <= 0
            g_list.append(g_Xi)

        elif self.model == "independent":
            # Xi1
            Xi1 = ca.blockcat([
                [-P,        A_true.T,     C.T      ],
                [A_true,   -P,            Z_chi_z  ],
                [C,         Z_z_chi,     -Iz      ]
            ])
            eig_Xi1 = ca.eig_symbolic(Xi1)
            min_eig_Xi1 = ca.mmin(eig_Xi1)
            g_Xi1 = min_eig_Xi1 + eps
            g_list.append(g_Xi1)

            # Xi2
            Xi2 = ca.blockcat([
                [-lam * Iw,          lam * Iw,          B.T,        D.T],
                [ lam * Iw,    -Q - lam * Iw,          Z_w_chi,    Z_w_z],
                [ B,                 Z_chi_w,         -P,          Z_chi_z],
                [ D,                 Z_z_w,           Z_z_chi,    -Iz]
            ])
            eig_Xi2 = ca.eig_symbolic(Xi2)
            min_eig_Xi2 = ca.mmin(eig_Xi2)
            g_Xi2 = min_eig_Xi2 + eps
            g_list.append(g_Xi2)

        else:
            raise ValueError("model must be 'correlated' or 'independent'")

        # Frobenius bounds on DeltaA, DeltaB
        g_DA = ca.sumsqr(DeltaA) - self.beta_A**2
        g_DB = ca.sumsqr(DeltaB) - self.beta_B**2
        g_list.extend([g_DA, g_DB])

        # PSD-ish on P, Q: lambda_min >= eps
        eig_P = ca.eig_symbolic(P)
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

