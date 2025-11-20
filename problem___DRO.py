import numpy as np
import cvxpy as cp
from utils___systems import DROLMIResult, Noise
from utils___matrices import MatricesAPI, recover_deltas
from utils___simulate import Open_Loop
from utils___ambiguity import Disturbances
import sys



class DRO: 
    def __init__(self, 
                 vals: tuple, model: str,
                 api: MatricesAPI, noise: Noise, 
                 rho: float = 1e-2, eps: float = 1e-6, sim_N: int = 20,
                 Bw_mode: str = "known_cov", real_perf_mats: bool = False, 
                 aug_mode: str = "std", eval_from_ol: bool = True, estm_noise: bool = False,
                 reg_fro: bool = False, reg_beta: bool = True, new: bool = True,
                 ):

        Bw_mode = "proj" if estm_noise else Bw_mode
        self.api = api
        self.eps, self.rho, self.sim_N = eps, rho, sim_N
        self.model, self.Bw_mode, self.vals = model, Bw_mode, vals
        self.real_perf_mats, self.augmented, self.aug_mode, self.eval_from_ol, self.estm_noise \
              = real_perf_mats, vals[3], aug_mode, eval_from_ol, estm_noise
        self.gamma, self.var, self.Sigma_nom = noise.gamma, noise.var, noise.Sigma_nom
        self.inp = vals[4]

        self.reg_fro, self.reg_beta, self.reg_vect = reg_fro, reg_beta, vals[2]
        self.new = new



    # ============================================================================ #

    def run(self):
        self.simulate_()    # -> builds self.data & self.dims
        self.estm_mats()    # -> builds self.mats
        self.build_var()    # -> builds self.vars
        self.build_con()    # -> builds self.cons
        self.build_obj()    # -> builds self.objs
        self.build_reg()    # -> builds self.regs

        self.solve_prb()    # -> MOSEK/SCS solver
        self.pack_outs()    # -> builds self.outs & self.others

        return self.outs, (self.get_mats()), self.Sigma_nom, self.others, (self.violations, self.total_constraints)


    # ============================================================================ #

    def _I(self, n:int, m: int = None): 
        m = n if m is None else m
        return np.eye(n, m)
    
    def _Z(self, n: int, m: int = None): 
        m = n if m is None else m
        return np.zeros((n, m))

    def _pseudo_inv(self, M: np.ndarray): 
        return M.T @ np.linalg.inv(M @ M.T + self.eps * self._I(M.shape[0]))
    
    def _sym(self, M: np.ndarray): 
        return 0.5 * (M + M.T) + self.eps * self._I(M.shape[0])
    
    def _negdef(self, M: np.ndarray, which: str = "relaxed"): 
        if which == "strict":
            return M << 0
        else:
            return M << -self.eps * self._I(M.shape[0], M.shape[1])
    
    def _posdef(self, M: np.ndarray, which: str = "relaxed"):
        if which == "strict": 
            return M >> 0
        else:
            return M >> self.eps * self._I(M.shape[0], M.shape[1])

    def _val(self, m): 
        if m is None: return None
        return float(m) if np.isscalar(m) else m


    # ============================================================================ #

    def get_data(self): 
        return self.data["X_"], self.data["U_"], self.data["X"], self.data["Y_"], self.data["Z_"]
    
    def get_dims(self): 
        return self.dims["nx"], self.dims["nu"], self.dims["nw"], self.dims["ny"], self.dims["nz"]
    
    def get_mats(self): 
        return self.mats["Ax"], self.mats["Bw"], self.mats["Bu"], self.mats["Cy"], self.mats["Dyw"], self.mats["Cz"], self.mats["Dzw"], self.mats["Dzu"] 

    def get_vars(self, which: str = "main"): 
        if which == "main":
            return self.vars["lam"], self.vars["Q"], self.vars["P"]
        elif which == "inner":
            return self.vars["K"], self.vars["L"], self.vars["Y"], self.vars["X"], self.vars["M"], self.vars["N"]
        elif which == "mats":
            return self.vars["A"], self.vars["B"], self.vars["C"], self.vars["D"]
        elif which == "t":
            return self.vars["tK"], self.vars["tL"], self.vars["tY"], self.vars["tX"], self.vars["tM"], self.vars["tN"]#, self.vars["tP"]


    # ============================================================================ #

    def select_representative_run(self, datasets, keys=("X","U","Y","Z","X_next"), weights=None):
        """
        Choose the medoid (most representative) dataset across the given keys.
        Returns a dict with aligned X,U,Y,Z,X_next from that single seed.
        """
        if not datasets:
            raise ValueError("datasets is empty")

        # 1) Align horizons
        T_min = min(d["meta"]["T"] for d in datasets)
        Teff      = T_min
        Teff_next = T_min - 1
        if Teff_next <= 0:
            raise ValueError("Need T >= 2 to align X_next")

        # 2) Stack aligned arrays
        stacks = {}
        for k in keys:
            if k == "X_next":
                stacks[k] = np.stack([d[k][..., :Teff_next] for d in datasets], axis=0)   # (N, n, T-1)
            else:
                stacks[k] = np.stack([d[k][..., :Teff]      for d in datasets], axis=0)   # (N, n, T)

        N = next(iter(stacks.values())).shape[0]
        if weights is None:
            weights = {k: 1.0 for k in keys}

        # 3) Per-key scaling so no single key dominates (Frobenius mean per seed)
        def fro_scale(S):  # S shape: (N, ...)
            return np.mean([np.linalg.norm(S[i]) for i in range(S.shape[0])]) + 1e-12

        scales = {k: fro_scale(S) for k, S in stacks.items()}

        # 4) Medoid index: minimize total weighted squared distance to others
        dists = np.zeros(N)
        for i in range(N):
            total = 0.0
            for k, S in stacks.items():
                Sk = S / scales[k]
                diff = Sk - Sk[i]              # (N, ...)
                total += float(weights[k]) * np.sum(diff**2)
            dists[i] = total
        i_star = int(np.argmin(dists))

        # 5) Build output using that single, real run (preserves dynamics)
        out = {
            "X":      stacks["X"][i_star],                        # (nx, T)
            "U":      stacks["U"][i_star],                        # (nu, T)
            "Y":      stacks["Y"][i_star],                        # (ny, T)
            "Z":      stacks["Z"][i_star],                        # (nz, T)
            # pad last column so X_next matches T
            "X_next": np.hstack([stacks["X_next"][i_star], stacks["X_next"][i_star][:, -1][:, None]]),
            "meta": {
                **datasets[0]["meta"],
                "T": T_min,
                "N": len(datasets),
                "selected_seed": i_star,
                "selection": "medoid_over_"+",".join(keys)
            }
        }
        return out

    def simulate_(self):
        if not self.eval_from_ol:
            upd, FROM_DATA, *_ = self.vals
            data = self.api.get_system(FROM_DATA=FROM_DATA, gamma=self.gamma, upd=upd)
            X_, X, U_, Y_, Z_ = data.get_data()

        else:
            op = Open_Loop(MAKE_DATA=False, EVAL_FROM_PATH=False, DATASETS=True, N=self.sim_N)
            datasets = op.datasets


            avg = self.select_representative_run(datasets)
            X_, U_, Y_, Z_, X = avg["X"], avg["U"], avg["Y"], avg["Z"], avg["X_next"]

        self.data = {
            "X": X, 
            "X_": X_, 
            "U_": U_, 
            "Y_": Y_, 
            "Z_": Z_,
        }

        self.dims = {
            "T": X_.shape[1], 
            "nx": X_.shape[0],
            "nu": U_.shape[0],
            "ny": Y_.shape[0],
            "nz": Z_.shape[0],
        }

    def spectral_norm_epigraph(self, A: cp.Expression, name: str):
        """
        Impone ||A||_2 <= t_name con un'epigrafe LMI:
            [[t I_m, A],
            [A.T,   t I_n]] >> 0
        Ritorna la variabile scalare t e la lista dei vincoli.
        """
        m, n = A.shape
        t = cp.Variable(nonneg=True, name=f"t_{name}")
        blk = cp.bmat([ [t * self._I(m),    A               ],
                        [A.T,               t * self._I(n)  ]])
        return t, [blk >> 0], [t >= 0]


    # ============================================================================ #

    def estm_mats(self): 
        X_, U_, X, Y_, Z_ = self.get_data()
        nx, nu = self.dims["nx"], self.dims["nu"]

        Dx = np.vstack([X_, U_])
        Ox = X @ self._pseudo_inv(Dx)
        Ax, Bu = Ox[:, :nx], Ox[:, nx:nx+nu]

        R = X - (Ax @ X_ + Bu @ U_)
        self.c = R @ self._pseudo_inv(Dx)
        self.c_a = np.sqrt(nx/(nx+nu)) * self.c
        self.c_b = np.sqrt(nu/(nx+nu)) * self.c

        Bw, nw, self._residual_anisotropy_weights = self.estm_Bw(R)
        W_ = self._pseudo_inv(Bw) @ R

        if self.estm_noise:
            d = Disturbances(n=nw)
            self.Sigma_nom = d.estm_Sigma_nom(W_.T)
            self.gamma, *_ = d._estimate_gamma_with_ci(W_.T)


        Dy = np.vstack([X_, W_])
        Oy = Y_ @ self._pseudo_inv(Dy)
        Cy, Dyw = Oy[:, :nx], Oy[:, nx:nx+nw]

        if self.real_perf_mats:
            Cz, Dzw, Dzu, *_ = self.api.build_out_matrices(nw=nw)
        else:
            Dz = np.vstack([X_, U_, W_])
            Oz = Z_ @ self._pseudo_inv(Dz)
            Cz, Dzu, Dzw = Oz[:, :nx], Oz[:, nx:nx+nu], Oz[:, nx+nu:nx+nu+nw]


        if self.augmented: 
            N = None if self.aug_mode == "std" else (1, 1)
            Bw, Dzw, Dyw, nw, self.Sigma_nom = self.api._augment_matrices(B_w=Bw, D_vw=Dzw, D_yw=Dyw, var=self.var, Sigma_nom=self.Sigma_nom, N=N)

        if self.reg_beta: 
            ss = np.linalg.svd(Dx, compute_uv=False)
            smin = float(ss[-1]) if ss.size else 0.0
            self.beta = np.linalg.norm(R, 'fro') / max(smin, 1e-12)


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

        # eig decomposition
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

            # eig of Sigma_nom
            lam, Q = np.linalg.eigh(Sigma_nom)
            lam = np.clip(lam, self.eps, None)
            Sigma_inv_sqrt = Q @ np.diag(lam**-0.5) @ Q.T

            Bw = Up @ np.diag(sp_sqrt) @ Sigma_inv_sqrt

            return Bw, nw, (U, s, w)


        # rank selection
        total = max(float(np.sum(s)), 1e-18)
        print(f"Total residual energy: {total}")
        cum = np.cumsum(s) / total
        nw = int(np.clip(np.searchsorted(cum, eta) + 1, 1, nx))
        if nw != 2:
            if self.inp: input("...")
            nw = 2

        Up = U[:, :nw]
        sp = s[:nw]
        sp_sqrt = np.sqrt(sp)

        if self.Bw_mode == "factor":
            # low-rank factor: S ≈ Bw Bw^T, w ~ N(0, I)
            Bw = Up @ np.diag(sp_sqrt)

        elif self.Bw_mode == "proj":
            # just the subspace: Bw Bw^T = Up Up^T
            Bw = Up

        elif self.Bw_mode == "white":
            # use factor but normalize so Σ_w = I in that nw-dim space
            # here it's effectively same as 'factor' if you treat w ~ N(0, I)
            Bw = Up @ np.diag(sp_sqrt)

        else:
            raise ValueError("Bw_mode must be in {'factor','proj','known_cov','white'}")

        return Bw, nw, (U, s, w)


    # ============================================================================ #

    def build_var(self):
        nx, nu, nw, ny, _ = self.get_dims()

        lam = cp.Variable(nonneg=True, name="lambda")
        Q = cp.Variable((nw, nw), PSD=True, name="Q")

        X = cp.Variable((nx, nx), symmetric=True, name="X")
        Y = cp.Variable((nx, nx), symmetric=True, name="Y")
        K = cp.Variable((nx, nx), name="K")
        L = cp.Variable((nx, ny), name="L")
        M = cp.Variable((nu, nx), name="M")
        N = cp.Variable((nu, ny), name="N")

        Ax, Bw, Bu, Cy, Dyw, Cz, Dzw, Dzu = self.get_mats()
        Ix = self._I(nx)


        # DRO matrices
        P = cp.bmat([
            [Y,     Ix], 
            [Ix,    X]
        ])

        A = cp.bmat([
            [Ax @ Y + Bu @ M,       Ax + Bu @ N @ Cy ], 
            [K,                     X @ Ax + L @ Cy]
        ])
        B = cp.bmat([
            [Bw + Bu @ N @ Dyw], 
            [X @ Bw + L @ Dyw]
        ])
        C = cp.bmat([ 
            [Cz @ Y + Dzu @ M,      Cz + Dzu @ N @ Cy]
        ])
        D = Dzw + Dzu @ N @ Dyw

        self.vars = {
            "lam": lam, 
            "Q": Q, 
            "X": X, 
            "Y": Y, 
            "K": K, 
            "L": L, 
            "M": M, 
            "N": N, 
            "P": P,
            "A": A, 
            "B": B,
            "C": C, 
            "D": D,
        }

    def build_con(self):
        if self.new: 
            self.build_con_new()
        else: 
            self.build_con_old()

    def build_con_old(self):
        cons = []
        lam, Q, P = self.get_vars(which="main")

        #cons += [lam >= 0]
        #cons += [self._posdef(Q, which="strict")]
        cons += [self._posdef(P)]

        A, B, C, D = self.get_vars(which="mats")
        nx, nu, nw, _, nz = self.get_dims()

        if self.reg_beta: 
            U_vect, _, w_vect = self._residual_anisotropy_weights
            w_vect = np.maximum(w_vect, self.eps)
            w_np = np.mean(w_vect)

            self.beta_a = self.beta * np.sqrt(nx/(nx+nu))
            self.beta_b = self.beta * np.sqrt(nu/(nx+nu))
            S_base = np.hstack([self._I(nx), self._Z(nx, nx)]).T

            if self.reg_vect:
                beta_A_dir_np = np.asarray(self.beta_a * w_vect, dtype=float)
                S_A  = S_base @ U_vect 
                self.tau_A = cp.Variable(nx, nonneg=True, name="tau_a_vec")
                self.s_A = cp.Variable(nx, nonneg=True, name="s_a_vec")
                self.beta_A = cp.Parameter(nx, nonneg=True, value=beta_A_dir_np)

                for i in range(nx):
                    block_i = cp.bmat([[self.s_A[i],     self.beta_A[i]], 
                                    [self.beta_A[i],   self.tau_A[i]]])
                    
                    #cons += [self.s_A[i] >= self.eps, self.tau_A[i] <= 1e3]
                    cons += [self._posdef(block_i, which="relaxed")]

                young_blk_A = -cp.diag(self.s_A)


                #"""
                beta_B_dir_np = np.asarray(self.beta_b * w_vect, dtype=float)
                S_B  = S_base @ U_vect 
                self.tau_B = cp.Variable(nx, nonneg=True, name="tau_b_vec")
                self.s_B = cp.Variable(nx, nonneg=True, name="s_b_vec")
                self.beta_B = cp.Parameter(nx, nonneg=True, value=beta_B_dir_np)

                for i in range(nx):
                    block_i = cp.bmat([[self.s_B[i],     self.beta_B[i]], 
                                    [self.beta_B[i],   self.tau_B[i]]])
                    
                    #cons += [self.s_B[i] >= self.eps, self.tau_B[i] <= 1e3]
                    cons += [self._posdef(block_i, which="relaxed")]

                young_blk_B = -cp.diag(self.s_B)    #"""

            else: 
                cons += [self._posdef(Q, which="relaxed")]

                beta_A_np = w_np * self.beta_a
                S_A = S_base
                self.tau_A = cp.Variable(nonneg=True, name="tau_a")
                self.s_A = cp.Variable(nonneg=True, name="s_a")
                self.beta_A = cp.Parameter(nonneg=True, value=float(np.clip(beta_A_np, 0.0, 1e3)))

                block_a = cp.bmat([[self.s_A, self.beta_A], 
                                [self.beta_A, self.tau_A]])
                
                #cons += [self.s_A >= self.eps, self.tau_A <= 1e3]
                cons += [self._posdef(block_a, which="strict")]
                young_blk_A = -self.s_A * self._I(nx)
            

                beta_B_np = w_np * self.beta_b
                S_B = S_base
                self.tau_B = cp.Variable(nonneg=True, name="tau_b")
                self.s_B = cp.Variable(nonneg=True, name="s_b")
                self.beta_B = cp.Parameter(nonneg=True, value=float(np.clip(beta_B_np, 0.0, 1e3)))

                block_b = cp.bmat([[self.s_B, self.beta_B], 
                                [self.beta_B, self.tau_B]])
                
                #cons += [self.s_B >= self.eps, self.tau_B <= 1e3]
                cons += [self._posdef(block_b, which="strict")]
                young_blk_B = -self.s_B * self._I(nx)


            state_blk = -P + (cp.sum(self.tau_A) + cp.sum(self.tau_B)) * self._I(2*nx)



        if self.model == "correlated": 
            if not self.reg_beta:
                blk = cp.bmat([
                    [-P,                    self._Z(2*nx, nw),  self._Z(2*nx, nw),      A.T,                C.T                 ],
                    [ self._Z(nw, 2*nx),   -lam*self._I(nw),    lam*self._I(nw),        B.T,                D.T                 ],
                    [ self._Z(nw, 2*nx),    lam*self._I(nw),   -Q - lam*self._I(nw),    self._Z(nw, 2*nx),  self._Z(nw, nz)     ],
                    [  A,                   B,                  self._Z(2*nx, nw),     -P,                  self._Z(2*nx, nz)   ],
                    [  C,                   D,                  self._Z(nz, nw),        self._Z(nz, 2*nx), -self._I(nz)         ],
                ])
            else: 
                blk = cp.bmat([
                    [-P,                    self._Z(2*nx, nw),  self._Z(2*nx, nw),      A.T,                C.T,                self._Z(2*nx, nx),      self._Z(2*nx, nx)       ],
                    [ self._Z(nw,2*nx),    -lam*self._I(nw),    lam*self._I(nw),        B.T,                D.T,                self._Z(nw,   nx),      self._Z(nw,   nx)       ],
                    [ self._Z(nw,2*nx),     lam*self._I(nw),   -Q - lam*self._I(nw),    self._Z(nw,2*nx),   self._Z(nw,  nz),   self._Z(nw,   nx),      self._Z(nw,   nx)       ],
                    [ A,                    B,                  self._Z(2*nx, nw),      state_blk,          self._Z(2*nx,nz),   S_A,                    S_B                     ],
                    [ C,                    D,                  self._Z(nz,  nw),       self._Z(nz, 2*nx), -self._I(nz),        self._Z(nz,   nx),      self._Z(nz,   nx)       ],
                    [ self._Z(nx,2*nx),     self._Z(nx,  nw),   self._Z(nx,  nw),       S_A.T,              self._Z(nx,  nz),   young_blk_A,            self._Z(nx,   nx)       ],
                    [ self._Z(nx,2*nx),     self._Z(nx,  nw),   self._Z(nx,  nw),       S_B.T,              self._Z(nx,  nz),   self._Z(nx, nx),        young_blk_B             ],
                ])

            cons += [self._negdef(blk, which="strict")]

        elif self.model == "independent":
            if not self.reg_beta:
                blk1 = cp.bmat([
                    [-P,    A.T,                C.T                 ],
                    [ A,   -P,                  self._Z(2*nx, nz)   ],
                    [ C,    self._Z(nz, 2*nx), -self._I(nz)         ],
                ])
            else:
                blk1 = cp.bmat([
                    [ -P,                   A.T,                C.T,                self._Z(2*nx, nx),  self._Z(2*nx, nx)       ],
                    [  A,                   state_blk,          self._Z(2*nx, nz),  S_A,                S_B                     ],
                    [  C,                   self._Z(nz, 2*nx), -self._I(nz)  ,      self._Z(nz, nx),    self._Z(nz, nx)         ],
                    [  self._Z(nx,2*nx),    S_A.T,              self._Z(nx, nz),    young_blk_A,        self._Z(nx,   nx)       ],
                    [  self._Z(nx,2*nx),    S_B.T,              self._Z(nx, nz),    self._Z(nx,   nx),  young_blk_B             ],
                ])  

            blk2 = cp.bmat([
                [-lam*self._I(nw),  lam*self._I(nw),    B.T,                D.T                 ],
                [ lam*self._I(nw), -Q-lam*self._I(nw),  self._Z(nw, 2*nx),  self._Z(nw, nz)     ],
                [ B,                self._Z(2*nx, nw), -P,                  self._Z(2*nx, nz)   ],
                [ D,                self._Z(nz, nw),    self._Z(nz, 2*nx), -self._I(nz)         ],
            ]) 

            cons += [self._negdef(blk1, which="strict")]
            cons += [self._negdef(blk2, which="strict")]
        

        if self.reg_fro:
            K, L, Y, X, M, N = self.get_vars(which="inner")
            tK, consK, _ = self.spectral_norm_epigraph(K, "K")
            tL, consL, _ = self.spectral_norm_epigraph(L, "L")
            tM, consM, _ = self.spectral_norm_epigraph(M, "M")
            tN, consN, _ = self.spectral_norm_epigraph(N, "N")
            #tP, consP = self.spectral_norm_epigraph(P, "P") 

            cons += consK + consL + consM + consN #+ consP

            self.vars["tK"] = tK
            self.vars["tL"] = tL
            self.vars["tM"] = tM
            self.vars["tN"] = tN
            #self.vars["tP"] = tP

        self.cons = cons

    def build_con_new(self):
        cons = []
        lam, Q, P = self.get_vars(which="main")
        self.c_e = None

        cons += [lam >= 0]
        cons += [self._posdef(Q)]
        cons += [self._posdef(P)]

        A, B, C, D = self.get_vars(which="mats")
        nx, nu, nw, _, nz = self.get_dims()    

        # ------ Build C_e ------------------------------ #
        kappa_A = np.linalg.norm(self.c_a, 2)
        kappa_B = np.linalg.norm(self.c_b, 2)
        c_y     = np.linalg.norm(self.mats["Cy"], 2)

        tY_hi, tM_hi, tN_hi, tX_hi = 0.9, 0.13, 0.36, 2.8e5

        e11 = kappa_A * tY_hi + kappa_B * tM_hi 
        e12 = kappa_A + kappa_B * tN_hi  * c_y
        e22 = tX_hi  * kappa_A

        gamma_E = e11**2 + e12**2 + e22**2
        beta_E_np = gamma_E * self._I(2*self.c.shape[0])
        #self.c_e = cp.Constant(beta_E_np)
        # ----------------------------------------------- #


        U_R, _, w_R = self._residual_anisotropy_weights
        U_R = np.block([[U_R, np.zeros_like(U_R)], [np.zeros_like(U_R), U_R]])
        w_R = np.maximum(w_R, self.eps)

        self.beta_a = self.beta * np.sqrt(nx/(nx+nu))
        self.beta_b = self.beta * np.sqrt(nu/(nx+nu))

        beta_A = np.asarray(self.beta_a * w_R, dtype=float)
        beta_B = np.asarray(self.beta_b * w_R, dtype=float)

        self.beta_A = cp.Parameter(nx, value=beta_A)
        self.beta_B = cp.Parameter(nx, value=beta_B)

        self.beta = cp.diag(cp.hstack([self.beta_A, self.beta_B]))
        self.beta_E = U_R @ self.beta @ U_R.T if self.c_e is None else self.c_e

        self.s_A = cp.Variable(nx, name="s_a")
        self.s_B = cp.Variable(nx, name="s_b")
        S = cp.diag(cp.hstack([self.s_A, self.s_B]))


        self.sigma_s = cp.Variable(name="sigma_s")
        self.sigma_p = cp.Variable(name="sigma_p")


        Is = self._I(S.shape[0])
        blkS = self.sigma_s * Is - S
        cons += [self._posdef(blkS)] + [self._posdef(S)]

        Np = P.shape[0]
        Ip = self._I(Np)
        blkP = cp.bmat([
            [self.sigma_p * Ip, Ip  ], 
            [Ip,                P   ],
        ])
        cons += [self._posdef(blkP)]


        G = cp.Variable((Np, Np), name="G")
        blkG = cp.bmat([
            [G,     Ip  ], 
            [Ip,    P   ],
        ])
        cons += [self._posdef(blkG)] + [self._posdef(G)]

        H = cp.Variable((Np, Np), name="H")
        blkH = cp.bmat([
            [H, G],
            [G, S],
        ])
        cons += [self._posdef(blkH)]  + [self._posdef(H)]

        Ps = cp.Variable((Np, Np), name="Ps")
        blkPs = cp.bmat([   #G + H - inv(Ps)
            [G + H, Ip],
            [Ip,    Ps],
        ])
        cons += [self._posdef(blkPs)] + [self._posdef(Ps)]


        P_bar = P - (self.sigma_s + self.sigma_p) * self.beta_E
        cons += [self._posdef(P_bar)] 
        

        # NOTE: I switched Ps and P_bar position, I did the calculus for P_bar in (1,1) and Ps in (2,2)-(4,4) but MOSEK doen't break if those are switched

        if self.model == "correlated": 
            blk = cp.bmat([
                [-P_bar,                   self._Z(2*nx, nw),  self._Z(2*nx, nw),      A.T,                C.T                 ],
                [ self._Z(nw, 2*nx),   -lam*self._I(nw),    lam*self._I(nw),        B.T,                D.T                 ],
                [ self._Z(nw, 2*nx),    lam*self._I(nw),   -Q - lam*self._I(nw),    self._Z(nw, 2*nx),  self._Z(nw, nz)     ],
                [  A,                   B,                  self._Z(2*nx, nw),     -Ps,              self._Z(2*nx, nz)   ],
                [  C,                   D,                  self._Z(nz, nw),        self._Z(nz, 2*nx), -self._I(nz)         ],
            ])

            cons += [self._negdef(blk, which="strict")]

        elif self.model == "independent":
            blk1 = cp.bmat([
                [-P_bar,       A.T,                C.T                 ],
                [ A,       -Ps,              self._Z(2*nx, nz)   ],
                [ C,        self._Z(nz, 2*nx), -self._I(nz)         ],
            ])

            blk2 = cp.bmat([
                [-lam*self._I(nw),  lam*self._I(nw),    B.T,                D.T                 ],
                [ lam*self._I(nw), -Q-lam*self._I(nw),  self._Z(nw, 2*nx),  self._Z(nw, nz)     ],
                [ B,                self._Z(2*nx, nw), -Ps,                  self._Z(2*nx, nz)   ],
                [ D,                self._Z(nz, nw),    self._Z(nz, 2*nx), -self._I(nz)         ],
            ]) 

            cons += [self._negdef(blk1, which="strict")]
            cons += [self._negdef(blk2, which="strict")]



        if self.reg_fro:
            K, L, Y, X, M, N = self.get_vars(which="inner")
            tK, consK, consk = self.spectral_norm_epigraph(K, "K")
            tL, consL, consl = self.spectral_norm_epigraph(L, "L")
            tY, consY, consy = self.spectral_norm_epigraph(Y, "Y")
            tX, consX, consx = self.spectral_norm_epigraph(X, "X")
            tM, consM, consm = self.spectral_norm_epigraph(M, "M")
            tN, consN, consn = self.spectral_norm_epigraph(N, "N")

            cons += consK + consL + consX + consY + consM + consN

            self.vars["tK"] = tK
            self.vars["tL"] = tL
            self.vars["tX"] = tX
            self.vars["tY"] = tY
            self.vars["tM"] = tM
            self.vars["tN"] = tN

            """self.c_e = cp.mat([
                [self.c_a * tY + self.c_b * tM, self.c_a + tN * self.c_b @ self.mats["Cy"]],
                [self._Z(self.c_a.shape[0]),    tX * self.c_a]
            ])"""

        self.cons = cons

    def build_obj(self): 
        lam, Q, _ = self.get_vars(which="main")
        obj_dro = cp.trace(Q @ self.Sigma_nom) + lam * (self.gamma ** 2)


        self.obj = obj_dro

    def build_reg(self): 
        self.reg_t= self.reg_a = self.reg_b = cp.Parameter(value=0.0)

        if self.reg_fro: 
            tK, tL, tY, tX, tM, tN = self.get_vars(which="t")
            self.reg_t = tK + tL + tY + tX + tM + tN
        
        if self.reg_beta and not self.new: 
            self.reg_a = cp.sum(self.s_A) + cp.sum(self.tau_A / (self.beta_A + self.eps))
            self.reg_b = cp.sum(self.s_B) + cp.sum(self.tau_B / (self.beta_B + self.eps))


        self.reg = self.rho * (self.reg_t + self.reg_a + self.reg_b) + self.eps


    # ============================================================================ #

    def solve_prb(self):
        obj = cp.Minimize(self.obj + self.reg)
        prob = cp.Problem(obj, self.cons)

        success_MOSEK = success_SCS = False
        solver = "MOSEK"
        print("\n===================================================\nAttempting to solve with MOSEK...")
        try:
            prob.solve(solver=cp.MOSEK, verbose=True, mosek_params={
                'MSK_DPAR_INTPNT_CO_TOL_PFEAS': 1e-7,
                'MSK_DPAR_INTPNT_CO_TOL_DFEAS': 1e-7,
                'MSK_DPAR_INTPNT_CO_TOL_REL_GAP': 1e-7,
                'MSK_DPAR_INTPNT_TOL_STEP_SIZE': 1e-6,
                'MSK_IPAR_INTPNT_SCALING': 1, # 0: no scaling, 1: geometric mean, 2: equilibrate
            })
            print(f"MOSEK status: {prob.status}")
            if prob.status == cp.OPTIMAL:
                success_MOSEK = True
        except Exception as mosek_e:
            print(f"MOSEK error: {mosek_e}")

        if not success_MOSEK:
            solver = "SCS"
            print("\n===================================================\nCVXOPT failed, trying SCS...")
            try:
                prob.solve(solver=cp.SCS, verbose=True, eps=1e-4, max_iters=10000)
                print(f"SCS status: {prob.status}")
                if prob.status in [cp.OPTIMAL, cp.OPTIMAL_INACCURATE]:
                    success_SCS = True
                    if prob.status == cp.OPTIMAL_INACCURATE:
                        print("Warning: SCS returned 'optimal_inaccurate'.")
                else:
                    print(f"SCS failed with status: {prob.status}")
            except Exception as scs_e:
                print(f"SCS error: {scs_e}")

        if success_MOSEK or success_SCS:
            print(f"Solve succeeded ({solver}) with value:", prob.value)
        else:
            print("Optimization error: All solvers failed.")

        self.solver = solver
        self.status = prob.status
        self.value = prob.value


        self.total_constraints = len(self.cons)
        self.violations = 0
        self.violation_values = []  # optional, if you want to keep the actual numbers

        try:
            for c in self.cons:
                v = float(c.violation())   # CVXPY residual for this constraint
                self.violation_values.append(v)
                print(c, "violation:", v)

                # count as violation only if it's larger than tolerance
                if v > 1e-6:
                    self.violations += 1
        except Exception as e: 
            pass

        print(f"total_constraints: {self.total_constraints}, num violations: {self.violations}")
        print(f"Objective value: {self.value}")
        print(f"Q: {self.vars["Q"].value}")

        if self.inp: input("Waiting...")


    def pack_outs(self): 
        lam, Q, P = self.get_vars(which="main")
        K, L, Y, X, M, N = self.get_vars(which="inner")
        A, B, C, D = self.get_vars(which="mats")


        P_val, Q_val, lam_val \
            = self._val(P.value), self._val(Q.value), self._val(lam.value)
        K_val, L_val, M_val, N_val, X_val, Y_val \
            = self._val(K.value), self._val(L.value), self._val(M.value), self._val(N.value), self._val(X.value), self._val(Y.value)
        A_val, B_val, C_val, D_val \
            = self._val(A.value), self._val(B.value), self._val(C.value), self._val(D.value)

        # Results -------------------------
        dro = DROLMIResult(
            solver=self.solver,
            status=self.status,
            obj_value=float(self.value),
            gamma=self.gamma,
            lambda_opt=lam_val,
            Q=Q_val, X=X_val, Y=Y_val, K=K_val, L=L_val, M=M_val, N=N_val,
            Pbar=P_val, Abar=A_val, Bbar=B_val, Cbar=C_val, Dbar=D_val, 
            Tp=None, P=None,
        )

        self.outs = dro

        if self.reg_beta or self.new:
            Ax, Bu, Cy = self.mats["Ax"], self.mats["Bu"], self.mats["Cy"]
            DeltaA, DeltaB, EAA, EAB = recover_deltas(
                P=P_val, X=X_val, Y=Y_val, M=M_val, N=N_val, Cy=Cy,
                Ahat=Ax, Buhat=Bu,
                beta_AA=np.mean(self.beta_A.value), beta_AB=np.mean(self.beta_B.value),
            )

            self.mats["Ax"] = Ax + DeltaA
            self.mats["Bu"] = Bu + DeltaB
            
            D = (DeltaA, DeltaB)
            E = (EAA, EAB)
            B = (self.beta, self.beta_a, self.beta_b, self.beta_A.value, self.beta_B.value)
            S = (self.s_A.value, self.s_B.value)
            T = None if self.new else (self.tau_A.value, self.tau_B.value)
            R = (self.obj.value, self.reg.value, self.reg_t.value, self.reg_a.value, self.reg_b.value)

            self.others = D, E, B, S, T, R
        else: 
            self.others = 0.0


    # ============================================================================ #
