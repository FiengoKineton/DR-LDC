import sys
import numpy as np, cvxpy as cp

from core import MatricesAPI, recover_deltas
from utils import (
    DROLMIResult, Noise, YoungDROConfig,
    I, Z, negdef, _val, _safe_scalar, _print_header, _print_scale_dict,
)

from .SimEstm import Data_Estimator_and_Simulator
from .Solvers import SolverManager


# =============================================================================================== #

class Young_dro_lmi:
    """
    Solver for the data-driven Young-based DRO-LMI synthesis problem.

    Responsibilities:
    - simulate/load datasets
    - estimate state, disturbance, and output matrices
    - build the DRO-LMI optimization problem
    - solve and package the results

    Public methods can be called independently to inspect intermediate quantities.
    """

    def __init__(
        self,
        api: MatricesAPI,
        vals: tuple,
        noise: Noise,
        config: YoungDROConfig | None = None,
    ):
        self.api = api
        self.vals = vals
        self.noise = noise
        self.cfg = config or YoungDROConfig()

        self.gamma = noise.gamma
        self.var = noise.var
        _, _, self.plot = vals

        self.estimator = Data_Estimator_and_Simulator(
            api=api,
            vals=vals,
            noise=noise,
            cfg=self.cfg,
        )


        # raw data
        self.datasets = None
        self.avg = None
        self.x = None
        self.u = None
        self.y = None
        self.z = None
        self.x_next = None

        # dimensions
        self.T = None
        self.nx = None
        self.nu = None
        self.ny = None
        self.nz = None
        self.nw = None

        # estimated matrices
        self.Ax = None
        self.Bu = None
        self.Bw = None
        self.Cy = None
        self.Dyw = None
        self.Cz = None
        self.Dzu = None
        self.Dzw = None
        self.Sigma_nom = None
        self.R = None
        self.w = None
        self.beta = None
        self.beta_a = None
        self.beta_b = None

        # cvxpy objects
        self.adds = self.vars = self.expr = {}
        self.cons = []
        self.problem = None

        # outputs
        self.result = None
        self.other = None
        self.violations = None

    # -------------------------------------------------------------------------------------- #

    def _sync_from_estimator(self):
        """
        Copy relevant estimated/simulated quantities from the estimator
        into the solver namespace, so old solver code can still use self.Ax,
        self.Bu, self.Bw, etc.
        """
        est = self.estimator
        est.eval()  # ensure all estimates are up to date

        self.datasets = est.datasets
        self.avg = est.avg

        self.x = est.x
        self.u = est.u
        self.y = est.y
        self.z = est.z
        self.x_next = est.x_next

        self.T = est.T
        self.nx = est.nx
        self.nu = est.nu
        self.ny = est.ny
        self.nz = est.nz
        self.nw = est.nw

        self.Ax = est.Ax
        self.Bu = est.Bu
        self.Bw = est.Bw

        self.Cy = est.Cy
        self.Dyw = est.Dyw
        self.Cz = est.Cz
        self.Dzu = est.Dzu
        self.Dzw = est.Dzw

        self.R = est.R
        self.w = est.w
        self.Sigma_nom = est.Sigma_nom

        self.beta = est.beta
        self.beta_a = est.beta_a
        self.beta_b = est.beta_b
        self.gamma = est.gamma
        self.var = est.var

        # ================= DEBUG PRINTS =================
        print("\n\n\n========== SYNC FROM ESTIMATOR ==========")

        print("\n--- Dimensions ---")
        print(f"T  : {self.T}")
        print(f"nx : {self.nx}, nu : {self.nu}, ny : {self.ny}, nz : {self.nz}, nw : {self.nw}")

        print("\n--- Data shapes ---")
        print(f"x        : {None if self.x is None else self.x.shape}")
        print(f"x_next   : {None if self.x_next is None else self.x_next.shape}")
        print(f"u        : {None if self.u is None else self.u.shape}")
        print(f"y        : {None if self.y is None else self.y.shape}")
        print(f"z        : {None if self.z is None else self.z.shape}")

        print("\n--- System matrices ---")
        print(f"Ax : {None if self.Ax is None else self.Ax.shape}")
        print(f"Bu : {None if self.Bu is None else self.Bu.shape}")
        print(f"Bw : {None if self.Bw is None else self.Bw.shape}")

        print(f"Cy  : {None if self.Cy is None else self.Cy.shape}")
        print(f"Dyw : {None if self.Dyw is None else self.Dyw.shape}")
        print(f"Cz  : {None if self.Cz is None else self.Cz.shape}")
        print(f"Dzu : {None if self.Dzu is None else self.Dzu.shape}")
        print(f"Dzw : {None if self.Dzw is None else self.Dzw.shape}")

        print("\n--- Noise / DRO ---")
        print(f"R          : {None if self.R is None else self.R.shape}")
        print(f"w          : {None if self.w is None else self.w.shape}")
        print(f"Sigma_nom  : {None if self.Sigma_nom is None else self.Sigma_nom.shape}")

        print("\n--- Scalars ---")
        print(f"beta   : {self.beta}")
        print(f"beta_a : {self.beta_a}")
        print(f"beta_b : {self.beta_b}")
        print(f"gamma  : {self.gamma}")
        print(f"var    : {self.var}")

        print("=========================================\n")

    # -------------------------------------------------------------------------------------- #

    def _spectral_norm_epigraph(self, A: cp.Expression, name: str):
        m, n = A.shape
        t = cp.Variable(nonneg=True, name=f"t_{name}")
        blk = cp.bmat([
            [t * I(m), A],
            [A.T,      t * I(n)]
        ])
        return t, [blk >> 0]

    def _postprocess(self, solver_name: str):
        violation_values = []
        violations = 0
        for c in self.cons:
            v = float(c.violation())
            violation_values.append(v)
            if v > 1e-6:
                violations += 1

        self.violations = (violations, len(self.cons))

        lam = self.vars["lam"]
        Q = self.vars["Q"]
        X = self.vars["X"]
        Y = self.vars["Y"]
        K = self.vars["K"]
        L = self.vars["L"]
        M = self.vars["M"]
        N = self.vars["N"]

        P = self.expr["P"]
        A = self.expr["A"]
        B = self.expr["B"]
        C = self.expr["C"]
        D = self.expr["D"]

        self.result = DROLMIResult(
            solver=solver_name,
            status=self.problem.status,
            obj_value=float(self.problem.value) if self.problem.value is not None else np.inf,
            gamma=self.gamma,
            lambda_opt=_val(lam.value),
            Q=_val(Q.value),
            X=_val(X.value),
            Y=_val(Y.value),
            K=_val(K.value),
            L=_val(L.value),
            M=_val(M.value),
            N=_val(N.value),
            Pbar=_val(P.value),
            Abar=_val(A.value),
            Bbar=_val(B.value),
            Cbar=_val(C.value),
            Dbar=_val(D.value),
            Tp=None,
            P=None,
        )

    def _build_return_tuple(self):
        """
        Build the final return tuple with the exact same format
        as the original Young_dro_lmi function.
        """

        P_val = _val(self.expr["P"].value)
        M_val = _val(self.vars["M"].value)
        N_val = _val(self.vars["N"].value)
        X_val = _val(self.vars["X"].value)
        Y_val = _val(self.vars["Y"].value)
        dro = self.result

        DeltaA, DeltaB, EAA, EAB = recover_deltas(
            P=P_val,
            X=X_val,
            Y=Y_val,
            M=M_val,
            N=N_val,
            Cy=self.Cy,
            Ahat=self.Ax,
            Buhat=self.Bu,
            beta_AA=np.mean(self.adds["beta_AA"].value) if np.ndim(self.adds["beta_AA"].value) > 0 else self.adds["beta_AA"].value,
            beta_AB=self.adds["beta_AB"].value,
        )

        Ax_rec = self.Ax + DeltaA
        Bu_rec = self.Bu + DeltaB

        P_tuple = (
            Ax_rec,
            self.Bw,
            Bu_rec,
            self.Cy,
            self.Dyw,
            self.Cz,
            self.Dzw,
            self.Dzu,
        )

        other = (
            (DeltaA, DeltaB),
            (EAA, EAB),
            (
                self.adds["beta"],
                self.adds["beta_a"],
                self.adds["beta_b"],
                self.adds["beta_AA"].value,
                self.adds["beta_ab"],
            ),
            (self.adds["s_AA"].value, self.adds["s_AB"].value),
            (self.adds["tau_AA"].value, self.adds["tau_AB"].value),
            (self.expr["obj_dro"].value, self.expr["reg"].value),
        )

        return dro, P_tuple, self.Sigma_nom, other, self.violations

    # -------------------------------------------------------------------------------------- #

    def build_problem(self):
        if self.Cy is None:
            self.estimate_output_mats()

        nx, nu, ny, nz, nw = self.nx, self.nu, self.ny, self.nz, self.nw
        gamma = self.gamma
        eps = self.cfg.eps
        mu = self.cfg.mu
        model = self.cfg.model
        approach = self.cfg.approach

        lam = cp.Variable(nonneg=True, name="lambda")
        Q = cp.Variable((nw, nw), PSD=True, name="Q")

        X = cp.Variable((nx, nx), symmetric=True, name="X")
        Y = cp.Variable((nx, nx), symmetric=True, name="Y")
        K = cp.Variable((nx, nx), name="K")
        L = cp.Variable((nx, ny), name="L")
        M = cp.Variable((nu, nx), name="M")
        N = cp.Variable((nu, ny), name="N")

        Ix = I(nx)
        Iw = I(nw)
        Iz = I(nz)

        P = cp.bmat([[Y, Ix], [Ix, X]])

        A = cp.bmat([
            [self.Ax @ Y + self.Bu @ M,       self.Ax + self.Bu @ N @ self.Cy],
            [K,                              X @ self.Ax + L @ self.Cy]
        ])
        B = cp.bmat([
            [self.Bw + self.Bu @ N @ self.Dyw],
            [X @ self.Bw + L @ self.Dyw]
        ])
        C = cp.bmat([
            [self.Cz @ Y + self.Dzu @ M,     self.Cz + self.Dzu @ N @ self.Cy]
        ])
        D = self.Dzw + self.Dzu @ N @ self.Dyw

        cons = [lam >= 0, Q >> 0]
        if model.lower() in ["correlated", "corr", "1"]:
            cons += [P >> 0]
        else:
            cons += [P >> eps * I(2 * nx)]

        obj_dro = cp.trace(Q @ self.Sigma_nom) + lam * (gamma ** 2)
        reg = 0.0


        Cy_norm = np.linalg.norm(self.Cy, 2)
        if model.lower() in ["correlated", "corr", "1"]:
            M_norm, N_norm, Y_norm, X_norm = 0.1, 0.27, 0.76, 2e5
        else: 
            M_norm, N_norm, Y_norm, X_norm = 0.15, 0.2, 1.0, 8e3 #3.0
        
        

        if approach == "Young":
            beta_aa, beta_ab = np.sqrt(1 + X_norm**2 + Y_norm**2) * self.beta_a, np.sqrt(M_norm**2 + N_norm**2 * Cy_norm**2) * self.beta_b
            print(f"Beta: {self.beta}\nComputed beta_a: {self.beta_a}, beta_b: {self.beta_b} \nComputed beta_aa: {beta_aa}, beta_ab: {beta_ab}")

            beta_AA = cp.Parameter(nonneg=True, value=float(np.clip(beta_aa, 0.0, 1e3)))
            beta_AB = cp.Parameter(nonneg=True, value=float(np.clip(beta_ab, 0.0, 1e3)))

            tau_AA = cp.Variable(nonneg=True, name="tau_aa")
            s_AA = cp.Variable(nonneg=True, name="s_aa") 
            S_AA = np.hstack([Ix, Z(nx, nx)]).T
            tau_AB = cp.Variable(nonneg=True, name="tau_ab")
            s_AB = cp.Variable(nonneg=True, name="s_ab") 
            S_AB = np.hstack([Ix, Z(nx, nx)]).T
            
            tK, consK = self._spectral_norm_epigraph(K, "K")   # usa I_nx e I_nx
            tL, consL = self._spectral_norm_epigraph(L, "L")   # usa I_nx e I_ny
            tM, consM = self._spectral_norm_epigraph(M, "M")   # usa I_nu e I_nx
            tN, consN = self._spectral_norm_epigraph(N, "N")   # usa I_nu e I_ny
            tP, consP = self._spectral_norm_epigraph(P, "P") 

            # Reg
            mhu_AA = mhu_AB = mu
            rhoK = rhoL = rhoM = rhoN = mu #/10
            rhoP = mu*0
            reg += mhu_AA * (s_AA + tau_AA / beta_aa**2) + mhu_AB * (s_AB + tau_AB / beta_ab**2)
            reg += rhoK * tK + rhoL * tL + rhoM * tM + rhoN * tN
            reg += rhoP * tP * (beta_aa + beta_ab)**2

            # Cons
            cons += [s_AA >= 1e-9, s_AB >= 1e-9]
            cons += [tau_AA <= 1e3, tau_AB <= 1e3]
            cons += [cp.bmat([[s_AA, beta_AA], [beta_AA, tau_AA]]) >> 0]
            cons += [cp.bmat([[s_AB, beta_AB], [beta_AB, tau_AB]]) >> 0]
            cons += consK + consL + consM + consN
            cons += consP

            # Final
            state_blk = -P + (tau_AA + tau_AB) * I(2*nx)
            young_blk = -s_AA * Ix
        
        elif approach == "Mats":
            # 1) residual anisotropy (directions and weights)
            U_A, _, w_A = self.estimator._residual_anisotropy_weights(self.R, floor=1e-12, mode="sqrt")
            # Optional: cap tiny directions to avoid numerical issues
            w_A = np.maximum(w_A, 1e-6)
            w_B = np.mean(w_A)

            # 2) per-direction β for the A-part: scale β_a with weights
            beta_AA_dir_np = np.asarray(self.beta_a * w_A, dtype=float)  # shape (nx,)

            # 3) rotate the selector S_AA into U_A basis so each slack acts along an eigendirection
            #    Original S_AA had shape (2nx x nx) with block [I; 0]. Keep the same but rotate columns.
            S_AA_base = np.hstack([Ix, Z(nx, nx)]).T               # (2nx x nx)
            S_AA  = S_AA_base @ U_A                            # (2nx x nx)

            # 4) replace scalar slacks by per-direction vectors
            tau_AA = cp.Variable(nx, nonneg=True, name="tau_aa_vec")
            s_AA   = cp.Variable(nx, nonneg=True, name="s_aa_vec")

            # parameters for β per direction
            beta_AA = cp.Parameter(nx, nonneg=True, value=beta_AA_dir_np)


            #Cy_norm, M_norm, N_norm, X_norm, Y_norm = np.linalg.norm(self.Cy, 2), 0.15, 0.6, 3.0, 1.0 # 2.5e5, 1.0
            beta_ab = w_B * self.beta_b #np.sqrt(M_norm**2 + N_norm**2 * Cy_norm**2) * beta_b
            beta_AB = cp.Parameter(nonneg=True, value=float(np.clip(beta_ab, 0.0, 1e3)))

            # 5) epigraphs for each direction:
            #    For every i, [[s_i, β_i],[β_i, τ_i]] >= 0
            for i in range(nx):
                if model.lower() in ["correlated", "corr", "1"]:    cons += [s_AA[i] >= 1e-9]
                else:                                               cons += [s_AA[i] >= 1e-9, tau_AA[i] <= 1e3]
                cons += [cp.bmat([[s_AA[i],      beta_AA[i]], [beta_AA[i],   tau_AA[i]]]) >> 0]

            # 6) keep AB part as-is for now (still isotropic), or do the same for AB if you want
            tau_AB = cp.Variable(nonneg=True, name="tau_ab")  # unchanged
            s_AB   = cp.Variable(nonneg=True, name="s_ab")
            S_AB   = S_AA_base                                # you can also rotate with a basis for AB if desired


            # 7) update regularization terms (small nudges to keep slacks bounded)
            mhu_AA = mhu_AB = rhoK = rhoL = rhoM = rhoN = mu

            reg += mhu_AA * (cp.sum(s_AA) + cp.sum(tau_AA / (beta_AA**2 + 1e-18)))
            reg += mhu_AB * (s_AB + tau_AB / (beta_ab**2 + 1e-18))

            # 8) state block: minimally invasive variant (keeps your scalar-identity structure)
            state_blk = -P + (cp.sum(tau_AA) + tau_AB) * I(2*nx)
            young_blk = -cp.diag(s_AA)

            # 9) use the rotated selector in the big LMI wherever S_AA appears
            if model.lower() in ["correlated", "corr", "1"]:    cons += [s_AB >= 1e-9]
            else:                                               cons += [s_AB >= 1e-9, tau_AB <= 1e3]
            cons += [cp.bmat([[s_AB, beta_AB], [beta_AB, tau_AB]]) >> 0]


            # 10) matrices
            tK, consK = self._spectral_norm_epigraph(K, "K")   # usa I_nx e I_nx
            tL, consL = self._spectral_norm_epigraph(L, "L")   # usa I_nx e I_ny
            tM, consM = self._spectral_norm_epigraph(M, "M")   # usa I_nu e I_nx
            tN, consN = self._spectral_norm_epigraph(N, "N")   # usa I_nu e I_ny
            #tP, consP = self._spectral_norm_epigraph(P, "P")   # usa I_nu e I_ny

            reg += rhoK * tK + rhoL * tL + rhoM * tM + rhoN * tN #+ mu * tP * (cp.sum(beta_AA) + beta_AB)**2

            cons += consK + consL + consM + consN     # += consP
        
        else:
            raise ValueError("approach must be 'Young' or 'Mats'.")


        obj = obj_dro + reg

        if model.lower() in ["correlated", "corr", "1"]:
            big_corr = cp.bmat([
                [ -P,            Z(2*nx, nw),  Z(2*nx, nw),  A.T,          C.T,         Z(2*nx, nx),    Z(2*nx, nx)     ],
                [ Z(nw,2*nx),   -lam*Iw,       lam*Iw,       B.T,          D.T,         Z(nw,   nx),    Z(nw,   nx)     ],
                [ Z(nw,2*nx),    lam*Iw,      -Q - lam*Iw,   Z(nw,2*nx),   Z(nw,  nz),  Z(nw,   nx),    Z(nw,   nx)     ],
                [ A,             B,            Z(2*nx, nw),  state_blk,    Z(2*nx,nz),  S_AA,           S_AB            ],
                [ C,             D,            Z(nz,  nw),   Z(nz, 2*nx), -Iz,          Z(nz,   nx),    Z(nz,   nx)     ],
                [ Z(nx,2*nx),    Z(nx,  nw),   Z(nx,  nw),   S_AA.T,       Z(nx,  nz),  young_blk,      Z(nx,   nx)     ],
                [ Z(nx,2*nx),    Z(nx,  nw),   Z(nx,  nw),   S_AB.T,       Z(nx,  nz),  Z(nx, nx),     -s_AB * Ix       ],
            ])
            cons += [negdef(state_blk, eps)]

            cons += [negdef(big_corr, eps)]

        elif model.lower() in ["independent", "indep", "2"]:
            blk1 = cp.bmat([
                [ -P,           A.T,          C.T,          Z(2*nx, nx),    Z(2*nx, nx)     ],
                [  A,           state_blk,    Z(2*nx, nz),  S_AA,           S_AB            ],
                [  C,           Z(nz, 2*nx), -Iz,           Z(nz, nx),      Z(nz, nx)       ],
                [  Z(nx,2*nx),  S_AA.T,       Z(nx, nz),    young_blk,      Z(nx,   nx)     ],
                [  Z(nx,2*nx),  S_AB.T,       Z(nx, nz),    Z(nx,   nx),   -s_AB * Ix       ],
            ])       
            cons += [negdef(state_blk, eps)]     
            
            blk2 = cp.bmat([
                [-lam*Iw,   lam*Iw,         B.T,            D.T         ],
                [ lam*Iw,  -Q - lam*Iw,     Z(nw, 2*nx),    Z(nw, nz)   ],
                [ B,        Z(2*nx, nw),   -P,              Z(2*nx, nz) ],
                [ D,        Z(nz, nw),      Z(nz, 2*nx),   -Iz          ],
            ])  # Tot size: (2nx + 2nw + nz) x (2nx + 2nw + nz)

            cons += [negdef(blk1, eps)]
            cons += [negdef(blk2, eps)]

        else:
            raise ValueError("model must be 'correlated' or 'independent'.")


        obj = cp.Minimize(obj)
        self.problem = cp.Problem(obj, cons)

        self.adds = {
            "beta_AA": beta_AA,
            "beta_AB": beta_AB,
            "beta_a": self.beta_a,
            "beta_b": self.beta_b,
            "beta": self.beta,
            "beta_ab": beta_ab,
            "s_AA": s_AA,
            "s_AB": s_AB,
            "tau_AA": tau_AA,
            "tau_AB": tau_AB,
        }
        self.vars = {
            "lam": lam, "Q": Q, "X": X, "Y": Y, "K": K, "L": L, "M": M, "N": N
        }
        self.expr = {
            "P": P, "A": A, "B": B, "C": C, "D": D,
            "obj_dro": obj_dro, "reg": reg,
        }
        self.cons = cons

        return self.problem

    def solve(self):
        if self.problem is None:
            self.build_problem()

        solver = SolverManager(
            solver_order=self.cfg.solver_order,
            verbose=self.cfg.verbose
        )

        sol = solver.solve(self.problem)

        used_solver = sol["solver"]
        self.problem = sol["problem"]

        if not sol["success"]:
            print("Optimization error: all solvers failed.")

        self._postprocess(used_solver)
        print(f"\nSolver: {used_solver}, Status: {self.problem.status}, Objective value: {self.problem.value}")
        #input("Press Enter to continue...")

        return self.result

    # -------------------------------------------------------------------------------------- #

    def run(self):
        self._sync_from_estimator()  # ensure we have the latest estimates before building the problem
        self.build_problem()
        self.solve()
        return self._build_return_tuple()

# =============================================================================================== #