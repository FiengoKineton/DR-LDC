import sys
import numpy as np, casadi as ca

from utils import DROLMIResult
from ._dro_utils import Constructor


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
