import sys
import numpy as np, cvxpy as cp

from core import MatricesAPI, recover_deltas
from disturbances import Disturbances
from simulate import Open_Loop
from utils import DROLMIResult, Noise


# =============================================================================================== #

def Young_dro_lmi(
    api: MatricesAPI,
    #data: Data,
    vals: tuple,
    noise: Noise,
    model: str = "correlated",  # or "independent"
    eps: float = 1e-5,
    mu: float = 1e-3,  # keep K, L, M, N from exploding
    approach: str = "DeePC",
    real_Z_mats: bool = False,
    N_sims: int = 1,
) -> DROLMIResult:
    """
    Builds and solves the DRO-LMI you specified.
    model = "correlated"  implements (1)
    model = "independent" implements (2)
    """
    gamma, var = noise.gamma, noise.var
    _, _, plot = vals

    op = Open_Loop(MAKE_DATA=False, EVAL_FROM_PATH=False, DATASETS=True, N=N_sims)
    datasets = op.datasets


    from analysis import select_representative_run
    avg = select_representative_run(datasets) if N_sims!=1 else datasets
    x, u, y, z, x_next = avg["X"], avg["U"], avg["Y"], avg["Z"], avg["X_next"]

    if plot:
        from analysis import plot_first3_and_mean
        plot_first3_and_mean(datasets, out=avg, key="X", title_prefix="Closed-loop")
        plot_first3_and_mean(datasets, out=avg, key="U", title_prefix="Closed-loop")
        plot_first3_and_mean(datasets, out=avg, key="Y", title_prefix="Closed-loop")
        plot_first3_and_mean(datasets, out=avg, key="Z", title_prefix="Closed-loop")

    T, nx, nu, ny, nz = x.shape[1], x.shape[0], u.shape[0], y.shape[0], z.shape[0]
    
    def _pseudo_inv(D, r=1e-6):
        return D.T @ np.linalg.inv(D @ D.T + r * np.eye(D.shape[0]))
    
    def I(n):
        return np.eye(n)

    def Z(r, c): 
        return np.zeros((r, c)) 

    def negdef(M): 
        return (M << -eps * np.eye(M.shape[0]))

    def _val(x):
        if x is None:
            return None
        return float(x) if np.isscalar(x) else x

    def _residual_anisotropy_weights(R, *, floor=1e-12, mode="sqrt"):
        """
        From residuals R (nx x T), return (U, s, w) where:
        - S = (R R^T)/T = U diag(s) U^T, s sorted desc
        - w_i are directional weights (proportional to sqrt(s_i) by default)
        """
        nx, T = R.shape
        S = (R @ R.T) / max(T, 1)
        S = 0.5 * (S + S.T) + floor * np.eye(nx)
        s_vals, U = np.linalg.eigh(S)          # ascending
        idx = np.argsort(s_vals)[::-1]         # descending
        s = np.clip(s_vals[idx], 0.0, None)
        U = U[:, idx]
        if mode == "sqrt":
            w = np.sqrt(s)
        elif mode == "linear":
            w = s
        else:
            raise ValueError("mode must be 'sqrt' or 'linear'")
        # Normalize so max weight is 1 (keeps scales comparable to scalar β logic)
        w = w / max(np.max(w), 1e-18)
        return U, s, w

    def estimate_Bw_from_residuals(
        Ax_hat, Bu_hat,
        eta=0.95,
        eps=1e-12,
        mode="default",          # "white" | "factor" | "known_cov"
        Sigma_w_known=None     # only used if mode == "known_cov"
    ):
        """
        Returns:
            Bw_hat, Sigma_w_hat, info
        where:
            - mode == "white":   Sigma_w_hat = I, Bw_hat absorbs the scaling
            - mode == "factor":  Bw_hat @ sqrtm(Sigma_w_hat) equals the principal factor of S_R
            - mode == "known_cov": Sigma_w_hat = Sigma_w_known, Bw_hat rescales accordingly
        """

        # Residuals R = x_{k+1} - Ax x_k - Bu u_k
        R = x_next - (Ax_hat @ x + Bu_hat @ u)              # (nx, T)
        nx, T = R.shape

        # Empirical residual covariance S_R = (1/(T)) R R^T
        # Use T > 0 guard, SPD symmetrization, and tiny ridge for numerics
        denom = max(T, 1)
        S = (R @ R.T) / denom
        S = 0.5 * (S + S.T) + eps * np.eye(nx)

        # Spectral decomposition
        # (S is SPD-ish; SVD is fine, eigh is a hair faster and numerically cleaner)
        s_vals, U = np.linalg.eigh(S)                       # s_vals ascending
        s_vals = np.clip(s_vals, 0.0, None)
        order = np.argsort(s_vals)[::-1]                    # descending
        s = s_vals[order]
        U = U[:, order]

        # Rank selection by cumulative energy
        total = max(float(np.sum(s)), 1e-18)
        print(f"Total residual energy: {total}")
        cum = np.cumsum(s) / total
        nw = int(np.clip(np.searchsorted(cum, eta) + 1, 1, nx))

        # Principal block
        Up = U[:, :nw]
        sp = s[:nw]
        sp_sqrt = np.sqrt(sp)

        if mode == "default":
            R = x_next - (Ax_hat @ x + Bu_hat @ u)
            S = (R @ R.T) / max(T, 1)
            S = 0.5 * (S + S.T) + 1e-12 * np.eye(nx)
            U_s, s_s, _ = np.linalg.svd(S, full_matrices=False)
            cum = np.cumsum(s_s) / max(np.sum(s_s), 1e-18)
            nw = int(np.clip(np.searchsorted(cum, 0.95) + 1, 1, nx))
            Bw = U_s[:, :nw] @ np.diag(np.sqrt(s_s[:nw]))  
            return Bw, R, None, {"nw": nw, "energy": float(np.sum(s_s[:nw])) / float(np.sum(s_s)), "s_vals": s_s}
            
        elif mode == "white":
            # White-normalized convention: Sigma_w_hat = I, Bw absorbs scaling
            Bw_hat = Up @ np.diag(sp_sqrt)                  # (nx, nw)
            Sigma_w_hat = np.eye(nw)

        elif mode == "factor":
            # Keep factorization as Bw * Sigma_w^{1/2} := Up diag(sp^{1/2})
            # Choose a convenient split; by default put all scaling in Sigma_w^{1/2}
            Bw_hat = Up                                     # (nx, nw), orthonormal columns
            Sigma_w_hat = np.diag(sp)                       # so Bw_hat @ Sigma_w_hat @ Bw_hat.T = S (rank-nw approx)

            # If you prefer the literal half-split, uncomment:
            #Bw_hat = Up @ np.diag(sp_sqrt)
            # Sigma_w_hat = np.eye(nw)

        elif mode == "known_cov":
            if Sigma_w_known is None:
                raise ValueError("mode='known_cov' requires Sigma_w_known.")
            Sigma_w_known = np.atleast_2d(Sigma_w_known)
            if Sigma_w_known.shape != (nw, nw):
                # If user supplied full-size Sigma_w over-estimate, reduce to nw via top eigenspace
                # or complain. Here we try to be helpful: project to top-nw block.
                # Safer alternative: raise an error instead of silently projecting.
                raise ValueError(f"Sigma_w_known must be shape ({nw},{nw}).")

            # We want Bw_hat Sigma_w_known Bw_hat.T ≈ Up diag(sp) Up.T.
            # Set Bw_hat = Up diag(sp^{1/2}) Sigma_w_known^{-1/2}.
            # Compute Sigma_w_known^{-1/2} stably via eigendecomposition.
            lam, Q = np.linalg.eigh(Sigma_w_known)
            lam = np.clip(lam, eps, None)
            Sigma_w_mhalf = Q @ np.diag(lam**-0.5) @ Q.T

            Bw_hat = Up @ np.diag(sp_sqrt) @ Sigma_w_mhalf
            Sigma_w_hat = Sigma_w_known

        else:
            raise ValueError("mode must be one of {'white','factor','known_cov'}.")

        info = {
            "nw": nw,
            "energy": float(np.sum(sp)) / total if total > 0 else 0.0,
            "s_vals": s,
        }
        return Bw_hat, R, Sigma_w_hat, info

    def spectral_norm_epigraph(A: cp.Expression, name: str):
        """
        Impone ||A||_2 <= t_name con un'epigrafe LMI:
            [[t I_m, A],
            [A.T,   t I_n]] >> 0
        Ritorna la variabile scalare t e la lista dei vincoli.
        """
        m, n = A.shape
        t = cp.Variable(nonneg=True, name=f"t_{name}")
        blk = cp.bmat([ [t * I(m),  A       ],
                        [A.T,       t * I(n)]])
        return t, [blk >> 0]
    

    Dx = np.vstack([x, u])
    Ox = x_next @ _pseudo_inv(Dx)
    Ax, Bu = Ox[:, :nx], Ox[:, nx:nx+nu]


    Bw, R, *_ = estimate_Bw_from_residuals(Ax_hat=Ax, Bu_hat=Bu, mode="factor")
    nw = Bw.shape[1]
    w = _pseudo_inv(Bw) @ R
    d = Disturbances(n=nw)
    Sigma_nom = var * np.eye(nw) #d.estm_Sigma_nom(w.T)
    print(f"Estimated Sigma_nom:\n{Sigma_nom}")
    print(f"True Sigma_nom:\n{d.Sigma_test}")

    #gamma, *_ = d.estimate_gamma_with_ci(w.T)
    gamma2, *_ = d._estimate_gamma_with_ci(w.T)
    print(f"Estimated disturbance dimension nw: {nw}, gamma: {gamma}, gamma2: {gamma2}")
    print(f"True gamma: {d.gamma}")

    #Delta = cp.Variable((Ox.shape[0], Ox.shape[1]), name='Delta')
    #Ax, Bu = Ax_hat + Delta[:, :nx], Bu_hat + Delta[:, nx:nx+nu]

    ss = np.linalg.svd(Dx, compute_uv=False)    # Singular Value Decomposition
    smin = float(ss[-1]) if ss.size else 0.0
    beta = np.linalg.norm(R, 'fro') / max(smin, 1e-12)


    Dy = np.vstack([x, w])
    Dz = np.vstack([x, u, w])
    Oy = y @ _pseudo_inv(Dy)
    Oz = z @ _pseudo_inv(Dz)
    Cy, Dyw = Oy[:, :nx], Oy[:, nx:nx+nw]
    
    if not real_Z_mats:
        Cz, Dzu, Dzw = Oz[:, :nx], Oz[:, nx:nx+nu], Oz[:, nx+nu:nx+nu+nw]
    else:
        Cz, Dzw, Dzu, *_ = api.build_out_matrices(nw=nw)

    Bw, Dzw, Dyw, nw, Sigma_nom = api._augment_matrices(B_w=Bw, D_vw=Dzw, D_yw=Dyw, var=var, Sigma_nom=Sigma_nom, N=(1,1))


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


    # DRO matrices --------------------
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


    # Constraints ---------------------
    cons = []
    cons += [lam >= 0]
    cons += [Q >> 0]
    if model.lower() in ["correlated", "corr", "1"]:    cons += [P >> 0]
    else:                                               cons += [P >> eps * I(2*nx)]

    obj_dro = cp.trace(Q @ Sigma_nom) + lam * (gamma ** 2)
    reg = 0.0

    if approach == "Young":
        Cy_norm, M_norm, N_norm, X_norm, Y_norm = np.linalg.norm(Cy, 2), 0.15, 0.6, 3.0, 1.0 # 2.5e5, 1.0
        beta_a, beta_b = beta * np.sqrt(nx/(nx+nu)), beta * np.sqrt(nu/(nx+nu))
        beta_aa, beta_ab = np.sqrt(1 + X_norm**2 + Y_norm**2) * beta_a, np.sqrt(M_norm**2 + N_norm**2 * Cy_norm**2) * beta_b
        print(f"Beta: {beta}\nComputed beta_a: {beta_a}, beta_b: {beta_b} \nComputed beta_aa: {beta_aa}, beta_ab: {beta_ab}")

        beta_AA = cp.Parameter(nonneg=True, value=float(np.clip(beta_aa, 0.0, 1e3)))
        beta_AB = cp.Parameter(nonneg=True, value=float(np.clip(beta_ab, 0.0, 1e3)))

        tau_AA = cp.Variable(nonneg=True, name="tau_aa")
        s_AA = cp.Variable(nonneg=True, name="s_aa") 
        S_AA = np.hstack([Ix, Z(nx, nx)]).T
        tau_AB = cp.Variable(nonneg=True, name="tau_ab")
        s_AB = cp.Variable(nonneg=True, name="s_ab") 
        S_AB = np.hstack([Ix, Z(nx, nx)]).T
        
        tK, consK = spectral_norm_epigraph(K, "K")   # usa I_nx e I_nx
        tL, consL = spectral_norm_epigraph(L, "L")   # usa I_nx e I_ny
        tM, consM = spectral_norm_epigraph(M, "M")   # usa I_nu e I_nx
        tN, consN = spectral_norm_epigraph(N, "N")   # usa I_nu e I_ny
        tP, consP = spectral_norm_epigraph(P, "P") 

        # Reg
        mhu_AA = mhu_AB = rhoK = rhoL = rhoM = rhoN = mu
        rhoP = 0 #1e-7
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
        U_A, _, w_A = _residual_anisotropy_weights(R, floor=1e-12, mode="sqrt")
        # Optional: cap tiny directions to avoid numerical issues
        w_A = np.maximum(w_A, 1e-6)
        w_B = np.mean(w_A)

        # ... your existing code that computes beta, beta_a, beta_b ...
        beta_a, beta_b = beta * np.sqrt(nx/(nx+nu)), beta * np.sqrt(nu/(nx+nu))

        # 2) per-direction β for the A-part: scale β_a with weights
        beta_AA_dir_np = np.asarray(beta_a * w_A, dtype=float)  # shape (nx,)

        # 3) rotate the selector S_AA into U_A basis so each slack acts along an eigendirection
        #    Original S_AA had shape (2nx x nx) with block [I; 0]. Keep the same but rotate columns.
        S_AA_base = np.hstack([Ix, Z(nx, nx)]).T               # (2nx x nx)
        S_AA  = S_AA_base @ U_A                            # (2nx x nx)

        # 4) replace scalar slacks by per-direction vectors
        tau_AA = cp.Variable(nx, nonneg=True, name="tau_aa_vec")
        s_AA   = cp.Variable(nx, nonneg=True, name="s_aa_vec")

        # parameters for β per direction
        beta_AA = cp.Parameter(nx, nonneg=True, value=beta_AA_dir_np)


        Cy_norm, M_norm, N_norm, X_norm, Y_norm = np.linalg.norm(Cy, 2), 0.15, 0.6, 3.0, 1.0 # 2.5e5, 1.0
        beta_ab = w_B * beta_b #np.sqrt(M_norm**2 + N_norm**2 * Cy_norm**2) * beta_b
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
        tK, consK = spectral_norm_epigraph(K, "K")   # usa I_nx e I_nx
        tL, consL = spectral_norm_epigraph(L, "L")   # usa I_nx e I_ny
        tM, consM = spectral_norm_epigraph(M, "M")   # usa I_nu e I_nx
        tN, consN = spectral_norm_epigraph(N, "N")   # usa I_nu e I_ny
        #tP, consP = spectral_norm_epigraph(P, "P")   # usa I_nu e I_ny

        reg += rhoK * tK + rhoL * tL + rhoM * tM + rhoN * tN #+ mu * tP * (cp.sum(beta_AA) + beta_AB)**2

        cons += consK + consL + consM + consN     # += consP
    else:
        pass

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
        cons += [negdef(state_blk)]

        cons += [negdef(big_corr)]

    elif model.lower() in ["independent", "indep", "2"]:
        blk1 = cp.bmat([
            [ -P,           A.T,          C.T,          Z(2*nx, nx),    Z(2*nx, nx)     ],
            [  A,           state_blk,    Z(2*nx, nz),  S_AA,           S_AB            ],
            [  C,           Z(nz, 2*nx), -Iz,           Z(nz, nx),      Z(nz, nx)       ],
            [  Z(nx,2*nx),  S_AA.T,       Z(nx, nz),    young_blk,      Z(nx,   nx)     ],
            [  Z(nx,2*nx),  S_AB.T,       Z(nx, nz),    Z(nx,   nx),   -s_AB * Ix       ],
        ])       
        cons += [negdef(state_blk)]     
        
        blk2 = cp.bmat([
            [-lam*Iw,   lam*Iw,         B.T,            D.T         ],
            [ lam*Iw,  -Q - lam*Iw,     Z(nw, 2*nx),    Z(nw, nz)   ],
            [ B,        Z(2*nx, nw),   -P,              Z(2*nx, nz) ],
            [ D,        Z(nz, nw),      Z(nz, 2*nx),   -Iz          ],
        ])  # Tot size: (2nx + 2nw + nz) x (2nx + 2nw + nz)

        cons += [negdef(blk1)]
        cons += [negdef(blk2)]
    else:
        raise ValueError("model must be 'correlated' or 'independent'.")


    # Optimisation Problem ------------
    obj = cp.Minimize(obj)
    prob = cp.Problem(obj, cons)

    success_MOSEK = success_CVXOPT = success_SCS = False
    solver = "MOSEK"
    print("\n===================================================\nAttempting to solve with MOSEK...")
    try:
        prob.solve(solver=cp.MOSEK, verbose=True, mosek_params={
            'MSK_DPAR_INTPNT_CO_TOL_PFEAS': 1e-7,
            'MSK_DPAR_INTPNT_CO_TOL_DFEAS': 1e-7,
            'MSK_DPAR_INTPNT_CO_TOL_REL_GAP': 1e-7,
            #'MSK_DPAR_INTPNT_TOL_STEP_SIZE': 1e-6,
            'MSK_IPAR_INTPNT_SCALING': 1, # 0: no scaling, 1: geometric mean, 2: equilibrate
        })
        print(f"MOSEK status: {prob.status}")
        if prob.status == cp.OPTIMAL:
            success_MOSEK = True
    except Exception as mosek_e:
        print(f"MOSEK error: {mosek_e}")

    if not success_MOSEK: # and 0:
        """solver = "CVXOPT"
        print("\n===================================================\nMOSEK failed, trying CVXOPT...")
        try:
            prob.solve(solver=cp.CVXOPT,
                    kktsolver='chol',  # if available; otherwise remove
                    maxiters=80,       # you can increase if needed
                    abstol=1e-9, reltol=1e-9, feastol=1e-9,
                    verbose=True)
            print(f"CVXOPT status: {prob.status}")
            if prob.status in (cp.OPTIMAL,):
                success_CVXOPT = True
        except Exception as e:
            print(f"CVXOPT error: {e}")

    if not (success_MOSEK or success_CVXOPT):"""
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

    if success_MOSEK or success_CVXOPT or success_SCS:
        print(f"Solve succeeded ({solver}) with value:", prob.value)
    else:
        print("Optimization error: All solvers failed.")


    total_constraints = len(cons)
    violation_values = []
    violations = 0
    for c in cons:
        v = float(c.violation())
        violation_values.append(v)
        print(c, "violation:", c.violation())
        if v > 1e-6:    violations += 1

    # Returning solutions -------------
    P_val, Q_val, K_val, L_val, M_val, N_val, X_val, Y_val \
        = _val(P.value), _val(Q.value), _val(K.value), _val(L.value), _val(M.value), _val(N.value), _val(X.value), _val(Y.value)
    A_val, B_val, C_val, D_val \
        = _val(A.value), _val(B.value), _val(C.value), _val(D.value)

    # Results -------------------------
    dro = DROLMIResult(
        solver=solver,
        status=prob.status,
        obj_value=float(prob.value),
        gamma=gamma,
        lambda_opt=_val(lam.value),
        Q=Q_val, X=X_val, Y=Y_val, K=K_val, L=L_val, M=M_val, N=N_val,
        Pbar=P_val, Abar=A_val, Bbar=B_val, Cbar=C_val, Dbar=D_val, 
        Tp=None, P=None,
    )

    DeltaA, DeltaB, EAA, EAB = recover_deltas(
        P=P_val, X=X_val, Y=Y_val, M=M_val, N=N_val, Cy=Cy,
        Ahat=Ax, Buhat=Bu,
        beta_AA=np.mean(beta_AA.value), beta_AB=beta_AB.value,
    )

    Ax = Ax + DeltaA
    Bu = Bu + DeltaB

    P = (Ax, Bw, Bu, Cy, Dyw, Cz, Dzw, Dzu)

    other = (DeltaA, DeltaB), (EAA, EAB), (beta, beta_a, beta_b, beta_AA.value, beta_ab), (s_AA.value, s_AB.value), (tau_AA.value, tau_AB.value), (obj_dro.value, reg.value)

    return dro, P, Sigma_nom, other, (violations, total_constraints)

# =============================================================================================== #
