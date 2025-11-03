# dro_lmi.py
import numpy as np
import cvxpy as cp
from utils___systems import Plant, DROLMIResult, DROLMIResultUpd, Noise, Data
from utils___matrices import MatricesAPI
from utils___ambiguity import Disturbances
import sys

# ================================================================================================

def build_and_solve_dro_lmi(
    plant: Plant,
    api: MatricesAPI,
    noise: Noise,
    model: str = "correlated",  # or "independent"
    eps_def: float = 1e-5,
    alpha_cap: float = 1e2,  # keep X, Y from exploding
    fro_cap: float = 1e2,  # keep K, L, M, N from exploding
    additional_constraints: bool = False,
) -> DROLMIResult:
    """
    Builds and solves the DRO-LMI you specified.
    model = "correlated"  implements (1)
    model = "independent" implements (2)
    """
    A, Bw, Bu, Cz, Dzw, Dzu, Cy, Dyw = plant.A, plant.Bw, plant.Bu, plant.Cz, plant.Dzw, plant.Dzu, plant.Cy, plant.Dyw
    nx, nw, nu, nz, ny = plant.dims()

    Sigma_nom, gamma, var = noise.Sigma_nom, noise.gamma, noise.var

    Bw, Dzw, Dyw, nw, Sigma_nom = api._augment_matrices(Bw, Dzw, Dyw, var)

    # Decision variables
    lam = cp.Variable(nonneg=True, name="lambda")
    Q = cp.Variable((nw, nw), symmetric=True, name="Q")     # PSD=True

    X = cp.Variable((nx, nx), symmetric=True, name="X")
    Y = cp.Variable((nx, nx), symmetric=True, name="Y")
    K = cp.Variable((nx, nx), name="K")
    L = cp.Variable((nx, ny), name="L")
    M = cp.Variable((nu, nx), name="M")
    N = cp.Variable((nu, ny), name="N")

    # Construct mathbb{P} = [[Y, I], [I, X]]
    I_x = np.eye(nx)
    Pbar = cp.bmat([[Y, I_x],                   # 2nx x 2nx
                    [I_x, X]])

    # Construct mathbb{A}, mathbb{B}, mathbb{C}, mathbb{D}
    Abar_11 = A @ Y + Bu @ M                    # nx x nx
    Abar_12 = A + Bu @ N @ Cy                   # nx x nx
    Abar_21 = K                                 # nx x nx
    Abar_22 = X @ A + L @ Cy                    # nx x nx

    Bbar_11 = Bw + Bu @ N @ Dyw                 # nx x nw
    Bbar_12 = X @ Bw + L @ Dyw                  # nx x nw

    Cbar_11 = Cz @ Y + Dzu @ M                  # nz x nx
    Cbar_12 = Cz + Dzu @ N @ Cy                 # nz x nx

    Dbar_1  = Dzw + Dzu @ N @ Dyw               # nz x nw

    Abar = cp.bmat([[Abar_11,   Abar_12],       # 2nx x 2nx
                    [Abar_21,   Abar_22]])
    Bbar = cp.vstack([Bbar_11,                  # 2nx x nw
                      Bbar_12])
    Cbar = cp.hstack([Cbar_11, Cbar_12])        # nz x 2nx
    Dbar = Dbar_1                               # nz x nw

    Tp = np.eye(2*nx)
    Tp_t = Tp.T
    Tp_inv, Tp_t_inv = np.linalg.inv(Tp), np.linalg.inv(Tp_t)


    def get_Acl(Abar, Pbar):
        P = Tp_t_inv @ Pbar @ Tp_inv
        Acl = Tp_inv @ Abar @ np.linalg.inv(Tp_t @ P)
        return Acl
    
    def is_stable(A, tol=1e-9):
        eigvals = np.linalg.eigvals(A)
        spectral_radius = np.max(np.abs(eigvals))
        return spectral_radius < 1- tol

    cons = []
    if additional_constraints:
        # Avoid explosions
        cons += [X << alpha_cap * np.eye(nx), Y << alpha_cap * np.eye(nx)]
        # optional: bound Frobenius norms of “gain-like” variables
        cons += [cp.norm(K, 'fro') <= fro_cap,
                cp.norm(L, 'fro') <= fro_cap,
                cp.norm(M, 'fro') <= fro_cap,
                cp.norm(N, 'fro') <= fro_cap]

        # Symmetry / positivity
        cons += [X >> eps_def * np.eye(nx), Y >> eps_def * np.eye(nx)]
        cons += [Pbar >> eps_def * np.eye(2*nx)]

        # Objective: tr(Q Sigma_nom) + lambda * gamma^2
        reg = 1e-4 * (
            cp.trace(X) + cp.trace(Y)
            + 0.1*cp.sum_squares(K) + 0.1*cp.sum_squares(L)
            + 0.1*cp.sum_squares(M) + 0.1*cp.sum_squares(N)
        )
    else:
        cons += [Pbar >> 0]
        #cons += [is_stable(np.linalg.inv(Pbar.value) @ Abar.value)]
        reg = 0.0

    # Negative definiteness helpers (strict -> with epsilon)
    def negdef(M): return (M << -eps_def * np.eye(M.shape[0])) if additional_constraints else (M << 0)

    Iw = np.eye(nw)
    Iz = np.eye(nz)

    # handy zero of the right size
    def Z(r, c): return np.zeros((r, c)) # cp.Constant(np.zeros((r, c)))

    if model.lower() in ["correlated", "corr", "1"]:
        # Block sizes by columns: [2nx, nw, nw, 2nx, nz]
        # Row heights: [2nx, nw, nw, 2nx, nz]
        big_corr = cp.bmat([
            # row 1: size 2nx x (4nx + 2nw + nz)
            [ -Pbar,          Z(2*nx, nw),    Z(2*nx, nw),    Abar.T,           Cbar.T          ],
            # row 2: size nw x (4nx + 2nw + nz)
            [ Z(nw, 2*nx),   -lam*Iw,         lam*Iw,         Bbar.T,           Dbar.T          ],
            # row 3: size nw x (4nx + 2nw + nz)
            [ Z(nw, 2*nx),    lam*Iw,        -Q - lam*Iw,     Z(nw, 2*nx),      Z(nw, nz)       ],
            # row 4: size 2nx x (4nx + 2nw + nz)
            [  Abar,           Bbar,           Z(2*nx, nw),   -Pbar,             Z(2*nx, nz)    ],
            # row 5: size nz x (4nx + 2nw + nz)
            [  Cbar,           Dbar,           Z(nz, nw),      Z(nz, 2*nx),     -Iz             ],
        ])  # Tot size: (4nx + 2nw + nz) x (4nx + 2nw + nz)
        cons += [negdef(big_corr)]

    elif model.lower() in ["independent", "indep", "2"]:
        # (2a) sizes:
        # columns: [2nx, 2nx, nz]; rows: [2nx, 2nx, nz]
        blk1 = cp.bmat([
            [ -Pbar,  Abar.T,             Cbar.T             ],
            [  Abar, -Pbar,               Z(2*nx, nz)        ],
            [  Cbar,  Z(nz, 2*nx),       -Iz                 ],
        ])
        cons += [negdef(blk1)]

        # (2b) 4x4 with columns [nw, nw, 2nx, nz]; rows [nw, nw, 2nx, nz]
        blk2 = cp.bmat([
            [ -lam*Iw,        lam*Iw,         Bbar.T,            Dbar.T          ],
            [  lam*Iw,   -Q - lam*Iw,         Z(nw, 2*nx),       Z(nw, nz)       ],
            [  Bbar,          Z(2*nx, nw),   -Pbar,              Z(2*nx, nz)     ],
            [  Dbar,          Z(nz, nw),      Z(nz, 2*nx),      -Iz              ],
        ])
        cons += [negdef(blk2)]
    else:
        raise ValueError("model must be 'correlated' or 'independent'.")

    cons += [Q >> 0]
    obj = cp.Minimize(cp.trace(Q @ Sigma_nom) + lam * (gamma ** 2) + reg)
    prob = cp.Problem(obj, cons)

    # Solve
    success_MOSEK = success_CVXOPT = success_SCS = False
    solver = "MOSEK"
    print("\n===================================================\nAttempting to solve with MOSEK...")
    try:
        prob.solve(solver=cp.MOSEK, verbose=True, mosek_params={
            'MSK_DPAR_INTPNT_CO_TOL_PFEAS': 1e-8,
            'MSK_DPAR_INTPNT_CO_TOL_DFEAS': 1e-8,
            'MSK_DPAR_INTPNT_CO_TOL_REL_GAP': 1e-8,
            'MSK_DPAR_INTPNT_TOL_STEP_SIZE': 1e-6
        })
        print(f"MOSEK status: {prob.status}")
        if prob.status == cp.OPTIMAL:
            success_MOSEK = True
    except Exception as mosek_e:
        print(f"MOSEK error: {mosek_e}")

    if not success_MOSEK:
        solver = "CVXOPT"
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

    if not (success_MOSEK or success_CVXOPT):
        solver = "SCS"
        print("\n===================================================\nCVXOPT failed, trying SCS...")
        try:
            prob.solve(solver=cp.SCS, verbose=True, eps=1e-4, max_iters=100000)
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


    # Safe extraction
    def _val(x):
        if x is None:
            return None
        return float(x) if np.isscalar(x) else x


    status = prob.status
    val = float(prob.value)
    lam_val = _val(lam.value)
    Q_val  = _val(Q.value)
    X_val, Y_val = _val(X.value), _val(Y.value)
    K_val, L_val, M_val, N_val = _val(K.value), _val(L.value), _val(M.value), _val(N.value)
    Pbar_val = _val(Pbar.value)
    Abar_val, Bbar_val, Cbar_val = _val(Abar.value), _val(Bbar.value), _val(Cbar.value)
    Dbar_val = _val(Dbar.value) if "Dbar" in locals() else None  # guard if you didn’t build it

    return DROLMIResult(
        solver=solver,
        status=status,
        obj_value=val,
        gamma=gamma,
        lambda_opt=lam_val,
        Q=Q_val, X=X_val, Y=Y_val, K=K_val, L=L_val, M=M_val, N=N_val,
        Pbar=Pbar_val, Abar=Abar_val, Bbar=Bbar_val, Cbar=Cbar_val, Dbar=Dbar_val, 
        Tp = Tp, P=Tp_t_inv @ Pbar_val @ Tp_inv
    )

# ================================================================================================

def build_and_solve_dro_lmi_upd(
    data: Data,
    noise: Noise,
    model: str = "correlated",  # or "independent"
    eps_def: float = 1e-5,
    alpha_cap: float = 1e2,  # keep X, Y from exploding
    fro_cap: float = 1e2,  # keep K, L, M, N from exploding
    mhu_x: float = 1.0,
    mhu_y: float = 0.5,
    mhu_z: float = 0.5,
    approach: str = "DeePC",
) -> DROLMIResult:
    """
    Builds and solves the DRO-LMI you specified.
    model = "correlated"  implements (1)
    model = "independent" implements (2)
    """

    x, x_next, u, y, z = data.get_data()
    T, nx, nu, ny, nz = x.shape[1], x.shape[0], u.shape[0], y.shape[0], z.shape[0]
    gamma, var = noise.gamma, noise.var


    def _pseudo_inv(D, r=1e-6):
        return D.T @ np.linalg.inv(D @ D.T + r * np.eye(D.shape[0]))
    
    def I(n):
        return np.eye(n)

    def _is_stable(M, tol=1e-9):
        eigvals = np.linalg.eigvals(M)
        spectral_radius = np.max(np.abs(eigvals))
        return spectral_radius < 1- tol

    def Z(r, c): 
        return np.zeros((r, c)) 

    def negdef(M): 
        return (M << -eps_def * np.eye(M.shape[0]))

    def _val(x):
        if x is None:
            return None
        return float(x) if np.isscalar(x) else x


    if approach == "DeePC":
        nw = 1
        Sigma_nom = var * np.eye(nw)
        d = Disturbances(n=nw)
        w = d.sample(T=T, Sigma=Sigma_nom).T

        Dx = np.vstack([x, u, w])
        Dy = np.vstack([x, w])
        Dz = np.vstack([x, u, w])

        rx = cp.Parameter(nonneg=True, value=1e-6)
        ry = cp.Parameter(nonneg=True, value=1e-6)
        rz = cp.Parameter(nonneg=True, value=1e-6)

        Gx = cp.Variable((Dx.shape[0], Dx.shape[0]), PSD=True, name='Gx')
        Gy = cp.Variable((Dy.shape[0], Dy.shape[0]), PSD=True, name='Gy')
        Gz = cp.Variable((Dz.shape[0], Dz.shape[0]), PSD=True, name='Gz')


        # Matrices approximation ----------
        OX = x_next @ Dx.T @ Gx
        OY = y @ Dy.T @ Gy
        OZ = z @ Dz.T @ Gz

        Ax = OX[:, :nx]
        Bu = OX[:, nx:nx+nu]
        Bw = OX[:, nx+nu:nx+nu+nw]
        Cy = OY[:, :nx]
        Dyw = OY[:, nx:nx+nw]
        Cz = OZ[:, :nx]
        Dzu = OZ[:, nx:nx+nu]
        Dzw = OZ[:, nx+nu:nx+nu+nw]
    
    elif approach == "Young":
        Dx = np.vstack([x, u])
        Ox = x_next @ _pseudo_inv(Dx)
        Ax, Bu = Ax_hat, Bu_hat = Ox[:, :nx], Ox[:, nx:nx+nu]

        R = x_next - (Ax_hat @ x + Bu_hat @ u)
        S = (R @ R.T) / max(T, 1)
        S = 0.5 * (S + S.T) + 1e-12 * np.eye(nx)
        U_s, s_s, _ = np.linalg.svd(S, full_matrices=False)
        cum = np.cumsum(s_s) / max(np.sum(s_s), 1e-18)
        nw = int(np.clip(np.searchsorted(cum, 0.95) + 1, 1, nx))
        Bw = U_s[:, :nw] @ np.diag(np.sqrt(s_s[:nw]))     
        w = _pseudo_inv(Bw) @ R
        Sigma_nom = 0.5*((w @ w.T)/max(T-1,1) + ((w @ w.T)/max(T-1,1)).T) + 1e-9*np.eye(nw)

        Delta = cp.Variable((Ox.shape[0], Ox.shape[1]), name='Delta')
        #Ax, Bu = Ax_hat + Delta[:, :nx], Bu_hat + Delta[:, nx:nx+nu]

        ss = np.linalg.svd(Dx, compute_uv=False)
        smin = float(ss[-1]) if ss.size else 0.0
        beta_data = np.linalg.norm(R, 'fro') / max(smin, 1e-12)

        Dy = np.vstack([x, w])
        Dz = np.vstack([x, u, w])
        Oy = y @ _pseudo_inv(Dy)
        Oz = z @ _pseudo_inv(Dz)
        Cy, Dyw = Oy[:, :nx], Oy[:, nx:nx+nw]
        Cz, Dzu, Dzw = Oz[:, :nx], Oz[:, nx:nx+nu], Oz[:, nx+nu:nx+nu+nw]


    elif approach == "Vertices":
        pass 

    else:
        raise ValueError("approach must be 'DeePC', 'Young' or 'Vertices'")


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
    cons += [P >> 0]

    obj_dro = cp.trace(Q @ Sigma_nom) + lam * (gamma ** 2)


    if approach == 'DeePc':
        obj_est = mhu_x * (cp.trace((Dx @ Dx.T + rx * I(Dx.shape[0])) @ Gx) - cp.log_det(Gx)) + \
                    mhu_y * (cp.trace((Dy @ Dy.T + ry * I(Dy.shape[0])) @ Gy) - cp.log_det(Gy)) + \
                    mhu_z * (cp.trace((Dz @ Dz.T + rz * I(Dz.shape[0])) @ Gz) - cp.log_det(Gz))
        reg = 0.0

        obj = obj_dro + obj_est + reg
    elif approach == "Young":
        beta = cp.Parameter(nonneg=True, value=float(np.clip(beta_data, 0.0, 1e3)))
        tau = cp.Variable(nonneg=True, name="tau")
        s = cp.Variable(nonneg=True, name="s") 
        S = np.hstack([Ix, Z(nx, nx)]).T

        #cons += [cp.norm(Delta, 'fro') <= beta]
        reg = 0.0

        cons += [cp.bmat([[s, beta], [beta, tau]]) >> 0]

        obj = obj_dro + reg
    else:
        reg = 0.0

        obj = obj_dro + reg


    if model.lower() in ["correlated", "corr", "1"]:
        if approach == "DeePC":
            big_corr = cp.bmat([
                [ -P,           Z(2*nx, nw),    Z(2*nx, nw),    A.T,            C.T         ],
                [ Z(nw, 2*nx), -lam*Iw,         lam*Iw,         B.T,            D.T         ],
                [ Z(nw, 2*nx),  lam*Iw,        -Q - lam*Iw,     Z(nw, 2*nx),    Z(nw, nz)   ],
                [  A,           B,              Z(2*nx, nw),   -P,              Z(2*nx, nz) ],
                [  C,           D,              Z(nz, nw),      Z(nz, 2*nx),   -Iz          ],
            ])  # Tot size: (4nx + 2nw + nz) x (4nx + 2nw + nz)
        elif approach == "Young":
            big_corr = cp.bmat([
                [ -P,            Z(2*nx, nw),  Z(2*nx, nw),  A.T,          C.T,         Z(2*nx, nx)     ],
                [ Z(nw,2*nx),   -lam*Iw,       lam*Iw,       B.T,          D.T,         Z(nw,   nx)     ],
                [ Z(nw,2*nx),    lam*Iw,      -Q - lam*Iw,   Z(nw,2*nx),   Z(nw,  nz),  Z(nw,   nx)     ],
                [  A,            B,            Z(2*nx, nw), -P,            Z(2*nx,nz),  S               ],
                [  C,            D,            Z(nz,  nw),   Z(nz, 2*nx), -Iz,          Z(nz,   nx)     ],
                [  Z(nx,2*nx),   Z(nx,  nw),   Z(nx,  nw),   S.T,          Z(nx,  nz), -s * Ix          ],
            ])

        cons += [negdef(big_corr)]

    elif model.lower() in ["independent", "indep", "2"]:
        if approach == "DeePC":
            blk1 = cp.bmat([
                [-P,    A.T,            C.T         ],
                [ A,   -P,              Z(2*nx, nz) ],
                [ C,    Z(nz, 2*nx),   -Iz          ],
            ])  # Tot size: (4nx + nz) x (4nx + nz)
        elif approach == "Young":
            blk1 = cp.bmat([
                [ -P,           A.T,          C.T,          np.zeros((2*nx, nx))],
                [  A,          -P,            Z(2*nx, nz),  S                   ],
                [  C,           Z(nz, 2*nx), -Iz,           Z(nz, nx)           ],
                [  Z(nx,2*nx),  S.T,          Z(nx, nz),   -s * Ix              ],
            ])        

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
            'MSK_DPAR_INTPNT_CO_TOL_PFEAS': 1e-8,
            'MSK_DPAR_INTPNT_CO_TOL_DFEAS': 1e-8,
            'MSK_DPAR_INTPNT_CO_TOL_REL_GAP': 1e-8,
            'MSK_DPAR_INTPNT_TOL_STEP_SIZE': 1e-6
        })
        print(f"MOSEK status: {prob.status}")
        if prob.status == cp.OPTIMAL:
            success_MOSEK = True
    except Exception as mosek_e:
        print(f"MOSEK error: {mosek_e}")

    if not success_MOSEK:
        solver = "CVXOPT"
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

    if not (success_MOSEK or success_CVXOPT):
        solver = "SCS"
        print("\n===================================================\nCVXOPT failed, trying SCS...")
        try:
            prob.solve(solver=cp.SCS, verbose=True, eps=1e-4, max_iters=100000)
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

    P = (Ax, Bw, Bu, Cy, Dyw, Cz, Dzw, Dzu)

    if approach == 'DeePC':
        rx_val, ry_val, rz_val \
            = _val(rx.value), _val(ry.value), _val(rz.value)
        other = (rx_val, ry_val, rz_val)
    elif approach == 'Young':
        other = (_val(Delta.value), _val(beta.value))

    return dro, P, Sigma_nom, other

# ================================================================================================

