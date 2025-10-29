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
        sys.exit(1)


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
    api: MatricesAPI,
    noise: Noise,
    model: str = "correlated",  # or "independent"
    eps_def: float = 1e-5,
    alpha_cap: float = 1e2,  # keep X, Y from exploding
    fro_cap: float = 1e2,  # keep K, L, M, N from exploding
    additional_constraints: bool = False,
    mhu_x: float = 1.0,
    mhu_y: float = 0.5,
    mhu_z: float = 0.5,
) -> DROLMIResult:
    """
    Builds and solves the DRO-LMI you specified.
    model = "correlated"  implements (1)
    model = "independent" implements (2)
    """

    x, x_next, u, y, z = data.get_data()
    T, nx, nu, ny, nz = x.shape[1], x.shape[0], u.shape[0], y.shape[0], z.shape[0]

    gamma, var = noise.gamma, noise.var
    nw = 1
    Sigma_nom = var * np.eye(nw)
    d = Disturbances(n=nw)
    w = d.sample(T=T, Sigma=Sigma_nom).T


    Dx = np.vstack([x, u, w])
    Dy = np.vstack([x, w])
    Dz = np.vstack([x, u, w])


    def _pseudo_inv(D, r):
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
        return (M << -eps_def * np.eye(M.shape[0])) if additional_constraints else (M << 0)

    def _val(x):
        if x is None:
            return None
        return float(x) if np.isscalar(x) else x
    

    #rx = cp.Variable(nonneg=True, name="ridge_x")
    #ry = cp.Variable(nonneg=True, name="ridge_y")
    #rz = cp.Variable(nonneg=True, name="ridge_z")

    rx = cp.Parameter(nonneg=True, value=1e-6)
    ry = cp.Parameter(nonneg=True, value=1e-6)
    rz = cp.Parameter(nonneg=True, value=1e-6)

    Gx = cp.Variable((Dx.shape[0], Dx.shape[0]), PSD=True, name='Gx')
    Gy = cp.Variable((Dy.shape[0], Dy.shape[0]), PSD=True, name='Gy')
    Gz = cp.Variable((Dz.shape[0], Dz.shape[0]), PSD=True, name='Gz')

    Ac = cp.Variable((nx, nx), name='Ac')
    Bc = cp.Variable((nx, ny), name='Bc')
    Cc = cp.Variable((nu, nx), name='Cc')
    Dc = cp.Variable((nu, ny), name='Dc')

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

    #Dx_inv = _pseudo_inv(Dx, rx)
    #Dy_inv = _pseudo_inv(Dy, ry)
    #Dz_inv = _pseudo_inv(Dz, rz)

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
    

    # Closed Loop sys -----------------
    Acl = cp.bmat([
        [Ax + Bu @ Dc @ Cy,     Bu @ Cc], 
        [Bc @ Cy,               Ac]
    ])

    Bcl = cp.bmat([
        [Bw + Bu @ Dc @ Dyw],
        [Bc @ Dyw]
    ])

    Ccl = cp.bmat([
        [Cz + Dzu @ Dc @ Cy,    Dzu @ Cc]
    ])

    Dcl = Dzw + Dzu @ Dc @ Dyw


    # DRO matrices --------------------
    P = cp.bmat([
        [Y,     Ix], 
        [Ix,    X]
    ])

    A1 = P @ Acl
    B1 = P @ Bcl
    C1 = Ccl
    D1 = Dcl


    A2 = cp.bmat([
        [Ax @ Y + Bu @ M,       Ax + Bu @ N @ Cy ], 
        [K,                     X @ Ax + L @ Cy]
    ])
    B2 = cp.bmat([
        [Bw + Bu @ N @ Dyw], 
        [X @ Bw + L @ Dyw]
    ])
    C2 = cp.bmat([
        [Cz @ Y + Dzu @ M,      Cz + Dzu @ N @ Cy]
    ])
    D2 = Dzw + Dzu @ N @ Dyw

    A, B, C, D = A1, B1, C1, D1


    # Constraints ---------------------
    cons = []
    cons += [lam >= 0]
    cons += [P >> 0]
    cons += [A1 == A2]
    cons += [B1 == B2]
    cons += [C1 == C2]
    cons += [D1 == D2]
    #cons += [_is_stable(Acl)]

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
        cons += [P >> eps_def * np.eye(2*nx)]

        # Objective: tr(Q Sigma_nom) + lambda * gamma^2
        reg = 1e-4 * (
            cp.trace(X) + cp.trace(Y)
            + 0.1*cp.sum_squares(K) + 0.1*cp.sum_squares(L)
            + 0.1*cp.sum_squares(M) + 0.1*cp.sum_squares(N)
        )
    else:
        reg = 0.0


    if model.lower() in ["correlated", "corr", "1"]:
        big_corr = cp.bmat([
            [ -P,           Z(2*nx, nw),    Z(2*nx, nw),    A.T,            C.T         ],
            [ Z(nw, 2*nx), -lam*Iw,         lam*Iw,         B.T,            D.T         ],
            [ Z(nw, 2*nx),  lam*Iw,        -Q - lam*Iw,     Z(nw, 2*nx),    Z(nw, nz)   ],
            [  A,           B,              Z(2*nx, nw),   -P,              Z(2*nx, nz) ],
            [  C,           D,              Z(nz, nw),      Z(nz, 2*nx),   -Iz          ],
        ])  # Tot size: (4nx + 2nw + nz) x (4nx + 2nw + nz)
        cons += [negdef(big_corr)]

    elif model.lower() in ["independent", "indep", "2"]:
        blk1 = cp.bmat([
            [-P,    A.T,            C.T         ],
            [ A,   -P,             Z(2*nx, nz)  ],
            [ C,    Z(nz, 2*nx),   -Iz          ],
        ])  # Tot size: (4nx + nz) x (4nx + nz)

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
    obj_dro = cp.trace(Q @ Sigma_nom) + lam * (gamma ** 2)
    obj_est = mhu_x * (cp.trace((Dx @ Dx.T + rx * I(Dx.shape[0])) @ Gx) - cp.log_det(Gx)) + \
                mhu_y * (cp.trace((Dy @ Dy.T + ry * I(Dy.shape[0])) @ Gy) - cp.log_det(Gy)) + \
                mhu_z * (cp.trace((Dz @ Dz.T + rz * I(Dz.shape[0])) @ Gz) - cp.log_det(Gz))

    obj = cp.Minimize(obj_dro + obj_est + reg)
    prob = cp.Problem(obj, cons)

    # Solve ---------------------------
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
        sys.exit(1)


    # Returning solutions
    status = prob.status
    val = float(prob.value)
    rx_val, ry_val, rz_val, lam_val \
        = _val(rx.value), _val(ry.value), _val(rz.value), _val(lam.value)
    P_val, Q_val, K_val, L_val, M_val, N_val, X_val, Y_val \
        = _val(P.value), _val(Q.value), _val(K.value), _val(L.value), _val(M.value), _val(N.value), _val(X.value), _val(Y.value)
    A1_val, B1_val, C1_val, D1_val \
        = _val(A1.value), _val(B1.value), _val(C1.value), _val(D1.value)
    A2_val, B2_val, C2_val, D2_val \
        = _val(A2.value), _val(B2.value), _val(C2.value), _val(D2.value)
    Acl_val, Bcl_val, Ccl_val, Dcl_val \
        = _val(Acl.value), _val(Bcl.value), _val(Ccl.value), _val(Dcl.value)
    Ac_val, Bc_val, Cc_val, Dc_val \
        = _val(Ac.value), _val(Bc.value), _val(Cc.value), _val(Dc.value)
    Ax_val, Bu_val, Bw_val, Cy_val, Dyw_val, Cz_val, Dzu_val, Dzw_val \
        = _val(Ax.value), _val(Bu.value), _val(Bw.value), _val(Cy.value), _val(Dyw.value), _val(Cz.value), _val(Dzu.value), _val(Dzw.value)


    return DROLMIResultUpd(
        solver=solver,
        status=status,
        obj_value=val,
        gamma=gamma,
        Sigma=Sigma_nom,

        rx=rx_val, ry=ry_val, rz=rz_val,
        lamda=lam_val,

        Q=Q_val, P=P_val, X=X_val, Y=Y_val, K=K_val, L=L_val, M=M_val, N=N_val,
        A1=A1_val, B1=B1_val, C1=C1_val, D1=D2_val, A2=A2_val, B2=B2_val, C2=C2_val, D2=D2_val, 
        A_same=(A1_val==A2_val), B_same=(B1_val==B2_val), C_same=(C1_val==C2_val), D_same=(D1_val==D2_val),

        Ax=Ax_val, Bu=Bu_val, Bw=Bw_val, Cy=Cy_val, Dyw=Dyw_val, Cz=Cz_val, Dzu=Dzu_val, Dzw=Dzw_val,
        Ac=Ac_val, Bc=Bc_val, Cc=Cc_val, Dc=Dc_val,
        Acl=Acl_val, Bcl=Bcl_val, Ccl=Ccl_val, Dcl=Dcl_val,
    )

# ================================================================================================

