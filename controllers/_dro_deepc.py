import sys
import numpy as np, cvxpy as cp

from core import MatricesAPI
from disturbances import Disturbances
from utils import DROLMIResult, Noise

# =============================================================================================== #

def DeePC_dro_lmi(                      # (Gx, Gy, Gz) written
    api: MatricesAPI,
    #data: Data,
    vals: tuple,
    noise: Noise,
    model: str = "correlated",  # or "independent"
    eps: float = 1e-5,
    mhu_x: float = 1.0,
    mhu_y: float = 0.5,
    mhu_z: float = 0.5,
) -> DROLMIResult:
    """
    Builds and solves the DRO-LMI you specified.
    model = "correlated"  implements (1)
    model = "independent" implements (2)
    """
    gamma, var = noise.gamma, noise.var
    upd, FROM_DATA, _ = vals

    data = api.get_system(FROM_DATA=FROM_DATA, gamma=gamma, upd=upd)
    x, x_next, u, y, z = data.get_data()
    T, nx, nu, ny, nz = x.shape[1], x.shape[0], u.shape[0], y.shape[0], z.shape[0]


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

    obj_est = mhu_x * (cp.trace((Dx @ Dx.T + rx * I(Dx.shape[0])) @ Gx) - cp.log_det(Gx)) + \
                mhu_y * (cp.trace((Dy @ Dy.T + ry * I(Dy.shape[0])) @ Gy) - cp.log_det(Gy)) + \
                mhu_z * (cp.trace((Dz @ Dz.T + rz * I(Dz.shape[0])) @ Gz) - cp.log_det(Gz))

    obj_dro += obj_est/10

    obj = obj_dro + reg


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
            [ A,   -P,              Z(2*nx, nz) ],
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
        #sys.exit(0)

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
        try: 
            v = float(c.violation())
            violation_values.append(v)
            print(c, "violation:", c.violation())
            if v > 1e-6:    violations += 1
        except Exception as e:
            pass

    # Returning solutions -------------
    P_val, Q_val, K_val, L_val, M_val, N_val, X_val, Y_val \
        = _val(P.value), _val(Q.value), _val(K.value), _val(L.value), _val(M.value), _val(N.value), _val(X.value), _val(Y.value)
    A_val, B_val, C_val, D_val \
        = _val(A.value), _val(B.value), _val(C.value), _val(D.value)

    # Results -------------------------
    try:
        value = float(prob.value)
    except Exception as e:
        value = None
    
    dro = DROLMIResult(
        solver=solver,
        status=prob.status,
        obj_value=value,
        gamma=gamma,
        lambda_opt=_val(lam.value),
        Q=Q_val, X=X_val, Y=Y_val, K=K_val, L=L_val, M=M_val, N=N_val,
        Pbar=P_val, Abar=A_val, Bbar=B_val, Cbar=C_val, Dbar=D_val, 
        Tp=None, P=None,
    )

    P = (Ax, Bw, Bu, Cy, Dyw, Cz, Dzw, Dzu)

    rx_val, ry_val, rz_val \
        = _val(rx.value), _val(ry.value), _val(rz.value)
    other = (rx_val, ry_val, rz_val)

    return dro, P, Sigma_nom, other, (violations, total_constraints)

# =============================================================================================== #
