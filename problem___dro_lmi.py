import json, sys, time, psutil, os, numpy as np, cvxpy as cp, casadi as ca
import matplotlib.pyplot as plt

from pathlib import Path
from typing import Dict, Any, List
from utils___systems import Plant, Plant_cl, Controller, DROLMIResult, Noise
from utils___matrices import MatricesAPI, recover_deltas, compose_closed_loop, Recover
from utils___ambiguity import Disturbances
from utils___simulate import Open_Loop, Closed_Loop
from utils___Nsims_mats import NsimsMatricesAnalyzer, mean_dict, select_representative_run, plot_first3_and_mean
from problem___dro_wfl import WFL


# =============================================================================================== #

def Baseline_dro_lmi(
    plant: Plant,
    api: MatricesAPI,
    noise: Noise,
    model: str = "correlated",  # or "independent"
    eps_def: float = 1e-5,
    alpha_cap: float = 1e2,  # keep X, Y from exploding
    fro_cap: float = 1e2,  # keep K, L, M, N from exploding
    additional_constraints: bool = False,
    SOLVER: str = None,
) -> DROLMIResult:
    """
    Builds and solves the DRO-LMI you specified.
    model = "correlated"  implements (1)
    model = "independent" implements (2)
    """
    A, Bw, Bu, Cz, Dzw, Dzu, Cy, Dyw = plant.A, plant.Bw, plant.Bu, plant.Cz, plant.Dzw, plant.Dzu, plant.Cy, plant.Dyw
    nx, nw, nu, nz, ny = plant.dims()

    Sigma_nom, gamma, var = noise.Sigma_nom, noise.gamma, noise.var

    Bw, Dzw, Dyw, nw, Sigma_nom = api._augment_matrices(B_w=Bw, D_vw=Dzw, D_yw=Dyw, var=var, Sigma_nom=Sigma_nom)

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
            [  Abar,          Bbar,           Z(2*nx, nw),   -Pbar,             Z(2*nx, nz)     ],
            # row 5: size nz x (4nx + 2nw + nz)
            [  Cbar,          Dbar,           Z(nz, nw),      Z(nz, 2*nx),     -Iz              ],
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

    #cons += [Q >> 0]
    obj = cp.Minimize(cp.trace(Q @ Sigma_nom) + lam * (gamma ** 2) + reg)
    prob = cp.Problem(obj, cons)

    # Solve
    success_MOSEK = success_CVXOPT = success_SCS = False
    if SOLVER == "MOSEK":
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

    if not success_MOSEK or SOLVER == "CVXOPT":
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

    if not (success_MOSEK or success_CVXOPT) or SOLVER == "SCS":
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

    dro = DROLMIResult(
        solver=solver,
        status=status,
        obj_value=val,
        gamma=gamma,
        lambda_opt=lam_val,
        Q=Q_val, X=X_val, Y=Y_val, K=K_val, L=L_val, M=M_val, N=N_val,
        Pbar=Pbar_val, Abar=Abar_val, Bbar=Bbar_val, Cbar=Cbar_val, Dbar=Dbar_val, 
        Tp = Tp, P=Tp_t_inv @ Pbar_val @ Tp_inv
    )

    return dro, (violations, total_constraints)

# =============================================================================================== #

def Estm_dro_lmi(
    api: MatricesAPI,
    noise: Noise,
    model: str = "correlated",  # or "independent"
    eps_def: float = 1e-5,
    alpha_cap: float = 1e2,  # keep X, Y from exploding
    fro_cap: float = 1e2,  # keep K, L, M, N from exploding
    additional_constraints: bool = False,
    SOLVER: str = None,
    real_Z_mats: bool = True,
    N_sims: int = 1,
) -> DROLMIResult:
    """
    Builds and solves the DRO-LMI you specified.
    model = "correlated"  implements (1)
    model = "independent" implements (2)
    """
    Sigma_nom, gamma, var = noise.Sigma_nom, noise.gamma, noise.var

    """data = api.get_system(FROM_DATA=True, gamma=gamma, upd=True)
    x, x_next, u, y, z = data.get_data()"""

    op = Open_Loop(MAKE_DATA=False, EVAL_FROM_PATH=False, DATASETS=True, N=N_sims)
    datasets = op.datasets

    avg = select_representative_run(datasets) if N_sims!=1 else datasets
    x, u, y, z, x_next = avg["X"], avg["U"], avg["Y"], avg["Z"], avg["X_next"]
    nx, nu, ny, nz = x.shape[0], u.shape[0], y.shape[0], z.shape[0]

    (A, Bu, Bw, Cy, Dyw, Cz, Dzu, Dzw), (_, nw, _), (Sigma_nom, gamma) \
        = api.estm_mats(X_=x, U_=u, X=x_next, Y_=y, Z_=z, Sigma_nom=Sigma_nom, real_perf_mats=real_Z_mats, gamma=gamma, estm_noise=False)
    


    Bw, Dzw, Dyw, nw, Sigma_nom = api._augment_matrices(B_w=Bw, D_vw=Dzw, D_yw=Dyw, var=var, Sigma_nom=Sigma_nom)
    plant = Plant(A=A, Bw=Bw, Bu=Bu, Cz=Cz, Dzw=Dzw, Dzu=Dzu, Cy=Cy, Dyw=Dyw)

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
            [  Abar,          Bbar,           Z(2*nx, nw),   -Pbar,             Z(2*nx, nz)     ],
            # row 5: size nz x (4nx + 2nw + nz)
            [  Cbar,          Dbar,           Z(nz, nw),      Z(nz, 2*nx),     -Iz              ],
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

    #cons += [Q >> 0]
    obj = cp.Minimize(cp.trace(Q @ Sigma_nom) + lam * (gamma ** 2) + reg)
    prob = cp.Problem(obj, cons)

    # Solve
    success_MOSEK = success_CVXOPT = success_SCS = False
    if SOLVER == "MOSEK":
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

    if not success_MOSEK or SOLVER == "CVXOPT":
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

    if not (success_MOSEK or success_CVXOPT) or SOLVER == "SCS":
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

    dro = DROLMIResult(
        solver=solver,
        status=status,
        obj_value=val,
        gamma=gamma,
        lambda_opt=lam_val,
        Q=Q_val, X=X_val, Y=Y_val, K=K_val, L=L_val, M=M_val, N=N_val,
        Pbar=Pbar_val, Abar=Abar_val, Bbar=Bbar_val, Cbar=Cbar_val, Dbar=Dbar_val, 
        Tp = Tp, P=Tp_t_inv @ Pbar_val @ Tp_inv
    )

    return dro, plant, Sigma_nom, (violations, total_constraints)

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


    avg = select_representative_run(datasets) if N_sims!=1 else datasets
    x, u, y, z, x_next = avg["X"], avg["U"], avg["Y"], avg["Z"], avg["X_next"]

    if plot:
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

def DeePC_dro_lmi(
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
    upd, FROM_DATA, plot = vals

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

    obj_dro += obj_est

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

    P = (Ax, Bw, Bu, Cy, Dyw, Cz, Dzw, Dzu)

    rx_val, ry_val, rz_val \
        = _val(rx.value), _val(ry.value), _val(rz.value)
    other = (rx_val, ry_val, rz_val)

    return dro, P, Sigma_nom, other, (violations, total_constraints)

# =============================================================================================== #

class Young_Schur_dro_lmi: 
    def __init__(self, 
                 vals: tuple, model: str,
                 api: MatricesAPI, noise: Noise, 
                 rho: float = 1e-2, eps: float = 1e-6, N_sims: int = 1,
                 Bw_mode: str = "known_cov", real_Z_mats: bool = False, 
                 aug_mode: str = "std", eval_from_ol: bool = True, estm_noise: bool = False,
                 reg_fro: bool = False, reg_beta: bool = True, new: bool = True,
                 estm_with_bounds: bool = True,
                 ):

        Bw_mode = "proj" if estm_noise else Bw_mode
        self.api = api
        self.eps, self.rho, self.N_sims = eps, rho, N_sims
        self.model, self.Bw_mode, self.vals = model, Bw_mode, vals
        self.real_perf_mats, self.augmented, self.aug_mode, self.eval_from_ol, self.estm_noise \
              = real_Z_mats, vals[3], aug_mode, eval_from_ol, estm_noise
        self.gamma, self.var, self.Sigma_nom = noise.gamma, noise.var, noise.Sigma_nom
        self.inp = vals[4]

        self.reg_fro, self.reg_beta, self.reg_vect = reg_fro, reg_beta, vals[2]
        self.new = new
        self.estm_with_bounds = estm_with_bounds



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

    def get_dataset(self, N_sims: int):
        op = Open_Loop(MAKE_DATA=False, EVAL_FROM_PATH=False, DATASETS=True, N=N_sims)
        return op.datasets

    def controllability_matrix(self, A, B, T):
        """
        Build the finite-horizon controllability matrix:
        [A^{T-1}B, A^{T-2}B, ..., B]

        Parameters
        ----------
        A : (nx, nx)
        B : (nx, nu)
        T : int

        Returns
        -------
        C_T : (nx, T*nu)
        """
        C_blocks = []

        A_power = np.eye(A.shape[0])
        powers = [A_power]

        # Precompute powers of A up to A^{T-1}
        for _ in range(1, T):
            A_power = A_power @ A
            powers.append(A_power)

        # Build blocks: A^{T-1}B ... B
        for k in reversed(range(T)):
            C_blocks.append(powers[k] @ B)

        C_T = np.hstack(C_blocks)
        return C_T


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
            datasets = self.get_dataset(N_sims=self.N_sims)
            if self.estm_with_bounds: self.full_dataset = datasets

            avg = self.select_representative_run(datasets) if self.N_sims!=1 else datasets
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
        nx, nu, T = self.dims["nx"], self.dims["nu"], self.dims["T"]
        X_, U_, X, Y_, Z_ = self.get_data()
        Dx = np.vstack([X_, U_])

        if not self.estm_with_bounds:
            Ox = X @ self._pseudo_inv(Dx)
            Ax, Bu = Ox[:, :nx], Ox[:, nx:nx+nu]

            R = X - (Ax @ X_ + Bu @ U_)
            self.c = R @ self._pseudo_inv(Dx)
            self.c_a = np.sqrt(nx/(nx+nu)) * self.c
            self.c_b = np.sqrt(nu/(nx+nu)) * self.c

            Bw, nw, self._residual_anisotropy_weights = self.estm_Bw(R)
            W_ = self._pseudo_inv(Bw) @ R


        else: 
            delta = 0.05
            N_sims_new = int(np.floor(8 * (nx + nu) + 16 * np.log(4/delta))) + 1
            full_datasets = self.get_dataset(N_sims=N_sims_new)

            S1 = np.zeros((nx, nx+nu), dtype=float)
            S2 = np.zeros((nx+nu, nx+nu), dtype=float)
            for data in full_datasets:
                X_reg = np.asarray(data["X_reg"], dtype=float)
                U_reg = np.asarray(data["U_reg"], dtype=float)
                X_next = np.asarray(data["X_next"], dtype=float)

                Phi = np.vstack([X_reg, U_reg])
                S1 += X_next @ Phi.T
                S2 += Phi @ Phi.T
            
            S2 += self.eps * np.eye(nx + nu)
            Theta = S1 @ np.linalg.inv(S2)
            Ax, Bu = Theta[:, :nx], Theta[:, nx:]

            sum_r, sum_u = np.zeros((nx,T), dtype=float), 0.0
            count_r, count_u = 0, 0

            for data in full_datasets:
                X_reg = np.asarray(data["X_reg"], dtype=float)
                U_reg = np.asarray(data["U_reg"], dtype=float)
                X_next = np.asarray(data["X_next"], dtype=float)

                R_i = X_next - (Ax @ X_reg + Bu @ U_reg)
                sum_r += R_i
                count_r += 1

                U_c = U_reg - np.mean(U_reg, axis=1, keepdims=True)
                sum_u += np.sum(U_c**2)
                count_u += U_c.shape[1]

            R = sum_r / max(count_r, 1) 
            Bw, nw, self._residual_anisotropy_weights = self.estm_Bw(R)
            W_ = self._pseudo_inv(Bw) @ R

            R_c = R - np.mean(R, axis=1, keepdims=True)
            sigma_U = np.sqrt(sum_u / max(count_u, 1) + self.eps)
            sigma_W = np.sqrt(np.sum(R_c**2) / max(R_c.shape[1], 1) + self.eps)
            sigma_u = np.sqrt(sigma_U)
            sigma_w = np.sqrt(sigma_W)

            G_T = self.controllability_matrix(Ax, Bu, T)
            F_T = self.controllability_matrix(Ax, np.eye(nx), T)

            M = sigma_U * (G_T @ G_T.T) + sigma_W * (F_T @ F_T.T)
            eigvals = np.linalg.eigvalsh(M)
            lambda_min = max(eigvals[0], self.eps)

            const = 16 * sigma_w *  np.sqrt((nx+2*nu)/N_sims_new * np.log(36/delta))

            self.c_a = const / sigma_w #np.sqrt(lambda_min)
            self.c_b = const / sigma_u
            self.N_sims_new = N_sims_new

            print(f"Estimated Bw with nw={nw} using {N_sims_new} simulations (delta={delta})")
            print(f"sigma_u = {sigma_u}, sigma_w = {sigma_w}, lambda_min = {lambda_min}")
            print(f"beta_a = {self.c_a}, beta_b = {self.c_b}")
            print(f"Ax: {Ax}, Bu: {Bu}, Bw: {Bw}")
            input("...")

            Cz, Dzw, Dzu, Cy, Dyw = self.api.build_out_matrices(nw=nw)

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
        try:
            kappa_A = np.linalg.norm(self.c_a, 2)
            kappa_B = np.linalg.norm(self.c_b, 2)
        except:
            kappa_A = self.c_a
            kappa_B = self.c_b
        c_y = np.linalg.norm(self.mats["Cy"], 2)

        tY_hi, tM_hi, tN_hi, tX_hi = 0.9, 0.13, 0.36, 2.8#e5

        e11 = kappa_A * tY_hi + kappa_B * tM_hi 
        e12 = kappa_A + kappa_B * tN_hi  * c_y
        e22 = tX_hi  * kappa_A

        gamma_E = e11**2 + e12**2 + e22**2
        beta_E_np = gamma_E * self._I(P.shape[0])
        self.c_e = cp.Constant(beta_E_np)
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


        self.sigma_s = cp.Variable(nonneg=True, name="sigma_s")
        self.sigma_p = cp.Variable(nonneg=True, name="sigma_p")


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


        G = self.sigma_p * Ip 
        """cp.Variable((Np, Np), nonneg=True, name="G")
        blkG = cp.bmat([
            [G,     Ip  ], 
            [Ip,    P   ],
        ])
        cons += [self._posdef(blkG)]#"""

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
                [-P_bar,                self._Z(2*nx, nw),  self._Z(2*nx, nw),      A.T,                C.T                 ],
                [ self._Z(nw, 2*nx),   -lam*self._I(nw),    lam*self._I(nw),        B.T,                D.T                 ],
                [ self._Z(nw, 2*nx),    lam*self._I(nw),   -Q - lam*self._I(nw),    self._Z(nw, 2*nx),  self._Z(nw, nz)     ],
                [  A,                   B,                  self._Z(2*nx, nw),     -Ps,                 self._Z(2*nx, nz)   ],
                [  C,                   D,                  self._Z(nz, nw),        self._Z(nz, 2*nx), -self._I(nz)         ],
            ])

            cons += [self._negdef(blk, which="strict")]

        elif self.model == "independent":
            blk1 = cp.bmat([
                [-P_bar,    A.T,                C.T                 ],
                [ A,       -Ps,                 self._Z(2*nx, nz)   ],
                [ C,        self._Z(nz, 2*nx), -self._I(nz)         ],
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

        if 0 and not success_MOSEK:
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
            print(e)

        print(f"total_constraints: {self.total_constraints}, num violations: {self.violations}")
        print(f"Objective value: {self.value}")
        print(f"Q: {self.vars["Q"].value}")

        #print(np.linalg.eigh(self.vars["P"].value)); sys.exit(0)

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
            B = (self.beta, self.beta_a, self.beta_b, self.beta_A.value, self.beta_B.value, self.c_a, self.c_b, self.c_e.value if self.c_e is not None else None)
            S = (self.s_A.value, self.s_B.value)
            T = None if self.new else (self.tau_A.value, self.tau_B.value)
            R = (self.obj.value, self.reg.value, self.reg_t.value, self.reg_a.value, self.reg_b.value)

            self.others = D, E, B, S, T, R
        else: 
            self.others = 0.0


    # ============================================================================ #

# =============================================================================================== #


# ------------------------- DRO-LMI PIPELINE OPTIMIZATION PROBLEM ------------------

class lmi_pipeline_optim_problem(): 
    def __init__(self, params: dict, out: Path, noise: Noise, 
                 upd: bool = False, plot: bool = False, save: bool = False, 
                 FROM_DATA: bool = False, init_cond: str = "zero", N_sims: int = None,
                 disturbance_type: str = "Gaussian"):

        recover = Recover()
        api = MatricesAPI()
        cl = Closed_Loop() 

        STABLE, i = False, 0
        tot_time = 0
        gamma, Sigma_nom = noise.gamma, noise.Sigma_nom

        model = params.get("model", "correlated") if params.get("ambiguity", {}).get("model", "W2") != "Gaussian" else "independent"
        old = bool(params.get("old_upd", 1))
        inp = bool(params.get("inp", 0))
        estm_only = bool(params.get("estm_only", 0))
        N_sims = int(params.get("N_sims", 1)) if N_sims is None else N_sims
        non_convex = bool(params.get("non_convex", 0))
        Nsims_mats = bool(params.get("Nsims_mats", 0))


        if Nsims_mats:
            an = NsimsMatricesAnalyzer(
                api=api,
                noise=noise,
                out_dir="out/EstmMats_Nsims",
                recompute=False,        # set True if you change estimation logic
            )

            an.run()  # estimates (or loads), then generates all plots
            sys.exit(0)

        self.proc = psutil.Process(os.getpid())
        self.solve_stats = []

        while not STABLE and i<7:
            t0 = time.perf_counter()
            cpu0 = self.proc.cpu_times()
            cpu0_total = cpu0.user + cpu0.system


            # 1) Define plant and nominal disturbance covariance (keep consistent with your LMI)
            if not upd or not FROM_DATA:
                plant, _ = api.get_system(FROM_DATA=FROM_DATA, gamma=noise.gamma, upd=upd)
                real_Z_mats = True
                N_sims = 0

                # 2) Solve DRO-LMI (choose "correlated" or "independent")
                res, num_violations = Baseline_dro_lmi(
                    plant=plant,
                    api=api,
                    noise=noise,
                    model=model,
                    SOLVER=params.get("solver", "MOSEK"),
                )

                A, Bw, Bu, Cz, Dzw, Dzu, Cy, Dyw = plant.A, plant.Bw, plant.Bu, plant.Cz, plant.Dzw, plant.Dzu, plant.Cy, plant.Dyw
                Bw, Dzw, Dyw, _, Sigma_nom = api._augment_matrices(Bw, Dzw, Dyw, noise.var)
                plant = Plant(A=A, Bw=Bw, Bu=Bu, Cz=Cz, Dzw=Dzw, Dzu=Dzu, Cy=Cy, Dyw=Dyw)
                ADD = False

                problem_params = {
                    "Methodology": "Baseline",
                    "FROM_DATA": FROM_DATA, # False
                    "augmented": True,
                    "reg_fro": False,
                    "non_convex": False,
                    "N_sims": 0,
                }

            else:
                approach = params.get("approach", "Young")
                vect = model == "independent"
                augmented = False #model == "correlated"
                reg_fro, reg_beta = True, True

                if not non_convex:

                    # 2) Solve DRO-LMI (choose "correlated" or "independent")
                    if not estm_only:
                        if old:
                            if approach == "DeePC":
                                real_Z_mats = False
                                N_sims = 0
                                res, P, Sigma_nom, other, num_violations = DeePC_dro_lmi(
                                    api=api,
                                    vals=(upd, FROM_DATA, plot),
                                    noise=noise,
                                    model=model,
                                )
                                problem_params = {
                                    "Methodology": "DeePC",
                                    "FROM_DATA": FROM_DATA, # True
                                    "augmented": False,
                                    "reg_fro": False,
                                    "non_convex": False,
                                    "N_sims": N_sims,
                                }
                            else:
                                real_Z_mats = False
                                res, P, Sigma_nom, other, num_violations = Young_dro_lmi(
                                    api=api,
                                    vals=(upd, FROM_DATA, plot),
                                    noise=noise,
                                    model=model,
                                    approach=approach,
                                    real_Z_mats=real_Z_mats,
                                    N_sims=N_sims,
                                ) 
                                problem_params = {
                                    "Methodology": approach,
                                    "FROM_DATA": FROM_DATA, # True
                                    "augmented": False,
                                    "reg_fro": False,
                                    "non_convex": False,
                                    "N_sims": N_sims,
                                }        
                        else:
                            real_Z_mats = False
                            estm_with_bounds = bool(params.get("estm_with_bounds", 0))
                            dro = Young_Schur_dro_lmi(
                                                    vals=(upd, FROM_DATA, vect, augmented, inp), 
                                                    model=model, N_sims=N_sims,
                                                    api=api, noise=noise, 
                                                    reg_fro=reg_fro, reg_beta=reg_beta, real_Z_mats=real_Z_mats,
                                                    estm_with_bounds=estm_with_bounds,
                                                    )
                            
                            res, plant, Sigma_nom, other, num_violations = dro.run()
                            if estm_with_bounds: N_sims = dro.N_sims_new

                            problem_params = {
                                "Methodology": "YoungSchur",
                                "FROM_DATA": FROM_DATA, # True
                                "augmented": augmented,
                                "reg_fro": reg_fro,
                                "reg_beta": reg_beta,
                                "non_convex": False,
                                "N_sims": N_sims,
                            }
                            
                        A, Bw, Bu, Cy, Dyw, Cz, Dzw, Dzu = plant
                        plant = Plant(A=A, Bw=Bw, Bu=Bu, Cz=Cz, Dzw=Dzw, Dzu=Dzu, Cy=Cy, Dyw=Dyw)
                        ADD = True
                    
                    else:
                        real_Z_mats = True
                        res, plant, Sigma_nom, num_violations = Estm_dro_lmi(
                            api=api,
                            noise=noise,
                            model=model,
                            SOLVER=params.get("solver", "MOSEK"),
                            real_Z_mats=real_Z_mats,
                            N_sims=N_sims,
                        )

                        A, Bw, Bu, Cz, Dzw, Dzu, Cy, Dyw = plant.A, plant.Bw, plant.Bu, plant.Cz, plant.Dzw, plant.Dzu, plant.Cy, plant.Dyw

                        problem_params = {
                            "Methodology": "EstmMat",
                            "FROM_DATA": FROM_DATA, # True
                            "augmented": True,
                            "reg_fro": False,
                            "non_convex": True,
                            "N_sims": N_sims,
                        }

                        ADD = False

                else: 
                    real_Z_mats = True
                    dro = WFL(vals=(upd, FROM_DATA, vect, augmented, inp), 
                              Bw_type = params.get("plant", {}).get("Bw_mode", "ident"),
                              model=model, N_sims=N_sims,
                              api=api, noise=noise, reg_fro=reg_fro, 
                              real_Z_mats=real_Z_mats)
                    
                    res, P, Sigma_nom, other, num_violations = dro.run()

                    problem_params = {
                        "Methodology": "WFL",
                        "FROM_DATA": FROM_DATA, # True
                        "augmented": augmented,
                        "reg_fro": reg_fro,
                        "reg_beta": reg_beta,
                        "non_convex": non_convex,
                        "N_sims": N_sims,
                    }

                    A, Bw, Bu, Cy, Dyw, Cz, Dzw, Dzu = P
                    plant = Plant(A=A, Bw=Bw, Bu=Bu, Cz=Cz, Dzw=Dzw, Dzu=Dzu, Cy=Cy, Dyw=Dyw)
                    ADD = True
            
            t1 = time.perf_counter()
            cpu1 = self.proc.cpu_times()
            cpu1_total = cpu1.user + cpu1.system
            d_cpu = cpu1_total - cpu0_total


            Time = t1 - t0
            stress = d_cpu / Time if Time > 0 else 0.0
            tot_time += Time

            if res.status not in ("optimal", "optimal_inaccurate"):
                raise RuntimeError(f"DRO-LMI solve failed: status={res.status}")


            # 3) From (Pbar, Abar, Bbar, Cbar, Dbar) build composite (Acl, Bcl, Ccl, Dcl) in original coords

            Ac, Bc, Cc, Dc = recover.Mc_from_bar(res, plant)
            ctrl = Controller(Ac=Ac, Bc=Bc, Cc=Cc, Dc=Dc)
            Acl, Bcl, Ccl, Dcl = compose_closed_loop(plant, ctrl)
            plant_cl = Plant_cl(Acl=Acl, Bcl=Bcl, Ccl=Ccl, Dcl=Dcl)

            # 4) Recover (Ac, Bc, Cc, Dc) from composite and plant, with residual diagnostics
            # ctrl_rec, residuals = recover.recover_controller_from_closed_loop(plant, api, (Acl, Bcl, Ccl, Dcl))
            

            eig = np.linalg.eigvals(Acl)
            rho = float(np.max(np.abs(eig)))
            if rho < 1.00:
                STABLE = True
            i += 1
            break

        eig_cart = [
            {"re": float(ev.real), "im": float(ev.imag), "abs": float(np.abs(ev))}
            for ev in eig
        ]
        eig_polar = [
            {"abs": float(np.abs(ev)), "ang_rad": float(np.angle(ev))}
            for ev in eig
        ]

        warn_margin = 1.0 - 1e-6     # warn if too close to 1 from below
        hard_fail   = 1.0 + 1e-9     # fail if ≥ 1 within numerical wiggle

        if not np.isfinite(rho):
            print("Closed loop produced non-finite eigenvalues.")

        if rho >= hard_fail:
            print(f"Unstable: spectral radius {rho:.6g} ≥ 1.")

        if rho >= warn_margin:
            print(f"Warning: near-marginal stability, spectral radius {rho:.6g}.")

        if any(abs(eig) >= warn_margin):
            print("Warning: Closed-loop system may be unstable")
            print(abs(eig))
        else:
            print("Closed-loop system is stable")
            print(abs(eig))
        

        # 6) Simulate with the recovered controller using the SAME plant and nominal Σ
        #    If you prefer covariance inflation for robustness testing, replace Sigma_nom here.
        sim = cl.simulate_closed_loop(plant=plant, ctrl=ctrl, Sigma_w=Sigma_nom, gamma=gamma, init_cond=init_cond)
        out_npz = out + f"___closed_loop_run.npz"

        sim_composite = cl.simulate_composite(Pcl=plant_cl, Sigma_w=Sigma_nom, gamma=gamma, init_cond=init_cond)
        out_composite = out + f"___closed_loop_composite.npz"

        sim_cost = cl.simulate_Z_cost(Z=sim["Z"], plot=plot)
        self.final_cost = sim_cost["J"]
        print("\nFinal closed-loop cost J = ", self.final_cost)
        out_cost = out + f"___closed_loop_run_cost.npz"

        sim_snr = cl.simulate_ZW_snr(Z=sim["Z"], W=sim["W"], plot=plot)
        self.snr = sim_snr["snr_db"]
        print("\nGlobal SNR = ", self.snr)
        out_snr = out + f"___closed_loop_snr.npz"

        self.rho = rho
        self.lamda = res.lambda_opt
        self.Time = Time
        self.attempt = i
        self.stress = stress
        self.obj = res.obj_value
        self.solver = res.solver
        self.ratio_violation = num_violations[0] / num_violations[1]
        #dist = Disturbances(n=Bw.shape[1])

        # JSON
        payload = {
            "prob": "DDD" if FROM_DATA else "MBD",
            "meta": {
                "model": model,
                "disturbance_type": disturbance_type,
                "gamma": res.gamma,
                "objective": self.obj,
                "spectral_radius_Acl": rho,
                "Z_cost": self.final_cost,
                "lambda_opt": res.lambda_opt,
                "trace_Q_Sigma": np.trace(res.Q @ Sigma_nom),
                "real_Z_mats": real_Z_mats,
                "N_sims": N_sims,
                "SNR": self.snr,
            },
            "solver_performance": {
                "solver": self.solver,
                "status": res.status,
                "num_violations": int(num_violations[0]),
                "tot_constraints:": int(num_violations[1]),
                "Time_seconds": Time,
                "attempt": i,
                "Total_time": tot_time,
                "stress": self.stress,
            },
            "problem_params": problem_params,
            "controller": self.controller_to_dict(ctrl),
            "plant": self.plant_to_dict(plant),
            "plant_dims": {
                "nx": A.shape[0], 
                "nu": Bu.shape[1], 
                "nw": Bw.shape[1], 
                "ny": Cy.shape[0], 
                "nz": Cz.shape[0],
            },
            "initial_conds": {
                "X": sim["x_0"].tolist(), 
                "Xc": sim["xc_0"].tolist(), 
            },
            "composite_closed_loop": self.plant_cl_to_dict(plant_cl),
            "Acl_eigenvals": {
                "cartesian": eig_cart,
                "polar": eig_polar
            },            
            "disturbance": {
                "Sigma_nom": Sigma_nom.tolist(),
                #"Sigma": dist.Sigma_test.totlist() if dist.Sigma_test is not None else None,
            },
            "dro_variables": {
                "Q": None if res.Q is None else res.Q.tolist(),
                "X": None if res.X is None else res.X.tolist(),
                "Y": None if res.Y is None else res.Y.tolist(),
                "K": None if res.K is None else res.K.tolist(),
                "L": None if res.L is None else res.L.tolist(),
                "M": None if res.M is None else res.M.tolist(),
                "N": None if res.N is None else res.N.tolist(),
                "Pbar": None if res.Pbar is None else res.Pbar.tolist(),
                "Abar": None if res.Abar is None else res.Abar.tolist(),
                "Bbar": None if res.Bbar is None else res.Bbar.tolist(),
                "Cbar": None if res.Cbar is None else res.Cbar.tolist(),
                "Dbar": None if res.Dbar is None else res.Dbar.tolist(),
                "Tp": None if res.Tp is None else res.Tp.tolist(),
            },
        }

        if ADD: 
            if (approach == 'Young' or approach == 'Mats') and old or not old and reg_beta:
                D, E, B, S, T, R = other
                payload["Young_approach"] = {
                    "approach": approach,
                    "D": self._to_serializable(D),  # handles (DeltaA, DeltaB) as matrices/tuples
                    "E": self._to_serializable(E),  # handles (EAA, EAB) as matrices/tuples
                    "B": self._to_serializable(B),  # now OK if matrix, vector, or scalars tuple
                    "S": self._to_serializable(S),  # now OK if matrix
                    "T": self._to_serializable(T),  # now OK if matrix
                    "R": self._to_serializable(R),
                }

        out_json = out + f"___results_run.json"
        if save: 
            cl.save_npz(sim, str(out_npz))
            print(f"[saved] {out_npz}")
            cl.save_npz(sim_composite, str(out_composite))
            print(f"[saved] {out_composite}")
            cl.save_npz(sim_cost, str(out_cost))   
            print(f"[saved] {out_cost}")
            cl.save_npz(sim_snr, str(out_snr))
            print(f"[saved] {out_snr}")
            self.save_json(out_json, payload)
            print(f"[saved] {out_json}")

        # 7) Plot results
        if plot: 
            cl.plot_timeseries(sim=sim, save=save, out=out)
            cl.plot_composite(sim=sim_composite, save=save, out=out)

        Data = {
            'gamma': res.gamma, 'lambda': res.lambda_opt, 'Sigma_nom': Sigma_nom,
            'A_c': Ac, 'B_c':Bc, 'C_c': Cc, 'D_c': Dc, 'A_cl': Acl, 'rho': rho, 'var' : noise.var,
            #'rx': None if res.rx is None else res.rx, 'ry': None if res.ry is None else res.ry, 'rz': None if res.rz is None else res.rz
        }
        for key in ['gamma', 'lambda', 'Sigma_nom', 'A_c', 'B_c', 'C_c', 'D_c', 'A_cl']:#, 'rho', 'var', 'rx', 'ry', 'rz']:
            print(f"\n{key} =")
            print(Data[key])
        
        self.plant, self.ctrl, self.sim, self.Sigma_nom = plant, ctrl, sim, Sigma_nom

    def _return_final_infos(self):
        infos = {
            "J": self.final_cost,
            "lamda": self.lamda,
            "rho": self.rho,
            "time": self.Time, 
            "attempts": self.attempt,
            "stress": self.stress,
            "obj": self.obj,
            "solver": self.solver,
            "ratio_violation": self.ratio_violation,
            "snr": self.snr,
        }
        return infos

    def plant_to_dict(self, P: Plant):
        return {
            "A": P.A.tolist(),
            "Bw": P.Bw.tolist(),
            "Bu": P.Bu.tolist(),
            "Cz": P.Cz.tolist(),
            "Dzw": P.Dzw.tolist(),
            "Dzu": P.Dzu.tolist(),
            "Cy": P.Cy.tolist(),
            "Dyw": P.Dyw.tolist(),
        }

    def plant_cl_to_dict(self, P: Plant_cl):
        return {
            "Acl": P.Acl.tolist(),
            "Bcl": P.Bcl.tolist(), 
            "Ccl": P.Ccl.tolist(), 
            "Dcl": P.Dcl.tolist()
        }

    def controller_to_dict(self, C: Controller):
        return {"Ac": C.Ac.tolist(), "Bc": C.Bc.tolist(), "Cc": C.Cc.tolist(), "Dc": C.Dc.tolist()}

    def save_json(self, path: Path, payload: dict):
        with open(path, "w", encoding="utf-8") as f:
            json.dump(payload, f, indent=2)

    def get_snr_vars(self):
        return self.plant, self.ctrl, self.sim, self.Sigma_nom

    def _to_serializable(self, x):
        """
        Convert scalars / arrays / nested tuples-lists of arrays into JSON-safe types.
        - CVXPy expressions -> use .value first (caller’s job), otherwise we try best-effort.
        - NumPy arrays -> .tolist()
        - Scalars -> float/int/bool
        - Lists/Tuples -> map recursively
        """
        import numpy as np

        # cvxpy objects: try to unwrap gracefully if user forgot to pass .value
        try:
            import cvxpy as cp
            if isinstance(x, (cp.Expression, cp.atoms.atom.Atom)):
                x = x.value
        except Exception:
            pass

        if x is None:
            return None
        if isinstance(x, (list, tuple)):
            return [self._to_serializable(xx) for xx in x]
        if hasattr(x, "toarray"):             # e.g., scipy.sparse
            x = x.toarray()
        if isinstance(x, np.ndarray):
            return x.tolist()
        # numpy scalar
        if hasattr(x, "item") and callable(getattr(x, "item")):
            try:
                return x.item()
            except Exception:
                pass
        # plain scalars
        if isinstance(x, (int, float, bool, str)):
            return x
        # last resort: try numpy conversion
        try:
            import numpy as np
            return np.asarray(x).tolist()
        except Exception:
            # give up: string-ify so JSON doesn't choke
            return str(x)


