# dro_lmi.py
import numpy as np
import cvxpy as cp
from utils___systems import Plant, DROLMIResult
from utils___matrices import MatricesAPI


def build_and_solve_dro_lmi(
    plant: Plant,
    api: MatricesAPI,
    Sigma_nom: np.ndarray,
    gamma: float,
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


    Bw, Dzw, Dyw, nw, Sigma_nom = api._augment_matrices(Bw, Dzw, Dyw)

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
        reg = 0.0

    # Negative definiteness helpers (strict -> with epsilon)
    def negdef(M):
        return (M << -eps_def * np.eye(M.shape[0])) if additional_constraints else (M << 0)

    Iw = np.eye(nw)
    Iz = np.eye(nz)

    # handy zero of the right size
    def Z(r, c):
        return np.zeros((r, c)) # cp.Constant(np.zeros((r, c)))

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
    success = False
    print("Attempting to solve with MOSEK...")
    try:
        prob.solve(solver=cp.MOSEK, verbose=True, mosek_params={
            'MSK_DPAR_INTPNT_CO_TOL_PFEAS': 1e-8,
            'MSK_DPAR_INTPNT_CO_TOL_DFEAS': 1e-8,
            'MSK_DPAR_INTPNT_CO_TOL_REL_GAP': 1e-8,
            'MSK_DPAR_INTPNT_TOL_STEP_SIZE': 1e-6
        })
        print(f"MOSEK status: {prob.status}")
        if prob.status == cp.OPTIMAL:
            success = True
    except Exception as mosek_e:
        print(f"MOSEK error: {mosek_e}")

    if not success:
        print("MOSEK failed, trying SCS...")
        try:
            prob.solve(solver=cp.SCS, verbose=True, eps=1e-4, max_iters=100000)
            print(f"SCS status: {prob.status}")
            if prob.status in [cp.OPTIMAL, cp.OPTIMAL_INACCURATE]:
                success = True
                if prob.status == cp.OPTIMAL_INACCURATE:
                    print("Warning: SCS returned 'optimal_inaccurate'.")
            else:
                print(f"SCS failed with status: {prob.status}")
        except Exception as scs_e:
            print(f"SCS error: {scs_e}")

    if not success:
        print("Optimization error: Both solvers failed.")
        exit(1)


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
        status=status,
        obj_value=val,
        gamma=gamma,
        lambda_opt=lam_val,
        Q=Q_val, X=X_val, Y=Y_val, K=K_val, L=L_val, M=M_val, N=N_val,
        Pbar=Pbar_val, Abar=Abar_val, Bbar=Bbar_val, Cbar=Cbar_val, Dbar=Dbar_val
    )
