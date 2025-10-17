# dro_lmi.py
import numpy as np
import cvxpy as cp
from dataclasses import dataclass
from systems import Plant

@dataclass
class DROLMIResult:
    status: str
    obj_value: float | None
    gamma: float
    lambda_opt: float | None
    Q: np.ndarray | None
    X: np.ndarray | None
    Y: np.ndarray | None
    K: np.ndarray | None
    L: np.ndarray | None
    M: np.ndarray | None
    N: np.ndarray | None
    Pbar: np.ndarray | None
    Abar: np.ndarray | None
    Bbar: np.ndarray | None
    Cbar: np.ndarray | None
    Dbar: np.ndarray | None

def build_and_solve_dro_lmi(
    plant: Plant,
    Sigma_nom: np.ndarray,
    gamma: float,
    model: str = "correlated",  # or "independent"
    eps_def: float = 1e-5,
    alpha_cap: float = 1e2,  # keep X, Y from exploding
    fro_cap: float = 1e2,  # keep K, L, M, N from exploding
    solver: str | None = None,
    verbose: bool = False,
) -> DROLMIResult:
    """
    Builds and solves the DRO-LMI you specified.
    model = "correlated"  implements (1)
    model = "independent" implements (2)
    """
    A, Bw, Bu, Cz, Dzw, Dzu, Cy, Dyw = plant.A, plant.Bw, plant.Bu, plant.Cz, plant.Dzw, plant.Dzu, plant.Cy, plant.Dyw
    nx, nw, nu, nz, ny = plant.dims()

    # Decision variables
    lam = cp.Variable(nonneg=True, name="lambda")
    Q = cp.Variable((nw, nw), PSD=True, name="Q")

    X = cp.Variable((nx, nx), symmetric=True, name="X")
    Y = cp.Variable((nx, nx), symmetric=True, name="Y")
    K = cp.Variable((nx, nx), name="K")
    L = cp.Variable((nx, ny), name="L")
    M = cp.Variable((nu, nx), name="M")
    N = cp.Variable((nu, ny), name="N")

    # Construct mathbb{P} = [[Y, I], [I, X]]
    I_x = np.eye(nx)
    Pbar = cp.bmat([[Y, I_x],
                    [I_x, X]])

    # Construct mathbb{A}, mathbb{B}, mathbb{C}, mathbb{D}
    AY_plus_BuM   = A @ Y + Bu @ M                  # nx x nx
    A_plus_BuNCy  = A + Bu @ N @ Cy                 # nx x nx
    XBw_plus_LDyw = X @ Bw + L @ Dyw                # nx x nw
    CzY_plus_DzuM = Cz @ Y + Dzu @ M                # nz x nx
    Cz_plus_DzuNCy= Cz + Dzu @ N @ Cy               # nz x nx
    Bw_plus_BuNDy = Bw + Bu @ N @ Dyw               # nx x nw
    Dzw_plus_DzuND= Dzw + Dzu @ N @ Dyw             # nz x nw

    Abar = cp.bmat([[AY_plus_BuM,   A_plus_BuNCy],
                    [K,             X @ A + L @ Cy]])
    Bbar = cp.vstack([Bw_plus_BuNDy,
                      XBw_plus_LDyw])
    Cbar = cp.hstack([CzY_plus_DzuM, Cz_plus_DzuNCy])
    Dbar = Dzw_plus_DzuND

    cons = []
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
    obj = cp.trace(Q @ Sigma_nom) + lam * (gamma ** 2) + reg

    # Negative definiteness helpers (strict -> with epsilon)
    def negdef(M):
        return M << -eps_def * np.eye(M.shape[0])

    Iw = np.eye(nw)
    Iz = np.eye(nz)

    # handy zero of the right size
    def Z(r, c):
        return cp.Constant(np.zeros((r, c)))

    if model.lower() in ["correlated", "corr", "1"]:
        # Block sizes by columns: [2nx, nw, nw, 2nx, nz]
        # Row heights: [2nx, nw, nw, 2nx, nz]
        big_corr = cp.bmat([
            # row 1: size 2nx
            [ -Pbar,          Z(2*nx, nw),    Z(2*nx, nw),    Abar.T,           Cbar.T          ],
            # row 2: size nw
            [ Z(nw, 2*nx),   -lam*Iw,         lam*Iw,         Bbar.T,           Dbar.T          ],
            # row 3: size nw
            [ Z(nw, 2*nx),    lam*Iw,        -Q - lam*Iw,     Z(nw, 2*nx),      Z(nw, nz)       ],
            # row 4: size 2nx
            [  Abar,           Bbar,           Z(2*nx, nw),   -Pbar,             Z(2*nx, nz)    ],
            # row 5: size nz
            [  Cbar,           Dbar,           Z(nz, nw),      Z(nz, 2*nx),     -Iz             ],
        ])
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


    prob = cp.Problem(cp.Minimize(obj), cons)

    # Pick solver
    solve_kwargs = dict(verbose=verbose)
    if solver is not None:
        solve_kwargs["solver"] = solver
        if solver.upper() == "SCS":
            solve_kwargs.update(dict(max_iters=150000, eps=5e-7))
            solve_kwargs.update(dict(acceleration_lookback=50, normalize=True, scale=5.0))
        if solver.upper() == "MOSEK":
            # if you have it, enjoy your life
            pass
    else:
        # default: try MOSEK else SCS
        try:
            solve_kwargs["solver"] = cp.MOSEK
        except Exception:
            solve_kwargs["solver"] = cp.SCS
            solve_kwargs.update(dict(max_iters=50000, eps=1e-6))

    val = None
    lam_val = None
    Q_val = X_val = Y_val = K_val = L_val = M_val = N_val = None
    Pbar_val = Abar_val = Bbar_val = Cbar_val = Dbar_val = None

    try:
        prob.solve(**solve_kwargs)
    except Exception as e:
        return DROLMIResult(
            status=f"solve_error: {e}",
            obj_value=None, gamma=gamma, lambda_opt=None,
            Q=None, X=None, Y=None, K=None, L=None, M=None, N=None,
            Pbar=None, Abar=None, Bbar=None, Cbar=None, Dbar=None
        )

    status = prob.status
    if status in (cp.OPTIMAL, cp.OPTIMAL_INACCURATE):
        val = float(prob.value)
        lam_val = float(lam.value)
        Q_val = Q.value
        X_val, Y_val = X.value, Y.value
        K_val, L_val, M_val, N_val = K.value, L.value, M.value, N.value
        Pbar_val = Pbar.value
        Abar_val, Bbar_val, Cbar_val, Dbar_val = Abar.value, Bbar.value, Cbar.value, Dbar.value

    return DROLMIResult(
        status=status,
        obj_value=val,
        gamma=gamma,
        lambda_opt=lam_val,
        Q=Q_val, X=X_val, Y=Y_val, K=K_val, L=L_val, M=M_val, N=N_val,
        Pbar=Pbar_val, Abar=Abar_val, Bbar=Bbar_val, Cbar=Cbar_val, Dbar=Dbar_val
    )
