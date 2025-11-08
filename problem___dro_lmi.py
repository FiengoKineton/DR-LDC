# dro_lmi.py
import numpy as np
import cvxpy as cp
import matplotlib.pyplot as plt
from utils___systems import Plant, DROLMIResult, Noise, Data
from utils___matrices import MatricesAPI, recover_deltas
from utils___ambiguity import Disturbances
from utils___simulate import Open_Loop
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

def mean_dict(datasets):
    """Elementwise mean over a list of dataset dicts with identical shapes."""
    out = {}
    T = min(d["meta"]["T"] for d in datasets)
    keys = ["X","U","Y","Z","X_next"]
    for k in keys:
        arrs = []
        for d in datasets:
            A = d[k]
            arrs.append(A[..., :T-1] if k in {"X_reg","X_next"} else A[..., :T])
        out[k] = np.stack(arrs, axis=0).mean(axis=0)
    out["meta"] = {**datasets[0]["meta"], "T": T, "N": len(datasets)}
    return out

def select_representative_run(datasets, keys=("X","U","Y","Z","X_next"), weights=None):
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

def plot_first3_and_mean(datasets, key="X", out=None, title_prefix=None,
                         show_band=True, symmetric_ylim=True):
    if not datasets:
        raise ValueError("datasets is empty.")

    # Align horizon for datasets
    T_min = min(d["meta"]["T"] for d in datasets)
    use_Tm1 = key in {"X_next", "X_reg"}
    Teff = T_min - 1 if use_Tm1 else T_min
    if Teff <= 0:
        raise ValueError("Effective horizon <= 0; check inputs.")

    # Stack datasets: (N, n, Teff) and get per-dataset row-mean (N, Teff)
    stack = np.stack([d[key][..., :Teff] for d in datasets], axis=0)
    per_ds_mean = np.nanmean(stack, axis=1)
    N = per_ds_mean.shape[0]
    global_mean = np.nanmean(per_ds_mean, axis=0)
    global_std  = np.nanstd(per_ds_mean, axis=0, ddof=1) if N > 1 else np.zeros_like(global_mean)

    # Time vector
    t = datasets[0].get("t", None)
    t = t[:Teff] if (t is not None and t.shape[-1] >= Teff) else np.arange(Teff)

    # Optional overlay from `out` using the SAME key
    overlay = None
    if out is not None and key in out:
        arr = np.asarray(out[key])
        overlay = arr if arr.ndim == 1 else np.nanmean(arr, axis=0)
        # clip/align overlay length to Teff
        overlay = overlay[:Teff] if overlay.shape[-1] >= Teff else np.pad(
            overlay, (0, Teff - overlay.shape[-1]), mode="edge"
        )

    # Y-limits from curves actually shown
    curves = [per_ds_mean[i] for i in range(min(3, N))] + [global_mean]
    if overlay is not None:
        curves.append(overlay)
    ymin = np.min([np.nanmin(c) for c in curves])
    ymax = np.max([np.nanmax(c) for c in curves])
    L = float(max(abs(ymin), abs(ymax))) or 1.0

    fig, axes = plt.subplots(2, 2, figsize=(12, 8), sharex=True)
    axes = axes.flatten()

    def plot_mean(ax, m, title):
        ax.plot(t, m, label="dataset avg")
        if show_band:
            ax.fill_between(t, global_mean - global_std, global_mean + global_std,
                            alpha=0.2, linewidth=0, label="±1σ (global)")
        if overlay is not None:
            ax.plot(t, overlay, linewidth=2.0, label=f"{key} (precomputed)")
        ax.set_title(title if title else "")
        ax.set_xlabel("time"); ax.set_ylabel(f"mean({key})")
        ax.set_ylim((-L, L) if symmetric_ylim else (ymin, ymax))
        ax.grid(True, alpha=0.3); ax.legend(loc="best")

    for i in range(min(3, N)):
        plot_mean(axes[i], per_ds_mean[i],
                  f"{title_prefix+' - ' if title_prefix else ''}{key}: dataset {i+1}")

    plot_mean(axes[3], global_mean,
              f"{title_prefix+' - ' if title_prefix else ''}{key}: mean over N={N}")

    fig.suptitle(f"{key}: first three dataset means + global{(' + overlay' if overlay is not None else '')}", y=0.98)
    fig.tight_layout()
    return fig, axes



def build_and_solve_dro_lmi_upd(
    api: MatricesAPI,
    #data: Data,
    vals: tuple,
    noise: Noise,
    model: str = "correlated",  # or "independent"
    eps: float = 1e-5,
    mu: float = 1e-3,  # keep K, L, M, N from exploding
    mhu_x: float = 1.0,
    mhu_y: float = 0.5,
    mhu_z: float = 0.5,
    approach: str = "DeePC",
    d: bool = False,
) -> DROLMIResult:
    """
    Builds and solves the DRO-LMI you specified.
    model = "correlated"  implements (1)
    model = "independent" implements (2)
    """
    gamma, var = noise.gamma, noise.var
    upd, FROM_DATA, plot = vals

    if approach == "DeePC":
        data = api.get_system(FROM_DATA=FROM_DATA, gamma=gamma, upd=upd)
        x, x_next, u, y, z = data.get_data()
    else: 
        op = Open_Loop(MAKE_DATA=False, EVAL_FROM_PATH=False, DATASETS=True, N=100)
        datasets = op.datasets


        avg = select_representative_run(datasets)
        x, u, y, z, x_next = avg["X"], avg["U"], avg["Y"], avg["Z"], avg["X_next"]
        T, nx, nu, ny, nz = x.shape[1], x.shape[0], u.shape[0], y.shape[0], z.shape[0]

        if plot:
            plot_first3_and_mean(datasets, out=avg, key="X", title_prefix="Closed-loop")
            plot_first3_and_mean(datasets, out=avg, key="U", title_prefix="Closed-loop")
            plot_first3_and_mean(datasets, out=avg, key="Y", title_prefix="Closed-loop")
            plot_first3_and_mean(datasets, out=avg, key="Z", title_prefix="Closed-loop")


    
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
        return (M << -eps * np.eye(M.shape[0]))

    def _val(x):
        if x is None:
            return None
        return float(x) if np.isscalar(x) else x

    def _fro(M):
        return cp.norm(M, 'fro')

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
            # Bw_hat = Up @ np.diag(sp_sqrt)
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
    
    elif approach == "Young" or approach == "Mats":
        Dx = np.vstack([x, u])
        Ox = x_next @ _pseudo_inv(Dx)
        Ax, Bu = Ox[:, :nx], Ox[:, nx:nx+nu]
   

        Bw, R, *_ = estimate_Bw_from_residuals(Ax_hat=Ax, Bu_hat=Bu, mode="factor")
        nw = Bw.shape[1]
        w = _pseudo_inv(Bw) @ R
        d = Disturbances(n=nw)
        Sigma_nom = d.estm_Sigma_nom(w)
        print(f"Estimated Sigma_nom:\n{Sigma_nom}")
        gamma, *_ = d.estimate_gamma_with_ci(w)
        print(f"Estimated disturbance dimension nw: {nw}, gamma: {gamma}")

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
        Cz, Dzu, Dzw = Oz[:, :nx], Oz[:, nx:nx+nu], Oz[:, nx+nu:nx+nu+nw]
        #Cz, Dzw, Dzu, *_ = api.build_out_matrices(nw=nw)

    else:
        raise ValueError("approach must be 'DeePC', 'Young' or 'Mats'")


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
    cons += [P >> eps * I(2*nx)]

    obj_dro = cp.trace(Q @ Sigma_nom) + lam * (gamma ** 2)
    reg = 0.0

    if approach == 'DeePc':
        obj_est = mhu_x * (cp.trace((Dx @ Dx.T + rx * I(Dx.shape[0])) @ Gx) - cp.log_det(Gx)) + \
                    mhu_y * (cp.trace((Dy @ Dy.T + ry * I(Dy.shape[0])) @ Gy) - cp.log_det(Gy)) + \
                    mhu_z * (cp.trace((Dz @ Dz.T + rz * I(Dz.shape[0])) @ Gz) - cp.log_det(Gz))

        obj_dro += obj_est
    elif approach == "Young":
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
    
    elif approach == "Mats":
        # 1) residual anisotropy (directions and weights)
        U_A, s_A, w_A = _residual_anisotropy_weights(R, floor=1e-12, mode="sqrt")
        # Optional: cap tiny directions to avoid numerical issues
        w_A = np.maximum(w_A, 1e-6)

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
        beta_ab = np.sqrt(M_norm**2 + N_norm**2 * Cy_norm**2) * beta_b
        beta_AB = cp.Parameter(nonneg=True, value=float(np.clip(beta_ab, 0.0, 1e3)))

        # 5) epigraphs for each direction:
        #    For every i, [[s_i, β_i],[β_i, τ_i]] >= 0
        for i in range(nx):
            cons += [cp.bmat([[s_AA[i],      beta_AA[i]],
                            [beta_AA[i],   tau_AA[i]]]) >> 0]

        # 6) keep AB part as-is for now (still isotropic), or do the same for AB if you want
        tau_AB = cp.Variable(nonneg=True, name="tau_ab")  # unchanged
        s_AB   = cp.Variable(nonneg=True, name="s_ab")
        S_AB   = S_AA_base                                # you can also rotate with a basis for AB if desired

        # Existing spectral norm epigraphs
        tK, consK = spectral_norm_epigraph(K, "K")
        tL, consL = spectral_norm_epigraph(L, "L")
        tM, consM = spectral_norm_epigraph(M, "M")
        tN, consN = spectral_norm_epigraph(N, "N")
        tP, consP = spectral_norm_epigraph(P, "P")

        # 7) update regularization terms (small nudges to keep slacks bounded)
        mhu_AA, mhu_AB = 1e-3, 1e-3
        rhoK, rhoL, rhoM, rhoN, rhoP = 1e-3, 1e-3, 1e-3, 1e-3, 0.0
        # Sum of vector slacks replaces scalar s_AA and τ_AA
        reg += mhu_AA * (cp.sum(s_AA) + cp.sum(tau_AA / (beta_AA**2 + 1e-18)))
        reg += mhu_AB * (s_AB + tau_AB / (beta_ab**2 + 1e-18))
        reg += rhoK * tK + rhoL * tL + rhoM * tM + rhoN * tN + rhoP * tP * (cp.sum(beta_AA) + beta_ab)**2

        # 8) state block: minimally invasive variant (keeps your scalar-identity structure)
        state_blk = -P + (cp.sum(tau_AA) + tau_AB) * I(2*nx)

        # 9) use the rotated selector in the big LMI wherever S_AA appears
        #    Replace S_AA with S_AA_rot, and the -s_AA*I block with -diag(s_AA_vec)

    else:
        pass

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
                [ -P,            Z(2*nx, nw),  Z(2*nx, nw),  A.T,          C.T,         Z(2*nx, nx),    Z(2*nx, nx)     ],
                [ Z(nw,2*nx),   -lam*Iw,       lam*Iw,       B.T,          D.T,         Z(nw,   nx),    Z(nw,   nx)     ],
                [ Z(nw,2*nx),    lam*Iw,      -Q - lam*Iw,   Z(nw,2*nx),   Z(nw,  nz),  Z(nw,   nx),    Z(nw,   nx)     ],
                [ A,             B,            Z(2*nx, nw),  state_blk,    Z(2*nx,nz),  S_AA,           S_AB            ],
                [ C,             D,            Z(nz,  nw),   Z(nz, 2*nx), -Iz,          Z(nz,   nx),    Z(nz,   nx)     ],
                [ Z(nx,2*nx),    Z(nx,  nw),   Z(nx,  nw),   S_AA.T,       Z(nx,  nz), -s_AA * Ix,      Z(nx,   nx)     ],
                [ Z(nx,2*nx),    Z(nx,  nw),   Z(nx,  nw),   S_AB.T,       Z(nx,  nz),  Z(nx, nx),     -s_AB * Ix       ],
            ])
        elif approach == "Mats":
            big_corr = cp.bmat([
                [ -P,            Z(2*nx, nw),  Z(2*nx, nw),  A.T,          C.T,          S_AA,           S_AB            ],
                [ Z(nw,2*nx),   -lam*Iw,       lam*Iw,       B.T,          D.T,          Z(nw,   nx),    Z(nw,   nx)     ],
                [ Z(nw,2*nx),    lam*Iw,      -Q - lam*Iw,   Z(nw,2*nx),   Z(nw,  nz),   Z(nw,   nx),    Z(nw,   nx)     ],
                [ A,             B,            Z(2*nx, nw),  state_blk,    Z(2*nx,nz),   S_AA,           S_AB            ],
                [ C,             D,            Z(nz,  nw),   Z(nz, 2*nx), -Iz,           Z(nz,   nx),    Z(nz,   nx)     ],
                [ S_AA.T,        Z(nx,  nw),   Z(nx,  nw),   S_AA.T,       Z(nx,  nz),  -cp.diag(s_AA),  Z(nx, nx)       ],
                [ Z(nx,2*nx),    Z(nx,  nw),   Z(nx,  nw),   S_AB.T,       Z(nx,  nz),   Z(nx, nx),     -s_AB * Ix       ],
            ])

        cons += [negdef(big_corr)]
        cons += [negdef(state_blk)]

    elif model.lower() in ["independent", "indep", "2"]:
        if approach == "DeePC":
            blk1 = cp.bmat([
                [-P,    A.T,            C.T         ],
                [ A,   -P,              Z(2*nx, nz) ],
                [ C,    Z(nz, 2*nx),   -Iz          ],
            ])  # Tot size: (4nx + nz) x (4nx + nz)
        elif approach == "Young":
            blk1 = cp.bmat([
                [ -P,           A.T,          C.T,          Z(2*nx, nx),    Z(2*nx, nx)     ],
                [  A,           state_blk,    Z(2*nx, nz),  S_AA,           S_AB            ],
                [  C,           Z(nz, 2*nx), -Iz,           Z(nz, nx),      Z(nz, nx)       ],
                [  Z(nx,2*nx),  S_AA.T,       Z(nx, nz),   -s_AA * Ix,      Z(nx,   nx)     ],
                [  Z(nx,2*nx),  S_AB.T,       Z(nx, nz),    Z(nx,   nx),   -s_AB * Ix       ],
            ])   
        elif approach == "Mats":
            blk1 = cp.bmat([
                [ -P,           A.T,          C.T,          S_AA,           S_AB            ],
                [  A,           state_blk,    Z(2*nx, nz),  S_AA,           S_AB            ],
                [  C,           Z(nz, 2*nx), -Iz,           Z(nz, nx),      Z(nz, nx)       ],
                [ S_AA.T,       S_AA.T,       Z(nx, nz),   -cp.diag(s_AA),  Z(nx, nx)       ],
                [  Z(nx,2*nx),  S_AB.T,       Z(nx, nz),    Z(nx, nx),     -s_AB * Ix       ],
            ])
        
        blk2 = cp.bmat([
            [-lam*Iw,   lam*Iw,         B.T,            D.T         ],
            [ lam*Iw,  -Q - lam*Iw,     Z(nw, 2*nx),    Z(nw, nz)   ],
            [ B,        Z(2*nx, nw),   -P,              Z(2*nx, nz) ],
            [ D,        Z(nz, nw),      Z(nz, 2*nx),   -Iz          ],
        ])  # Tot size: (2nx + 2nw + nz) x (2nx + 2nw + nz)

        cons += [negdef(blk1)]
        cons += [negdef(blk2)]
        if d: cons += [negdef(state_blk)]
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
            #'MSK_DPAR_INTPNT_TOL_STEP_SIZE': 1e-6,
            'MSK_IPAR_INTPNT_SCALING': 1, # 0: no scaling, 1: geometric mean, 2: equilibrate
        })
        print(f"MOSEK status: {prob.status}")
        if prob.status == cp.OPTIMAL:
            success_MOSEK = True
    except Exception as mosek_e:
        print(f"MOSEK error: {mosek_e}")

    if not success_MOSEK and 0:
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

    if approach == 'Young' or approach == "Mats":
        DeltaA, DeltaB, EAA, EAB = recover_deltas(
            P=P_val, X=X_val, Y=Y_val, M=M_val, N=N_val, Cy=Cy,
            Ahat=Ax, Buhat=Bu,
            beta_AA=np.mean(beta_AA.value), beta_AB=beta_AB.value,
        )

        Ax = Ax + DeltaA
        Bu = Bu + DeltaB

    P = (Ax, Bw, Bu, Cy, Dyw, Cz, Dzw, Dzu)

    if approach == 'DeePC':
        rx_val, ry_val, rz_val \
            = _val(rx.value), _val(ry.value), _val(rz.value)
        other = (rx_val, ry_val, rz_val)
    elif approach == 'Young' or approach == "Mats":
        other = (DeltaA, DeltaB), (EAA, EAB), (beta, beta_a, beta_b, beta_AA.value, beta_ab), (s_AA.value, s_AB.value), (tau_AA.value, tau_AB.value)

    return dro, P, Sigma_nom, other

# ================================================================================================

