import numpy as np
import cvxpy as cp

def synthesize_state_feedback(A, B, Cz, Dzu, Sigma_nom, gamma, solver="MOSEK"):
    """
    State-feedback DRC via a convex surrogate of the moment-SDP.
    Decision variables: X (Lyapunov), Y = K X, and disturbance moment blocks (Σ).
    Enforces:
      - Discrete-time Lyapunov LMI in (X,Y) for stability of A+BK.
      - Σ_xx <= X linking stationary covariance to Lyapunov bound.
      - Wasserstein/Gelbrich: Tr(Σ_ww + Σ_wH - 2 Σ_wwH) <= gamma^2
      - Σ_wH = Sigma_nom  (nominal covariance for the auxiliary variable)
      - PSD of the assembled Σ.
    Objective: minimize Tr((Cz + Dzu K) X (Cz + Dzu K)^T).

    Returns dict with keys: status, optval, K.
    """
    n, m = B.shape

    # Lyapunov variables
    X = cp.Variable((n, n), PSD=True)
    Y = cp.Variable((m, n))  # Y = K X

    # Moment blocks for [x; w; w_hat]
    S_xx   = cp.Variable((n, n), PSD=True)
    S_xw   = cp.Variable((n, n))
    S_ww   = cp.Variable((n, n), PSD=True)
    S_xwH  = cp.Variable((n, n))
    S_wwH  = cp.Variable((n, n))
    S_wHwH = cp.Variable((n, n), PSD=True)

    eps = 1e-6
    cons = [X >> eps * np.eye(n)]

    # Stability surrogate: [[X, (A X + B Y)^T], [A X + B Y, X]] >= 0
    M = cp.bmat([
        [X,                 (A @ X + B @ Y).T],
        [A @ X + B @ Y,     X]
    ])
    cons += [M >> eps * np.eye(2 * n)]

    # Stationary covariance upper bound
    cons += [S_xx << X]

    # Wasserstein/Gelbrich surrogate: E||w - w_hat||^2 <= gamma^2
    gelbrich = cp.trace(S_ww + S_wHwH - 2 * S_wwH)
    cons += [gelbrich <= gamma**2]

    # Nominal covariance for w_hat
    cons += [S_wHwH == Sigma_nom]

    # PSD of full Σ
    S = cp.bmat([
        [S_xx,    S_xw,    S_xwH],
        [S_xw.T,  S_ww,    S_wwH],
        [S_xwH.T, S_wwH.T, S_wHwH]
    ])
    cons += [S >> 0]

    # Recover K = Y X^{-1}
    K = Y @ cp.inverse(X)
    Cz_cl = Cz + Dzu @ K

    # Upper bound robust H2-like cost
    obj = cp.Minimize(cp.trace(Cz_cl @ X @ Cz_cl.T))
    prob = cp.Problem(obj, cons)
    try:
        prob.solve(solver=solver, verbose=False)
    except Exception:
        prob.solve(solver="SCS", verbose=False, max_iters=20000)

    status = prob.status
    K_val = None
    if status in ("optimal", "optimal_inaccurate"):
        Xv = X.value
        Yv = Y.value
        if Xv is not None:
            try:
                K_val = Yv @ np.linalg.inv(Xv)
            except np.linalg.LinAlgError:
                K_val = None
    return dict(status=status, optval=prob.value, K=K_val)
