# optim_problem.py
import numpy as np
from numpy.linalg import eigvals, norm
from scipy.optimize import minimize
from compose import compose_closed_loop
from systems import Plant, Controller

def is_stable_discrete(A, spectral_radius_tol=0.999):
    rho = max(abs(eigvals(A)))
    return float(rho.real) < spectral_radius_tol, float(rho.real)

def simulate_cost(plant: Plant, ctrl: Controller, Sigma_w, T=3000, burnin=300, seed=0):
    """
    Long-run average E[||z_t||^2] under zero-mean Gaussian w_t ~ N(0, Σ_eff).
    We estimate by Monte Carlo with one long trajectory. Increase T for better accuracy.
    """
    rng = np.random.default_rng(seed)
    nx, nw, nu, nz, ny = plant.dims()
    nxc = ctrl.dims()

    A_cl, B_cl, C_cl, D_cl = compose_closed_loop(plant, ctrl)

    X = np.zeros((nx + nxc, 1))
    cost_accum = 0.0
    cnt = 0

    # Precompute a Cholesky (or fallback) for sampling
    try:
        L = np.linalg.cholesky(Sigma_w)
    except np.linalg.LinAlgError:
        # jitter if needed
        eps = 1e-9
        L = np.linalg.cholesky(Sigma_w + eps*np.eye(Sigma_w.shape[0]))

    for t in range(T):
        w = (L @ rng.standard_normal((nw, 1))).astype(float)
        # output
        z = C_cl @ X + D_cl @ w
        # state update
        X = A_cl @ X + B_cl @ w

        if t >= burnin:
            cost_accum += float((z.T @ z).item())
            cnt += 1

    return cost_accum / max(cnt, 1)

def pack_vars(Ac, Bc, Cc, Dc):
    return np.concatenate([Ac.flatten(), Bc.flatten(), Cc.flatten(), Dc.flatten()])

def unpack_vars(theta, shapes):
    Ac_shape, Bc_shape, Cc_shape, Dc_shape = shapes
    nA = np.prod(Ac_shape)
    nB = np.prod(Bc_shape)
    nC = np.prod(Cc_shape)
    nD = np.prod(Dc_shape)
    assert theta.size == nA + nB + nC + nD
    i0 = 0
    Ac = theta[i0:i0+nA].reshape(Ac_shape); i0 += nA
    Bc = theta[i0:i0+nB].reshape(Bc_shape); i0 += nB
    Cc = theta[i0:i0+nC].reshape(Cc_shape); i0 += nC
    Dc = theta[i0:i0+nD].reshape(Dc_shape)
    return Ac, Bc, Cc, Dc

def stability_project(ctrl: Controller, clip=0.995):
    """
    If 𝒜 is unstable, shrink controller dynamics slightly.
    Crude but effective: scale Ac toward zero and Dc toward small gain.
    This is a projection heuristic; penalization is also applied in the objective.
    """
    Ac, Bc, Cc, Dc = ctrl.Ac.copy(), ctrl.Bc.copy(), ctrl.Cc.copy(), ctrl.Dc.copy()
    # simple damping on Ac, mild on Dc
    Ac *= clip
    Dc *= clip
    return Controller(Ac, Bc, Cc, Dc)

def objective(theta, shapes, plant: Plant, Sigma_w, seeds, T, burnin, rho_penalty=1e4):
    Ac, Bc, Cc, Dc = unpack_vars(theta, shapes)
    ctrl = Controller(Ac, Bc, Cc, Dc)

    # Compose and check stability
    A_cl, B_cl, C_cl, D_cl = compose_closed_loop(plant, ctrl)
    stable, rho = is_stable_discrete(A_cl)
    penalty = 0.0
    if not stable:
        penalty = rho_penalty * (rho - 0.999)**2

    # Monte Carlo average over seeds
    vals = []
    for s in seeds:
        vals.append(simulate_cost(plant, ctrl, Sigma_w, T=T, burnin=burnin, seed=s))
    val = float(np.mean(vals))
    return val + penalty

def optimize_controller(plant: Plant,
                        ctrl_init: Controller,
                        Sigma_w,
                        T=2500,
                        burnin=300,
                        seeds=(0,1,2,3),
                        maxiter=80):
    shapes = (ctrl_init.Ac.shape, ctrl_init.Bc.shape, ctrl_init.Cc.shape, ctrl_init.Dc.shape)
    theta0 = pack_vars(ctrl_init.Ac, ctrl_init.Bc, ctrl_init.Cc, ctrl_init.Dc)

    # Bounds can keep things sane. Here we use loose symmetric bounds.
    bnd = 5.0
    bounds = [(-bnd, bnd)] * theta0.size

    res = minimize(
        fun=lambda th: objective(th, shapes, plant, Sigma_w, seeds, T, burnin),
        x0=theta0,
        method="L-BFGS-B",
        bounds=bounds,
        options={"maxiter": maxiter, "ftol": 1e-6}
    )

    Ac, Bc, Cc, Dc = unpack_vars(res.x, shapes)
    ctrl = Controller(Ac, Bc, Cc, Dc)

    # Last-chance projection if unstable
    A_cl, *_ = compose_closed_loop(plant, ctrl)
    stable, _ = is_stable_discrete(A_cl)
    if not stable:
        ctrl = stability_project(ctrl)

    return ctrl, res
