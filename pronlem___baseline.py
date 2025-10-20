import numpy as np
from utilis___matrices import MatricesAPI, compose_closed_loop

from numpy.linalg import eigvals
from scipy.optimize import minimize
from utilis___systems import Plant, Controller, Ambiguity


# ------------------------- SUPPORTING FUNCTIONS FOR OPTIMIZATION --------

class Optim_Problem():
    def __init__(self):
        pass

    def is_stable_discrete(self, A, spectral_radius_tol=0.999):
        rho = max(abs(eigvals(A)))
        return float(rho.real) < spectral_radius_tol, float(rho.real)

    def simulate_cost(self, plant: Plant, ctrl: Controller, Sigma_w, T=3000, burnin=300, seed=0):
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

    def pack_vars(self, Ac, Bc, Cc, Dc):
        return np.concatenate([Ac.flatten(), Bc.flatten(), Cc.flatten(), Dc.flatten()])

    def unpack_vars(self, theta, shapes):
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

    def stability_project(self, ctrl: Controller, clip=0.995):
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

    def objective(self, theta, shapes, plant: Plant, Sigma_w, seeds, T, burnin, rho_penalty=1e4):
        Ac, Bc, Cc, Dc = self.unpack_vars(theta, shapes)
        ctrl = Controller(Ac, Bc, Cc, Dc)

        # Compose and check stability
        A_cl, B_cl, C_cl, D_cl = compose_closed_loop(plant, ctrl)
        stable, rho = self.is_stable_discrete(A_cl)
        penalty = 0.0
        if not stable:
            penalty = rho_penalty * (rho - 0.999)**2

        # Monte Carlo average over seeds
        vals = []
        for s in seeds:
            vals.append(self.simulate_cost(plant, ctrl, Sigma_w, T=T, burnin=burnin, seed=s))
        val = float(np.mean(vals))
        return val + penalty

    def optimize_controller(self, plant: Plant,
                            ctrl_init: Controller,
                            Sigma_w,
                            T=2500,
                            burnin=300,
                            seeds=(0,1,2,3),
                            maxiter=80):
        shapes = (ctrl_init.Ac.shape, ctrl_init.Bc.shape, ctrl_init.Cc.shape, ctrl_init.Dc.shape)
        theta0 = self.pack_vars(ctrl_init.Ac, ctrl_init.Bc, ctrl_init.Cc, ctrl_init.Dc)

        # Bounds can keep things sane. Here we use loose symmetric bounds.
        bnd = 5.0
        bounds = [(-bnd, bnd)] * theta0.size

        res = minimize(
            fun=lambda th: self.objective(th, shapes, plant, Sigma_w, seeds, T, burnin),
            x0=theta0,
            method="L-BFGS-B",
            bounds=bounds,
            options={"maxiter": maxiter, "ftol": 1e-6}
        )

        Ac, Bc, Cc, Dc = self.unpack_vars(res.x, shapes)
        ctrl = Controller(Ac, Bc, Cc, Dc)

        # Last-chance projection if unstable
        A_cl, *_ = compose_closed_loop(plant, ctrl)
        stable, _ = self.is_stable_discrete(A_cl)
        if not stable:
            ctrl = self.stability_project(ctrl)

        return ctrl, res


# ------------------------- SINGLE RUN FUNCTION --------------------------

def run_once(seed_plant: int = 7,
             T_cost_init: int = 2000,
             T_cost_opt: int = 2500,
             burnin_init: int = 200,
             burnin_opt: int = 300,
             seeds_opt = (0,1,2,3,4),
             maxiter: int = 120, 
             model="correlated"):
    
    opt = Optim_Problem()
    api = MatricesAPI()

    # 1) Define system
    plant, ctrl0 = api.get_system(seed=seed_plant, FROM_DATA=True)
    nx, nw, nu, nz, ny = plant.dims()
    print(f"Plant dims nx={nx}, nw={nw}, nu={nu}, nz={nz}, ny={ny}")

    # 2) Define ambiguity set (W2-ball around N(0, Σ_nom) with radius γ)
    Sigma_nom = api.make_nominal_covariances(nw)
    amb = Ambiguity(Sigma_nom, alpha=None)
    Sigma_eff = amb.sigma_effective()
    print("Effective Σ_w:\n", Sigma_eff)

    # 3) Baseline cost with initial controller
    base_cost = opt.simulate_cost(plant, ctrl0, Sigma_eff, T=T_cost_init, burnin=burnin_init, seed=0)
    print(f"Baseline long-run cost E||z||^2 ≈ {base_cost:.4f}")

    # 4) Optimize
    ctrl_opt, res = opt.optimize_controller(
        plant,
        ctrl0,
        Sigma_eff,
        T=T_cost_opt,
        burnin=burnin_opt,
        seeds=seeds_opt,
        maxiter=maxiter,
    )
    print("Optimizer status:", res.message)

    # 5) Evaluate optimized controller
    cost_opt = opt.simulate_cost(plant, ctrl_opt, Sigma_eff, T=4000, burnin=400, seed=11)
    print(f"Optimized long-run cost E||z||^2 ≈ {cost_opt:.4f}")

    # 6) Report closed-loop stability
    A_cl, *_ = compose_closed_loop(plant, ctrl_opt)
    rho = max(abs(np.linalg.eigvals(A_cl)))
    print(f"Spectral radius of 𝒜: {rho:.6f}")

    # 7) Dump controller
    np.set_printoptions(suppress=True, linewidth=140, precision=4)
    print("\nOptimized controller matrices:")
    print("Ac=\n", ctrl_opt.Ac)
    print("Bc=\n", ctrl_opt.Bc)
    print("Cc=\n", ctrl_opt.Cc)
    print("Dc=\n", ctrl_opt.Dc)

    return Sigma_eff, float(base_cost), str(res.message), float(cost_opt), rho, ctrl_opt, plant


# ------------------------- MAIN SCRIPT ENTRY POINT ----------------------

if __name__ == "__main__":
    Sigma_eff, base_cost, msg, cost_opt, rho, ctrl_opt, plant = run_once()
    np.set_printoptions(suppress=True, linewidth=140, precision=4)
