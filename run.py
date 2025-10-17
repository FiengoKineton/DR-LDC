# run.py
import numpy as np
from define_matrices import get_system, make_nominal_covariances
from ambiguity import Ambiguity
from optim_problem import optimize_controller, simulate_cost
from compose import compose_closed_loop

def run_once(seed_plant: int = 7,
             gamma: float = 0.2,
             T_cost_init: int = 2000,
             T_cost_opt: int = 2500,
             burnin_init: int = 200,
             burnin_opt: int = 300,
             seeds_opt = (0,1,2,3,4),
             maxiter: int = 120, 
             model="correlated"):
    
    # 1) Define system
    plant, ctrl0 = get_system(seed=seed_plant, FROM_DATA=True)
    nx, nw, nu, nz, ny = plant.dims()
    print(f"Plant dims nx={nx}, nw={nw}, nu={nu}, nz={nz}, ny={ny}")

    # 2) Define ambiguity set (W2-ball around N(0, Σ_nom) with radius γ)
    Sigma_nom = make_nominal_covariances(nw)
    amb = Ambiguity(Sigma_nom, gamma, model=model, alpha=None)
    Sigma_eff = amb.sigma_effective()
    print("Effective Σ_w:\n", Sigma_eff)

    # 3) Baseline cost with initial controller
    base_cost = simulate_cost(plant, ctrl0, Sigma_eff, T=T_cost_init, burnin=burnin_init, seed=0)
    print(f"Baseline long-run cost E||z||^2 ≈ {base_cost:.4f}")

    # 4) Optimize
    ctrl_opt, res = optimize_controller(
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
    cost_opt = simulate_cost(plant, ctrl_opt, Sigma_eff, T=4000, burnin=400, seed=11)
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


if __name__ == "__main__":
    Sigma_eff, base_cost, msg, cost_opt, rho, ctrl_opt, plant = run_once()
    np.set_printoptions(suppress=True, linewidth=140, precision=4)
