import json, numpy as np
from utils___matrices import compose_closed_loop

from numpy.linalg import eigvals
from scipy.optimize import minimize
from utils___systems import Plant, Controller
from pathlib import Path

from utils___systems import Plant, Controller, Plant_cl, Noise
from utils___simulate import Closed_Loop 
from utils___matrices import Recover, MatricesAPI, compose_closed_loop


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
        nx, nw, *_ = plant.dims()
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
        If Abar is unstable, shrink controller dynamics slightly.
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
        A_cl, *_ = compose_closed_loop(plant, ctrl)
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

def run_once(plant: Plant = None,
             ctrl0: Controller = None,
             Sigma_nom: np.ndarray = np.array([[1.0, 0.0], [0.0, 1.0]]),
             T_cost_init: int = 2000,
             T_cost_opt: int = 2500,
             burnin_init: int = 200,
             burnin_opt: int = 300,
             seeds_opt = (0,1,2,3,4),
             maxiter: int = 120):
    
    opt = Optim_Problem()

    # 1) Define system
    nx, nw, nu, nz, ny = plant.dims()
    print(f"Plant dims nx={nx}, nw={nw}, nu={nu}, nz={nz}, ny={ny}")

    # 3) Baseline cost with initial controller
    if Sigma_nom[0].size != nw: 
        Sigma_nom = np.eye(nw)
    base_cost = opt.simulate_cost(plant, ctrl0, Sigma_nom, T=T_cost_init, burnin=burnin_init, seed=0)
    print(f"Baseline long-run cost E||z||^2 ≈ {base_cost:.4f}")

    # 4) Optimize
    ctrl_opt, res = opt.optimize_controller(
        plant,
        ctrl0,
        Sigma_nom,
        T=T_cost_opt,
        burnin=burnin_opt,
        seeds=seeds_opt,
        maxiter=maxiter,
    )
    print("Optimizer status:", res.message)

    # 5) Evaluate optimized controller
    cost_opt = opt.simulate_cost(plant, ctrl_opt, Sigma_nom, T=4000, burnin=400, seed=11)
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

    return Sigma_nom, float(base_cost), str(res.message), float(cost_opt), rho, ctrl_opt


# ------------------------- BASELINE OPTIMIZATION PROBLEM --------------------------

class baseline_optim_problem(): 
    def __init__(self, out: Path, Sigma_nom: np.ndarray, gamma: float, plot: bool = False, save: bool = False, FROM_DATA: bool = None, init_cond: str = "zero"):

        # Run optimization AND capture the exact plant used
        cl = Closed_Loop()  # instantiate simulation class
        api = MatricesAPI()

        plant, ctrl0 = api.get_system(FROM_DATA=FROM_DATA, gamma=gamma)
        api.print_plant(plant)

        Sigma_nom, base_cost, msg, cost_opt, rho, ctrl_opt = run_once(plant=plant, ctrl0=ctrl0, Sigma_nom=Sigma_nom)
        Acl, Bcl, Ccl, Dcl = compose_closed_loop(plant, ctrl_opt)
        plant_cl = Plant_cl(Acl=Acl, Bcl=Bcl, Ccl=Ccl, Dcl=Dcl)

        # Persist everything needed for reproducible simulation
        json_path = out + f"___results_run.json"
        if save: self.save_results_json(json_path, Sigma_nom, base_cost, msg, cost_opt, rho, ctrl_opt, plant, plant_cl)
        print(f"[saved] {json_path}")

        # Load back the exact same objects and simulate
        Sigma_loaded, ctrl_loaded, plant_loaded, _ = self.load_results_json(json_path)
        sim = cl.simulate_closed_loop(plant=plant_loaded, ctrl=ctrl_loaded, Sigma_w=Sigma_loaded, gamma=gamma, init_cond=init_cond)
        out_npz = out + f"___closed_loop_run.npz"
        if save: cl.save_npz(sim, str(out_npz))
        print(f"[saved] {out_npz}")

        sim_composite = cl.simulate_composite(Pcl=plant_cl, gamma=gamma, init_cond=init_cond)
        out_composite = out + f"___closed_loop_composite.npz"
        if save: cl.save_npz(sim_composite, str(out_composite))
        print(f"[saved] {out_composite}")

        sim_cost = cl.simulate_Z_cost(Z=sim_composite["Z"], plot=plot)
        out_cost = out + f"___closed_loop_run_cost.npz"
        if save: cl.save_npz(sim_cost, str(out_cost))   
        print(f"[saved] {out_cost}")


        if plot: 
            cl.plot_timeseries(sim=sim, save=save, out=out)
            cl.plot_composite(sim=sim_composite, save=save, out=out)
    
        self.plant, self.ctrl, self.sim, self.Sigma_nom = plant, ctrl_opt, sim, Sigma_nom


    def plant_to_dict(self, P: Plant):
        return {
            "A": P.A.tolist(), "Bw": P.Bw.tolist(), "Bu": P.Bu.tolist(),
            "Cz": P.Cz.tolist(), "Dzw": P.Dzw.tolist(), "Dzu": P.Dzu.tolist(),
            "Cy": P.Cy.tolist(), "Dyw": P.Dyw.tolist(),
        }

    def plant_cl_to_dict(self, P: Plant_cl):
        return {
            "Acl": P.Acl.tolist(),
            "Bcl": P.Bcl.tolist(), 
            "Ccl": P.Ccl.tolist(), 
            "Dcl": P.Dcl.tolist()
        }
    
    def plant_from_dict(self, d: dict) -> Plant:
        return Plant(
            A=np.array(d["A"], dtype=float),
            Bw=np.array(d["Bw"], dtype=float),
            Bu=np.array(d["Bu"], dtype=float),
            Cz=np.array(d["Cz"], dtype=float),
            Dzw=np.array(d["Dzw"], dtype=float),
            Dzu=np.array(d["Dzu"], dtype=float),
            Cy=np.array(d["Cy"], dtype=float),
            Dyw=np.array(d["Dyw"], dtype=float),
        )

    def controller_to_dict(self, C: Controller):
        return {"Ac": C.Ac.tolist(), "Bc": C.Bc.tolist(), "Cc": C.Cc.tolist(), "Dc": C.Dc.tolist()}

    def controller_from_dict(self, d: dict) -> Controller:
        return Controller(
            Ac=np.array(d["Ac"], dtype=float),
            Bc=np.array(d["Bc"], dtype=float),
            Cc=np.array(d["Cc"], dtype=float),
            Dc=np.array(d["Dc"], dtype=float),
        )

    def save_results_json(self, path, Sigma_nom, base_cost, msg, cost_opt, rho, ctrl_opt, plant, plant_cl):
        payload = {
            "Sigma_nom": Sigma_nom.tolist(),
            "baseline_cost": base_cost,
            "optimizer_status": msg,
            "optimized_cost": cost_opt,
            "spectral_radius_Acl": rho,
            "controller": self.controller_to_dict(ctrl_opt),
            "plant": self.plant_to_dict(plant),
            "composite_closed_loop": self.plant_cl_to_dict(plant_cl),
        }
        with open(path, "w", encoding="utf-8") as f:
            json.dump(payload, f, indent=2)

    def load_results_json(self, path):
        with open(path, "r", encoding="utf-8") as f:
            d = json.load(f)
        Sigma_nom = np.array(d["Sigma_nom"], dtype=float)
        ctrl = self.controller_from_dict(d["controller"])
        plant = self.plant_from_dict(d["plant"])
        meta = {
            "baseline_cost": d["baseline_cost"],
            "optimized_cost": d["optimized_cost"],
            "optimizer_status": d["optimizer_status"],
            "spectral_radius_Acl": d["spectral_radius_Acl"],
        }
        return Sigma_nom, ctrl, plant, meta

    def get_snr_vars(self):
        return self.plant, self.ctrl, self.sim, self.Sigma_nom


# ------------------------- MAIN SCRIPT ENTRY POINT ----------------------

if __name__ == "__main__":
    Sigma_nom, base_cost, msg, cost_opt, rho, ctrl_opt, plant = run_once()
    np.set_printoptions(suppress=True, linewidth=140, precision=4)
