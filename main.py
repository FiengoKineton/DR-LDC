# main.py
import json, argparse, yaml
import numpy as np
from pathlib import Path

from problem___baseline import run_once
from problem___dro_lmi import build_and_solve_dro_lmi

from utilis___systems import Plant, Controller
from utilis___simulate import Closed_Loop 
from utilis___matrices import Recover, MatricesAPI


# ------------------------- BASELINE OPTIMIZATION PROBLEM --------------------------

class baseline_optim_problem(): 
    def __init__(self, params: dict):
        ARTIFACTS = Path("out/artifacts/baseline")
        ARTIFACTS.mkdir(exist_ok=True)

        # Run optimization AND capture the exact plant used
        cl = Closed_Loop()  # instantiate simulation class
        FROM_DATA = params.get("FROM_DATA", False)
        if FROM_DATA:
            print("Evaluating plant from data files.")

        Sigma_eff, base_cost, msg, cost_opt, rho, ctrl_opt, plant = run_once(FROM_DATA=FROM_DATA)

        # Persist everything needed for reproducible simulation
        json_path = ARTIFACTS / "results_run.json"
        self.save_results_json(json_path, Sigma_eff, base_cost, msg, cost_opt, rho, ctrl_opt, plant)

        # Also stash arrays for quick re-loads
        npz_path = ARTIFACTS / "results_run_arrays.npz"
        np.savez_compressed(
            npz_path,
            Sigma_eff=Sigma_eff,
            Ac=ctrl_opt.Ac, Bc=ctrl_opt.Bc, Cc=ctrl_opt.Cc, Dc=ctrl_opt.Dc,
            A=plant.A, Bw=plant.Bw, Bu=plant.Bu, Cz=plant.Cz, Dzw=plant.Dzw, Dzu=plant.Dzu, Cy=plant.Cy, Dyw=plant.Dyw,
            baseline_cost=np.array(base_cost), optimized_cost=np.array(cost_opt), spectral_radius_Acl=np.array(rho),
        )
        print(f"[saved] {json_path}")
        print(f"[saved] {npz_path}")

        # Load back the exact same objects and simulate
        Sigma_loaded, ctrl_loaded, plant_loaded, meta = self.load_results_json(json_path)
        sim = cl.simulate_closed_loop(plant_loaded, ctrl_loaded, Sigma_loaded, T=800, seed=11)
        out_npz = ARTIFACTS / "closed_loop_run_seed11_T800.npz"
        cl.save_npz(sim, str(out_npz))
        print(f"[saved] {out_npz}")

        cl.plot_timeseries(sim)

    def plant_to_dict(self, P: Plant):
        return {
            "A": P.A.tolist(), "Bw": P.Bw.tolist(), "Bu": P.Bu.tolist(),
            "Cz": P.Cz.tolist(), "Dzw": P.Dzw.tolist(), "Dzu": P.Dzu.tolist(),
            "Cy": P.Cy.tolist(), "Dyw": P.Dyw.tolist(),
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

    def save_results_json(self, path, Sigma_eff, base_cost, msg, cost_opt, rho, ctrl_opt, plant):
        payload = {
            "Sigma_eff": Sigma_eff.tolist(),
            "baseline_cost": base_cost,
            "optimizer_status": msg,
            "optimized_cost": cost_opt,
            "spectral_radius_Acl": rho,
            "controller": self.controller_to_dict(ctrl_opt),
            "plant": self.plant_to_dict(plant),
        }
        with open(path, "w", encoding="utf-8") as f:
            json.dump(payload, f, indent=2)

    def load_results_json(self, path):
        with open(path, "r", encoding="utf-8") as f:
            d = json.load(f)
        Sigma_eff = np.array(d["Sigma_eff"], dtype=float)
        ctrl = self.controller_from_dict(d["controller"])
        plant = self.plant_from_dict(d["plant"])
        meta = {
            "baseline_cost": d["baseline_cost"],
            "optimized_cost": d["optimized_cost"],
            "optimizer_status": d["optimizer_status"],
            "spectral_radius_Acl": d["spectral_radius_Acl"],
        }
        return Sigma_eff, ctrl, plant, meta


# ------------------------- DRO-LMI PIPELINE OPTIMIZATION PROBLEM ------------------

class lmi_pipeline_optim_problem(): 
    def __init__(self, params: dict):
        ART = Path("out/artifacts/lmi")
        ART.mkdir(exist_ok=True)

        recover = Recover()
        api = MatricesAPI()
        FROM_DATA = params.get("FROM_DATA", False)
        if FROM_DATA:
            print("Evaluating plant from data files.")

        # 1) Define plant and nominal disturbance covariance (keep consistent with your LMI)
        plant, _ = api.get_system(seed=7, FROM_DATA=FROM_DATA)
        Sigma_nom = api.make_nominal_covariances(plant.Bw.shape[1])

        solver = params.get("solver", "SCS")                     # "MOSEK" or "SCS"
        gamma = params.get("ambiguity", {}).get("gamma", 0.5)    # Wasserstein radius (set as you wish)

        cl = Closed_Loop()  # instantiate simulation class

        # 2) Solve DRO-LMI (choose "correlated" or "independent")
        model = params.get("model", "independent")                          # \in {"correlated", "independent"}
        res = build_and_solve_dro_lmi(
            plant=plant,
            Sigma_nom=Sigma_nom,
            gamma=gamma,
            model=model,
            solver=solver,       # MOSEK if available, else SCS (set to "MOSEK" explicitly if you have it)
            verbose=False
        )

        if res.status not in ("optimal", "optimal_inaccurate"):
            raise RuntimeError(f"DRO-LMI solve failed: status={res.status}")

        # Debug prints (one time)
        print("Abar", np.shape(res.Abar), "Bbar", np.shape(res.Bbar), 
            "Cbar", np.shape(res.Cbar), "Dbar", np.shape(res.Dbar), "Pbar", np.shape(res.Pbar))

        # 3) From (Pbar, Abar, Bbar, Cbar, Dbar) build composite (Acl, Bcl, Ccl, Dcl) in original coords
        Acl, Bcl, Ccl, Dcl = recover.closed_loop_from_bar(res.Pbar, res.Abar, res.Bbar, res.Cbar, res.Dbar)

        # 4) Recover (Ac, Bc, Cc, Dc) from composite and plant, with residual diagnostics
        ctrl_rec, residuals = recover.recover_controller_from_closed_loop(plant, Acl, Bcl, Ccl, Dcl)
        rho = float(np.max(np.abs(np.linalg.eigvals(Acl))))
        print(f"spectral radius(Acl) ≈ {rho:.6g}")
        if not np.isfinite(rho) or rho >= 1.05:
            raise RuntimeError("Closed loop is unstable/ill-conditioned (rho>=1.05). "
                            "Tighten regularization or reduce gamma before simulating.")

        # 5) Persist everything meaningful into a single JSON
        payload = {
            "meta": {
                "model": model,
                "status": res.status,
                "objective": res.obj_value,
                "gamma": res.gamma,
                "lambda_opt": res.lambda_opt,
                "spectral_radius_Acl": rho,
            },
            "disturbance": {
                "Sigma_nom": Sigma_nom.tolist(),
            },
            "recovered_controller": self.controller_to_dict(ctrl_rec),
            "plant": self.plant_to_dict(plant),
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
            },
            "composite_closed_loop": {
                "Acl": Acl.tolist(),
                "Bcl": Bcl.tolist(),
                "Ccl": Ccl.tolist(),
                "Dcl": Dcl.tolist(),
            },
            "recovery_residuals_rel": residuals,  # dimensionless relative errors
        }

        out_json = ART / f"dro_pipeline_{model}.json"
        self.save_json(out_json, payload)
        print(f"[saved] {out_json}")

        # 6) Simulate with the recovered controller using the SAME plant and nominal Σ
        #    If you prefer covariance inflation for robustness testing, replace Sigma_nom here.
        sim = cl.simulate_closed_loop(plant, ctrl_rec, Sigma_nom, T=800, seed=11)
        out_npz = ART / f"dro_pipeline_{model}_sim_T800_seed11.npz"
        cl.save_npz(sim, str(out_npz))
        print(f"[saved] {out_npz}")

        # 7) Plot results
        cl.plot_timeseries(sim)


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

    def controller_to_dict(self, C: Controller):
        return {"Ac": C.Ac.tolist(), "Bc": C.Bc.tolist(), "Cc": C.Cc.tolist(), "Dc": C.Dc.tolist()}

    def save_json(self, path: Path, payload: dict):
        with open(path, "w", encoding="utf-8") as f:
            json.dump(payload, f, indent=2)


# ------------------------- MAIN SCRIPT ENTRY POINT -------------------------------

if __name__ == "__main__":
    parser = argparse.ArgumentParser(description="DRO LMI Optimization")
    parser.add_argument("--base", action="store_true", help="Run baseline optimization")
    parser.add_argument("--lmi", action="store_true", help="Run LMI pipeline optimization")
    args = parser.parse_args()

    if yaml is None:
        raise ImportError("PyYAML not available. Install with `pip install pyyaml`.")
    with open("problem___parameters.yaml", "r", encoding="utf-8") as f:
        cfg = yaml.safe_load(f)

    p = cfg.get("params", {})


    if args.base:
        print("Running baseline optimization...")
        baseline_optim_problem(params=p)
    if args.lmi:
        print("Running LMI pipeline optimization...")
        lmi_pipeline_optim_problem(params=p)
