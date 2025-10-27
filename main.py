# main.py
import json, argparse, yaml, sys
import numpy as np
from pathlib import Path

from problem___baseline import run_once
from problem___dro_lmi import build_and_solve_dro_lmi

from utils___systems import Plant, Controller, Plant_cl
from utils___simulate import Closed_Loop 
from utils___matrices import Recover, MatricesAPI, compose_closed_loop
from utils___SolutionComparison import ResultsComparator


# ------------------------- BASELINE OPTIMIZATION PROBLEM --------------------------

class baseline_optim_problem(): 
    def __init__(self, out: Path, Sigma_nom: np.ndarray, gamma: float, plot: bool = False, FROM_DATA: bool = None):

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
        self.save_results_json(json_path, Sigma_nom, base_cost, msg, cost_opt, rho, ctrl_opt, plant, plant_cl)
        print(f"[saved] {json_path}")

        # Load back the exact same objects and simulate
        Sigma_loaded, ctrl_loaded, plant_loaded, _ = self.load_results_json(json_path)
        sim = cl.simulate_closed_loop(plant_loaded, ctrl_loaded, Sigma_loaded, gamma)
        out_npz = out + f"___closed_loop_run.npz"
        cl.save_npz(sim, str(out_npz))
        print(f"[saved] {out_npz}")

        sim_composite = cl.simulate_composite(plant_cl, gamma)
        out_composite = out + f"___closed_loop_composite.npz"
        cl.save_npz(sim_composite, str(out_composite))
        print(f"[saved] {out_composite}")

        if plot: 
            cl.plot_timeseries(sim)
            cl.plot_composite(sim_composite)

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


# ------------------------- DRO-LMI PIPELINE OPTIMIZATION PROBLEM ------------------

class lmi_pipeline_optim_problem(): 
    def __init__(self, params: dict, out: Path, gamma: float, Sigma_nom: np.ndarray, plot: bool = False, FROM_DATA: bool = False):

        recover = Recover()
        api = MatricesAPI()
        cl = Closed_Loop() 

        # 1) Define plant and nominal disturbance covariance (keep consistent with your LMI)
        plant, _ = api.get_system(FROM_DATA=FROM_DATA, gamma=gamma)
        api.print_plant(plant)

        # 2) Solve DRO-LMI (choose "correlated" or "independent")
        model = params.get("model", "correlated")                # \in {"correlated", "independent"}
        res = build_and_solve_dro_lmi(
            plant=plant,
            api=api,
            Sigma_nom=Sigma_nom,
            gamma=gamma,
            model=model,
        )

        if res.status not in ("optimal", "optimal_inaccurate"):
            raise RuntimeError(f"DRO-LMI solve failed: status={res.status}")

        # Debug prints (one time)
        print("Abar", np.shape(res.Abar), "Bbar", np.shape(res.Bbar), 
            "Cbar", np.shape(res.Cbar), "Dbar", np.shape(res.Dbar), "Pbar", np.shape(res.Pbar))

        # 3) From (Pbar, Abar, Bbar, Cbar, Dbar) build composite (Acl, Bcl, Ccl, Dcl) in original coords
        Ac, Bc, Cc, Dc = recover.Mc_from_bar(res, plant)
        A, Bw, Bu, Cz, Dzw, Dzu, Cy, Dyw = plant.A, plant.Bw, plant.Bu, plant.Cz, plant.Dzw, plant.Dzu, plant.Cy, plant.Dyw
        Bw, Dzw, Dyw, _, Sigma_nom = api._augment_matrices(Bw, Dzw, Dyw)

        plant = Plant(A=A, Bw=Bw, Bu=Bu, Cz=Cz, Dzw=Dzw, Dzu=Dzu, Cy=Cy, Dyw=Dyw)
        ctrl = Controller(Ac=Ac, Bc=Bc, Cc=Cc, Dc=Dc)
        Acl, Bcl, Ccl, Dcl = compose_closed_loop(plant, ctrl)
        plant_cl = Plant_cl(Acl=Acl, Bcl=Bcl, Ccl=Ccl, Dcl=Dcl)

        # 4) Recover (Ac, Bc, Cc, Dc) from composite and plant, with residual diagnostics
        # ctrl_rec, residuals = recover.recover_controller_from_closed_loop(plant, api, (Acl, Bcl, Ccl, Dcl))
        
        
        eig = np.linalg.eigvals(Acl)
        rho = float(np.max(np.abs(eig)))

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

        # 5) Persist everything meaningful into a single JSON
        payload = {
            "meta": {
                "model": model,
                "solver": res.solver,
                "status": res.status,
                "objective": res.obj_value,
                "gamma": res.gamma,
                "lambda_opt": res.lambda_opt,
                "spectral_radius_Acl": rho,
            },
            "disturbance": {
                "Sigma_nom": Sigma_nom.tolist(),
            },
            "controller": self.controller_to_dict(ctrl),
            "plant": self.plant_to_dict(plant),
            "composite_closed_loop": self.plant_cl_to_dict(plant_cl),
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
                "P": None if res.P is None else res.P.tolist(),
            },
        }

        out_json = out + f"___results_run.json"
        self.save_json(out_json, payload)
        print(f"[saved] {out_json}")

        # 6) Simulate with the recovered controller using the SAME plant and nominal Σ
        #    If you prefer covariance inflation for robustness testing, replace Sigma_nom here.
        sim = cl.simulate_closed_loop(plant, ctrl, Sigma_nom, gamma)
        out_npz = out + f"___closed_loop_run.npz"
        cl.save_npz(sim, str(out_npz))
        print(f"[saved] {out_npz}")

        sim_composite = cl.simulate_composite(plant_cl, gamma)
        out_composite = out + f"___closed_loop_composite.npz"
        cl.save_npz(sim_composite, str(out_composite))
        print(f"[saved] {out_composite}")

        # 7) Plot results
        if plot: 
            cl.plot_timeseries(sim)
            cl.plot_composite(sim_composite)


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


# ------------------------- MAIN SCRIPT ENTRY POINT -------------------------------

def main(gamma: float = None, FROM_DATA: bool = None, comp: bool = None):
    #parser = argparse.ArgumentParser(description="DRO LMI Optimization")
    #parser.add_argument("--comp", action="store_true", help="Run comparison btw baseline and LMI pipeline")
    #parser.add_argument("--base", action="store_true", help="Run baseline optimization")
    #parser.add_argument("--p", action="store_true", help="Force Plot")
    #parser.add_argument("--lmi", action="store_true", help="Run LMI pipeline optimization")
    #args = parser.parse_args()

    if yaml is None:
        raise ImportError("PyYAML not available. Install with `pip install pyyaml`.")
    with open("problem___parameters.yaml", "r", encoding="utf-8") as f:
        cfg = yaml.safe_load(f)

    p = cfg.get("params", {})
    out = p.get("directories", {}).get("artifacts", "./out/artifacts/")
    runID = p.get("directories", {}).get("runID", "temp")
    _type = p.get("plant", {}).get("type", "explicit")
    _model = p.get("model", "independent")
    _method = p.get("method", "lmi")
    FROM_DATA = bool(p.get("FROM_DATA", False)) if FROM_DATA is None else FROM_DATA
    plot = bool(p.get("plot", False)) if runID != "GammaOpt" else False
    _data = "DDD" if FROM_DATA else "MBD"
    gamma = p.get("ambiguity", {}).get("gamma", 0.5) if gamma is None else gamma
    #comp = args.comp if comp is None else comp

    Sigma_nom = np.array(p.get("ambiguity", {})["Sigma_nom"], dtype=float)

    path_name = f"/run_{runID}___{_type}_{_model}_{_data}"

    if comp:
        cmp = ResultsComparator(out_root=out)
        return cmp.compare_mbd_vs_ddd(path_name=path_name, method=_method, plot=plot)
        # cmp.compare_baseline_vs_lmi(path_name=path_name, plot=True)
    else:
        out = Path(out).with_suffix("").as_posix() + f"/{_method}" + path_name

        if _method == "base":
            print("\nRunning baseline optimization...")
            baseline_optim_problem(out=out, Sigma_nom=Sigma_nom, gamma=gamma, plot=plot, FROM_DATA=FROM_DATA)
        else:
            print("\nRunning LMI pipeline optimization...")
            lmi_pipeline_optim_problem(params=p, out=out, Sigma_nom=Sigma_nom, gamma=gamma, plot=plot, FROM_DATA=FROM_DATA)



if __name__ == "__main__":
    if yaml is None:
        raise ImportError("PyYAML not available. Install with `pip install pyyaml`.")
    with open("problem___parameters.yaml", "r", encoding="utf-8") as f:
        cfg = yaml.safe_load(f)

    p = cfg.get("params", {})

    if p.get("method", "lmi") == "lmi":
        if bool(p.get("use_set_out_mats", False)):
            gamma = 0.17960675006309104
    else:
        gamma = p.get("ambiguity", {}).get("gamma", 0.5)


    if bool(p.get("ALL", False)):
        main(FROM_DATA=False, gamma=gamma)
        main(FROM_DATA=True, gamma=gamma)
        main(comp=True, gamma=gamma)
    else:
        main(gamma=gamma)
