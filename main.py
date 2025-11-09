# main.py
import json, argparse, yaml, sys
import numpy as np
from pathlib import Path
import matplotlib.pyplot as plt

from problem___baseline import run_once
from problem___dro_lmi import build_and_solve_dro_lmi, build_and_solve_dro_lmi_upd

from utils___systems import Plant, Controller, Plant_cl, Noise
from utils___simulate import Closed_Loop 
from utils___matrices import Recover, MatricesAPI, compose_closed_loop
from utils___SolutionComparison import ResultsComparator
from utils___SNR import SNRAnalyzer


# ------------------------- BASELINE OPTIMIZATION PROBLEM --------------------------

class baseline_optim_problem(): 
    def __init__(self, out: Path, Sigma_nom: np.ndarray, gamma: float, plot: bool = False, save: bool = False, FROM_DATA: bool = None):

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
        sim = cl.simulate_closed_loop(plant_loaded, ctrl_loaded, Sigma_loaded, gamma)
        out_npz = out + f"___closed_loop_run.npz"
        if save: cl.save_npz(sim, str(out_npz))
        print(f"[saved] {out_npz}")

        sim_composite = cl.simulate_composite(plant_cl, gamma)
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


# ------------------------- DRO-LMI PIPELINE OPTIMIZATION PROBLEM ------------------

class lmi_pipeline_optim_problem(): 
    def __init__(self, params: dict, out: Path, noise: Noise, upd: bool = False, plot: bool = False, save: bool = False, FROM_DATA: bool = False):

        recover = Recover()
        api = MatricesAPI()
        cl = Closed_Loop() 

        model = params.get("model", "correlated") if params.get("ambiguity", {}).get("model", "W2") != "Gaussian" else "independent"

        # 1) Define plant and nominal disturbance covariance (keep consistent with your LMI)
        if not upd or not FROM_DATA:
            plant, _ = api.get_system(FROM_DATA=FROM_DATA, gamma=gamma, upd=upd)
            api.print_plant(plant)

            # 2) Solve DRO-LMI (choose "correlated" or "independent")
            res = build_and_solve_dro_lmi(
                plant=plant,
                api=api,
                noise=noise,
                model=model,
                SOLVER=params.get("solver", "MOSEK"),
            )

            A, Bw, Bu, Cz, Dzw, Dzu, Cy, Dyw = plant.A, plant.Bw, plant.Bu, plant.Cz, plant.Dzw, plant.Dzu, plant.Cy, plant.Dyw
            Bw, Dzw, Dyw, _, Sigma_nom = api._augment_matrices(Bw, Dzw, Dyw, noise.var)
            plant = Plant(A=A, Bw=Bw, Bu=Bu, Cz=Cz, Dzw=Dzw, Dzu=Dzu, Cy=Cy, Dyw=Dyw)
            ADD = False

        else:
            approach = params.get("approach", "Young")

            # 2) Solve DRO-LMI (choose "correlated" or "independent")
            res, P, Sigma_nom, other = build_and_solve_dro_lmi_upd(
                api=api,
                vals=(upd, FROM_DATA, plot),
                noise=noise,
                model=model,
                approach=approach,
                d=(params.get("ambiguity", {}).get("model", "W2") == "Gaussian")
            )

            A, Bw, Bu, Cy, Dyw, Cz, Dzw, Dzu = P
            plant = Plant(A=A, Bw=Bw, Bu=Bu, Cz=Cz, Dzw=Dzw, Dzu=Dzu, Cy=Cy, Dyw=Dyw)
            ADD = True
        
        if res.status not in ("optimal", "optimal_inaccurate"):
            raise RuntimeError(f"DRO-LMI solve failed: status={res.status}")


        # 3) From (Pbar, Abar, Bbar, Cbar, Dbar) build composite (Acl, Bcl, Ccl, Dcl) in original coords

        Ac, Bc, Cc, Dc = recover.Mc_from_bar(res, plant)
        ctrl = Controller(Ac=Ac, Bc=Bc, Cc=Cc, Dc=Dc)
        Acl, Bcl, Ccl, Dcl = compose_closed_loop(plant, ctrl)
        plant_cl = Plant_cl(Acl=Acl, Bcl=Bcl, Ccl=Ccl, Dcl=Dcl)

        # 4) Recover (Ac, Bc, Cc, Dc) from composite and plant, with residual diagnostics
        # ctrl_rec, residuals = recover.recover_controller_from_closed_loop(plant, api, (Acl, Bcl, Ccl, Dcl))
        
        
        eig = np.linalg.eigvals(Acl)
        rho = float(np.max(np.abs(eig)))

        eig_cart = [
            {"re": float(ev.real), "im": float(ev.imag), "abs": float(np.abs(ev))}
            for ev in eig
        ]
        eig_polar = [
            {"abs": float(np.abs(ev)), "ang_rad": float(np.angle(ev))}
            for ev in eig
        ]

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
        

        # JSON
        payload = {
            "meta": {
                "model": model,
                "solver": res.solver,
                "status": res.status,
                "objective": res.obj_value,
                "gamma": res.gamma,
                "lambda_opt": res.lambda_opt,
                "spectral_radius_Acl": rho,
                #"rx": None if res.rx is None else res.rx,
                #"ry": None if res.ry is None else res.ry,
                #"rz": None if res.rz is None else res.rz,
            },
            "controller": self.controller_to_dict(ctrl),
            "plant": self.plant_to_dict(plant),
            "composite_closed_loop": self.plant_cl_to_dict(plant_cl),
            "Acl_eigenvals": {
                "cartesian": eig_cart,
                "polar": eig_polar
            },            
            "disturbance": {
                "Sigma_nom": Sigma_nom.tolist(),
            },
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
            },
        }

        if ADD: 
            if approach == 'Young' or approach == 'Mats':
                D, E, B, S, T = other
                payload["Young_approach"] = {
                    "approach": approach,
                    "D": self._to_serializable(D),  # handles (DeltaA, DeltaB) as matrices/tuples
                    "E": self._to_serializable(E),  # handles (EAA, EAB) as matrices/tuples
                    "B": self._to_serializable(B),  # now OK if matrix, vector, or scalars tuple
                    "S": self._to_serializable(S),  # now OK if matrix
                    "T": self._to_serializable(T),  # now OK if matrix
                }

        out_json = out + f"___results_run.json"
        if save: self.save_json(out_json, payload)
        print(f"[saved] {out_json}")

        # 6) Simulate with the recovered controller using the SAME plant and nominal Σ
        #    If you prefer covariance inflation for robustness testing, replace Sigma_nom here.
        sim = cl.simulate_closed_loop(plant, ctrl, Sigma_nom, gamma)
        out_npz = out + f"___closed_loop_run.npz"
        if save: cl.save_npz(sim, str(out_npz))
        print(f"[saved] {out_npz}")

        sim_composite = cl.simulate_composite(plant_cl, Sigma_nom, gamma)
        out_composite = out + f"___closed_loop_composite.npz"
        if save: cl.save_npz(sim_composite, str(out_composite))
        print(f"[saved] {out_composite}")

        sim_cost = cl.simulate_Z_cost(Z=sim_composite["Z"], plot=plot)
        self.final_cost = sim_cost["J"]
        print("\nFinal closed-loop cost J =", self.final_cost)
        out_cost = out + f"___closed_loop_run_cost.npz"
        if save: cl.save_npz(sim_cost, str(out_cost))   
        print(f"[saved] {out_cost}")

        # 7) Plot results
        if plot: 
            cl.plot_timeseries(sim=sim, save=save, out=out)
            cl.plot_composite(sim=sim_composite, save=save, out=out)

        Data = {
            'gamma': res.gamma, 'lambda': res.lambda_opt, 'Sigma_nom': Sigma_nom,
            'A_c': Ac, 'B_c':Bc, 'C_c': Cc, 'D_c': Dc, 'A_cl': Acl, 'rho': rho, 'var' : noise.var,
            #'rx': None if res.rx is None else res.rx, 'ry': None if res.ry is None else res.ry, 'rz': None if res.rz is None else res.rz
        }
        for key in ['gamma', 'lambda', 'Sigma_nom', 'A_c', 'B_c', 'C_c', 'D_c', 'A_cl']:#, 'rho', 'var', 'rx', 'ry', 'rz']:
            print(f"\n{key} =")
            print(Data[key])
        
        self.plant, self.ctrl, self.sim, self.Sigma_nom = plant, ctrl, sim, Sigma_nom

    def _return_final_cost(self):
        return self.final_cost

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

    def get_snr_vars(self):
        return self.plant, self.ctrl, self.sim, self.Sigma_nom

    def _to_serializable(self, x):
        """
        Convert scalars / arrays / nested tuples-lists of arrays into JSON-safe types.
        - CVXPy expressions -> use .value first (caller’s job), otherwise we try best-effort.
        - NumPy arrays -> .tolist()
        - Scalars -> float/int/bool
        - Lists/Tuples -> map recursively
        """
        import numpy as np

        # cvxpy objects: try to unwrap gracefully if user forgot to pass .value
        try:
            import cvxpy as cp
            if isinstance(x, (cp.Expression, cp.atoms.atom.Atom)):
                x = x.value
        except Exception:
            pass

        if x is None:
            return None
        if isinstance(x, (list, tuple)):
            return [self._to_serializable(xx) for xx in x]
        if hasattr(x, "toarray"):             # e.g., scipy.sparse
            x = x.toarray()
        if isinstance(x, np.ndarray):
            return x.tolist()
        # numpy scalar
        if hasattr(x, "item") and callable(getattr(x, "item")):
            try:
                return x.item()
            except Exception:
                pass
        # plain scalars
        if isinstance(x, (int, float, bool, str)):
            return x
        # last resort: try numpy conversion
        try:
            import numpy as np
            return np.asarray(x).tolist()
        except Exception:
            # give up: string-ify so JSON doesn't choke
            return str(x)


# ------------------------- MAIN SCRIPT ENTRY POINT -------------------------------

def main(gamma: float = None, FROM_DATA: bool = None, comp: bool = None, plot: bool = None, ALL: bool = False, COST: bool = False):
    parser = argparse.ArgumentParser(description="DRO LMI Optimization")
    #parser.add_argument("--comp", action="store_true", help="Run comparison btw baseline and LMI pipeline")
    #parser.add_argument("--base", action="store_true", help="Run baseline optimization")
    #parser.add_argument("--p", action="store_true", help="Force Plot")
    #parser.add_argument("--lmi", action="store_true", help="Run LMI pipeline optimization")
    args = parser.parse_args()

    if yaml is None:
        raise ImportError("PyYAML not available. Install with `pip install pyyaml`.")
    with open("problem___parameters.yaml", "r", encoding="utf-8") as f:
        cfg = yaml.safe_load(f)

    p = cfg.get("params", {})
    out = Path(p.get("directories", {}).get("artifacts", "./out/artifacts/")).with_suffix("")#.as_posix()
    m = p.get("ambiguity", {}).get("model", "W2")
    FROM_DATA = bool(p.get("FROM_DATA", False)) if FROM_DATA is None else FROM_DATA

    _runID = p.get("directories", {}).get("runID", "temp")
    _type = p.get("plant", {}).get("type", "explicit")
    _model = p.get("model", "independent") if m == "W2" else m
    _upd = bool(p.get("upd", 0))
    _method = p.get("method", "lmi")
    _plot = bool(p.get("plot", False)) if plot is None and not COST else plot
    _data = "DDD" if FROM_DATA else "MBD"
    _save = p.get("save", False) if not COST else False
    _comp = bool(p.get("comp", 0)) if comp is None else comp
    _ts = p.get("simulation", {}).get("ts", 0.5)

    #gamma = p.get("ambiguity", {}).get("gamma", 0.5) if gamma is None else gamma
    if gamma is None or m != "W2":
        gamma = p.get("ambiguity", {}).get("gamma", 0.5)
    else:
        gamma = gamma
    
    if _method=="lmi" and _upd: _method = "lmi-upd"

    var = float(p.get("ambiguity", {})["var"])
    n = p.get("dimensions", {}).get("nw", 2)
    Sigma_nom = np.array(p.get("ambiguity", {})["Sigma_nom"], dtype=float) if m!="Gaussian" else var * np.eye(n)

    noise = Noise(Sigma_nom=Sigma_nom, avrg=0, var=var, n=n, gamma=gamma)

    path_name = f"{_type}_{_model}_{_data}"

    if _comp:
        cmp = ResultsComparator(out_root=out, save=_save, ts=_ts)
        return cmp.compare_mbd_vs_ddd(path_name=path_name, method=_method, ID=_runID, plot=_plot)
        # cmp.compare_baseline_vs_lmi(path_name=path_name, plot=True)
    else:
        out = out / f"{_method}" / f"run_{_runID}"
        out.mkdir(parents=True, exist_ok=True)
        out = out / path_name
        out = out.as_posix()

        if _method == "base":
            print("\nRunning baseline optimization...")
            opt = baseline_optim_problem(out=out, Sigma_nom=Sigma_nom, gamma=gamma, plot=_plot if not ALL else False, save=_save if not ALL else True, FROM_DATA=FROM_DATA)
        else:
            print("\nRunning LMI pipeline optimization...")
            opt = lmi_pipeline_optim_problem(params=p, out=out, upd=_upd, noise=noise, plot=_plot if not ALL else False, save=_save if not ALL else True, FROM_DATA=FROM_DATA)

        if COST:
            return opt._return_final_cost()
        
    if bool(p.get("SNR", 1)): 
        plant, ctrl, sim, Sigma = opt.get_snr_vars()
        an = SNRAnalyzer(plant=plant, ctrl=ctrl, Sigma=Sigma)
        res = an.snr()
        print({k: v for k, v in res.items()}) # if k.endswith("_dB") or k=="spectral_radius_Acl"})
        an.plot_bars(title="SNR for my controller")

        if 1:
            an.plot_output_psd(sim["X"], "x", fs=1.0/p.get("simulation", {}).get("ts", 0.05), nfft=4096)
            an.plot_output_psd(sim["Y"], "y", fs=1.0/p.get("simulation", {}).get("ts", 0.05), nfft=4096)
            an.plot_output_psd(sim["Z"], "z", fs=1.0/p.get("simulation", {}).get("ts", 0.05), nfft=4096)
            an.plot_output_psd(sim["U"], "u", fs=1.0/p.get("simulation", {}).get("ts", 0.05), nfft=4096)

        # 1) Evaluate SNR via kernels for any Σ
        res = an.snr_from_kernels()
        print(res)

        # 2) Worst/best directions (trace-normalized Σ)
        print("Z worst/best:", an.worst_best_snr("z"))

        # 3) Sweep Σ orientation in a 2D subspace and plot
        thetas, traces = an.plot_snr_rotation_sweep(dims=(0,1), n_angles=181)

        # 4) Plot worst/best SNR bands
        an.plot_worst_best_lines()


# ----------------------------------------------------------------------------------

if __name__ == "__main__":
    if yaml is None:
        raise ImportError("PyYAML not available. Install with `pip install pyyaml`.")
    with open("problem___parameters.yaml", "r", encoding="utf-8") as f:
        cfg = yaml.safe_load(f)

    p = cfg.get("params", {})
    if not bool(p.get("ambiguity", {}).get("fixGamma", False)):     # set fixGamma: 0
        if p.get("method", "lmi") == "lmi":                         # ------|set method: "lmi"
            if p.get("model", "correlated") == "correlated":        # ------|------| set model: "correlated"
                if bool(p.get("ident", {}).get("stabilise", True)): # ------|------|-------| set stabilise: true
                    if bool(p.get("use_set_out_mats", False)):      # ------|------|-------|-------| set use_set_out_mats: true            | runID: Opt&SetOutMats&Stabilise
                        gamma = 0.41640786499873816
                    else:                                           # ------|------|-------|-------| set use_set_out_mats: false           | runID: Opt&Stabilise
                        gamma = 0.6180339887498949
                else:                                               # ------|------|-------| set stabilise: false
                    if bool(p.get("use_set_out_mats", False)):      # ------|------|-------|-------| set use_set_out_mats: true            | runID: Opt&SetOutMats
                        gamma = 0.06888370749726605
                    else:                                           # ------|------|-------|-------| set use_set_out_mats: false           | runID: Opt
                        gamma = 0.9016994374947425
            else:                                                   # ------|------| set model: "independent"
                gamma = p.get("ambiguity", {}).get("gamma", 0.5)
        else:
            gamma = p.get("ambiguity", {}).get("gamma", 0.5)
    else:                                                           # set fixGamma: 1
        gamma = p.get("ambiguity", {}).get("gamma", 0.5)


    COST = bool(p.get("COST", 0))
    if not COST:
        ALL = bool(p.get("ALL", False))
        if ALL:
            main(FROM_DATA=False, gamma=gamma, comp=False, ALL=ALL)
            main(FROM_DATA=True, gamma=gamma, comp=False, ALL=ALL)
            main(comp=True, gamma=gamma, ALL=ALL)
        else:
            main(gamma=gamma)
    else: 
        N = 20
        model = p.get("model", "independent")
        c_MBD, c_DDD = [], []
        for i in range(N):
            print(f"\n\n----- RUN {i+1}/100 -----")
            c_mbd = main(FROM_DATA=False, gamma=gamma, comp=False, ALL=False, COST=COST)
            c_ddd = main(FROM_DATA=True, gamma=gamma, comp=False, ALL=False, COST=COST)
            c_MBD.append(c_mbd)
            c_DDD.append(c_ddd)

        # After your loop:

        print("\n\n===== COST STATISTICS OVER ALL RUNS =====")
        c_MBD = np.array(c_MBD, dtype=float)
        c_DDD = np.array(c_DDD, dtype=float)

        # Stats
        mu_mbd, sd_mbd = float(np.mean(c_MBD)), float(np.std(c_MBD, ddof=1))
        mu_ddd, sd_ddd = float(np.mean(c_DDD)), float(np.std(c_DDD, ddof=1))
        print(f"MBD: mean={mu_mbd:.6g}, std={sd_mbd:.6g}")
        print(f"DDD: mean={mu_ddd:.6g}, std={sd_ddd:.6g}")

        # 1) Bar-with-errorbars
        labels = ["MBD", "DDD"]
        means = [mu_mbd, mu_ddd]
        stds  = [sd_mbd, sd_ddd]

        fig1, ax1 = plt.subplots(figsize=(6,4))
        x = np.arange(len(labels))
        ax1.bar(x, means, yerr=stds, capsize=6)
        ax1.set_xticks(x, labels)
        ax1.set_ylabel("Cost")
        ax1.set_title("Mean ± 1 SD over N runs")
        ax1.grid(True, axis="y", alpha=0.3)
        fig1.tight_layout()
        fig1.savefig("cost_mean_std_bar.pdf")
        #fig1.savefig("cost_mean_std_bar.png", dpi=200)

        # 2) Scatter of all runs + mean lines
        fig2, ax2 = plt.subplots(figsize=(7,4))
        ax2.scatter(np.zeros_like(c_MBD), c_MBD, alpha=0.6, label="MBD")
        ax2.scatter(np.ones_like(c_DDD),  c_DDD, alpha=0.6, label="DDD")
        ax2.hlines(mu_mbd, -0.2, 0.2)
        ax2.hlines(mu_ddd,  0.8, 1.2)
        ax2.set_xticks([0,1], labels)
        ax2.set_ylabel("Cost")
        ax2.set_title(f"Per-run costs with mean lines ({model})")
        ax2.grid(True, axis="y", alpha=0.3)
        fig2.tight_layout()
        fig2.savefig("cost_runs_scatter.pdf")
        #fig2.savefig("cost_runs_scatter.png", dpi=200)

        plt.show()

        fig3, ax3 = plt.subplots(figsize=(6,4))
        for i, (name, arr, mu, sd) in enumerate([("MBD", c_MBD, mu_mbd, sd_mbd),
                                                ("DDD", c_DDD, mu_ddd, sd_ddd)]):
            ax3.errorbar(i, mu, yerr=sd, fmt="o", capsize=6)
            ax3.vlines(i, mu - sd, mu + sd)

        ax3.set_xticks([0,1], ["MBD","DDD"])
        ax3.set_ylabel("Cost")
        ax3.set_title(f"Mean with ±1 SD whiskers ({model})")
        ax3.grid(True, axis="y", alpha=0.3)
        fig3.tight_layout()
        fig3.savefig("cost_mean_whiskers.pdf")
        plt.show()
