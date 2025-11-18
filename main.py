# main.py
import json, argparse, yaml, sys, time, psutil, os
import numpy as np
from pathlib import Path
import matplotlib.pyplot as plt
import matplotlib.patches as mpatches
import pandas as pd


from matplotlib.lines import Line2D
from problem___baseline import run_once
from problem___dro_lmi import build_and_solve_dro_lmi, build_and_solve_dro_lmi_upd
from problem___DRO import DRO

from utils___systems import Plant, Controller, Plant_cl, Noise
from utils___simulate import Closed_Loop 
from utils___matrices import Recover, MatricesAPI, compose_closed_loop
from utils___SolutionComparison import ResultsComparator
from utils___SNR import SNRAnalyzer


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


# ------------------------- DRO-LMI PIPELINE OPTIMIZATION PROBLEM ------------------

class lmi_pipeline_optim_problem(): 
    def __init__(self, params: dict, out: Path, noise: Noise, upd: bool = False, plot: bool = False, save: bool = False, FROM_DATA: bool = False, init_cond: str = "zero"):

        recover = Recover()
        api = MatricesAPI()
        cl = Closed_Loop() 

        STABLE, i = False, 0
        tot_time = 0
        model = params.get("model", "correlated") if params.get("ambiguity", {}).get("model", "W2") != "Gaussian" else "independent"

        self.proc = psutil.Process(os.getpid())
        self.solve_stats = []

        while not STABLE and i<=5:
            t0 = time.perf_counter()
            cpu0 = self.proc.cpu_times()
            cpu0_total = cpu0.user + cpu0.system


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
                old = params.get("old_upd", True)

                # 2) Solve DRO-LMI (choose "correlated" or "independent")
                if old:
                    res, P, Sigma_nom, other = build_and_solve_dro_lmi_upd(
                        api=api,
                        vals=(upd, FROM_DATA, plot),
                        noise=noise,
                        model=model,
                        approach=approach,
                        d=(params.get("ambiguity", {}).get("model", "W2") == "Gaussian")
                    )
                else:
                    dro = DRO(vals=(upd, FROM_DATA, plot), model=model, 
                              api=api, noise=noise, )
                    
                    res, P, Sigma_nom, other = dro.run()

                A, Bw, Bu, Cy, Dyw, Cz, Dzw, Dzu = P
                plant = Plant(A=A, Bw=Bw, Bu=Bu, Cz=Cz, Dzw=Dzw, Dzu=Dzu, Cy=Cy, Dyw=Dyw)
                ADD = True


            t1 = time.perf_counter()
            cpu1 = self.proc.cpu_times()
            cpu1_total = cpu1.user + cpu1.system
            d_cpu = cpu1_total - cpu0_total


            Time = t1 - t0
            stress = d_cpu / Time if Time > 0 else 0.0
            tot_time += Time

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
            if rho < 1.00:
                STABLE = True
            i += 1

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
        


        # 6) Simulate with the recovered controller using the SAME plant and nominal Σ
        #    If you prefer covariance inflation for robustness testing, replace Sigma_nom here.
        sim = cl.simulate_closed_loop(plant=plant, ctrl=ctrl, Sigma_w=Sigma_nom, gamma=gamma, init_cond=init_cond)
        out_npz = out + f"___closed_loop_run.npz"

        sim_composite = cl.simulate_composite(Pcl=plant_cl, Sigma_w=Sigma_nom, gamma=gamma, init_cond=init_cond)
        out_composite = out + f"___closed_loop_composite.npz"

        sim_cost = cl.simulate_Z_cost(Z=sim["Z"], plot=plot)
        self.final_cost = sim_cost["J"]
        print("\nFinal closed-loop cost J =", self.final_cost)
        out_cost = out + f"___closed_loop_run_cost.npz"

        self.rho = rho
        self.lamda = res.lambda_opt
        self.Time = Time
        self.attempt = i
        self.stress = stress

        # JSON
        payload = {
            "prob": "DDD" if FROM_DATA else "MBD",
            "meta": {
                "model": model,
                "solver": res.solver,
                "status": res.status,
                "objective": res.obj_value,
                "gamma": res.gamma,
                "lambda_opt": res.lambda_opt,
                "spectral_radius_Acl": rho,
                "Z_cost": self.final_cost,
                "Time_seconds": Time,
                "attempt": i,
                "Total_time": tot_time,
                "stress": self.stress,
            },
            "controller": self.controller_to_dict(ctrl),
            "plant": self.plant_to_dict(plant),
            "plant_dims": {
                "nx": A.shape[0], 
                "nu": Bu.shape[1], 
                "nw": Bw.shape[1], 
                "ny": Cy.shape[0], 
                "nz": Cz.shape[0],
            },
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
        if save: 
            cl.save_npz(sim, str(out_npz))
            print(f"[saved] {out_npz}")
            cl.save_npz(sim_composite, str(out_composite))
            print(f"[saved] {out_composite}")
            cl.save_npz(sim_cost, str(out_cost))   
            print(f"[saved] {out_cost}")
            self.save_json(out_json, payload)
            print(f"[saved] {out_json}")

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

    def _return_final_infos(self):
        infos = {
            "J": self.final_cost,
            "lamda": self.lamda,
            "rho": self.rho,
            "time": self.Time, 
            "attempts": self.attempt,
            "stress": self.stress,
        }
        return infos

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

def main(gamma: float = None, FROM_DATA: bool = None, comp: bool = None, plot: bool = None, ALL: bool = False, COST: bool = False, info: bool = False):
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
    out = Path(p.get("directories", {}).get("artifacts", "./out/artifacts/")).with_suffix("")#.as_posix()
    m = p.get("ambiguity", {}).get("model", "W2")
    FROM_DATA = bool(p.get("FROM_DATA", False)) if FROM_DATA is None else FROM_DATA

    _runID = p.get("directories", {}).get("runID", "temp")
    _type = p.get("plant", {}).get("type", "explicit")
    _upd = bool(p.get("upd", 0))
    _re_evaluate = bool(p.get("re_evaluate", 0)) if not ALL else False
    _method = p.get("method", "lmi")
    _plot = bool(p.get("plot", False)) if plot is None and not COST else plot
    _data = "DDD" if FROM_DATA else "MBD"
    _save = p.get("save", False) if not COST else False
    _comp = bool(p.get("comp", 0)) if comp is None else comp
    _ts = p.get("simulation", {}).get("ts", 0.5)
    _init_cond = p.get("init_cond", "rand")

    if m == "W2":
        _model = p.get("model", "independent")
    elif m == "2W":
        _model = m + "_" + p.get("model", "independent")
    else:
        _model = m

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
        return cmp.compare_mbd_vs_ddd(path_name=path_name, method=_method, ID=_runID, plot=_plot, re_evaluate=_re_evaluate, init_cond=_init_cond)
        # cmp.compare_baseline_vs_lmi(path_name=path_name, plot=True)
    else:
        out = out / f"{_method}" / f"run_{_runID}"
        out.mkdir(parents=True, exist_ok=True)
        out = out / path_name
        out = out.as_posix()

        if _method == "base":
            print("\nRunning baseline optimization...")
            opt = baseline_optim_problem(out=out, Sigma_nom=Sigma_nom, gamma=gamma, plot=_plot if not ALL else False, save=_save if not ALL else True, FROM_DATA=FROM_DATA, init_cond=_init_cond)
        else:
            print("\nRunning LMI pipeline optimization...")
            opt = lmi_pipeline_optim_problem(params=p, out=out, upd=_upd, noise=noise, plot=_plot if not ALL else False, save=_save if not ALL else True, FROM_DATA=FROM_DATA, init_cond=_init_cond)

        if COST or info:
            return opt._return_final_infos(), _model
        
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


def select_gamma(p):
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
    
    return gamma

# ------------------------- Evaluation --------------------------------------------

def MutipleRunsEvaluation(p, COST: bool = True, N: int = None):
    N = 20 if N is None else N
    model = p.get("model", "independent")
    save = bool(p.get("save", False))
    plot = bool(p.get("plot", False))
    c_MBD, c_DDD = [], []
    l_MBD, l_DDD = [], []
    r_MBD, r_DDD = [], []
    t_MBD, t_DDD = [], []
    a_MBD, a_DDD = [], []
    s_MBD, s_DDD = [], []

    out = Path(p.get("directories", {}).get("artifacts", "./out/artifacts/")).with_suffix("")
    out = out / "MutipleRunsEvaluation"
    out.mkdir(parents=True, exist_ok=True)
    path = out / p.get("directories", {}).get("runID", "temp")
    path.mkdir(parents=True, exist_ok=True)
    mbd_file = path / f"_{model}_MBD_runs.csv"
    ddd_file = path / f"_{model}_DDD_runs.csv"

    NOT_FOUND = not (mbd_file.is_file() and ddd_file.is_file())


    if bool(p.get("re_evaluate", 0)) or NOT_FOUND:
        K_RECENT = 5
        unstable_hit = False

        for i in range(N):
            print("\n\n\n\n"
                "==============================\n"
                f"----- RUN {i+1}/{N} -----\n"
                "==============================\n"
                "\n\n\n\n")
            try:
                infos_mbd, *_ = main(FROM_DATA=False, gamma=gamma, comp=False, ALL=False, COST=COST)
                c_mbd = infos_mbd["J"]
                l_mbd = infos_mbd["lamda"]
                r_mbd = infos_mbd["rho"]
                t_mbd = infos_mbd["time"]
                a_mbd = infos_mbd["attempts"]
                s_mbd = infos_mbd["stress"]

                c_MBD.append(c_mbd)
                l_MBD.append(l_mbd)
                r_MBD.append(r_mbd)
                t_MBD.append(t_mbd)
                a_MBD.append(a_mbd)
                s_MBD.append(s_mbd)
            except Exception as e:
                print(f"Error occurred in MBD run {i+1}: {e}")

            try:
                infos_ddd, *_ = main(FROM_DATA=True, gamma=gamma, comp=False, ALL=False, COST=COST)
                c_ddd = infos_ddd["J"]
                l_ddd = infos_ddd["lamda"]
                r_ddd = infos_ddd["rho"]
                t_ddd = infos_ddd["time"]
                a_ddd = infos_ddd["attempts"]
                s_ddd = infos_ddd["stress"]

                c_DDD.append(c_ddd)
                l_DDD.append(l_ddd)
                r_DDD.append(r_ddd)
                t_DDD.append(t_ddd)
                a_DDD.append(a_ddd)
                s_DDD.append(s_ddd)

                # ---------- NEW: check last K rho_DDD ----------
                if len(r_DDD) >= K_RECENT:
                    recent_rho = np.array(r_DDD[-K_RECENT:], dtype=float)
                    if np.all(recent_rho >= 1.0):
                        print(
                            f"[WARN] Last {K_RECENT} DDD rho values are >= 1.0; "
                            f"recent_rho = {recent_rho}. Stopping further runs."
                        )
                        unstable_hit = True
                        break

            except Exception as e:
                print(f"Error occurred in DDD run {i+1}: {e}")

        print(f"Completed {len(c_MBD)} MBD runs, {len(c_DDD)} DDD runs. unstable_hit={unstable_hit}")

        print("\n\n===== COST STATISTICS OVER ALL RUNS =====")
        c_MBD = np.array(c_MBD, dtype=float)
        c_DDD = np.array(c_DDD, dtype=float)
        l_MBD = np.array(l_MBD, dtype=float)
        l_DDD = np.array(l_DDD, dtype=float)
        r_MBD = np.array(r_MBD, dtype=float)
        r_DDD = np.array(r_DDD, dtype=float)
        t_MBD = np.array(t_MBD, dtype=float)
        t_DDD = np.array(t_DDD, dtype=float)
        a_MBD = np.array(a_MBD, dtype=float)
        a_DDD = np.array(a_DDD, dtype=float)
        s_MBD = np.array(s_MBD, dtype=float)
        s_DDD = np.array(s_DDD, dtype=float)

        # ------------------------------------------------------------------
        # SAVE TO CSV (one file for MBD, one for DDD)
        # ------------------------------------------------------------------

        # In case some runs failed on one side, truncate to common length
        n_mbd = len(c_MBD)
        n_ddd = len(c_DDD)
        n_common_mbd = min(n_mbd, len(l_MBD), len(r_MBD), len(t_MBD), len(a_MBD), len(s_MBD))
        n_common_ddd = min(n_ddd, len(l_DDD), len(r_DDD), len(t_DDD), len(a_DDD), len(s_DDD))

        # MBD table: [run, J, lambda, rho, time, attempts, stress]
        mbd_data = np.column_stack([
            np.arange(n_common_mbd),
            c_MBD[:n_common_mbd],
            l_MBD[:n_common_mbd],
            r_MBD[:n_common_mbd],
            t_MBD[:n_common_mbd],
            a_MBD[:n_common_mbd],
            s_MBD[:n_common_mbd],
        ])

        # DDD table
        ddd_data = np.column_stack([
            np.arange(n_common_ddd),
            c_DDD[:n_common_ddd],
            l_DDD[:n_common_ddd],
            r_DDD[:n_common_ddd],
            t_DDD[:n_common_ddd],
            a_DDD[:n_common_ddd],
            s_DDD[:n_common_ddd],
        ])

        if save:
            np.savetxt(
                mbd_file,
                mbd_data,
                delimiter=",",
                header="run,J,lambda,rho,time,attempts,stress",
                comments=""
            )

            np.savetxt(
                ddd_file,
                ddd_data,
                delimiter=",",
                header="run,J,lambda,rho,time,attempts,stress",
                comments=""
            )

    else:
        # load, skip header row
        mbd_data = np.loadtxt(mbd_file, delimiter=",", skiprows=1)
        ddd_data = np.loadtxt(ddd_file, delimiter=",", skiprows=1)

        # robustify: if only 1 row, loadtxt returns 1D
        mbd_data = np.atleast_2d(mbd_data)
        ddd_data = np.atleast_2d(ddd_data)

        # columns: run,J,lambda,rho,time,attempts,stress
        runs_MBD = mbd_data[:, 0]
        c_MBD    = mbd_data[:, 1]
        l_MBD    = mbd_data[:, 2]
        r_MBD    = mbd_data[:, 3]
        t_MBD    = mbd_data[:, 4]
        a_MBD    = mbd_data[:, 5]
        s_MBD    = mbd_data[:, 6]

        runs_DDD = ddd_data[:, 0]
        c_DDD    = ddd_data[:, 1]
        l_DDD    = ddd_data[:, 2]
        r_DDD    = ddd_data[:, 3]
        t_DDD    = ddd_data[:, 4]
        a_DDD    = ddd_data[:, 5]
        s_DDD    = ddd_data[:, 6]

    def analyze_and_plot_metric(
        y_MBD,
        y_DDD,
        metric_name: str,
        model: str,
        path: Path,
        save: bool,
        N_runs: int | None = None,
    ):
        """
        y_MBD, y_DDD : 1D arrays (list or np.ndarray)
            Values of the metric for each run.
        metric_name  : str
            Used in labels / titles / filenames, e.g. "cost", "lambda", "rho", "time", "attempts", "stress".
        model        : str
            Model name for titles / filenames, e.g. "independent", "correlated".
        path         : Path
            Base directory where to save artifacts.
        save         : bool
            If True, save figures and CSV.
        N_runs       : int or None
            Total planned runs, for labeling / padding. If None, use max(len(MBD), len(DDD)).
        """

        y_MBD = np.asarray(y_MBD, dtype=float)
        y_DDD = np.asarray(y_DDD, dtype=float)

        # ----- basic stats (ignoring NaNs if present) -----
        mu_mbd = float(np.nanmean(y_MBD))
        sd_mbd = float(np.nanstd(y_MBD, ddof=1))
        mu_ddd = float(np.nanmean(y_DDD))
        sd_ddd = float(np.nanstd(y_DDD, ddof=1))

        print(f"[{metric_name}] MBD: mean={mu_mbd:.6g}, std={sd_mbd:.6g}")
        print(f"[{metric_name}] DDD: mean={mu_ddd:.6g}, std={sd_ddd:.6g}")

        mbd_label = rf"MBD  ($\mu$={mu_mbd:.3g}, $\sigma$={sd_mbd:.3g})"
        ddd_label = rf"DDD  ($\mu$={mu_ddd:.3g}, $\sigma$={sd_ddd:.3g})"

        labels = ["MBD", "DDD"]

        # =========================
        # 1) scatter + mean lines
        # =========================
        fig2, ax2 = plt.subplots(figsize=(7, 4))
        ax2.scatter(np.zeros_like(y_MBD), y_MBD, alpha=0.6, label="MBD", color="blue")
        ax2.scatter(np.ones_like(y_DDD),  y_DDD, alpha=0.6, label="DDD", color="orange")
        ax2.hlines(mu_mbd, -0.2, 0.2)
        ax2.hlines(mu_ddd,  0.8, 1.2)
        ax2.set_xticks([0, 1], labels)
        ax2.set_ylabel(metric_name.capitalize())
        ax2.set_title(f"Per-run {metric_name} over runs ({model})")
        ax2.grid(True, axis="y", alpha=0.3)

        handles = [
            Line2D([0], [0], marker='o', linestyle='None', label=mbd_label, color="blue"),
            Line2D([0], [0], marker='o', linestyle='None', label=ddd_label, color="orange"),
        ]
        ax2.legend(handles=handles, title=f"{metric_name.capitalize()} stats", loc="best",
                frameon=True, framealpha=0.9)
        fig2.tight_layout()

        # =========================
        # 2) time-like overlay
        # =========================

        # Align on common index
        t_max = min(len(y_MBD), len(y_DDD))
        t = np.arange(1, t_max + 1, dtype=int)

        y_mbd = y_MBD[:t_max]
        y_ddd = y_DDD[:t_max]

        finite = np.isfinite(y_mbd) & np.isfinite(y_ddd)
        better = (y_ddd < y_mbd) & finite   # DDD better (green)
        worse  = (~better) & finite         # MBD better or equal (red)

        fig4, ax4 = plt.subplots(figsize=(9, 4.8))

        ax4.plot(t, y_mbd, marker="o", linewidth=1.5, alpha=0.9, label="MBD")
        ax4.plot(t, y_ddd, marker="s", linewidth=1.5, alpha=0.9, label="DDD")

        ax4.fill_between(t, y_mbd, y_ddd, where=better, interpolate=True,
                        color="green", alpha=0.12)
        ax4.fill_between(t, y_mbd, y_ddd, where=worse, interpolate=True,
                        color="red", alpha=0.12)

        ax4.set_xlabel("Run")
        ax4.set_ylabel(metric_name.capitalize())
        ax4.set_title(f"Per-run {metric_name} comparison with conditional shading ({model})")
        ax4.grid(True, alpha=0.3)

        handles, leg_labels = ax4.get_legend_handles_labels()
        handles += [
            mpatches.Patch(color="green", alpha=0.12, label="DDD < MBD"),
            mpatches.Patch(color="red",   alpha=0.12, label="DDD ≥ MBD"),
        ]
        leg_labels += ["DDD < MBD", "DDD ≥ MBD"]

        ax4.legend(handles, leg_labels, loc="best")
        fig4.tight_layout()

        # =========================
        # 3) CSV save
        # =========================
        if save:
            path.mkdir(parents=True, exist_ok=True)

            fig2.savefig(path / f"{model}_{metric_name}_runs_scatter.pdf")
            fig4.savefig(path / f"{model}_{metric_name}_runs_timeseries_overlay_shaded.pdf")

            if N_runs is None:
                N_runs = max(len(y_MBD), len(y_DDD))
            runs = np.arange(1, N_runs + 1)

            def pad(a, n):
                out = np.full(n, np.nan, dtype=float)
                out[:len(a)] = a
                return out

            df = pd.DataFrame({
                "run": runs,
                f"{metric_name}_MBD": pad(y_MBD, N_runs),
                f"{metric_name}_DDD": pad(y_DDD, N_runs),
            })

            csv_path = path / f"{model}_per_run_{metric_name}.csv"
            df.to_csv(csv_path, index=False)
            print(f"Saved per-run {metric_name} to {csv_path}")

        if plot: 
            plt.show()

    analyze_and_plot_metric(c_MBD, c_DDD, "cost",    model, path, save, N_runs=N)
    analyze_and_plot_metric(l_MBD, l_DDD, "lambda",  model, path, save, N_runs=N)
    analyze_and_plot_metric(r_MBD, r_DDD, "rho",     model, path, save, N_runs=N)
    analyze_and_plot_metric(t_MBD, t_DDD, "time",    model, path, save, N_runs=N)
    analyze_and_plot_metric(a_MBD, a_DDD, "attempts",model, path, save, N_runs=N)
    analyze_and_plot_metric(s_MBD, s_DDD, "stress",  model, path, save, N_runs=N)


def print_infos_comparison(m: str, infos_mbd: dict, infos_ddd: dict):
    """
    Pretty-print a comparison table between MBD and DDD info dicts.

    Expected keys:
        "J", "lamda", "rho", "time", "attempts", "stress"
    """
    metrics = [
        ("J",       "Cost J"),
        ("lamda",   "λ"),
        ("rho",     "ρ"),
        ("time",    "Time [s]"),
        ("attempts","Attempts"),
        ("stress",  "Stress"),
    ]

    def fmt(v):
        # crude but effective formatter
        if isinstance(v, (int, float)):
            return f"{v:.4g}"
        return str(v)

    print("\n" + "=" * 70)
    print(f" {m} summary ".center(70, "="))
    print("=" * 70)

    header = f"{'Metric':<15}{'MBD':>15}{'DDD':>15}{'DDD - MBD':>15}"
    print(header)
    print("-" * 70)

    for key, label in metrics:
        v_m = infos_mbd.get(key, None)
        v_d = infos_ddd.get(key, None)

        # difference only if both are numeric
        if isinstance(v_m, (int, float)) and isinstance(v_d, (int, float)):
            diff = v_d - v_m
            diff_str = f"{diff:+.3g}"
        else:
            diff_str = ""

        print(f"{label:<15}{fmt(v_m):>15}{fmt(v_d):>15}{diff_str:>15}")

    print("=" * 70 + "\n")


# ----------------------------------------------------------------------------------

if __name__ == "__main__":
    if yaml is None:
        raise ImportError("PyYAML not available. Install with `pip install pyyaml`.")
    with open("problem___parameters.yaml", "r", encoding="utf-8") as f:
        cfg = yaml.safe_load(f)

    p = cfg.get("params", {})
    gamma = select_gamma(p)

    COST = bool(p.get("COST", 0))
    if not COST:
        ALL = bool(p.get("ALL", False))
        if ALL:
            infos_mbd, m = main(FROM_DATA=False, gamma=gamma, comp=False, ALL=ALL, info=True)
            infos_ddd, _ = main(FROM_DATA=True, gamma=gamma, comp=False, ALL=ALL, info=True)
            main(comp=True, gamma=gamma, ALL=ALL)

            print_infos_comparison(m, infos_mbd, infos_ddd)
        else:
            main(gamma=gamma)
    else: 
        MutipleRunsEvaluation(p, N=15)
 