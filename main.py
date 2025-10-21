# main.py
import json, argparse, yaml, sys
import numpy as np
from pathlib import Path

from problem___baseline import run_once
from problem___dro_lmi import build_and_solve_dro_lmi

from utils___systems import Plant, Controller
from utils___simulate import Closed_Loop 
from utils___matrices import Recover, MatricesAPI, compose_closed_loop


# ------------------------- BASELINE OPTIMIZATION PROBLEM --------------------------

class baseline_optim_problem(): 
    def __init__(self, out: Path, Sigma_nom: np.ndarray):

        # Run optimization AND capture the exact plant used
        cl = Closed_Loop()  # instantiate simulation class
        api = MatricesAPI()

        plant, ctrl0 = api.get_system()
        api.print_plant(plant)

        Sigma_nom, base_cost, msg, cost_opt, rho, ctrl_opt = run_once(plant=plant, ctrl0=ctrl0, Sigma_nom=Sigma_nom)

        # Persist everything needed for reproducible simulation
        json_path = out + f"___results_run.json"
        self.save_results_json(json_path, Sigma_nom, base_cost, msg, cost_opt, rho, ctrl_opt, plant)
        print(f"[saved] {json_path}")

        # Load back the exact same objects and simulate
        Sigma_loaded, ctrl_loaded, plant_loaded, _ = self.load_results_json(json_path)
        sim = cl.simulate_closed_loop(plant_loaded, ctrl_loaded, Sigma_loaded)
        out_npz = out + f"___closed_loop_run.npz"
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

    def save_results_json(self, path, Sigma_nom, base_cost, msg, cost_opt, rho, ctrl_opt, plant):
        payload = {
            "Sigma_nom": Sigma_nom.tolist(),
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
    def __init__(self, params: dict, out: Path, Sigma_nom: np.ndarray):

        recover = Recover()
        api = MatricesAPI()
        cl = Closed_Loop() 

        # 1) Define plant and nominal disturbance covariance (keep consistent with your LMI)
        plant, _ = api.get_system()
        api.print_plant(plant)

        # 2) Solve DRO-LMI (choose "correlated" or "independent")
        gamma = params.get("ambiguity", {}).get("gamma", 0.5)    # Wasserstein radius (set as you wish)
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
        }

        out_json = out + f"___results_run.json"
        self.save_json(out_json, payload)
        print(f"[saved] {out_json}")

        # 6) Simulate with the recovered controller using the SAME plant and nominal Σ
        #    If you prefer covariance inflation for robustness testing, replace Sigma_nom here.
        sim = cl.simulate_closed_loop(plant, ctrl, Sigma_nom)
        out_npz = out + f"___closed_loop_run.npz"
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


# ------------------------- COMPARISON: BASELINE vs LMI ---------------------------

def compare_baseline_vs_lmi(out_root: str, path_name: str) -> None:
    """
    Looks for:
      baseline{path_name}/___results_run.json
      lmi{path_name}/___results_run.json

    Rebuilds plant/controller/Sigma for each, re-simulates with Closed_Loop,
    re-plots, and prints a tidy metric table. Also dumps a comparison JSON.
    """

    def _plant_from_dict(d: dict) -> Plant:
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

    def _controller_from_dict(d: dict) -> Controller:
        return Controller(
            Ac=np.array(d["Ac"], dtype=float),
            Bc=np.array(d["Bc"], dtype=float),
            Cc=np.array(d["Cc"], dtype=float),
            Dc=np.array(d["Dc"], dtype=float),
        )

    def _safe_l2(x):
        if x is None:
            return np.nan
        x = np.asarray(x, dtype=float)
        return float(np.sum(x * x))

    def _safe_peak(x):
        if x is None:
            return np.nan
        x = np.asarray(x, dtype=float)
        return float(np.max(np.abs(x)))

    def _compute_metrics(sim: dict) -> dict:
        """
        Tries to be minimally nosy. Looks for common keys and computes boring but useful scalars.
        Falls back to NaN if your simulation dict is an enigma.
        """
        keys = sim.keys() if isinstance(sim, dict) else []
        x = sim.get("x") if "x" in keys else None
        y = sim.get("y") if "y" in keys else None
        z = sim.get("z") if "z" in keys else None
        u = sim.get("u") if "u" in keys else None
        e = sim.get("e") if "e" in keys else None  # in case you store error

        # Energy-like integrals
        Jz = _safe_l2(z)          # "H2-ish" proxy on performance output
        Ju = _safe_l2(u)          # control effort
        Jx = _safe_l2(x)          # state energy
        Je = _safe_l2(e)          # tracking error, if any

        # Peaks
        z_peak = _safe_peak(z)
        u_peak = _safe_peak(u)

        # Optional crude settling metric: last-10% window std
        def _tail_std(a):
            if a is None:
                return np.nan
            a = np.asarray(a)
            n = a.shape[0]
            if n < 10:
                return np.nan
            tail = a[int(0.9 * n):]
            return float(np.std(tail))

        z_tail_std = _tail_std(z)
        y_tail_std = _tail_std(y)

        return {
            "Jz_sum_sq": Jz,
            "Ju_sum_sq": Ju,
            "Jx_sum_sq": Jx,
            "Je_sum_sq": Je,
            "z_peak_abs": z_peak,
            "u_peak_abs": u_peak,
            "z_tail_std": z_tail_std,
            "y_tail_std": y_tail_std,
        }

    def _load_json(path: Path) -> dict:
        with open(path, "r", encoding="utf-8") as f:
            return json.load(f)

    base_json = Path(out_root).with_suffix("").as_posix() + "/baseline" + path_name + "___results_run.json"
    lmi_json  = Path(out_root).with_suffix("").as_posix() + "/lmi" + path_name + "___results_run.json"

    if not Path(base_json).exists():
        raise FileNotFoundError(f"Baseline JSON not found at: {base_json}. Run the baseline pipeline first.")
    if not Path(lmi_json).exists():
        raise FileNotFoundError(f"LMI JSON not found at: {lmi_json}. Run the LMI pipeline first.")

    print(f"[compare] loading baseline: {base_json}")
    db = _load_json(base_json)
    Sigma_b = np.array(db.get("Sigma_nom"), dtype=float)
    plant_b = _plant_from_dict(db["plant"])
    ctrl_b  = _controller_from_dict(db["controller"])
    meta_b  = {
        "baseline_cost": db.get("baseline_cost"),
        "optimized_cost": db.get("optimized_cost"),
        "optimizer_status": db.get("optimizer_status"),
        "spectral_radius_Acl": db.get("spectral_radius_Acl"),
    }

    print(f"[compare] loading lmi: {lmi_json}")
    dl = _load_json(lmi_json)
    # LMI JSON layout is different by design, naturally
    Sigma_l = np.array(dl["disturbance"]["Sigma_nom"], dtype=float)
    plant_l = _plant_from_dict(dl["plant"])
    ctrl_l  = _controller_from_dict(dl["recovered_controller"])
    meta_l  = {
        "status": dl["meta"].get("status"),
        "objective": dl["meta"].get("objective"),
        "gamma": dl["meta"].get("gamma"),
        "lambda_opt": dl["meta"].get("lambda_opt"),
        "spectral_radius_Acl": dl["meta"].get("spectral_radius_Acl"),
        "model": dl["meta"].get("model"),
    }

    # Re-simulate with the exact same Closed_Loop machinery
    cl = Closed_Loop()
    print("[compare] simulating baseline closed loop...")
    sim_b = cl.simulate_closed_loop(plant_b, ctrl_b, Sigma_b)
    print("[compare] simulating lmi closed loop...")
    sim_l = cl.simulate_closed_loop(plant_l, ctrl_l, Sigma_l)

    # Re-plot both (because seeing is believing, and spreadsheets lie)
    print("[compare] plotting baseline...")
    cl.plot_timeseries(sim_b)
    print("[compare] plotting lmi...")
    cl.plot_timeseries(sim_l)

    # Compute metrics
    mb = _compute_metrics(sim_b)
    ml = _compute_metrics(sim_l)

    # Collate a small but meaningful report
    report = {
        "paths": {"baseline_json": base_json, "lmi_json": lmi_json},
        "baseline_meta": meta_b,
        "lmi_meta": meta_l,
        "metrics": {"baseline": mb, "lmi": ml},
        "deltas": {k: (ml.get(k, np.nan) - mb.get(k, np.nan)) for k in set(mb) | set(ml)},
    }

    # Save comparison next to the LMI run, because the LMI is the one that gets judged anyway
    comp_out = Path(out_root).with_suffix("").as_posix() + path_name + "___comparison_baseline_vs_lmi.json"
    with open(comp_out, "w", encoding="utf-8") as f:
        json.dump(report, f, indent=2)
    print(f"[compare] saved summary: {comp_out}")

    # Print a compact table to stdout for humans who distrust JSON
    def _fmt(x):
        return "nan" if x is None or (isinstance(x, float) and (np.isnan(x) or not np.isfinite(x))) else f"{x:.6g}"

    print("\n=== Comparison (lower is better for energies/peaks) ===")
    headers = [
        ("Jz_sum_sq", "∑ z^2"),
        ("Ju_sum_sq", "∑ u^2"),
        ("Jx_sum_sq", "∑ x^2"),
        ("Je_sum_sq", "∑ e^2"),
        ("z_peak_abs", "max|z|"),
        ("u_peak_abs", "max|u|"),
        ("z_tail_std", "tail std(z)"),
        ("y_tail_std", "tail std(y)"),
    ]
    print(f"{'metric':<16} {'baseline':>14} {'lmi':>14} {'lmi-baseline':>14}")
    for key, label in headers:
        b = mb.get(key, np.nan)
        l = ml.get(key, np.nan)
        d = l - b if np.isfinite(b) and np.isfinite(l) else np.nan
        print(f"{label:<16} {_fmt(b):>14} {_fmt(l):>14} {_fmt(d):>14}")

    # Meta sanity
    print("\n=== Meta sanity ===")
    print(f"baseline.rho(Acl) ≈ {meta_b.get('spectral_radius_Acl')}")
    print(f"lmi     .rho(Acl) ≈ {meta_l.get('spectral_radius_Acl')}")
    print(f"lmi.status = {meta_l.get('status')}, objective ≈ {meta_l.get('objective')}, model = {meta_l.get('model')}, gamma = {meta_l.get('gamma')}")


# ------------------------- MAIN SCRIPT ENTRY POINT -------------------------------

if __name__ == "__main__":
    parser = argparse.ArgumentParser(description="DRO LMI Optimization")
    parser.add_argument("--comp", action="store_true", help="Run comparison btw baseline and LMI pipeline")
    parser.add_argument("--base", action="store_true", help="Run baseline optimization")
    parser.add_argument("--lmi", action="store_true", help="Run LMI pipeline optimization")
    args = parser.parse_args()

    if yaml is None:
        raise ImportError("PyYAML not available. Install with `pip install pyyaml`.")
    with open("problem___parameters.yaml", "r", encoding="utf-8") as f:
        cfg = yaml.safe_load(f)

    p = cfg.get("params", {})
    out = p.get("directories", {}).get("artifacts", "./out/artifacts/")
    _type = p.get("plant", {}).get("type", "explicit")
    _model = p.get("model", "independent")
    FROM_DATA = bool(p.get("FROM_DATA", False))
    _data = "DDD" if FROM_DATA else "MBD"

    Sigma_nom = np.array(p.get("ambiguity", {})["Sigma_nom"], dtype=float)

    path_name = f"/run_02___{_type}_{_model}_{_data}"

    if args.comp:
        print("\nRunning comparison between baseline and LMI pipeline...")
        compare_baseline_vs_lmi(out_root=out, path_name=path_name)
    else:
        if args.base:
            print("\nRunning baseline optimization...")
            out = Path(out).with_suffix("").as_posix() + "/baseline" + path_name
            baseline_optim_problem(params=p, out=out, Sigma_nom=Sigma_nom)
        if args.lmi:
            print("\nRunning LMI pipeline optimization...")
            out = Path(out).with_suffix("").as_posix() + "/lmi" + path_name
            lmi_pipeline_optim_problem(params=p, out=out, Sigma_nom=Sigma_nom)

