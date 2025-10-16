# main.py
import json
import numpy as np
from pathlib import Path

from run import run_once
from systems import Plant, Controller
from simulate_closed_loop import simulate_closed_loop, plot_timeseries, save_npz

ARTIFACTS = Path("artifacts")
ARTIFACTS.mkdir(exist_ok=True)

def plant_to_dict(P: Plant):
    return {
        "A": P.A.tolist(), "Bw": P.Bw.tolist(), "Bu": P.Bu.tolist(),
        "Cz": P.Cz.tolist(), "Dzw": P.Dzw.tolist(), "Dzu": P.Dzu.tolist(),
        "Cy": P.Cy.tolist(), "Dyw": P.Dyw.tolist(),
    }

def plant_from_dict(d: dict) -> Plant:
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

def controller_to_dict(C: Controller):
    return {"Ac": C.Ac.tolist(), "Bc": C.Bc.tolist(), "Cc": C.Cc.tolist(), "Dc": C.Dc.tolist()}

def controller_from_dict(d: dict) -> Controller:
    return Controller(
        Ac=np.array(d["Ac"], dtype=float),
        Bc=np.array(d["Bc"], dtype=float),
        Cc=np.array(d["Cc"], dtype=float),
        Dc=np.array(d["Dc"], dtype=float),
    )

def save_results_json(path, Sigma_eff, base_cost, msg, cost_opt, rho, ctrl_opt, plant):
    payload = {
        "Sigma_eff": Sigma_eff.tolist(),
        "baseline_cost": base_cost,
        "optimizer_status": msg,
        "optimized_cost": cost_opt,
        "spectral_radius_Acl": rho,
        "controller": controller_to_dict(ctrl_opt),
        "plant": plant_to_dict(plant),
    }
    with open(path, "w", encoding="utf-8") as f:
        json.dump(payload, f, indent=2)

def load_results_json(path):
    with open(path, "r", encoding="utf-8") as f:
        d = json.load(f)
    Sigma_eff = np.array(d["Sigma_eff"], dtype=float)
    ctrl = controller_from_dict(d["controller"])
    plant = plant_from_dict(d["plant"])
    meta = {
        "baseline_cost": d["baseline_cost"],
        "optimized_cost": d["optimized_cost"],
        "optimizer_status": d["optimizer_status"],
        "spectral_radius_Acl": d["spectral_radius_Acl"],
    }
    return Sigma_eff, ctrl, plant, meta

def main():
    # Run optimization AND capture the exact plant used
    Sigma_eff, base_cost, msg, cost_opt, rho, ctrl_opt, plant = run_once()

    # Persist everything needed for reproducible simulation
    json_path = ARTIFACTS / "results_run.json"
    save_results_json(json_path, Sigma_eff, base_cost, msg, cost_opt, rho, ctrl_opt, plant)

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
    Sigma_loaded, ctrl_loaded, plant_loaded, meta = load_results_json(json_path)
    sim = simulate_closed_loop(plant_loaded, ctrl_loaded, Sigma_loaded, T=800, seed=11)
    out_npz = ARTIFACTS / "closed_loop_run_seed11_T800.npz"
    save_npz(sim, str(out_npz))
    print(f"[saved] {out_npz}")

    plot_timeseries(sim)

if __name__ == "__main__":
    main()
