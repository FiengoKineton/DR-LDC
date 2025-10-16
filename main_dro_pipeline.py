# main_dro_pipeline.py
import json
from pathlib import Path
import numpy as np

from define_matrices import make_example_system, make_nominal_covariances
from dro_lmi import build_and_solve_dro_lmi
from recover_controller import (
    closed_loop_from_bar,
    recover_controller_from_closed_loop,
)
from simulate_closed_loop import simulate_closed_loop, plot_timeseries, save_npz
from systems import Plant, Controller


ART = Path("artifacts_lmi")
ART.mkdir(exist_ok=True)

def plant_to_dict(P: Plant):
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

def controller_to_dict(C: Controller):
    return {"Ac": C.Ac.tolist(), "Bc": C.Bc.tolist(), "Cc": C.Cc.tolist(), "Dc": C.Dc.tolist()}

def save_json(path: Path, payload: dict):
    with open(path, "w", encoding="utf-8") as f:
        json.dump(payload, f, indent=2)

def main():
    # 1) Define plant and nominal disturbance covariance (keep consistent with your LMI)
    plant, _ = make_example_system(seed=7)        # replace with your real matrices if needed
    Sigma_nom = make_nominal_covariances(plant.Bw.shape[1])
    gamma = 0.5                                   # Wasserstein radius (set as you wish)

    # 2) Solve DRO-LMI (choose "correlated" or "independent")
    model = "independent"                          # \in {"correlated", "independent"}
    res = build_and_solve_dro_lmi(
        plant=plant,
        Sigma_nom=Sigma_nom,
        gamma=gamma,
        model=model,
        solver="SCS",       # MOSEK if available, else SCS (set to "MOSEK" explicitly if you have it)
        verbose=False
    )

    if res.status not in ("optimal", "optimal_inaccurate"):
        raise RuntimeError(f"DRO-LMI solve failed: status={res.status}")

    # Debug prints (one time)
    print("Abar", np.shape(res.Abar), "Bbar", np.shape(res.Bbar), 
        "Cbar", np.shape(res.Cbar), "Dbar", np.shape(res.Dbar), "Pbar", np.shape(res.Pbar))

    # 3) From (Pbar, Abar, Bbar, Cbar, Dbar) build composite (Acl, Bcl, Ccl, Dcl) in original coords
    Acl, Bcl, Ccl, Dcl = closed_loop_from_bar(res.Pbar, res.Abar, res.Bbar, res.Cbar, res.Dbar)

    # 4) Recover (Ac, Bc, Cc, Dc) from composite and plant, with residual diagnostics
    ctrl_rec, residuals = recover_controller_from_closed_loop(plant, Acl, Bcl, Ccl, Dcl)
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
        "recovered_controller": controller_to_dict(ctrl_rec),
        "plant": plant_to_dict(plant),
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
    save_json(out_json, payload)
    print(f"[saved] {out_json}")

    # 6) Simulate with the recovered controller using the SAME plant and nominal Σ
    #    If you prefer covariance inflation for robustness testing, replace Sigma_nom here.
    sim = simulate_closed_loop(plant, ctrl_rec, Sigma_nom, T=800, seed=11)
    out_npz = ART / f"dro_pipeline_{model}_sim_T800_seed11.npz"
    save_npz(sim, str(out_npz))
    print(f"[saved] {out_npz}")

    # 7) Plot results
    plot_timeseries(sim)

if __name__ == "__main__":
    main()
