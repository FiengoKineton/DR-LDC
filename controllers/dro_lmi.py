import sys
import json, time, psutil, os, numpy as np
from pathlib import Path


from core import MatricesAPI, compose_closed_loop, Recover      # matrices.py
from simulate import Closed_Loop
from utils import Plant, Plant_cl, Controller, Noise

from ._dro_base import Baseline_dro_lmi
from ._dro_deepc import DeePC_dro_lmi
from ._dro_estm import Estm_dro_lmi
from ._dro_young import Young_dro_lmi
from ._dro_youngschur import Young_Schur_dro_lmi
from .non_convex import WFL


# ------------------------- DRO-LMI PIPELINE OPTIMIZATION PROBLEM ------------------

class lmi_pipeline_optim_problem(): 
    def __init__(self, params: dict, out: Path, noise: Noise, 
                 upd: bool = False, plot: bool = False, save: bool = False, 
                 FROM_DATA: bool = False, init_cond: str = "zero", N_sims: int = None,
                 disturbance_type: str = "Gaussian"):

        recover = Recover()
        api = MatricesAPI()
        cl = Closed_Loop() 

        STABLE, i = False, 0
        tot_time = 0
        gamma, Sigma_nom = noise.gamma, noise.Sigma_nom

        model = params.get("model", "correlated") if params.get("ambiguity", {}).get("model", "W2") != "Gaussian" else "independent"
        old = bool(params.get("old_upd", 1))
        inp = bool(params.get("inp", 0))
        estm_only = bool(params.get("estm_only", 0))
        N_sims = int(params.get("N_sims", 1)) if N_sims is None else N_sims
        non_convex = bool(params.get("non_convex", 0))
        Nsims_mats = bool(params.get("Nsims_mats", 0))


        if Nsims_mats:
            from analysis import NsimsMatricesAnalyzer
            an = NsimsMatricesAnalyzer(
                api=api,
                noise=noise,
                out_dir="out/EstmMats_Nsims",
                recompute=False,        # set True if you change estimation logic
            )

            an.run()  # estimates (or loads), then generates all plots
            sys.exit(0)

        self.proc = psutil.Process(os.getpid())
        self.solve_stats = []

        while not STABLE and i<7:
            t0 = time.perf_counter()
            cpu0 = self.proc.cpu_times()
            cpu0_total = cpu0.user + cpu0.system


            # 1) Define plant and nominal disturbance covariance (keep consistent with your LMI)
            if not upd or not FROM_DATA:
                plant, _ = api.get_system(FROM_DATA=FROM_DATA, gamma=noise.gamma, upd=upd)
                real_Z_mats = True
                N_sims = 0

                # 2) Solve DRO-LMI (choose "correlated" or "independent")
                res, num_violations = Baseline_dro_lmi(
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

                problem_params = {
                    "Methodology": "Baseline",
                    "FROM_DATA": FROM_DATA, # False
                    "augmented": True,
                    "reg_fro": False,
                    "non_convex": False,
                    "N_sims": 0,
                }

            else:
                approach = params.get("approach", "Young")
                vect = model == "independent"
                augmented = False #model == "correlated"
                reg_fro, reg_beta = True, True

                if not non_convex:

                    # 2) Solve DRO-LMI (choose "correlated" or "independent")
                    if not estm_only:
                        if old:
                            if approach == "DeePC":
                                real_Z_mats = False
                                N_sims = 0
                                res, plant, Sigma_nom, other, num_violations = DeePC_dro_lmi(
                                    api=api,
                                    vals=(upd, FROM_DATA, plot),
                                    noise=noise,
                                    model=model,
                                )
                                problem_params = {
                                    "Methodology": "DeePC",
                                    "FROM_DATA": FROM_DATA, # True
                                    "augmented": False,
                                    "reg_fro": False,
                                    "non_convex": False,
                                    "N_sims": N_sims,
                                }
                            else:
                                real_Z_mats = False
                                res, plant, Sigma_nom, other, num_violations = Young_dro_lmi(
                                    api=api,
                                    vals=(upd, FROM_DATA, plot),
                                    noise=noise,
                                    model=model,
                                    approach=approach,
                                    real_Z_mats=real_Z_mats,
                                    N_sims=N_sims,
                                ) 
                                problem_params = {
                                    "Methodology": approach,
                                    "FROM_DATA": FROM_DATA, # True
                                    "augmented": False,
                                    "reg_fro": False,
                                    "non_convex": False,
                                    "N_sims": N_sims,
                                }        
                        else:
                            real_Z_mats = False
                            estm_with_bounds = bool(params.get("estm_with_bounds", 0))
                            dro = Young_Schur_dro_lmi(
                                                    vals=(upd, FROM_DATA, vect, augmented, inp), 
                                                    model=model, N_sims=N_sims,
                                                    api=api, noise=noise, 
                                                    reg_fro=reg_fro, reg_beta=reg_beta, real_Z_mats=real_Z_mats,
                                                    estm_with_bounds=estm_with_bounds,
                                                    )
                            
                            res, plant, Sigma_nom, other, num_violations = dro.run()
                            if estm_with_bounds: N_sims = dro.N_sims_new

                            problem_params = {
                                "Methodology": "YoungSchur",
                                "FROM_DATA": FROM_DATA, # True
                                "augmented": augmented,
                                "reg_fro": reg_fro,
                                "reg_beta": reg_beta,
                                "non_convex": False,
                                "N_sims": N_sims,
                            }
                            
                        A, Bw, Bu, Cy, Dyw, Cz, Dzw, Dzu = plant
                        plant = Plant(A=A, Bw=Bw, Bu=Bu, Cz=Cz, Dzw=Dzw, Dzu=Dzu, Cy=Cy, Dyw=Dyw)
                        ADD = True
                    
                    else:
                        real_Z_mats = True
                        res, plant, Sigma_nom, num_violations = Estm_dro_lmi(
                            api=api,
                            noise=noise,
                            model=model,
                            SOLVER=params.get("solver", "MOSEK"),
                            real_Z_mats=real_Z_mats,
                            N_sims=N_sims,
                        )

                        A, Bw, Bu, Cz, Dzw, Dzu, Cy, Dyw = plant.A, plant.Bw, plant.Bu, plant.Cz, plant.Dzw, plant.Dzu, plant.Cy, plant.Dyw

                        problem_params = {
                            "Methodology": "EstmMat",
                            "FROM_DATA": FROM_DATA, # True
                            "augmented": True,
                            "reg_fro": False,
                            "non_convex": True,
                            "N_sims": N_sims,
                        }

                        ADD = False

                else: 
                    real_Z_mats = True
                    dro = WFL(vals=(upd, FROM_DATA, vect, augmented, inp), 
                              Bw_type = params.get("plant", {}).get("Bw_mode", "ident"),
                              model=model, N_sims=N_sims,
                              api=api, noise=noise, reg_fro=reg_fro, 
                              real_Z_mats=real_Z_mats)
                    
                    res, P, Sigma_nom, other = dro.run()
                    num_violations = [0, 1]

                    problem_params = {
                        "Methodology": "WFL",
                        "FROM_DATA": FROM_DATA, # True
                        "augmented": augmented,
                        "reg_fro": reg_fro,
                        "reg_beta": reg_beta,
                        "non_convex": non_convex,
                        "N_sims": N_sims,
                    }

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
            break

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
        print("\nFinal closed-loop cost J = ", self.final_cost)
        out_cost = out + f"___closed_loop_run_cost.npz"

        sim_snr = cl.simulate_ZW_snr(Z=sim["Z"], W=sim["W"], plot=plot)
        self.snr = sim_snr["snr_db"]
        print("\nGlobal SNR = ", self.snr)
        out_snr = out + f"___closed_loop_snr.npz"

        self.rho = rho
        self.lamda = res.lambda_opt
        self.Time = Time
        self.attempt = i
        self.stress = stress
        self.obj = res.obj_value
        self.solver = res.solver
        self.ratio_violation = num_violations[0] / num_violations[1]
        #dist = Disturbances(n=Bw.shape[1])

        # JSON
        payload = {
            "prob": "DDD" if FROM_DATA else "MBD",
            "meta": {
                "model": model,
                "disturbance_type": disturbance_type,
                "gamma": res.gamma,
                "objective": self.obj,
                "spectral_radius_Acl": rho,
                "Z_cost": self.final_cost,
                "lambda_opt": res.lambda_opt,
                "trace_Q_Sigma": np.trace(res.Q @ Sigma_nom),
                "real_Z_mats": real_Z_mats,
                "N_sims": N_sims,
                "SNR": self.snr,
            },
            "solver_performance": {
                "solver": self.solver,
                "status": res.status,
                "num_violations": int(num_violations[0]),
                "tot_constraints:": int(num_violations[1]),
                "Time_seconds": Time,
                "attempt": i,
                "Total_time": tot_time,
                "stress": self.stress,
            },
            "problem_params": problem_params,
            "controller": self.controller_to_dict(ctrl),
            "plant": self.plant_to_dict(plant),
            "plant_dims": {
                "nx": A.shape[0], 
                "nu": Bu.shape[1], 
                "nw": Bw.shape[1], 
                "ny": Cy.shape[0], 
                "nz": Cz.shape[0],
            },
            "initial_conds": {
                "X": sim["x_0"].tolist(), 
                "Xc": sim["xc_0"].tolist(), 
            },
            "composite_closed_loop": self.plant_cl_to_dict(plant_cl),
            "Acl_eigenvals": {
                "cartesian": eig_cart,
                "polar": eig_polar
            },            
            "disturbance": {
                "Sigma_nom": Sigma_nom.tolist(),
                #"Sigma": dist.Sigma_test.totlist() if dist.Sigma_test is not None else None,
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
            if (approach == 'Young' or approach == 'Mats') and old or not old and reg_beta:
                D, E, B, S, T, R = other
                payload["Young_approach"] = {
                    "approach": approach,
                    "D": self._to_serializable(D),  # handles (DeltaA, DeltaB) as matrices/tuples
                    "E": self._to_serializable(E),  # handles (EAA, EAB) as matrices/tuples
                    "B": self._to_serializable(B),  # now OK if matrix, vector, or scalars tuple
                    "S": self._to_serializable(S),  # now OK if matrix
                    "T": self._to_serializable(T),  # now OK if matrix
                    "R": self._to_serializable(R),
                }

        out_json = out + f"___results_run.json"
        if save: 
            cl.save_npz(sim, str(out_npz))
            print(f"[saved] {out_npz}")
            cl.save_npz(sim_composite, str(out_composite))
            print(f"[saved] {out_composite}")
            cl.save_npz(sim_cost, str(out_cost))   
            print(f"[saved] {out_cost}")
            cl.save_npz(sim_snr, str(out_snr))
            print(f"[saved] {out_snr}")
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
            "obj": self.obj,
            "solver": self.solver,
            "ratio_violation": self.ratio_violation,
            "snr": self.snr,
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

# =============================================================================================== #
