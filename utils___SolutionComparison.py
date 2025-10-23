import json, sys
from pathlib import Path
import numpy as np
import matplotlib.pyplot as plt
from typing import Tuple, Optional, Mapping


from utils___systems import Plant, Controller, Plant_cl
from utils___simulate import Closed_Loop
from utils___matrices import MatricesAPI



class ResultsComparator:
    def __init__(self, out_root: str):
        """
        out_root: artifacts root, e.g. "./out/artifacts/"
        """
        self.out_root = Path(out_root).with_suffix("")

    # ------------------------ helpers: i/o & reconstruction ------------------------

    @staticmethod
    def _load_json(p: Path) -> dict:
        with open(p, "r", encoding="utf-8") as f:
            return json.load(f)

    @staticmethod
    def _plant_from_dict(d: dict) -> 'Plant':
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

    @staticmethod
    def _plant_cl_from_dict(d: dict) -> 'Plant_cl':
        return Plant_cl(
            Acl=np.array(d["Acl"], dtype=float),
            Bcl=np.array(d["Bcl"], dtype=float),
            Ccl=np.array(d["Ccl"], dtype=float),
            Dcl=np.array(d["Dcl"], dtype=float),
        )
    
    @staticmethod
    def _controller_from_dict(d: dict) -> 'Controller':
        return Controller(
            Ac=np.array(d["Ac"], dtype=float),
            Bc=np.array(d["Bc"], dtype=float),
            Cc=np.array(d["Cc"], dtype=float),
            Dc=np.array(d["Dc"], dtype=float),
        )

    @staticmethod
    def _fro_stats(A: np.ndarray, B: np.ndarray):
        D = np.asarray(A, float) - np.asarray(B, float)
        return {
            "shape": str(A.shape),
            "fro_norm": float(np.linalg.norm(D, ord='fro')),
            "max_abs": float(np.max(np.abs(D))) if D.size else 0.0,
            "mean_abs": float(np.mean(np.abs(D))) if D.size else 0.0,
        }

    @staticmethod
    def _safe_l2(x):
        if x is None:
            return np.nan
        x = np.asarray(x, dtype=float)
        return float(np.sum(x * x))

    @staticmethod
    def _safe_peak(x):
        if x is None:
            return np.nan
        x = np.asarray(x, dtype=float)
        return float(np.max(np.abs(x)))

    def _compute_metrics(self, sim: dict) -> dict:
        keys = sim.keys() if isinstance(sim, dict) else []
        x = sim.get("x") if "x" in keys else None
        y = sim.get("y") if "y" in keys else None
        z = sim.get("z") if "z" in keys else None
        u = sim.get("u") if "u" in keys else None
        e = sim.get("e") if "e" in keys else None

        def _tail_std(a):
            if a is None:
                return np.nan
            a = np.asarray(a)
            n = a.shape[0]
            if n < 10:
                return np.nan
            tail = a[int(0.9 * n):]
            return float(np.std(tail))

        return {
            "Jz_sum_sq": self._safe_l2(z),
            "Ju_sum_sq": self._safe_l2(u),
            "Jx_sum_sq": self._safe_l2(x),
            "Je_sum_sq": self._safe_l2(e),
            "z_peak_abs": self._safe_peak(z),
            "u_peak_abs": self._safe_peak(u),
            "z_tail_std": _tail_std(z),
            "y_tail_std": _tail_std(y),
        }

    @staticmethod
    def _npz_extract_states(npz: Mapping) -> Tuple[np.ndarray, np.ndarray,
                                                   Optional[np.ndarray],
                                                   Optional[np.ndarray],
                                                   Optional[np.ndarray], 
                                                   Optional[np.ndarray]]:
        """
        Extract time vector t and trajectories X (state), and optionally Y (measured),
        Z (performance), U (input) from an npz-like mapping.

        Returns:
            t : (T,)               time index (np.arange(T) if absent)
            X : (T, nx)            states (required)
            Y : (T, ny) or None    measured outputs if present
            Z : (T, nz) or None    performance outputs if present
            U : (T, nu) or None    inputs if present

        Rules:
        - Accepts many plausible keys for robustness.
        - If an array is shaped (n, T), it is transposed to (T, n).
        - 1D arrays are promoted to (T, 1).
        - Missing streams are returned as None.
        """

        def _get_first(npz_map, keys):
            for k in keys:
                if k in npz_map:
                    return np.array(npz_map[k])
            return None

        def _to_time_major(arr: Optional[np.ndarray]) -> Optional[np.ndarray]:
            if arr is None:
                return None
            arr = np.array(arr)
            # Squeeze harmless singleton dims
            arr = np.squeeze(arr)
            if arr.ndim == 1:
                # Promote to (T, 1)
                return arr.reshape(-1, 1)
            if arr.ndim != 2:
                raise ValueError(f"Expected 1D or 2D array, got shape {arr.shape}")
            T, n = arr.shape
            # Heuristic: if it's (n, T) with n << T, flip to (T, n)
            if T < n:
                arr = arr.T
            return arr

        # Time
        t = _get_first(npz, keys=("t", "time", "times"))
        if t is not None:
            t = np.ravel(t)

        # State X (required)
        X = _get_first(npz, keys=("x", "X", "states", "state", "traj_X", "traj"))
        if X is None:
            raise KeyError("Could not find a state matrix in NPZ (x/X/states/state/traj_X/traj).")
        X = _to_time_major(X)

        # Y, Z, U (optional; multiple likely aliases)
        Y = _get_first(npz, keys=("y", "Y", "meas", "measured", "outputs_y", "traj_Y"))
        Z = _get_first(npz, keys=("z", "Z", "perf", "performance", "outputs_z", "traj_Z"))
        U = _get_first(npz, keys=("u", "U", "input", "inputs", "traj_U"))
        Xc = _get_first(npz, keys=("xc", "x_c", "Xc", "X_c"))

        Y = _to_time_major(Y)
        Z = _to_time_major(Z)
        U = _to_time_major(U)
        Xc = _to_time_major(Xc)

        # If time missing, synthesize from X length
        if t is None:
            t = np.arange(X.shape[0])

        # Basic length consistency check; mismatch won’t hard-fail, just warn in logs if you use logging
        T = X.shape[0]
        for name, arr in (("Y", Y), ("Z", Z), ("U", U), ("Xc", Xc)):
            if arr is not None and arr.shape[0] != T:
                # Align by truncation to the minimum T to avoid broadcasting disasters.
                Tmin = min(T, arr.shape[0])
                X = X[:Tmin]
                t = t[:Tmin]
                if Y is not None: Y = Y[:Tmin]
                if Z is not None: Z = Z[:Tmin]
                if U is not None: U = U[:Tmin]
                if Xc is not None: Xc = Xc[:Tmin]
                break

        return t, X, Y, Z, U, Xc

    @staticmethod
    def _plot_overlay_states(t, XM, XD, l, title: str):
        XM = np.atleast_2d(XM)
        XD = np.atleast_2d(XD)
        T = min(XM.shape[0], XD.shape[0])
        XM, XD = XM[:T], XD[:T]
        t = np.arange(T) if len(t) != T else t[:T]
        nx = XM.shape[1] if XM.ndim == 2 else 1
        if XM.ndim == 1:
            XM = XM.reshape(-1, 1)
            XD = XD.reshape(-1, 1)
            nx = 1
        fig, axes = plt.subplots(nx, 1, figsize=(10, max(3, 2 * nx)), sharex=True)
        if nx == 1:
            axes = [axes]
        for i, ax in enumerate(axes):
            ax.plot(t, XM[:, i], label="MBD", linewidth=1.6)
            ax.plot(t, XD[:, i], label="DDD", linewidth=1.6, linestyle="--")
            ax.set_ylabel(f"{l}[{i}]")
            ax.grid(True, alpha=0.3)
        axes[-1].set_xlabel("time")
        axes[0].set_title(title)
        axes[0].legend(loc="best")
        plt.tight_layout()
        #plt.show()

    def _eig_stats(self, A: np.ndarray) -> dict:
        vals = np.linalg.eigvals(A)
        absvals = np.abs(vals)
        rho = float(absvals.max()) if absvals.size else float("nan")
        dist_uc = float((1.0 - absvals).min()) if absvals.size else float("nan")  # positive is good
        out = {
            "spectral_radius": rho,
            "is_stable_disc": bool(rho < 1.0),
            "min_margin_to_unit_circle": dist_uc,
            "eigvals_real": np.real(vals).tolist(),
            "eigvals_imag": np.imag(vals).tolist(),
            "fro_norm": float(np.linalg.norm(A, "fro")),
            "two_norm": float(np.linalg.norm(A, 2)),
            "shape": list(A.shape),
            "rank": int(np.linalg.matrix_rank(A)),
        }
        try:
            out["cond_2"] = float(np.linalg.cond(A))
        except Exception:
            out["cond_2"] = None
        return out

    def _stream_stats(self, arr: np.ndarray, name: str) -> dict:
        """Accepts (T,n) or None. Returns scalar KPIs per signal and aggregated."""
        if arr is None:
            return {"present": False}
        arr = np.asarray(arr)
        if arr.ndim == 1:
            arr = arr.reshape(-1, 1)
        T, n = arr.shape
        finite = np.isfinite(arr)
        nan_count = int(np.isnan(arr).sum())
        inf_count = int(np.isinf(arr).sum())
        any_bad = bool((~finite).any())
        rms = float(np.sqrt(np.mean(arr[finite]**2))) if finite.any() else float("nan")
        mean = float(np.mean(arr[finite])) if finite.any() else float("nan")
        std = float(np.std(arr[finite])) if finite.any() else float("nan")
        max_abs = float(np.max(np.abs(arr[finite]))) if finite.any() else float("nan")
        l2_energy = float(np.sqrt(np.sum(arr[finite]**2))) if finite.any() else float("nan")
        return {
            "present": True, "name": name, "T": int(T), "n_signals": int(n),
            "nan_count": nan_count, "inf_count": inf_count, "any_nonfinite": any_bad,
            "rms": rms, "mean": mean, "std": std, "max_abs": max_abs, "l2_energy": l2_energy,
        }

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

    def _traj_error_stats(
        self,
        A, B,
        *,
        name: str,
        burn_in: int = 0,
        normalize: str = None,   # {None, "std", "range"}
    ) -> dict:
        """
        Compare two time-series arrays (T,n): stats for A vs B.

        normalize:
        None    => plain errors
        "std"   => NRMSE = RMSE / std(B) per channel (fallback to global std if zero)
        "range" => NRMSE = RMSE / (max(B)-min(B)) per channel (fallback if degenerate)

        Returns overall scalars plus per-channel arrays.
        """
        import numpy as np
        if A is None or B is None:
            return {"present": False, "name": name}

        A = np.asarray(A); B = np.asarray(B)
        if A.ndim == 1: A = A.reshape(-1, 1)
        if B.ndim == 1: B = B.reshape(-1, 1)

        T = min(A.shape[0], B.shape[0])
        A = A[:T]; B = B[:T]

        if burn_in > 0:
            A = A[burn_in:]
            B = B[burn_in:]

        finite = np.isfinite(A) & np.isfinite(B)
        if not finite.any():
            return {"present": True, "name": name, "T": int(A.shape[0]), "n": int(A.shape[1]),
                    "any_nonfinite": True}

        # mask out non-finite entries columnwise
        E_list, rmse_cols, mae_cols, maxabs_cols, corr_cols, nrmse_cols = [], [], [], [], [], []
        l2_energy_cols = []
        ncols = A.shape[1]
        for j in range(ncols):
            f = finite[:, j]
            a = A[f, j]; b = B[f, j]
            e = a - b
            E_list.append(e)
            if e.size == 0:
                rmse_cols.append(np.nan); mae_cols.append(np.nan)
                maxabs_cols.append(np.nan); l2_energy_cols.append(np.nan)
                corr_cols.append(np.nan); nrmse_cols.append(np.nan)
                continue
            rmse = float(np.sqrt(np.mean(e**2)))
            mae  = float(np.mean(np.abs(e)))
            maxa = float(np.max(np.abs(e)))
            l2e  = float(np.sqrt(np.sum(e**2)))
            rmse_cols.append(rmse); mae_cols.append(mae)
            maxabs_cols.append(maxa); l2_energy_cols.append(l2e)
            # correlation is optional; guard degenerate variance
            if np.std(a) > 0 and np.std(b) > 0:
                corr = float(np.corrcoef(a, b)[0, 1])
            else:
                corr = np.nan
            corr_cols.append(corr)

            if normalize == "std":
                denom = np.std(b)
                nrmse = rmse / denom if denom > 0 else np.nan
            elif normalize == "range":
                rng = np.max(b) - np.min(b)
                nrmse = rmse / rng if rng > 0 else np.nan
            else:
                nrmse = np.nan
            nrmse_cols.append(float(nrmse))

        # aggregate over columns using finite values
        rmse_cols_np = np.array(rmse_cols, dtype=float)
        mae_cols_np  = np.array(mae_cols, dtype=float)
        max_cols_np  = np.array(maxabs_cols, dtype=float)
        l2_cols_np   = np.array(l2_energy_cols, dtype=float)
        nrmse_cols_np = np.array(nrmse_cols, dtype=float)
        corr_cols_np  = np.array(corr_cols, dtype=float)

        def _finite_mean(x): 
            x = x[np.isfinite(x)]
            return float(np.mean(x)) if x.size else float("nan")

        overall = {
            "present": True,
            "name": name,
            "T_used": int(A.shape[0]),
            "n_signals": int(ncols),
            "rmse_mean": _finite_mean(rmse_cols_np),
            "mae_mean":  _finite_mean(mae_cols_np),
            "maxabs_mean": _finite_mean(max_cols_np),
            "l2_energy_mean": _finite_mean(l2_cols_np),
            "corr_mean": _finite_mean(corr_cols_np),
            "nrmse_mean": _finite_mean(nrmse_cols_np),
            "rmse_cols": rmse_cols,
            "mae_cols": mae_cols,
            "maxabs_cols": maxabs_cols,
            "l2_energy_cols": l2_energy_cols,
            "corr_cols": corr_cols,
            "nrmse_cols": nrmse_cols,
            "normalize": normalize,
            "burn_in": int(burn_in),
        }
        # a cleaner single scalar to optimize on if you need just one:
        overall["rmse_overall"] = overall["rmse_mean"]
        overall["nrmse_overall"] = overall["nrmse_mean"]
        return overall

    # ------------------------ public: MBD vs DDD (same method) ------------------------

    def compare_mbd_vs_ddd(self, *, path_name: str, method: str = "lmi", plot: bool = True, re_evaluate: bool = False,
                           burn_in: int = 0, normalize: str = None, error_weights: dict = None) -> dict:
        """
        Compare MBD vs DDD for the SAME method ('lmi' or 'baseline').

        path_name must end with '_MBD' or '_DDD'.
        Looks for:
          <out_root>/<method>/<base>_MBD/___results_run.json
          <out_root>/<method>/<base>_DDD/___results_run.json
          and their ___closed_loop_run.npz
        """
        method = method.strip().lower()
        assert method in ("lmi", "base"), "method must be 'lmi' or 'baseline'"

        suffix = path_name.lstrip("/\\")
        if not (suffix.endswith("_MBD") or suffix.endswith("_DDD")):
            raise ValueError("path_name must end with '_MBD' or '_DDD'.")

        base = suffix.rsplit("_", 1)[0]
        run_dir = self.out_root / method
        j_mbd = run_dir / f"{base}_MBD___results_run.json"
        j_ddd = run_dir / f"{base}_DDD___results_run.json"
        z_mbd = run_dir / f"{base}_MBD___closed_loop_run.npz"
        z_ddd = run_dir / f"{base}_DDD___closed_loop_run.npz"        
        c_mbd = run_dir / f"{base}_MBD___closed_loop_composite.npz"
        c_ddd = run_dir / f"{base}_DDD___closed_loop_composite.npz"

        if not j_mbd.exists() or not j_ddd.exists():
            raise FileNotFoundError(f"Missing JSON(s): {j_mbd} or {j_ddd}")

        JM = self._load_json(j_mbd)
        JD = self._load_json(j_ddd)

        # LMI payloads might store controller under either key; baseline uses "controller"
        def _ctrl_from_any(J):
            if "controller" in J:
                return self._controller_from_dict(J["controller"])
            if "recovered_controller" in J:
                return self._controller_from_dict(J["recovered_controller"])
            raise KeyError("No controller found in JSON payload.")

        def _plnt_from_any(J):
            if "plant" in J:
                return self._plant_from_dict(J["plant"])
            raise KeyError("No plant found in JSON payload.")

        def _plnt_cl_from_any(J):
            if "composite_closed_loop" in J:
                return self._plant_cl_from_dict(J["composite_closed_loop"])
            raise KeyError("No composite_closed_loop found in JSON payload.")

        ctrlM = _ctrl_from_any(JM)
        plntM = _plnt_from_any(JM)
        plnt_clM = _plnt_cl_from_any(JM)
        ctrlD = _ctrl_from_any(JD)
        plntD = _plnt_from_any(JD)
        plnt_clD = _plnt_cl_from_any(JD)

        # Objective values if present
        objM = JM.get("meta", {}).get("objective", JM.get("optimized_cost"))
        objD = JD.get("meta", {}).get("objective", JD.get("optimized_cost"))
        obj_gap = None
        obj_winner = None
        try:
            obj_gap = float(objD) - float(objM) if (objM is not None and objD is not None) else None
            if obj_gap is not None:
                obj_winner = "DDD" if float(objD) < float(objM) else ("MBD" if float(objM) < float(objD) else "tie")
        except Exception:
            pass

        # deltas_ctrl
        deltas_ctrl = {
            "Ac": self._fro_stats(ctrlD.Ac, ctrlM.Ac),
            "Bc": self._fro_stats(ctrlD.Bc, ctrlM.Bc),
            "Cc": self._fro_stats(ctrlD.Cc, ctrlM.Cc),
            "Dc": self._fro_stats(ctrlD.Dc, ctrlM.Dc),
        }

        print("\n=== Controller matrix deltas_ctrl (DDD minus MBD) ===")
        for k, st in deltas_ctrl.items():
            print(f"{k}: shape={st['shape']}, ‖Δ‖_F={st['fro_norm']:.3e}, "
                  f"max|Δ|={st['max_abs']:.3e}, mean|Δ|={st['mean_abs']:.3e}")

        # deltas_plnt
        deltas_plnt = {
            "A": self._fro_stats(plntD.A, plntM.A),
            "Bu": self._fro_stats(plntD.Bu, plntM.Bu),
            #"Bw": self._fro_stats(plntD.Bw, plntM.Bw),
            "Cy": self._fro_stats(plntD.Cy, plntM.Cy),
            #"Dyw": self._fro_stats(plntD.Dyw, plntM.Dyw),
            "Cz": self._fro_stats(plntD.Cz, plntM.Cz),
            "Dzu": self._fro_stats(plntD.Dzu, plntM.Dzu),
            #"Dzw": self._fro_stats(plntD.Dzw, plntM.Dzw),
        }

        deltas_composite = {
            "Acl": self._fro_stats(plnt_clD.Acl, plnt_clM.Acl),
            #"Bcl": self._fro_stats(plnt_clD.Bcl, plnt_clM.Bcl),
            "Ccl": self._fro_stats(plnt_clD.Ccl, plnt_clM.Ccl),
            #"Dcl": self._fro_stats(plnt_clD.Dcl, plnt_clM.Dcl),
        }

        # Spectral/stability stats on Acl
        spec_M = self._eig_stats(plnt_clM.Acl)
        spec_D = self._eig_stats(plnt_clD.Acl)

        # Shapes/ranks snapshot
        def _shape_rank(M):
            return {"shape": list(M.shape), "rank": int(np.linalg.matrix_rank(M))}
        shapes_ranks = {
            "ctrl_MBD": {k: _shape_rank(getattr(ctrlM, k)) for k in ("Ac","Bc","Cc","Dc")},
            "ctrl_DDD": {k: _shape_rank(getattr(ctrlD, k)) for k in ("Ac","Bc","Cc","Dc")},
            "plant_MBD": {k: _shape_rank(getattr(plntM, k)) for k in ("A","Bu","Bw","Cy","Dyw","Cz","Dzu","Dzw")},
            "plant_DDD": {k: _shape_rank(getattr(plntD, k)) for k in ("A","Bu","Bw","Cy","Dyw","Cz","Dzu","Dzw")},
            "comp_MBD": {k: _shape_rank(getattr(plnt_clM, k)) for k in ("Acl","Bcl","Ccl","Dcl")},
            "comp_DDD": {k: _shape_rank(getattr(plnt_clD, k)) for k in ("Acl","Bcl","Ccl","Dcl")},
        }

        # Time-series KPIs (if NPZ present)
        kpis_M, kpis_D = {}, {}
        if z_mbd.exists():
            dataM = np.load(z_mbd, allow_pickle=True)
            tM, XM, YM, ZM, UM, XcM = self._npz_extract_states(dataM)
            kpis_M = {
                "x":  self._stream_stats(XM, "x"),
                "xc": self._stream_stats(XcM, "xc"),
                "u":  self._stream_stats(UM, "u"),
                "y":  self._stream_stats(YM, "y"),
                "z":  self._stream_stats(ZM, "z"),
            }
        if z_ddd.exists():
            dataD = np.load(z_ddd, allow_pickle=True)
            tD, XD, YD, ZD, UD, XcD = self._npz_extract_states(dataD)
            kpis_D = {
                "x":  self._stream_stats(XD, "x"),
                "xc": self._stream_stats(XcD, "xc"),
                "u":  self._stream_stats(UD, "u"),
                "y":  self._stream_stats(YD, "y"),
                "z":  self._stream_stats(ZD, "z"),
            }

        # Optional composite time-series KPIs
        comp_kpis_M, comp_kpis_D = {}, {}
        if c_mbd.exists():
            cM = np.load(c_mbd, allow_pickle=True)
            t, X, _, Z, *_ = self._npz_extract_states(cM)
            comp_kpis_M = {"x": self._stream_stats(X, "x"), "z": self._stream_stats(Z, "z")}
        if c_ddd.exists():
            cD = np.load(c_ddd, allow_pickle=True)
            t, X, _, Z, *_ = self._npz_extract_states(cD)
            comp_kpis_D = {"x": self._stream_stats(X, "x"), "z": self._stream_stats(Z, "z")}


        print("\n=== Plant matrix deltas_plnt (DDD minus MBD) ===")
        for k, st in deltas_plnt.items():
            print(f"{k}: shape={st['shape']}, ‖Δ‖_F={st['fro_norm']:.3e}, "
                  f"max|Δ|={st['max_abs']:.3e}, mean|Δ|={st['mean_abs']:.3e}")


        print("\n=== Objective values (as saved by the pipeline) ===")
        print(f"{method.upper()} MBD objective: {objM}")
        print(f"{method.upper()} DDD objective: {objD}")


        traj_errors = {}
        aggregate = {"weighted_rmse": None, "weights": None}

        if z_mbd.exists() and z_ddd.exists():
            # Pairwise errors: MBD vs DDD
            err_x  = self._traj_error_stats(XM, XD, name="x",  burn_in=burn_in, normalize=normalize)
            err_xc = self._traj_error_stats(XcM, XcD, name="xc", burn_in=burn_in, normalize=normalize)
            err_u  = self._traj_error_stats(UM, UD, name="u",  burn_in=burn_in, normalize=normalize)
            err_y  = self._traj_error_stats(YM, YD, name="y",  burn_in=burn_in, normalize=normalize)
            err_z  = self._traj_error_stats(ZM, ZD, name="z",  burn_in=burn_in, normalize=normalize)
            traj_errors = {"x": err_x, "xc": err_xc, "u": err_u, "y": err_y, "z": err_z}

            # Aggregate scalar for tuning: default to y,z only unless you say otherwise
            if error_weights is None:
                error_weights = {"y": 0.5, "z": 0.5}
            num = 0.0; den = 0.0
            for k, w in error_weights.items():
                e = traj_errors.get(k, {})
                rmse = e.get("nrmse_overall") if normalize else e.get("rmse_overall")
                if rmse is not None and np.isfinite(rmse):
                    num += float(w) * float(rmse)
                    den += float(w)
            aggregate["weighted_rmse"] = float(num / den) if den > 0 else None
            aggregate["weights"] = {k: float(v) for k, v in error_weights.items()}

        # Optional: composite closed-loop error if you saved composite NPZs
        comp_errors = {}
        if c_mbd.exists() and c_ddd.exists():
            cM = np.load(c_mbd, allow_pickle=True); cD = np.load(c_ddd, allow_pickle=True)
            _, XMc, _, ZMc, *_ = self._npz_extract_states(cM)
            _, XDc, _, ZDc, *_ = self._npz_extract_states(cD)
            comp_errors = {
                "x": self._traj_error_stats(XMc, XDc, name="x_comp", burn_in=burn_in, normalize=normalize),
                "z": self._traj_error_stats(ZMc, ZDc, name="z_comp", burn_in=burn_in, normalize=normalize),
            }


        api = MatricesAPI()
        p_MDB = Plant(A=plntM.A, Bu=plntM.Bu, Bw=plntM.Bw, Cy=plntM.Cy, Dyw=plntM.Dyw, Cz=plntM.Cz, Dzu=plntM.Dzu, Dzw=plntM.Dzw)
        c_MDB = Controller(Ac=ctrlM.Ac, Bc=ctrlM.Bc, Cc=ctrlM.Cc, Dc=ctrlM.Dc)
        p_DDD = Plant(A=plntD.A, Bu=plntD.Bu, Bw=plntD.Bw, Cy=plntD.Cy, Dyw=plntD.Dyw, Cz=plntD.Cz, Dzu=plntD.Dzu, Dzw=plntD.Dzw)
        c_DDD = Controller(Ac=ctrlD.Ac, Bc=ctrlD.Bc, Cc=ctrlD.Cc, Dc=ctrlD.Dc)
        if plot:
            print("\n=== Matrices MBD ===")
            api.print_plant(plant=p_MDB)
            api.print_controller(ctrl=c_MDB)
            api.print_plant_cl(plant_cl=plnt_clM)
            print("\n=== Matrices DDD ===")
            api.print_plant(plant=p_DDD)
            api.print_controller(ctrl=c_DDD)
            api.print_plant_cl(plant_cl=plnt_clD)


        # Plots
        if z_mbd.exists() and z_ddd.exists() and plot:
            dataM = np.load(z_mbd)
            dataD = np.load(z_ddd)
            t, XM, YM, ZM, UM, XcM = self._npz_extract_states(dataM)
            _, XD, YD, ZD, UD, XcD = self._npz_extract_states(dataD)
            T = min(XM.shape[0], XD.shape[0])

            title = f"{method.upper()} closed-loop: states (MBD vs DDD)"
            self._plot_overlay_states(t if len(t) == T else np.arange(T), XM, XD, "x", title)            
            title = f"{method.upper()} closed-loop: cntrl states (MBD vs DDD)"
            self._plot_overlay_states(t if len(t) == T else np.arange(T), XcM, XcD, "x", title)
            title = f"{method.upper()} closed-loop: inputs (MBD vs DDD)"
            self._plot_overlay_states(t if len(t) == T else np.arange(T), UM, UD, "u", title)
            title = f"{method.upper()} closed-loop: outputs (MBD vs DDD)"
            self._plot_overlay_states(t if len(t) == T else np.arange(T), YM, YD, "y", title)            
            title = f"{method.upper()} closed-loop: perf. outputs (MBD vs DDD)"
            self._plot_overlay_states(t if len(t) == T else np.arange(T), ZM, ZD, "z", title)
            plt.show()

            if re_evaluate:
                # Re-simulate both
                cl = Closed_Loop()
                print("simulating closed loop of MDB...")
                sim_m = cl.simulate_closed_loop(plant=p_MDB, ctrl=c_MDB)
                print("simulating closed loop of DDD...")
                sim_d = cl.simulate_closed_loop(plant=p_DDD, ctrl=c_DDD)

                print("plotting MDB...")
                cl.plot_timeseries(sim_m)
                print("plotting DDD...")
                cl.plot_timeseries(sim_d)
        else:
            print("\n[warn] Missing NPZ for one/both runs; skipping plots.")

        if c_mbd.exists() and c_ddd.exists() and plot:
            dataM = np.load(c_mbd)
            dataD = np.load(c_ddd)
            t, XM, _, ZM, *_ = self._npz_extract_states(dataM)
            _, XD, _, ZD, *_ = self._npz_extract_states(dataD)
            T = min(XM.shape[0], XD.shape[0])

            title = f"{method.upper()} composite closed-loop: states (MBD vs DDD)"
            self._plot_overlay_states(t if len(t) == T else np.arange(T), XM, XD, "x", title)            
            title = f"{method.upper()} composite closed-loop: perf. outputs (MBD vs DDD)"
            self._plot_overlay_states(t if len(t) == T else np.arange(T), ZM, ZD, "z", title)
            plt.show()

            if re_evaluate:
                # Re-simulate both
                cl = Closed_Loop()
                print("simulating closed loop of MDB...")
                sim_m = cl.simulate_composite(Pcl=plnt_clM)
                print("simulating closed loop of DDD...")
                sim_d = cl.simulate_closed_loop(Pcl=plnt_clD)

                print("plotting composite MDB...")
                cl.plot_timeseries(sim_m)
                print("plotting composite DDD...")
                cl.plot_timeseries(sim_d)
        else:
            print("\n[warn] Missing NPZ for one/both runs; skipping plots.")

        # Assemble report
        report = {
            "paths": {
                "mbd_json": str(j_mbd), "ddd_json": str(j_ddd),
                "mbd_npz": str(z_mbd),  "ddd_npz": str(z_ddd),
                "mbd_comp_npz": str(c_mbd), "ddd_comp_npz": str(c_ddd),
            },
            "meta": {
                "method": method,
                "mbd_meta": JM.get("meta", {}),
                "ddd_meta": JD.get("meta", {}),
            },
            "objectives": {
                "MBD": objM, "DDD": objD,
                "gap_DDD_minus_MBD": obj_gap, "winner_lower_is_better": obj_winner,
            },
            "controller_deltas": deltas_ctrl,
            "plant_deltas": deltas_plnt,
            "composite_deltas": deltas_composite,
            "shapes_ranks": shapes_ranks,
            "stability": {
                "MBD": spec_M,
                "DDD": spec_D,
            },
            "timeseries_kpis": {
                "MBD": kpis_M,
                "DDD": kpis_D,
                "composite_MBD": comp_kpis_M,
                "composite_DDD": comp_kpis_D,
            },
            "Matrices": {
                "Controllers": {
                    "MBD": self.controller_to_dict(c_MDB), 
                    "DDD": self.controller_to_dict(c_DDD),
                },
                "Composite Plants": {
                    "MBD": self.plant_cl_to_dict(plnt_clM), 
                    "DDD": self.plant_cl_to_dict(plnt_clD),
                },
                "Plants": {
                    "MBD": self.plant_to_dict(p_MDB), 
                    "DDD": self.plant_to_dict(p_DDD),
                },
            },
            "trajectory_errors": {
                "signals": traj_errors,
                "composite": comp_errors,
                "aggregate": aggregate,
            },
        }
        # Save side-by-side comparison JSON next to the MBD run
        out_comp = (self.out_root / method / f"{base}___comparison_MBD_vs_DDD.json")
        with open(out_comp, "w", encoding="utf-8") as f:
            json.dump(report, f, indent=2)
        print(f"[compare] saved summary: {out_comp}")
        return report

    # ------------------------ public: baseline vs LMI (your provided logic) ------------------------

    def compare_baseline_vs_lmi(self, *, path_name: str, plot: bool = True) -> dict:
        """
        Looks for:
          baseline{path_name}/___results_run.json
          lmi{path_name}/___results_run.json

        Rebuilds plant/controller/Sigma for each, re-simulates with Closed_Loop,
        re-plots (optional), and prints a tidy metric table. Also dumps a comparison JSON.
        """
        base_json = self.out_root.as_posix() + "/baseline" + path_name + "___results_run.json"
        lmi_json  = self.out_root.as_posix() + "/lmi" + path_name + "___results_run.json"

        if not Path(base_json).exists():
            raise FileNotFoundError(f"Baseline JSON not found at: {base_json}")
        if not Path(lmi_json).exists():
            raise FileNotFoundError(f"LMI JSON not found at: {lmi_json}")

        print(f"[compare] loading baseline: {base_json}")
        db = self._load_json(Path(base_json))
        # Baseline file format per your earlier code
        Sigma_b = np.array(db.get("Sigma_nom") or db.get("disturbance", {}).get("Sigma_nom"), dtype=float)
        plant_b = self._plant_from_dict(db["plant"])
        ctrl_b  = self._controller_from_dict(db["controller"])
        meta_b  = {
            "baseline_cost": db.get("baseline_cost"),
            "optimized_cost": db.get("optimized_cost"),
            "optimizer_status": db.get("optimizer_status"),
            "spectral_radius_Acl": db.get("spectral_radius_Acl"),
        }

        print(f"[compare] loading lmi: {lmi_json}")
        dl = self._load_json(Path(lmi_json))
        Sigma_l = np.array(dl.get("disturbance", {}).get("Sigma_nom") or dl.get("Sigma_nom"), dtype=float)
        plant_l = self._plant_from_dict(dl["plant"])
        # LMI JSON sometimes stores "controller", sometimes "recovered_controller"
        ctrl_key = "controller" if "controller" in dl else "recovered_controller"
        ctrl_l  = self._controller_from_dict(dl[ctrl_key])
        meta_l  = {
            "status": dl.get("meta", {}).get("status"),
            "objective": dl.get("meta", {}).get("objective"),
            "gamma": dl.get("meta", {}).get("gamma"),
            "lambda_opt": dl.get("meta", {}).get("lambda_opt"),
            "spectral_radius_Acl": dl.get("meta", {}).get("spectral_radius_Acl"),
            "model": dl.get("meta", {}).get("model"),
        }

        # Re-simulate both
        cl = Closed_Loop()
        print("[compare] simulating baseline closed loop...")
        sim_b = cl.simulate_closed_loop(plant_b, ctrl_b, Sigma_b)
        print("[compare] simulating lmi closed loop...")
        sim_l = cl.simulate_closed_loop(plant_l, ctrl_l, Sigma_l)

        if plot:
            print("[compare] plotting baseline...")
            cl.plot_timeseries(sim_b)
            print("[compare] plotting lmi...")
            cl.plot_timeseries(sim_l)

        # Metrics
        mb = self._compute_metrics(sim_b)
        ml = self._compute_metrics(sim_l)

        report = {
            "paths": {"baseline_json": base_json, "lmi_json": lmi_json},
            "baseline_meta": meta_b,
            "lmi_meta": meta_l,
            "metrics": {"baseline": mb, "lmi": ml},
            "deltas_ctrl": {k: (ml.get(k, np.nan) - mb.get(k, np.nan)) for k in set(mb) | set(ml)},
        }

        comp_out = self.out_root.as_posix() + path_name + "___comparison_baseline_vs_lmi.json"
        with open(comp_out, "w", encoding="utf-8") as f:
            json.dump(report, f, indent=2)
        print(f"[compare] saved summary: {comp_out}")

        # Compact console table
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

        print("\n=== Meta sanity ===")
        print(f"baseline.rho(Acl) ≈ {meta_b.get('spectral_radius_Acl')}")
        print(f"lmi     .rho(Acl) ≈ {meta_l.get('spectral_radius_Acl')}")
        print(f"lmi.status = {meta_l.get('status')}, objective ≈ {meta_l.get('objective')}, model = {meta_l.get('model')}, gamma = {meta_l.get('gamma')}")
        return report

