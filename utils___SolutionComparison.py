import json, sys
from pathlib import Path
import numpy as np
import matplotlib.pyplot as plt

from utils___systems import Plant, Controller
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
    def _npz_extract_states(npz: dict):
        # time
        t = None
        for tk in ("t", "time", "times"):
            if tk in npz:
                t = npz[tk]
                break
        # states
        X = None
        for xk in ("x", "X", "states", "state", "traj_X", "traj"):
            if xk in npz:
                X = npz[xk]
                break
        if X is None:
            raise KeyError("Could not find a state matrix in NPZ (x/X/states/state/traj_X/traj).")
        X = np.array(X)
        # We prefer (T, nx). If it's (nx, T) and nx<T, leave it; either way we’ll plot row-wise time.
        if t is None:
            t = np.arange(X.shape[0])
        return t, X

    @staticmethod
    def _plot_overlay_states(t, XM, XD, title: str):
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
            ax.set_ylabel(f"x[{i}]")
            ax.grid(True, alpha=0.3)
        axes[-1].set_xlabel("time")
        axes[0].set_title(title)
        axes[0].legend(loc="best")
        plt.tight_layout()
        plt.show()

    # ------------------------ public: MBD vs DDD (same method) ------------------------

    def compare_mbd_vs_ddd(self, *, path_name: str, method: str = "lmi", plot: bool = True) -> dict:
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

        ctrlM = _ctrl_from_any(JM)
        ctrlD = _ctrl_from_any(JD)

        # Objective values if present
        objM = JM.get("meta", {}).get("objective", JM.get("optimized_cost"))
        objD = JD.get("meta", {}).get("objective", JD.get("optimized_cost"))

        # Deltas
        deltas = {
            "Ac": self._fro_stats(ctrlD.Ac, ctrlM.Ac),
            "Bc": self._fro_stats(ctrlD.Bc, ctrlM.Bc),
            "Cc": self._fro_stats(ctrlD.Cc, ctrlM.Cc),
            "Dc": self._fro_stats(ctrlD.Dc, ctrlM.Dc),
        }

        print("\n=== Controller matrix deltas (DDD minus MBD) ===")
        for k, st in deltas.items():
            print(f"{k}: shape={st['shape']}, ‖Δ‖_F={st['fro_norm']:.3e}, "
                  f"max|Δ|={st['max_abs']:.3e}, mean|Δ|={st['mean_abs']:.3e}")

        print("\n=== Objective values (as saved by the pipeline) ===")
        print(f"{method.upper()} MBD objective: {objM}")
        print(f"{method.upper()} DDD objective: {objD}")

        api = MatricesAPI()
        c_MDB = Controller(Ac=ctrlM.Ac, Bc=ctrlM.Bc, Cc=ctrlM.Cc, Dc=ctrlM.Dc)
        c_DDD = Controller(Ac=ctrlD.Ac, Bc=ctrlD.Bc, Cc=ctrlD.Cc, Dc=ctrlD.Dc)
        print("\n=== Matrices MBD ===")
        api.print_controller(ctrl=c_MDB)
        print("\n=== Matrices DDD ===")
        api.print_controller(ctrl=c_DDD)
        

        # Plots
        if z_mbd.exists() and z_ddd.exists() and plot:
            dataM = np.load(z_mbd)
            dataD = np.load(z_ddd)
            tM, XM = self._npz_extract_states(dataM)
            tD, XD = self._npz_extract_states(dataD)
            T = min(XM.shape[0], XD.shape[0])
            title = f"{method.upper()} closed-loop: states (MBD vs DDD)"
            self._plot_overlay_states(tM if len(tM) == T else np.arange(T), XM, XD, title)
        else:
            print("\n[warn] Missing NPZ for one/both runs; skipping plots.")

        report = {
            "paths": {
                "mbd_json": str(j_mbd),
                "ddd_json": str(j_ddd),
                "mbd_npz": str(z_mbd),
                "ddd_npz": str(z_ddd),
            },
            "objectives": {"MBD": objM, "DDD": objD},
            "matrix_deltas": deltas,
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
            "deltas": {k: (ml.get(k, np.nan) - mb.get(k, np.nan)) for k in set(mb) | set(ml)},
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
