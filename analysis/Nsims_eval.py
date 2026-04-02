# main.py
import sys
import numpy as np, pandas as pd
import matplotlib.pyplot as plt
import matplotlib.patches as mpatches

from pathlib import Path
from matplotlib.lines import Line2D
from typing import Callable, Dict, Any




# ------------------------- Evaluation --------------------------------------------

def MutipleRunsEvaluation(
        p, 
        run_fn: Callable[..., Dict[str, Any]],
        gamma: float = 0.5, 
        COST: bool = True, 
        N: int = None,
        ):
    
    N = 20 if N is None else N
    model = p.get("model", "independent")
    save = bool(p.get("save", False))
    plot = bool(p.get("plot", False))
    use = True

    c_MBD, c_DDD = [], []
    l_MBD, l_DDD = [], []
    r_MBD, r_DDD = [], []
    t_MBD, t_DDD = [], []
    a_MBD, a_DDD = [], []
    s_MBD, s_DDD = [], []
    v_MBD, v_DDD = [], []
    o_MBD, o_DDD = [], []
    p_MBD, p_DDD = [], []
    n_MBD, n_DDD = [], []

    out = Path(p.get("directories", {}).get("artifacts", "./out/artifacts/")).with_suffix("")
    out = out / "MutipleRunsEvaluation"
    out.mkdir(parents=True, exist_ok=True)
    path = out / p.get("directories", {}).get("runID", "temp")
    path.mkdir(parents=True, exist_ok=True)
    mbd_file = path / f"_{model}_MBD_runs.csv"
    ddd_file = path / f"_{model}_DDD_runs.csv"

    NOT_FOUND = not (mbd_file.is_file() and ddd_file.is_file())


    if bool(p.get("re_evaluate", 0)) or NOT_FOUND:
        k = 0

        for i in range(N):
            print("\n\n\n\n"
                "==============================\n"
                f"----- RUN {i+1}/{N} -----\n"
                "==============================\n"
                "\n\n\n\n")
            infos_mbd, *_ = run_fn(FROM_DATA=False, gamma=gamma, comp=False, ALL=False, COST=COST)
            c_mbd = infos_mbd["J"]
            l_mbd = infos_mbd["lamda"]
            r_mbd = infos_mbd["rho"]
            t_mbd = infos_mbd["time"]
            a_mbd = infos_mbd["attempts"]
            s_mbd = infos_mbd["stress"]
            v_mbd = infos_mbd["ratio_violation"]
            p_mbd = 1 if infos_mbd["solver"] in ["MOSEK", "mosek"] else 0
            o_mbd = infos_mbd["obj"]
            n_mbd = infos_mbd["snr"]

            c_MBD.append(c_mbd)
            l_MBD.append(l_mbd)
            r_MBD.append(r_mbd)
            t_MBD.append(t_mbd)
            a_MBD.append(a_mbd)
            s_MBD.append(s_mbd)
            v_MBD.append(v_mbd)
            p_MBD.append(p_mbd)
            o_MBD.append(o_mbd)
            n_MBD.append(n_mbd)

            infos_ddd, *_ = run_fn(FROM_DATA=True, gamma=gamma, comp=False, ALL=False, COST=COST)
            c_ddd = infos_ddd["J"]
            l_ddd = infos_ddd["lamda"]
            r_ddd = infos_ddd["rho"]
            t_ddd = infos_ddd["time"]
            a_ddd = infos_ddd["attempts"]
            s_ddd = infos_ddd["stress"]
            v_ddd = infos_ddd["ratio_violation"]
            p_ddd = 1 if infos_ddd["solver"] in ["MOSEK", "mosek"] else 0
            o_ddd = infos_ddd["obj"]
            n_ddd = infos_ddd["snr"]

            c_DDD.append(c_ddd)
            l_DDD.append(l_ddd)
            r_DDD.append(r_ddd)
            t_DDD.append(t_ddd)
            a_DDD.append(a_ddd)
            s_DDD.append(s_ddd)
            v_DDD.append(v_ddd)
            p_DDD.append(p_ddd)
            o_DDD.append(o_ddd)
            n_DDD.append(n_ddd)

            if r_ddd > 1.0:
                k += 1

            """# ---------- NEW: check last K rho_DDD ----------
            if len(r_DDD) >= K_RECENT:
                recent_rho = np.array(r_DDD[-K_RECENT:], dtype=float)
                if np.all(recent_rho >= 1.0):
                    print(
                        f"[WARN] Last {K_RECENT} DDD rho values are >= 1.0; "
                        f"recent_rho = {recent_rho}. Stopping further runs."
                    )
                    unstable_hit = True
                    break
            
            try:
                pass
            except Exception as e: 
                print(f"Error occurred in MBD run {i+1}: {e}")"""

        unstable_hit = k > int(N/5)
        print(f"Completed {len(c_MBD)} MBD runs, {len(c_DDD)} DDD runs. unstable_hit={unstable_hit}")

        print(f"\n\n===== COST STATISTICS OVER {N} RUNS =====")
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
        v_MBD = np.array(v_MBD, dtype=float)
        v_DDD = np.array(v_DDD, dtype=float)
        p_MBD = np.array(p_MBD, dtype=float)
        p_DDD = np.array(p_DDD, dtype=float)        
        o_MBD = np.array(o_MBD, dtype=float)
        o_DDD = np.array(o_DDD, dtype=float)
        n_MBD = np.array(n_MBD, dtype=float)
        n_DDD = np.array(n_DDD, dtype=float)

        # ------------------------------------------------------------------
        # SAVE TO CSV (one file for MBD, one for DDD)
        # ------------------------------------------------------------------

        # In case some runs failed on one side, truncate to common length
        n_mbd = len(c_MBD)
        n_ddd = len(c_DDD)
        n_common_mbd = min(n_mbd, len(l_MBD), len(r_MBD), len(t_MBD), len(a_MBD), len(s_MBD), len(n_MBD))
        n_common_ddd = min(n_ddd, len(l_DDD), len(r_DDD), len(t_DDD), len(a_DDD), len(s_DDD), len(n_DDD))

        # MBD table: [run, J, lambda, rho, time, attempts, stress]
        mbd_data = np.column_stack([
            np.arange(n_common_mbd),
            c_MBD[:n_common_mbd],
            l_MBD[:n_common_mbd],
            r_MBD[:n_common_mbd],
            t_MBD[:n_common_mbd],
            a_MBD[:n_common_mbd],
            s_MBD[:n_common_mbd],
            v_MBD[:n_common_mbd],
            o_MBD[:n_common_mbd],
            p_MBD[:n_common_mbd],
            n_MBD[:n_common_mbd],
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
            v_DDD[:n_common_ddd],
            o_DDD[:n_common_ddd],
            p_DDD[:n_common_ddd],
            n_DDD[:n_common_ddd],
        ])

        if save:
            np.savetxt(
                mbd_file,
                mbd_data,
                delimiter=",",
                header="run,J,lambda,rho,time,attempts,stress,ratio_violation",
                comments=""
            )

            np.savetxt(
                ddd_file,
                ddd_data,
                delimiter=",",
                header="run,J,lambda,rho,time,attempts,stress,ratio_violation",
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

        try:
            v_MBD = mbd_data[:, 7]
            v_DDD = ddd_data[:, 7]
            o_MBD = mbd_data[:, 8]
            o_DDD = ddd_data[:, 8]
            p_MBD = mbd_data[:, 9]
            p_DDD = ddd_data[:, 9]
            n_MBD = mbd_data[:, 10]
            n_DDD = ddd_data[:, 10]
        except Exception as e:
            use = False


    idx = np.where(c_DDD < 80)[0]

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

        # Align on common index
        t_max = min(len(y_MBD), len(y_DDD))
        t = np.arange(1, t_max + 1, dtype=int)[idx]

        y_mbd = y_MBD[:t_max][idx]
        y_ddd = y_DDD[:t_max][idx]


        # ----- basic stats (ignoring NaNs if present) -----
        mu_mbd = float(np.nanmean(y_mbd))
        sd_mbd = float(np.nanstd(y_mbd, ddof=1))
        mu_ddd = float(np.nanmean(y_ddd))
        sd_ddd = float(np.nanstd(y_ddd, ddof=1))

        print(f"[{metric_name}] MBD: mean={mu_mbd:.6g}, std={sd_mbd:.6g}")
        print(f"[{metric_name}] DDD: mean={mu_ddd:.6g}, std={sd_ddd:.6g}")

        mbd_label = rf"MBD  ($\mu$={mu_mbd:.3g}, $\sigma$={sd_mbd:.3g})"
        ddd_label = rf"DDD  ($\mu$={mu_ddd:.3g}, $\sigma$={sd_ddd:.3g})"

        labels = ["MBD", "DDD"]

        # =========================
        # 1) scatter + mean lines
        # =========================
        fig2, ax2 = plt.subplots(figsize=(7, 4))
        ax2.scatter(np.zeros_like(y_mbd), y_mbd, alpha=0.6, label="MBD", color="blue")
        ax2.scatter(np.ones_like(y_ddd),  y_ddd, alpha=0.6, label="DDD", color="orange")
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
        
        return metric_name, (mu_mbd, sd_mbd), (mu_ddd, sd_ddd)


    c_n, c_m, c_d = analyze_and_plot_metric(c_MBD, c_DDD, "cost",     model, path, save, N_runs=N)
    l_n, l_m, l_d = analyze_and_plot_metric(l_MBD, l_DDD, "lambda",   model, path, save, N_runs=N)
    r_n, r_m, r_d = analyze_and_plot_metric(r_MBD, r_DDD, "rho",      model, path, save, N_runs=N)
    t_n, t_m, t_d = analyze_and_plot_metric(t_MBD, t_DDD, "time",     model, path, save, N_runs=N)
    a_n, a_m, a_d = analyze_and_plot_metric(a_MBD, a_DDD, "attempts", model, path, save, N_runs=N)
    s_n, s_m, s_d = analyze_and_plot_metric(s_MBD, s_DDD, "stress",   model, path, save, N_runs=N)
    
    metrics = [
        (c_n, c_m, c_d),
        (l_n, l_m, l_d),
        (r_n, r_m, r_d),
        (t_n, t_m, t_d),
        (a_n, a_m, a_d),
        (s_n, s_m, s_d),
    ]

    if use: 
        v_n, v_m, v_d = analyze_and_plot_metric(v_MBD, v_DDD, "ratio_violation", model, path, save, N_runs=N)
        o_n, o_m, o_d = analyze_and_plot_metric(o_MBD, o_DDD, "objective", model, path, save, N_runs=N)
        p_n, p_m, p_d = analyze_and_plot_metric(p_MBD, p_DDD, "solver", model, path, save, N_runs=N)
        n_n, n_m, n_d = analyze_and_plot_metric(n_MBD, n_DDD, "snr", model, path, save, N_runs=N)

        metrics.extend([
            (v_n, v_m, v_d),
            (o_n, o_m, o_d),
            (p_n, p_m, p_d),
            (n_n, n_m, n_d),
        ])

    rows = []
    for name, (mu_mbd, sd_mbd), (mu_ddd, sd_ddd) in metrics:
        rows.append({
            "metric": name,
            "MBD": f"{mu_mbd:.6g} ± {sd_mbd:.6g}",
            "DDD": f"{mu_ddd:.6g} ± {sd_ddd:.6g}",
        })

    summary_df = pd.DataFrame(rows, columns=["metric", "MBD", "DDD"])

    csv_path = path / f"_{model}_metrics_summary_{N}_runs.csv"
    summary_df.to_csv(csv_path, index=False)
    print(f"Saved summary metrics to {csv_path}")

def select_best_N_sims(N_sims_list, means, stds, prefer_small: bool = True, rel_eps: float = 1e-3):
    """
    Pick best N_sims given mean and std arrays.

    - prefer_small=True: minimize mean (good for cost J, rho, etc.).
    - Among near-equal means (within rel_eps), choose smallest std.

    N_sims_list, means, stds: 1D arrays / lists of same length.
    """
    N_sims_arr = np.asarray(N_sims_list, dtype=float)
    means_arr = np.asarray(means, dtype=float)
    stds_arr = np.asarray(stds, dtype=float)

    if len(N_sims_arr) == 0:
        raise ValueError("select_best_N_sims: empty input.")

    if prefer_small:
        base = np.min(means_arr)
        # "Near best" in relative sense
        tol = rel_eps * max(abs(base), 1e-12)
        candidates = np.where(means_arr <= base + tol)[0]
    else:
        base = np.max(means_arr)
        tol = rel_eps * max(abs(base), 1e-12)
        candidates = np.where(means_arr >= base - tol)[0]

    # Among candidates, choose the one with smallest std
    best_local_idx = np.argmin(stds_arr[candidates])
    best_idx = candidates[best_local_idx]

    return int(N_sims_arr[best_idx]), float(means_arr[best_idx]), float(stds_arr[best_idx])



# ------------------------- N_sims sweep for FROM_DATA --------------------------------------------

def NsimSweep_FROM_DATA(
    p: dict,
    run_fn: Callable[..., Dict[str, Any]],
    gamma: float = 0.5,
    COST: bool = True,
    N_sims_values: list[int] | None = None,
    runs_per_N: int = 10,
    ):
    """
    Sweep N_sims for FROM_DATA=True, run `main` multiple times per value,
    and aggregate mean / std of J and rho.

    - N_sims_values: list of N_sims to test; if None -> [1, 6, 11, ..., 46]
    - runs_per_N: how many runs per N_sims (default 10)
    """

    # ------------------------------------------------------------
    # Setup
    # ------------------------------------------------------------
    if N_sims_values is None:
        # 10 steps: 1, 6, 11, ..., 46
        N_sims_values = list(range(1, 50, 5))

    model = p.get("model", "independent")
    save = bool(p.get("save", False))
    plot = bool(p.get("plot", False))

    out_root = Path(p.get("directories", {}).get("artifacts", "./out/artifacts/")).with_suffix("")
    out = out_root / "NsimSweep_FROM_DATA"
    out.mkdir(parents=True, exist_ok=True)

    path = out / p.get("directories", {}).get("runID", "temp")
    path.mkdir(parents=True, exist_ok=True)

    csv_file = path / f"_{model}_Nsim_sweep_FROM_DATA.csv"



    if bool(p.get("re_evaluate", 0)) or not csv_file.is_file():
        # ------------------------------------------------------------
        # Storage for aggregated statistics
        # ------------------------------------------------------------
        agg_N_sims = []
        agg_J_mean = []
        agg_J_std = []
        agg_rho_mean = []
        agg_rho_std = []
        agg_obj_mean = []
        agg_obj_std = []

        # ------------------------------------------------------------
        # Main sweep loop
        # ------------------------------------------------------------
        for idx, N_sims in enumerate(N_sims_values, start=1):
            print(
                "\n\n\n"
                "============================================\n"
                f"  N_sims sweep step {idx}/{len(N_sims_values)}  (N_sims = {N_sims})\n"
                "============================================\n"
            )

            J_vals = []
            rho_vals = []
            obj_vals = []

            for run_idx in range(runs_per_N):
                print(
                    "\n--------------------------------------------\n"
                    f"  Inner run {run_idx + 1}/{runs_per_N}  (N_sims = {N_sims})\n"
                    "--------------------------------------------\n"
                )

                # Only FROM_DATA=True
                infos_ddd, *_ = run_fn(
                    FROM_DATA=True,
                    gamma=gamma,
                    comp=False,
                    ALL=False,
                    COST=COST,
                    N_sims=N_sims,
                )

                J_vals.append(float(infos_ddd["J"]))
                rho_vals.append(float(infos_ddd["rho"]))
                obj_vals.append(float(infos_ddd["obj"]))

            # Convert to numpy and compute stats
            J_vals = np.asarray(J_vals, dtype=float)
            rho_vals = np.asarray(rho_vals, dtype=float)
            obj_vals = np.asarray(obj_vals, dtype=float)

            J_mean = float(np.mean(J_vals))
            J_std = float(np.std(J_vals, ddof=1)) if len(J_vals) > 1 else 0.0
            rho_mean = float(np.mean(rho_vals))
            rho_std = float(np.std(rho_vals, ddof=1)) if len(rho_vals) > 1 else 0.0
            obj_mean = float(np.mean(obj_vals))
            obj_std = float(np.std(obj_vals, ddof=1)) if len(obj_vals) > 1 else 0.0

            print(f"[N_sims={N_sims}] J:   mean = {J_mean:.6g}, std = {J_std:.6g}")
            print(f"[N_sims={N_sims}] rho: mean = {rho_mean:.6g}, std = {rho_std:.6g}")
            print(f"[N_sims={N_sims}] obj: mean = {obj_mean:.6g}, std = {obj_std:.6g}")

            agg_N_sims.append(N_sims)
            agg_J_mean.append(J_mean)
            agg_J_std.append(J_std)
            agg_rho_mean.append(rho_mean)
            agg_rho_std.append(rho_std)
            agg_obj_mean.append(obj_mean)
            agg_obj_std.append(obj_std)

        # ------------------------------------------------------------
        # Build DataFrame & save CSV
        # ------------------------------------------------------------
        df = pd.DataFrame(
            {
                "N_sims": agg_N_sims,
                "J_mean": agg_J_mean,
                "J_std": agg_J_std,
                "rho_mean": agg_rho_mean,
                "rho_std": agg_rho_std,
                "obj_mean": agg_obj_mean, 
                "obj_std": agg_obj_std,
            }
        )

        if save:
            df.to_csv(csv_file, index=False)
            print(f"Saved N_sims sweep stats to {csv_file}")

    else:
        # load, skip header row
        data = np.loadtxt(csv_file, delimiter=",", skiprows=1)

        # robustify: if only 1 row, loadtxt returns 1D
        data = np.atleast_2d(data)

        # columns: run,J,lambda,rho,time,attempts,stress
        agg_N_sims      = data[:, 0]
        agg_J_mean      = data[:, 1]
        agg_J_std       = data[:, 2]
        agg_rho_mean    = data[:, 3]
        agg_rho_std     = data[:, 4]
        agg_obj_mean    = data[:, 5]
        agg_obj_std     = data[:, 6]


    # ------------------------------------------------------------
    # Select best N_sims for J and rho
    # ------------------------------------------------------------
    # J: want small mean and small std
    best_N_J, best_J_mean, best_J_std = select_best_N_sims(
        agg_N_sims, agg_J_mean, agg_J_std, prefer_small=True
    )
    print(
        f"\n[Best J] N_sims = {best_N_J}, "
        f"J_mean = {best_J_mean:.6g}, J_std = {best_J_std:.6g}"
    )

    # obj: want pos and small std
    best_N_obj, best_obj_mean, best_obj_std = select_best_N_sims(
        agg_N_sims, agg_obj_mean, agg_obj_std, prefer_small=True
    )
    print(
        f"\n[Best obj] N_sims = {best_N_obj}, "
        f"obj_mean = {best_obj_mean:.6g}, obj_std = {best_obj_std:.6g}"
    )

    # rho: usually want mean rho < 1 and as small as possible
    agg_N_sims_arr = np.asarray(agg_N_sims, dtype=float)
    agg_rho_mean_arr = np.asarray(agg_rho_mean, dtype=float)
    agg_rho_std_arr = np.asarray(agg_rho_std, dtype=float)

    stable_mask = agg_rho_mean_arr < 1.0  # only consider stable ones
    if np.any(stable_mask):
        best_N_rho, best_rho_mean, best_rho_std = select_best_N_sims(
            agg_N_sims_arr[stable_mask],
            agg_rho_mean_arr[stable_mask],
            agg_rho_std_arr[stable_mask],
            prefer_small=True,
        )
        print(
            f"[Best rho (stable)] N_sims = {best_N_rho}, "
            f"rho_mean = {best_rho_mean:.6g}, rho_std = {best_rho_std:.6g}"
        )
    else:
        best_N_rho = None
        best_rho_mean = None
        best_rho_std = None
        print("[Best rho] No N_sims with mean rho < 1.0; cannot select a 'stable' best.")

    # ------------------------------------------------------------
    # Plotting: errorbars vs N_sims
    # ------------------------------------------------------------
    # 1) J vs N_sims
    fig_J, ax_J = plt.subplots(figsize=(7, 4))
    ax_J.errorbar(
        agg_N_sims,
        agg_J_mean,
        yerr=agg_J_std,
        fmt="o-",
        capsize=4,
    )
    ax_J.set_xscale("log")
    ax_J.set_xlabel("N_sims")
    ax_J.set_ylabel("J (mean ± std)")
    ax_J.set_title(f"FROM_DATA=True: J vs N_sims ({model})")
    ax_J.grid(True, alpha=0.3)

    # Mark best J
    ax_J.axvline(best_N_J, linestyle="--", alpha=0.6)
    star_J = ax_J.scatter([best_N_J], [best_J_mean], marker="*", s=120)
    handles, labels = ax_J.get_legend_handles_labels()

    best_handle_J = Line2D(
        [0], [0],
        marker="*",
        linestyle="None",
        color=star_J.get_facecolor()[0] if hasattr(star_J, "get_facecolor") else "C1",
        label=f"Best N_sims (J): {best_N_J}",
    )

    handles.append(best_handle_J)
    labels.append(f"Best N_sims (J): {best_N_J}")

    ax_J.legend(handles, labels, loc="best")
    fig_J.tight_layout()


    # 1) J vs N_sims
    fig_obj, ax_obj = plt.subplots(figsize=(7, 4))
    ax_obj.errorbar(
        agg_N_sims,
        agg_obj_mean,
        yerr=agg_obj_std,
        fmt="o-",
        capsize=4,
    )
    ax_obj.set_xscale("log")
    ax_obj.set_xlabel("N_sims")
    ax_obj.set_ylabel("obj (mean ± std)")
    ax_obj.set_title(f"FROM_DATA=True: obj vs N_sims ({model})")
    ax_obj.grid(True, alpha=0.3)

    # Mark best obj
    ax_obj.axvline(best_N_obj, linestyle="--", alpha=0.6)
    star_obj = ax_obj.scatter([best_N_obj], [best_obj_mean], marker="*", s=120)
    handles, labels = ax_obj.get_legend_handles_labels()

    best_handle_obj = Line2D(
        [0], [0],
        marker="*",
        linestyle="None",
        color=star_obj.get_facecolor()[0] if hasattr(star_obj, "get_facecolor") else "C1",
        label=f"Best N_sims (obj): {best_N_obj}",
    )

    handles.append(best_handle_obj)
    labels.append(f"Best N_sims (obj): {best_N_obj}")

    ax_obj.legend(handles, labels, loc="best")
    fig_obj.tight_layout()

    # 2) rho vs N_sims
    fig_rho, ax_rho = plt.subplots(figsize=(7, 4))
    ax_rho.errorbar(
        agg_N_sims,
        agg_rho_mean,
        yerr=agg_rho_std,
        fmt="o-",
        capsize=4,
    )
    ax_rho.set_xscale("log")
    ax_rho.set_xlabel("N_sims")
    ax_rho.set_ylabel(r"$\rho$ (mean ± std)")
    ax_rho.set_title(f"FROM_DATA=True: rho vs N_sims ({model})")
    ax_rho.grid(True, alpha=0.3)

    # Mark best rho if any stable candidate exists
    if best_N_rho is not None:
        ax_rho.axvline(best_N_rho, linestyle="--", alpha=0.6)
        star_rho = ax_rho.scatter([best_N_rho], [best_rho_mean], marker="*", s=120)
        handles, labels = ax_rho.get_legend_handles_labels()

        best_handle_rho = Line2D(
            [0], [0],
            marker="*",
            linestyle="None",
            color=star_rho.get_facecolor()[0] if hasattr(star_rho, "get_facecolor") else "C1",
            label=f"Best N_sims (ρ): {best_N_rho}",
        )

        handles.append(best_handle_rho)
        labels.append(f"Best N_sims (ρ): {best_N_rho}")

    ax_rho.legend(handles, labels, loc="best")
    fig_rho.tight_layout()


    if save:
        fig_J.savefig(path / f"{model}_J_vs_Nsims_FROM_DATA.pdf")
        fig_rho.savefig(path / f"{model}_rho_vs_Nsims_FROM_DATA.pdf")
        fig_obj.savefig(path / f"{model}_obj_vs_Nsims_FROM_DATA.pdf")

    if plot:
        plt.show()

