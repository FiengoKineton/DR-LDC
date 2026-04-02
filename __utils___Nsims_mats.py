import json, numpy as np, cvxpy as cp, casadi as ca
import matplotlib.pyplot as plt

from pathlib import Path
from typing import Dict, Any, List
from __utils___simulate import Open_Loop


# === UTILS ===================================================================================== #

def mean_dict(datasets):
    """Elementwise mean over a list of dataset dicts with identical shapes."""
    out = {}
    T = min(d["meta"]["T"] for d in datasets)
    keys = ["X","U","Y","Z","X_next"]
    for k in keys:
        arrs = []
        for d in datasets:
            A = d[k]
            arrs.append(A[..., :T-1] if k in {"X_reg","X_next"} else A[..., :T])
        out[k] = np.stack(arrs, axis=0).mean(axis=0)
    out["meta"] = {**datasets[0]["meta"], "T": T, "N": len(datasets)}
    return out

def select_representative_run(datasets, keys=("X","U","Y","Z","X_next"), weights=None):
    """
    Choose the medoid (most representative) dataset across the given keys.
    Returns a dict with aligned X,U,Y,Z,X_next from that single seed.
    """
    if not datasets:
        raise ValueError("datasets is empty")

    # 1) Align horizons
    T_min = min(d["meta"]["T"] for d in datasets)
    Teff      = T_min
    Teff_next = T_min - 1
    if Teff_next <= 0:
        raise ValueError("Need T >= 2 to align X_next")

    # 2) Stack aligned arrays
    stacks = {}
    for k in keys:
        if k == "X_next":
            stacks[k] = np.stack([d[k][..., :Teff_next] for d in datasets], axis=0)   # (N, n, T-1)
        else:
            stacks[k] = np.stack([d[k][..., :Teff]      for d in datasets], axis=0)   # (N, n, T)

    N = next(iter(stacks.values())).shape[0]
    if weights is None:
        weights = {k: 1.0 for k in keys}

    # 3) Per-key scaling so no single key dominates (Frobenius mean per seed)
    def fro_scale(S):  # S shape: (N, ...)
        return np.mean([np.linalg.norm(S[i]) for i in range(S.shape[0])]) + 1e-12

    scales = {k: fro_scale(S) for k, S in stacks.items()}

    # 4) Medoid index: minimize total weighted squared distance to others
    dists = np.zeros(N)
    for i in range(N):
        total = 0.0
        for k, S in stacks.items():
            Sk = S / scales[k]
            diff = Sk - Sk[i]              # (N, ...)
            total += float(weights[k]) * np.sum(diff**2)
        dists[i] = total
    i_star = int(np.argmin(dists))

    # 5) Build output using that single, real run (preserves dynamics)
    meta_sel = dict(datasets[i_star]["meta"])  # copy
    meta_sel.update({
        "T": T_min,
        "N": len(datasets),
        "selected_index": i_star,
        "selection": "medoid_over_"+",".join(keys),
    })
    out = {
        "X":      stacks["X"][i_star],                        # (nx, T)
        "U":      stacks["U"][i_star],                        # (nu, T)
        "Y":      stacks["Y"][i_star],                        # (ny, T)
        "Z":      stacks["Z"][i_star],                        # (nz, T)
        # pad last column so X_next matches T
        "X_next": np.hstack([stacks["X_next"][i_star], stacks["X_next"][i_star][:, -1][:, None]]),
        "meta": meta_sel,
    }
    return out

def plot_first3_and_mean(datasets, key="X", out=None, title_prefix=None,
                         show_band=True, symmetric_ylim=True):
    if not datasets:
        raise ValueError("datasets is empty.")

    # Align horizon for datasets
    T_min = min(d["meta"]["T"] for d in datasets)
    use_Tm1 = key in {"X_next", "X_reg"}
    Teff = T_min - 1 if use_Tm1 else T_min
    if Teff <= 0:
        raise ValueError("Effective horizon <= 0; check inputs.")

    # Stack datasets: (N, n, Teff) and get per-dataset row-mean (N, Teff)
    stack = np.stack([d[key][..., :Teff] for d in datasets], axis=0)
    per_ds_mean = np.nanmean(stack, axis=1)
    N = per_ds_mean.shape[0]
    global_mean = np.nanmean(per_ds_mean, axis=0)
    global_std  = np.nanstd(per_ds_mean, axis=0, ddof=1) if N > 1 else np.zeros_like(global_mean)

    # Time vector
    t = datasets[0].get("t", None)
    t = t[:Teff] if (t is not None and t.shape[-1] >= Teff) else np.arange(Teff)

    # Optional overlay from `out` using the SAME key
    overlay = None
    if out is not None and key in out:
        arr = np.asarray(out[key])
        overlay = arr if arr.ndim == 1 else np.nanmean(arr, axis=0)
        # clip/align overlay length to Teff
        overlay = overlay[:Teff] if overlay.shape[-1] >= Teff else np.pad(
            overlay, (0, Teff - overlay.shape[-1]), mode="edge"
        )

    # Y-limits from curves actually shown
    curves = [per_ds_mean[i] for i in range(min(3, N))] + [global_mean]
    if overlay is not None:
        curves.append(overlay)
    ymin = np.min([np.nanmin(c) for c in curves])
    ymax = np.max([np.nanmax(c) for c in curves])
    L = float(max(abs(ymin), abs(ymax))) or 1.0

    fig, axes = plt.subplots(2, 2, figsize=(12, 8), sharex=True)
    axes = axes.flatten()

    def plot_mean(ax, m, title):
        ax.plot(t, m, label="dataset avg")
        if show_band:
            ax.fill_between(t, global_mean - global_std, global_mean + global_std,
                            alpha=0.2, linewidth=0, label="±1σ (global)")
        if overlay is not None:
            ax.plot(t, overlay, linewidth=2.0, label=f"{key} (precomputed)")
        ax.set_title(title if title else "")
        ax.set_xlabel("time"); ax.set_ylabel(f"mean({key})")
        ax.set_ylim((-L, L) if symmetric_ylim else (ymin, ymax))
        ax.grid(True, alpha=0.3); ax.legend(loc="best")

    for i in range(min(3, N)):
        plot_mean(axes[i], per_ds_mean[i],
                  f"{title_prefix+' - ' if title_prefix else ''}{key}: dataset {i+1}")

    plot_mean(axes[3], global_mean,
              f"{title_prefix+' - ' if title_prefix else ''}{key}: mean over N={N}")

    fig.suptitle(f"{key}: first three dataset means + global{(' + overlay' if overlay is not None else '')}", y=0.98)
    fig.tight_layout()
    return fig, axes


# =============================================================================================== #

class NsimsMatricesAnalyzer:
    def __init__(
        self,
        api,
        noise,
        out_dir: str = "results/estm_mats_nsims",
        N_sims_values: List[int] | None = None,
        real_Z_mats: bool = True,
        recompute: bool = False,
    ):
        """
        Parameters
        ----------
        api : MatricesAPI
            Your MatricesAPI instance (must have .estm_mats)
        noise : Noise
            Your Noise object (must have Sigma_nom, gamma)
        out_dir : str
            Root directory where results and plots are stored.
        N_sims_values : list[int] | None
            Values of N_sims to sweep. If None, use default list.
        real_Z_mats : bool
            Passed to api.estm_mats (real_perf_mats flag).
        recompute : bool
            If True, ignore cache and re-estimate everything.
        """
        self.api = api
        self.noise = noise
        self.out_dir = Path(out_dir)
        self.out_dir.mkdir(parents=True, exist_ok=True)

        if N_sims_values is None:
            N_sims_values = [
                1, 2, 3, 5,          # Emergence of structure
                8, 12, 16,           # Early stability range
                20, 25, 30,          # Practical medoid stability region
                40, 50, 65,          # Larger-sample variance reduction
                80, 100, 120, 150    # High-data (plateau) regime
            ]

        self.N_sims_values = list(N_sims_values)
        self.real_Z_mats = real_Z_mats
        self.recompute = recompute

        self.data: Dict[str, Any] | None = None

        # Paths for cache
        self.cache_npz = self.out_dir / "matrices.npz"
        self.cache_meta = self.out_dir / "meta.json"

        plant, _ = api.get_system(FROM_DATA=False, gamma=noise.gamma, upd=False)
        A, Bw, Bu, Cz, Dzw, Dzu, Cy, Dyw = plant.A, plant.Bw, plant.Bu, plant.Cz, plant.Dzw, plant.Dzu, plant.Cy, plant.Dyw

        self.true_mats = {
            "A": A, "Bu": Bu, "Bw": Bw, 
            "Cy": Cy, "Dyw": Dyw, 
            "Cz": Cz, "Dzw": Dzw, "Dzu": Dzu,
        }

    # ------------------------------------------------------------------
    # Public API
    # ------------------------------------------------------------------
    def run(self) -> Dict[str, Any]:
        """
        Main entry point.
        - Load cached matrices if available and recompute=False
        - Otherwise estimate everything and save
        - Then run all analyses / plots and save figures
        Returns the data dict with all matrices and metadata.
        """
        if (not self.recompute) and self.cache_npz.exists() and self.cache_meta.exists():
            self._load_results()
        else:
            self._estimate_all()
            self._save_results()

        self._run_all_plots()
        return self.data

    # ------------------------------------------------------------------
    # Estimation & caching
    # ------------------------------------------------------------------
    def _estimate_all(self):
        """
        Estimate matrices for all N_sims and store in self.data.
        """
        Sigma_nom, gamma = self.noise.Sigma_nom, self.noise.gamma

        A_list, Bu_list, Bw_list = [], [], []
        Cy_list, Dyw_list = [], []
        Cz_list, Dzu_list, Dzw_list = [], [], []

        dims = None

        for N_sims in self.N_sims_values:
            op = Open_Loop(MAKE_DATA=False, EVAL_FROM_PATH=False, DATASETS=True, N=N_sims)
            datasets = op.datasets

            avg = select_representative_run(datasets) if N_sims != 1 else datasets
            x, u, y, z, x_next = avg["X"], avg["U"], avg["Y"], avg["Z"], avg["X_next"]

            nx, nu = x.shape[0], u.shape[0]
            ny, nz = y.shape[0], z.shape[0]

            (A, Bu, Bw, Cy, Dyw, Cz, Dzu, Dzw), (_, nw, _), (Sigma_nom, gamma) = self.api.estm_mats(
                X_=x,
                U_=u,
                X=x_next,
                Y_=y,
                Z_=z,
                Sigma_nom=Sigma_nom,
                real_perf_mats=self.real_Z_mats,
                gamma=gamma,
                estm_noise=False,
            )

            if dims is None:
                dims = {
                    "nx": nx,
                    "nu": nu,
                    "nw": nw,
                    "ny": ny,
                    "nz": nz,
                }

            A_list.append(A)
            Bu_list.append(Bu)
            Bw_list.append(Bw)

            Cy_list.append(Cy)
            Dyw_list.append(Dyw)

            Cz_list.append(Cz)
            Dzu_list.append(Dzu)
            Dzw_list.append(Dzw)

        # Stack along first axis: index ↔ index in N_sims_values
        A_arr = np.stack(A_list, axis=0)
        Bu_arr = np.stack(Bu_list, axis=0)
        Bw_arr = np.stack(Bw_list, axis=0)

        Cy_arr = np.stack(Cy_list, axis=0)
        Dyw_arr = np.stack(Dyw_list, axis=0)

        Cz_arr = np.stack(Cz_list, axis=0)
        Dzu_arr = np.stack(Dzu_list, axis=0)
        Dzw_arr = np.stack(Dzw_list, axis=0)

        self.data = {
            "N_sims_values": self.N_sims_values,
            "state": {
                "A": A_arr,
                "Bu": Bu_arr,
                "Bw": Bw_arr,
            },
            "output": {
                "Cy": Cy_arr,
                "Dyw": Dyw_arr,
            },
            "performance": {
                "Cz": Cz_arr,
                "Dzu": Dzu_arr,
                "Dzw": Dzw_arr,
            },
            "dims": dims,
        }

    def _save_results(self):
        """
        Save matrices (npz) + metadata (json).
        """
        assert self.data is not None, "No data to save."

        np.savez_compressed(
            self.cache_npz,
            A=self.data["state"]["A"],
            Bu=self.data["state"]["Bu"],
            Bw=self.data["state"]["Bw"],
            Cy=self.data["output"]["Cy"],
            Dyw=self.data["output"]["Dyw"],
            Cz=self.data["performance"]["Cz"],
            Dzu=self.data["performance"]["Dzu"],
            Dzw=self.data["performance"]["Dzw"],
        )

        meta = {
            "N_sims_values": self.data["N_sims_values"],
            "dims": self.data["dims"],
            "real_Z_mats": self.real_Z_mats,
        }
        with open(self.cache_meta, "w", encoding="utf-8") as f:
            json.dump(meta, f, indent=2)

    def _load_results(self):
        """
        Load matrices + metadata from disk.
        """
        with np.load(self.cache_npz) as npz:
            A = npz["A"]
            Bu = npz["Bu"]
            Bw = npz["Bw"]
            Cy = npz["Cy"]
            Dyw = npz["Dyw"]
            Cz = npz["Cz"]
            Dzu = npz["Dzu"]
            Dzw = npz["Dzw"]

        with open(self.cache_meta, "r", encoding="utf-8") as f:
            meta = json.load(f)

        self.N_sims_values = meta["N_sims_values"]
        dims = meta["dims"]

        self.data = {
            "N_sims_values": self.N_sims_values,
            "state": {
                "A": A,
                "Bu": Bu,
                "Bw": Bw,
            },
            "output": {
                "Cy": Cy,
                "Dyw": Dyw,
            },
            "performance": {
                "Cz": Cz,
                "Dzu": Dzu,
                "Dzw": Dzw,
            },
            "dims": dims,
        }

    # ------------------------------------------------------------------
    # Analysis & plots
    # ------------------------------------------------------------------
    def _run_all_plots(self):
        """
        Generate all analyses and figures, save them in subfolders.
        """
        assert self.data is not None, "Data not available."

        # Create plot subdirectories
        (self.out_dir / "spectral_radius").mkdir(exist_ok=True, parents=True)
        (self.out_dir / "eigA").mkdir(exist_ok=True, parents=True)
        (self.out_dir / "step_response").mkdir(exist_ok=True, parents=True)
        (self.out_dir / "fro_diff").mkdir(exist_ok=True, parents=True)
        (self.out_dir / "matrix_diff").mkdir(exist_ok=True, parents=True)
        (self.out_dir / "sin_response").mkdir(exist_ok=True, parents=True)

        #self._plot_spectral_radius()
        #self._plot_eigA_all()
        self._plot_step_responses_all()
        #self._plot_fro_differences()
        #self._plot_matrix_diff_heatmaps()
        self._plot_sinusoidal_responses_all(
            horizon=getattr(self, "step_horizon", 250),
            omega=0.1,          # change if you want
            amplitude=1.0,
            single_input=True,
        )

    # -------------- spectral radius -----------------------------------
    def _plot_spectral_radius(self):
        """
        Plot spectral radius of estimated A vs N_sims,
        with the TRUE spectral radius as a horizontal red line.
        """
        A_arr = self.data["state"]["A"]          # shape: (K, nx, nx)
        N_sims_values = self.N_sims_values

        # true model
        A_true = self.true_mats["A"]
        eig_true = np.linalg.eigvals(A_true)
        rho_true = np.max(np.abs(eig_true))

        # estimated spectral radii
        radii = []
        for A in A_arr:
            eigvals = np.linalg.eigvals(A)
            rho = np.max(np.abs(eigvals))
            radii.append(rho)

        fig, ax = plt.subplots(figsize=(7, 4))
        ax.plot(N_sims_values, radii, marker="o", label="estimated ρ(A)")
        ax.axhline(rho_true, color="red", linestyle="--", label="true ρ(A)")

        ax.set_xlabel("N_sims")
        ax.set_ylabel("spectral radius ρ(A)")
        ax.set_title("Spectral radius of A vs N_sims (estimates vs true)")
        ax.grid(True, alpha=0.3)
        ax.legend(loc="best")
        fig.tight_layout()

        out_path = self.out_dir / "spectral_radius" / "rho_A_vs_Nsims_with_true.pdf"
        fig.savefig(out_path, dpi=200)
        plt.close(fig)

    # -------------- eigenvalues of A ----------------------------------
    def _plot_eigA_all(self):
        A_arr = self.data["state"]["A"]
        N_sims_values = self.N_sims_values

        # One combined plot
        fig, ax = plt.subplots(figsize=(6, 6))
        for idx, (A, N) in enumerate(zip(A_arr, N_sims_values)):
            eigvals = np.linalg.eigvals(A)
            ax.scatter(
                eigvals.real,
                eigvals.imag,
                s=20,
                alpha=0.5,
                label=f"N={N}" if idx in (0, len(N_sims_values) - 1) else None,
            )
        # unit circle
        theta = np.linspace(0, 2 * np.pi, 400)
        ax.plot(np.cos(theta), np.sin(theta), linestyle="--", alpha=0.4)
        ax.set_xlabel("Re(λ)")
        ax.set_ylabel("Im(λ)")
        ax.set_title("Eigenvalues of A (all N_sims)")
        ax.grid(True, alpha=0.3)
        ax.set_aspect("equal", "box")
        if any(lbl is not None for lbl in ax.get_legend_handles_labels()[1]):
            ax.legend()
        fig.tight_layout()
        fig.savefig(self.out_dir / "eigA" / "eigA_all_overlay.pdf", dpi=200)
        plt.close(fig)

        # Also one plot per N_sims, if you want per-case inspection
        for A, N in zip(A_arr, N_sims_values):
            eigvals = np.linalg.eigvals(A)
            fig, ax = plt.subplots(figsize=(6, 6))
            ax.scatter(eigvals.real, eigvals.imag, s=30)
            ax.plot(np.cos(theta), np.sin(theta), linestyle="--", alpha=0.4)
            ax.set_xlabel("Re(λ)")
            ax.set_ylabel("Im(λ)")
            ax.set_title(f"Eigenvalues of A, N_sims={N}")
            ax.grid(True, alpha=0.3)
            ax.set_aspect("equal", "box")
            fig.tight_layout()
            fig.savefig(self.out_dir / "eigA" / f"eigA_N{N}.pdf", dpi=200)
            plt.close(fig)

    # -------------- step responses for all A --------------------------
    def _plot_step_responses_all(self, horizon: int | None = None):
        """
        Step response of the first state for all estimated (A, Bu),
        with the TRUE (A, Bu) overlaid in red.

        Legend in the aggregate plot shows ONLY:
            - TRUE
            - the 2 closest estimated trajectories.
        """
        A_arr  = self.data["state"]["A"]    # (K, nx, nx)
        Bu_arr = self.data["state"]["Bu"]   # (K, nx, nu)
        N_sims_values = self.N_sims_values

        if horizon is None:
            horizon = getattr(self, "step_horizon", 250)

        K, nx, _ = A_arr.shape
        _, _, nu = Bu_arr.shape
        k_grid = np.arange(horizon)

        # ---------- TRUE model ----------
        A_true  = self.true_mats["A"]
        Bu_true = self.true_mats["Bu"]

        if A_true.shape != (nx, nx):
            raise ValueError(f"True A has shape {A_true.shape}, expected {(nx, nx)}")
        if Bu_true.shape != (nx, nu):
            raise ValueError(f"True Bu has shape {Bu_true.shape}, expected {(nx, nu)}")

        u_step_true = np.ones((nu, 1))
        x_true = np.zeros((nx, 1))
        traj_true = []

        for _ in range(horizon):
            x_true = A_true @ x_true + Bu_true @ u_step_true
            traj_true.append(x_true.copy())

        traj_true = np.hstack(traj_true)   # (nx, horizon)

        # ---------- collect all estimated trajectories ----------
        traj_hats = np.zeros((K, nx, horizon))

        for i, (A_hat, Bu_hat) in enumerate(zip(A_arr, Bu_arr)):
            u_step = np.ones((nu, 1))
            x = np.zeros((nx, 1))
            tmp = []

            for _ in range(horizon):
                x = A_hat @ x + Bu_hat @ u_step
                tmp.append(x.copy())

            traj_hats[i] = np.hstack(tmp)

        # ---------- select 2 closest ----------
        closest_idx = self._select_closest_indices(traj_true, traj_hats, k=2)
        closest_idx_set = set(closest_idx)

        # ---------- aggregate overlay plot ----------
        fig_all, ax_all = plt.subplots(figsize=(7, 4))

        for i, N in enumerate(N_sims_values):
            label = None
            if i in closest_idx_set:
                label = f"estimate (N={N})"

            ax_all.plot(
                k_grid,
                traj_hats[i, 0, :],
                alpha=0.35 if i not in closest_idx_set else 0.9,
                linewidth=1.0 if i not in closest_idx_set else 1.8,
                label=label,
            )

        # TRUE in red, always labeled
        ax_all.plot(
            k_grid,
            traj_true[0, :],
            color="red",
            linewidth=2.0,
            label="TRUE",
        )

        ax_all.set_xlabel("k")
        ax_all.set_ylabel("x[0]")
        ax_all.set_title("Step response (first state) vs N_sims\n(TRUE + 2 closest estimates)")
        ax_all.grid(True, alpha=0.3)
        ax_all.legend(loc="best")
        fig_all.tight_layout()
        fig_all.savefig(self.out_dir / "step_response" / "step_all_overlay_true_2closest.pdf", dpi=200)
        plt.close(fig_all)

        # ---------- individual plots per N_sims (keep as before, full legend) ----------
        for i, N in enumerate(N_sims_values):
            fig, ax = plt.subplots(figsize=(7, 4))
            ax.plot(
                k_grid,
                traj_hats[i, 0, :],
                label=f"estimate (N={N})",
            )
            ax.plot(
                k_grid,
                traj_true[0, :],
                color="red",
                linestyle="--",
                linewidth=2.0,
                label="TRUE",
            )
            ax.set_xlabel("k")
            ax.set_ylabel("x[0]")
            ax.set_title(f"Step response (first state), N_sims={N} vs TRUE")
            ax.grid(True, alpha=0.3)
            ax.legend(loc="best")
            fig.tight_layout()
            fig.savefig(self.out_dir / "step_response" / f"step_A_N{N}_vs_true.pdf", dpi=200)
            plt.close(fig)

    def _plot_sinusoidal_responses_all(
        self,
        horizon: int | None = None,
        omega: float = 0.1,
        amplitude: float = 1.0,
        single_input: bool = True,
    ):
        """
        Sinusoidal response of the first state for all estimated (A, Bu),
        with the TRUE (A, Bu) overlaid in red.

        Legend in the aggregate plot shows ONLY:
            - TRUE
            - the 2 closest estimated trajectories.

        Dynamics: x_{k+1} = A x_k + Bu u_k,  x_0 = 0
        Input:    u_k = amplitude * sin(omega * k)
        """
        A_arr  = self.data["state"]["A"]    # (K, nx, nx)
        Bu_arr = self.data["state"]["Bu"]   # (K, nx, nu)
        N_sims_values = self.N_sims_values

        if horizon is None:
            horizon = getattr(self, "step_horizon", 1000)

        K, nx, _ = A_arr.shape
        _, _, nu = Bu_arr.shape
        k_grid = np.arange(horizon)

        # ---------- TRUE model ----------
        A_true  = self.true_mats["A"]
        Bu_true = self.true_mats["Bu"]

        if A_true.shape != (nx, nx):
            raise ValueError(f"True A has shape {A_true.shape}, expected {(nx, nx)}")
        if Bu_true.shape != (nx, nu):
            raise ValueError(f"True Bu has shape {Bu_true.shape}, expected {(nx, nu)}")

        # input for TRUE model
        if single_input:
            u_true = np.zeros((nu, horizon))
            u_true[0, :] = amplitude * np.sin(omega * k_grid)
        else:
            u_true = amplitude * np.sin(omega * k_grid)[None, :].repeat(nu, axis=0)

        x_true = np.zeros((nx, 1))
        traj_true = []

        for k in range(horizon):
            u_k = u_true[:, [k]]
            x_true = A_true @ x_true + Bu_true @ u_k
            traj_true.append(x_true.copy())

        traj_true = np.hstack(traj_true)   # (nx, horizon)

        # ---------- collect all estimated trajectories ----------
        traj_hats = np.zeros((K, nx, horizon))

        for i, (A_hat, Bu_hat) in enumerate(zip(A_arr, Bu_arr)):
            if single_input:
                u_hat = np.zeros((nu, horizon))
                u_hat[0, :] = amplitude * np.sin(omega * k_grid)
            else:
                u_hat = amplitude * np.sin(omega * k_grid)[None, :].repeat(nu, axis=0)

            x = np.zeros((nx, 1))
            tmp = []

            for k in range(horizon):
                u_k = u_hat[:, [k]]
                x = A_hat @ x + Bu_hat @ u_k
                tmp.append(x.copy())

            traj_hats[i] = np.hstack(tmp)

        # ---------- select 2 closest ----------
        closest_idx = self._select_closest_indices(traj_true, traj_hats, k=2)
        closest_idx_set = set(closest_idx)

        # ensure directory exists
        (self.out_dir / "sin_response").mkdir(exist_ok=True, parents=True)

        # ---------- aggregate overlay plot ----------
        fig_all, ax_all = plt.subplots(figsize=(7, 4))

        for i, N in enumerate(N_sims_values):
            label = None
            if i in closest_idx_set:
                label = f"estimate (N={N})"

            ax_all.plot(
                k_grid,
                traj_hats[i, 0, :],
                alpha=0.35 if i not in closest_idx_set else 0.9,
                linewidth=1.0 if i not in closest_idx_set else 1.8,
                label=label,
            )

        # TRUE in red
        ax_all.plot(
            k_grid,
            traj_true[0, :],
            color="red",
            linewidth=2.0,
            label="TRUE",
        )

        ax_all.set_xlabel("k")
        ax_all.set_ylabel("x[0]")
        ax_all.set_title(
            "Sinusoidal response (first state) vs N_sims\n"
            f"(ω={omega:.3f}, amp={amplitude}, TRUE + 2 closest)"
        )
        ax_all.grid(True, alpha=0.3)
        ax_all.legend(loc="best")
        fig_all.tight_layout()
        fig_all.savefig(
            self.out_dir / "sin_response" / "sin_all_overlay_true_2closest.pdf",
            dpi=200,
        )
        plt.close(fig_all)

        # ---------- individual plots per N_sims ----------
        for i, N in enumerate(N_sims_values):
            fig, ax = plt.subplots(figsize=(7, 4))
            ax.plot(
                k_grid,
                traj_hats[i, 0, :],
                label=f"estimate (N={N})",
            )
            ax.plot(
                k_grid,
                traj_true[0, :],
                color="red",
                linestyle="--",
                linewidth=2.0,
                label="TRUE",
            )

            ax.set_xlabel("k")
            ax.set_ylabel("x[0]")
            ax.set_title(
                f"Sinusoidal response (first state), N_sims={N} vs TRUE\n"
                f"(ω={omega:.3f}, amp={amplitude})"
            )
            ax.grid(True, alpha=0.3)
            ax.legend(loc="best")
            fig.tight_layout()
            fig.savefig(
                self.out_dir / "sin_response" / f"sin_A_N{N}_vs_true.pdf",
                dpi=200,
            )
            plt.close(fig)

    def _select_closest_indices(self, traj_true: np.ndarray, traj_hats: np.ndarray, k: int = 2):
        """
        Given:
            traj_true : (nx, T)
            traj_hats : (K, nx, T)
        return indices of the k closest trajectories in RMS error
        (computed on the first state).

        Returns a list of indices (length <= k).
        """
        nx, T = traj_true.shape
        K, nx_hat, T_hat = traj_hats.shape
        assert nx_hat == nx and T_hat == T, "Shape mismatch in trajectories."

        err = []
        for i in range(K):
            diff = traj_hats[i, 0, :] - traj_true[0, :]
            rms = np.sqrt(np.mean(diff**2))
            err.append(rms)

        err = np.array(err)
        order = np.argsort(err)
        k = min(k, K)
        return order[:k].tolist()

    # -------------- Frobenius differences vs reference ----------------
    def _plot_fro_differences(self):
        N_sims_values = self.N_sims_values

        mats = {
            "A":   self.data["state"]["A"],
            "Bu":  self.data["state"]["Bu"],
            "Bw":  self.data["state"]["Bw"],
            "Cy":  self.data["output"]["Cy"],
            "Dyw": self.data["output"]["Dyw"],
            "Cz":  self.data["performance"]["Cz"],
            "Dzu": self.data["performance"]["Dzu"],
            "Dzw": self.data["performance"]["Dzw"],
        }

        for name, arr in mats.items():
            if name not in self.true_mats:
                # just in case, but given your dict it should all be there
                continue

            true_M = self.true_mats[name]
            true_norm = np.linalg.norm(true_M, "fro") + 1e-12

            norms = []
            rel_diffs = []

            for M in arr:
                norms.append(np.linalg.norm(M, "fro"))
                rel_diffs.append(np.linalg.norm(M - true_M, "fro") / true_norm)

            fig, ax1 = plt.subplots(figsize=(7, 4))
            ax1.plot(N_sims_values, norms, marker="o", label="||M_hat||_F")
            ax1.axhline(true_norm, linestyle="--", alpha=0.5, label="||M_true||_F")
            ax1.set_xlabel("N_sims")
            ax1.set_ylabel("||M||_F")
            ax1.grid(True, alpha=0.3)

            ax2 = ax1.twinx()
            ax2.plot(
                N_sims_values,
                rel_diffs,
                marker="s",
                linestyle="--",
                label="rel diff vs true",
            )
            ax2.set_ylabel("relative Frobenius difference")

            lines1, labels1 = ax1.get_legend_handles_labels()
            lines2, labels2 = ax2.get_legend_handles_labels()
            ax1.legend(lines1 + lines2, labels1 + labels2, loc="best")

            fig.suptitle(f"Frobenius norm & relative diff vs TRUE for {name}")
            fig.tight_layout()
            fig.savefig(
                self.out_dir / "fro_diff" / f"{name}_fro_vs_Nsims_vs_true.pdf",
                dpi=200,
            )
            plt.close(fig)

    # -------------- matrix difference heatmaps ------------------------
    def _plot_matrix_diff_heatmaps(self):
        mats = {
            "A":   self.data["state"]["A"],
            "Bu":  self.data["state"]["Bu"],
            "Bw":  self.data["state"]["Bw"],
            "Cy":  self.data["output"]["Cy"],
            "Dyw": self.data["output"]["Dyw"],
            "Cz":  self.data["performance"]["Cz"],
            "Dzu": self.data["performance"]["Dzu"],
            "Dzw": self.data["performance"]["Dzw"],
        }

        N_sims_values = self.N_sims_values

        for name, arr in mats.items():
            if name not in self.true_mats:
                continue

            true_M = self.true_mats[name]
            subdir = self.out_dir / "matrix_diff" / f"{name}_vs_true"
            subdir.mkdir(exist_ok=True, parents=True)

            for M, N in zip(arr, N_sims_values):
                diff = M - true_M

                fig, ax = plt.subplots(figsize=(5, 4))
                im = ax.imshow(diff, aspect="auto")
                ax.set_title(f"{name}_hat - {name}_true, N_sims={N}")
                ax.set_xlabel("columns")
                ax.set_ylabel("rows")
                fig.colorbar(im, ax=ax, fraction=0.046, pad=0.04)
                fig.tight_layout()
                fig.savefig(subdir / f"{name}_diff_vs_true_N{N}.pdf", dpi=200)
                plt.close(fig)


