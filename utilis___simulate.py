#!/usr/bin/env python3
import argparse, csv, yaml
import numpy as np
from pathlib import Path
from utilis___matrices import MatricesAPI

import matplotlib.pyplot as plt
from utilis___systems import Plant, Controller


## ------------------------- CLOSED-LOOP SIMULATION CLASS --------------------------

class Closed_Loop():
    def __init__(self, TEST=False):
        if TEST: self.test()
    
    def test(self):
        # Use the same plant as the optimization example (seed=7)
        api = MatricesAPI()
        plant, _ = api.get_system(seed=7, FROM_DATA=False)

        Ac = np.array([
            [ 0.3449, -0.4085,  0.    ,  0.    ],
            [-0.4279,  0.4803,  0.    ,  0.    ],
            [ 0.    ,  0.    ,  0.    ,  0.    ],
            [ 0.    ,  0.    ,  0.    ,  0.    ],
        ], dtype=float)

        Bc = np.array([
            [ 0.2538, -0.2417],
            [-0.2802,  0.3522],
            [ 0.    ,  0.    ],
            [ 0.    ,  0.    ],
        ], dtype=float)

        Cc = np.array([
            [ 0.1093, -0.0614,  0.    ,  0.    ],
            [ 0.1129, -0.0553,  0.    ,  0.    ],
        ], dtype=float)

        Dc = np.array([
            [-1.0166,  0.1460],
            [-1.0123,  0.1628],
        ], dtype=float)

        ctrl = Controller(Ac=Ac, Bc=Bc, Cc=Cc, Dc=Dc)
        Sigma_w = 0.7 * np.eye(2)

        sim = self.simulate_closed_loop(plant, ctrl, Sigma_w, T=800, seed=11)
        print("Simulated shapes:",
            {k: v.shape for k, v in sim.items() if isinstance(v, np.ndarray)})

        out = self.save_npz(sim, "closed_loop_run_seed11_T800.npz")
        print(f"Saved time series to {out}")

        self.plot_timeseries(sim)

    def simulate_closed_loop(self, plant: Plant,
                            ctrl: Controller,
                            Sigma_w: np.ndarray,
                            T: int = 500,
                            seed: int = 0,
                            x0: np.ndarray | None = None,
                            xc0: np.ndarray | None = None):
        """
        Simulate the interconnection (no composite shortcut) so we can view y, u explicitly.

        y_t = C_y x_t + D_yw w_t
        u_t = C_c x_c_t + D_c y_t
        x_{t+1} = A x_t + B_u u_t + B_w w_t
        x_{c,t+1} = A_c x_{c,t} + B_c y_t
        z_t = C_z x_t + D_zu u_t + D_zw w_t
        """
        A, Bw, Bu, Cz, Dzw, Dzu, Cy, Dyw = \
            plant.A, plant.Bw, plant.Bu, plant.Cz, plant.Dzw, plant.Dzu, plant.Cy, plant.Dyw
        Ac, Bc, Cc, Dc = ctrl.Ac, ctrl.Bc, ctrl.Cc, ctrl.Dc

        nx = A.shape[0]
        nw = Bw.shape[1]
        nu = Bu.shape[1]
        nz = Cz.shape[0]
        ny = Cy.shape[0]
        nxc = Ac.shape[0]

        # Initial conditions
        x = np.zeros((nx, 1)) if x0 is None else np.asarray(x0, float).reshape(nx, 1)
        xc = np.zeros((nxc, 1)) if xc0 is None else np.asarray(xc0, float).reshape(nxc, 1)

        # Precompute a sampling factor for w
        rng = np.random.default_rng(seed)
        try:
            L = np.linalg.cholesky(Sigma_w)
        except np.linalg.LinAlgError:
            L = np.linalg.cholesky(Sigma_w + 1e-12 * np.eye(nw))

        # Storage
        X  = np.zeros((T, nx))
        Xc = np.zeros((T, nxc))
        Y  = np.zeros((T, ny))
        U  = np.zeros((T, nu))
        Z  = np.zeros((T, nz))
        W  = np.zeros((T, nw))

        for t in range(T):
            w = (L @ rng.standard_normal((nw, 1))).astype(float)
            y = Cy @ x + Dyw @ w
            u = Cc @ xc + Dc @ y
            z = Cz @ x + Dzu @ u + Dzw @ w

            # Log
            X[t, :]  = x.ravel()
            Xc[t, :] = xc.ravel()
            Y[t, :]  = y.ravel()
            U[t, :]  = u.ravel()
            Z[t, :]  = z.ravel()
            W[t, :]  = w.ravel()

            # Update
            x_next  = A @ x + Bu @ u + Bw @ w
            xc_next = Ac @ xc + Bc @ y
            x, xc = x_next, xc_next

        return {
            "X": X, "Xc": Xc, "Y": Y, "U": U, "Z": Z, "W": W,
            "T": T, "nx": nx, "nxc": nxc, "ny": ny, "nu": nu, "nz": nz, "nw": nw
        }

    def plot_timeseries(self, sim):
        T = sim["T"]
        t = np.arange(T)

        # States x
        plt.figure(figsize=(10, 6))
        for i in range(sim["nx"]):
            plt.plot(t, sim["X"][:, i], label=f"x[{i}]")
        plt.title("Plant states x")
        plt.xlabel("t")
        plt.legend()
        plt.grid(True, alpha=0.3)

        # Controller states x_c
        plt.figure(figsize=(10, 6))
        for i in range(sim["nxc"]):
            plt.plot(t, sim["Xc"][:, i], label=f"x_c[{i}]")
        plt.title("Controller states x_c")
        plt.xlabel("t")
        plt.legend()
        plt.grid(True, alpha=0.3)

        # Measured output y
        plt.figure(figsize=(10, 6))
        for i in range(sim["ny"]):
            plt.plot(t, sim["Y"][:, i], label=f"y[{i}]")
        plt.title("Measured output y")
        plt.xlabel("t")
        plt.legend()
        plt.grid(True, alpha=0.3)

        # Control input u
        plt.figure(figsize=(10, 5))
        for i in range(sim["nu"]):
            plt.plot(t, sim["U"][:, i], label=f"u[{i}]")
        plt.title("Control input u")
        plt.xlabel("t")
        plt.legend()
        plt.grid(True, alpha=0.3)

        # Performance output z
        plt.figure(figsize=(10, 6))
        for i in range(sim["nz"]):
            plt.plot(t, sim["Z"][:, i], label=f"z[{i}]")
        plt.title("Performance output z")
        plt.xlabel("t")
        plt.legend()
        plt.grid(True, alpha=0.3)

        plt.show()

    def save_npz(self, sim, fname="cl_timeseries.npz"):
        np.savez_compressed(fname, **sim)
        return fname


## ------------------------- OPEN-LOOP SIMULATION CLASS ----------------------------

class Open_Loop():
    def __init__(self, MAKE_DATA=False, EVAL_FROM_PATH=True, PLOT=False, yaml_path="problem___parameters.yaml"):
        if yaml is None:
            raise ImportError("PyYAML not available. Install with `pip install pyyaml`.")
        with open(yaml_path, "r", encoding="utf-8") as f:
            cfg = yaml.safe_load(f)

        self.p = cfg.get("params", {})
        out = self.p.get("directories", {}).get("data", "./out/data/session_01")
        _type = self.p.get("plant", {}).get("type", "explicit")
        _model = self.p.get("model", "independent")
        _data = "DDD" if bool(self.p.get("FROM_DATA", False)) else "MBD"

        self.csv_path = out + f"___{_type}_{_model}_{_data}.csv"

        self.out = Path(self.csv_path)
        self.out.parent.mkdir(parents=True, exist_ok=True)
        self.estim_path = self.out.with_suffix("").as_posix() + "_estmMat.npz"
        self.truth_path = self.out.with_suffix("").as_posix() + "_trueMat.npz"



        if MAKE_DATA: self.make_data()
        if EVAL_FROM_PATH: self.evaluate_from_path()
        if PLOT: 
            metrics = self.plot_est_vs_truth(nsteps=80, x0_mode="e1", show=True)
            print("\n[PLT] Plotting completed. Metrics:", metrics)

    
    """
    python simulate_pe_openloop.py --T 3000 --nx 4 --nu 2 --nw 2 --ny 2 --nz 3 --input multisine --amp 1.0 --w_std 0.1 --seed 42 --out out/data/session01.csv
    """


    def make_data(self):
        ap = argparse.ArgumentParser(description="Simulate open-loop with persistently exciting input and save CSV.")
        ap.add_argument("--T", type=int, default=3000, help="number of time steps")
        ap.add_argument("--seed", type=int, default=0)
        ap.add_argument("--input", choices=["multisine", "prbs"], default="multisine")
        ap.add_argument("--amp", type=float, default=1.0, help="input amplitude")
        ap.add_argument("--w_std", type=float, default=0.1, help="disturbance std scaling")
        ap.add_argument("--delimiter", type=str, default=",")
        args = ap.parse_args()


        def prbs(nu, T, shift=7, seed=0, amp=1.0):
            """ PRBS generator per input channel. shift sets LFSR length. """
            rng = np.random.default_rng(seed)
            U = np.zeros((nu, T))
            for i in range(nu):
                # primitive-ish taps for small shift values
                taps = {
                    5: (5, 2), 6: (6, 1), 7: (7, 1), 9: (9, 5),
                    10: (10, 3), 11: (11, 2), 15: (15, 1)
                }
                s = shift if shift in taps else 7
                reg = rng.integers(1, 2**s, dtype=np.int64)
                ti, tj = taps[s]
                seq = np.empty(T)
                for t in range(T):
                    bit = (reg >> (ti - 1)) ^ (reg >> (tj - 1))
                    bit &= 1
                    reg = ((reg << 1) & ((1 << s) - 1)) | bit
                    seq[t] = 1.0 if (reg & 1) else -1.0
                U[i, :] = amp * seq
            return U

        def multisine(nu, T, ntones=8, seed=0, amp=1.0):
            """ Sum of randomized sines and cosines per channel. """
            rng = np.random.default_rng(seed)
            U = np.zeros((nu, T))
            t = np.arange(T)
            for i in range(nu):
                freqs = rng.choice(np.arange(1, max(2, T // 10)), size=ntones, replace=False)
                phases = 2 * np.pi * rng.random(ntones)
                ui = np.zeros(T)
                for k, f in enumerate(freqs):
                    ui += np.sin(2 * np.pi * f * t / T + phases[k])
                    ui += np.cos(2 * np.pi * f * t / T + phases[k] / 3.0)
                ui /= np.max(np.abs(ui)) + 1e-12
                U[i, :] = amp * ui
            return U

        def pe_check(X, U):
            """ Report condition number of DD^T where D=[X;U]. """
            D = np.vstack([X, U])
            G = D @ D.T
            svals = np.linalg.svd(G, compute_uv=False)
            cond = np.inf if svals[-1] <= 1e-14 else svals[0] / svals[-1]
            return cond, svals

        def simulate_open_loop(A, Bu, Bw, T, x0, U, w_std=0.1, seed=0):
            rng = np.random.default_rng(seed)
            nx, nu = Bu.shape
            nw = Bw.shape[1]
            X = np.zeros((nx, T))
            X[:, 0] = x0
            W = rng.normal(0.0, 1.0, size=(nw, T)) * w_std
            for t in range(T - 1):
                X[:, t + 1] = (A @ X[:, t] + Bu @ U[:, t] + Bw @ W[:, t])
            return X, W

        def synth_outputs(X, U, R, ny, nz):
            nx, T = X.shape
            nu = U.shape[0]
            # Measured output: pick first ny states, no direct disturbance by default
            ny = min(ny, nx)
            Cy = np.zeros((ny, nx))
            Cy[np.arange(ny), np.arange(ny)] = 1.0
            Dyw = np.zeros((ny, nx))  # maps residual proxy to y; keep zero unless you know better
            Y = Cy @ X[:, :-1] + Dyw @ R  # aligned to t=0..T-2

            # Performance output: states + mild control penalty, no Dzw
            nz_eff = min(nz, nx)
            Cz = np.zeros((nz, nx))
            for i in range(nz_eff):
                Cz[i, i] = 1.0
            Dzu = 0.05 * np.eye(nz, nu)
            Dzw = np.zeros((nz, nx))
            Z = Cz @ X[:, :-1] + Dzu @ U[:, :-1] + Dzw @ R
            return Y, Z

        def synth_outputs_with_mats(X, U, R, Cy, Dyw, Cz, Dzu, Dzw, ny, nz):
            """
            Returns Y, Z plus the exact matrices used to generate them:
            Cy, Dyw, Cz, Dzu, Dzw.
            Shapes:
            X: (nx,T), U: (nu,T), R: (nx,T-1)
            Y: (ny,T-1), Z: (nz,T-1)
            """
            nx, T = X.shape
            nu = U.shape[0]
            Y = Cy @ X[:, :-1] + Dyw @ R        # aligned to t = 0..T-2
            Z = Cz @ X[:, :-1] + Dzu @ U[:, :-1] + Dzw @ R

            return Y, Z


        T = args.T
        rng = np.random.default_rng(args.seed)

        api = MatricesAPI()
        A, Bu, Bw = api.build_AB_from_yaml()
        Cz, Dzw, Dzu, Cy, Dyw = api.build_out_matrices()
        nx, nw, nu, ny, nz = api.get_dimensions_from_yaml()

        # Persistently exciting input
        if args.input == "prbs":
            U = prbs(nu, T, shift=11, seed=args.seed + 17, amp=args.amp)
        else:
            U = multisine(nu, T, ntones=12, seed=args.seed + 23, amp=args.amp)

        # Simulate
        x0 = rng.normal(0, 1.0, size=nx)
        X, W = simulate_open_loop(A, Bu, Bw, T, x0, U, w_std=args.w_std, seed=args.seed + 101)

        # One-step alignment and residual proxy R = X+ - AX - BU
        X_reg = X[:, :-1]       # x_0..x_{T-2}
        X_next = X[:, 1:]       # x_1..x_{T-1}
        R = X_next - (A @ X_reg + Bu @ U[:, :-1])  # nx x (T-1)

        # Outputs Y,Z and the exact matrices used to generate them
        #Y, Z = synth_outputs_with_mats(X, U, R, Cy, Dyw, Cz, Dzu, Dzw, ny=ny, nz=nz)

        # PE sanity check
        cond, svals = pe_check(X_reg, U[:, :-1])
        print(f"[PE] cond( D D^T ) = {cond:.2e}   rank={np.sum(svals>1e-10)}/{nx+nu}")

        # Save CSV

        headers = []
        for i in range(nx): headers.append(f"x{i+1}")
        for j in range(nu): headers.append(f"u{j+1}")
        """for k in range(ny): headers.append(f"y{k+1}")
        for k in range(nz): headers.append(f"z{k+1}")"""

        rows = []
        for t in range(T):
            row = []
            row.extend(X[:, t].tolist())
            row.extend(U[:, t].tolist())
            """if t < T-1:
                row.extend(Y[:, t].tolist())
                row.extend(Z[:, t].tolist())
            else:
                row.extend(Y[:, -1].tolist())
                row.extend(Z[:, -1].tolist())"""
            rows.append(row)

        with open(self.out, "w", newline="", encoding="utf-8") as f:
            wr = csv.writer(f, delimiter=args.delimiter)
            wr.writerow(headers)
            wr.writerows(rows)

        print(f"[OK] Saved {self.out} with shape ({len(rows)} rows, {len(headers)} cols).")

        # >>> NEW: save ground-truth matrices and minimal metadata <<<
        np.savez_compressed(
            self.truth_path,
            # true plant
            A=A, Bu=Bu, Bw=Bw,
            # true output/performance blocks used to synthesize Y,Z
            Cy=Cy, Dyw=Dyw, Cz=Cz, Dzu=Dzu, Dzw=Dzw,
            # optional metadata for reproducibility
            nx=nx, nu=nu, nw=nw, ny=ny, nz=nz, T=T,
            seed=args.seed, input=args.input, amp=args.amp, w_std=args.w_std
        )
        print(f"[OK] Saved ground-truth matrices to {self.truth_path}")

    def evaluate_from_path(
        self, 
        ridge: float = 1e-6,
        delimiter: str = ",",
    ):
        """
        Rebuilds the plant directly from CSV (x*, u*, optional y*, z*),
        prints estimated matrices, and if a ground-truth .npz is provided,
        prints relative errors.

        truth_npz (optional) must contain any subset of:
        A, Bu, Bw, Cy, Dyw, Cz, Dzu, Dzw
        Anything missing will just be skipped in the comparison.
        """

        def _rel_or_abs_err(est: np.ndarray, true: np.ndarray, eps: float = 1e-12) -> tuple[float, str]:
            """
            Returns (value, mode), where mode is 'rel' or 'abs'.
            If ||true|| is tiny, report absolute ||est||; else relative ||est-true||/||true||.
            """
            nt = np.linalg.norm(true, "fro")
            if nt < 1e-10:
                return float(np.linalg.norm(est, "fro")), "abs"
            return float(np.linalg.norm(est - true, "fro") / (nt + eps)), "rel"

        def _print_block(name: str, M: np.ndarray, indent: int = 2):
            pad = " " * indent
            print(f"{name:<4} shape={M.shape}  ||·||_F={np.linalg.norm(M, 'fro'):.4g}")
            # tiny preview to avoid screen spam
            r, c = M.shape
            rshow, cshow = min(r, 15), min(c, 15)
            preview = M[:rshow, :cshow]
            with np.printoptions(precision=3, suppress=True):
                print(pad + str(preview))
                if r > rshow or c > cshow:
                    print(pad + f"... ({r}x{c} total)")

        def _proj_subspace(M: np.ndarray) -> np.ndarray:
            Q, _ = np.linalg.qr(M)  # orthonormal basis of the column space
            return Q

        def _subspace_gap(A: np.ndarray, B: np.ndarray) -> float:
            """
            Principal-angle based gap between colspaces(A) and colspaces(B).
            0 = identical subspaces, 1 = orthogonal.
            """
            Qa = _proj_subspace(A)
            Qb = _proj_subspace(B)
            s = np.linalg.svd(Qa.T @ Qb, compute_uv=False)
            return float(np.sqrt(1.0 - np.min(s)**2))

        def _procrustes_align(A: np.ndarray, B: np.ndarray) -> np.ndarray:
            """
            Find R (square) minimizing ||A - B R||_F, assuming A,B have same row count.
            Returns A_aligned = B R.
            """
            U, _, Vt = np.linalg.svd(B.T @ A, full_matrices=False)
            R = U @ Vt
            return B @ R


        csv_path = str(self.csv_path)
        print(f"\n[DDD] Loading CSV: {csv_path}")

        api = MatricesAPI()
        nx, nw, nu, ny, nz = api.get_dimensions_from_yaml()

        plant_est, ctrl0 = api.make_matrices_from_data(
            data_csv=csv_path,
            delimiter=delimiter,
            ridge=ridge,
        )

        Ahat, Buhat, Bwhat = plant_est.A, plant_est.Bu, plant_est.Bw
        Cyhat, Dywhat = plant_est.Cy, plant_est.Dyw
        Czhat, Dzuhat, Dzwhat = plant_est.Cz, plant_est.Dzu, plant_est.Dzw

        print("\n[DDD] Estimated matrices (from data)")
        _print_block("Ahat", Ahat)
        _print_block("Buhat", Buhat)
        _print_block("Bwhat", Bwhat)
        _print_block("Cyhat", Cyhat)
        _print_block("Dywhat", Dywhat)
        _print_block("Czhat", Czhat)
        _print_block("Dzuhat", Dzuhat)
        _print_block("Dzwhat", Dzwhat)

        truth_npz = Path(csv_path).with_suffix("").as_posix() + "_trueMat.npz" #truth_npz=out.with_suffix("").as_posix() + "_truth.npz"
        if truth_npz is None:
            print("\n[DDD] No ground-truth .npz provided. Skipping comparisons.")
            return plant_est, ctrl0

        truth_npz = str(truth_npz)
        if not Path(truth_npz).exists():
            print(f"\n[WARN] truth_npz file not found: {truth_npz}. Skipping comparisons.")
            return plant_est, ctrl0

        truth = np.load(truth_npz)
        print(f"\n[GT ] Loaded ground truth: {truth_npz}")

        pairs = [
        ("Ahat",  "A"),
        ("Buhat", "Bu"),
        ("Bwhat", "Bw"),
        ("Cyhat", "Cy"),
        ("Dywhat","Dyw"),
        ("Czhat", "Cz"),
        ("Dzuhat","Dzu"),
        ("Dzwhat","Dzw"),
        ]

        print("\n[GT ] True matrices (matching estimate order)")
        for est_label, true_key in pairs:
            if true_key in truth:
                # label aligns visually with the est block names
                _print_block(est_label.replace("hat","tru"), truth[true_key])

        def cmp(name_est, M_est, name_true):
            if name_true in truth:
                M_true = truth[name_true]
                if name_true == "Bw":
                    gap = _subspace_gap(M_est, M_true)
                    M_est_aligned = _procrustes_align(M_true, M_est)
                    val, mode = _rel_or_abs_err(M_est_aligned, M_true)
                    print(f"[cmp] {name_est:>6} vs {name_true:<6}  subspace_gap={gap:.3e}  {mode}={val:.3e}")
                else:
                    val, mode = _rel_or_abs_err(M_est, M_true)
                    print(f"[cmp] {name_est:>6} vs {name_true:<6}  {mode}={val:.3e}")
            else:
                print(f"[cmp] {name_est:>6}: no '{name_true}' in truth file; skipped.")


        # Compare whatever the truth file actually contains
        print("\n[CMP] Comparison of estimates vs ground-truth:")
        cmp("Ahat",   Ahat,   "A")
        cmp("Buhat",  Buhat,  "Bu")
        cmp("Bwhat",  Bwhat,  "Bw")
        cmp("Cyhat",  Cyhat,  "Cy")
        cmp("Dywhat", Dywhat, "Dyw")
        cmp("Czhat",  Czhat,  "Cz")
        cmp("Dzuhat", Dzuhat, "Dzu")
        cmp("Dzwhat", Dzwhat, "Dzw")


        np.savez_compressed(
            self.estim_path,
            # true plant
            Ahat=Ahat, Buhat=Buhat, Bwhat=Bwhat,
            # true output/performance blocks used to synthesize Y,Z
            Cyhat=Cyhat, Dywhat=Dywhat, Czhat=Czhat, Dzuhat=Dzuhat, Dzwhat=Dzwhat,
            # optional metadata for reproducibility
            nx=nx, nu=nu, nw=nw, ny=ny, nz=nz,
        )
        print(f"[OK] Saved ground-estimated matrices to {self.estim_path}")


        return plant_est, ctrl0

    def plot_est_vs_truth(
        self,
        nsteps: int = 60,
        x0_mode: str = "e1",   # "e1" | "ones" | "random"
        seed: int = 0,
        save_dir: str | None = None,
        show: bool = True
    ):
        """
        Compare estimated vs true matrices with a set of informative plots.

        - Loads A (true) from `truth_path` and Ahat from `estim_path` (.npz files).
        - Simulates autonomous trajectories x_{t+1} = A x_t and Ahat x_t
          from identical initial states and overlays them.
        - Plots eigenvalues on the complex plane, singular values, and heatmaps.

        Parameters
        ----------
        estim_path : str | None
            Path to .npz containing Ahat (and optional Buhat, etc.).
            Defaults to <csv_path base>_estmMat.npz.
        truth_path : str | None
            Path to .npz containing A (and optional Bu, etc.).
            Defaults to <csv_path base>_trueMat.npz.
        nsteps : int
            Number of steps for the autonomous simulation.
        x0_mode : {"e1","ones","random"}
            Initial state used for unit response style comparison.
        seed : int
            RNG seed when x0_mode == "random".
        save_dir : str | None
            Directory where figures are saved as PNG. Defaults next to csv.
        show : bool
            Whether to call plt.show() at the end.

        Returns
        -------
        dict
            A dictionary of basic metrics for programmatic checks.
        """
        # Resolve default paths
        base = Path(self.csv_path).with_suffix("")
        estim_path = Path(self.estim_path)
        truth_path = Path(self.truth_path)

        if not estim_path.exists():
            raise FileNotFoundError(f"Estimated matrices file not found: {estim_path}")
        if not truth_path.exists():
            raise FileNotFoundError(f"Ground-truth matrices file not found: {truth_path}")

        E = np.load(estim_path)
        T = np.load(truth_path)
        if "Ahat" not in E:
            raise KeyError(f"'Ahat' not found in {estim_path}")
        if "A" not in T:
            raise KeyError(f"'A' not found in {truth_path}")

        Ahat = E["Ahat"]
        Atru = T["A"]
        if Ahat.shape != Atru.shape:
            raise ValueError(f"Shape mismatch: Ahat {Ahat.shape} vs A {Atru.shape}")
        nx = Atru.shape[0]

        # initial condition
        rng = np.random.default_rng(seed)
        if x0_mode == "e1":
            x0 = np.zeros(nx); x0[0] = 1.0
        elif x0_mode == "ones":
            x0 = np.ones(nx) / np.sqrt(nx)
        elif x0_mode == "random":
            x0 = rng.normal(0, 1, size=nx)
            nrm = np.linalg.norm(x0)
            if nrm > 0:
                x0 = x0 / nrm
        else:
            raise ValueError("x0_mode must be 'e1', 'ones', or 'random'")

        # simulate autonomous responses
        def sim(A, x0, Tn):
            X = np.zeros((nx, Tn))
            X[:, 0] = x0
            for t in range(Tn - 1):
                X[:, t + 1] = A @ X[:, t]
            return X

        Xtru = sim(Atru, x0, nsteps)
        Xhat = sim(Ahat, x0, nsteps)
        t = np.arange(nsteps)

        # metrics
        fro_A = float(np.linalg.norm(Atru, "fro"))
        fro_diff = float(np.linalg.norm(Ahat - Atru, "fro"))
        rel_err = float(fro_diff / (fro_A + 1e-12))
        traj_rel = float(np.linalg.norm(Xhat - Xtru) / (np.linalg.norm(Xtru) + 1e-12))
        rho_true = float(max(abs(np.linalg.eigvals(Atru))))
        rho_hat  = float(max(abs(np.linalg.eigvals(Ahat))))
        metrics = {
            "fro_rel_error": rel_err,
            "traj_rel_error": traj_rel,
            "rho_true": rho_true,
            "rho_hat": rho_hat,
        }
        print("[A vs Ahat] metrics:", metrics)

        # 1) trajectories overlay
        plt.figure(figsize=(9, 5))
        for i in range(nx):
            plt.plot(t, Xtru[i], linestyle="-", label="true" if i == 0 else None)
            plt.plot(t, Xhat[i], linestyle="--", label="est" if i == 0 else None)
        plt.title(f"Autonomous response from x0='{x0_mode}'")
        plt.xlabel("t"); plt.ylabel("state"); plt.grid(True, alpha=0.3); plt.legend()

        # 2) error norm over time
        err = np.linalg.norm(Xhat - Xtru, axis=0)
        plt.figure(figsize=(8, 4))
        plt.plot(t, err, marker="o")
        plt.title("State error norm over time  ||x̂_t − x_t||₂")
        plt.xlabel("t"); plt.ylabel("error norm"); plt.grid(True, alpha=0.3)

        # 3) eigenvalues
        evals_true = np.linalg.eigvals(Atru)
        evals_hat  = np.linalg.eigvals(Ahat)
        theta = np.linspace(0, 2*np.pi, 400)
        plt.figure(figsize=(6, 6))
        plt.plot(np.cos(theta), np.sin(theta))                 # unit circle
        plt.scatter(evals_true.real, evals_true.imag, marker="o", label="eig(A)")
        plt.scatter(evals_hat.real,  evals_hat.imag,  marker="x", label="eig(Ahat)")
        plt.axhline(0, linewidth=0.8); plt.axvline(0, linewidth=0.8)
        plt.gca().set_aspect("equal", adjustable="box")
        plt.title("Eigenvalues"); plt.grid(True, alpha=0.3); plt.legend()

        if show:
            plt.show()
        else:
            plt.close("all")

        return metrics


## ------------------------------ MAIN ENTRY POINT ---------------------------------

if __name__ == "__main__":
    yaml_path="problem___parameters.yaml"

    CL = False
    OL = True

    if CL: Closed_Loop()
    if OL: Open_Loop(MAKE_DATA=True, EVAL_FROM_PATH=True, PLOT=True, yaml_path=yaml_path)


