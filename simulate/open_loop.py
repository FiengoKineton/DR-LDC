#!/usr/bin/env python3
import argparse, csv
import numpy as np
from pathlib import Path
import matplotlib.pyplot as plt


from config import get_cfg                              # loader.py
from disturbances import Disturbances                   # disturbances.py
from utils import Plant                                 # systems.py

from .initial_conditions import _initial_condition_from_eigenvalues


# =============================================================================================== #

class Open_Loop():
    def __init__(self, MAKE_DATA=True, EVAL_FROM_PATH=True, DATASETS=False, gamma: float = None, p: bool = False, x0_mode: str = None, s: bool = None, N: int = None):
        
        cfg = get_cfg()
        p = cfg.get("params", {})
        self.p = p
        out = self.p.get("directories", {}).get("data", "./out/data/session_")
        m = self.p.get("ambiguity", {}).get("model", "W2")
        runID = self.p.get("directories", {}).get("runID", ".temp")
        _type = self.p.get("plant", {}).get("type", "explicit")
        _model = self.p.get("model", "independent") if m == "W2" else m
        PLOT = bool(self.p.get("plot", "false")) if not p else p
        
        var = float(p.get("ambiguity", {})["var"])
        n = p.get("dimensions", {}).get("nw", 2)
        Sigma_nom = np.array(p.get("ambiguity", {})["Sigma_nom"], dtype=float) if m!="Gaussian" else var * np.eye(n)
        
        self.csv_path = out + f"{runID}___{_type}_{_model}.csv"    # _{_data}

        self.out = Path(self.csv_path)
        self.out.parent.mkdir(parents=True, exist_ok=True)
        self.estim_path = self.out.with_suffix("").as_posix() + "_estmMat.npz"
        self.truth_path = self.out.with_suffix("").as_posix() + "_trueMat.npz"

        self.ts = self.p.get("simulation", {}).get("ts", 0.05)
        self.Tf = self.p.get("simulation", {}).get("TotTime", 0.05)

        from core import MatricesAPI
        api = MatricesAPI()
        plant, _ = api.get_system(FROM_DATA=False, gamma=gamma)


        if DATASETS:
            self.datasets = self.make_multiple_data(plant=plant, gamma=gamma, Sigma=Sigma_nom, N=N if N is not None else 5)
        else: 
            if MAKE_DATA: self.data = self.make_data(plant=plant, gamma=gamma, Sigma=Sigma_nom)
            if EVAL_FROM_PATH: self.evaluate_from_path()
            if PLOT: 
                metrics = self.plot_est_vs_truth(x0_mode="e1" if x0_mode is None else x0_mode, show=True if s is None else s)
                print("\n[PLT] Plotting completed. Metrics:", metrics)
    
    # ------------------------------------------------------------------------------------------- #

    def make_multiple_data(self, plant: Plant, N: int = 5, gamma: float = None, Sigma: np.ndarray = None):
        datasets = []
        init = ["rand"]#, "zeros"]
        input = ["multisine"]#, "prbs"]
        for i in range(N):
            print(f"[DATA] Generating dataset {i+1}/{N}...")
            data = self.make_data(plant=plant, gamma=gamma, Sigma=Sigma, multiple_datasets=True, init=init[i % len(init)], input=input[i % len(input)])
            if data["PE"]: 
                if N==1: 
                    datasets = {
                        "X_next": data["X_next"],
                        "X": data["X"][:, :-1],
                        "Y": data["Y"][:, :-1],
                        "Z": data["Z"][:, :-1],
                        "U": data["U"][:, :-1],
                    }
                    break
                else:
                    datasets.append(data)
            else: 
                print(f"[DATA] Dataset {i+1} failed PE check. Regenerating...")
                i -= 1
        return datasets

    def make_data(self, plant: Plant, gamma: float = None, Sigma: np.ndarray = None, multiple_datasets: bool = False, init: str = "zeros", input: str = "multisine"):
        ap = argparse.ArgumentParser(description="Simulate open-loop with persistently exciting input and save CSV.")
        ap.add_argument("--T", type=int, default=3000, help="number of time steps")
        ap.add_argument("--seed", type=int, default=0)
        ap.add_argument("--input", choices=["multisine", "prbs"], default="multisine")
        ap.add_argument("--amp", type=float, default=1.0, help="input amplitude")
        ap.add_argument("--w_std", type=float, default=0.1, help="disturbance std scaling")
        ap.add_argument("--delimiter", type=str, default=",")
        args = ap.parse_args()


        T = args.T
        rng = np.random.default_rng(args.seed)

        wass = Disturbances(gamma=gamma, n=Sigma[0].size, var=1)
        W = wass.sample(T=T, Sigma=Sigma).T
        #api = MatricesAPI()
        #plant, _ = api.get_system(Generating_data=True)
        A, Bu, Bw, Cz, Dzw, Dzu, Cy, Dyw = plant.A, plant.Bu, plant.Bw, plant.Cz, plant.Dzw, plant.Dzu, plant.Cy, plant.Dyw
        #A, Bu, Bw = api.build_AB_from_yaml()
        #Cz, Dzw, Dzu, Cy, Dyw = api.build_out_matrices()
        #nx, nw, nu, ny, nz = api.get_dimensions_from_yaml()
        nx = A.shape[0]
        nw = Bw.shape[1]
        nu = Bu.shape[1]
        nz = Cz.shape[0]
        ny = Cy.shape[0]

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
            # rng = np.random.default_rng(seed); W = rng.normal(0.0, 1.0, size=(nw, T)) * w_std

            if init == "zeros":         X = np.zeros((nx, T))
            elif init == "from_eig":    X = _initial_condition_from_eigenvalues(A)
            else:                       X = rng.normal(0, 1.0, size=(nx, T))

            X[:, 0] = x0
            for t in range(T - 1):
                X[:, t + 1] = (A @ X[:, t] + Bu @ U[:, t] + Bw @ W[:, t])
            return X

        def synth_outputs_with_mats(X, U):
            """
            Returns Y, Z plus the exact matrices used to generate them:
            Cy, Dyw, Cz, Dzu, Dzw.
            Shapes:
            X: (nx,T), U: (nu,T), R: (nx,T-1)
            Y: (ny,T-1), Z: (nz,T-1)
            """

            Y = Cy @ X[:, :-1] + Dyw @ W[:, :-1]
            Z = Cz @ X[:, :-1] + Dzu @ U[:, :-1] + Dzw @ W[:, :-1]

            return Y, Z



        # Persistently exciting input
        input_mode = args.input if not multiple_datasets else input
        if input_mode == "prbs":
            U = prbs(nu, T, shift=11, seed=args.seed + 17, amp=args.amp)
        else:
            U = multisine(nu, T, ntones=12, seed=args.seed + 23, amp=args.amp)

        # Simulate
        x0 = rng.normal(0, 1.0, size=nx)
        X = simulate_open_loop(A, Bu, Bw, T, x0, U, w_std=args.w_std, seed=args.seed + 101)

        # One-step alignment and residual proxy R = X+ - AX - BU
        X_reg = X[:, :-1]       # x_0..x_{T-2}
        X_next = X[:, 1:]       # x_1..x_{T-1}
        R = X_next - (A @ X_reg + Bu @ U[:, :-1])  # nx x (T-1)

        # Outputs Y,Z and the exact matrices used to generate them
        Y, Z = synth_outputs_with_mats(X, U)

        # PE sanity check
        cond, svals = pe_check(X_reg, U[:, :-1])
        print(f"\t[PE] cond( D D^T ) = {cond:.2e}   rank={np.sum(svals>1e-10)}/{nx+nu}")
        if np.sum(svals>1e-10) == nx + nu:
            PE = True
        else:
            PE = False
            print("[PE] Warning: input is not persistently exciting of order nx+nu.")

        # ----------------------- build dataset dict -----------------------
        # Align Y,Z to length T by repeating the last column (keeps your current semantics)
        Y_aligned = Y if Y.shape[1] == T else np.hstack([Y, Y[:, [-1]]])
        Z_aligned = Z if Z.shape[1] == T else np.hstack([Z, Z[:, [-1]]])

        t = np.arange(T, dtype=float)

        data = {
            "X": X,                     # shape (nx, T)
            "U": U,                     # shape (nu, T)
            "Y": Y_aligned,             # shape (ny, T)
            "Z": Z_aligned,             # shape (nz, T)
            "X_reg": X_reg,             # shape (nx, T-1)
            "U_reg": U[:, :-1],         # shape (nu, T-1)
            "X_next": X_next,           # shape (nx, T-1)
            "W": W,                     # shape (nw, T)
            "R": R,                     # residual proxy, shape (nx, T-1)
            "t": t,                     # time index
            "meta": {
                "nx": nx, "nu": nu, "nw": nw, "ny": ny, "nz": nz, "T": T,
                "seed": args.seed, "input": input_mode, "amp": args.amp, "w_std": args.w_std,
                "csv_path": self.out, "truth_path": self.truth_path, "init": init,
            }, 
            "PE": PE,
        }

        # Save CSV

        headers = []
        for i in range(nx): headers.append(f"x{i+1}")
        for j in range(nu): headers.append(f"u{j+1}")
        for k in range(ny): headers.append(f"y{k+1}")
        for k in range(nz): headers.append(f"z{k+1}")

        rows = []
        for t in range(T):
            row = []
            row.extend(X[:, t].tolist())
            row.extend(U[:, t].tolist())
            if t < T-1:
                row.extend(Y[:, t].tolist())
                row.extend(Z[:, t].tolist())
            else:
                row.extend(Y[:, -1].tolist())
                row.extend(Z[:, -1].tolist())
            rows.append(row)


        if not multiple_datasets:
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
                seed=args.seed, input=input_mode, amp=args.amp, w_std=args.w_std
            )
            print(f"[OK] Saved ground-truth matrices to {self.truth_path}")

        data["headers"] = headers
        data["rows"] = np.asarray(rows)
        return data

    # ------------------------------------------------------------------------------------------- #

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
        from core import MatricesAPI
        api = MatricesAPI()
        nx, nw, nu, ny, nz = api.get_dimensions_from_yaml()

        plant_est, _ = api.make_matrices_from_data(
            delimiter=delimiter,
            ridge=ridge,
            eval=True,
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
            return plant_est

        truth_npz = str(truth_npz)
        if not Path(truth_npz).exists():
            print(f"\n[WARN] truth_npz file not found: {truth_npz}. Skipping comparisons.")
            return plant_est

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


        return plant_est

    def plot_est_vs_truth(self, x0_mode="e1", seed=0, show=True):
        """
        DT diagnostic:
        - Simulate x_{k+1} = A x_k and x_{k+1} = Ahat x_k from same x0.
        - One subplot per state for trajectories.
        - Eigenvalues on unit circle (DT).
        - Matrix comparison figure: for every pair (K, Khat) found in the .npz files,
        draw heatmaps [K_true, K_est, K_est - K_true] and run a shape check.
        Returns metrics and a dict of dimension checks.
        """

        nsteps = int(self.Tf / self.ts)
        estim_path = Path(self.estim_path)
        truth_path = Path(self.truth_path)
        if not estim_path.exists():
            raise FileNotFoundError(f"Estimated matrices file not found: {estim_path}")
        if not truth_path.exists():
            raise FileNotFoundError(f"Ground-truth matrices file not found: {truth_path}")

        E = np.load(estim_path)
        T = np.load(truth_path)

        # Required pair for trajectories
        if "Ahat" not in E:
            raise KeyError(f"'Ahat' not found in {estim_path}")
        if "A" not in T:
            raise KeyError(f"'A' not found in {truth_path}")

        Ahat = E["Ahat"]
        Atru = T["A"]
        if Ahat.shape != Atru.shape:
            raise ValueError(f"Shape mismatch: Ahat {Ahat.shape} vs A {Atru.shape}")
        nx = Atru.shape[0]

        # Initial condition
        rng = np.random.default_rng(seed)
        if x0_mode == "e1":
            x0 = np.zeros(nx); x0[0] = 1.0
        elif x0_mode == "ones":
            x0 = np.ones(nx) / np.sqrt(nx)
        elif x0_mode == "random":
            x0 = rng.normal(0, 1, size=nx)
            nrm = np.linalg.norm(x0);  x0 = x0 / (nrm + 1e-15)
        else:
            raise ValueError("x0_mode must be 'e1', 'ones', or 'random'")

        # DT simulation x_{k+1} = A x_k
        def sim(A, x0, Tn, cap=1e8):
            X = np.zeros((nx, Tn))
            X[:, 0] = x0
            for t in range(Tn - 1):
                X[:, t + 1] = A @ X[:, t]
                if not np.isfinite(X[:, t + 1]).all() or np.linalg.norm(X[:, t + 1]) > cap:
                    return X[:, :t+2]
            return X

        Xtru = sim(Atru, x0, nsteps)
        Xhat = sim(Ahat, x0, nsteps)
        Tlen = min(Xtru.shape[1], Xhat.shape[1])
        t = np.arange(Tlen)*self.ts

        # Metrics
        fro_A = float(np.linalg.norm(Atru, "fro"))
        fro_diff = float(np.linalg.norm(Ahat - Atru, "fro"))
        rel_err = float(fro_diff / (fro_A + 1e-12))
        traj_rel = float(
            np.linalg.norm(Xhat[:, :Tlen] - Xtru[:, :Tlen])
            / (np.linalg.norm(Xtru[:, :Tlen]) + 1e-12)
        )
        rho_true = float(np.max(np.abs(np.linalg.eigvals(Atru))))
        rho_hat  = float(np.max(np.abs(np.linalg.eigvals(Ahat))))
        metrics = {
            "fro_rel_error": rel_err,
            "traj_rel_error": traj_rel,
            "rho_true": rho_true,
            "rho_hat": rho_hat,
        }
        print("[A vs Ahat] metrics:", metrics)

        # 1) Trajectories: one subplot per state
        fig_traj, axs = plt.subplots(nx, 1, figsize=(9, max(3, 1.6*nx)), sharex=True)
        if nx == 1:
            axs = [axs]
        for i in range(nx):
            axs[i].plot(t, Xtru[i, :Tlen], "-",  label="true")
            axs[i].plot(t, Xhat[i, :Tlen], "--", label="est")
            axs[i].grid(alpha=0.3)
            axs[i].set_ylabel(f"x[{i}]")
            if i == 0:
                ttl = f"Autonomous DT response from x0='{x0_mode}'"
                if rho_true >= 1 or rho_hat >= 1:
                    ttl += "  [unstable eigenvalues detected]"
                axs[i].set_title(ttl)
        axs[-1].set_xlabel("time") #k (steps)")
        axs[0].legend()

        # 2) Error norm over time
        err = np.linalg.norm(Xhat[:, :Tlen] - Xtru[:, :Tlen], axis=0)
        plt.figure(figsize=(8, 4))
        plt.plot(np.arange(Tlen)*self.ts, err, marker="o")
        plt.title("State error norm over time  ||x̂_k − x_k||₂")
        plt.xlabel("time") #k (steps)")
        plt.ylabel("error norm"); plt.grid(alpha=0.3)

        # 3) Eigenvalues on unit circle (DT)
        evals_true = np.linalg.eigvals(Atru)
        evals_hat  = np.linalg.eigvals(Ahat)
        theta = np.linspace(0, 2*np.pi, 600)
        circle = np.c_[np.cos(theta), np.sin(theta)]
        plt.figure(figsize=(6, 6))
        plt.plot(circle[:,0], circle[:,1], linewidth=1.0, label="unit circle")
        plt.scatter(evals_true.real, evals_true.imag, marker="o", label="eig(A)")
        plt.scatter(evals_hat.real,  evals_hat.imag,  marker="x", label="eig(Â)")
        lim = max(1.1, np.max(np.abs(np.r_[evals_true.real, evals_true.imag,
                                        evals_hat.real,  evals_hat.imag, 1.0])))*1.05
        plt.xlim([-lim, lim]); plt.ylim([-lim, lim])
        plt.axhline(0, linewidth=0.8); plt.axvline(0, linewidth=0.8)
        plt.gca().set_aspect("equal", adjustable="box")
        plt.title("DT eigenvalues"); plt.grid(alpha=0.3); plt.legend()

        # 4) Matrix comparison figure (true vs est vs diff) for all matched pairs
        # Pair any key 'Khat' in E with 'K' in T
        pairs = []
        for k_est in E.files:
            if not k_est.endswith("hat"):
                continue
            k_true = k_est[:-3]
            if k_true in T.files:
                pairs.append((k_true, k_est))

        dim_checks = {}
        if pairs:
            nrows = len(pairs)
            fig, axes = plt.subplots(nrows, 3, figsize=(12, max(3.0, 2.2*nrows)))
            if nrows == 1:
                axes = np.array([axes])
            for i, (ktru, kest) in enumerate(pairs):
                M_true = T[ktru]
                M_est  = E[kest]
                same_shape = (M_true.shape == M_est.shape)
                dim_checks[(ktru, kest)] = {
                    "true_shape": tuple(M_true.shape),
                    "est_shape":  tuple(M_est.shape),
                    "match": bool(same_shape),
                }
                if not same_shape:
                    # scream into the plot so future-you notices
                    fig.suptitle("WARNING: shape mismatches detected", color="crimson")
                # heatmaps
                ax1, ax2, ax3 = axes[i, 0], axes[i, 1], axes[i, 2]
                im1 = ax1.imshow(M_true, aspect='auto'); ax1.set_title(f"{ktru} (true)"); ax1.grid(False)
                im2 = ax2.imshow(M_est,  aspect='auto'); ax2.set_title(f"{kest} (est)");  ax2.grid(False)
                # difference uses a symmetric colormap around 0
                diff = M_est - M_true if same_shape else np.zeros_like(M_true)
                vmax = np.max(np.abs(diff)) + 1e-12
                im3 = ax3.imshow(diff, vmin=-vmax, vmax=vmax, aspect='auto'); ax3.set_title(f"{kest} − {ktru}"); ax3.grid(False)
                for ax in (ax1, ax2, ax3):
                    ax.set_xlabel("cols"); ax.set_ylabel("rows")
                # optional colorbars (comment out if you hate margins)
                fig.colorbar(im1, ax=ax1, fraction=0.046, pad=0.04)
                fig.colorbar(im2, ax=ax2, fraction=0.046, pad=0.04)
                fig.colorbar(im3, ax=ax3, fraction=0.046, pad=0.04)
            fig.tight_layout()
        else:
            print("No matrix pairs found of the form (K in truth, KhAT in estim). Add keys like 'Bu' and 'Buhat' if you want them compared.")

        if show:
            plt.show()
        else:
            plt.close("all")

        # include dimension checks in return for programmatic gating
        out = {"metrics": metrics, "dimension_checks": dim_checks}
        return out

# =============================================================================================== #