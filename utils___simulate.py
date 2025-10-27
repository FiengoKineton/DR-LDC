#!/usr/bin/env python3
import argparse, csv, yaml, sys
import numpy as np
from pathlib import Path
import matplotlib.pyplot as plt
from typing import Optional, Dict, Any, Iterable

#from utils___matrices import MatricesAPI
from utils___systems import Plant, Controller, Plant_cl
from utils___ambiguity import Disturbances


yaml_path="problem___parameters.yaml"
if yaml is None:
    raise ImportError("PyYAML not available. Install with `pip install pyyaml`.")
with open(yaml_path, "r", encoding="utf-8") as f:
    cfg = yaml.safe_load(f)


## ------------------------- CLOSED-LOOP SIMULATION CLASS --------------------------

class Closed_Loop():
    def __init__(self, TEST=False):
        self.p = cfg.get("params", {})
        sim = self.p.get("simulation", {})
        self.Tf = sim.get("TotTime", 25)
        self.ts = sim.get("ts", 0.5)

        if TEST: self.test()
    
    def test(self):
        # Use the same plant as the optimization example (seed=7)
        from utils___matrices import MatricesAPI
        api = MatricesAPI()
        plant, ctrl = api.get_system()

        """Ac = np.array([
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
        Sigma_w = 0.7 * np.eye(2)"""

        sim = self.simulate_closed_loop(plant, ctrl)
        print("Simulated shapes:",
            {k: v.shape for k, v in sim.items() if isinstance(v, np.ndarray)})

        out = self.save_npz(sim, "closed_loop_run_seed11_T800.npz")
        print(f"Saved time series to {out}")

        self.plot_timeseries(sim)

    def simulate_closed_loop(self, plant: Plant,
                            ctrl: Controller,
                            Sigma_w: np.ndarray = None,
                            gamma: float = None,
                            seed: int = 11,
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
        T = int(self.Tf / self.ts)
        x = np.zeros((nx, 1)) #if x0 is None else np.asarray(x0, float).reshape(nx, 1)
        xc = np.zeros((nxc, 1)) #if xc0 is None else np.asarray(xc0, float).reshape(nxc, 1)

        # Precompute a sampling factor for w
        if Sigma_w is not None:
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

        wass = Disturbances(gamma=gamma)
        W = wass.sample(T=T)
        if W.ndim == 1:
            W = W.reshape(T, 1)

        if W.shape[1] == nw:
            # Great. Use directly.
            pass
        else:
            # Case 2: mismatch. DO NOT "average columns".
            # Sample *state-space* disturbance then project onto span(Bw_hat).
            # Build a target state covariance. Easiest defensible choice:
            #   Sigma_state = Bw_hat Bw_hat^T (+ tiny isotropic pad).
            eps = 1e-6
            Sigma_state = Bw @ Bw.T + eps * np.eye(nx)

            # Sample state disturbances with that covariance (shape: T x nx)
            D = wass.sample(T=T, Sigma=Sigma_state)

            # Project each d_t onto col(Bw_hat) via pseudoinverse
            Bw_pinv = np.linalg.pinv(Bw)     # (nw_hat x nx)
            W = (D @ Bw_pinv.T)                  # (T x nw_hat)
        
        for t in range(T):
            w = W[t, :].reshape(nw, 1) # (L @ rng.standard_normal((nw, 1))).astype(float)
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
        t = np.arange(T) * self.ts

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

    def simulate_composite(self, Pcl: Plant_cl, gamma: float) -> Dict[str, Any]:
        """
        Simulate the composed system:
            X_{t+1} = Acl X_t + Bcl w_t
            z_t     = Ccl X_t + Dcl w_t
        with X_0 = 0 (or provided X0).

        Args:
            Pcl: Plant_cl with (Acl,Bcl,Ccl,Dcl)
            W: disturbance trajectory (T, nw). If provided, overrides Sigma_w.
            Sigma_w: covariance of w_t (nw,nw), i.i.d. Gaussian if W is None.
            seed: RNG seed when sampling w_t.
            X0: initial state (nX,) or (nX,1). Defaults to zeros.

        Returns dict with:
            'X' (T, nX), 'Z' (T, nz), 'W' (T, nw),
            sizes and timing metadata.
        """

        A = np.asarray(Pcl.Acl, dtype=float)
        B = np.asarray(Pcl.Bcl, dtype=float)
        C = np.asarray(Pcl.Ccl, dtype=float)
        D = np.asarray(Pcl.Dcl, dtype=float)

        nX = A.shape[0]
        nw = B.shape[1]
        nz = C.shape[0]

        # Horizon
        T = int(round(self.Tf / self.ts))
        if T <= 0:
            raise ValueError("Non-positive simulation horizon. Check Tf and ts.")

        # Initial condition
        x = np.zeros((nX, 1))

        # Disturbance generation
        wass = Disturbances(gamma=gamma)
        W = wass.sample(T=T)
        if W.ndim == 1:
            W = W.reshape(T, 1)

        if W.shape[1] == nw:
            # Great. Use directly.
            pass
        else:
            # Case 2: mismatch. DO NOT "average columns".
            # Sample *state-space* disturbance then project onto span(Bw_hat).
            # Build a target state covariance. Easiest defensible choice:
            #   Sigma_state = Bw_hat Bw_hat^T (+ tiny isotropic pad).
            eps = 1e-6
            Sigma_state = B @ B.T + eps * np.eye(nX)

            # Sample state disturbances with that covariance (shape: T x nx)
            d = wass.sample(T=T, Sigma=Sigma_state)

            # Project each d_t onto col(Bw_hat) via pseudoinverse
            B_pinv = np.linalg.pinv(B)     # (nw_hat x nx)
            W = (d @ B_pinv.T)                  # (T x nw_hat)

        # Storage
        X = np.zeros((T, nX))
        Z = np.zeros((T, nz))

        # Rollout
        for t in range(T):
            w = W[t, :].reshape(nw, 1)
            z = C @ x + D @ w
            X[t, :] = x.ravel()
            Z[t, :] = z.ravel()
            x = A @ x + B @ w

        return {
            "X": X,               # (T, nX)
            "Z": Z,               # (T, nz)
            "W": W,               # (T, nw)
            "T": T, "ts": self.ts,
            "nX": nX, "nw": nw, "nz": nz,
        }

    def plot_composite(self,
                        sim: dict,
                        show_X: bool = True,
                        show_Z: bool = True,
                        show_W: bool = False,
                        X_idx: Optional[Iterable[int]] = None,
                        Z_idx: Optional[Iterable[int]] = None,
                        W_idx: Optional[Iterable[int]] = None,
                        suptitle: Optional[str] = None):
        """
        Plot time series for the composed closed-loop simulation.

        Args:
            sim: dict from simulate_composite()
                 must contain keys 'X', 'Z', 'W', 'T', 'ts', 'nX','nz','nw'
            show_X, show_Z, show_W: enable/disable each panel
            *_idx: optional iterable of component indices to plot; defaults to all
            suptitle: optional overall title

        Behavior:
            - x-axis is physical time (k * ts)
            - each signal gets its own figure to avoid clutter
        """
        T   = int(sim["T"])
        ts  = float(sim["ts"])
        t   = np.arange(T) * ts

        def _ensure_indices(n: int, sel: Optional[Iterable[int]]):
            if sel is None:
                return list(range(n))
            # validate and unique-preserve order
            idx = list(dict.fromkeys(int(i) for i in sel))
            bad = [i for i in idx if i < 0 or i >= n]
            if bad:
                raise IndexError(f"indices {bad} out of range [0, {n-1}]")
            return idx

        if show_X:
            X = np.asarray(sim["X"])
            nX = int(sim["nX"])
            xi = _ensure_indices(nX, X_idx)
            plt.figure(figsize=(10, 6))
            for i in xi:
                plt.plot(t, X[:, i], label=f"X[{i}]")
            ttl = "Composite state X(t)"
            if suptitle: ttl = f"{suptitle} — {ttl}"
            plt.title(ttl)
            plt.xlabel("time")
            plt.ylabel("state")
            plt.grid(True, alpha=0.3)
            if len(xi) <= 12:
                plt.legend(ncol=2, framealpha=0.8)

        if show_Z:
            Z = np.asarray(sim["Z"])
            nz = int(sim["nz"])
            zi = _ensure_indices(nz, Z_idx)
            plt.figure(figsize=(10, 6))
            for i in zi:
                plt.plot(t, Z[:, i], label=f"z[{i}]")
            ttl = "Performance output z(t)"
            if suptitle: ttl = f"{suptitle} — {ttl}"
            plt.title(ttl)
            plt.xlabel("time")
            plt.ylabel("output")
            plt.grid(True, alpha=0.3)
            if len(zi) <= 12:
                plt.legend(ncol=2, framealpha=0.8)

        if show_W:
            W = np.asarray(sim["W"])
            nw = int(sim["nw"])
            wi = _ensure_indices(nw, W_idx)
            plt.figure(figsize=(10, 5))
            for i in wi:
                plt.plot(t, W[:, i], label=f"w[{i}]")
            ttl = "Disturbance w(t)"
            if suptitle: ttl = f"{suptitle} — {ttl}"
            plt.title(ttl)
            plt.xlabel("time")
            plt.ylabel("disturbance")
            plt.grid(True, alpha=0.3)
            if len(wi) <= 12:
                plt.legend(ncol=2, framealpha=0.8)

        plt.show()



## ------------------------- OPEN-LOOP SIMULATION CLASS ----------------------------

class Open_Loop():
    def __init__(self, MAKE_DATA=True, EVAL_FROM_PATH=True, gamma: float = None, p: bool = None, x0_mode: str = None, s: bool = None):
        self.p = cfg.get("params", {})
        out = self.p.get("directories", {}).get("data", "./out/data/session_")
        m = self.p.get("ambiguity", {}).get("model", "W2")
        runID = self.p.get("directories", {}).get("runID", ".temp")
        _type = self.p.get("plant", {}).get("type", "explicit")
        _model = self.p.get("model", "independent") if m == "W2" else m
        PLOT = bool(self.p.get("plot", "false")) if p is None else p

        self.csv_path = out + f"{runID}___{_type}_{_model}.csv"    # _{_data}

        self.out = Path(self.csv_path)
        self.out.parent.mkdir(parents=True, exist_ok=True)
        self.estim_path = self.out.with_suffix("").as_posix() + "_estmMat.npz"
        self.truth_path = self.out.with_suffix("").as_posix() + "_trueMat.npz"

        self.ts = self.p.get("simulation", {}).get("ts", 0.05)
        self.Tf = self.p.get("simulation", {}).get("TotTime", 0.05)

        from utils___matrices import MatricesAPI
        api = MatricesAPI()
        plant, _ = api.get_system(FROM_DATA=False, gamma=gamma)

        if MAKE_DATA: self.make_data(plant=plant, gamma=gamma)
        if EVAL_FROM_PATH: self.evaluate_from_path()
        if PLOT: 
            metrics = self.plot_est_vs_truth(x0_mode="e1" if x0_mode is None else x0_mode, show=True if s is None else s)
            print("\n[PLT] Plotting completed. Metrics:", metrics)

    
    """
    python simulate_pe_openloop.py --T 3000 --nx 4 --nu 2 --nw 2 --ny 2 --nz 3 --input multisine --amp 1.0 --w_std 0.1 --seed 42 --out out/data/session01.csv
    """


    def make_data(self, plant: Plant, gamma: float = None):
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

        wass = Disturbances(gamma=gamma)
        W = wass.sample(T=T).T
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

            X = np.zeros((nx, T))
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
        if args.input == "prbs":
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
        print(f"[PE] cond( D D^T ) = {cond:.2e}   rank={np.sum(svals>1e-10)}/{nx+nu}")

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
        from utils___matrices import MatricesAPI
        api = MatricesAPI()
        nx, nw, nu, ny, nz = api.get_dimensions_from_yaml()

        plant_est, _ = api.make_matrices_from_data(
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
        t = np.arange(Tlen)

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
        axs[-1].set_xlabel("k (steps)")
        axs[0].legend()

        # 2) Error norm over time
        err = np.linalg.norm(Xhat[:, :Tlen] - Xtru[:, :Tlen], axis=0)
        plt.figure(figsize=(8, 4))
        plt.plot(np.arange(Tlen), err, marker="o")
        plt.title("State error norm over time  ||x̂_k − x_k||₂")
        plt.xlabel("k (steps)"); plt.ylabel("error norm"); plt.grid(alpha=0.3)

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


## ------------------------------ MAIN ENTRY POINT ---------------------------------

if __name__ == "__main__":
    CL = 0
    OL = 1

    m = ["e1", "ones", "random"]
    if CL: Closed_Loop(TEST=False)
    if OL: Open_Loop(MAKE_DATA=1, EVAL_FROM_PATH=1, p=True, x0_mode=m[0], s=False)

    """
    PaperLike
        NoiseFree
            e1
                [CMP] Comparison of estimates vs ground-truth:
                [cmp]   Ahat vs A       rel=1.044e-05
                [cmp]  Buhat vs Bu      rel=1.135e-07
                [cmp]  Bwhat vs Bw      subspace_gap=0.000e+00  rel=1.000e+00       
                [cmp]  Cyhat vs Cy      rel=3.233e-08
                [cmp] Dywhat vs Dyw     abs=3.284e-07
                [cmp]  Czhat vs Cz      rel=4.080e-06
                [cmp] Dzuhat vs Dzu     rel=1.097e-07
                [cmp] Dzwhat vs Dzw     abs=3.210e-04
                [PLT] Plotting completed. Metrics: {'metrics': {'fro_rel_error': 1.0439859481910079e-05, 'traj_rel_error': 1.5464040578905646e-05, 'rho_true': 0.9998750394917907, 'rho_hat': 0.9998750118201665}, 'dimension_checks': {('A', 'Ahat'): {'true_shape': (7, 7), 'est_shape': (7, 7), 'match': True}, ('Bu', 'Buhat'): {'true_shape': (7, 1), 'est_shape': (7, 1), 'match': True}, ('Bw', 'Bwhat'): {'true_shape': (7, 2), 'est_shape': (7, 7), 'match': False}, ('Cy', 'Cyhat'): {'true_shape': (3, 7), 'est_shape': (3, 7), 'match': True}, ('Dyw', 'Dywhat'): {'true_shape': (3, 2), 'est_shape': (3, 7), 'match': False}, ('Cz', 'Czhat'): {'true_shape': (8, 7), 'est_shape': (8, 7), 'match': True}, ('Dzu', 'Dzuhat'): {'true_shape': (8, 1), 'est_shape': (8, 1), 'match': True}, ('Dzw', 'Dzwhat'): {'true_shape': (8, 2), 'est_shape': (8, 7), 'match': False}}}
            ones
                [CMP] Comparison of estimates vs ground-truth:
                [cmp]   Ahat vs A       rel=1.044e-05
                [cmp]  Buhat vs Bu      rel=1.135e-07
                [cmp]  Bwhat vs Bw      subspace_gap=0.000e+00  rel=1.000e+00       
                [cmp]  Cyhat vs Cy      rel=3.233e-08
                [cmp] Dywhat vs Dyw     abs=3.284e-07
                [cmp]  Czhat vs Cz      rel=4.080e-06
                [cmp] Dzuhat vs Dzu     rel=1.097e-07
                [cmp] Dzwhat vs Dzw     abs=3.210e-04
                [PLT] Plotting completed. Metrics: {'metrics': {'fro_rel_error': 1.0439859481910079e-05, 'traj_rel_error': 0.00011236165770693395, 'rho_true': 0.9998750394917907, 'rho_hat': 0.9998750118201665}, 'dimension_checks': {('A', 'Ahat'): {'true_shape': (7, 7), 'est_shape': (7, 7), 'match': True}, ('Bu', 'Buhat'): {'true_shape': (7, 1), 'est_shape': (7, 1), 'match': True}, ('Bw', 'Bwhat'): {'true_shape': (7, 2), 'est_shape': (7, 7), 'match': False}, ('Cy', 'Cyhat'): {'true_shape': (3, 7), 'est_shape': (3, 7), 'match': True}, ('Dyw', 'Dywhat'): {'true_shape': (3, 2), 'est_shape': (3, 7), 'match': False}, ('Cz', 'Czhat'): {'true_shape': (8, 7), 'est_shape': (8, 7), 'match': True}, ('Dzu', 'Dzuhat'): {'true_shape': (8, 1), 'est_shape': (8, 1), 'match': True}, ('Dzw', 'Dzwhat'): {'true_shape': (8, 2), 'est_shape': (8, 7), 'match': False}}}
            random
                [CMP] Comparison of estimates vs ground-truth:
                [cmp]   Ahat vs A       rel=1.044e-05
                [cmp]  Buhat vs Bu      rel=1.135e-07
                [cmp]  Bwhat vs Bw      subspace_gap=0.000e+00  rel=1.000e+00       
                [cmp]  Cyhat vs Cy      rel=3.233e-08
                [cmp] Dywhat vs Dyw     abs=3.284e-07
                [cmp]  Czhat vs Cz      rel=4.080e-06
                [cmp] Dzuhat vs Dzu     rel=1.097e-07
                [cmp] Dzwhat vs Dzw     abs=3.210e-04
                [PLT] Plotting completed. Metrics: {'metrics': {'fro_rel_error': 1.0439859481910079e-05, 'traj_rel_error': 0.00010167845678260035, 'rho_true': 0.9998750394917907, 'rho_hat': 0.9998750118201665}, 'dimension_checks': {('A', 'Ahat'): {'true_shape': (7, 7), 'est_shape': (7, 7), 'match': True}, ('Bu', 'Buhat'): {'true_shape': (7, 1), 'est_shape': (7, 1), 'match': True}, ('Bw', 'Bwhat'): {'true_shape': (7, 2), 'est_shape': (7, 7), 'match': False}, ('Cy', 'Cyhat'): {'true_shape': (3, 7), 'est_shape': (3, 7), 'match': True}, ('Dyw', 'Dywhat'): {'true_shape': (3, 2), 'est_shape': (3, 7), 'match': False}, ('Cz', 'Czhat'): {'true_shape': (8, 7), 'est_shape': (8, 7), 'match': True}, ('Dzu', 'Dzuhat'): {'true_shape': (8, 1), 'est_shape': (8, 1), 'match': True}, ('Dzw', 'Dzwhat'): {'true_shape': (8, 2), 'est_shape': (8, 7), 'match': False}}}
        Gaussian
            e1
                [CMP] Comparison of estimates vs ground-truth:
                [cmp]   Ahat vs A       rel=1.566e-02
                [cmp]  Buhat vs Bu      rel=2.137e-02
                [cmp]  Bwhat vs Bw      subspace_gap=0.000e+00  rel=5.200e-03        
                [cmp]  Cyhat vs Cy      rel=1.097e-08
                [cmp] Dywhat vs Dyw     abs=1.214e-11
                [cmp]  Czhat vs Cz      rel=3.365e-06
                [cmp] Dzuhat vs Dzu     rel=3.082e-08
                [cmp] Dzwhat vs Dzw     abs=1.566e-08
                [PLT] Plotting completed. Metrics: {'metrics': {'fro_rel_error': 0.015663134336058482, 'traj_rel_error': 0.03878359786768288, 'rho_true': 0.9998750394917907, 'rho_hat': 0.9999424386670328}, 'dimension_checks': {('A', 'Ahat'): {'true_shape': (7, 7), 'est_shape': (7, 7), 'match': True}, ('Bu', 'Buhat'): {'true_shape': (7, 1), 'est_shape': (7, 1), 'match': True}, ('Bw', 'Bwhat'): {'true_shape': (7, 2), 'est_shape': (7, 1), 'match': False}, ('Cy', 'Cyhat'): {'true_shape': (3, 7), 'est_shape': (3, 7), 'match': True}, ('Dyw', 'Dywhat'): {'true_shape': (3, 2), 'est_shape': (3, 1), 'match': False}, ('Cz', 'Czhat'): {'true_shape': (8, 7), 'est_shape': (8, 7), 'match': True}, ('Dzu', 'Dzuhat'): {'true_shape': (8, 1), 'est_shape': (8, 1), 'match': True}, ('Dzw', 'Dzwhat'): {'true_shape': (8, 2), 'est_shape': (8, 1), 'match': False}}}
            ones
                [CMP] Comparison of estimates vs ground-truth:
                [cmp]   Ahat vs A       rel=3.412e-02
                [cmp]  Buhat vs Bu      rel=3.925e-02
                [cmp]  Bwhat vs Bw      subspace_gap=0.000e+00  rel=1.152e-02        
                [cmp]  Cyhat vs Cy      rel=1.911e-08
                [cmp] Dywhat vs Dyw     abs=1.236e-11
                [cmp]  Czhat vs Cz      rel=3.308e-06
                [cmp] Dzuhat vs Dzu     rel=3.073e-08
                [PLT] Plotting completed. Metrics: {'metrics': {'fro_rel_error': 0.03412001842685607, 'traj_rel_error': 0.07181927769098195, 'rho_true': 0.9998750394917907, 'rho_hat': 0.9996616173080205}, 'dimension_checks': {('A', 'Ahat'): {'true_shape': (7, 7), 'est_shape': (7, 7), 'match': True}, ('Bu', 'Buhat'): {'true_shape': (7, 1), 'est_shape': (7, 1), 'match': True}, ('Bw', 'Bwhat'): {'true_shape': (7, 2), 'est_shape': (7, 1), 'match': False}, ('Cy', 'Cyhat'): {'true_shape': (3, 7), 'est_shape': (3, 7), 'match': True}, ('Dyw', 'Dywhat'): {'true_shape': (3, 2), 'est_shape': (3, 1), 'match': False}, ('Cz', 'Czhat'): {'true_shape': (8, 7), 'est_shape': (8, 7), 'match': True}, ('Dzu', 'Dzuhat'): {'true_shape': (8, 1), 'est_shape': (8, 1), 'match': True}, ('Dzw', 'Dzwhat'): {'true_shape': (8, 2), 'est_shape': (8, 1), 'match': False}}}
            random
                [CMP] Comparison of estimates vs ground-truth:
                [cmp]   Ahat vs A       rel=5.388e-02
                [cmp]  Buhat vs Bu      rel=5.539e-03
                [cmp]  Bwhat vs Bw      subspace_gap=1.490e-08  rel=4.559e-03        
                [cmp]  Cyhat vs Cy      rel=2.383e-08
                [cmp] Dywhat vs Dyw     abs=1.610e-11
                [cmp]  Czhat vs Cz      rel=3.352e-06
                [cmp] Dzuhat vs Dzu     rel=3.071e-08
                [cmp] Dzwhat vs Dzw     abs=1.560e-08
                [PLT] Plotting completed. Metrics: {'metrics': {'fro_rel_error': 0.053878907694304694, 'traj_rel_error': 0.07588044995799628, 'rho_true': 0.9998750394917907, 'rho_hat': 1.0000002762830784}, 'dimension_checks': {('A', 'Ahat'): {'true_shape': (7, 7), 'est_shape': (7, 7), 'match': True}, ('Bu', 'Buhat'): {'true_shape': (7, 1), 'est_shape': (7, 1), 'match': True}, ('Bw', 'Bwhat'): {'true_shape': (7, 2), 'est_shape': (7, 1), 'match': False}, ('Cy', 'Cyhat'): {'true_shape': (3, 7), 'est_shape': (3, 7), 'match': True}, ('Dyw', 'Dywhat'): {'true_shape': (3, 2), 'est_shape': (3, 1), 'match': False}, ('Cz', 'Czhat'): {'true_shape': (8, 7), 'est_shape': (8, 7), 'match': True}, ('Dzu', 'Dzuhat'): {'true_shape': (8, 1), 'est_shape': (8, 1), 'match': True}, ('Dzw', 'Dzwhat'): {'true_shape': (8, 2), 'est_shape': (8, 1), 'match': False}}}
        correlated
            e1
                [CMP] Comparison of estimates vs ground-truth:
                [cmp]   Ahat vs A       rel=2.931e-01
                [cmp]  Buhat vs Bu      rel=4.784e-02
                [cmp]  Bwhat vs Bw      subspace_gap=1.490e-08  rel=1.491e-01        
                [cmp]  Cyhat vs Cy      rel=9.349e-09
                [cmp] Dywhat vs Dyw     abs=1.802e-11
                [cmp]  Czhat vs Cz      rel=3.992e-06
                [cmp] Dzuhat vs Dzu     rel=3.086e-08
                [cmp] Dzwhat vs Dzw     abs=1.855e-08
                [PLT] Plotting completed. Metrics: {'metrics': {'fro_rel_error': 0.29306403094359046, 'traj_rel_error': 0.1638847521499199, 'rho_true': 0.9998750394917907, 'rho_hat': 0.9998932839433232}, 'dimension_checks': {('A', 'Ahat'): {'true_shape': (7, 7), 'est_shape': (7, 7), 'match': True}, ('Bu', 'Buhat'): {'true_shape': (7, 1), 'est_shape': (7, 1), 'match': True}, ('Bw', 'Bwhat'): {'true_shape': (7, 2), 'est_shape': (7, 1), 'match': False}, ('Cy', 'Cyhat'): {'true_shape': (3, 7), 'est_shape': (3, 7), 'match': True}, ('Dyw', 'Dywhat'): {'true_shape': (3, 2), 'est_shape': (3, 1), 'match': False}, ('Cz', 'Czhat'): {'true_shape': (8, 7), 'est_shape': (8, 7), 'match': True}, ('Dzu', 'Dzuhat'): {'true_shape': (8, 1), 'est_shape': (8, 1), 'match': True}, ('Dzw', 'Dzwhat'): {'true_shape': (8, 2), 'est_shape': (8, 1), 'match': False}}}
            ones
                [CMP] Comparison of estimates vs ground-truth:
                [cmp]   Ahat vs A       rel=2.994e-01
                [cmp]  Buhat vs Bu      rel=2.484e-02
                [cmp]  Bwhat vs Bw      subspace_gap=2.107e-08  rel=1.024e-01        
                [cmp]  Cyhat vs Cy      rel=9.934e-09
                [cmp] Dywhat vs Dyw     abs=1.804e-11
                [cmp]  Czhat vs Cz      rel=3.958e-06
                [cmp] Dzuhat vs Dzu     rel=3.069e-08
                [cmp] Dzwhat vs Dzw     abs=1.839e-08
                [PLT] Plotting completed. Metrics: {'metrics': {'fro_rel_error': 0.29944352958422815, 'traj_rel_error': 0.682993439757955, 'rho_true': 0.9998750394917907, 'rho_hat': 0.999909842125003}, 'dimension_checks': {('A', 'Ahat'): {'true_shape': (7, 7), 'est_shape': (7, 7), 'match': True}, ('Bu', 'Buhat'): {'true_shape': (7, 1), 'est_shape': (7, 1), 'match': True}, ('Bw', 'Bwhat'): {'true_shape': (7, 2), 'est_shape': (7, 1), 'match': False}, ('Cy', 'Cyhat'): {'true_shape': (3, 7), 'est_shape': (3, 7), 'match': True}, ('Dyw', 'Dywhat'): {'true_shape': (3, 2), 'est_shape': (3, 1), 'match': False}, ('Cz', 'Czhat'): {'true_shape': (8, 7), 'est_shape': (8, 7), 'match': True}, ('Dzu', 'Dzuhat'): {'true_shape': (8, 1), 'est_shape': (8, 1), 'match': True}, ('Dzw', 'Dzwhat'): {'true_shape': (8, 2), 'est_shape': (8, 1), 'match': False}}}
            random
                [CMP] Comparison of estimates vs ground-truth:
                [cmp]   Ahat vs A       rel=2.827e-01
                [cmp]  Buhat vs Bu      rel=4.212e-02
                [cmp]  Bwhat vs Bw      subspace_gap=2.581e-08  rel=1.344e-01        
                [cmp]  Cyhat vs Cy      rel=9.298e-09
                [cmp] Dywhat vs Dyw     abs=1.891e-11
                [cmp]  Czhat vs Cz      rel=3.969e-06
                [cmp] Dzuhat vs Dzu     rel=3.072e-08
                [cmp] Dzwhat vs Dzw     abs=1.845e-08
                [PLT] Plotting completed. Metrics: {'metrics': {'fro_rel_error': 0.2826890836107835, 'traj_rel_error': 0.7094004030534825, 'rho_true': 0.9998750394917907, 'rho_hat': 1.0004165243920082}, 'dimension_checks': {('A', 'Ahat'): {'true_shape': (7, 7), 'est_shape': (7, 7), 'match': True}, ('Bu', 'Buhat'): {'true_shape': (7, 1), 'est_shape': (7, 1), 'match': True}, ('Bw', 'Bwhat'): {'true_shape': (7, 2), 'est_shape': (7, 1), 'match': False}, ('Cy', 'Cyhat'): {'true_shape': (3, 7), 'est_shape': (3, 7), 'match': True}, ('Dyw', 'Dywhat'): {'true_shape': (3, 2), 'est_shape': (3, 1), 'match': False}, ('Cz', 'Czhat'): {'true_shape': (8, 7), 'est_shape': (8, 7), 'match': True}, ('Dzu', 'Dzuhat'): {'true_shape': (8, 1), 'est_shape': (8, 1), 'match': True}, ('Dzw', 'Dzwhat'): {'true_shape': (8, 2), 'est_shape': (8, 1), 'match': False}}}

        independent
            e1
                [CMP] Comparison of estimates vs ground-truth:
                [cmp]   Ahat vs A       rel=1.010e-01
                [cmp]  Buhat vs Bu      rel=5.270e-02
                [cmp]  Bwhat vs Bw      subspace_gap=1.490e-08  rel=1.058e-02 
                [cmp]  Cyhat vs Cy      rel=1.368e-08
                [cmp] Dywhat vs Dyw     abs=1.181e-11
                [cmp]  Czhat vs Cz      rel=3.214e-06
                [cmp] Dzuhat vs Dzu     rel=3.071e-08
                [cmp] Dzwhat vs Dzw     abs=1.496e-08
                [PLT] Plotting completed. Metrics: {'metrics': {'fro_rel_error': 0.10096265794854022, 'traj_rel_error': 0.14807958837622764, 'rho_true': 0.9998750394917907, 'rho_hat': 0.9995914289145464}, 'dimension_checks': {('A', 'Ahat'): {'true_shape': (7, 7), 'est_shape': (7, 7), 'match': True}, ('Bu', 'Buhat'): {'true_shape': (7, 1), 'est_shape': (7, 1), 'match': True}, ('Bw', 'Bwhat'): {'true_shape': (7, 2), 'est_shape': (7, 1), 'match': False}, ('Cy', 'Cyhat'): {'true_shape': (3, 7), 'est_shape': (3, 7), 'match': True}, ('Dyw', 'Dywhat'): {'true_shape': (3, 2), 'est_shape': (3, 1), 'match': False}, ('Cz', 'Czhat'): {'true_shape': (8, 7), 'est_shape': (8, 7), 'match': True}, ('Dzu', 'Dzuhat'): {'true_shape': (8, 1), 'est_shape': (8, 1), 'match': True}, ('Dzw', 'Dzwhat'): {'true_shape': (8, 2), 'est_shape': (8, 1), 'match': False}}}
            ones
                [CMP] Comparison of estimates vs ground-truth:
                [cmp]   Ahat vs A       rel=4.281e-02
                [cmp]  Buhat vs Bu      rel=4.345e-04
                C:\Users\g7fie\OneDrive\Documenti\GitHub\DR-LDC\utils___simulate.py:641: RuntimeWarning: invalid value encountered in sqrt
                return float(np.sqrt(1.0 - np.min(s)**2))
                [cmp]  Bwhat vs Bw      subspace_gap=nan  rel=6.894e-03
                [cmp]  Cyhat vs Cy      rel=1.184e-08
                [cmp] Dywhat vs Dyw     abs=1.222e-11
                [cmp]  Czhat vs Cz      rel=3.397e-06
                [cmp] Dzuhat vs Dzu     rel=3.073e-08
                [cmp] Dzwhat vs Dzw     abs=1.580e-08
                [PLT] Plotting completed. Metrics: {'metrics': {'fro_rel_error': 0.04281034514578502, 'traj_rel_error': 0.07339410225924949, 'rho_true': 0.9998750394917907, 'rho_hat': 0.999662667483667}, 'dimension_checks': {('A', 'Ahat'): {'true_shape': (7, 7), 'est_shape': (7, 7), 'match': True}, ('Bu', 'Buhat'): {'true_shape': (7, 1), 'est_shape': (7, 1), 'match': True}, ('Bw', 'Bwhat'): {'true_shape': (7, 2), 'est_shape': (7, 1), 'match': False}, ('Cy', 'Cyhat'): {'true_shape': (3, 7), 'est_shape': (3, 7), 'match': True}, ('Dyw', 'Dywhat'): {'true_shape': (3, 2), 'est_shape': (3, 1), 'match': False}, ('Cz', 'Czhat'): {'true_shape': (8, 7), 'est_shape': (8, 7), 'match': True}, ('Dzu', 'Dzuhat'): {'true_shape': (8, 1), 'est_shape': (8, 1), 'match': True}, ('Dzw', 'Dzwhat'): {'true_shape': (8, 2), 'est_shape': (8, 1), 'match': False}}}
            random
                [CMP] Comparison of estimates vs ground-truth:
                [cmp]   Ahat vs A       rel=2.327e-02
                [cmp]  Buhat vs Bu      rel=1.088e-02
                [cmp]  Bwhat vs Bw      subspace_gap=2.107e-08  rel=1.537e-02        
                [cmp]  Cyhat vs Cy      rel=7.769e-09
                [cmp] Dywhat vs Dyw     abs=1.046e-11
                [cmp]  Czhat vs Cz      rel=3.210e-06
                [cmp] Dzuhat vs Dzu     rel=3.069e-08
                [cmp] Dzwhat vs Dzw     abs=1.495e-08
                [PLT] Plotting completed. Metrics: {'metrics': {'fro_rel_error': 0.02327116400551359, 'traj_rel_error': 0.07182629545805232, 'rho_true': 0.9998750394917907, 'rho_hat': 1.0001053771032629}, 'dimension_checks': {('A', 'Ahat'): {'true_shape': (7, 7), 'est_shape': (7, 7), 'match': True}, ('Bu', 'Buhat'): {'true_shape': (7, 1), 'est_shape': (7, 1), 'match': True}, ('Bw', 'Bwhat'): {'true_shape': (7, 2), 'est_shape': (7, 1), 'match': False}, ('Cy', 'Cyhat'): {'true_shape': (3, 7), 'est_shape': (3, 7), 'match': True}, ('Dyw', 'Dywhat'): {'true_shape': (3, 2), 'est_shape': (3, 1), 'match': False}, ('Cz', 'Czhat'): {'true_shape': (8, 7), 'est_shape': (8, 7), 'match': True}, ('Dzu', 'Dzuhat'): {'true_shape': (8, 1), 'est_shape': (8, 1), 'match': True}, ('Dzw', 'Dzwhat'): {'true_shape': (8, 2), 'est_shape': (8, 1), 'match': False}}}
    """

