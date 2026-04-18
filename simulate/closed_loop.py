#!/usr/bin/env python3
import numpy as np
import matplotlib.pyplot as plt
from typing import Optional, Dict, Any, Iterable


from config import get_cfg                              # loader.py
from disturbances import Disturbances                   # disturbances.py
from utils import Plant, Controller, Plant_cl           # systems.py

from .initial_conditions import _initial_condition_from_eigenvalues


# =============================================================================================== #

class Closed_Loop():
    def __init__(self, TEST=False):
        cfg = get_cfg()
        self.p = cfg.get("params", {})
        sim = self.p.get("simulation", {})
        self.Tf = sim.get("TotTime", 25)
        self.ts = sim.get("ts", 0.5)

        if TEST: self.test()
    
    # ------------------------------------------------------------------------------------------- #

    def test(self):
        from core import MatricesAPI
        api = MatricesAPI()
        plant, ctrl = api.get_system()

        sim = self.simulate_closed_loop(plant, ctrl)
        print("Simulated shapes:",
            {k: v.shape for k, v in sim.items() if isinstance(v, np.ndarray)})

        out = self.save_npz(sim, "closed_loop_run_seed11_T800.npz")
        print(f"Saved time series to {out}")

        self.plot_timeseries(sim)

    def save_npz(self, sim, fname="cl_timeseries.npz"):
        np.savez_compressed(fname, **sim)
        return fname

    # ------------------------------------------------------------------------------------------- #

    def simulate_closed_loop(self, plant: Plant,
                            ctrl: Controller,
                            Sigma_w: np.ndarray = None,
                            gamma: float = None,
                            seed: int = 11,
                            init_cond: str = "zeros", 
                            x_init: np.ndarray = None,
                            xc_init: np.ndarray = None, ):
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

        if init_cond == "zeros":
            x = np.zeros((nx, 1))
            xc = np.zeros((nxc, 1))
        elif init_cond == "from_eig":            
            x = _initial_condition_from_eigenvalues(A)
            xc = _initial_condition_from_eigenvalues(Ac)
        elif init_cond == "e1":
            x = np.zeros((nx, 1))
            x[0, 0] = 0.07
            xc = np.zeros((nxc, 1))
        elif init_cond == "set": 
            x = x_init
            xc = xc_init
        else:   # rand
            rng = np.random.default_rng(seed)
            x = rng.standard_normal((nx, 1))
            xc = rng.standard_normal((nxc, 1))
        
        x_0, xc_0 = x, xc

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
        step = np.zeros((T,), dtype=int)

        wass = Disturbances(gamma=gamma, n=Sigma_w[0].size, var=1)
        W = wass.sample(T=T, Sigma=Sigma_w)
        if W.ndim == 1:
            W = W.reshape(T, 1)

        if W.shape[1] == nw:
            pass
        
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
            step[t] = t

            # Update
            x_next  = A @ x + Bu @ u + Bw @ w
            xc_next = Ac @ xc + Bc @ y
            x, xc = x_next, xc_next

        return {
            "X": X, "Xc": Xc, "Y": Y, "U": U, "Z": Z, "W": W,
            "T": T, "nx": nx, "nxc": nxc, "ny": ny, "nu": nu, "nz": nz, "nw": nw, "step": step,
            "x_0": x_0, "xc_0": xc_0,
        }
    
    def simulate_composite(self, Pcl: Plant_cl, Sigma_w: np.ndarray = None, gamma: float = 0.5, init_cond: str = "zeros") -> Dict[str, Any]:
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
        if init_cond == "zeros":
            x = np.zeros((nX, 1))
        elif init_cond == "from_eig":           
            x = _initial_condition_from_eigenvalues(A)
        elif init_cond == "e1":
            x = np.zeros((nX, 1))
            x[0, 0] = 1.0
        else:   # rand
            rng = np.random.default_rng()
            x = rng.standard_normal((nX, 1))        

        # Disturbance generation
        wass = Disturbances(gamma=gamma, n=Sigma_w[0].size, var=1)
        W = wass.sample(T=T, Sigma=Sigma_w)
        if W.ndim == 1:
            W = W.reshape(T, 1)

        if W.shape[1] == nw:
            # Great. Use directly.
            pass
        """else:
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
            W = (d @ B_pinv.T)                  # (T x nw_hat)"""

        # Storage
        X = np.zeros((T, nX))
        Z = np.zeros((T, nz))
        step = np.zeros((T,), dtype=int)

        # Rollout
        for t in range(T):
            w = W[t, :].reshape(nw, 1)
            z = C @ x + D @ w
            X[t, :] = x.ravel()
            Z[t, :] = z.ravel()
            step[t] = t
            x = A @ x + B @ w

        return {
            "X": X,               # (T, nX)
            "Z": Z,               # (T, nz)
            "W": W,               # (T, nw)
            "T": T, "ts": self.ts, "step": step,
            "nX": nX, "nw": nw, "nz": nz,
        }

    # ------------------------------------------------------------------------------------------- #

    def simulate_Z_cost(self, Z: np.ndarray, Q: np.ndarray = None, plot: bool = True):
        """
        Compute quadratic cost from performance output time series Z and (optionally) plot it.

        Z: array with shape (nz, T), (T, nz), or (T, nz, N) for N trajectories.
        If batched, E[·] is taken as the empirical mean over N.
        Q: optional (nz, nz) PSD weighting; defaults to identity → ||z||_2^2.
        plot: whether to plot instantaneous cost and running average.

        Returns:
            dict with:
                'inst'      : (T,) instantaneous costs c_t
                'running'   : (T,) running average J_t
                'J'         : float, final time-average cost J_T
                'T'         : int, number of timesteps
        """
        Z = np.asarray(Z)

        # Normalize to shape (T, nz) or (T, nz, N)
        if Z.ndim == 1:
            Z = Z.reshape(1, -1)  # (nz=1, T)
        if Z.ndim == 2:
            # Guess orientation: if first dim matches self.T, keep; otherwise transpose.
            T_guess = int(round(self.Tf / self.ts))
            Z = Z if Z.shape[0] == T_guess or Z.shape[0] > Z.shape[1] else Z.T
        elif Z.ndim == 3:
            # Expect (T, nz, N). If it looks like (nz, T, N), swap first two axes.
            if Z.shape[0] < Z.shape[1]:
                Z = np.swapaxes(Z, 0, 1)
        else:
            raise ValueError("Z must be 1D, 2D, or 3D.")

        T = Z.shape[0]
        nz = Z.shape[1]
        t = np.arange(T) * self.ts

        # Weighting
        if Q is None:
            # c_t = ||z(t)||^2
            if Z.ndim == 2:
                inst = np.sum(Z**2, axis=1)
            else:  # (T, nz, N) → average over N
                inst = np.mean(np.sum(Z**2, axis=1), axis=1)
        else:
            Q = np.asarray(Q)
            if Q.shape != (nz, nz):
                raise ValueError(f"Q must have shape ({nz},{nz}), got {Q.shape}.")
            # c_t = z(t)^T Q z(t); handle batch by averaging over N
            if Z.ndim == 2:
                # einsum: (T,nz),(nz,nz),(T,nz) → (T,)
                inst = np.einsum("ti,ij,tj->t", Z, Q, Z)
            else:
                # (T,nz,N) → average over N
                inst = np.mean(np.einsum("tin,ij,tjn->t n", Z, Q, Z), axis=1)

        # Running average
        running = np.cumsum(inst) / np.arange(1, T + 1)
        J = float(running[-1])

        if plot:
            # Instantaneous cost
            plt.figure(figsize=(8, 3.2))
            plt.plot(t, inst, linewidth=1.5)
            plt.xlabel("t")
            plt.ylabel(r"$\|z(t)\|^2$" if Q is None else r"$z(t)^\top Q z(t)$")
            plt.title("Instantaneous quadratic cost")
            plt.grid(True, alpha=0.3)

            # Running average
            plt.figure(figsize=(8, 3.2))
            plt.plot(t, running, linewidth=1.5)
            plt.xlabel("t")
            plt.ylabel(r"$\frac{1}{t}\sum_{k=0}^{t-1}\|z(k)\|^2$")
            plt.title("Running time-average cost")
            plt.grid(True, alpha=0.3)
            plt.show()

        return {"inst": inst, "running": running, "J": J, "T": T}

    def simulate_ZW_snr(self,
                        Z: np.ndarray,
                        W: np.ndarray,
                        plot: bool = True,
                        eps: float = 1e-12):
        """
        Compute Signal-to-Noise Ratio (SNR) between performance output Z and disturbance W,
        and optionally plot its evolution in time.

        Z: array with shape (nz, T), (T, nz), or (T, nz, N) for N trajectories.
        Treated as the "signal".
        W: array with shape (nw, T), (T, nw), or (T, nw, N) for N trajectories.
        Treated as the "noise" / disturbance.
        If batched, E[·] is taken as the empirical mean over N.

        SNR is defined as:
            SNR(t)      = P_signal(t) / P_noise(t)
            SNR_dB(t)   = 10 log10( SNR(t) )
        where P_signal(t) = ||z(t)||_2^2, P_noise(t) = ||w(t)||_2^2
        (with batch-average if N > 1), and the global SNR is
            SNR_dB = 10 log10( mean_t P_signal(t) / mean_t P_noise(t) ).

        Returns:
            dict with:
                'snr_t'     : (T,) SNR(t) (linear scale)
                'snr_db_t'  : (T,) SNR_dB(t) in dB
                'snr_db'    : float, global SNR in dB (time-averaged power ratio)
                'T'         : int, number of timesteps
        """

        def _normalize_TSZN(arr: np.ndarray, name: str) -> np.ndarray:
            """
            Normalize input to shape (T, n, N) or (T, n) if unbatched.
            Mirrors the orientation logic of simulate_Z_cost.
            """
            arr = np.asarray(arr)

            if arr.ndim == 1:
                arr = arr.reshape(1, -1)  # (n=1, T)

            if arr.ndim == 2:
                # Guess orientation using (Tf, ts) as in simulate_Z_cost
                T_guess = int(round(self.Tf / self.ts))
                # If first dim looks like time, keep; else transpose
                if not (arr.shape[0] == T_guess or arr.shape[0] > arr.shape[1]):
                    arr = arr.T  # now (T, n)
            elif arr.ndim == 3:
                # Expect (T, n, N). If it looks like (n, T, N), swap first two axes.
                if arr.shape[0] < arr.shape[1]:
                    arr = np.swapaxes(arr, 0, 1)
            else:
                raise ValueError(f"{name} must be 1D, 2D, or 3D.")

            return arr

        # Normalize Z and W to (T, n, N?) / (T, n)
        Z = _normalize_TSZN(Z, "Z")
        W = _normalize_TSZN(W, "W")

        # Broadcast/batch compatibility check (time dimension must match)
        if Z.shape[0] != W.shape[0]:
            raise ValueError(f"Time dimensions differ: Z.shape[0]={Z.shape[0]}, W.shape[0]={W.shape[0]}")

        T = Z.shape[0]
        t = np.arange(T) * self.ts

        # Ensure both have an explicit batch dimension if needed
        if Z.ndim == 2:
            Z_ = Z[..., None]  # (T, nz, 1)
        else:
            Z_ = Z             # (T, nz, N)

        if W.ndim == 2:
            W_ = W[..., None]  # (T, nw, 1)
        else:
            W_ = W             # (T, nw, N)

        # Power per time, averaged over batch N
        # P_signal(t) = E_N[ sum_i Z_i(t)^2 ]
        # P_noise(t)  = E_N[ sum_j W_j(t)^2 ]
        signal_power_t = np.mean(np.sum(Z_**2, axis=1), axis=-1)  # (T,)
        noise_power_t  = np.mean(np.sum(W_**2, axis=1), axis=-1)  # (T,)

        # Avoid division by zero
        noise_power_t_clipped = np.maximum(noise_power_t, eps)

        # SNR(t) and SNR_dB(t)
        snr_t = signal_power_t / noise_power_t_clipped
        snr_db_t = 10.0 * np.log10(snr_t + eps)

        # Global SNR: ratio of mean powers
        signal_power_mean = float(np.mean(signal_power_t))
        noise_power_mean  = float(np.mean(noise_power_t_clipped))

        if noise_power_mean <= 0:
            snr_db = np.inf
        else:
            snr_db = float(10.0 * np.log10(signal_power_mean / noise_power_mean))

        if plot:
            # SNR_dB(t)
            plt.figure(figsize=(8, 3.2))
            plt.plot(t, snr_db_t, linewidth=1.5)
            plt.xlabel("t")
            plt.ylabel(r"$\mathrm{SNR}(t)\;[\mathrm{dB}]$")
            plt.title("Instantaneous SNR between Z (signal) and W (disturbance)")
            plt.grid(True, alpha=0.3)

            # Optional: linear SNR(t) on separate plot (if you care)
            plt.figure(figsize=(8, 3.2))
            plt.plot(t, snr_t, linewidth=1.5)
            plt.xlabel("t")
            plt.ylabel(r"$\mathrm{SNR}(t)$")
            plt.title("Instantaneous SNR (linear scale)")
            plt.grid(True, alpha=0.3)

            plt.show()

        return {
            "snr_t": snr_t,
            "snr_db_t": snr_db_t,
            "snr_db": snr_db,
            "T": T,
        }

    # ------------------------------------------------------------------------------------------- #            

    def plot_timeseries(self, sim, save: bool = False, out: str = None, fmt: str = "pdf", dpi: int = 300, tight: bool = True):
        T = sim["T"]
        t = np.arange(T) * self.ts

        saved = {}

        def _finalize(fig, name: str):
            nonlocal saved
            if tight:
                try:
                    fig.tight_layout()
                except Exception:
                    pass
            if save:
                save_path = out + f"{name}.{fmt}"
                fig.savefig(save_path, dpi=dpi, bbox_inches="tight")
                plt.close(fig)
                saved[name] = save_path
            else:
                fig.show()

        # 1) Plant states x
        fig1 = plt.figure(figsize=(10, 6))
        for i in range(sim["nx"]):
            plt.plot(t, sim["X"][:, i], label=f"x[{i}]")
        plt.title("Plant states x")
        plt.xlabel("t")
        plt.legend()
        plt.grid(True, alpha=0.3)
        _finalize(fig1, "___plant_states_x")

        # 2) Controller states x_c
        fig2 = plt.figure(figsize=(10, 6))
        for i in range(sim["nxc"]):
            plt.plot(t, sim["Xc"][:, i], label=f"x_c[{i}]")
        plt.title("Controller states x_c")
        plt.xlabel("t")
        plt.legend()
        plt.grid(True, alpha=0.3)
        _finalize(fig2, "___controller_states_xc")

        # 3) Measured output y
        fig3 = plt.figure(figsize=(10, 6))
        for i in range(sim["ny"]):
            plt.plot(t, sim["Y"][:, i], label=f"y[{i}]")
        plt.title("Measured output y")
        plt.xlabel("t")
        plt.legend()
        plt.grid(True, alpha=0.3)
        _finalize(fig3, "___measured_output_y")

        # 4) Control input u
        fig4 = plt.figure(figsize=(10, 5))
        for i in range(sim["nu"]):
            plt.plot(t, sim["U"][:, i], label=f"u[{i}]")
        plt.title("Control input u")
        plt.xlabel("t")
        plt.legend()
        plt.grid(True, alpha=0.3)
        _finalize(fig4, "___control_input_u")

        # 5) Performance output z
        fig5 = plt.figure(figsize=(10, 6))
        for i in range(sim["nz"]):
            plt.plot(t, sim["Z"][:, i], label=f"z[{i}]")
        plt.title("Performance output z")
        plt.xlabel("t")
        plt.legend()
        plt.grid(True, alpha=0.3)
        _finalize(fig5, "___performance_output_z")

        plt.show()
        if save: return saved

    def plot_composite(self,
                        sim: dict,
                        show_X: bool = True,
                        show_Z: bool = True,
                        show_W: bool = False,
                        X_idx: Optional[Iterable[int]] = None,
                        Z_idx: Optional[Iterable[int]] = None,
                        W_idx: Optional[Iterable[int]] = None,
                        suptitle: Optional[str] = None,
                        save: bool = False, out: str = None, fmt: str = "pdf", dpi: int = 300, tight: bool = True):
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

        saved = {}

        def _finalize(fig, name: str):
            nonlocal saved
            if tight:
                try:
                    fig.tight_layout()
                except Exception:
                    pass
            if save:
                save_path = out + f"{name}.{fmt}"
                fig.savefig(save_path, dpi=dpi, bbox_inches="tight")
                plt.close(fig)
                saved[name] = save_path
            else:
                fig.show()


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
            figX = plt.figure(figsize=(10, 6))
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
            _finalize(figX, "___plant_states_X_CL")

        if show_Z:
            Z = np.asarray(sim["Z"])
            nz = int(sim["nz"])
            zi = _ensure_indices(nz, Z_idx)
            figZ = plt.figure(figsize=(10, 6))
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
            _finalize(figZ, "___performance_output_Z_CL")

        if show_W:
            W = np.asarray(sim["W"])
            nw = int(sim["nw"])
            wi = _ensure_indices(nw, W_idx)
            figW = plt.figure(figsize=(10, 5))
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
            _finalize(figW, "___plant_disturbances_W")

        plt.show()
        if save: return saved

# =============================================================================================== #