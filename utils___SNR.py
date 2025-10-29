# snr_analyzer.py
import numpy as np
from numpy.linalg import eigvals
from scipy.linalg import solve_discrete_lyapunov
import matplotlib.pyplot as plt
import sys

class SNRAnalyzer:
    """
    Steady-state SNR for closed-loop LTI systems with additive disturbance w ~ N(0, Sigma).

    Inputs at init (dicts or arrays):
      plant: {
        "A","Bu","Bw","Cy","Dyw","Cz","Dzw","Dzu"
      }
      ctrl:  {
        "Ac","Bc","Cc","Dc"
      }
      plant_cl (optional/minimal): {
        "Acl" : composite closed-loop A (for X=[x; xc]).
        # If not provided, Acl is composed from plant+ctrl.
      }
      Sigma: disturbance covariance (nw x nw), SPD.

    Ports:
      y = Cy x + Dyw w
      u = Cc xc + Dc y
      z = Cz x + Dzu u + Dzw w

    SNR definition (trace-based):
      SNR_q = tr(C_q^cl P C_q^cl^T) / tr(D_qw Σ D_qw^T),  q ∈ {y,u,z}
      where P solves P = Acl P Acl^T + Bcl Σ Bcl^T.
    """

    # ---------------------------- ctor ----------------------------
    def __init__(self, plant: dict, ctrl: dict, Sigma: np.ndarray):
        # plant
        self.A   = plant.A
        self.Bu  = plant.Bu
        self.Bw  = plant.Bw
        self.Cy  = plant.Cy
        self.Dyw = plant.Dyw
        self.Cz  = plant.Cz
        self.Dzw = plant.Dzw
        self.Dzu = plant.Dzu

        # controller
        self.Ac = ctrl.Ac
        self.Bc = ctrl.Bc
        self.Cc = ctrl.Cc
        self.Dc = ctrl.Dc

        # disturbance covariance
        Sigma = np.array(Sigma, dtype=float)
        self.Sigma = 0.5 * (Sigma + Sigma.T)

        # Compose closed-loop blocks
        self.Acl, self.Bcl, self.Cy_cl, self.Cu_cl, self.Cz_cl, \
        self.Dyw_cl, self.Duw_cl, self.Dzw_cl = self._compose_closed_loop()

        # Solve steady-state covariance (raises if unstable)
        self.P = self._solve_steady_state()

        # Precompute covariance splits
        self._covs = self._compute_covariances()

        self._precompute_kernels()

    # --------------------- closed-loop composition ---------------------
    def _compose_closed_loop(self):
        # Acl
        Acl = np.block([
            [self.A + self.Bu @ self.Dc @ self.Cy,      self.Bu @ self.Cc],
            [self.Bc @ self.Cy,                          self.Ac         ],
        ])
        # Bcl (map from w to augmented state X=[x;xc])
        Bcl = np.vstack([
            self.Bw + self.Bu @ self.Dc @ self.Dyw,
            self.Bc @ self.Dyw
        ])
        # Output maps (augmented)
        nx  = self.A.shape[0]
        nxc = self.Ac.shape[0]
        zeros_xc = np.zeros((self.Cy.shape[0], nxc))
        Cy_cl = np.hstack([self.Cy, zeros_xc])                          # y = Cy_cl X + Dyw w
        Cu_cl = np.hstack([self.Dc @ self.Cy, self.Cc])                 # u = Cu_cl X + Duw w
        Cz_cl = np.hstack([self.Cz + self.Dzu @ self.Dc @ self.Cy,      # z = Cz_cl X + Dzw_cl w
                           self.Dzu @ self.Cc])
        Dyw_cl = self.Dyw
        Duw_cl = self.Dc @ self.Dyw
        Dzw_cl = self.Dzw + self.Dzu @ self.Dc @ self.Dyw
        return Acl, Bcl, Cy_cl, Cu_cl, Cz_cl, Dyw_cl, Duw_cl, Dzw_cl

    # ------------------------ steady-state P ------------------------
    def _solve_steady_state(self):
        rho = float(np.max(np.abs(eigvals(self.Acl))))
        if not np.isfinite(rho) or rho >= 1.0:
            raise ValueError(f"Acl not Schur-stable (spectral radius {rho:.6g}). "
                             "Steady-state SNR is undefined.")
        Q = self.Bcl @ self.Sigma @ self.Bcl.T
        return solve_discrete_lyapunov(self.Acl, Q)

    # -------------------- covariance components --------------------
    def _compute_covariances(self):
        Sig_y_sig   = self.Cy_cl @ self.P @ self.Cy_cl.T
        Sig_y_noise = self.Dyw_cl @ self.Sigma @ self.Dyw_cl.T

        Sig_u_sig   = self.Cu_cl @ self.P @ self.Cu_cl.T
        Sig_u_noise = self.Duw_cl @ self.Sigma @ self.Duw_cl.T

        Sig_z_sig   = self.Cz_cl @ self.P @ self.Cz_cl.T
        Sig_z_noise = self.Dzw_cl @ self.Sigma @ self.Dzw_cl.T

        return {
            "y_signal": Sig_y_sig,  "y_noise": Sig_y_noise,
            "u_signal": Sig_u_sig,  "u_noise": Sig_u_noise,
            "z_signal": Sig_z_sig,  "z_noise": Sig_z_noise,
        }

    # --------------------------- SNRs ---------------------------
    @staticmethod
    def _todB(x, eps=1e-16):
        return 10*np.log10(np.maximum(x, eps))

    def snr(self):
        """Aggregate SNRs (trace-based) and per-channel ratios."""
        eps = 1e-16
        c = self._covs
        tr = np.trace
        snr_y = tr(c["y_signal"]) / max(tr(c["y_noise"]), eps)
        snr_u = tr(c["u_signal"]) / max(tr(c["u_noise"]), eps)
        snr_z = tr(c["z_signal"]) / max(tr(c["z_noise"]), eps)

        def diag_ratio(Ssig, Snoise):
            num = np.clip(np.diag(Ssig),   0.0, None)
            den = np.clip(np.diag(Snoise), eps, None)
            return num / den

        return {
            "spectral_radius_Acl": float(np.max(np.abs(eigvals(self.Acl)))),
            "SNR_y": float(snr_y), "SNR_y_dB": float(self._todB(snr_y)),
            "SNR_u": float(snr_u), "SNR_u_dB": float(self._todB(snr_u)),
            "SNR_z": float(snr_z), "SNR_z_dB": float(self._todB(snr_z)),
            "SNR_y_channels": diag_ratio(c["y_signal"], c["y_noise"]),
            "SNR_u_channels": diag_ratio(c["u_signal"], c["u_noise"]),
            "SNR_z_channels": diag_ratio(c["z_signal"], c["z_noise"]),
        }

    # --------------------------- plots ---------------------------
    def plot_bars(self, title="SNR (trace-based)", show=True):
        r = self.snr()
        agg = np.array([r["SNR_y"], r["SNR_u"], r["SNR_z"]])
        fig, axes = plt.subplots(1, 2, figsize=(9, 4.2))

        axes[0].bar(["y","u","z"], self._todB(agg))
        axes[0].set_ylabel("dB"); axes[0].set_title(title)

        chans = [r["SNR_y_channels"], r["SNR_u_channels"], r["SNR_z_channels"]]
        labels = ["y","u","z"]
        L = max(x.size for x in chans)
        x = np.arange(L); w = 0.25
        for k,(lab,vec) in enumerate(zip(labels, chans)):
            pad = np.pad(vec, (0, L - vec.size), constant_values=np.nan)
            axes[1].bar(x + (k-1)*w, self._todB(pad), width=w, label=lab)
        axes[1].set_xticks(x); axes[1].set_xlabel("channel idx")#; axes[1].set_ylabel("dB")
        axes[1].legend()
        fig.tight_layout()
        if show: plt.show()
        return fig

    def plot_output_psd(self, y: np.ndarray, tlt: str, fs: float, nfft: int = 2048, show=True):
        """
        Periodogram PSD for y (T x ny). One axis, one curve per channel.
        Detrends mean. Zero-pads to 'pad' = next power of two >= max(nfft, T).
        """
        import numpy as np
        import matplotlib.pyplot as plt

        y = np.asarray(y)
        if y.ndim == 1:
            y = y[:, None]
        T, m = y.shape
        if T < 2:
            raise ValueError("Need at least 2 samples to compute a PSD. This isn’t quantum magic.")

        # Detrend mean
        y = y - y.mean(axis=0, keepdims=True)

        # Frequency grid
        pad = int(2**np.ceil(np.log2(max(nfft, T))))
        freqs = np.fft.rfftfreq(pad, d=1.0/fs)

        # Periodogram
        psd = np.empty((freqs.size, m))
        for j in range(m):
            Y = np.fft.rfft(y[:, j], n=pad)
            P = (Y * np.conj(Y)).real / T   # simple (non-Welch) periodogram
            psd[:, j] = P[:freqs.size]

        # Plot (in dB), guarding zeros
        psd_db = 10 * np.log10(np.maximum(psd, 1e-300))

        plt.figure(figsize=(6.4, 4.2))
        for j in range(m):
            plt.plot(freqs, psd_db[:, j], label=f"{tlt}[{j}]")
        plt.xlabel("Frequency [Hz]")
        plt.ylabel("PSD [dB]")
        plt.title("Output PSD (periodogram)")
        plt.legend(fontsize=8)
        plt.tight_layout()
        if show:
            plt.show()

        return freqs, psd


    # ===================== KERNEL (Σ-linear) SNR FORMULATION =====================

    def _lyap_obs(self, C):
        """Solve X = Acl^T X Acl + C^T C (observer Lyapunov)."""
        return solve_discrete_lyapunov(self.Acl.T, C.T @ C)

    def _sym(self, M):
        return 0.5 * (M + M.T)

    def _precompute_kernels(self):
        """
        Precompute X_q, K_q = Bcl^T X_q Bcl and R_q = D_qw^T D_qw for q in {y,u,z}.
        Then SNR_q(Σ) = <K_q, Σ>/<R_q, Σ>, independent of the scale of Σ.
        """
        Xy = self._lyap_obs(self.Cy_cl)
        Xu = self._lyap_obs(self.Cu_cl)
        Xz = self._lyap_obs(self.Cz_cl)

        Ky = self._sym(self.Bcl.T @ Xy @ self.Bcl)
        Ku = self._sym(self.Bcl.T @ Xu @ self.Bcl)
        Kz = self._sym(self.Bcl.T @ Xz @ self.Bcl)

        Ry = self._sym(self.Dyw_cl.T @ self.Dyw_cl)
        Ru = self._sym(self.Duw_cl.T @ self.Duw_cl)
        Rz = self._sym(self.Dzw_cl.T @ self.Dzw_cl)

        self._K = {"y": Ky, "u": Ku, "z": Kz}
        self._R = {"y": Ry, "u": Ru, "z": Rz}

    # call kernels once at the end of __init__
    #   add this line at the end of your current __init__:
    #   self._precompute_kernels()

    def snr_from_kernels(self, Sig: np.ndarray = None):
        """
        Evaluate SNR_y/u/z using the kernel formula:
            SNR_q(Σ) = tr(K_q Σ) / tr(R_q Σ).
        Useful to sweep Σ without re-solving Lyapunov for P.
        """
        Sigma = self.Sigma if Sig is None else Sig

        def ratio(K, R):
            num = float(np.trace(K @ Sigma))
            den = float(np.trace(R @ Sigma))
            return np.inf if den <= 1e-16 else num / den

        SNRy = ratio(self._K["y"], self._R["y"])
        SNRu = ratio(self._K["u"], self._R["u"])
        SNRz = ratio(self._K["z"], self._R["z"])
        return {
            "SNR_y": SNRy, "SNR_y_dB": float(self._todB(SNRy)),
            "SNR_u": SNRu, "SNR_u_dB": float(self._todB(SNRu)),
            "SNR_z": SNRz, "SNR_z_dB": float(self._todB(SNRz)),
        }

    # -------- generalized eigen analysis: worst/best Σ directions (trace-normed) -----

    def _pinv_sqrt(self, M, rcond=1e-12):
        """Return M^{-1/2} on range(M) (Moore-Penrose via eig)."""
        w, V = np.linalg.eigh(self._sym(M))
        w = np.clip(w, 0.0, None)
        keep = w > rcond * (w.max() if w.size else 1.0)
        if not np.any(keep):   # R is numerically zero ⇒ no direct noise feedthrough
            return None
        Dmh = np.zeros_like(w)
        Dmh[keep] = 1.0 / np.sqrt(w[keep])
        return V @ np.diag(Dmh) @ V.T

    def worst_best_snr(self, port: str = "z"):
        """
        Max/min SNR_q over PSD Σ with tr(Σ)=1:
            SNR extremes are eigenvalues of R^{-1/2} K R^{-1/2} on range(R).
        Returns (snr_min, snr_max).
        """
        K = self._K[port]; R = self._R[port]
        Rmh = self._pinv_sqrt(R)
        if Rmh is None:
            return (np.inf, np.inf)  # no noise in that port; SNR is infinite
        S = self._sym(Rmh @ K @ Rmh)  # similar to the generalized eigen pencil (K,R)
        w = np.linalg.eigvalsh(S)
        w = w[np.isfinite(w)]
        return float(w.min()), float(w.max())

    # ----------------------- plotting: SNR vs Σ direction ---------------------------

    def plot_snr_rotation_sweep(self, dims=(0,1), n_angles: int = 181,
                                ports=("y","u","z"), show=True):
        Sigma0 = self.Sigma
        nw = Sigma0.shape[0]
        i, j = dims
        if i == j or i >= nw or j >= nw:
            raise ValueError("dims must be two distinct valid indices.")

        thetas = np.linspace(0, np.pi, n_angles)
        snr_traces = {p: np.zeros_like(thetas) for p in ports}

        for k, th in enumerate(thetas):
            c, s = np.cos(th), np.sin(th)
            R = np.eye(nw)
            # correct 2×2 rotation embedding
            R[np.ix_([i, j], [i, j])] = np.array([[c, -s],
                                                [s,  c]])
            Sig = R @ Sigma0 @ R.T
            vals = self.snr_from_kernels(Sig)
            for p in ports:
                snr_traces[p][k] = vals[f"SNR_{p}"]

        plt.figure(figsize=(7.4, 4.2))
        for p in ports:
            plt.plot(np.rad2deg(thetas), self._todB(snr_traces[p]), label=f"SNR_{p}(Σ(θ))")
        plt.xlabel(f"rotation angle θ [deg] in subspace {dims}")
        plt.ylabel("SNR [dB]")
        plt.title("SNR vs disturbance orientation (kernel method)")
        plt.legend()
        plt.tight_layout()
        if show: plt.show()
        return thetas, snr_traces


    def plot_worst_best_lines(self, ports=("y","u","z"), show=True):
        """
        Horizontal lines for worst/best-case SNR under tr(Σ)=1 for each port,
        using generalized eigenvalues of (K,R).
        """
        mins, maxs = [], []
        for p in ports:
            m, M = self.worst_best_snr(p)
            mins.append(self._todB(m)); maxs.append(self._todB(M))

        x = np.arange(len(ports))
        plt.figure(figsize=(6.2, 3.8))
        plt.hlines(mins, x-0.3, x+0.3, colors="tab:red",  label="min dB")
        plt.hlines(maxs, x-0.3, x+0.3, colors="tab:green",label="max dB")
        plt.xticks(x, [f"SNR_{p}" for p in ports]); plt.ylabel("dB")
        plt.title("Worst/best SNR over trace-normalized Σ")
        plt.legend(); plt.tight_layout()
        if show: plt.show()
        return dict(min_dB=np.array(mins), max_dB=np.array(maxs))
