import numpy as np
from numpy.linalg import cholesky
from scipy.linalg import sqrtm
from pathlib import Path
import yaml, sys
import matplotlib.pyplot as plt

yaml_path = Path(__file__).resolve().parent / "problem___parameters.yaml"
if yaml is None:
    raise ImportError("PyYAML not available. Install with `pip install pyyaml`.")
with open(yaml_path, "r", encoding="utf-8") as f:
    cfg = yaml.safe_load(f)


# =====================================================================================

class metric_2_Wasserstein:
    """
    2-Wasserstein ambiguity set around a fixed zero-mean Gaussian P_nom = N(0, Sigma_nom).

    Mathematically (paper's notation):

        W_cor  = { all disturbance sequences w : N0 -> L2(Ω, R^{n_w}) s.t.
                   W(P_{w(t)}, P_nom) <= gamma  for all t }

        W_ind  = subset of W_cor with w(t) independent across t.

    Here we:
      * Fix P_nom = N(0, Sigma_nom).
      * Use the Gelbrich / 2-Wasserstein distance specialized to Gaussians.
      * Provide:
          - membership checks for Gaussian marginals,
          - Gaussian sampling procedures whose marginals lie in the ball
            (independent or AR(1)-correlated).

    This is still *Gaussian* inside the ball; the full ambiguity set in the paper
    has no parametric restriction, but these Gaussians are legitimate members of it.
    """

    # -------------------------------------------------------------------------
    # basic helpers
    # -------------------------------------------------------------------------

    @staticmethod
    def _sym(A: np.ndarray) -> np.ndarray:
        return 0.5 * (A + A.T)

    @staticmethod
    def _spd_correction(S: np.ndarray, eps: float = 1e-12) -> np.ndarray:
        """Force SPD by eigenvalue clipping (numerical guard)."""
        S = metric_2_Wasserstein._sym(S)
        vals, vecs = np.linalg.eigh(S)
        vals = np.maximum(vals, eps)
        return vecs @ np.diag(vals) @ vecs.T

    # -------------------------------------------------------------------------
    # 2-Wasserstein / Gelbrich distances (Gaussian case)
    # -------------------------------------------------------------------------

    @staticmethod
    def w2_gaussian_zero_mean(S1: np.ndarray, S2: np.ndarray) -> float:
        """
        W_2 between N(0, S1) and N(0, S2).

        For zero-mean Gaussians, this equals the Gelbrich distance and can be
        written in terms of the Bures metric on SPD matrices.
        """
        S1 = metric_2_Wasserstein._spd_correction(S1)
        S2 = metric_2_Wasserstein._spd_correction(S2)

        S2h = sqrtm(S2)
        M = S2h @ S1 @ S2h
        M = np.real_if_close(M, tol=1e-8)
        Mhalf = sqrtm(M)
        Mhalf = np.real_if_close(Mhalf, tol=1e-8)

        val = np.trace(S1) + np.trace(S2) - 2.0 * np.trace(Mhalf)
        val = float(np.maximum(val, 0.0))
        return np.sqrt(val)

    @staticmethod
    def w2_gaussian(mu1: np.ndarray, S1: np.ndarray,
                    mu2: np.ndarray, S2: np.ndarray) -> float:
        """
        Full 2-Wasserstein (Gelbrich) distance between Gaussian
        N(mu1, S1) and N(mu2, S2):

            W^2 = ||mu1 - mu2||^2 + tr(S1 + S2 - 2 (S2^{1/2} S1 S2^{1/2})^{1/2})

        This is exactly Theorem 4.2(1)(2)(3) in the paper when both are normal.
        """
        mu1 = np.atleast_1d(mu1).astype(float)
        mu2 = np.atleast_1d(mu2).astype(float)
        d_mu2 = float(np.dot(mu1 - mu2, mu1 - mu2))

        w2_cov = metric_2_Wasserstein.w2_gaussian_zero_mean(S1, S2)
        return float(np.sqrt(d_mu2 + w2_cov**2))

    # -------------------------------------------------------------------------
    # init
    # -------------------------------------------------------------------------

    def __init__(self,
                 gamma: float,
                 n: int = None, var: float = None, alpha: float = 1.5, ellipse: bool = False, AfterBefore: bool = False):
        """
        Parameters
        ----------
        Sigma_nom : (n, n) SPD array
            Covariance matrix of the fixed nominal normal distribution P_nom.
        gamma : float
            Radius of the Wasserstein ball: W(P_w(t), P_nom) <= gamma.
        mode : {"independent", "correlated"}
            Which ambiguity set we use to generate processes:
            - "independent" -> W_ind
            - "correlated"  -> W_cor via AR(1).
        T : int
            Default time horizon.
        ts : float
            Sampling time (just stored for convenience).
        rng : np.random.Generator, optional
            RNG used for all sampling.
        """

        p = cfg.get("params", {})
        set = p.get("ambiguity", {})
        sim = p.get("simulation", {})
        self.var = float(set.get("var", 1)) if var is None else var

        Sigma_nom = np.array(set["Sigma_nom"], dtype=float) if n is None else self.var * np.eye(n)
        gamma = float(set.get("gamma", 0.5)) if gamma is None else gamma
        Tf = sim.get("TotTime", 100)
        ts = sim.get("ts", 0.5)
        
        mode = p.get("model", "independent")

        Sigma_nom = np.asarray(Sigma_nom, float)
        self.Sigma_nom = self._spd_correction(Sigma_nom)
        self.Nw = self.n = self.Sigma_nom.shape[0]
        self.mu_nom = np.zeros(self.n)
        self.gamma = float(gamma)

        assert mode in ("independent", "correlated")
        self.mode = mode
        self.T = int(Tf / ts)
        self.ts = float(ts)
        self.time = np.arange(self.T) * self.ts

        self.rng = np.random.default_rng()
        self.ellipse = ellipse
        self.AfterBefore= AfterBefore

        Sigma_raw = np.array([[0.68999448, 0.52701602],[0.52701602, 1.61000552],]) if self.AfterBefore else self.make_random_spd_around_nom(alpha=alpha)
        self.Sigma_test = self.project_zero_mean_cov_to_ball(Sigma_raw)

        self.gamma_estm = None
        self.Sigma_estm = None
        self.percent = set.get("percent", 1)


    # -------------------------------------------------------------------------
    # membership tests (Gaussian marginal)
    # -------------------------------------------------------------------------

    def w2_to_nominal(self, mu: np.ndarray, Sigma: np.ndarray) -> float:
        """W_2( N(mu, Sigma), P_nom )."""
        return self.w2_gaussian(mu, Sigma, self.mu_nom, self.Sigma_nom)

    def is_marginal_in_ball(self,
                            mu: np.ndarray,
                            Sigma: np.ndarray,
                            tol: float = 1e-10) -> bool:
        """
        Check whether the Gaussian N(mu, Sigma) satisfies W(P, P_nom) <= gamma.
        This is exactly the condition defining W_cor / W_ind in the paper.
        """
        d = self.w2_to_nominal(mu, Sigma)
        return d <= self.gamma + tol

    def make_random_spd_around_nom(self, alpha: float = 1.5) -> np.ndarray:
        """
        Build a random SPD covariance 'around' Sigma_nom.
        alpha controls how far we move away.
        """
        n = self.n
        A = self.rng.standard_normal((n, n))
        # random SPD core
        S = A @ A.T
        # normalize scale roughly to match Sigma_nom
        S = S / np.trace(S) * np.trace(self.Sigma_nom)
        # blend with Sigma_nom
        Sigma_test = (1 - alpha) * self.Sigma_nom + alpha * S
        # clean up numerically
        Sigma_test = self._spd_correction(Sigma_test)
        return Sigma_test

    # -------------------------------------------------------------------------
    # projection of zero-mean Gaussian to the W2-ball (covariance geodesic)
    # -------------------------------------------------------------------------

    def project_zero_mean_cov_to_ball(self,
                                      Sigma: np.ndarray,
                                      tol: float = 1e-7,
                                      maxit: int = 60) -> np.ndarray:
        """
        Project SPD Sigma (zero-mean Gaussian) onto the closed W2-ball
        { N(0, S) : W_2(N(0,S), N(0,Sigma_nom)) <= gamma }.

        We move along the Bures geodesic from Sigma towards Sigma_nom until
        the boundary W_2 = gamma is hit. For zero-mean Gaussians this respects
        the structure in the paper (Theorem 4.2(3)).
        """
        Sigma = self._spd_correction(Sigma)

        if self.w2_gaussian_zero_mean(Sigma, self.Sigma_nom) <= self.gamma:
            return Sigma

        Sigh = sqrtm(Sigma)
        Sigh = np.real_if_close(Sigh, tol=1e-8)
        M = Sigh @ self.Sigma_nom @ Sigh
        M = np.real_if_close(M, tol=1e-8)
        Ah = sqrtm(M)
        Ah = np.real_if_close(Ah, tol=1e-8)

        def cov_on_geodesic(t: float) -> np.ndarray:
            T_ = (1.0 - t) * Sigh + t * Ah
            G = T_ @ T_
            return self._spd_correction(G)

        lo, hi = 0.0, 1.0
        for _ in range(maxit):
            mid = 0.5 * (lo + hi)
            G = cov_on_geodesic(mid)
            d = self.w2_gaussian_zero_mean(G, self.Sigma_nom)
            if d > self.gamma:
                lo = mid
            else:
                hi = mid
            if hi - lo < tol:
                break
        return cov_on_geodesic(hi)

    # -------------------------------------------------------------------------
    # empirical helpers
    # -------------------------------------------------------------------------

    @staticmethod
    def empirical_mean_and_cov(w: np.ndarray,
                               unbiased: bool = True) -> tuple[np.ndarray, np.ndarray]:
        """
        Given samples w shape (T, n), compute empirical mean and covariance
        (L2 sense, as in the paper).
        """
        w = np.asarray(w, float)
        mu = w.mean(axis=0)
        z = w - mu
        denom = max(1, (w.shape[0] - 1) if unbiased else w.shape[0])
        S = z.T @ z / denom
        S = metric_2_Wasserstein._spd_correction(S)
        return mu, S

    def empirical_marginal_in_ball(self, w: np.ndarray, tol: float = 1e-10) -> bool:
        """
        Take a trajectory w(t) and check whether its empirical marginal
        distribution is inside the W2-ball wrt P_nom.
        """
        mu_hat, S_hat = self.empirical_mean_and_cov(w, unbiased=True)
        d = self.w2_to_nominal(mu_hat, S_hat)
        return d <= self.gamma + tol
    
    def _plot_AfterBefore(self, s0, s):
        """
        s0, s: arrays of shape (N, d), with d >= 2.
        First figure: histogram of ||w|| for s0 and s, with mean/std in legend.
        Second figure: scatter of w[0] vs w[1] for s0 and s.
        """
        s0 = np.asarray(s0)
        s  = np.asarray(s)

        if s0.ndim != 2 or s.ndim != 2:
            raise ValueError("s0 and s must be 2D arrays (N, d).")
        if s0.shape[1] < 2 or s.shape[1] < 2:
            raise ValueError("s0 and s must have at least 2 columns for w[0], w[1].")
        if s0.shape[0] == 0 or s.shape[0] == 0:
            raise ValueError("s0 and s must be non-empty.")

        # ---------- Figure 1: distribution of norms ----------
        r0 = np.linalg.norm(s0, axis=1)
        r  = np.linalg.norm(s,  axis=1)

        mu0, std0 = float(np.mean(r0)), float(np.std(r0))
        mu,  std  = float(np.mean(r)),  float(np.std(r))

        r_min = min(r0.min(), r.min())
        r_max = max(r0.max(), r.max())
        # avoid degenerate bins
        if r_max == r_min:
            r_max = r_min + 1e-6
        bins = np.linspace(r_min, r_max, 30)

        fig1, ax1 = plt.subplots(figsize=(6, 4))
        ax1.hist(
            r0,
            bins=bins,
            density=True,
            alpha=0.5,
            color="lightblue",
            label=fr"$s_0$: $\mu={mu0:.3g}$, $\sigma={std0:.3g}$",
        )
        ax1.hist(
            r,
            bins=bins,
            density=True,
            alpha=0.5,
            color="lightcoral",
            label=fr"$s$: $\mu={mu:.3g}$, $\sigma={std:.3g}$",
        )
        ax1.set_xlabel(r"$\|w\|$")
        ax1.set_ylabel("Density")
        ax1.set_title("Distribution of $s_0$ and $s$")
        ax1.grid(True, alpha=0.3)
        ax1.legend()

        # ---------- Figure 2: scatter w[0] vs w[1] ----------
        fig2, ax2 = plt.subplots(figsize=(6, 5))
        ax2.scatter(
            s0[:, 0],
            s0[:, 1],
            alpha=0.5,
            color="lightblue",
            edgecolors="none",
            label=r"$s_0$",
        )
        ax2.scatter(
            s[:, 0],
            s[:, 1],
            alpha=0.5,
            color="lightcoral",
            edgecolors="none",
            label=r"$s$",
        )
        ax2.set_xlabel(r"$w_0$")
        ax2.set_ylabel(r"$w_1$")
        ax2.set_title(r"$w_0$ vs $w_1$ for $s_0$ and $s$")
        ax2.grid(True, alpha=0.3)
        ax2.legend()

        plt.tight_layout()
        plt.show()

    def plot_AfterBefore(self, s0, s, max_dims_hist=6, bins=30, save_path=False):
        """
        Plot marginal distributions and 2D scatter for s0 (before) and s (after).

        s0, s : arrays with shape (n_w, T) or (T, n_w). Will be auto-oriented so
                that shape is (n_w, T), with n_w >= 2.
        """

        def _prep(W):
            W = np.asarray(W)
            if W.ndim != 2:
                raise ValueError("s0 and s must be 2D arrays.")
            n0, n1 = W.shape
            # heuristic: if rows > cols, assume (T, n_w) and transpose
            if n0 > n1:
                W = W.T
            n_w, T = W.shape
            return W, n_w, T

        W0, n_w0, T0 = _prep(s0)
        W,  n_w,  T  = _prep(s)

        if n_w0 != n_w:
            raise ValueError(f"s0 and s must have same number of components. "
                            f"Got {n_w0} and {n_w}.")
        if n_w < 1:
            raise ValueError("Need at least one component.")
        if n_w < 2:
            # then the scatter plot makes no sense
            raise ValueError("Need at least 2 components for w[0] vs w[1] scatter.")

    # --------- statistics for each set ----------
        # shape: (n_w,)
        mu0    = np.mean(W0, axis=1)
        Sigma0 = np.cov(W0, bias=False)
        std0   = np.sqrt(np.diag(Sigma0))

        mu     = np.mean(W, axis=1)
        Sigma  = np.cov(W, bias=False)
        std    = np.sqrt(np.diag(Sigma))

        # --------- Figure 1: univariate marginals, overlapping s0/s ----------
        n_hist = min(n_w, max_dims_hist)
        ncols = 3 if n_hist >= 3 else n_hist
        nrows = int(np.ceil(n_hist / ncols)) if n_hist else 0

        figs = []
        if n_hist > 0:
            fig1, axes = plt.subplots(
                nrows, ncols,
                figsize=(4.5 * ncols, 3.0 * nrows),
                squeeze=False
            )

            for i in range(n_hist):
                r, c = divmod(i, ncols)
                ax = axes[r][c]

                xi0 = W0[i, :]
                xi  = W[i, :]

                # common bin range across s0 and s for component i
                xmin = min(np.percentile(xi0, 0.5), np.percentile(xi, 0.5))
                xmax = max(np.percentile(xi0, 99.5), np.percentile(xi, 99.5))
                if xmax == xmin:
                    xmax = xmin + 1e-6

                bin_edges = np.linspace(xmin, xmax, bins)

                # histograms (density)
                ax.hist(
                    xi0,
                    bins=bin_edges,
                    density=True,
                    alpha=0.5,
                    edgecolor="none",
                    color="lightblue",
                )
                ax.hist(
                    xi,
                    bins=bin_edges,
                    density=True,
                    alpha=0.5,
                    edgecolor="none",
                    color="lightcoral",
                )

                xs = np.linspace(xmin, xmax, 400)

                # Gaussian fit for s0
                if std0[i] > 0:
                    pdf0 = (1.0 / (np.sqrt(2 * np.pi) * std0[i])) * np.exp(
                        -0.5 * ((xs - mu0[i]) / std0[i]) ** 2
                    )
                    ax.plot(
                        xs, pdf0,
                        linewidth=1.8,
                        color="lightblue",
                        label=fr"$s_0$: $\mu={mu0[i]:.3g}$, $\sigma={std0[i]:.3g}$"
                )

                # Gaussian fit for s
                if std[i] > 0:
                    pdf = (1.0 / (np.sqrt(2 * np.pi) * std[i])) * np.exp(
                        -0.5 * ((xs - mu[i]) / std[i]) ** 2
                    )
                    ax.plot(
                        xs, pdf,
                        linewidth=1.8,
                        color="lightcoral",
                        label=fr"$s$: $\mu={mu[i]:.3g}$, $\sigma={std[i]:.3g}$"
                    )

                ax.set_title(f"w[{i}]")
                ax.set_ylabel("Density")
                ax.grid(True, alpha=0.3)
                ax.legend(fontsize=8)

            # Hide empty subplots
            for j in range(n_hist, nrows * ncols):
                r, c = divmod(j, ncols)
                fig1.delaxes(axes[r][c])

            fig1.suptitle("Empirical marginals of disturbance components (before vs after)")
            fig1.tight_layout(rect=[0, 0, 1, 0.96])
            if save_path is not None:
                fig1.savefig(f"{save_path}__marginals_before_after.png",
                            dpi=150, bbox_inches="tight")
            figs.append(fig1)

        # --------- Figure 2: scatter of w[0] vs w[1] ----------
        fig2, ax2 = plt.subplots(figsize=(6, 5))
        ax2.scatter(
            W0[0, :],
            W0[1, :],
            alpha=0.5,
            color="lightblue",
            edgecolors="none",
            label=r"$s_0$",
        )
        ax2.scatter(
            W[0, :],
            W[1, :],
            alpha=0.5,
            color="lightcoral",
            edgecolors="none",
            label=r"$s$",
        )
        ax2.set_xlabel(r"$w[0]$")
        ax2.set_ylabel(r"$w[1]$")
        ax2.set_title(r"$w[0]$ vs $w[1]$ (before vs after)")
        ax2.grid(True, alpha=0.3)
        ax2.legend()

        fig2.tight_layout()
        if save_path is not None:
            fig2.savefig(f"{save_path}__scatter_w0_w1_before_after.png",
                        dpi=150, bbox_inches="tight")
        figs.append(fig2)
        plt.show()

        return tuple(figs)


    # -------------------------------------------------------------------------
    # Define ellipse
    # -------------------------------------------------------------------------

    def _confidence_ellipse(self, ax, cov2, mean2=None, nsig=2.0, label=None, lw=1.5, ls='-'):
        """
        Draw an nsig-sigma ellipse for a 2x2 covariance 'cov2' centered at 'mean2'.
        """
        if mean2 is None:
            mean2 = np.zeros(2)
        # eigen-decomp
        vals, vecs = np.linalg.eigh(cov2)
        # sort desc
        order = np.argsort(vals)[::-1]
        vals, vecs = vals[order], vecs[:, order]
        # param
        theta = np.linspace(0, 2*np.pi, 400)
        circle = np.vstack((np.cos(theta), np.sin(theta)))  # 2xM
        # nsig scaling (chi-square with 2 dof: radius^2 = nsig^2 is fine for visualization)
        L = vecs @ np.diag(np.sqrt(vals)) @ (nsig * np.eye(2))
        pts = (L @ circle).T + mean2  # Mx2
        ax.plot(pts[:,0], pts[:,1], linewidth=lw, linestyle=ls, label=label)

    def _w2_boundary_covariances(self, Sigma_nom, gamma):
        """
        Produce a few SPD covariances on the W2 boundary around Sigma_nom, sharing eigenvectors.
        For each principal axis k: set sqrt(lambda_k)' = max(0, sqrt(lambda_k) ± gamma),
        others unchanged. Return list of (Sigma_boundary, tag).
        """
        # eigendecompose Sigma_nom (assume SPD)
        d, U = np.linalg.eigh(0.5*(Sigma_nom + Sigma_nom.T))
        d = np.maximum(d, 1e-15)
        s = np.sqrt(d)
        out = []
        for k in range(len(d)):
            for sign, tag in [(+1, f"+γ along axis {k}"), (-1, f"-γ along axis {k}")]:
                s_new = s.copy()
                s_new[k] = max(1e-15, s[k] + sign*gamma)
                d_new = s_new**2
                Sig_new = U @ np.diag(d_new) @ U.T
                out.append((Sig_new, tag))
        return out

    def plot_samples_with_wasserstein_bounds(
        self, w, Sigma_nom, dims=(0,1), nsig=2.0, show_empirical=True, max_boundary=4
    ):
        """
        Scatter the samples w[:, dims] and overlay:
        - nominal nsig-σ ellipse for Sigma_nom[dims,dims]
        - empirical nsig-σ ellipse (optional)
        - a few W2-boundary ellipses constructed by axis-wise eigenvalue shifts of Sigma_nom
        Notes:
        • This visualizes *marginal covariance* bounds, not a hard envelope for points.
        • Accurate for the commuting case (shared eigenvectors); still informative otherwise.
        """
        w = np.asarray(w)
        i, j = dims
        if i == j:
            raise ValueError("Pick two distinct dimensions for plotting.")
        # 2D marginal samples
        W2 = w[:, [i, j]]
        mu = W2.mean(axis=0)
        # 2D covariances
        Sig_nom_2d = Sigma_nom[np.ix_([i,j],[i,j])]
        Sig_emp_2d = np.cov(W2.T, bias=False)  # zero-mean vs sample mean doesn’t matter visually

        # Build a few 2D boundary covariances
        boundaries = self._w2_boundary_covariances(Sigma_nom, self.gamma)
        # Keep at most 'max_boundary' ellipses, but favor ones that actually change i or j axes
        chosen = []
        for Sig_b, tag in boundaries:
            Sig_b_2d = Sig_b[np.ix_([i,j],[i,j])]
            chosen.append((Sig_b_2d, tag))
            if len(chosen) >= max_boundary:
                break

        # Plot
        fig, ax = plt.subplots(figsize=(6.2, 6.2))
        ax.scatter(W2[:,0], W2[:,1], s=10, alpha=0.35, label="samples")
        self._confidence_ellipse(ax, Sig_nom_2d, mean2=mu, nsig=nsig, label=f"nominal (Σ_nom), {nsig}σ", lw=2.0)
        if show_empirical:
            self._confidence_ellipse(ax, Sig_emp_2d, mean2=mu, nsig=nsig, label=f"empirical, {nsig}σ", ls='--')

        for Sig_b_2d, tag in chosen:
            self._confidence_ellipse(ax, Sig_b_2d, mean2=mu, nsig=nsig, label=f"W2-boundary: {tag}", ls=':')

        ax.set_aspect('equal', adjustable='box')
        ax.set_xlabel(f"w[{i}]")
        ax.set_ylabel(f"w[{j}]")
        ax.set_title(f"Samples and W2(·, Σ_nom) ≤ γ bounds (γ={self.gamma}, dims={dims})")
        ax.legend(loc="best", fontsize=8)
        plt.tight_layout()
        #plt.show()
        return ax

    def plot_all_pairs(self, w, Sigma_nom, max_pairs=6):
        n = w.shape[1]
        pairs = [(i, j) for i in range(n) for j in range(i+1, n)]
        for (i, j) in pairs[:max_pairs]:
            self.plot_samples_with_wasserstein_bounds(w, Sigma_nom, dims=(i, j))

    # -------------------------------------------------------------------------
    # sampling utilities (Gaussian elements of the ambiguity sets)
    # -------------------------------------------------------------------------

    def sample(self,
               T: int | None = None,
               Sigma: np.ndarray | None = None,
               rho: float = 0.9) -> np.ndarray:
        """
        Sample a disturbance trajectory of length T from a Gaussian distribution
        lying in the Wasserstein ball, using the selected mode:

        mode = "independent":
            w(t) iid ~ N(0, Sigma_use).
        mode = "correlated":
            AR(1): w_{t+1} = F w_t + eps_t with stationary marginal N(0, Sigma_use).

        Sigma_init:
            candidate covariance. It will be projected into the ball.
            If None, we use Sigma_nom (center of the ambiguity set).
        """
        T = self.T if T is None else int(T)
        Sigma_use = self.Sigma_test #if Sigma is None else self.project_zero_mean_cov_to_ball(Sigma)

        if self.mode == "independent":
            s0, s = self._sample_iid_gaussian(T, Sigma_use)
        else:
            s0, s = self._sample_correlated_gaussian(T, Sigma_use, rho=rho)

        if self.ellipse: 
            #self.plot_samples_with_wasserstein_bounds(s, S)
            self.plot_all_pairs(s, Sigma_use) #self.Sigma_nom)
            plt.show()
        
        if self.AfterBefore: 
            self.plot_AfterBefore(s0, s)
        return self.percent * s

    def _sample_iid_gaussian(self, T: int, Sigma: np.ndarray) -> np.ndarray:
        """
        Generate w(t) iid ~ N(0, Sigma), with Sigma already in the ball.
        This is one particular element of W_ind.
        """
        Sigma = self._spd_correction(Sigma)
        L = cholesky(Sigma)
        z = self.rng.standard_normal(size=(T, Sigma.shape[0]))
        return z, z @ L.T

    def _sample_correlated_gaussian(self,
                                    T: int,
                                    Sigma: np.ndarray,
                                    rho: float = 0.9) -> np.ndarray:
        """
        Generate an AR(1) process with stationary marginal N(0, Sigma):

            w_{t+1} = F w_t + eps_t,    F = rho I,   eps_t ~ N(0, Q),

        where Q solves the discrete-time Lyapunov equation

            Sigma = F Sigma F^T + Q.

        Since the marginal at every t is N(0, Sigma) and Sigma is in the ball,
        this trajectory belongs to W_cor.
        """
        Sigma = self._spd_correction(Sigma)
        n = Sigma.shape[0]
        F = rho * np.eye(n)
        Q = self._sym(Sigma - F @ Sigma @ F.T)
        Q = self._spd_correction(Q)

        LQ = cholesky(Q)
        Ls = cholesky(Sigma)

        w = np.zeros((T, n))
        # initialize in the stationary distribution
        w[0] = self.rng.standard_normal(n) @ Ls.T
        for t in range(T - 1):
            eps = self.rng.standard_normal(n) @ LQ.T
            w[t + 1] = F @ w[t] + eps

        return self.rng.standard_normal(size=(T, n)), w

    # -------------------------------------------------------------------------
    #  NEW: estimate gamma (radius) and Pw
    # -------------------------------------------------------------------------

    def estm_Sigma_nom(self, w: np.ndarray, unbiased: bool = True) -> np.ndarray:
        """
        Estimate Sigma_nom from disturbance data w (T, n).

        This uses the empirical covariance of (w - mean(w)), consistent with
        Sigma_nom := ∫ v v^T dP_nom(v) for a zero-mean nominal distribution.

        Returns the estimate and updates self.Sigma_nom, self.mu_nom.
        """
        w = np.asarray(w, float)
        _, S_hat = self.empirical_mean_and_cov(w, unbiased=unbiased)

        #self.Sigma_nom = S_hat
        self.mu_nom = np.zeros(self.n)
        self.Sigma_estm = S_hat
        return S_hat

    def _estimate_gamma_with_ci(self,
                                w: np.ndarray,
                                include_mean: bool = False,
                                unbiased: bool = True,
                                set_internal: bool = True) -> float:
        """
        Estimate the Wasserstein radius gamma from disturbance samples w (T, n),
        assuming that (mu_nom, Sigma_nom) have ALREADY been set, e.g. via
        estm_Sigma_nom(w_nom) on some (possibly different) data.

        This uses a simple plug-in estimator:
            gamma_hat = W_2( N(mu_hat, Sigma_hat), N(mu_nom, Sigma_nom) ).

        Parameters
        ----------
        w : array (T, n)
            Disturbance time series from the 'true' Pw.
        include_mean : bool
            If True, include the mean shift in W_2, otherwise assume both zero-mean.
        unbiased : bool
            Use 1/(T-1) vs 1/T for covariance.
        set_internal : bool
            If True, store gamma_hat in self.gamma.

        Returns
        -------
        gamma_hat : float
        """
        w = np.asarray(w, float)
        T, n = w.shape
        if n != self.n: 
            Sigma_nom = self.var * np.eye(n)
        else:
            Sigma_nom = self.Sigma_nom

        if self.Sigma_estm is None: self.estm_Sigma_nom(w)

        if include_mean:
            gamma_hat = self.w2_gaussian(self.mu_hat, self.Sigma_estm, self.mu_nom, Sigma_nom)
        else:
            gamma_hat = self.w2_gaussian_zero_mean(self.Sigma_estm, Sigma_nom)

        if set_internal:
            self.gamma = float(gamma_hat)

        return float(gamma_hat), 

    def estimate_gamma_with_ci(self, w, beta=0.1):
        """
        Heuristic data-driven W2 radius γ_N(β) for DRO.
        w : (N, d) array of disturbance samples.
        beta : confidence tail probability (e.g. 0.1 for 90%).
        """
        N, d = w.shape

        # Scale estimates
        m2 = np.mean(np.sum(w**2, axis=1))  # E[||w||^2]
        s_w = np.sqrt(m2)                   # typical magnitude

        # Dimension-dependent exponent alpha
        alpha = max(d / 2.0, 2.0)

        # Heuristic constants (you can tune these)
        c0 = 7.5
        C0 = 10.0

        radius = c0 * s_w * ((np.log(C0 / beta) / N) ** (1.0 / alpha))
        return float(radius),

# =====================================================================================

class WassersteinAmbiguitySet:
    """
    2-Wasserstein ambiguity set around N(0, Sigma_nom) with radius gamma.
    - W_ind: iid disturbances with that marginal
    - W_cor: arbitrary temporal correlation but each marginal in the ball
    """
    def __init__(self, gamma: float = None, ellipse: bool = False, n: int = None, var: float = None, alpha: float = 1.5):

        p = cfg.get("params", {})
        set = p.get("ambiguity", {})
        sim = p.get("simulation", {})
        Sigma_nom = np.array(set["Sigma_nom"], dtype=float) if n is None or var is None else var * np.eye(n)
        gamma = float(set.get("gamma", 0.5)) if gamma is None else gamma
        Tf = sim.get("TotTime", 100)
        self.ts = sim.get("ts", 0.5)
        
        self.mode = p.get("model", "independent")
        self.T = int(Tf / self.ts)
        self.Sigma_nom = 0.5 * (Sigma_nom + Sigma_nom.T)
        self.gamma = gamma
        self.rng = np.random.default_rng()
        self._chol_nom = None
        self.time = np.arange(self.T)*self.ts
        self.ellipse = ellipse

        rng = np.random.default_rng()
        M = rng.uniform(low=-alpha/8, high=alpha/8, size=(n, n))
        K = M - M.T # skew-symmetric
        self.Sigma_test = alpha * np.eye(n) + K #(1 + gamma/np.sqrt(n))**2 * np.eye(n)
        self.Nw = n


    @staticmethod
    def _sym(A):
        return 0.5 * (A + A.T)

    @staticmethod
    def w2_gaussian(S1: np.ndarray, S2: np.ndarray) -> float:
        """W2(N(0,S1), N(0,S2)) for SPD S1, S2."""
        S1 = 0.5 * (S1 + S1.T)
        S2 = 0.5 * (S2 + S2.T)
        S2h = sqrtm(S2)
        # print(S2h.size, S1.size, S2.size); sys.exit(0)
        mid = sqrtm(S2h @ S1 @ S2h)
        # numerical cleanup: sqrtm returns complex with tiny imag parts sometimes
        mid = np.real_if_close(mid, tol=1e-8)
        val = np.trace(S1) + np.trace(S2) - 2.0 * np.trace(mid)
        val = float(np.maximum(val, 0.0))
        return np.sqrt(val)

    def is_member_gaussian(self, Sigma: np.ndarray) -> bool:
        """Check if N(0,Sigma) lies in the ball of radius gamma around N(0,Sigma_nom)."""
        d = self.w2_gaussian(Sigma, self.Sigma_nom)
        return d <= self.gamma + 1e-10

    def project_cov_to_ball(self, Sigma: np.ndarray, tol: float = 1e-7, maxit: int = 60) -> np.ndarray:
        """
        Project SPD Sigma onto the closed Bures/W2 ball B(Sigma_nom, gamma) by moving along the
        Bures geodesic toward Sigma_nom until the boundary is hit.
        This is not the unique projection in general, but it’s a valid contraction that enforces membership.
        """
        if self.is_member_gaussian(Sigma):
            return self._sym(Sigma)

        # Bures geodesic: G(t) = Sigma^(1/2) ( (1-t)^2 I + t(2-t) A ) Sigma^(1/2),
        # where A solves Sigma^(1/2) A Sigma^(1/2) = Sigma^(1/2) Sigma_nom Sigma^(1/2).
        # We'll use the equivalent path implemented via the optimal transport map.
        Sigma = self._sym(Sigma)
        Sigh = sqrtm(Sigma); Sigh = np.real_if_close(Sigh, tol=1e-8)
        M = Sigh @ self.Sigma_nom @ Sigh
        Ah = sqrtm(M); Ah = np.real_if_close(Ah, tol=1e-8)

        # Define geodesic cov: G(t) = ((1-t) * Sigh + t * Ah) @ ((1-t) * Sigh + t * Ah)
        def cov_on_geodesic(t):
            T = (1.0 - t) * Sigh + t * Ah
            G = T @ T
            return self._sym(np.real_if_close(G, tol=1e-8))

        # Bisection on t in [0,1] to hit distance = gamma
        lo, hi = 0.0, 1.0
        for _ in range(maxit):
            mid = 0.5 * (lo + hi)
            G = cov_on_geodesic(mid)
            d = self.w2_gaussian(G, self.Sigma_nom)
            if d > self.gamma:
                lo = mid
            else:
                hi = mid
            if hi - lo < tol:
                break
        return cov_on_geodesic(hi)


    # ---------- Define ellipse --------------

    def _confidence_ellipse(self, ax, cov2, mean2=None, nsig=2.0, label=None, lw=1.5, ls='-'):
        """
        Draw an nsig-sigma ellipse for a 2x2 covariance 'cov2' centered at 'mean2'.
        """
        if mean2 is None:
            mean2 = np.zeros(2)
        # eigen-decomp
        vals, vecs = np.linalg.eigh(cov2)
        # sort desc
        order = np.argsort(vals)[::-1]
        vals, vecs = vals[order], vecs[:, order]
        # param
        theta = np.linspace(0, 2*np.pi, 400)
        circle = np.vstack((np.cos(theta), np.sin(theta)))  # 2xM
        # nsig scaling (chi-square with 2 dof: radius^2 = nsig^2 is fine for visualization)
        L = vecs @ np.diag(np.sqrt(vals)) @ (nsig * np.eye(2))
        pts = (L @ circle).T + mean2  # Mx2
        ax.plot(pts[:,0], pts[:,1], linewidth=lw, linestyle=ls, label=label)

    def _w2_boundary_covariances(self, Sigma_nom, gamma):
        """
        Produce a few SPD covariances on the W2 boundary around Sigma_nom, sharing eigenvectors.
        For each principal axis k: set sqrt(lambda_k)' = max(0, sqrt(lambda_k) ± gamma),
        others unchanged. Return list of (Sigma_boundary, tag).
        """
        # eigendecompose Sigma_nom (assume SPD)
        d, U = np.linalg.eigh(0.5*(Sigma_nom + Sigma_nom.T))
        d = np.maximum(d, 1e-15)
        s = np.sqrt(d)
        out = []
        for k in range(len(d)):
            for sign, tag in [(+1, f"+γ along axis {k}"), (-1, f"-γ along axis {k}")]:
                s_new = s.copy()
                s_new[k] = max(1e-15, s[k] + sign*gamma)
                d_new = s_new**2
                Sig_new = U @ np.diag(d_new) @ U.T
                out.append((Sig_new, tag))
        return out

    def plot_samples_with_wasserstein_bounds(
        self, w, Sigma_nom, dims=(0,1), nsig=2.0, show_empirical=True, max_boundary=4
    ):
        """
        Scatter the samples w[:, dims] and overlay:
        - nominal nsig-σ ellipse for Sigma_nom[dims,dims]
        - empirical nsig-σ ellipse (optional)
        - a few W2-boundary ellipses constructed by axis-wise eigenvalue shifts of Sigma_nom
        Notes:
        • This visualizes *marginal covariance* bounds, not a hard envelope for points.
        • Accurate for the commuting case (shared eigenvectors); still informative otherwise.
        """
        w = np.asarray(w)
        i, j = dims
        if i == j:
            raise ValueError("Pick two distinct dimensions for plotting.")
        # 2D marginal samples
        W2 = w[:, [i, j]]
        mu = W2.mean(axis=0)
        # 2D covariances
        Sig_nom_2d = Sigma_nom[np.ix_([i,j],[i,j])]
        Sig_emp_2d = np.cov(W2.T, bias=False)  # zero-mean vs sample mean doesn’t matter visually

        # Build a few 2D boundary covariances
        boundaries = self._w2_boundary_covariances(Sigma_nom, self.gamma)
        # Keep at most 'max_boundary' ellipses, but favor ones that actually change i or j axes
        chosen = []
        for Sig_b, tag in boundaries:
            Sig_b_2d = Sig_b[np.ix_([i,j],[i,j])]
            chosen.append((Sig_b_2d, tag))
            if len(chosen) >= max_boundary:
                break

        # Plot
        fig, ax = plt.subplots(figsize=(6.2, 6.2))
        ax.scatter(W2[:,0], W2[:,1], s=10, alpha=0.35, label="samples")
        self._confidence_ellipse(ax, Sig_nom_2d, mean2=mu, nsig=nsig, label=f"nominal (Σ_nom), {nsig}σ", lw=2.0)
        if show_empirical:
            self._confidence_ellipse(ax, Sig_emp_2d, mean2=mu, nsig=nsig, label=f"empirical, {nsig}σ", ls='--')

        for Sig_b_2d, tag in chosen:
            self._confidence_ellipse(ax, Sig_b_2d, mean2=mu, nsig=nsig, label=f"W2-boundary: {tag}", ls=':')

        ax.set_aspect('equal', adjustable='box')
        ax.set_xlabel(f"w[{i}]")
        ax.set_ylabel(f"w[{j}]")
        ax.set_title(f"Samples and W2(·, Σ_nom) ≤ γ bounds (γ={self.gamma}, dims={dims})")
        ax.legend(loc="best", fontsize=8)
        plt.tight_layout()
        #plt.show()
        return ax

    def plot_all_pairs(self, w, Sigma_nom, max_pairs=6):
        n = w.shape[1]
        pairs = [(i, j) for i in range(n) for j in range(i+1, n)]
        for (i, j) in pairs[:max_pairs]:
            self.plot_samples_with_wasserstein_bounds(w, Sigma_nom, dims=(i, j))


    # ---------- sampling utilities ----------

    def sample(self, T:int = None, Sigma: np.ndarray = None) -> np.ndarray:
        """
        Sample a disturbance sequence of length T from the ambiguity set.
        Uses the specified mode ("independent" or "correlated").
        Returns array shape (T, n).
        """
        T = T if T is not None else self.T
        if self.mode == "independent":
            s, S = self.sample_iid(T=T, Sigma=Sigma)
        else:
            s, S = self.sample_correlated(T=T, Sigma=Sigma)
        
        if self.ellipse: 
            #self.plot_samples_with_wasserstein_bounds(s, S)
            self.plot_all_pairs(s, S) #self.Sigma_nom)
            plt.show()
        return s


    def sample_iid(self, T: int, Sigma: np.ndarray = None) -> np.ndarray:
        """
        Sample an iid sequence in W_ind.
        If Sigma is provided, it will be projected into the ball first.
        Otherwise Sigma_nom is used (on the boundary only if gamma=0).
        Returns array shape (T, n).
        """
        if Sigma is None:
            Sigma_use = self.Sigma_test
        else:
            Sigma_use = self.project_cov_to_ball(Sigma)

        L = cholesky(self._sym(Sigma_use))
        n = L.shape[0]
        z = self.rng.standard_normal(size=(T, n))
        return z @ L.T, Sigma_use

    def sample_correlated(self, T: int, rho: float = 0.9, Sigma: np.ndarray = None) -> np.ndarray:
        """
        Sample a correlated sequence in W_cor using an AR(1) model:
            w_{t+1} = F w_t + eps_t,   eps_t ~ N(0,Q)
        with stationary marginal covariance = Sigma_proj ∈ ball.
        """
        if Sigma is None:
            Sigma_target = self.Sigma_test
        else:
            #self.Sigma_nom = np.eye(Sigma[0].size)
            Sigma_target = self.project_cov_to_ball(Sigma)

        assert self.is_member_gaussian(Sigma_target), "Projected Sigma is not inside the W2-ball. Numerical issue."

        n = Sigma_target.shape[0]
        # stable isotropic dynamics
        F = rho * np.eye(n)
        # Solve for Q so that Sigma_target = F Sigma_target F^T + Q
        Q = self._sym(Sigma_target - F @ Sigma_target @ F.T)
        # numerical guard if tiny negatives
        evals, evecs = np.linalg.eigh(Q)
        Q = evecs @ np.diag(np.maximum(evals, 1e-12)) @ evecs.T

        LQ = cholesky(Q)
        w = np.zeros((T, n))
        # initialize from the stationary distribution
        Ls = cholesky(self._sym(Sigma_target))
        w[0] = self.rng.standard_normal(n) @ Ls.T
        for t in range(T - 1):
            eps = self.rng.standard_normal(n) @ LQ.T
            w[t + 1] = F @ w[t] + eps
        return w, Sigma_target

    # ---------- empirical utilities ----------

    def empirical_marginal_cov(self, w: np.ndarray) -> np.ndarray:
        """
        Given samples w shape (T, n), return sample covariance of the marginal (zero-mean enforced).
        For iid it’s the usual covariance. For correlated it’s still the marginal when T is large.
        """
        w0 = w - w.mean(axis=0, keepdims=True)
        return self._sym(w0.T @ w0 / max(1, w0.shape[0] - 1))

    def is_member_empirical(self, w: np.ndarray) -> bool:
        """Check membership by plugging in the sample covariance into the Gaussian W2 formula."""
        S = self.empirical_marginal_cov(w)
        return self.is_member_gaussian(S)
    
    # ---------- estimate gamma from data -----

    @staticmethod
    def w2_gaussian_full(mu1: np.ndarray, S1: np.ndarray,
                        mu2: np.ndarray, S2: np.ndarray) -> float:
        """
        W2 between Gaussians N(mu1,S1) and N(mu2,S2). Calls the existing
        w2_gaussian (which handles the covariance/Bures part) and adds the mean shift.
        """
        mu1 = np.atleast_1d(mu1).astype(float)
        mu2 = np.atleast_1d(mu2).astype(float)
        d_mu2 = float(np.dot(mu1 - mu2, mu1 - mu2))
        # reuse covariance-only distance
        w2_cov = WassersteinAmbiguitySet.w2_gaussian(S1, S2)
        return float(np.sqrt(d_mu2 + w2_cov**2))

    def estm_Sigma_nom(self, s):
        n, T = s.shape
        return 0.5 * ((s @ s.T)/max(T,1) + ((s @ s.T)/max(T,1)).T) + 1e-9*np.eye(n)

    def empirical_mean_and_cov(self, w: np.ndarray, unbiased_cov: bool = True):
        """
        Empirical mean and covariance of samples w shape (T,n).
        unbiased_cov=False uses 1/T, True uses 1/(T-1).
        """
        w = np.asarray(w, float)
        mu = w.mean(axis=0)
        z = w - mu
        denom = max(1, (w.shape[0] - 1) if unbiased_cov else w.shape[0])
        S = self._sym(z @ z.T / denom)
        return mu, S

    def estimate_gamma_from_samples(self, w: np.ndarray, include_mean: bool = True):
        """
        Plug-in estimator of gamma: W2( N(mu_hat, Sigma_hat), N(0, Sigma_nom) ).
        If include_mean=False, ignores mean shift and uses the zero-mean formula.
        Returns (gamma_hat, diagnostics_dict).
        """
        mu_hat, S_hat = self.empirical_mean_and_cov(w, unbiased_cov=True)
        if include_mean:
            gamma_hat = self.w2_gaussian_full(mu_hat, S_hat, np.zeros_like(mu_hat), self.estm_Sigma_nom(w))
        else:
            gamma_hat = self.w2_gaussian(S_hat, self.Sigma_nom)
        diag = {"mu_hat": mu_hat, "Sigma_hat": S_hat, "include_mean": include_mean}
        return float(gamma_hat), diag

    def estimate_gamma_with_ci(self, w: np.ndarray, include_mean: bool = True,
                            correlated: bool = True, B: int = 300, alpha: float = 0.10,
                            block_len: int | None = None, rng: np.random.Generator | None = None):
        """
        Bootstrap CI for gamma. Uses ordinary bootstrap if iid; moving block bootstrap if correlated.
        - correlated=True -> moving block bootstrap with block_len ~ T**(1/3) if not given.
        Returns (gamma_hat, (lo, hi), diagnostics).
        """
        rng = np.random.default_rng() if rng is None else rng
        T, n = w.shape
        gamma_hat, _ = self.estimate_gamma_from_samples(w, include_mean=include_mean)

        # resampling helpers
        def draw_series_iid():
            idx = rng.integers(0, T, size=T)
            return w[idx]

        def draw_series_block():
            L = block_len or max(2, int(round(T**(1/3))))
            n_blocks = int(np.ceil(T / L))
            starts = rng.integers(0, max(1, T - L + 1), size=n_blocks)
            pieces = [w[s:s+L] for s in starts]
            ww = np.vstack(pieces)[:T]
            return ww

        boots = []
        draw = draw_series_block if correlated else draw_series_iid
        for _ in range(B):
            wb = draw()
            gb, _ = self.estimate_gamma_from_samples(wb, include_mean=include_mean)
            boots.append(gb)
        boots = np.array(boots, float)
        lo, hi = np.quantile(boots, [alpha/2, 1 - alpha/2])

        diag = {"B": B, "alpha": alpha, "boot_samples": boots, "correlated": correlated}
        self.gamma = gamma_hat
        return gamma_hat, (float(lo), float(hi)), diag

# =====================================================================================

class WithoutNoise:
    def __init__(self):
        p = cfg.get("params", {})
        sim = p.get("simulation", {})
        Tf = sim.get("TotTime", 100)
        self.ts = sim.get("ts", 0.5)
        
        self.mode = p.get("model", "independent")
        self.T = int(Tf / self.ts)
        self.Sigma_nom = None
        self.gamma = None
        self.rng = np.random.default_rng()
        self._chol_nom = None
        self.time = np.arange(self.T)*self.ts
        self.p = p

    def sample(self, T:int = None, Sigma: np.ndarray = None) -> np.ndarray:
        T = T if T is not None else self.T
        nw = Sigma[0].size if Sigma is not None else self.p.get("dimensions", {}).get("nw", 2)
        return np.zeros((T, nw))

    def is_member_empirical(self, w: np.ndarray) -> bool:
        return True

# =====================================================================================

class GaussianNoise:
    def __init__(self, n: int = None, var: float = None):
        p   = cfg.get("params", {})
        amb = p.get("ambiguity", {})
        sim = p.get("simulation", {})

        self.n          = p.get("dimensions", {}).get("nw", 2) if n is None else n
        self.var        = amb.get("var", 0) if var is None else var
        self.Sigma_nom  = np.array(amb["Sigma_nom"], dtype=float) if self.var==0 else self.var * np.eye(self.n)
        self.gamma      = None

        Tf          = float(sim.get("TotTime", 100.0))
        self.ts     = float(sim.get("ts", 0.5))
        self.T      = int(round(Tf / self.ts))
        self.time   = np.arange(self.T) * self.ts
        self.rng    = np.random.default_rng()

    def sample(self, T: int = None, Sigma: np.ndarray = None) -> np.ndarray:
        T = int(T) if T is not None else self.T
        Sigma = Sigma if Sigma is not None else self.Sigma_nom
        L = cholesky(0.5*(Sigma + Sigma.T))
        z = self.rng.standard_normal(size=(T, L.shape[0]))
        return z @ L.T
    
    def is_member_empirical(self, w: np.ndarray) -> bool:
        return True

# =====================================================================================

class Disturbances:
    """
    Thin delegating wrapper. Disturbances.impl holds the real object.
    You can call dist.sample(T=...) regardless of which model you selected.
    """
    def __init__(self, gamma: float = None, model: str = None, n: int = None, var: float = None, ellipse: bool = None, AfterBefore: bool = None):
        p   = cfg.get("params", {})
        amb = p.get("ambiguity", {})
        model = amb.get("model", "W2") if model is None else model
        self.impl = self._select(model, gamma, n, var, ellipse, AfterBefore)

        # convenience mirrors so your demo prints work
        self.mode = getattr(self.impl, "mode", model)
        self.gamma = getattr(self.impl, "gamma", gamma)
        self.Sigma_nom = getattr(self.impl, "Sigma_nom", None)
        self.Sigma_test = getattr(self.impl, "Sigma_test", None)
        self.time = getattr(self.impl, "time", None)
        self.ts = getattr(self.impl, "ts", None)
        self.T = getattr(self.impl, "T", None)

    def _select(self, model: str, gamma: float, n: int, var: float, ellipse: bool, AfterBefore: bool):
        if model == "W2":
            if gamma is not None and gamma < 0:
                raise ValueError("gamma must be nonnegative")
            return WassersteinAmbiguitySet(gamma=gamma, ellipse=ellipse, n=n, var=var)
        if model == "zero":
            return WithoutNoise()
        if model == "Gaussian":
            return GaussianNoise(n=n, var=var)
        if model == "2W":
            return metric_2_Wasserstein(gamma=gamma, n=n, var=var, ellipse=ellipse, AfterBefore=AfterBefore)
        raise ValueError(f"Unknown ambiguity model: {model}")

    def __getattr__(self, name: str):
        # delegate all other attributes/methods to the concrete implementation
        return getattr(self.impl, name)

    def __repr__(self):
        return f"Disturbances(mode={self.mode!r}, gamma={self.gamma!r})"

    def plot_disturbance_distribution(self, w, *, bins=40, max_dims_hist=8, plot_pairs=True,
                                    ellipse_levels=(1.0, 2.0, 3.0), save_path=None):
        """
        Plot empirical distribution of a disturbance w in R^{n_w x T}.

        Parameters
        ----------
        w : array_like
            Disturbance samples with shape (n_w, T) or (T, n_w). If T is mistaken
            for n_w, the function will transpose to get (n_w, T).
        bins : int
            Number of histogram bins for marginal plots.
        max_dims_hist : int
            Max number of dimensions to show as univariate histograms.
        plot_pairs : bool
            If True and n_w >= 2, also show a 2D scatter with covariance ellipses for (w_0, w_1).
        ellipse_levels : tuple of floats
            Sigma levels for covariance ellipses (e.g., 1,2,3).
        save_path : str or None
            If provided, saves the figure(s) to this path prefix. Files will be suffixed automatically.

        Returns
        -------
        stats : dict
            {"mu": mean (n_w,), "Sigma": covariance (n_w,n_w), "std": std (n_w,)}
        """    
        
        def _draw_cov_ellipses(ax, mu2, Sigma2, levels=(1.0, 2.0, 3.0)):
            """
            Draw covariance ellipses for a 2D Gaussian with mean mu2 and covariance Sigma2.
            levels are sigma radii (1,2,3), i.e., scale of the principal axes.
            """
            if Sigma2.shape != (2, 2):
                return
            # Eigen-decomposition
            vals, vecs = np.linalg.eigh(Sigma2)  # guaranteed symmetric
            vals = np.maximum(vals, 0.0)
            # Unit circle
            theta = np.linspace(0, 2*np.pi, 400)
            unit = np.vstack([np.cos(theta), np.sin(theta)])  # (2, N)

            for s in levels:
                # Scale along principal axes
                axes = vecs @ (np.sqrt(vals)[:, None] * unit)  # (2,N)
                pts = (mu2[:, None] + s * axes)                # (2,N)
                ax.plot(pts[0, :], pts[1, :], lw=1.2, alpha=0.9, label=f"{s}σ")
            # Avoid duplicate legend entries
            handles, labels = ax.get_legend_handles_labels()
            uniq = dict(zip(labels, handles))
            ax.legend(uniq.values(), uniq.keys(), loc="best", frameon=True)

        W = np.asarray(w)
        if W.ndim != 2:
            raise ValueError("w must be 2D, got shape {}".format(W.shape))

        # Ensure shape (n_w, T)
        n0, n1 = W.shape
        if n0 < n1:
            # ambiguous, but typical convention is (n_w, T); keep as is
            n_w, T = n0, n1
        else:
            # if rows >> cols, likely (T, n_w) -> transpose
            if n0 > n1:
                W = W.T
            n_w, T = W.shape

        # Sample mean/cov
        mu = np.mean(W, axis=1)                           # (n_w,)
        Sigma = np.cov(W, bias=False)                     # (n_w, n_w)
        std = np.sqrt(np.diag(Sigma))

        # ---------- Univariate marginals ----------
        n_hist = min(n_w, max_dims_hist)
        ncols = 3 if n_hist >= 3 else n_hist
        nrows = int(np.ceil(n_hist / ncols)) if n_hist else 0

        if n_hist > 0:
            fig1, axes = plt.subplots(nrows, ncols, figsize=(4.5*ncols, 3.0*nrows), squeeze=False)
            for i in range(n_hist):
                r, c = divmod(i, ncols)
                ax = axes[r][c]
                xi = W[i, :]
                # Histogram (density)
                ax.hist(xi, bins=bins, density=True, alpha=0.6, edgecolor="none")
                # Gaussian fit overlay
                xi_min, xi_max = np.percentile(xi, [0.5, 99.5])
                xs = np.linspace(xi_min, xi_max, 400)
                if std[i] > 0:
                    pdf = (1.0/(np.sqrt(2*np.pi)*std[i])) * np.exp(-0.5*((xs-mu[i])/std[i])**2)
                    ax.plot(xs, pdf, lw=1.8)
                ax.set_title(f"w[{i}]  μ={mu[i]:.3g}, σ={std[i]:.3g}")
                ax.set_ylabel("Density")
                ax.grid(True, alpha=0.3)
            # Hide empty subplots
            for j in range(n_hist, nrows*ncols):
                r, c = divmod(j, ncols)
                fig1.delaxes(axes[r][c])

            fig1.suptitle("Empirical marginals of disturbance components")
            fig1.tight_layout(rect=[0, 0, 1, 0.97])
            if save_path:
                fig1.savefig(f"{save_path}__marginals.png", dpi=150, bbox_inches="tight")

        # ---------- 2D scatter + covariance ellipses for first two dims ----------
        if plot_pairs and n_w >= 2:
            fig2, ax2 = plt.subplots(1, 1, figsize=(5.5, 5.0))
            ax2.scatter(W[0, :], W[1, :], s=8, alpha=0.35)
            _draw_cov_ellipses(ax2, mu[:2], Sigma[:2, :2], levels=ellipse_levels)
            ax2.set_xlabel("w[0]")
            ax2.set_ylabel("w[1]")
            ax2.set_title("Scatter and covariance ellipses (w[0], w[1])")
            ax2.grid(True, alpha=0.3)
            ax2.set_aspect("equal")
            if save_path:
                fig2.savefig(f"{save_path}__pair01.png", dpi=150, bbox_inches="tight")
            plt.show()

        return {"mu": mu, "Sigma": Sigma, "std": std}

# =====================================================================================

if __name__ == "__main__":
    model = "2W"
    n, var = 2, 1 # None
    ellipse, AfterBefore = True, True

    # simple test
    wass = Disturbances(model=model, gamma=0.5, n=n, var=var, ellipse=ellipse, AfterBefore=AfterBefore)
    print(f"{model} initialized: mode={wass.mode}, gamma={wass.gamma}, n={wass.Nw}")
    print(f" Nominal covariance Sigma_nom:\n{wass.Sigma_nom}")
    print(f" Test covariance Sigma_test:\n{wass.Sigma_test}")


    # sample iid
    t = wass.time
    w = wass.sample()
    #print(f"Is within bounds? {wass.is_member_empirical(w)}")
    Sigma_nom_estm = 1/(2*w.shape[0]) * ((w.T @ w) + (w.T @ w).T)
    print("Sigma_nom_estm:", Sigma_nom_estm)

    if 1:
        if n > 1:
            # distribution plots
            stats = wass.plot_disturbance_distribution(w, bins=40, plot_pairs=True, save_path=None)
            print("mu:", stats["mu"])
            print("Sigma:\n", stats["Sigma"])
            
        if 1:
            plt.figure()
            plt.title(f"{wass.mode} samples")
            #plt.plot(t, wass.gamma * np.ones_like(t), 'r--', label="gamma")
            plt.plot(t, w)
            plt.xlabel("Time")
            plt.grid()
            plt.show()

    # plain plug-in estimate
    Sigma_estm = wass.estm_Sigma_nom(w)
    print("Estimated Sigma from samples:\n", Sigma_estm)
    gamma_estm, *_ = wass.estimate_gamma_with_ci(w)
    print("Estimated gamma from samples:", gamma_estm)