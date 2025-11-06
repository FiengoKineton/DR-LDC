import numpy as np
from numpy.linalg import cholesky
from scipy.linalg import sqrtm
import yaml, sys
import matplotlib.pyplot as plt

yaml_path = "problem___parameters.yaml"
if yaml is None:
    raise ImportError("PyYAML not available. Install with `pip install pyyaml`.")
with open(yaml_path, "r", encoding="utf-8") as f:
    cfg = yaml.safe_load(f)

# =====================================================================================

class WassersteinAmbiguitySet:
    """
    2-Wasserstein ambiguity set around N(0, Sigma_nom) with radius gamma.
    - W_ind: iid disturbances with that marginal
    - W_cor: arbitrary temporal correlation but each marginal in the ball
    """
    def __init__(self, gamma: float = None, ellipse: bool = False, n: int = None, var: float = None):

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
            Sigma_use = self.Sigma_nom
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
            Sigma_target = self.Sigma_nom
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
    def __init__(self, gamma: float = None, model: str = None, n: int = None, var: float = None, ellipse: bool = None):
        p   = cfg.get("params", {})
        amb = p.get("ambiguity", {})
        model = amb.get("model", "W2") if model is None else model
        self.impl = self._select(model, gamma, n, var, ellipse)

        # convenience mirrors so your demo prints work
        self.mode = getattr(self.impl, "mode", model)
        self.gamma = getattr(self.impl, "gamma", gamma)
        self.Sigma_nom = getattr(self.impl, "Sigma_nom", None)
        self.time = getattr(self.impl, "time", None)
        self.ts = getattr(self.impl, "ts", None)
        self.T = getattr(self.impl, "T", None)

    def _select(self, model: str, gamma: float, n: int, var: float, ellipse: bool):
        if model == "W2":
            if gamma is not None and gamma < 0:
                raise ValueError("gamma must be nonnegative")
            return WassersteinAmbiguitySet(gamma=gamma, ellipse=ellipse, n=n, var=var)
        if model == "zero":
            return WithoutNoise()
        if model == "Gaussian":
            return GaussianNoise(n=n, var=var)
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
    # simple test
    wass = Disturbances(model="Gaussian", gamma=0.5, n=2, var=1, ellipse=True)
    print("Nominal Sigma:\n", wass.Sigma_nom)
    print("Gamma:", wass.gamma)
    print("Mode:", wass.mode)

    # sample iid
    t = wass.time
    w = wass.sample()
    print(wass.is_member_empirical(w))


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
        plt.legend()
        plt.show()
