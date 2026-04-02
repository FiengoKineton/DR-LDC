import numpy as np, matplotlib.pyplot as plt

from config import cfg
from ._wasserstein import WassersteinAmbiguitySet
from ._metric_2w import Metric2Wasserstein
from ._gaussian import GaussianNoise
from ._zero import WithoutNoise




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
            return Metric2Wasserstein(gamma=gamma, n=n, var=var, ellipse=ellipse, AfterBefore=AfterBefore)
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
