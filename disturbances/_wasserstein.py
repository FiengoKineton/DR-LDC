import numpy as np
import matplotlib.pyplot as plt

from scipy.linalg import sqrtm
from numpy.linalg import cholesky
from config import get_cfg


# =============================================================================================== #

class WassersteinAmbiguitySet:
    """
    2-Wasserstein ambiguity set around N(0, Sigma_nom) with radius gamma.
    - W_ind: iid disturbances with that marginal
    - W_cor: arbitrary temporal correlation but each marginal in the ball
    """
    def __init__(self, gamma: float = None, ellipse: bool = False, n: int = None, var: float = None, alpha: float = 1.5):
        cfg = get_cfg()
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

# =============================================================================================== #
