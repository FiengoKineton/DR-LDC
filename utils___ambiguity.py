import numpy as np
from numpy.linalg import cholesky
from scipy.linalg import sqrtm

class WassersteinAmbiguitySet:
    """
    2-Wasserstein ambiguity set around N(0, Sigma_nom) with radius gamma.
    - W_ind: iid disturbances with that marginal
    - W_cor: arbitrary temporal correlation but each marginal in the ball
    """
    def __init__(self, Sigma_nom: np.ndarray, gamma: float, rng: np.random.Generator = None):
        Sigma_nom = 0.5 * (Sigma_nom + Sigma_nom.T)
        self.Sigma_nom = Sigma_nom
        self.gamma = float(gamma)
        self.rng = np.random.default_rng() if rng is None else rng
        self._chol_nom = None

    @staticmethod
    def _sym(A):
        return 0.5 * (A + A.T)

    @staticmethod
    def w2_gaussian(S1: np.ndarray, S2: np.ndarray) -> float:
        """W2(N(0,S1), N(0,S2)) for SPD S1, S2."""
        S1 = 0.5 * (S1 + S1.T)
        S2 = 0.5 * (S2 + S2.T)
        S2h = sqrtm(S2)
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
        return z @ L.T

    def sample_correlated(self, T: int, rho: float = 0.8, Sigma: np.ndarray = None) -> np.ndarray:
        """
        Sample a correlated sequence in W_cor using an AR(1) model:
            w_{t+1} = F w_t + eps_t,   eps_t ~ N(0,Q)
        with stationary marginal covariance = Sigma_proj ∈ ball.
        """
        if Sigma is None:
            Sigma_target = self.Sigma_nom
        else:
            Sigma_target = self.project_cov_to_ball(Sigma)

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
        return w

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
