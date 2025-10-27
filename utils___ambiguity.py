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
    def __init__(self, gamma: float = None):

        p = cfg.get("params", {})
        set = p.get("ambiguity", {})
        sim = p.get("simulation", {})
        Sigma_nom = np.array(set["Sigma_nom"], dtype=float) 
        gamma = float(set.get("gamma", 0.0)) if gamma is None else gamma
        Tf = sim.get("TotTime", 100)
        self.ts = sim.get("ts", 0.5)
        
        self.mode = p.get("model", "independent")
        self.T = int(Tf / self.ts)
        self.Sigma_nom = 0.5 * (Sigma_nom + Sigma_nom.T)
        self.gamma = gamma
        self.rng = np.random.default_rng()
        self._chol_nom = None
        self.time = np.arange(self.T)*self.ts


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


    # ---------- sampling utilities ----------

    def sample(self, T:int = None, Sigma: np.ndarray = None) -> np.ndarray:
        """
        Sample a disturbance sequence of length T from the ambiguity set.
        Uses the specified mode ("independent" or "correlated").
        Returns array shape (T, n).
        """
        T = T if T is not None else self.T
        if self.mode == "independent":
            return self.sample_iid(T=T, Sigma=Sigma)
        elif self.mode == "correlated":
            return self.sample_correlated(T=T, Sigma=Sigma)
        else:
            raise ValueError(f"Unknown ambiguity mode: {self.mode}")

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

    def sample_correlated(self, T: int, rho: float = 0.9, Sigma: np.ndarray = None) -> np.ndarray:
        """
        Sample a correlated sequence in W_cor using an AR(1) model:
            w_{t+1} = F w_t + eps_t,   eps_t ~ N(0,Q)
        with stationary marginal covariance = Sigma_proj ∈ ball.
        """
        if Sigma is None:
            Sigma_target = self.Sigma_nom
        else:
            self.Sigma_nom = np.eye(Sigma[0].size)
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

# =====================================================================================

class WithoutNoise():
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
    def __init__(self):
        p   = cfg.get("params", {})
        amb = p.get("ambiguity", {})
        sim = p.get("simulation", {})
        self.Sigma_nom = np.array(amb["Sigma_nom"], dtype=float)
        self.gamma = None
        Tf      = float(sim.get("TotTime", 100.0))
        self.ts = float(sim.get("ts", 0.5))
        self.T  = int(round(Tf / self.ts))
        self.time = np.arange(self.T) * self.ts
        self.rng = np.random.default_rng()

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
    def __init__(self, gamma: float = None, model: str = None):
        p   = cfg.get("params", {})
        amb = p.get("ambiguity", {})
        model = amb.get("model", "W2") if model is None else model
        self.impl = self._select(model, gamma)

        # convenience mirrors so your demo prints work
        self.mode = getattr(self.impl, "mode", model)
        self.gamma = getattr(self.impl, "gamma", gamma)
        self.Sigma_nom = getattr(self.impl, "Sigma_nom", None)
        self.time = getattr(self.impl, "time", None)
        self.ts = getattr(self.impl, "ts", None)
        self.T = getattr(self.impl, "T", None)

    def _select(self, model: str, gamma: float):
        if model == "W2":
            if gamma is not None and gamma < 0:
                raise ValueError("gamma must be nonnegative")
            return WassersteinAmbiguitySet(gamma=gamma)
        if model == "zero":
            return WithoutNoise()
        if model == "Gaussian":
            return GaussianNoise()
        raise ValueError(f"Unknown ambiguity model: {model}")

    def __getattr__(self, name: str):
        # delegate all other attributes/methods to the concrete implementation
        return getattr(self.impl, name)

    def __repr__(self):
        return f"Disturbances(mode={self.mode!r}, gamma={self.gamma!r})"

# =====================================================================================

if __name__ == "__main__":
    # simple test
    wass = Disturbances(model="Gaussian")
    print("Nominal Sigma:\n", wass.Sigma_nom)
    print("Gamma:", wass.gamma)
    print("Mode:", wass.mode)

    # sample iid
    t = wass.time
    w = wass.sample()
    print(wass.is_member_empirical(w))

    plt.figure()
    plt.title(f"{wass.mode} samples")
    #plt.plot(t, wass.gamma * np.ones_like(t), 'r--', label="gamma")
    plt.plot(t, w)
    plt.xlabel("Time")
    plt.grid()
    plt.legend()
    plt.show()
