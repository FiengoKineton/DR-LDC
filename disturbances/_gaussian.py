import numpy as np

from numpy.linalg import cholesky
from config import cfg


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
