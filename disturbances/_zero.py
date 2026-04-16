import numpy as np
from config import get_cfg

# =====================================================================================

class WithoutNoise:
    def __init__(self):
        cfg = get_cfg()
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
