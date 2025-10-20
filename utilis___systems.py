import numpy as np
import yaml
from dataclasses import dataclass

@dataclass
class Plant:
    A: np.ndarray
    Bw: np.ndarray
    Bu: np.ndarray
    Cz: np.ndarray
    Dzw: np.ndarray
    Dzu: np.ndarray
    Cy: np.ndarray
    Dyw: np.ndarray

    def dims(self):
        nx = self.A.shape[0]
        nw = self.Bw.shape[1]
        nu = self.Bu.shape[1]
        nz = self.Cz.shape[0]
        ny = self.Cy.shape[0]
        return nx, nw, nu, nz, ny

@dataclass
class Controller:
    Ac: np.ndarray
    Bc: np.ndarray
    Cc: np.ndarray
    Dc: np.ndarray

    def dims(self):
        nxc = self.Ac.shape[0]
        return nxc



class Ambiguity:
    """
    Wasserstein-2 ball around a zero-mean Gaussian nominal with covariance Σ_nom.
    We use a defensible covariance-inflation surrogate:
        Σ_eff = Σ_nom + α(γ) I
    for 'independent' and 'correlated' models.
    This upper-bounds the worst-case second moment within the W2 ball.
    Data is limited; exact tight factors depend on cost structure. Use α as a knob.
    """

    def __init__(self, Sigma_nom: np.ndarray):
        """
        model ∈ {"correlated","independent"} just tags your modeling assumption.
        alpha: if None, we pick alpha = gamma (units: same as Σ entries).
        """
        if yaml is None:
            raise ImportError("PyYAML not available. Install with `pip install pyyaml`.")
        with open("problem___parameters.yaml", "r", encoding="utf-8") as f:
            cfg = yaml.safe_load(f)

        p = cfg.get("params", {})
        amb = p.get("ambiguity", {})

        self.model = p.get("model", "correlated")
        self.gamma = float(amb.get("gamma", 0.0))
        self.alpha = float(amb.get("alpha", self.gamma))
        self.Sigma_nom = Sigma_nom


    def sigma_effective(self):
        n = self.Sigma_nom.shape[0]
        return self.Sigma_nom + self.alpha * np.eye(n)
