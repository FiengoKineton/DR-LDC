# systems.py
import numpy as np
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

    def __init__(self, Sigma_nom: np.ndarray, gamma: float, model: str = "correlated", alpha: float = None):
        """
        model ∈ {"correlated","independent"} just tags your modeling assumption.
        alpha: if None, we pick alpha = gamma (units: same as Σ entries).
        """
        self.Sigma_nom = Sigma_nom
        self.gamma = float(gamma)
        self.model = model
        self.alpha = self.gamma if alpha is None else float(alpha)

    def sigma_effective(self):
        n = self.Sigma_nom.shape[0]
        return self.Sigma_nom + self.alpha * np.eye(n)
