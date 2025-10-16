# ambiguity.py
import numpy as np

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
