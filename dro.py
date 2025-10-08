import numpy as np

def wasserstein_radius_chernoff(dim, N, delta=0.1, scale=1.0):
    """
    Toy picker for W2 radius: gamma ~ scale * sqrt((dim + log(1/delta)) / N).
    Replace with your preferred finite-sample DRO calibration if you care
    about tightness more than your weekend.
    """
    return float(scale * np.sqrt((dim + np.log(1.0 / max(delta, 1e-12))) / max(N, 1)))
