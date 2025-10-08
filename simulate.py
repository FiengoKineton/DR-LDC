import numpy as np

def simulate_closed_loop(A, B, Cz, Dzu, K, T=2000, x0=None, Sigma_w=None, rng=0):
    rng = np.random.default_rng(rng)
    n = A.shape[0]
    if x0 is None:
        x = np.zeros(n)
    else:
        x = x0.copy()

    zsum = 0.0
    for _ in range(T):
        w = rng.multivariate_normal(mean=np.zeros(n), cov=Sigma_w)
        z = (Cz + Dzu @ K) @ x
        zsum += float(z @ z)
        x = (A + B @ K) @ x + w
    return zsum / T
