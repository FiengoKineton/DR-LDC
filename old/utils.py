import numpy as np
from numpy.linalg import eigvals
from scipy.linalg import solve_discrete_are

def make_stable_A(n, spectral_radius=0.9, rng=None):
    rng = np.random.default_rng(rng)
    Q, _ = np.linalg.qr(rng.standard_normal((n, n)))
    vals = spectral_radius * rng.uniform(0.2, 1.0, size=n)
    A = Q @ np.diag(vals) @ np.linalg.inv(Q)
    return A

def lqr(A, B, Q, R):
    """Discrete-time LQR via Riccati equation."""
    P = solve_discrete_are(A, B, Q, R)
    K = np.linalg.inv(B.T @ P @ B + R) @ (B.T @ P @ A)
    return K

def is_stable(Acl, tol=1e-7):
    return np.max(np.abs(eigvals(Acl))) < 1 - tol
