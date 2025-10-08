import numpy as np
from dataclasses import dataclass
from utils import make_stable_A

@dataclass
class Plant:
    A: np.ndarray
    B: np.ndarray
    Cz: np.ndarray
    Dzu: np.ndarray

def make_demo_plant(n=4, m=2, z_dim=4, rng=0):
    rng = np.random.default_rng(rng)
    A = make_stable_A(n, spectral_radius=0.9, rng=rng)
    B = rng.standard_normal((n, m))
    Cz = np.eye(n)[:z_dim, :]
    Dzu = 0.1 * rng.standard_normal((z_dim, m))
    return Plant(A=A, B=B, Cz=Cz, Dzu=Dzu)
