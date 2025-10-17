# define_matrices.py
import numpy as np
from systems import Plant, Controller

def get_system(seed=0, FROM_DATA=False):
    if FROM_DATA:
        return make_data_matrices(seed=seed)
    else:
        return make_example_system(seed=seed)

def make_data_matrices(seed=0):
    """
    Replace this with your real matrices.
    Discrete-time example with modest dimensions.
    """
    rng = np.random.default_rng(seed)

    nx, nw, nu, nz, ny = 4, 2, 2, 3, 2

    A  = np.array([[0.95, 0.1, 0.0, 0.0],
                   [0.0,  0.92, 0.1, 0.0],
                   [0.0,  0.0,  0.90,0.1],
                   [0.0,  0.0,  0.0, 0.88]])
    Bu = rng.normal(0, 0.3, (nx, nu))
    Bw = rng.normal(0, 0.3, (nx, nw))

    Cz = rng.normal(0, 0.5, (nz, nx))
    Dzu = rng.normal(0, 0.2, (nz, nu))
    Dzw = rng.normal(0, 0.2, (nz, nw))

    Cy = rng.normal(0, 0.5, (ny, nx))
    Dyw = rng.normal(0, 0.2, (ny, nw))

    plant = Plant(A=A, Bw=Bw, Bu=Bu, Cz=Cz, Dzw=Dzw, Dzu=Dzu, Cy=Cy, Dyw=Dyw)

    # Full-order controller as a starting point; zero dynamics + small static gain
    nxc = nx
    Ac0 = 0.0 * np.eye(nxc)
    Bc0 = 0.1 * np.eye(nxc, ny)
    Cc0 = 0.1 * np.eye(nu, nxc)
    Dc0 = 0.0 * np.eye(nu, ny)

    ctrl0 = Controller(Ac=Ac0, Bc=Bc0, Cc=Cc0, Dc=Dc0)
    return plant, ctrl0

def make_example_system(seed=0):
    """
    Replace this with your real matrices.
    Discrete-time example with modest dimensions.
    """
    rng = np.random.default_rng(seed)

    nx, nw, nu, nz, ny = 4, 2, 2, 3, 2
    A  = np.array([[0.95, 0.1, 0.0, 0.0],
                   [0.0,  0.92, 0.1, 0.0],
                   [0.0,  0.0,  0.90,0.1],
                   [0.0,  0.0,  0.0, 0.88]])
    Bu = rng.normal(0, 0.3, (nx, nu))
    Bw = rng.normal(0, 0.3, (nx, nw))

    Cz = rng.normal(0, 0.5, (nz, nx))
    Dzu = rng.normal(0, 0.2, (nz, nu))
    Dzw = rng.normal(0, 0.2, (nz, nw))

    Cy = rng.normal(0, 0.5, (ny, nx))
    Dyw = rng.normal(0, 0.2, (ny, nw))

    plant = Plant(A=A, Bw=Bw, Bu=Bu, Cz=Cz, Dzw=Dzw, Dzu=Dzu, Cy=Cy, Dyw=Dyw)

    # Full-order controller as a starting point; zero dynamics + small static gain
    nxc = nx
    Ac0 = 0.0 * np.eye(nxc)
    Bc0 = 0.1 * np.eye(nxc, ny)
    Cc0 = 0.1 * np.eye(nu, nxc)
    Dc0 = 0.0 * np.eye(nu, ny)

    ctrl0 = Controller(Ac=Ac0, Bc=Bc0, Cc=Cc0, Dc=Dc0)
    return plant, ctrl0

def make_nominal_covariances(nw):
    # nominal zero-mean Gaussian covariance for w
    Sigma_nom = 0.5 * np.eye(nw)
    return Sigma_nom
