# simulate_closed_loop.py
import numpy as np
import matplotlib.pyplot as plt

from matrices import get_system
from systems import Plant, Controller

def simulate_closed_loop(plant: Plant,
                         ctrl: Controller,
                         Sigma_w: np.ndarray,
                         T: int = 500,
                         seed: int = 0,
                         x0: np.ndarray | None = None,
                         xc0: np.ndarray | None = None):
    """
    Simulate the interconnection (no composite shortcut) so we can view y, u explicitly.

      y_t = C_y x_t + D_yw w_t
      u_t = C_c x_c_t + D_c y_t
      x_{t+1} = A x_t + B_u u_t + B_w w_t
      x_{c,t+1} = A_c x_{c,t} + B_c y_t
      z_t = C_z x_t + D_zu u_t + D_zw w_t
    """
    A, Bw, Bu, Cz, Dzw, Dzu, Cy, Dyw = \
        plant.A, plant.Bw, plant.Bu, plant.Cz, plant.Dzw, plant.Dzu, plant.Cy, plant.Dyw
    Ac, Bc, Cc, Dc = ctrl.Ac, ctrl.Bc, ctrl.Cc, ctrl.Dc

    nx = A.shape[0]
    nw = Bw.shape[1]
    nu = Bu.shape[1]
    nz = Cz.shape[0]
    ny = Cy.shape[0]
    nxc = Ac.shape[0]

    # Initial conditions
    x = np.zeros((nx, 1)) if x0 is None else np.asarray(x0, float).reshape(nx, 1)
    xc = np.zeros((nxc, 1)) if xc0 is None else np.asarray(xc0, float).reshape(nxc, 1)

    # Precompute a sampling factor for w
    rng = np.random.default_rng(seed)
    try:
        L = np.linalg.cholesky(Sigma_w)
    except np.linalg.LinAlgError:
        L = np.linalg.cholesky(Sigma_w + 1e-12 * np.eye(nw))

    # Storage
    X  = np.zeros((T, nx))
    Xc = np.zeros((T, nxc))
    Y  = np.zeros((T, ny))
    U  = np.zeros((T, nu))
    Z  = np.zeros((T, nz))
    W  = np.zeros((T, nw))

    for t in range(T):
        w = (L @ rng.standard_normal((nw, 1))).astype(float)
        y = Cy @ x + Dyw @ w
        u = Cc @ xc + Dc @ y
        z = Cz @ x + Dzu @ u + Dzw @ w

        # Log
        X[t, :]  = x.ravel()
        Xc[t, :] = xc.ravel()
        Y[t, :]  = y.ravel()
        U[t, :]  = u.ravel()
        Z[t, :]  = z.ravel()
        W[t, :]  = w.ravel()

        # Update
        x_next  = A @ x + Bu @ u + Bw @ w
        xc_next = Ac @ xc + Bc @ y
        x, xc = x_next, xc_next

    return {
        "X": X, "Xc": Xc, "Y": Y, "U": U, "Z": Z, "W": W,
        "T": T, "nx": nx, "nxc": nxc, "ny": ny, "nu": nu, "nz": nz, "nw": nw
    }

def plot_timeseries(sim):
    T = sim["T"]
    t = np.arange(T)

    # States x
    plt.figure(figsize=(10, 6))
    for i in range(sim["nx"]):
        plt.plot(t, sim["X"][:, i], label=f"x[{i}]")
    plt.title("Plant states x")
    plt.xlabel("t")
    plt.legend()
    plt.grid(True, alpha=0.3)

    # Controller states x_c
    plt.figure(figsize=(10, 6))
    for i in range(sim["nxc"]):
        plt.plot(t, sim["Xc"][:, i], label=f"x_c[{i}]")
    plt.title("Controller states x_c")
    plt.xlabel("t")
    plt.legend()
    plt.grid(True, alpha=0.3)

    # Measured output y
    plt.figure(figsize=(10, 6))
    for i in range(sim["ny"]):
        plt.plot(t, sim["Y"][:, i], label=f"y[{i}]")
    plt.title("Measured output y")
    plt.xlabel("t")
    plt.legend()
    plt.grid(True, alpha=0.3)

    # Control input u
    plt.figure(figsize=(10, 5))
    for i in range(sim["nu"]):
        plt.plot(t, sim["U"][:, i], label=f"u[{i}]")
    plt.title("Control input u")
    plt.xlabel("t")
    plt.legend()
    plt.grid(True, alpha=0.3)

    # Performance output z
    plt.figure(figsize=(10, 6))
    for i in range(sim["nz"]):
        plt.plot(t, sim["Z"][:, i], label=f"z[{i}]")
    plt.title("Performance output z")
    plt.xlabel("t")
    plt.legend()
    plt.grid(True, alpha=0.3)

    plt.show()

def save_npz(sim, fname="cl_timeseries.npz"):
    np.savez_compressed(fname, **sim)
    return fname

if __name__ == "__main__":
    # Use the same plant as the optimization example (seed=7)
    plant, _ = get_system(seed=7, FROM_DATA=False)

    Ac = np.array([
        [ 0.3449, -0.4085,  0.    ,  0.    ],
        [-0.4279,  0.4803,  0.    ,  0.    ],
        [ 0.    ,  0.    ,  0.    ,  0.    ],
        [ 0.    ,  0.    ,  0.    ,  0.    ],
    ], dtype=float)

    Bc = np.array([
        [ 0.2538, -0.2417],
        [-0.2802,  0.3522],
        [ 0.    ,  0.    ],
        [ 0.    ,  0.    ],
    ], dtype=float)

    Cc = np.array([
        [ 0.1093, -0.0614,  0.    ,  0.    ],
        [ 0.1129, -0.0553,  0.    ,  0.    ],
    ], dtype=float)

    Dc = np.array([
        [-1.0166,  0.1460],
        [-1.0123,  0.1628],
    ], dtype=float)

    ctrl = Controller(Ac=Ac, Bc=Bc, Cc=Cc, Dc=Dc)
    Sigma_w = 0.7 * np.eye(2)

    sim = simulate_closed_loop(plant, ctrl, Sigma_w, T=800, seed=11)
    print("Simulated shapes:",
          {k: v.shape for k, v in sim.items() if isinstance(v, np.ndarray)})

    out = save_npz(sim, "closed_loop_run_seed11_T800.npz")
    print(f"Saved time series to {out}")

    plot_timeseries(sim)
