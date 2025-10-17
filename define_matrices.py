# define_matrices.py
import re
import numpy as np
from systems import Plant, Controller
from typing import Dict, Tuple, Optional, List

# ------------------------- CSV LOADING HELPERS -------------------------

def _read_csv_with_headers(path: str, delimiter: str = ",") -> Tuple[List[str], np.ndarray]:
    """
    Minimal CSV reader that grabs the first row as headers, remaining as float data.
    No pandas dependency, no excuses.
    """
    with open(path, "r", encoding="utf-8") as f:
        header = f.readline().strip()
    headers = [h.strip() for h in header.split(delimiter)]
    data = np.loadtxt(path, delimiter=delimiter, skiprows=1)
    if data.ndim == 1:
        data = data[None, :]
    if data.shape[1] != len(headers):
        raise ValueError(f"Column count mismatch: headers={len(headers)} vs data={data.shape[1]}")
    return headers, data

def _pick_columns(headers: List[str], data: np.ndarray, prefix: str) -> np.ndarray:
    """
    Pick columns whose names match prefix + integer, e.g., x1,x2,... or y3,...
    Returns matrix with shape (count, T) in ROW-MAJOR (variables x time).
    """
    pat = re.compile(rf"^{re.escape(prefix)}(\d+)$", re.IGNORECASE)
    indices = [(i, int(m.group(1))) for i, h in enumerate(headers) if (m := pat.match(h)) is not None]
    if not indices:
        return np.empty((0, data.shape[0]))  # no such block
    # sort by numeric suffix to keep x1,x2,... order
    indices.sort(key=lambda t: t[1])
    cols = [i for i, _ in indices]
    block = data[:, cols].T  # (count x T)
    return block

def _build_blocks_from_csv(path: str, delimiter: str = ","):
    headers, data = _read_csv_with_headers(path, delimiter=delimiter)

    X = _pick_columns(headers, data, "x")
    U = _pick_columns(headers, data, "u")
    Y = _pick_columns(headers, data, "y")
    Z = _pick_columns(headers, data, "z")

    if X.size == 0 or U.size == 0:
        raise ValueError("CSV must contain at least x* and u* columns.")

    # Raw lengths
    Tx = X.shape[1]
    Tu = U.shape[1]
    Ty = Y.shape[1] if Y.size else np.inf
    Tz = Z.shape[1] if Z.size else np.inf

    # We need pairs (x_t, u_t) and their successor x_{t+1}.
    # So the maximum valid regressor length is (min(Tx, Tu, Ty, Tz) - 1).
    Tpair = int(min(Tx, Tu, Ty, Tz)) - 1
    if Tpair < 1:
        raise ValueError(f"Not enough samples to form (X, X_next): got Tx={Tx}, Tu={Tu}, Ty={Ty}, Tz={Tz}")

    # Build aligned blocks with identical column counts
    X_reg  = X[:, :Tpair]          # x_0..x_{Tpair-1}
    U_reg  = U[:, :Tpair]          # u_0..u_{Tpair-1}
    X_next = X[:, 1:Tpair+1]       # x_1..x_{Tpair}
    Y_reg  = Y[:, :Tpair] if Y.size else None
    Z_reg  = Z[:, :Tpair] if Z.size else None

    return dict(X=X_reg, U=U_reg, X_next=X_next, Y=Y_reg, Z=Z_reg)

# ------------------------- DDD ESTIMATION HELPERS -------------------------

def _lsq_right_inverse(D: np.ndarray, ridge: float) -> np.ndarray:
    # D in R^{(nx+nu) x T}; return V = D^T (D D^T + ridge I)^{-1}
    r, _ = D.shape
    G = D @ D.T + ridge * np.eye(r)
    return D.T @ np.linalg.inv(G)

def _bw_from_residual(R: np.ndarray, nw: Optional[int], eps: float = 1e-9) -> Tuple[np.ndarray, int, np.ndarray]:
    """
    Build Bw from residual covariance: R in R^{nx x T}.
    Bw uses top-nw eigen-directions of cov(R). Returns (Bw, nw_eff, Sigma_res).
    """
    nx, T = R.shape
    S = (R @ R.T) / max(T, 1)
    S = 0.5 * (S + S.T) + eps * np.eye(nx)
    vals, vecs = np.linalg.eigh(S)
    idx = np.argsort(vals)[::-1]
    vals = vals[idx]
    vecs = vecs[:, idx]
    if nw is None:
        # pick as many modes as needed to cover 95% energy, at least 1
        cum = np.cumsum(vals) / max(np.sum(vals), eps)
        nw = int(np.clip(np.searchsorted(cum, 0.95) + 1, 1, nx))
    nw = int(min(max(1, nw), nx))
    Bw = vecs[:, :nw] @ np.diag(np.sqrt(np.maximum(vals[:nw], eps)))
    return Bw, nw, S

# ------------------------- PUBLIC API -------------------------

def get_system(seed=0, FROM_DATA=False, **kwargs):
    """
    If FROM_DATA=True, pass data_csv="path/to/file.csv" (and optional settings).
    Example:
      get_system(FROM_DATA=True,
                 data_csv="data/run01.csv",
                 delimiter=",",
                 nw=None, ny=None, nz=None,
                 ridge=1e-6)
    """
    if FROM_DATA:
        return make_matrices_from_data(seed=seed, **kwargs)
    else:
        return make_example_system(seed=seed)

def make_matrices_from_data(
    seed=0,
    data_csv: str = "",
    delimiter: str = ",",
    nw: Optional[int] = None,
    ny: Optional[int] = None,
    nz: Optional[int] = None,
    ridge: float = 1e-6,
):
    """
    Direct data-driven construction of (A, Bu, Bw, Cz, Dzw, Dzu, Cy, Dyw)
    from a single CSV file. The CSV must have headers x1..x{nx}, u1..u{nu}
    and may optionally include y1..y{ny}, z1..z{nz}.

    The function automatically aligns sequences to form (X, U, X_next) over a
    common T-1 horizon, then applies projection-based estimates:
      [A  Bu] = X_next @ V^T,   V = D^T (D D^T + ridge I)^{-1},  D=[X;U]
      Bw     from principal directions of residual covariance.
    If Y/Z supplied, it regresses:
      Y ≈ Cy X + Dyw R,   Z ≈ Cz X + Dzu U + Dzw R,
    with R = X_next - (A X + Bu U).
    """

    if not data_csv:
        #raise ValueError("Provide data_csv='path/to/file.csv' to read data.")
        return make_example_system(seed=seed)

    blocks = _build_blocks_from_csv(data_csv, delimiter=delimiter)
    X = blocks["X"]          # (nx x T)
    U = blocks["U"]          # (nu x T)
    X_next = blocks["X_next"]
    Y = blocks["Y"]          # may be None
    Z = blocks["Z"]          # may be None

    nx, T = X.shape
    nu = U.shape[0]

    # Data-driven A, Bu by right-inverse projection
    D = np.vstack([X, U])       # (nx+nu) x T
    V = _lsq_right_inverse(D, ridge=ridge)  # (T x nx+nu)
    Vx = V[:, :nx]  # (T x nx)
    Vu = V[:, nx:]  # (T x nu)

    A = X_next @ Vx
    Bu = X_next @ Vu

    # Residual and Bw
    R = X_next - (A @ X + Bu @ U)  # (nx x T)
    Bw, nw_eff, Sigma_res = _bw_from_residual(R, nw=nw)

    # Outputs: measured Y
    if Y is None or Y.size == 0:
        # Educated defaults: measure first ny states; no direct disturbance to sensors
        if ny is None:
            ny = min(2, nx)
        Cy = np.zeros((ny, nx))
        Cy[np.arange(ny), np.arange(ny)] = 1.0
        Dyw = np.zeros((ny, Bw.shape[1]))
    else:
        ny = Y.shape[0] if ny is None else ny
        # Regress Y on [X; R] using ridge
        ThetaY = np.vstack([X, R])        # (nx+nx) x T
        GY = ThetaY @ ThetaY.T + ridge * np.eye(2 * nx)
        WY = Y @ ThetaY.T @ np.linalg.inv(GY)   # (ny x 2nx)
        Cy = WY[:, :nx]
        Dyw = WY[:, nx:]
        # If desired ny < observed, truncate; if ny > observed, pad zeros
        if Cy.shape[0] != ny:
            if Cy.shape[0] > ny:
                Cy = Cy[:ny, :]
                Dyw = Dyw[:ny, :]
            else:
                pad = ny - Cy.shape[0]
                Cy = np.vstack([Cy, np.zeros((pad, nx))])
                Dyw = np.vstack([Dyw, np.zeros((pad, Dyw.shape[1]))])

    # Outputs: performance Z
    if Z is None or Z.size == 0:
        if nz is None:
            nz = min(3, nx + nu)
        # Cz: identity rows if nz <= nx, else principal directions of state cov
        if nz <= nx:
            Cz = np.zeros((nz, nx))
            Cz[np.arange(nz), np.arange(nz)] = 1.0
        else:
            Sx = X @ X.T / max(T, 1)
            Sx = 0.5 * (Sx + Sx.T) + 1e-9 * np.eye(nx)
            vals, vecs = np.linalg.eigh(Sx)
            idx = np.argsort(vals)[::-1]
            Cz = vecs[:, idx[:nz]].T  # (nz x nx)
        Dzu = 0.05 * np.eye(nz, nu)                 # mild control penalty
        Dzw = np.zeros((nz, Bw.shape[1]))           # no direct w to z by default
    else:
        nz = Z.shape[0] if nz is None else nz
        ThetaZ = np.vstack([X, U, R])               # (nx+nu+nx) x T
        GZ = ThetaZ @ ThetaZ.T + ridge * np.eye(2 * nx + nu)
        WZ = Z @ ThetaZ.T @ np.linalg.inv(GZ)       # (nz x (nx+nu+nx))
        Cz  = WZ[:, :nx]
        Dzu = WZ[:, nx:nx+nu]
        Dzw = WZ[:, nx+nu:]
        if Cz.shape[0] != nz:
            if Cz.shape[0] > nz:
                Cz  = Cz[:nz, :]
                Dzu = Dzu[:nz, :]
                Dzw = Dzw[:nz, :]
            else:
                pad = nz - Cz.shape[0]
                Cz  = np.vstack([Cz,  np.zeros((pad, nx))])
                Dzu = np.vstack([Dzu, np.zeros((pad, nu))])
                Dzw = np.vstack([Dzw, np.zeros((pad, Dzw.shape[1]))])

    # Build plant
    plant = Plant(A=A, Bw=Bw, Bu=Bu, Cz=Cz, Dzw=Dzw, Dzu=Dzu, Cy=Cy, Dyw=Dyw)

    # Neutral controller seed (full-order, tiny static gains)
    nxc = nx
    Ac0 = np.zeros((nxc, nxc))
    Bc0 = 0.05 * np.eye(nxc, Cy.shape[0])
    Cc0 = 0.05 * np.eye(Bu.shape[1], nxc)
    Dc0 = np.zeros((Bu.shape[1], Cy.shape[0]))
    ctrl0 = Controller(Ac=Ac0, Bc=Bc0, Cc=Cc0, Dc=Dc0)

    return plant, ctrl0

# ------------------------- LEGACY EXAMPLE (unchanged) -------------------------

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


"""
USAGE

plant, ctrl0 = get_system(FROM_DATA=True,
                          data_csv="data/session01.csv",
                          delimiter=",",
                          nw=None, ny=None, nz=None,
                          ridge=1e-6)

"""