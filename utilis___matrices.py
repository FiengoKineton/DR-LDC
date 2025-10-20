import re, json, yaml
import numpy as np
from utilis___systems import Plant, Controller
from typing import Tuple, Optional, List
from numpy.linalg import eigvals, norm


# ------------------------- COMPOSE MATRICES FROM LMI --------------------------

def compose_closed_loop(plant: Plant, ctrl: Controller):
    """
    Build the composite matrices (𝒜|𝓑; 𝒞|𝒟) for
      [ X_{t+1} ]   [ 𝒜  𝓑 ] [ X_t ]
      [   z_t   ] = [ 𝒞  𝒟 ] [ w_t ]
    with X = [x; x_c].
    Formula matches the screenshot: blue terms are controller blocks.
    """
    A, Bw, Bu, Cz, Dzw, Dzu, Cy, Dyw = \
        plant.A, plant.Bw, plant.Bu, plant.Cz, plant.Dzw, plant.Dzu, plant.Cy, plant.Dyw
    Ac, Bc, Cc, Dc = ctrl.Ac, ctrl.Bc, ctrl.Cc, ctrl.Dc

    # Top-left block 𝒜:
    A11 = A + Bu @ Dc @ Cy
    A12 = Bu @ Cc
    A21 = Bc @ Cy
    A22 = Ac
    A_cl = np.block([[A11, A12],
                     [A21, A22]])

    # Top-right block 𝓑:
    B1 = Bw + Bu @ Dc @ Dyw
    B2 = Bc @ Dyw
    B_cl = np.vstack([B1, B2])

    # Bottom-left block 𝒞:
    C1 = Cz + Dzu @ Dc @ Cy
    C2 = Dzu @ Cc
    C_cl = np.hstack([C1, C2])

    # Bottom-right block 𝒟:
    D_cl = Dzw + Dzu @ Dc @ Dyw

    return A_cl, B_cl, C_cl, D_cl


# ------------------------- RECOVER MATRICES FROM CLOSED-LOOP ------------------

class Recover():
    def __init__(self):
        pass

    def load_dro_json(self, path: str):
        with open(path, "r", encoding="utf-8") as f:
            d = json.load(f)
        # Required pieces
        Pbar = np.array(d["Pbar"], dtype=float)
        Abar = np.array(d["Abar"], dtype=float)
        Bbar = np.array(d["Bbar"], dtype=float)
        Cbar = np.array(d["Cbar"], dtype=float)
        Dbar = np.array(d["Dbar"], dtype=float)
        return Pbar, Abar, Bbar, Cbar, Dbar, d

    def closed_loop_from_bar(self, Pbar, Abar, Bbar, Cbar, Dbar, jitter=1e-9):
        """
        Direct gauge: set T = I, P = Pbar. Then:
        A_cl = Pbar^{-1} Abar
        B_cl = Pbar^{-1} Bbar
        C_cl = Cbar
        D_cl = Dbar
        This damps explosive scalings when Pbar has huge entries.
        """
        Pbar = np.array(Pbar, dtype=float)
        Abar = np.array(Abar, dtype=float)
        Bbar = np.array(Bbar, dtype=float)
        Cbar = np.array(Cbar, dtype=float)
        Dbar = np.array(Dbar, dtype=float)

        # Symmetrize and ensure invertibility (nearest PD if needed)
        Pbar = 0.5 * (Pbar + Pbar.T)
        try:
            Pinv = np.linalg.inv(Pbar)
        except np.linalg.LinAlgError:
            # project to nearest PD then invert
            w, V = np.linalg.eigh(Pbar)
            w_clip = np.maximum(w, jitter)
            Pbar_pd = (V * w_clip) @ V.T
            Pinv = np.linalg.inv(Pbar_pd)

        A_cl = Pinv @ Abar
        B_cl = Pinv @ Bbar
        C_cl = Cbar
        D_cl = Dbar
        return A_cl, B_cl, C_cl, D_cl

    def recover_controller_from_closed_loop(self, plant: Plant, A_cl, B_cl, C_cl, D_cl, rcond=1e-9):
        """
        Solve for Dc, Cc, Bc, Ac using least-squares when needed.
        Returns Controller and a residual report.
        """
        A, Bw, Bu, Cz, Dzw, Dzu, Cy, Dyw = plant.A, plant.Bw, plant.Bu, plant.Cz, plant.Dzw, plant.Dzu, plant.Cy, plant.Dyw
        nx = A.shape[0]
        nxc = A_cl.shape[0] - nx
        if nxc <= 0:
            raise ValueError("Composite A_cl has invalid size relative to plant nx.")

        # Partition composite matrices
        A11 = A_cl[:nx, :nx]
        A12 = A_cl[:nx, nx:]
        A21 = A_cl[nx:, :nx]
        A22 = A_cl[nx:, nx:]

        B1 = B_cl[:nx, :]
        B2 = B_cl[nx:, :]

        C1 = C_cl[:, :nx]
        C2 = C_cl[:, nx:]

        # 1) Recover Dc from D_cl = Dzw + Dzu Dc Dyw  ->  Dzu Dc Dyw = D_cl - Dzw
        RHS = D_cl - Dzw
        def _tikhonov_left(A, B, alpha=1e-8):
            # Solve A X ≈ B: (A^T A + alpha I) X = A^T B
            _, n = A.shape
            return np.linalg.solve(A.T @ A + alpha*np.eye(n), A.T @ B)

        # Dc from Dzu Dc Dyw = RHS using two-sided Tikhonov
        Dc_mid = _tikhonov_left(Dyw.T, RHS.T, alpha=1e-8).T
        Dc = _tikhonov_left(Dzu, Dc_mid, alpha=1e-8)

        # Cc = Dzu^\dagger C2   and  Bc = B2 Dyw^\dagger  with Tikhonov
        Cc = _tikhonov_left(Dzu, C2, alpha=1e-8)
        Bc = (_tikhonov_left(Dyw.T, B2.T, alpha=1e-8)).T

        # 4) Recover Ac directly
        Ac = A22

        # Residual checks (sanity)
        res = {}
        res["A12"] = norm(A12 - Bu @ Cc) / (1 + norm(A12))
        res["A21"] = norm(A21 - Bc @ Cy) / (1 + norm(A21))
        res["A11"] = norm(A11 - (A + Bu @ Dc @ Cy)) / (1 + norm(A11))
        res["B1"]  = norm(B1  - (Bw + Bu @ Dc @ Dyw)) / (1 + norm(B1))
        res["C1"]  = norm(C1  - (Cz + Dzu @ Dc @ Cy)) / (1 + norm(C1))
        res["D"]   = norm(D_cl - (Dzw + Dzu @ Dc @ Dyw)) / (1 + norm(D_cl))

        return Controller(Ac=Ac, Bc=Bc, Cc=Cc, Dc=Dc), res

    def recover_controller_from_dro_json(self, json_path: str, plant: Plant):
        Pbar, Abar, Bbar, Cbar, Dbar, meta = self.load_dro_json(json_path)
        A_cl, B_cl, C_cl, D_cl = self.closed_loop_from_bar(Pbar, Abar, Bbar, Cbar, Dbar)
        ctrl, residuals = self.recover_controller_from_closed_loop(plant, A_cl, B_cl, C_cl, D_cl)
        # Quick stability peek on composite A
        rho = max(abs(eigvals(A_cl)))
        return ctrl, residuals, float(rho)


# ------------------------- PUBLIC API -----------------------------------------

class MatricesAPI():
    def __init__(self):
        pass

    def make_nominal_covariances(self, nw):
        # nominal zero-mean Gaussian covariance for w
        Sigma_nom = 0.5 * np.eye(nw)
        return Sigma_nom


    def get_system(self, seed=0, FROM_DATA=False, **kwargs):
        """
        If FROM_DATA=True, pass data_csv="path/to/file.csv" (and optional settings).
        Example:
        get_system(FROM_DATA=True,
                    data_csv="out/data/run01.csv",
                    delimiter=",",
                    nw=None, ny=None, nz=None,
                    ridge=1e-6)
        """
        if FROM_DATA:
            return self.make_matrices_from_data(**kwargs)
        else:
            return self.make_example_system()


    # ------------------------- EXAMPLE SYSTEM CONSTRUCTION -------------------------

    def build_out_matrices(self, yaml_path: str) -> tuple[np.ndarray, np.ndarray, np.ndarray, np.ndarray, np.ndarray]:

        # ======================================================================
        # Output-construction helpers (drop-in, no mystery defaults elsewhere)
        # ======================================================================

        def make_Cy_Dyw(nx: int, ny: int, nw: int, select: str = "first", dyw_zero: bool = True) -> tuple[np.ndarray, np.ndarray]:
            """
            Measured output model y = Cy x + Dyw w.
            - select="first": Cy picks the first ny states (identity rows).
            - select="random": random orthonormal rows (stable numerics).
            - dyw_zero=True -> Dyw = 0 (typical unless sensor sees w).
            """
            ny = int(min(ny, nx))
            if select == "random":
                Q, _ = np.linalg.qr(np.random.randn(nx, nx))
                Cy = Q[:ny, :]
            else:
                Cy = np.zeros((ny, nx))
                Cy[np.arange(ny), np.arange(ny)] = 1.0

            Dyw = np.zeros((ny, nw)) if dyw_zero else 0.05 * np.random.randn(ny, nw)
            return Cy, Dyw

        def make_performance_A(nx: int, nu: int,
                            Qx_diag: np.ndarray | None = None,
                            Ru_diag: np.ndarray | None = None) -> tuple[np.ndarray, np.ndarray, np.ndarray]:
            """
            Option A (state-input H2-style):
                z = [ Qx^{1/2} x ; Ru^{1/2} u ]
            Returns (Cz, Dzw, Dzu) with Dzw = 0, nz = nx + nu.
            """
            if Qx_diag is None:
                Qx_diag = np.ones(nx)
            if Ru_diag is None:
                Ru_diag = 0.1 * np.ones(nu)

            Qx_sqrt = np.sqrt(np.maximum(Qx_diag, 0.0))
            Ru_sqrt = np.sqrt(np.maximum(Ru_diag, 1e-12))

            Cz_top = np.diag(Qx_sqrt)              # (nx x nx)
            Cz_bot = np.zeros((nu, nx))            # pad for input rows
            Cz = np.vstack([Cz_top, Cz_bot])       # (nx+nu) x nx

            Dzu_top = np.zeros((nx, nu))
            Dzu_bot = np.diag(Ru_sqrt)
            Dzu = np.vstack([Dzu_top, Dzu_bot])    # (nx+nu) x nu

            Dzw = np.zeros((nx + nu, 0))           # placeholder (use correct nw when instantiating)
            return Cz, Dzw, Dzu

        def make_performance_B(Cy: np.ndarray, Dyw: np.ndarray,
                            Wy_diag: np.ndarray | None,
                            Ru_diag: np.ndarray | None) -> tuple[np.ndarray, np.ndarray, np.ndarray]:
            """
            Option B (output-input H2-style):
                z = [ Wy y ; Ru^{1/2} u ]  with y = Cy x + Dyw w
            Returns (Cz, Dzw, Dzu) with nz = ny + nu.
            """
            ny, nx = Cy.shape
            nu = int(Ru_diag.shape[0]) if Ru_diag is not None else None

            if Wy_diag is None:
                Wy_diag = np.ones(ny)
            if Ru_diag is None:
                raise ValueError("Ru_diag must be provided for performance Option B.")

            Wy = np.diag(np.sqrt(np.maximum(Wy_diag, 0.0)))
            Ru_sqrt = np.sqrt(np.maximum(Ru_diag, 1e-12))

            Cz_top = Wy @ Cy                         # (ny x nx)
            Cz_bot = np.zeros((nu, nx))              # (nu x nx)
            Cz = np.vstack([Cz_top, Cz_bot])         # (ny+nu) x nx

            Dzw_top = Wy @ Dyw                       # (ny x nw)
            # Dzw_bot is zero (inputs don't map w directly here)
            # Allocate at runtime if nw known; we return empty placeholder for shape deferral.
            Dzw = np.vstack([Dzw_top, np.zeros((nu, Dzw_top.shape[1]))]) if Dzw_top.size else np.zeros((ny + nu, 0))

            Dzu_top = np.zeros((ny, nu))
            Dzu_bot = np.diag(Ru_sqrt)
            Dzu = np.vstack([Dzu_top, Dzu_bot])      # (ny+nu) x nu

            return Cz, Dzw, Dzu


        # ======================================================================
        # YAML-driven builder (optional)
        # problem___parameters.yaml schema:
        # params:
        #   dimensions: {nx: 4, nw: 2, nu: 2, nz: 3, ny: 2}
        #   outputs:
        #     mode: "A"        # "A" -> state-input, "B" -> output-input
        #     Qx_diag: [1,1,0.3,0.1]
        #     Ru_diag: [0.1,0.1]
        #     Wy_diag: [1,1]   # only for mode "B"
        #     measured: {select: "first", dyw_zero: true}
        # ======================================================================

        """
        Load dimensions + output specs from YAML and produce (Cz, Dzw, Dzu, Cy, Dyw).
        If PyYAML is unavailable, raise a helpful error.
        """

        with open(yaml_path, "r", encoding="utf-8") as f:
            cfg = yaml.safe_load(f)

        p = cfg.get("params", {})
        dims = p.get("dimensions", {})
        nx = int(dims.get("nx"))
        nw = int(dims.get("nw"))
        nu = int(dims.get("nu"))
        ny = int(dims.get("ny"))

        outspec = p.get("outputs", {})
        mode = str(outspec.get("mode", "A")).upper().strip()

        meas = outspec.get("measured", {}) or {}
        Cy, Dyw = make_Cy_Dyw(
            nx=nx, ny=ny, nw=nw,
            select=str(meas.get("select", "first")).lower(),
            dyw_zero=bool(meas.get("dyw_zero", True))
        )

        if mode == "A":
            Qx_diag = np.array(outspec.get("Qx_diag", [1.0] * nx), dtype=float)
            Ru_diag = np.array(outspec.get("Ru_diag", [0.1] * nu), dtype=float)
            Cz, _, Dzu = make_performance_A(nx, nu, Qx_diag=Qx_diag, Ru_diag=Ru_diag)
            # Fill Dzw with the correct width (nw), zeros by construction in Option A
            Dzw = np.zeros((Cz.shape[0], nw))
        elif mode == "B":
            Wy_diag = np.array(outspec.get("Wy_diag", [1.0] * ny), dtype=float)
            Ru_diag = np.array(outspec.get("Ru_diag", [0.1] * nu), dtype=float)
            Cz, Dzw, Dzu = make_performance_B(Cy, Dyw, Wy_diag=Wy_diag, Ru_diag=Ru_diag)
            # Ensure Dzw has correct width
            if Dzw.shape[1] == 0:
                Dzw = np.zeros((Cz.shape[0], nw))
            elif Dzw.shape[1] != nw:
                # pad or trim to nw just in case
                if Dzw.shape[1] > nw:
                    Dzw = Dzw[:, :nw]
                else:
                    Dzw = np.hstack([Dzw, np.zeros((Dzw.shape[0], nw - Dzw.shape[1]))])
        else:
            raise ValueError("outputs.mode must be 'A' or 'B'")

        return Cz, Dzw, Dzu, Cy, Dyw

    def build_AB_from_yaml(self, yaml_path: str) -> tuple[np.ndarray, np.ndarray, np.ndarray]:
        """
        Build (A, Bu, Bw) from YAML.
        YAML schema example:
        params:
            dimensions: {nx: 4, nw: 2, nu: 2}
            plant:
            type: "random_stable"   # or "explicit"
            seed: 42
            A:
                eig_min: 0.7
                eig_max: 0.95
            Bu:
                scale: 0.5
            Bw:
                rank: 2
                scale: 1.0
            # If type == "explicit", provide numeric arrays:
            # A_mat: [[...],[...],...]
            # Bu_mat: [[...],[...],...]
            # Bw_mat: [[...],[...],...]
        """

        def _stable_A_random(nx: int, eig_min: float, eig_max: float, rng: np.random.Generator) -> np.ndarray:
            """Random orthogonal similarity of diagonal eigenvalues in (eig_min, eig_max)."""
            M = rng.normal(size=(nx, nx))
            Q, _ = np.linalg.qr(M)
            eigvals = eig_min + (eig_max - eig_min) * rng.random(nx)
            return Q @ np.diag(eigvals) @ Q.T


        def _random_full(shape: tuple[int, int], scale: float, rng: np.random.Generator) -> np.ndarray:
            return scale * rng.normal(size=shape)


        def _random_orthonormal_columns(nx: int, rank: int, scale: float, rng: np.random.Generator) -> np.ndarray:
            Q, _ = np.linalg.qr(rng.normal(size=(nx, nx)))
            return Q[:, :rank] * scale

        if yaml is None:
            raise ImportError("PyYAML not available. Install with `pip install pyyaml`.")
        with open(yaml_path, "r", encoding="utf-8") as f:
            cfg = yaml.safe_load(f)

        p = cfg.get("params", {})
        dims = p.get("dimensions", {})
        nx = int(dims["nx"]); nw = int(dims["nw"]); nu = int(dims["nu"])
        plant_cfg = p.get("plant", {}) or {}
        ptype = str(plant_cfg.get("type", "random_stable")).lower()
        seed = int(plant_cfg.get("seed", 0))
        rng = np.random.default_rng(seed)

        if ptype == "explicit":
            A = np.array(plant_cfg["A_mat"], dtype=float)
            Bu = np.array(plant_cfg["Bu_mat"], dtype=float)
            Bw = np.array(plant_cfg["Bw_mat"], dtype=float)
            if A.shape != (nx, nx) or Bu.shape != (nx, nu) or Bw.shape != (nx, nw):
                raise ValueError("Explicit matrices do not match (nx,nu,nw) in YAML.")
            return A, Bu, Bw

        # random_stable (default)
        Aconf = plant_cfg.get("A", {}) or {}
        eig_min = float(Aconf.get("eig_min", 0.7))
        eig_max = float(Aconf.get("eig_max", 0.95))
        if not (0.0 < eig_min < eig_max < 1.0):
            raise ValueError("Require 0 < eig_min < eig_max < 1 for stability.")

        Buconf = plant_cfg.get("Bu", {}) or {}
        Bu_scale = float(Buconf.get("scale", 0.5))

        Bwconf = plant_cfg.get("Bw", {}) or {}
        Bw_rank = int(Bwconf.get("rank", nw))
        Bw_scale = float(Bwconf.get("scale", 1.0))
        Bw_rank = max(1, min(Bw_rank, min(nx, nw)))

        A = _stable_A_random(nx, eig_min, eig_max, rng)
        Bu = _random_full((nx, nu), Bu_scale, rng)
        Bw = _random_orthonormal_columns(nx, Bw_rank, Bw_scale, rng)

        # If requested Bw has fewer columns than nw, pad with zeros to match declared nw
        if Bw.shape[1] < nw:
            Bw = np.hstack([Bw, np.zeros((nx, nw - Bw.shape[1]))])

        return A, Bu, Bw

    def get_dimensions_from_yaml(self, yaml_path: str) -> tuple[int, int, int, int, int]:
        """
        Extract (nx, nw, nu, ny, nz) from YAML file.
        """
        if yaml is None:
            raise ImportError("PyYAML not available. Install with `pip install pyyaml`.")
        with open(yaml_path, "r", encoding="utf-8") as f:
            cfg = yaml.safe_load(f)

        p = cfg.get("params", {})
        dims = p.get("dimensions", {})
        nx = int(dims.get("nx", 0))
        nw = int(dims.get("nw", 0))
        nu = int(dims.get("nu", 0))
        ny = int(dims.get("ny", 0))

        out = p.get("outputs", {})
        mode = str(out.get("mode", "A")).upper().strip()
        nz = nx + nu if mode == "A" else ny + nu
        return nx, nw, nu, ny, nz

    # ------------------------- DATA-DRIVEN DDD CONSTRUCTION -------------------------

    def make_matrices_from_data(
        self, 
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


        ## UTILS functions
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


        if not data_csv:
            #raise ValueError("Provide data_csv='path/to/file.csv' to read data.")
            return self.make_example_system()

        blocks = _build_blocks_from_csv(data_csv, delimiter=delimiter)
        X = blocks["X"]          # (nx x T)
        U = blocks["U"]          # (nu x T)
        X_next = blocks["X_next"]

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
        Bw, *_ = _bw_from_residual(R, nw=nw)

        """
            Y = blocks["Y"]          # may be None
            Z = blocks["Z"]          # may be None


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
        """

        Cz, Dzw, Dzu, Cy, Dyw = self.build_out_matrices(yaml_path="problem___parameters.yaml")

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

    def make_example_system(self, yaml_path="problem___parameters.yaml"):
        """
        Replace this with your real matrices.
        Discrete-time example with modest dimensions.
        """

        if yaml is None:
            raise ImportError("PyYAML not available. Install with `pip install pyyaml`.")
        with open(yaml_path, "r", encoding="utf-8") as f:
            cfg = yaml.safe_load(f)

        p = cfg.get("params", {})
        dims = p.get("dimensions", {})
        nx = int(dims["nx"]); ny = int(dims["ny"]); nu = int(dims["nu"])

        A, Bu, Bw = self.build_AB_from_yaml(yaml_path=yaml_path)
        Cz, Dzw, Dzu, Cy, Dyw = self.build_out_matrices(yaml_path=yaml_path)

        plant = Plant(A=A, Bw=Bw, Bu=Bu, Cz=Cz, Dzw=Dzw, Dzu=Dzu, Cy=Cy, Dyw=Dyw)

        # Full-order controller as a starting point; zero dynamics + small static gain
        nxc = nx
        Ac0 = 0.0 * np.eye(nxc)
        Bc0 = 0.1 * np.eye(nxc, ny)
        Cc0 = 0.1 * np.eye(nu, nxc)
        Dc0 = 0.0 * np.eye(nu, ny)

        ctrl0 = Controller(Ac=Ac0, Bc=Bc0, Cc=Cc0, Dc=Dc0)
        return plant, ctrl0


    """
    USAGE

    plant, ctrl0 = get_system(FROM_DATA=True,
                            data_csv="out/data/session01.csv",
                            delimiter=",",
                            nw=None, ny=None, nz=None,
                            ridge=1e-6)

    """


# ------------------------------------------------------------------------------
