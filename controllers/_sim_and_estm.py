import sys, numpy as np

from disturbances import Disturbances
from simulate import Open_Loop
from utils.helpers import _pseudo_inv





class Data_Estimator_and_Simulator:
    def __init__(self, api, vals, noise, cfg):
        self.api = api
        self.vals = vals
        self.noise = noise
        self.cfg = cfg

        self.gamma = noise.gamma
        self.var = noise.var
        _, _, self.plot = vals

        self.datasets = None
        self.avg = None

        self.x = self.u = self.y = self.z = self.x_next = None
        self.T = self.nx = self.nu = self.ny = self.nz = self.nw = None

        self.Ax = self.Bu = self.Bw = None
        self.Cy = self.Dyw = self.Cz = self.Dzu = self.Dzw = None
        self.Sigma_nom = None
        self.R = None
        self.w = None
        self.beta = None
        self.gamma_est = None
        self.bw_info = None


    def simulate_datasets(self):
        op = Open_Loop(MAKE_DATA=False, EVAL_FROM_PATH=False, DATASETS=True, N=self.cfg.N_sims)
        self.datasets = op.datasets
        return self.datasets


    def select_dataset(self):
        from analysis import select_representative_run

        if self.datasets is None:
            self.simulate_datasets()

        self.avg = select_representative_run(self.datasets) if self.cfg.N_sims != 1 else self.datasets

        self.x = self.avg["X"]
        self.u = self.avg["U"]
        self.y = self.avg["Y"]
        self.z = self.avg["Z"]
        self.x_next = self.avg["X_next"]

        self.T = self.x.shape[1]
        self.nx = self.x.shape[0]
        self.nu = self.u.shape[0]
        self.ny = self.y.shape[0]
        self.nz = self.z.shape[0]

        if self.plot:
            from analysis import plot_first3_and_mean
            for key in ["X", "U", "Y", "Z"]:
                plot_first3_and_mean(self.datasets, out=self.avg, key=key, title_prefix="Closed-loop")

        return self.avg
    

    def evaluate_beta(self):

        # OLD (wrong) 
        self.beta = np.linalg.norm(self.R, "fro") / max(self.smin, 1e-12)



    def estimate_state_mats(self):
        if self.x is None:
            self.select_dataset()

        Dx = np.vstack([self.x, self.u])
        Ox = self.x_next @ _pseudo_inv(Dx)

        self.Ax = Ox[:, :self.nx]
        self.Bu = Ox[:, self.nx:self.nx + self.nu]

        ss = np.linalg.svd(Dx, compute_uv=False)
        self.smin = float(ss[-1]) if ss.size else 0.0

        self.R = self.x_next - (self.Ax @ self.x + self.Bu @ self.u)
        self.beta = self.evaluate_beta()

        return self.Ax, self.Bu, self.R, self.beta
    
    def estimate_disturbance_model(self, mode: str | None = None, eta: float | None = None):
        if self.Ax is None or self.Bu is None:
            self.estimate_state_mats()

        mode = mode or self.cfg.bw_mode
        eta = eta or self.cfg.bw_eta

        Bw, R, Sigma_w_hat, info = self._estimate_Bw_from_residuals(
            Ax_hat=self.Ax,
            Bu_hat=self.Bu,
            eta=eta,
            mode=mode,
        )

        self.Bw = Bw
        self.R = R
        self.nw = Bw.shape[1]

        self.w = _pseudo_inv(self.Bw) @ self.R
        d = Disturbances(n=self.nw)

        self.Sigma_nom = self.var * np.eye(self.nw)
        gamma2, *_ = d._estimate_gamma_with_ci(self.w.T)

        if self.cfg.verbose:
            print(f"Estimated disturbance dimension nw: {self.nw}")
            print(f"Configured gamma: {self.gamma}, estimated gamma2: {gamma2}")
            print(f"Estimated Sigma_nom:\n{self.Sigma_nom}")
            print(f"True Sigma_nom:\n{d.Sigma_test}")

        return self.Bw, self.R, self.Sigma_nom, info

    def estimate_output_mats(self):
        if self.Bw is None:
            self.estimate_disturbance_model()

        Dy = np.vstack([self.x, self.w])
        Dz = np.vstack([self.x, self.u, self.w])

        Oy = self.y @ _pseudo_inv(Dy)
        Oz = self.z @ _pseudo_inv(Dz)

        self.Cy = Oy[:, :self.nx]
        self.Dyw = Oy[:, self.nx:self.nx + self.nw]

        if not self.cfg.real_Z_mats:
            self.Cz = Oz[:, :self.nx]
            self.Dzu = Oz[:, self.nx:self.nx + self.nu]
            self.Dzw = Oz[:, self.nx + self.nu:self.nx + self.nu + self.nw]
        else:
            self.Cz, self.Dzw, self.Dzu, *_ = self.api.build_out_matrices(nw=self.nw)

        self.Bw, self.Dzw, self.Dyw, self.nw, self.Sigma_nom = self.api._augment_matrices(
            B_w=self.Bw,
            D_vw=self.Dzw,
            D_yw=self.Dyw,
            var=self.var,
            Sigma_nom=self.Sigma_nom,
            N=(1, 1),
        )

        return self.Cy, self.Dyw, self.Cz, self.Dzu, self.Dzw
       

    def _residual_anisotropy_weights(self, R, *, floor=1e-12, mode="sqrt"):
        nx, T = R.shape
        S = (R @ R.T) / max(T, 1)
        S = 0.5 * (S + S.T) + floor * np.eye(nx)

        s_vals, U = np.linalg.eigh(S)
        idx = np.argsort(s_vals)[::-1]
        s = np.clip(s_vals[idx], 0.0, None)
        U = U[:, idx]

        if mode == "sqrt":
            w = np.sqrt(s)
        elif mode == "linear":
            w = s
        else:
            raise ValueError("mode must be 'sqrt' or 'linear'")

        w = w / max(np.max(w), 1e-18)
        return U, s, w
    

    def _estimate_Bw_from_residuals(
        self,
        Ax_hat,
        Bu_hat,
        eta=0.95,
        eps=1e-12,
        mode="default",
        Sigma_w_known=None,
    ):
        R = self.x_next - (Ax_hat @ self.x + Bu_hat @ self.u)
        nx, T = R.shape

        S = (R @ R.T) / max(T, 1)
        S = 0.5 * (S + S.T) + eps * np.eye(nx)

        s_vals, U = np.linalg.eigh(S)
        s_vals = np.clip(s_vals, 0.0, None)
        order = np.argsort(s_vals)[::-1]
        s = s_vals[order]
        U = U[:, order]

        total = max(float(np.sum(s)), 1e-18)
        cum = np.cumsum(s) / total
        nw = int(np.clip(np.searchsorted(cum, eta) + 1, 1, nx))

        Up = U[:, :nw]
        sp = s[:nw]
        sp_sqrt = np.sqrt(sp)

        if mode == "default":
            U_s, s_s, _ = np.linalg.svd(S, full_matrices=False)
            cum = np.cumsum(s_s) / max(np.sum(s_s), 1e-18)
            nw = int(np.clip(np.searchsorted(cum, 0.95) + 1, 1, nx))
            Bw = U_s[:, :nw] @ np.diag(np.sqrt(s_s[:nw]))
            return Bw, R, None, {
                "nw": nw,
                "energy": float(np.sum(s_s[:nw])) / float(np.sum(s_s)),
                "s_vals": s_s,
            }

        elif mode == "white":
            Bw_hat = Up @ np.diag(sp_sqrt)
            Sigma_w_hat = np.eye(nw)

        elif mode == "factor":
            Bw_hat = Up
            Sigma_w_hat = np.diag(sp)

        elif mode == "known_cov":
            if Sigma_w_known is None:
                raise ValueError("mode='known_cov' requires Sigma_w_known.")

            lam, Q = np.linalg.eigh(np.atleast_2d(Sigma_w_known))
            lam = np.clip(lam, eps, None)
            Sigma_w_mhalf = Q @ np.diag(lam**-0.5) @ Q.T

            Bw_hat = Up @ np.diag(sp_sqrt) @ Sigma_w_mhalf
            Sigma_w_hat = Sigma_w_known
        else:
            raise ValueError("mode must be one of {'white','factor','known_cov'}.")

        info = {
            "nw": nw,
            "energy": float(np.sum(sp)) / total if total > 0 else 0.0,
            "s_vals": s,
        }
        return Bw_hat, R, Sigma_w_hat, info

