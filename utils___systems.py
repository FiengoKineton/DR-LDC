import numpy as np
import yaml
from dataclasses import dataclass

@dataclass
class Plant:
    A: np.ndarray
    Bw: np.ndarray
    Bu: np.ndarray
    Cz: np.ndarray
    Dzw: np.ndarray
    Dzu: np.ndarray
    Cy: np.ndarray
    Dyw: np.ndarray

    def dims(self):
        nx = self.A.shape[0]
        nw = self.Bw.shape[1]
        nu = self.Bu.shape[1]
        nz = self.Cz.shape[0]
        ny = self.Cy.shape[0]
        return nx, nw, nu, nz, ny

@dataclass
class Controller:
    Ac: np.ndarray
    Bc: np.ndarray
    Cc: np.ndarray
    Dc: np.ndarray

    def dims(self):
        nxc = self.Ac.shape[0]
        return nxc

@dataclass
class Plant_cl: 
    Acl: np.ndarray
    Bcl: np.ndarray
    Ccl: np.ndarray
    Dcl: np.ndarray

    def dims(self):
        nx_cl = self.Acl.shape[0]
        nw_cl = self.Bcl.shape[1]
        nz_cl = self.Ccl.shape[0]
        return nx_cl, nw_cl, nz_cl

@dataclass
class Plant_k:
    A: np.ndarray
    B: np.ndarray
    C: np.ndarray
    D: np.ndarray
    K: np.ndarray
    V: np.ndarray

    def dims(self):
        nx = self.A.shape[0]
        nu = self.B.shape[1]
        nz = self.C.shape[0]
        nw = self.V.shape[1]
        return nx, nu, nz, nw

@dataclass
class Noise:
    Sigma_nom: np.ndarray
    var: float
    n: int
    avrg: float
    gamma: float

    def dims(self):
        nw = self.Sigma_nom.shape[0]
        return nw

@dataclass
class Data:
    X: np.ndarray
    X_next: np.ndarray
    U: np.ndarray
    Y: np.ndarray
    Z: np.ndarray
    W: np.ndarray

    rx: float
    ry: float
    rz: float

    def dim(self, M):
        n_m = self.X.shape[0]
        T_m = self.X.shape[1]
        return n_m, T_m
    def dims(self):
        nx, Tx = self.dim(self.X)
        nu, Tu = self.dim(self.U)
        ny, Ty = self.dim(self.Y)
        nz, Tz = self.dim(self.Z)
        return (nx, Tx), (nu, Tu), (ny, Ty), (nz, Tz)
    def get_lamd(self):
        return self.rx, self.ry, self.rz
    def get_data(self):
        return self.X, self.X_next, self.U, self.Y, self.Z

@dataclass
class DROLMIResult:
    solver: str
    status: str
    obj_value: float | None
    gamma: float
    lambda_opt: float | None
    Q: np.ndarray | None
    X: np.ndarray | None
    Y: np.ndarray | None
    K: np.ndarray | None
    L: np.ndarray | None
    M: np.ndarray | None
    N: np.ndarray | None
    Pbar: np.ndarray | None
    Abar: np.ndarray | None
    Bbar: np.ndarray | None
    Cbar: np.ndarray | None
    Dbar: np.ndarray | None
    Tp: np.ndarray | None
    P: np.ndarray | None

@dataclass
class DROLMIResultUpd:
    solver: str
    status: str
    obj_value: float | None
    gamma: float
    Sigma: np.ndarray | None

    rx: float | None
    ry: float | None
    rz: float | None
    lamda: float | None

    Q: np.ndarray | None
    X: np.ndarray | None
    Y: np.ndarray | None
    K: np.ndarray | None
    L: np.ndarray | None
    M: np.ndarray | None
    N: np.ndarray | None
    P: np.ndarray | None

    A1: np.ndarray | None
    B1: np.ndarray | None
    C1: np.ndarray | None
    D1: np.ndarray | None    
    A2: np.ndarray | None
    B2: np.ndarray | None
    C2: np.ndarray | None
    D2: np.ndarray | None
    A_same: bool | None
    B_same: bool | None
    C_same: bool | None
    D_same: bool | None

    Ax: np.ndarray | None
    Bu: np.ndarray | None
    Bw: np.ndarray | None
    Cy: np.ndarray | None
    Dyw: np.ndarray | None
    Cz: np.ndarray | None
    Dzu: np.ndarray | None
    Dzw: np.ndarray | None

    Acl: np.ndarray | None
    Bcl: np.ndarray | None
    Ccl: np.ndarray | None
    Dcl: np.ndarray | None

    Ac: np.ndarray | None
    Bc: np.ndarray | None
    Cc: np.ndarray | None
    Dc: np.ndarray | None


    def _get_cl(self):
        return self.Acl, self.Bcl, self.Ccl, self.Dcl
    def _get_plant(self):
        return self.Ax, self.Bu, self.Bw, self.Cy, self.Dyw, self.Cz, self.Dzu, self.Dzw
    def _get_ctrl(self):
        return self.Ac, self.Bc, self.Cc, self.Dc
    def _get_r(self):
        return self.rx, self.ry, self.rz
    def _get_l(self):
        return self.lamda