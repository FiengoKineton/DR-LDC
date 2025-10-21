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


