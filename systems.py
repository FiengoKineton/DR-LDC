# systems.py
import numpy as np
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
