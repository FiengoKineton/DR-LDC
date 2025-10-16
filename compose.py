# compose.py
import numpy as np
from systems import Plant, Controller

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
