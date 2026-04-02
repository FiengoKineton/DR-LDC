import numpy as np
from scipy.linalg import expm
import cvxpy as cp

# Wind turbine blade control: Distributionally robust synthesis
# States: [omega, h, h_dot, phi, phi_dot, beta, beta_dot]
# Constraints: RMS h <= 2 m (E[h^2] <= 4), RMS phi <= 0.01 rad (E[phi^2] <= 1e-4)
# Turbulence: 8% intensity (sigma = 0.96 m/s), Dryden filter

# Turbine parameters
J = 4e7  # Rotor inertia (kg·m^2)
k_omega = -1e5  # Torque sensitivity to rotor speed (N·m·s/rad)
k_h = 1e4  # Torque sensitivity to flapwise displacement (N·m/m)
k_phi = -1e4  # Torque sensitivity to torsional angle (N·m/rad)
k_beta = -1e6  # Torque sensitivity to pitch angle (N·m/rad)
k_v = 5e5  # Torque sensitivity to wind speed (N·m·s/m)
m = 1e4  # Effective blade mass (kg)
omega_f = 6.28  # Flapwise natural frequency (rad/s)
zeta_f = 0.05  # Flapwise damping ratio
f_omega = 1e3  # Force sensitivity to rotor speed (N·s/rad)
f_beta = -1e4  # Force sensitivity to pitch angle (N/rad)
f_v = 2e4  # Force sensitivity to wind speed (N·s/m)
I_t = 1e5  # Torsional inertia (kg·m^2)
omega_t = 31.4  # Torsional natural frequency (rad/s)
zeta_t = 0.02  # Torsional damping ratio
m_omega = 1e2  # Moment sensitivity to rotor speed (N·m·s/rad)
m_beta = -1e3  # Moment sensitivity to pitch angle (N·m/rad)
m_v = 1e4  # Moment sensitivity to wind speed (N·m·s/m)
omega_p = 10  # Pitch actuator natural frequency (rad/s)
zeta_p = 0.7  # Pitch actuator damping ratio

# Turbine state-space matrices
A_continuous = np.array([
    [k_omega/J, k_h/J, 0, k_phi/J, 0, k_beta/J, 0],  # omega_dot
    [0, 0, 1, 0, 0, 0, 0],  # h_dot
    [f_omega/m, -omega_f**2, -2*zeta_f*omega_f, 0, 0, f_beta/m, 0],  # h_ddot
    [0, 0, 0, 0, 1, 0, 0],  # phi_dot
    [m_omega/I_t, 0, 0, -omega_t**2, -2*zeta_t*omega_t, m_beta/I_t, 0],  # phi_ddot
    [0, 0, 0, 0, 0, 0, 1],  # beta_dot
    [0, 0, 0, 0, 0, -omega_p**2, -2*zeta_p*omega_p]  # beta_ddot
])
B_continuous = np.array([[0], [0], [0], [0], [0], [0], [omega_p**2]])  # Input: beta_dot_c
E_continuous = np.array([
    [k_v/J, 0],  # omega (v_z only)
    [0, 0],      # h
    [f_v/m, f_v/m],  # h_dot (v_x, v_z)
    [0, 0],      # phi
    [m_v/I_t, m_v/I_t],  # phi_dot (v_x, v_z)
    [0, 0],      # beta
    [0, 0]       # beta_dot
])  # 7x2: [v_x, v_z]

# Discretize system (dt = 0.01s)
dt = 0.05
n_x = A_continuous.shape[0]
A = expm(A_continuous * dt)

def discretize_input(A_c, B_c, dt):
    n = A_c.shape[0]
    m = B_c.shape[1]
    Phi = expm(np.block([[A_c, B_c], [np.zeros((m, n)), np.zeros((m, m))]]) * dt)
    A_d = Phi[:n, :n]
    B_d = Phi[:n, n:]
    return A_d, B_d

_, B_u = discretize_input(A_continuous, B_continuous, dt)
_, B_w = discretize_input(A_continuous, E_continuous, dt)

# Consider scaling of random noise
B_w = B_w/np.sqrt(dt)

# System dimensions
n_y = 3
n_u = B_u.shape[1]
n_w = B_w.shape[1]
n_v = n_x + n_u  # 12

# Performance weights
Q = np.eye(n_x) #np.diag([1e2, 1e-1, 1e-1, 1e1, 1e1, 1e-1, 1e-1, 0, 0])
R = 1e0
C_v = np.vstack([np.diag(np.sqrt(Q.diagonal())),np.zeros((n_u,n_x))])
D_vu = np.vstack([np.zeros((n_x,n_u)),np.array([[np.sqrt(R)]])])
D_vw = np.zeros((n_x+n_u, n_w))
C_y = np.zeros((n_y,n_x))
C_y[0,0] = 1
C_y[1,2] = 1
C_y[2,4] = 1
D_yu = np.zeros((n_y,n_u))
D_yw = np.zeros((n_y,n_w))

# Regularize the problem with measurement noise
B_w = np.block([[B_w, (1e-4)*np.eye(n_x), np.zeros((n_x,n_y))]])
D_vw = np.block([[D_vw, np.zeros((n_v,n_x + n_y))]])
D_yw = np.block([[D_yw,np.zeros((n_y, n_x)),(1e-4)*np.eye(n_y)]])
n_w = B_w.shape[1]

# Constrained outputs
C_v1 = np.zeros((1, n_x))
C_v1[0, 1] = 1  # h
D_v1u = np.array([[0]])
D_v1w = np.zeros((1, n_w))

C_v2 = np.zeros((1, n_x))
C_v2[0, 3] = 1  # phi
D_v2u = np.array([[0]])
D_v2w = np.zeros((1, n_w))

# Wasserstein parameters
Sigma_nom = np.eye(n_w)
gamma = 0.5

# LMI variables
X = cp.Variable((n_x, n_x), symmetric=True)
Y = cp.Variable((n_x, n_x), symmetric=True)
K = cp.Variable((n_x, n_x))
L = cp.Variable((n_x, n_y))
M = cp.Variable((n_u, n_x))
N = cp.Variable((n_u, n_y))
P = cp.Variable((n_w, n_w), symmetric=True)

# LMI matrices
P_mat = cp.bmat([[Y, np.eye(n_x)], [np.eye(n_x), X]])
A_bar = cp.bmat([
    [A @ Y + B_u @ M, A + B_u @ N @ C_y],
    [K, X @ A + L @ C_y]
])
B_bar = cp.bmat([
    [B_w + B_u @ N @ D_yw],
    [X @ B_w + L @ D_yw]
])
C_bar = cp.bmat([
    [C_v @ Y + D_vu @ M, C_v + D_vu @ N @ C_y]
])
D_bar = D_vw + D_vu @ N @ D_yw
                 
LMI1 = cp.bmat([
    [-P_mat, A_bar.T, C_bar.T],
    [A_bar, -P_mat, np.zeros((2*n_x, n_v))],
    [C_bar, np.zeros((n_v, 2*n_x)), -np.eye(n_v)]
])
LMI2 = cp.bmat([
    [-P, B_bar.T, D_bar.T],
    [B_bar, -P_mat, np.zeros((2*n_x, n_v))],
    [D_bar, np.zeros((n_v, 2*n_x)), -np.eye(n_v)]
])

# Optimization problem
objective = cp.Minimize(cp.trace(P @ Sigma_nom))
constraints = [
    LMI1 << 0,
    LMI2 << 0,
    P_mat >> 0,
    P >> 0
]

# Solve
success = False
print("Attempting to solve with MOSEK...")
try:
    problem = cp.Problem(objective, constraints)
    problem.solve(solver=cp.MOSEK, verbose=True, mosek_params={
        'MSK_DPAR_INTPNT_CO_TOL_PFEAS': 1e-8,
        'MSK_DPAR_INTPNT_CO_TOL_DFEAS': 1e-8,
        'MSK_DPAR_INTPNT_CO_TOL_REL_GAP': 1e-8,
        'MSK_DPAR_INTPNT_TOL_STEP_SIZE': 1e-6
    })
    print(f"MOSEK status: {problem.status}")
    if problem.status == cp.OPTIMAL:
        success = True
except Exception as mosek_e:
    print(f"MOSEK error: {mosek_e}")

if not success:
    print("MOSEK failed, trying SCS...")
    try:
        problem.solve(solver=cp.SCS, verbose=True, eps=1e-4, max_iters=100000)
        print(f"SCS status: {problem.status}")
        if problem.status in [cp.OPTIMAL, cp.OPTIMAL_INACCURATE]:
            success = True
            if problem.status == cp.OPTIMAL_INACCURATE:
                print("Warning: SCS returned 'optimal_inaccurate'.")
        else:
            print(f"SCS failed with status: {problem.status}")
    except Exception as scs_e:
        print(f"SCS error: {scs_e}")

if not success:
    print("Optimization error: Both solvers failed.")
    exit(1)

# Extract controller
X_val = X.value
Y_val = Y.value
K_val = K.value
L_val = L.value
M_val = M.value
N_val = N.value

try:
    U, S, Vt = np.linalg.svd(np.eye(n_x) - X_val @ Y_val)
    V = Vt.T
    epsilon = 1e-10
    S_sqrt = np.diag(np.sqrt(np.maximum(S, epsilon)))
    S_sqrt_inv = np.diag(1.0 / np.sqrt(np.maximum(S, epsilon)))
    U_new = U @ S_sqrt
    V_new = V @ S_sqrt
    U_new_inv = S_sqrt_inv @ U.T
    V_new_inv_T = V @ S_sqrt_inv
    A_c = U_new_inv @ (K_val - X_val @ A @ Y_val - L_val @ C_y @ Y_val - X_val @ B_u @ (M_val - N_val @ C_y @ Y_val)) @ V_new_inv_T
    B_c = U_new_inv @ (L_val - X_val @ B_u @ N_val)
    C_c = (M_val - N_val @ C_y @ Y_val) @ V_new_inv_T
    D_c = N_val
except np.linalg.LinAlgError:
    print("Error: Singular matrix in controller reconstruction. Using fallback.")
    A_c = K_val - X_val @ A @ Y_val - L_val @ Y_val - X_val @ B_u @ M_val
    B_c = L_val
    C_c = M_val
    D_c = N_val

# Closed-loop system
n_c = n_x
A_cl = np.block([
    [A + B_u @ D_c @ C_y, B_u @ C_c],
    [B_c @ C_y, A_c]
])
B_cl = np.block([[B_w + B_u @ D_c @ D_yw], [B_c @ D_yw]])
C_cl = np.block([[C_v + D_vu @ D_c @ C_y, D_vu @ C_c]])
D_cl = np.block([[D_vw + D_vu @ D_c @ D_yw]])



# Stability check
eigvals_cl = np.linalg.eigvals(A_cl)
if any(abs(eigvals_cl) >= 1 - 1e-6):
    print("Warning: Closed-loop system may be unstable")
    print(abs(eigvals_cl))
else:
    print("Closed-loop system is stable")
    print(abs(eigvals_cl))

# Save data
controller_data = {
    'A_c': A_c, 'B_c': B_c, 'C_c': C_c, 'D_c': D_c,
    'A_cl': A_cl, 'B_cl': B_cl, 'C_cl': C_cl, 'D_cl': D_cl,
    'A': A, 'B_u': B_u, 'B_w': B_w,
    'C_v': C_v, 'D_vu': D_vu, 'D_vw': D_vw,
    'C_y': C_y, 'D_yw': D_yw,
    'Q': Q, 'R': R, 'gamma': gamma, 'Sigma_nom': Sigma_nom,
    'dt': dt
}
np.save('controller_data_h2.npy', controller_data)
print("Controller and system matrices saved to 'controller_data_h2.npy'")