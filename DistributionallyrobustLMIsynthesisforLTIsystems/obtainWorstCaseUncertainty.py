import numpy as np
from scipy.linalg import solve_discrete_lyapunov
import cvxpy as cp

# Load closed-loop matrices
robust_data = np.load('controller_data_robust.npy', allow_pickle=True).item()
h2_data = np.load('controller_data_h2.npy', allow_pickle=True).item()

# System dimensions
gamma = robust_data['gamma']  # 1.5
Sigma_nom = robust_data['Sigma_nom'][0:2,0:2]  # I_2

# Closed-loop matrices
controllers = {
    'robust': {
        'A_cl': robust_data['A_cl'],  # 22x22
        'B_cl': robust_data['B_cl'],  # 22x2
        'C_cl': robust_data['C_cl'],  # 12x22
        'D_cl': robust_data['D_cl'],  # 12x2
        'C_c': robust_data['C_c'],    # 1x11
        'D_c': robust_data['D_c'],    # 1x11
        'C_y': robust_data['C_y']     # 11x11
    },
    'H2': {
        'A_cl': h2_data['A_cl'],
        'B_cl': h2_data['B_cl'],
        'C_cl': h2_data['C_cl'],
        'D_cl': h2_data['D_cl'],
        'C_c': h2_data['C_c'],
        'D_c': h2_data['D_c'],
        'C_y': h2_data['C_y']
    }
}

# Display eigenvalues for robust controller
eigvals = np.linalg.eigvals(robust_data['A_cl'])
print("Robust controller eigenvalues (abs):")
print(abs(eigvals))

# Compute worst-case covariances
covariances = {}
for ctrl_type, matrices in controllers.items():
    # Full-order closed-loop system
    A_cl_full = matrices['A_cl']  # 22x22
    B_cl_full = matrices['B_cl'][:,0:2]  # 22x2

    # Redefine output matrices
    C_y = matrices['C_y']  # 11x11
    C_c = matrices['C_c']  # 1x11
    D_c = matrices['D_c']  # 1x11
    n_x,n_w = B_cl_full.shape
    outputs_to_attack = {'omega': (np.eye(n_x)[0:1, :],np.zeros((1, n_w))),
                         'h': (np.eye(n_x)[1:2, :],np.zeros((1, n_w))),
                         'phi': (np.eye(n_x)[3:4, :],np.zeros((1, n_w))),
                         'u': (np.hstack([D_c @ C_y, C_c]),np.zeros((1, n_w)))}
    covariances[ctrl_type] = {}

    # Loop over outputs to attack
    for output_name, (C_cl, D_cl) in outputs_to_attack.items():
        C_cl_full = np.block([
            [C_cl],                      # u = C_c x_c + D_c y
            [np.zeros((2, n_x))]                 # w1, w2 (via D_cl)
        ])  # 5x22
        D_cl_full = np.block([
            [D_cl],  # u
            [np.eye(n_w)]          # w1, w2
        ])  # 5x2

        # Balanced truncation
        try:
            # Compute discrete-time Gramians
            P = solve_discrete_lyapunov(A_cl_full, B_cl_full @ B_cl_full.T + (1e-8)*np.eye(n_x)) + (1e-8)*np.eye(n_x)
            Q = solve_discrete_lyapunov(A_cl_full.T, C_cl_full.T @ C_cl_full + (1e-8)*np.eye(n_x)) + (1e-8)*np.eye(n_x)

            # Check positive definiteness
            print(np.linalg.eigvals(P) <= 0)
            print(np.any(np.linalg.eigvals(P) <= 0))
            print(np.linalg.eigvals(Q) <= 0)
            print(np.any(np.linalg.eigvals(Q) <= 0))
            print(np.linalg.eigvals(P))
            print(np.linalg.eigvals(Q))
            if np.any(np.linalg.eigvals(P) <= 0) or np.any(np.linalg.eigvals(Q) <= 0):
                raise ValueError("Gramians are not positive definite.")

            # Cholesky decomposition
            try:
                L = np.linalg.cholesky(P)
                M = np.linalg.cholesky(Q)
            except:
                E,V = np.linalg.eig(P)
                print(E.shape)
                print(V.shape)
                L = V@np.diag(np.sqrt(E))
                E,V = np.linalg.eig(Q)
                M = V@np.diag(np.sqrt(E))



            # SVD of L^T M
            U, Sigma, Vt = np.linalg.svd(L.T @ M)
            V = Vt.T

            # Balancing transformation
            Sigma_sqrt = np.diag(np.sqrt(Sigma))
            Sigma_sqrt_inv = np.diag(1.0 / np.sqrt(Sigma + 1e-10))
            T = L @ U @ Sigma_sqrt_inv
            T_inv = Sigma_sqrt_inv @ V.T @ M.T

            # Transform closed-loop system
            A_cl_bal = T_inv @ A_cl_full @ T
            B_cl_bal = T_inv @ B_cl_full
            C_cl_bal = C_cl_full @ T
            D_cl_bal = D_cl_full

            # Truncate to n_r states
            n_r = np.sum(np.diag(Sigma) > 1e-6)
            A_cl_red = A_cl_bal[:n_r, :n_r]
            B_cl_red = B_cl_bal[:n_r, :]
            C_cl_red = C_cl_bal[:, :n_r]
            D_cl_red = D_cl_bal

            # Store Hankel singular values
            hankel_sv = Sigma
                                        
            # Stability check
            eigvals_red = np.linalg.eigvals(A_cl_red)
            if any(abs(eigvals_red) >= 1 - 1e-6):
                print(f"Warning: {ctrl_type.capitalize()} reduced-order closed-loop system may be unstable")
            else:
                print(f"{ctrl_type.capitalize()} reduced-order closed-loop system is stable")

        except Exception as e:
            print(f"{ctrl_type.capitalize()} balanced truncation error: {e}")
            print(f"Using full-order system for {ctrl_type} controller.")
            A_cl_red = A_cl_full
            B_cl_red = B_cl_full
            C_cl_red = C_cl_full
            D_cl_red = D_cl_full
            n_r = A_cl_red.shape[0]

        # Use reduced-order system for SDP
        A_cl = A_cl_red  # n_r x n_r
        B_cl = B_cl_red  # n_r x n_w
        C_cl = C_cl_red  # n_v x n_r
        D_cl = D_cl_red  # n_v x n_w

        # Define moment matrix Sigma
        Sigma_chi_chi = cp.Variable((n_r, n_r), symmetric=True)
        Sigma_chi_w = cp.Variable((n_r, n_w))
        Sigma_chi_hatw = cp.Variable((n_r, n_w))
        Sigma_ww = cp.Variable((n_w, n_w), symmetric=True)
        Sigma_w_hatw = cp.Variable((n_w, n_w))
        
        Sigma = cp.bmat([
            [Sigma_chi_chi, Sigma_chi_w, Sigma_chi_hatw],
            [Sigma_chi_w.T, Sigma_ww, Sigma_w_hatw],
            [Sigma_chi_hatw.T, Sigma_w_hatw.T, Sigma_nom]
        ])

        # Objective: maximize trace(C_cl Sigma_chi_chi C_cl.T + ...)
        cost = cp.trace(C_cl @ Sigma_chi_chi @ C_cl.T + C_cl @ Sigma_chi_w @ D_cl.T +
                        D_cl @ Sigma_chi_w.T @ C_cl.T + D_cl @ Sigma_ww @ D_cl.T)

        # Constraints
        constraints = [
            # Discrete-time Lyapunov equation
            Sigma_chi_chi == A_cl @ Sigma_chi_chi @ A_cl.T + A_cl @ Sigma_chi_w @ B_cl.T +
                            B_cl @ Sigma_chi_w.T @ A_cl.T + B_cl @ Sigma_ww @ B_cl.T + 0.0001*np.eye(n_r),
            # Wasserstein constraint
            cp.trace(Sigma_ww + Sigma_nom - 2 * Sigma_w_hatw) <= gamma**2,
            # Positive semi-definiteness
            Sigma >> 0
        ]

        # Solve SDP
        problem = cp.Problem(cp.Maximize(cost), constraints)
        success = False
        try:
            problem.solve(solver=cp.MOSEK, verbose=True, mosek_params={
                'MSK_DPAR_INTPNT_CO_TOL_PFEAS': 1e-8,
                'MSK_DPAR_INTPNT_CO_TOL_DFEAS': 1e-8,
                'MSK_DPAR_INTPNT_CO_TOL_REL_GAP': 1e-8
            })
            if problem.status == cp.OPTIMAL:
                success = True
                print(f"{ctrl_type.capitalize()} controller: MOSEK solved successfully")
            else:
                print(f"{ctrl_type.capitalize()} controller: MOSEK status {problem.status}")
        except Exception as e:
            print(f"{ctrl_type.capitalize()} controller: MOSEK error: {e}")

        if not success:
            print(f"{ctrl_type.capitalize()} controller: Trying SCS...")
            try:
                problem.solve(solver=cp.SCS, verbose=True, eps=1e-6, max_iters=100000)
                if problem.status in [cp.OPTIMAL, cp.OPTIMAL_INACCURATE]:
                    success = True
                    print(f"{ctrl_type.capitalize()} controller: SCS status {problem.status}")
                else:
                    print(f"{ctrl_type.capitalize()} controller: SCS failed with status {problem.status}")
            except Exception as e:
                print(f"{ctrl_type.capitalize()} controller: SCS error: {e}")

        if not success:
            print(f"{ctrl_type.capitalize()} controller: Optimization failed")
            covariances[ctrl_type] = None
            continue

    

        # Calculate output covariances
        Sigma_vv_val = (C_cl @ Sigma_chi_chi @ C_cl.T + C_cl @ Sigma_chi_w @ D_cl.T +
                        D_cl @ Sigma_chi_w.T @ C_cl.T + D_cl @ Sigma_ww @ D_cl.T).value

        # Store results
        covariances[ctrl_type][output_name] = {"variance": Sigma_vv_val[0,0],
                                                "Sigma_chi_chi": Sigma_chi_chi.value,
                                                "Sigma_chi_w": Sigma_chi_w.value,
                                                "Sigma_ww": Sigma_ww.value,
                                                "Sigma_w_hatw": Sigma_w_hatw.value}


print("Covariances:")
for ctrl_type, cov in covariances.items():
    print(f"{ctrl_type.capitalize()} controller:")
    for output_name, cov_data in cov.items():
        print(f"  {output_name}: {cov_data['variance']}")
    print()

print("Disturbance covariance matrices:")
for ctrl_type, cov in covariances.items():
    print(f"{ctrl_type.capitalize()} controller:")
    for output_name, cov_data in cov.items():
        print(f"  {output_name}:")
        print(f"    Sigma_ww:\n{cov_data['Sigma_ww']}")
    print()

# Save results
np.save('worst_case_covariances.npy', covariances)
print("Worst-case covariances saved to 'worst_case_covariances.npy'")