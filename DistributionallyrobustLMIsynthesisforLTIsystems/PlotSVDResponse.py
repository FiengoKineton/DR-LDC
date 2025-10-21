import numpy as np
import matplotlib
matplotlib.use('pgf')
import matplotlib.pyplot as plt
import os
from scipy.signal import ss2tf, freqz_zpk

# Create plots folder if it doesn't exist
os.makedirs('plots', exist_ok=True)

# Configure PGF backend
plt.rcParams.update({
    "pgf.texsystem": "pdflatex",
    "font.family": "serif",
    "text.usetex": True,
    "pgf.rcfonts": False,
    "font.size": 10,
    "axes.labelsize": 10,
    "xtick.labelsize": 8,
    "ytick.labelsize": 8,
    "legend.fontsize": 8
})

# Load controller data
h2_data = np.load('controller_data_h2.npy', allow_pickle=True).item()
robust_data = np.load('controller_data_robust.npy', allow_pickle=True).item()
h2_scaled_data = np.load('controller_data_h2_scaled.npy', allow_pickle=True).item()
robust_scaled_data = np.load('controller_data_robust_scaled.npy', allow_pickle=True).item()

# System parameters
dt = 0.05  # Sampling time (s)
omega_f = 6.28  # Flapwise frequency (rad/s)
omega_t = 12.56  # Torsional frequency (rad/s)
n_x = h2_data['A_cl'].shape[0]  # Closed-loop states (22)
n_w = h2_data['B_cl'].shape[1]  # Disturbance inputs (5)

# Define output matrices (outputs_to_attack)
outputs_to_attack = {
    'omega': (np.eye(n_x)[0:1, :], np.zeros((1, n_w))),
    'h': (np.eye(n_x)[1:2, :], np.zeros((1, n_w))),
    'phi': (np.eye(n_x)[3:4, :], np.zeros((1, n_w))),
    'u': (
        np.hstack([h2_data['D_c'] @ h2_data['C_y'], h2_data['C_c']]),
        np.zeros((1, n_w))
    )
}

# Update u for robust controller
outputs_to_attack_robust = outputs_to_attack.copy()
outputs_to_attack_robust['u'] = (
    np.hstack([robust_data['D_c'] @ robust_data['C_y'], robust_data['C_c']]),
    np.zeros((1, n_w))
)

# Create output matrices for scaled controllers
outputs_to_attack_h2_scaled = outputs_to_attack.copy()
outputs_to_attack_h2_scaled['u'] = (
    np.hstack([h2_scaled_data['D_c'] @ h2_scaled_data['C_y'], h2_scaled_data['C_c']]),
    np.zeros((1, n_w))
)

outputs_to_attack_robust_scaled = outputs_to_attack.copy()
outputs_to_attack_robust_scaled['u'] = (
    np.hstack([robust_scaled_data['D_c'] @ robust_scaled_data['C_y'], robust_scaled_data['C_c']]),
    np.zeros((1, n_w))
)

# Output labels and units
outputs = ['omega', 'h', 'phi', 'u']
output_units = ['rad/s', 'm', 'rad', 'rad']

# Frequency range (rad/s)
freqs = np.logspace(-4, 0.9, 500)  # 0.01 to 10 rad/s
omega = 2 * np.pi * freqs
z = np.exp(1j * omega * dt)

# Function to compute max singular value response
def compute_svd_response(A_cl, B_cl, C_cl, D_cl, freqs, dt):
    n_states = A_cl.shape[0]
    B_w = B_cl[:, :2]  # Disturbance inputs [v_x, v_z]
    svd_vals = []
    
    for w in 2 * np.pi * freqs:
        z = np.exp(1j * w * dt)
        G = C_cl @ np.linalg.inv(z * np.eye(n_states) - A_cl) @ B_w + D_cl[:, :2]
        svd = np.linalg.svd(G, compute_uv=False)[0]  # Maximum singular value
        svd_vals.append(svd)
    
    return np.array(svd_vals)

# Plot responses
for output, unit in zip(outputs, output_units):
    # Get output matrices
    C_cl_h2, D_cl_h2 = outputs_to_attack[output]
    C_cl_robust, D_cl_robust = outputs_to_attack_robust[output]
    C_cl_h2_scaled, D_cl_h2_scaled = outputs_to_attack_h2_scaled[output]
    C_cl_robust_scaled, D_cl_robust_scaled = outputs_to_attack_robust_scaled[output]
    
    # Compute SVD responses
    h2_svd = compute_svd_response(
        h2_data['A_cl'], h2_data['B_cl'], C_cl_h2, D_cl_h2, freqs, dt
    )
    robust_svd = compute_svd_response(
        robust_data['A_cl'], robust_data['B_cl'], C_cl_robust, D_cl_robust, freqs, dt
    )
    h2_scaled_svd = compute_svd_response(
        h2_scaled_data['A_cl'], h2_scaled_data['B_cl'], C_cl_h2_scaled, D_cl_h2_scaled, freqs, dt
    )
    robust_scaled_svd = compute_svd_response(
        robust_scaled_data['A_cl'], robust_scaled_data['B_cl'], C_cl_robust_scaled, D_cl_robust_scaled, freqs, dt
    )
    
    # Convert to dB
    h2_db = 20 * np.log10(np.maximum(h2_svd, 1e-10))  # Avoid log(0)
    robust_db = 20 * np.log10(np.maximum(robust_svd, 1e-10))
    h2_scaled_db = 20 * np.log10(np.maximum(h2_scaled_svd, 1e-10))
    robust_scaled_db = 20 * np.log10(np.maximum(robust_scaled_svd, 1e-10))
    
    # Create plot
    plt.figure(figsize=(4, 3))
    plt.semilogx(freqs, h2_db, label='H2 Controller', color='blue', linewidth=1)
    plt.semilogx(freqs, robust_db, label='Robust Controller', color='red', linestyle='--', linewidth=1)
    plt.semilogx(freqs, h2_scaled_db, label='H2 Scaled Controller', color='green', linestyle='-.', linewidth=1)
    plt.semilogx(freqs, robust_scaled_db, label='Robust Scaled Controller', color='black', linestyle=':', linewidth=1)
    
    # Add vertical lines for omega_f and omega_t
    plt.axvline(x=omega_f, color='gray', linestyle=':', alpha=0.5)
    plt.axvline(x=omega_t, color='gray', linestyle=':', alpha=0.5)
    
    # Formatting
    plt.xlabel('Frequency (rad/s)')
    plt.ylabel(f'Disturbance (dB) to {output} ({unit})')
    plt.grid(True, which='both', linestyle='--', alpha=0.7)
    plt.legend(loc='best')
    plt.tight_layout()
    
    # Save as PGF
    plt.savefig(f'plots/{output}_svd.pgf', format='pgf', bbox_inches='tight')
    
    # Save as PNG (switch to default backend temporarily)
    matplotlib.use('Agg')
    plt.savefig(f'plots/{output}_svd.png', format='png', dpi=300, bbox_inches='tight')
    matplotlib.use('pgf')
    plt.close()

print("Plots saved in 'plots' folder: omega_svd.pgf, h_svd.pgf, phi_svd.pgf, u_svd.pgf")
print("PNG versions saved: omega_svd.png, h_svd.png, phi_svd.png, u_svd.png")