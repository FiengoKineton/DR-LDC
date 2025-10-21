# Wind Turbine Blade Control: H2 and Robust Control Synthesis

This repository contains Python scripts for synthesizing distributionally robust and H2-optimal controllers for wind turbine blade control, incorporating a turbulence disturbance model. The project evaluates controller performance under wind disturbances (\( v_x, v_z \)) with 8% turbulence intensity (\( \sigma = 0.96 \, \text{m/s} \)) and generates frequency response plots for analysis. The system is modeled with an 7-state plant (rotor speed \( \omega \), flapwise displacement \( h \), flapwise velocity \( \dot{h} \), torsional angle \( \phi \), torsional rate \( \dot{\phi} \), pitch angle \( \beta \), pitch rate \( \dot{\beta} \)).

## Project Overview

The project focuses on:
- **H2 Controller Synthesis**: Optimizes a controller to minimize the H2 norm of the closed-loop system, with RMS constraints (\( \mathbb{E}[h^2] \leq 4 \, \text{m}^2 \), \( \mathbb{E}[\phi^2] \leq 10^{-4} \, \text{rad}^2 \)).
- **Distributionally Robust Controller**: Accounts for worst-case disturbance covariances, computed via `obtainWorstCaseUncertainty.py`.
- **Frequency Response Analysis**: Plots maximum singular value responses from wind disturbances to outputs (\( \omega, h, \phi, u \)) using Matplotlib’s PGF backend for LaTeX integration.

Key parameters:
- Rotor inertia: \( J = 3.5 \times 10^7 \, \text{kg·m}^2 \)
- Blade mass: \( m = 1.7 \times 10^4 \, \text{kg} \)
- Torsional frequency: \( \omega_t = 12.56 \, \text{rad/s} \)
- Flapwise frequency: \( \omega_f = 6.28 \, \text{rad/s} \)
- Sampling time: \( \Delta t = 0.05 \, \text{s} \)

## Dependencies

- **Python**: 3.8+
- **Packages**:
  - `numpy`: Matrix computations and system dynamics.
  - `scipy`: Discretization and Riccati equation solvers.
  - `cvxpy`: LMI optimization for H2 synthesis.
  - `matplotlib`: Plotting with PGF backend for LaTeX-compatible outputs.
  - Optional: `mosek` for faster LMI solving (requires license).
- **LaTeX**: TeX Live or similar for compiling PGF plots (e.g., `pdflatex`).
- Install dependencies:
  ```bash
  pip install numpy scipy cvxpy matplotlib
  ```

## Directory Structure

```
wind-turbine-control/
├── H2ControllerSynthesis.py        # H2 controller synthesis (LMI-based)
├── PlotSVDResponse.py              # Generates frequency response plots (.pgf, .png)
├── controller_data_h2.npy          # H2 controller and closed-loop matrices
├── controller_data_robust.npy      # Robust controller and closed-loop matrices
├── obtainWorstCaseUncertainty.py   # Computes worst-case disturbance covariance
├── generatePGFPlots.py             # Generates time-domain plots from simulations
├── plots/                          # Output folder for .pgf and .png plots
│   ├── omega_svd.pgf              # Frequency response for omega
│   ├── h_svd.pgf                  # Frequency response for h
│   ├── phi_svd.pgf                # Frequency response for phi
│   ├── u_svd.pgf                  # Frequency response for u
│   ├── omega_svd.png              # PNG version for quick inspection
│   ├── h_svd.png                  # PNG version
│   ├── phi_svd.png                # PNG version
│   ├── u_svd.png                  # PNG version
│   ├── h2_h_full.pgf              # Time-domain plot for h (H2)
│   ├── h2_phi_full.pgf            # Time-domain plot for phi (H2)
│   └── ...                        # Other plots from generatePGFPlots.py
├── README.md                       # This file
```

## Setup and Usage

1. **Clone the Repository**:
   ```bash
   git clone https://github.com/your-username/wind-turbine-control.git
   cd wind-turbine-control
   ```

2. **Run Controller Synthesis**:
   - Generate distributionally robust controller:
     ```bash
     python DRControllerSynthesis.py
     ```
     Outputs `controller_data_robust.npy`.
   - Generate distributionally robust controller with scaled outputs:
     ```bash
     python DRControllerSynthesis_scaled.py
     ```
     Outputs `controller_data_robust_scaled.npy`.
   - Generate H2 controller:
     ```bash
     python H2ControllerSynthesis.py
     ```
     Outputs `controller_data_h2.npy`.
   - Generate H2 controller with scaled outputs:
     ```bash
     python H2ControllerSynthesis_scaled.py
     ```
     Outputs `controller_data_h2_scaled.npy`.
     
3. **Generate worst-case uncertainties**:
     ```bash
     python obtainWorstCaseUncertainty.py
     ```
     Outputs `worst_case_covariances.npy`.

5. **Generate Frequency Response Plots**:
   - Plot maximum singular value responses from wind disturbances (\( v_x, v_z \)) to outputs (\( \omega, h, \phi, u \)):
     ```bash
     python PlotSVDResponse.py
     ```
     Outputs `.pgf` and `.png` files in `plots/` (e.g., `plots/omega_svd.pgf`, `plots/omega_svd.png`).

6. **Integrate with LaTeX**:
   - Include `.pgf` plots in your LaTeX document (e.g., `report.tex`):
     ```latex
     \documentclass{article}
     \usepackage{pgfplots}
     \pgfplotsset{compat=1.18}
     \begin{document}
     \begin{figure}[h]
         \centering
         \begin{tikzpicture}
             \begin{axis}[
                 width=0.45\textwidth,
                 xlabel={Frequency (rad/s)},
                 ylabel={Max Singular Value to $\omega$ (dB)},
                 xmode=log,
                 grid=major,
                 grid style={dashed,gray!30},
                 legend pos=north east
             ]
                 \addplot graphics {plots/omega_svd};
             \end{axis}
         \end{tikzpicture}
         \caption{Frequency response of maximum singular value from wind disturbance to rotor speed ($\omega$).}
         \label{fig:omega_svd}
     \end{figure}
     % Add similar figures for h_svd, phi_svd, u_svd
     \end{document}
     ```
   - Compile with `pdflatex`:
     ```bash
     pdflatex report.tex
     ```

## Key Outputs

- **Controller Data**:
  - `controller_data_h2.npy`: H2 controller matrices (\( A_c, B_c, C_c, D_c \)) and closed-loop system (\( A_{\text{cl}}, B_{\text{cl}}, C_{\text{cl}}, D_{\text{cl}} \)).
  - `controller_data_robust.npy`: Robust controller matrices and closed-loop system.
  - `worst_case_covariances.npy`: Worst-case disturbance covariance (\( \Sigma_{ww} \)).

- **Plots**:
  - Frequency-domain: `plots/omega_svd.pgf`, `plots/h_svd.pgf`, `plots/phi_svd.pgf`, `plots/u_svd.pgf` (and `.png` versions).

## License

This project is licensed under the MIT License. See `LICENSE` for details.
