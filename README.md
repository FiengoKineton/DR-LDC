# README.md

## Interconnection Diagram
simulate_pe_openloop.py → define_matrices.py (Direct Data-Driven Estimation)
│
▼
Plant, Controller₀
│
▼
compose.py → dro_lmi.py → recover_controller.py
│ │
▼ ▼
simulate_closed_loop.py ← main_dro_pipeline.py ← run.py
│
▼
main.py

---

## List of Files

1. `ambiguity.py`  
2. `systems.py`  
3. `define_matrices.py`  
4. `compose.py`  
5. `dro_lmi.py`  
6. `recover_controller.py`  
7. `optim_problem.py`  
8. `simulate_closed_loop.py`  
9. `simulate_pe_openloop.py`  
10. `run.py`  
11. `main_dro_pipeline.py`  
12. `main.py`


This pipeline performs **data-driven distributionally robust control (DRO-LDC)** synthesis and validation.
It links data generation, system identification, convex LMI synthesis, controller recovery, and closed-loop evaluation.

---

## Mathematical and Algorithmic Overview

### 1. `ambiguity.py`
Implements the **Wasserstein ambiguity model** defining the uncertainty ball:
\[
\mathbb{B}_\gamma(\mathbb{P}_{\Sigma_{\text{nom}}}) = \{ \mathbb{Q} \mid W_2(\mathbb{Q}, \mathbb{P}_{\Sigma_{\text{nom}}}) \le \gamma \}.
\]
For tractability, the ambiguity set is replaced by an inflated covariance:
\[
\Sigma_{\text{eff}} = \Sigma_{\text{nom}} + \alpha I,
\]
where α depends on the Wasserstein radius γ.  
This upper-bounds the worst-case second moment within the ball.

---

### 2. `systems.py`
Defines two main data structures:
- **Plant**: stores system matrices \((A, B_w, B_u, C_z, D_{zw}, D_{zu}, C_y, D_{yw})\)
  describing
  \[
  \begin{aligned}
  x^+ &= Ax + B_uu + B_ww, \\
  z &= C_zx + D_{zu}u + D_{zw}w, \\
  y &= C_yx + D_{yw}w.
  \end{aligned}
  \]
- **Controller**: stores feedback matrices \((A_c, B_c, C_c, D_c)\)
  in
  \[
  x_c^+ = A_cx_c + B_cy, \quad u = C_cx_c + D_cy.
  \]
Both include dimension getters for use in simulation and optimization.

---

### 3. `define_matrices.py`
Responsible for **creating system matrices**, either synthetically or from experimental data.

#### Direct Data-Driven Estimation
Given CSV data with columns `[x, u, y, z]`, it estimates:
\[
X^+ = [A \; B_u] 
\begin{bmatrix} X \\ U \end{bmatrix}, \quad
[A \; B_u] = X^+ D^\top (D D^\top)^{-1}, \quad D = [X; U].
\]

Residuals \(R = X^+ - A X - B_u U\) define \(B_w\):
\[
\Sigma_R = \frac{1}{T} R R^\top, \quad
B_w = V \operatorname{diag}(\sqrt{\lambda_i}),
\]
where \(V\) and \(\lambda_i\) are eigenvectors/values of \(\Sigma_R\).

Static outputs:
- \(C_y, C_z\): identity/selection matrices for measured/performance outputs.
- \(D_{zu}, D_{zw}, D_{yw}\): small or zero random matrices ensuring full-rank mappings.

---

### 4. `compose.py`
Constructs **closed-loop augmented matrices**:
\[
\mathcal{A}, \mathcal{B}, \mathcal{C}, \mathcal{D}
\]
from plant and controller components:
\[
\mathcal{A} = 
\begin{bmatrix}
A + B_uD_cC_y & B_uC_c \\
B_cC_y & A_c
\end{bmatrix}, \quad
\mathcal{B} =
\begin{bmatrix}
B_w + B_uD_cD_{yw} \\ 
B_cD_{yw}
\end{bmatrix}.
\]
These are the matrices used in LMI-based control synthesis.

---

### 5. `dro_lmi.py`
Solves the **distributionally robust LMI synthesis**:
\[
\begin{aligned}
&\min_{P,Q,\lambda} \; \text{tr}(Q \Sigma_{\text{eff}}) + \lambda \gamma^2 \\
\text{s.t.} &\quad
\begin{bmatrix}
A^\top P A - P + C^\top C & A^\top P B + C^\top D \\
B^\top P A + D^\top C & B^\top P B + D^\top D - Q
\end{bmatrix} \prec 0, \\
&\quad P \succ 0, \; \lambda \ge 0.
\end{aligned}
\]

This problem arises from convexifying the robust \(H_2\) control objective:
\[
J = \sup_{w\in\mathbb{B}_\gamma} \mathbb{E}[\|z\|^2],
\]
under Wasserstein uncertainty, as developed by Scherer & Yan (2025):contentReference[oaicite:0]{index=0}.

Outputs are serialized (`artifacts_lmi/*.json`).

---

### 6. `recover_controller.py`
Reconstructs \(A_c, B_c, C_c, D_c\) from the LMI solution using block decomposition.

Given closed-loop solution matrices \(\mathcal{A}, \mathcal{B}, \mathcal{C}, \mathcal{D}\) and transformation \(P = T^\top T\), it recovers controller parameters by solving:
\[
\mathcal{A} = T^{-1} A_{cl} T, \quad
A_{cl} = 
\begin{bmatrix}
A + B_u D_c C_y & B_u C_c \\
B_c C_y & A_c
\end{bmatrix}.
\]
Least-squares regression yields the controller blocks ensuring matching dimensions.

---

### 7. `optim_problem.py`
Implements a **numerical performance optimizer** over controller matrices via simulation.
Objective:
\[
J(A_c,B_c,C_c,D_c) = \mathbb{E}[\|z_t\|^2].
\]
It samples controllers, simulates their performance, and selects the one minimizing the empirical cost.

---

### 8. `simulate_closed_loop.py`
Simulates the **interconnected plant-controller** system:
\[
\begin{aligned}
x_{t+1} &= A x_t + B_u u_t + B_w w_t, \\
x_{c,t+1} &= A_c x_{c,t} + B_c y_t, \\
u_t &= C_c x_{c,t} + D_c y_t, \\
y_t &= C_y x_t + D_{yw} w_t, \\
z_t &= C_z x_t + D_{zu} u_t + D_{zw} w_t.
\end{aligned}
\]
It logs trajectories, checks stability (\(\rho(\mathcal{A}) < 1\)), and can visualize outputs.

---

### 9. `simulate_pe_openloop.py`
Generates persistently exciting (PE) data for identification.  
Inputs \(U_t\) are PRBS or multisine:
\[
u_t = \sum_i a_i \sin(\omega_i t + \phi_i).
\]
Simulates
\[
x_{t+1}=Ax_t+B_uu_t+B_ww_t,
\]
stores `(x,u,y,z)` trajectories, and exports:
- CSV with signals,
- NPZ with ground-truth matrices.

Also includes `evaluate_from_path()` to estimate matrices from the saved CSV and compare to the truth.

---

### 10. `run.py`
Executes the DRO control synthesis:
1. Load plant (`define_matrices.get_system()`).
2. Define ambiguity set (`Ambiguity`).
3. Call LMI solver (`dro_lmi.solve()`).
4. Recover controller (`recover_controller.py`).
5. Evaluate closed-loop stability and cost.

It thus provides a **one-shot synthesis routine**.

---

### 11. `main_dro_pipeline.py`
High-level orchestrator integrating all steps:
1. Data-driven system reconstruction.
2. DRO LMI synthesis.
3. Controller recovery.
4. Closed-loop simulation.
5. Artifact saving and plotting.

It functions as the **master experiment pipeline** coordinating all subsystems.

---

### 12. `main.py`
Entry point for ad-hoc tests — can call the pipeline or run components in isolation for debugging.

---

## Mathematical Interdependencies Summary

| Module | Mathematical Role | Inputs | Outputs | Depends On |
|--------|--------------------|---------|----------|-------------|
| `simulate_pe_openloop.py` | Generate PE dataset | A,Bu,Bw,noise | CSV, NPZ | — |
| `define_matrices.py` | Estimate system matrices | CSV | A,Bu,Bw,C,D | numpy |
| `compose.py` | Build closed-loop matrices | A,Bu,Bw,C,D,controller | 𝒜,𝒝,𝒞,𝒟 | systems |
| `dro_lmi.py` | Solve convex DRO-LMI | 𝒜,𝒝,𝒞,𝒟,Σeff | Ā,B̄,C̄,D̄,P̄ | cvxpy |
| `recover_controller.py` | Extract Ac,Bc,Cc,Dc | Ā,B̄,C̄,D̄,P̄ | Controller | numpy |
| `simulate_closed_loop.py` | Validate controller | Plant, Controller | performance metrics | compose |
| `run.py` | Execute synthesis | Plant | Results | all above |
| `main_dro_pipeline.py` | Full automation | — | final artifacts | all above |

---

## Algorithmic Flow Summary

1. **Data generation** (PE simulation)
   - Ensure \(D D^\top\) full rank (persistency of excitation).
   - Compute regression matrices.
2. **Identification**
   - Estimate \(A,B_u,B_w\) by least-squares + residual eigen-decomposition.
3. **DRO Synthesis**
   - Formulate convex LMI minimizing worst-case cost.
   - Solve for feasible \(P, Q, \lambda\).
4. **Controller Recovery**
   - Factorize transformation \(P=T^\top T\).
   - Recover controller blocks.
5. **Simulation**
   - Integrate plant + controller forward in time.
   - Compute expected cost and verify stability.

---

## References
- **Scherer & Yan (2025)** – *Distributionally Robust LMI Synthesis for LTI Systems*:contentReference[oaicite:1]{index=1}  
- **Taskesen et al. (2023)** – *Distributionally Robust Linear Quadratic Control*  
- **Villani (2008)** – *Optimal Transport: Old and New*

---
