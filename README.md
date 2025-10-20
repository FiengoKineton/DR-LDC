# Distributionally-Robust Output-Feedback Control  
### (Baseline Monte Carlo H₂ vs. DRO-LMI)

This repository implements two pipelines for discrete-time output-feedback controller synthesis and evaluation:

1. **Baseline** — stochastic H₂ optimization via Monte Carlo simulation.  
2. **DRO-LMI** — distributionally-robust control synthesis via convex LMIs and controller recovery.

Both produce controllers for the same plant model and evaluate closed-loop performance.

---

## 🧭 Repository Structure

```
.
├── main.py                     # CLI entry point
├── problem___baseline.py       # Baseline H₂ optimization (Monte Carlo + L-BFGS-B)
├── problem___dro_lmi.py        # Distributionally-Robust LMI formulation & solver
├── problem___parameters.yaml   # Central configuration (dimensions, solver, ambiguity, etc.)
│
├── utilis___systems.py         # Plant & controller data structures
├── utilis___matrices.py        # Closed-loop composition and controller recovery
├── utilis___simulate.py        # Simulation and plotting utilities
│
└── out/                        # Folder created automatically for JSON/NPZ artifacts
```

---

## ⚙️ How It Works

### Overview
The repository supports two synthesis modes:

| Mode | Description | Run Command |
|------|--------------|--------------|
| **Baseline** | Minimizes expected output energy under Gaussian disturbance using Monte Carlo rollouts. | `python main.py --base` |
| **DRO-LMI** | Solves a distributionally-robust H₂ problem using LMIs and recovers a controller. | `python main.py --lmi` |

---

## 🧩 System Diagram

```mermaid
flowchart TD
    P1[problem___parameters.yaml] --> A1[utilis___systems.py<br>Plant & Controller classes]
    A1 --> A2[utilis___matrices.py<br>Compose (A,B,C,D)]
    A2 --> A3[problem___baseline.py<br>Monte Carlo Optimization]
    A2 --> A4[problem___dro_lmi.py<br>DRO-LMI Synthesis]
    A3 --> S1[utilis___simulate.py<br>Closed-Loop Simulation]
    A4 --> R1[Recover Controller]
    R1 --> S1
    S1 --> OUT[out/artifacts/ JSON + NPZ]
```

---

## 🧠 Baseline Monte-Carlo H₂ Method

**Goal:**  
Minimize  
\[
J = \mathbb{E}\big[ \|z_t\|^2 \big]
\]
for the closed-loop system under white noise \( w_t \).

**Procedure:**
1. Randomly initialize controller matrices \( A_c,B_c,C_c,D_c \).  
2. Compose the closed-loop dynamics \((\mathcal{A}, \mathcal{B}, \mathcal{C}, \mathcal{D})\).  
3. Simulate for multiple trajectories and compute empirical H₂ cost.  
4. Optimize parameters with **L-BFGS-B**.  
5. Penalize instability and project back into stable region if necessary.

**Artifacts:**
- JSON: cost, plant/controller matrices, stability info.  
- NPZ: trajectories and numerical data.  
- Plots: state, control input, output evolution.

---

## 🧮 DRO-LMI Method

**Goal:**  
Design a controller minimizing the worst-case H₂ cost under covariance uncertainty within a Wasserstein-2 ball.

**Steps:**
1. Construct the **block matrices**  
   \[
   \mathbb{P}, \mathbb{A}, \mathbb{B}, \mathbb{C}, \mathbb{D}
   \]
   using the plant and ambiguity model (correlated or independent).
2. Formulate LMIs with decision variables \( X, Y, Q, \lambda, K, L, M, N \).  
3. Solve using **MOSEK** or **SCS** (configured in YAML).  
4. Recover a realizable controller \((A_c,B_c,C_c,D_c)\) via structured least squares.  
5. Simulate and store results as JSON/NPZ artifacts.

---

## 📊 Configuration (problem___parameters.yaml)

Key fields:

| Field | Description |
|--------|--------------|
| `params.dimensions` | System sizes (`nx`, `nw`, `nu`, `ny`). |
| `params.outputs.mode` | Output configuration for cost definition. |
| `params.ambiguity` | Wasserstein radius `gamma` and regularization `alpha`. |
| `params.solver` | Choose between `MOSEK` or `SCS`. |
| `params.plant` | Randomization seed and scaling for system generation. |

---

## ▶️ Running Experiments

### Baseline Optimization
```bash
python main.py --base
```
Artifacts stored in `out/artifacts/baseline/`.

### DRO-LMI Synthesis
```bash
python main.py --lmi
```
Artifacts stored in `out/artifacts/lmi/`.

---

## 📦 Outputs

Each run generates:
- `*.json` → parameters, matrices, metadata  
- `*.npz` → trajectories and arrays  
- `.png` plots under the same directory  

---

## 🧠 Dependencies

- Python ≥ 3.10  
- NumPy, SciPy, Matplotlib  
- CVXPY with a solver (MOSEK or SCS)  
- PyYAML for configuration  

---

## 🧾 Citation

If used in research or coursework, please acknowledge this repository as:

> Hybrid Baseline & DRO-LMI Pipeline for Robust Output-Feedback Control (2025)
