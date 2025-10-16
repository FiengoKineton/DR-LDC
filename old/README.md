# DRC Demo: Distributionally Robust LMI Synthesis (State-Feedback)

This is a minimal, multi-file Python project demonstrating a distributionally robust controller synthesis
using an LMI/SDP with a Wasserstein ambiguity set (state-feedback specialization).

## Structure
- `system.py` – create a demo plant
- `dro.py` – simple Wasserstein radius helper
- `lmi_synthesis.py` – convex surrogate of the moment-SDP; solve for K
- `simulate.py` – Monte Carlo evaluation
- `experiment.py` – end-to-end runner
- `utils.py` – helpers (LQR, stability)

## Run
```bash
pip install -r requirements.txt
python -m drc_demo.experiment
