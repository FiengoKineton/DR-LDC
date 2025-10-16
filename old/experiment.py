import numpy as np
import matplotlib.pyplot as plt

from system import make_demo_plant
from utils import lqr, is_stable
from dro import wasserstein_radius_chernoff
from lmi_synthesis import synthesize_state_feedback
from simulate import simulate_closed_loop

def main():
    rng = 7
    n, m, zdim = 4, 2, 4
    plant = make_demo_plant(n=n, m=m, z_dim=zdim, rng=rng)
    A, B, Cz, Dzu = plant.A, plant.B, plant.Cz, plant.Dzu

    Sigma_nom = 0.05 * np.eye(n)
    N = 1000
    gamma = wasserstein_radius_chernoff(dim=n, N=N, delta=0.1, scale=0.5)
    print(f"[Info] gamma ≈ {gamma:.4f} (dim={n}, N={N})")

    # Nominal LQR baseline
    Q = Cz.T @ Cz
    R = 0.1 * np.eye(m)
    K_lqr = lqr(A, B, Q, R)
    print(f"[Baseline] LQR stable? {is_stable(A + B @ K_lqr)}")

    # DRC synthesis via SDP
    res = synthesize_state_feedback(A, B, Cz, Dzu, Sigma_nom, gamma, solver="MOSEK")
    print(f"[SDP] status={res['status']}, optval={res['optval']:.4f}")
    K = res["K"]
    if K is None:
        print("[SDP] could not recover K; exiting.")
        return
    print(f"[SDP] stable? {is_stable(A + B @ K)}")

    # Monte Carlo evaluation
    T = 4000
    cost_nom_lqr = simulate_closed_loop(A, B, Cz, Dzu, K_lqr, T=T, Sigma_w=Sigma_nom, rng=rng)
    cost_nom_drc = simulate_closed_loop(A, B, Cz, Dzu, K,      T=T, Sigma_w=Sigma_nom, rng=rng+1)

    Sigma_shift = Sigma_nom + 0.03 * np.eye(n)
    cost_shift_lqr = simulate_closed_loop(A, B, Cz, Dzu, K_lqr, T=T, Sigma_w=Sigma_shift, rng=rng+2)
    cost_shift_drc = simulate_closed_loop(A, B, Cz, Dzu, K,      T=T, Sigma_w=Sigma_shift, rng=rng+3)

    print(f"[Eval] Avg ||z||^2 (nominal): LQR={cost_nom_lqr:.3f}, DRC={cost_nom_drc:.3f}")
    print(f"[Eval] Avg ||z||^2 (shifted): LQR={cost_shift_lqr:.3f}, DRC={cost_shift_drc:.3f}")

    # Plot
    labels = ["Nominal", "Shifted"]
    lqr_vals = [cost_nom_lqr, cost_shift_lqr]
    drc_vals = [cost_nom_drc, cost_shift_drc]
    x = np.arange(len(labels)); w = 0.35
    fig, ax = plt.subplots()
    ax.bar(x - w/2, lqr_vals, width=w, label="LQR")
    ax.bar(x + w/2, drc_vals, width=w, label="DRC (SDP)")
    ax.set_ylabel("Avg ||z||^2")
    ax.set_title("Distributionally robust synthesis demo")
    ax.set_xticks(x); ax.set_xticklabels(labels)
    ax.legend()
    plt.tight_layout()
    out = "drc_demo_eval.png"
    plt.savefig(out, dpi=150)
    print(f"[Plot] saved to {out}")

if __name__ == "__main__":
    main()
