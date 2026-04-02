import os, sys
import numpy as np
import matplotlib.pyplot as plt

from pathlib import Path
ROOT = Path(__file__).resolve().parents[4]
sys.path.append(str(ROOT))
from __utils___simulate import Closed_Loop




def extract_sim_from_npz(npz_path):
    """
    Load only the closed-loop arrays needed by Closed_Loop methods.

    Keeps only:
        [X, Xc, U, W, Y, Z, T]
    and reconstructs the dimension metadata needed by plot_timeseries().
    """
    data = np.load(npz_path, allow_pickle=True)

    required = ["X", "Xc", "U", "W", "Y", "Z", "T"]
    missing = [k for k in required if k not in data.files]
    if missing:
        raise KeyError(f"Missing required keys in {npz_path}: {missing}")

    sim = {k: np.asarray(data[k]) for k in required}

    # Convert T to plain int if stored as 0-d array
    sim["T"] = int(np.asarray(sim["T"]).item())

    # Rebuild shape metadata expected by plot_timeseries()
    sim["nx"]  = sim["X"].shape[1]  if sim["X"].ndim  == 2 else 1
    sim["nxc"] = sim["Xc"].shape[1] if sim["Xc"].ndim == 2 else 1
    sim["nu"]  = sim["U"].shape[1]  if sim["U"].ndim  == 2 else 1
    sim["nw"]  = sim["W"].shape[1]  if sim["W"].ndim  == 2 else 1
    sim["ny"]  = sim["Y"].shape[1]  if sim["Y"].ndim  == 2 else 1
    sim["nz"]  = sim["Z"].shape[1]  if sim["Z"].ndim  == 2 else 1

    # Optional step axis for generic plotting/debug
    sim["step"] = np.arange(sim["T"], dtype=int)

    return sim


def run_closed_loop_postprocessing(npz_path, out_dir=None, plot=True, save_timeseries=True):
    """
    Load sim from .npz, instantiate Closed_Loop, and call:
      - plot_timeseries(sim)
      - simulate_Z_cost(Z=sim["Z"])
      - simulate_ZW_snr(Z=sim["Z"], W=sim["W"])
    """
    sim = extract_sim_from_npz(npz_path)

    cl = Closed_Loop()

    # Important:
    # simulate_Z_cost() and simulate_ZW_snr() use self.Tf/self.ts internally
    # to guess orientation in some cases.
    # So we align Tf with the loaded trajectory length.
    cl.Tf = sim["T"] * cl.ts

    print("\nLoaded sim keys:")
    print({k: v.shape if isinstance(v, np.ndarray) else v for k, v in sim.items()})

    # 1) Plot time series from the loaded sim
    if plot:
        if save_timeseries:
            if out_dir is None:
                out_dir = os.getcwd()
            os.makedirs(out_dir, exist_ok=True)
            saved = cl.plot_timeseries(
                sim,
                save=True,
                out=os.path.join(out_dir, "loaded_sim"),
                fmt="pdf"
            )
            print("\nSaved timeseries plots:")
            for name, path in saved.items():
                print(f"  {name}: {path}")
        else:
            cl.plot_timeseries(sim)

    # 2) Cost from Z
    sim_cost = cl.simulate_Z_cost(Z=sim["Z"], plot=plot)
    print("\nFinal closed-loop cost J =", sim_cost["J"])

    # 3) SNR from Z and W
    sim_snr = cl.simulate_ZW_snr(Z=sim["Z"], W=sim["W"], plot=plot)
    print("Global SNR =", sim_snr["snr_db"], "dB")

    return {
        "sim": sim,
        "cost": sim_cost,
        "snr": sim_snr,
    }


# ------------------------------------------------------------------
# Your existing generic inspection/plot functions can stay as they are
# ------------------------------------------------------------------

def plot_all_npz_contents(npz_path, model="independent", out_dir=None, x_axis_key="step"):
    """
    Load an .npz file and save one PDF plot for each key.

    Rules:
    - scalar: save a figure with the scalar value as text
    - 1D array: single plot
    - 2D array: one subplot per column
    - higher-dimensional arrays: skipped with a warning
    """
    if out_dir is None:
        out_dir = os.getcwd()

    os.makedirs(out_dir, exist_ok=True)

    data = np.load(npz_path, allow_pickle=True)

    x_axis = None
    if x_axis_key in data.files:
        candidate_x = np.asarray(data[x_axis_key])
        if candidate_x.ndim == 1:
            x_axis = candidate_x

    print(f"\nSaving plots to: {out_dir}\n")

    for key in data.files:
        arr = np.asarray(data[key])

        safe_key = "".join(c if c.isalnum() or c in ("_", "-") else "_" for c in key)
        pdf_path = os.path.join(out_dir, f"{model}_{safe_key}.pdf")

        if arr.ndim == 0:
            fig, ax = plt.subplots(figsize=(6, 3))
            ax.axis("off")
            ax.text(
                0.5, 0.5,
                f"{key} = {arr.item()}",
                ha="center", va="center", fontsize=16
            )
            ax.set_title(f"{key} (scalar)")
            fig.tight_layout()
            fig.savefig(pdf_path, format="pdf", bbox_inches="tight")
            plt.close(fig)
            print(f"[saved] {pdf_path}")
            continue

        if arr.ndim == 1:
            fig, ax = plt.subplots(figsize=(10, 4))

            if x_axis is not None and len(x_axis) == len(arr) and key != x_axis_key:
                ax.plot(x_axis, arr)
                ax.set_xlabel(x_axis_key)
            else:
                ax.plot(arr)
                ax.set_xlabel("index")

            ax.set_title(f"{key} - shape {arr.shape}")
            ax.set_ylabel(key)
            ax.grid(True, alpha=0.3)

            fig.tight_layout()
            fig.savefig(pdf_path, format="pdf", bbox_inches="tight")
            plt.close(fig)
            print(f"[saved] {pdf_path}")
            continue

        if arr.ndim == 2:
            nrows, ncols = arr.shape
            fig, axes = plt.subplots(
                ncols, 1,
                figsize=(10, max(2.8 * ncols, 4)),
                squeeze=False,
                sharex=True
            )
            axes = axes.flatten()

            use_shared_x = x_axis is not None and len(x_axis) == nrows and key != x_axis_key

            for j in range(ncols):
                if use_shared_x:
                    axes[j].plot(x_axis, arr[:, j])
                    axes[j].set_xlabel(x_axis_key if j == ncols - 1 else "")
                else:
                    axes[j].plot(arr[:, j])
                    axes[j].set_xlabel("index" if j == ncols - 1 else "")

                axes[j].set_ylabel(f"{key}[:,{j}]")
                axes[j].set_title(f"{key} - column {j}")
                axes[j].grid(True, alpha=0.3)

            fig.suptitle(f"{key} - shape {arr.shape}", y=0.995)
            fig.tight_layout()
            fig.savefig(pdf_path, format="pdf", bbox_inches="tight")
            plt.close(fig)
            print(f"[saved] {pdf_path}")
            continue

        print(f"[skipped] {key}: ndim={arr.ndim} not supported")


def inspect_npz(path, verbose=True):
    data = np.load(path, allow_pickle=True)

    data_dict = {}
    info_dict = {}

    if verbose:
        print(f"\n📦 File: {path}")
        print(f"🔑 Keys found: {list(data.keys())}\n")

    for key in data.files:
        arr = data[key]
        data_dict[key] = arr

        info = {
            "shape": arr.shape,
            "dtype": arr.dtype,
            "ndim": arr.ndim
        }

        if np.issubdtype(arr.dtype, np.number):
            info.update({
                "min": float(np.min(arr)),
                "max": float(np.max(arr)),
                "mean": float(np.mean(arr))
            })

        info_dict[key] = info

        if verbose:
            print(f"--- {key} ---")
            print(f"shape: {arr.shape}")
            print(f"dtype: {arr.dtype}")
            if "min" in info:
                print(f"min/max: {info['min']:.4f} / {info['max']:.4f}")
                print(f"mean: {info['mean']:.4f}")
            print()

    return data_dict, info_dict


if __name__ == "__main__":
    model = "correlated"  # or "independent"
    path = f"PaperLike_2W_{model}_MBD___closed_loop_run.npz"

    # 1) generic inspection if you want
    # data_dict, info_dict = inspect_npz(path)

    # 2) generic per-key plots if you still want them
    # plot_all_npz_contents(path, model)

    # 3) use Closed_Loop methods directly on the loaded sim
    results = run_closed_loop_postprocessing(
        npz_path=path,
        out_dir=f"plots_{model}",
        plot=True,
        save_timeseries=True,
    )

    print("\nReturned summary:")
    print("J      =", results["cost"]["J"])
    print("SNR dB =", results["snr"]["snr_db"])
