import os
import numpy as np
import matplotlib.pyplot as plt

def plot_all_npz_contents(npz_path, model="independent", out_dir=None, x_axis_key="step"):
    """
    Load an .npz file and save one PDF plot for each key.

    Rules:
    - scalar: save a figure with the scalar value as text
    - 1D array: single plot
    - 2D array: one subplot per column
    - higher-dimensional arrays: skipped with a warning

    Parameters
    ----------
    npz_path : str
        Path to the .npz file
    out_dir : str or None
        Output directory. If None, uses current working directory.
    x_axis_key : str
        Key to use as x-axis when compatible (default: 'step')
    """
    if out_dir is None:
        out_dir = os.getcwd()

    os.makedirs(out_dir, exist_ok=True)

    data = np.load(npz_path, allow_pickle=True)

    # optional shared x-axis
    x_axis = None
    if x_axis_key in data.files:
        candidate_x = np.asarray(data[x_axis_key])
        if candidate_x.ndim == 1:
            x_axis = candidate_x

    print(f"\nSaving plots to: {out_dir}\n")

    for key in data.files:
        arr = np.asarray(data[key])

        # sanitize filename
        safe_key = "".join(c if c.isalnum() or c in ("_", "-") else "_" for c in key)
        pdf_path = os.path.join(out_dir, f"{model}_{safe_key}.pdf")

        # ----------------------------
        # Case 1: scalar
        # ----------------------------
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

        # ----------------------------
        # Case 2: 1D array
        # ----------------------------
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

        # ----------------------------
        # Case 3: 2D array
        # ----------------------------
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

        # ----------------------------
        # Case 4: ndim > 2
        # ----------------------------
        print(f"[skipped] {key}: ndim={arr.ndim} not supported")


def inspect_npz(path, verbose=True):
    """
    Inspect contents of a .npz file.

    Parameters
    ----------
    path : str
        Path to the .npz file
    verbose : bool
        If True, prints detailed info

    Returns
    -------
    data_dict : dict
        Dictionary with arrays (name -> numpy array)
    info_dict : dict
        Metadata (shape, dtype, stats)
    """
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

        # Try to compute stats if numeric
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
    # Example usage
    model = "correlated"  # or "correlated"
    path = f"PaperLike_2W_{model}_MBD___closed_loop_run.npz"
    #data_dict, info_dict = inspect_npz(path)
    plot_all_npz_contents(path, model)