import os




def print_infos_comparison(m: str, infos_mbd: dict, infos_ddd: dict, path: str):
    """
    Pretty-print AND save a comparison table between MBD and DDD info dicts.

    Args:
        m: title (e.g., "2W_independent")
        infos_mbd: dict for MBD
        infos_ddd: dict for DDD
        path: full file path (e.g., "results/summary.txt")
    """

    metrics = [
        ("J",               "Cost J"),
        ("obj",             "Objective"),
        ("lamda",           "λ"),
        ("rho",             "ρ"),
        ("snr",             "SNR [dB]"),
        ("time",            "Time [s]"),
        ("attempts",        "Attempts"),
        ("stress",          "Stress"),
        ("ratio_violation", "Violations [%]"),
        ("solver",          "Solver"),
    ]

    def fmt(v):
        if isinstance(v, (int, float)):
            return f"{v:.4g}"
        return str(v)

    # === BUILD STRING ===
    lines = []

    lines.append("\n" + "=" * 70)
    lines.append(f" {m} summary ".center(70, "="))
    lines.append("=" * 70)

    header = f"{'Metric':<15}{'MBD':>15}{'DDD':>15}{'DDD - MBD':>15}"
    lines.append(header)
    lines.append("-" * 70)

    for key, label in metrics:
        v_m = infos_mbd.get(key, None)
        v_d = infos_ddd.get(key, None)

        if isinstance(v_m, (int, float)) and isinstance(v_d, (int, float)):
            diff = v_d - v_m
            diff_str = f"{diff:+.3g}"
        else:
            diff_str = ""

        line = f"{label:<15}{fmt(v_m):>15}{fmt(v_d):>15}{diff_str:>15}"
        lines.append(line)

    lines.append("=" * 70 + "\n")

    # join everything
    table_str = "\n".join(lines)

    # === PRINT ===
    print(table_str)

    # === SAVE TO FILE ===
    if path is not None:
        path = path.rstrip("/")

        # remove trailing "_MBD" if present
        if path.endswith("_MBD"):
            path = path[:-4]

        # final file path (same directory, just rename)
        final_path = f"{path}_summary.txt"

        os.makedirs(os.path.dirname(final_path), exist_ok=True)

        with open(final_path, "w", encoding="utf-8") as f:
            f.write(table_str)

