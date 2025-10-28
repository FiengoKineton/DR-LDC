"""
make an optmisation problem based on the value of gamma such that i run

1. main.main(gamma, FROM_DATA=False)
2. OpenLoop.make_data(gamma)
3. main.main(gamma, FROM_DATA=True)
4. report = main.main(gamma, comp=True)
5. read from report the mse of the variable involved (select from x, u, y, z)
6. change gamma accordingly to reduce the mse

"""


import math, time, sys, yaml, csv, os, datetime
from pathlib import Path
import numpy as np
from typing import Callable, Dict, Any, Tuple, List

from main import main
from utils___simulate import Open_Loop

# ----------------------------- wiring to your pipeline -----------------------------

def run_main(gamma: float, *, FROM_DATA: bool = False, comp: bool = False, plot: bool = False) -> Dict[str, Any]:
    """
    Thin adapter over your main.main(...). Must return the 'report' dict
    exactly with the structure you showed. If main.main writes JSON,
    load and return it here instead.
    """
    # Example placeholder:
    return main(gamma=gamma, FROM_DATA=FROM_DATA, comp=comp, plot=plot)


def make_data_openloop(gamma: float) -> None:
    """
    Thin adapter over your OpenLoop.make_data(...).
    """
    # Example placeholder:
    Open_Loop(MAKE_DATA=True, EVAL_FROM_PATH=False, PLOT=False, gamma=gamma)

# ----------------------------- objective construction ------------------------------

def _safe_get(d: Dict, path: List[str], default=None):
    cur = d
    for k in path:
        if cur is None: return default
        if k not in cur: return default
        cur = cur[k]
    return cur

def _mse_from_traj_errors(report: Dict[str, Any], signal: str, *, use_nrmse: bool) -> float:
    """
    Pull a scalar from report['trajectory_errors']['signals'][signal].
    If normalize=True, use NRMSE; else RMSE. Falls back to np.inf if missing.
    """
    sig = _safe_get(report, ["trajectory_errors", "signals", signal], None)
    if not sig or not sig.get("present", False):
        return float("inf")
    key = "nrmse_overall" if use_nrmse else "rmse_overall"
    val = sig.get(key, None)
    if val is None or not np.isfinite(val):
        # fall back to mean of per-column, then to MAE
        alt = sig.get("rmse_mean", None) if not use_nrmse else sig.get("nrmse_mean", None)
        if alt is None or not np.isfinite(alt):
            alt = sig.get("mae_mean", None)
        return float(alt) if (alt is not None and np.isfinite(alt)) else float("inf")
    return float(val)

def _matrix_distance_from_deltas(report: Dict[str, Any],
                                 *,
                                 ctrl_keys: Tuple[str, ...] = ("Ac","Bc","Cc","Dc"),
                                 plant_keys: Tuple[str, ...] = ("A","Bu","Cy","Cz","Dzu"),
                                 composite_keys: Tuple[str, ...] = ("Acl","Ccl"),
                                 p_ctrl: float = 2.0,
                                 p_plant: float = 2.0,
                                 p_comp: float = 2.0,
                                 weights: Dict[str, float] = None) -> float:
    """
    Build a scalar “distance between matrices” using the Frobenius deltas already in the report.
    Uses an L^p aggregation per group, then sums with optional weights.
    """
    weights = weights or {"ctrl": 1.0, "plant": 1.0, "comp": 1.0}
    def grab(block_name, keys, p):
        block = report.get(f"{block_name}_deltas", {})  # e.g., 'controller_deltas'
        vals = []
        for k in keys:
            st = block.get(k, None)
            if st is None: 
                continue
            fn = st.get("fro_norm", None)
            if fn is None or not np.isfinite(fn):
                continue
            vals.append(float(fn))
        if not vals:
            return 0.0
        if math.isinf(p) or p <= 0:
            # default back to max
            return float(np.max(vals))
        return float((np.mean(np.array(vals) ** p)) ** (1.0 / p))

    d_ctrl = grab("controller", ctrl_keys, p_ctrl)
    d_plnt = grab("plant", plant_keys, p_plant)
    d_comp = grab("composite", composite_keys, p_comp)

    return float(weights.get("ctrl", 1.0) * d_ctrl
                 + weights.get("plant", 1.0) * d_plnt
                 + weights.get("comp", 1.0) * d_comp)

def _stability_penalty(report: Dict[str, Any],
                       *,
                       rho_target: float = 0.999,
                       slope: float = 100.0) -> float:
    """
    Penalize spectral radius above rho_target using a softplus-ish hinge.
    Uses composite Acl stats already in the report.
    """
    specM = _safe_get(report, ["stability", "MBD", "spectral_radius"], None)
    specD = _safe_get(report, ["stability", "DDD", "spectral_radius"], None)
    pen = 0.0
    for rho in (specM, specD):
        if rho is None or not np.isfinite(rho):
            pen += 1e3  # missing spectra is not a free lunch
        else:
            margin = float(rho) - rho_target
            if margin > 0:
                # smooth penalty
                pen += float(math.log1p(math.exp(slope * margin)) / slope)
    return pen

def _build_scalar_objective(report: dict,
                            *,
                            # you may pass either `signal` (str) or `signals` (list[str])
                            signal: str | None = None,
                            signals: list[str] | None = None,
                            signal_mix_weights: dict[str,float] | None = None,
                            use_nrmse: bool = True,
                            w_traj: float = 1.0,
                            w_mats: float = 0.1,
                            w_stab: float = 10.0,
                            delta_weights: dict[str,float] | None = None) -> dict[str,float]:
    """
    Compose f = w_traj*E + w_mats*D + w_stab*P.
    E is a (possibly weighted) average of per-signal RMSE/NRMSE across the selected signals.
    """
    import numpy as np

    # normalize input
    if signals is None:
        if signal is None:
            signal = "y"   # default
        signals = [signal]
    else:
        # make sure it's a clean list of unique strings
        signals = [str(s) for s in signals]
    if signal_mix_weights is None:
        # equal weights for whatever you passed
        signal_mix_weights = {s: 1.0 for s in signals}

    # gather per-signal errors
    per_sig_vals = []
    per_sig_used = []
    for s in signals:
        val = _mse_from_traj_errors(report, s, use_nrmse=use_nrmse)
        if np.isfinite(val):
            w = float(signal_mix_weights.get(s, 1.0))
            per_sig_vals.append((val, w))
            per_sig_used.append(s)

    if not per_sig_vals:
        E = float("inf")   # nothing usable, call it infeasible
    else:
        num = sum(v * w for v, w in per_sig_vals)
        den = sum(w for _, w in per_sig_vals)
        E = float(num / den) if den > 0 else float("inf")

    # matrix deltas
    D = _matrix_distance_from_deltas(
        report,
        ctrl_keys=("Ac","Bc","Cc","Dc"),
        plant_keys=("A","Bu","Cy","Cz","Dzu"),
        composite_keys=("Acl","Ccl"),
        p_ctrl=2.0, p_plant=2.0, p_comp=2.0,
        weights=delta_weights or {"ctrl": 1.0, "plant": 1.0, "comp": 0.5},
    )

    # stability penalty
    P = _stability_penalty(report, rho_target=0.999, slope=80.0)

    total = float(w_traj * E + w_mats * D + w_stab * P)
    return {
        "E_traj": E,
        "D_mats": D,
        "P_stab": P,
        "total": total,
        "signals_used": per_sig_used,
        "use_nrmse": bool(use_nrmse),
    }

# ----------------------------- evaluation wrapper ----------------------------------

def _evaluate_gamma_once(gamma: float,
                         *,
                         signal: str,
                         signals: list[str],
                         signal_mix_weights: dict[str,float],
                         use_nrmse: bool,
                         weights: Dict[str, float],
                         delta_weights: Dict[str, float],
                         cache: Dict[float, Dict[str, Any]]) -> Tuple[float, Dict[str, Any]]:
    """
    Run the 4-step pipeline for this gamma, compute objective, cache results.
    """
    if gamma in cache: 
        return cache[gamma]["objective"]["total"], cache[gamma]

    t0 = time.time()
    # 1) baseline MBD
    _ = run_main(gamma=gamma, FROM_DATA=False, comp=False, plot=False)
    # 2) generate data with this gamma
    #make_data_openloop(gamma=gamma)
    # 3) DDD run
    _ = run_main(gamma=gamma, FROM_DATA=True, comp=False, plot=False)
    # 4) final comparison (must return the 'report' dict)
    report = run_main(gamma=gamma, FROM_DATA=True, comp=True, plot=False)

    obj = _build_scalar_objective(
        report,
        signal=signal,               # may be ignored if `signals` is not None
        signals=signals,             # capture from outer scope via closure or add param
        signal_mix_weights=signal_mix_weights,
        use_nrmse=use_nrmse,
        w_traj=weights.get("traj", 1.0),
        w_mats=weights.get("mats", 0.1),
        w_stab=weights.get("stab", 10.0),
        delta_weights=delta_weights,
    )
    t1 = time.time()
    rec = {"gamma": float(gamma), "report": report, "objective": obj, "elapsed_sec": float(t1 - t0)}
    cache[gamma] = rec
    return obj["total"], rec

# ----------------------------- bounded 1-D search ----------------------------------

def _golden_section_minimize(f: Callable[[float], Tuple[float, Dict[str, Any]]],
                             a: float, b: float,
                             *,
                             tol: float = 1e-3,
                             max_iter: int = 50) -> Dict[str, Any]:
    """
    Derivative-free minimization on [a,b]. Returns dict with best gamma and history.
    f must return (obj_value, payload).
    """
    phi = (1 + 5 ** 0.5) / 2
    invphi = (5 ** 0.5 - 1) / 2  # 1/phi
    # interior points
    c = b - invphi * (b - a)
    d = a + invphi * (b - a)

    fc, pay_c = f(c)
    fd, pay_d = f(d)

    history = [{"gamma": c, "obj": fc}, {"gamma": d, "obj": fd}]
    k = 0
    while (b - a) > tol and k < max_iter:
        if fc < fd:
            b, d, fd = d, c, fc
            c = b - invphi * (b - a)
            fc, pay_c = f(c)
            history.append({"gamma": c, "obj": fc})
        else:
            a, c, fc = c, d, fd
            d = a + invphi * (b - a)
            fd, pay_d = f(d)
            history.append({"gamma": d, "obj": fd})
        k += 1

    # pick best seen
    best = min(history, key=lambda r: r["obj"])
    # Attach the full payload for the final best gamma
    _, final_payload = f(best["gamma"])
    return {
        "best_gamma": float(best["gamma"]),
        "best_obj": float(best["obj"]),
        "best_payload": final_payload,
        "iters": k,
        "interval": [float(a), float(b)],
        "history": history,
    }

# ----------------------------- public API ------------------------------------------

def optimize_gamma(
    *,
    gamma_bounds: Tuple[float, float],
    signal: str = "y",         # choose among {"x","u","y","z","xc"}
    signals: list[str] | None = None,  # NEW: to combine multiple
    signal_mix_weights: dict[str,float] | None = None,  # NEW: weights among signals
    use_nrmse: bool = True,    # True -> use NRMSE, False -> RMSE
    weights: Dict[str, float] = None,      # {"traj":1.0, "mats":0.1, "stab":10.0}
    delta_weights: Dict[str, float] = None,# {"ctrl":1.0,"plant":1.0,"comp":0.5}
    tol: float = 1e-3,
    max_iter: int = 40,
) -> Dict[str, Any]:
    """
    Tune gamma in [gamma_bounds] to minimize a scalar objective composed of:
      - trajectory error between MBD and DDD on chosen signal
      - aggregated Frobenius deltas between matrices
      - stability penalty (spectral radius of Acl)

    Returns a dict with:
      - best_gamma, best_obj
      - best_payload: {"gamma", "report", "objective", "elapsed_sec"}
      - history of evaluations
    """
    if weights is None:
        weights = {"traj": 1.0, "mats": 0.1, "stab": 10.0}
    if delta_weights is None:
        delta_weights = {"ctrl": 1.0, "plant": 1.0, "comp": 0.5}

    cache: Dict[float, Dict[str, Any]] = {}

    def f(g: float) -> Tuple[float, Dict[str, Any]]:
        try:
            return _evaluate_gamma_once(
                float(g),
                signal=signal,
                signals=signals,
                signal_mix_weights=signal_mix_weights,
                use_nrmse=use_nrmse,
                weights=weights,
                delta_weights=delta_weights,
                cache=cache,
            )
        except Exception as e:
            # If a run crashes, treat as infinite objective. Yes, harsh. That’s the point.
            print(e)
            sys.exit(0)
            return float("inf"), {"gamma": float(g), "error": repr(e)}

    res = _golden_section_minimize(f, float(gamma_bounds[0]), float(gamma_bounds[1]),
                                   tol=tol, max_iter=max_iter)

    # optional: pretty print summary
    best = res["best_payload"]
    obj = best["objective"]
    print("\n[gamma-opt] best_gamma = {:.6g}, total = {:.6g} | E_traj = {:.6g}, D_mats = {:.6g}, P_stab = {:.6g}".format(
        res["best_gamma"], obj["total"], obj["E_traj"], obj["D_mats"], obj["P_stab"]
    ))
    return res

# ----------------------------- save res --------------------------------------------

def save(res, 
        gamma_bounds: Tuple[float, float],
        signal: str = "y",         # choose among {"x","u","y","z","xc"}
        signals: list[str] | None = None,  # NEW: to combine multiple
        signal_mix_weights: dict[str,float] | None = None,  # NEW: weights among signals
        use_nrmse: bool = True,    # True -> use NRMSE, False -> RMSE
        weights: Dict[str, float] = None,      # {"traj":1.0, "mats":0.1, "stab":10.0}
        delta_weights: Dict[str, float] = None,# {"ctrl":1.0,"plant":1.0,"comp":0.5}
        tol: float = 1e-3,
        max_iter: int = 40,):
    
    if yaml is None:
        raise ImportError("PyYAML not available. Install with `pip install pyyaml`.")
    with open("problem___parameters.yaml", "r", encoding="utf-8") as f:
        cfg = yaml.safe_load(f)

    p = cfg.get("params", {})
    out = p.get("directories", {}).get("artifacts", "./out/artifacts/")
    m = p.get("ambiguity", {}).get("model", "W2")
    _runID = p.get("directories", {}).get("runID", "temp")
    _type = p.get("plant", {}).get("type", "explicit")
    _model = p.get("model", "independent") if m=="W2" else m
    _method = p.get("method", "lmi")

    path_name = f"/{_type}_{_model}_GammaSweep.csv"
    csv_path = Path(out).with_suffix("").as_posix() + f"/{_method}" + f"/run_{_runID}" + path_name

    # one unified schema for ALL rows
    fields = [
        # evaluation summary
        "timestamp","tag","gamma","obj_total","E_traj","D_mats","P_stab",
        "elapsed_sec","signals_used","use_nrmse","rho_mbd","rho_ddd",
        # optimizer result footprint
        "iters","interval_lo","interval_hi","history_len",
        # knobs passed in
        "gamma_lo","gamma_hi","signals","signal_mix_weights",
        "w_traj","w_mats","w_stab","dw_ctrl","dw_plant","dw_comp","tol","max_iter",
    ]

    is_new = not os.path.exists(csv_path)
    now = datetime.datetime.now().isoformat(timespec="seconds")

    with open(csv_path, "a", newline="", encoding="utf-8") as fcsv:
        w = csv.DictWriter(fcsv, fieldnames=fields)
        if is_new:
            w.writeheader()

        # write each history point as tag=eval
        for h in res.get("history", []):
            w.writerow({
                "timestamp": now,
                "tag": "eval",
                "gamma": h.get("gamma"),
                "obj_total": h.get("obj"),
                # no detailed metrics for eval points -> leave blank
                "E_traj": "", "D_mats": "", "P_stab": "",
                "elapsed_sec": "", "signals_used": "", "use_nrmse": "",
                "rho_mbd": "", "rho_ddd": "",
                # overall result context is constant; include for every row
                "iters": res.get("iters"),
                "interval_lo": (res.get("interval") or [None, None])[0],
                "interval_hi": (res.get("interval") or [None, None])[1],
                "history_len": len(res.get("history", [])),
                # knobs
                "gamma_lo": gamma_bounds[0],
                "gamma_hi": gamma_bounds[1],
                "signals": ",".join(signals),
                "signal_mix_weights": ";".join(f"{k}:{v}" for k, v in signal_mix_weights.items()),
                "w_traj": weights["traj"], "w_mats": weights["mats"], "w_stab": weights["stab"],
                "dw_ctrl": delta_weights["ctrl"], "dw_plant": delta_weights["plant"], "dw_comp": delta_weights["comp"],
                "use_nrmse": use_nrmse, "tol": tol, "max_iter": max_iter,
            })

        # write the best row with full metrics
        best = res["best_payload"]; obj = best["objective"]; rpt = best.get("report", {}) or {}
        rho_mbd = ((rpt.get("stability") or {}).get("MBD", {}) or {}).get("spectral_radius", "")
        rho_ddd = ((rpt.get("stability") or {}).get("DDD", {}) or {}).get("spectral_radius", "")

        w.writerow({
            "timestamp": now,
            "tag": "best",
            "gamma": res["best_gamma"],
            "obj_total": obj.get("total"),
            "E_traj": obj.get("E_traj"),
            "D_mats": obj.get("D_mats"),
            "P_stab": obj.get("P_stab"),
            "elapsed_sec": best.get("elapsed_sec", ""),
            "signals_used": ",".join(obj.get("signals_used", [])) if obj.get("signals_used") else "",
            "use_nrmse": obj.get("use_nrmse", use_nrmse),
            "rho_mbd": rho_mbd,
            "rho_ddd": rho_ddd,
            "iters": res.get("iters"),
            "interval_lo": (res.get("interval") or [None, None])[0],
            "interval_hi": (res.get("interval") or [None, None])[1],
            "history_len": len(res.get("history", [])),
            "gamma_lo": gamma_bounds[0],
            "gamma_hi": gamma_bounds[1],
            "signals": ",".join(signals),
            "signal_mix_weights": ";".join(f"{k}:{v}" for k, v in signal_mix_weights.items()),
            "w_traj": weights["traj"], "w_mats": weights["mats"], "w_stab": weights["stab"],
            "dw_ctrl": delta_weights["ctrl"], "dw_plant": delta_weights["plant"], "dw_comp": delta_weights["comp"],
            "tol": tol, "max_iter": max_iter,
        })


# ==========================================================================================================================================================

if __name__ == "__main__":
    gamma_bounds = (0.0, 1.0)
    signals = ["y", "z", "x", "u"]
    signal_mix_weights = {"y": 0.3, "z": 0.4, "x": 0.2, "u": 0.1}
    use_nrmse = True
    weights = {"traj": 1.0, "mats": 0.1, "stab": 20.0}
    delta_weights = {"ctrl": 1.0, "plant": 0.7, "comp": 0.3}
    tol = 5e-3
    max_iter = 30

    res = optimize_gamma(
        gamma_bounds=gamma_bounds,
        signals=signals,
        signal_mix_weights=signal_mix_weights,
        use_nrmse=use_nrmse,
        weights=weights,
        delta_weights=delta_weights,
        tol=tol, max_iter=max_iter,
    )

    save(res, 
        gamma_bounds=gamma_bounds,
        signals=signals,
        signal_mix_weights=signal_mix_weights,
        use_nrmse=use_nrmse,
        weights=weights,
        delta_weights=delta_weights,
        tol=tol, max_iter=max_iter,
    )
