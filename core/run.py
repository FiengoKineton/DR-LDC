# main.py
import sys
import numpy as np
from pathlib import Path

from config import get_cfg                      # loader.py
from controllers import (
    baseline_optim_problem,                 # baseline.py
    lmi_pipeline_optim_problem,             # dro_lmi.py
)
from utils import _generate_dir, Noise      # directory.py


# =============================================================================================== #

def run_exp(
        gamma: float = None, 
        FROM_DATA: bool = None, 
        comp: bool = None, 
        plot: bool = None, 
        ALL: bool = False, 
        COST: bool = False, 
        info: bool = False, 
        N_sims: int = None, 
        SINGLE_RUN: bool = False,
        ):


    p = get_cfg().get("params", {})
    out_base = Path(p.get("directories", {}).get("artifacts", "./out/artifacts/")).with_suffix("")#.as_posix()
    FROM_DATA = bool(p.get("FROM_DATA", False)) if FROM_DATA is None else FROM_DATA


    _upd = bool(p.get("upd", 0))
    _re_evaluate = bool(p.get("re_evaluate", 0)) if not ALL else False
    _plot = bool(p.get("plot", False)) if plot is None and not COST else plot
    _save = p.get("save", False) if not COST else False
    _comp = bool(p.get("comp", 0)) if comp is None else comp
    _ts = p.get("simulation", {}).get("ts", 0.5)
    _init_cond = p.get("simulation", {}).get("init_cond", "rand")

    _percent = int(p.get("ambiguity", {}).get("percent", 1)*100)
    
    # ------------------------------------------------------------------------------------------- #

    m, path_name, (_method, _runID, _model) = _generate_dir(p, ALL, FROM_DATA)

    #gamma = p.get("ambiguity", {}).get("gamma", 0.5) if gamma is None else gamma
    if gamma is None or m != "W2":
        gamma = p.get("ambiguity", {}).get("gamma", 0.5)
    else:
        gamma = gamma


    var = float(p.get("ambiguity", {})["var"])
    n = p.get("dimensions", {}).get("nw", 2)
    Sigma_nom = np.array(p.get("ambiguity", {})["Sigma_nom"], dtype=float) if m!="Gaussian" else var * np.eye(n)

    noise = Noise(Sigma_nom=Sigma_nom, avrg=0, var=var, n=n, gamma=gamma)

    # ------------------------------------------------------------------------------------------- #

    if _comp:
        from analysis.Comparator import ResultsComparator
        cmp = ResultsComparator(out_root=out_base, save=_save, ts=_ts)
        return cmp.compare_mbd_vs_ddd(path_name=path_name, method=_method, ID=_runID, plot=_plot, re_evaluate=_re_evaluate, init_cond=_init_cond, percent=_percent)
        # cmp.compare_baseline_vs_lmi(path_name=path_name, plot=True)
    else:
        out = out_base / f"{_method}" / f"run_{_runID}"
        out.mkdir(parents=True, exist_ok=True)
        out = out / path_name
        out = out.as_posix()

        if _method == "base":
            print("\nRunning baseline optimization...")
            opt = baseline_optim_problem(out=out, Sigma_nom=Sigma_nom, gamma=gamma, plot=_plot if not ALL else False, save=_save if not ALL else True, FROM_DATA=FROM_DATA, init_cond=_init_cond)
        else:
            print("\nRunning LMI pipeline optimization...")
            opt = lmi_pipeline_optim_problem(params=p, out=out, upd=_upd, noise=noise, N_sims=N_sims,
                                             plot=_plot if not ALL else False, save=_save if not ALL else True, 
                                             FROM_DATA=FROM_DATA, init_cond=_init_cond, disturbance_type=_model, )

        if COST or info:
            return opt._return_final_infos(), _model, out
        
        if SINGLE_RUN: 
            from analysis.Comparator import ResultsComparator
            cmp = ResultsComparator(out_root=out_base, save=_save, ts=_ts)
            cmp.plot_single_mbd_or_ddd(path_name=path_name, method=_method, ID=_runID, plot=_plot, re_evaluate=_re_evaluate, init_cond=_init_cond, percent=_percent)
            return opt._return_final_infos(), _model, out
        
    # ------------------------------------------------------------------------------------------- #

    if bool(p.get("SNR", 1)): 
        plant, ctrl, sim, Sigma = opt.get_snr_vars()

        from analysis.SNR import SNRAnalyzer
        an = SNRAnalyzer(plant=plant, ctrl=ctrl, Sigma=Sigma)
        res = an.snr()

        print({k: v for k, v in res.items()}) # if k.endswith("_dB") or k=="spectral_radius_Acl"})
        an.plot_bars(title="SNR for my controller")

        if 1:
            an.plot_output_psd(sim["X"], "x", fs=1.0/p.get("simulation", {}).get("ts", 0.05), nfft=4096)
            an.plot_output_psd(sim["Y"], "y", fs=1.0/p.get("simulation", {}).get("ts", 0.05), nfft=4096)
            an.plot_output_psd(sim["Z"], "z", fs=1.0/p.get("simulation", {}).get("ts", 0.05), nfft=4096)
            an.plot_output_psd(sim["U"], "u", fs=1.0/p.get("simulation", {}).get("ts", 0.05), nfft=4096)

        # 1) Evaluate SNR via kernels for any Σ
        res = an.snr_from_kernels()
        print(res)

        # 2) Worst/best directions (trace-normalized Σ)
        print("Z worst/best:", an.worst_best_snr("z"))

        # 3) Sweep Σ orientation in a 2D subspace and plot
        n.plot_snr_rotation_sweep(dims=(0,1), n_angles=181)

        # 4) Plot worst/best SNR bands
        an.plot_worst_best_lines()

# =============================================================================================== #