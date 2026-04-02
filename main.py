import sys

from analysis import (
    print_infos_comparison, print_infos,            # print_info.py
    MutipleRunsEvaluation, NsimSweep_FROM_DATA,     # Nsims_eval.py  
)
from config import cfg                              # loader.py
from core import run_exp                            # run.py
from utils import select_gamma                      # gamma_selection.py

# ----------------------------------------------------------------------------------

if __name__ == "__main__":
    p = cfg.get("params", {})
    gamma = select_gamma(p)

    if bool(p.get("FIND", 0)):
        from analysis.find_opt_gamma import opt_gamma
        opt_gamma(run_fn=run_exp)
        sys.exit(0)

    COST = bool(p.get("COST", 0))
    if not COST and not bool(p.get("test_Nsims", 0)):
        ALL = bool(p.get("ALL", False))
        if ALL:
            infos_mbd, m, o = run_exp(FROM_DATA=False, gamma=gamma, comp=False, ALL=ALL, info=True)
            infos_ddd, _, _ = run_exp(FROM_DATA=True, gamma=gamma, comp=False, ALL=ALL, info=True)
            run_exp(comp=True, gamma=gamma, ALL=ALL)
            print_infos_comparison(m, infos_mbd, infos_ddd, o)
        else:
            info, m, o = run_exp(gamma=gamma, SINGLE_RUN=True)
            print_infos(m, info, o, bool(p.get("FROM_DATA", 0)))
    
    else: 
        if not bool(p.get("test_Nsims", 0)):
            MutipleRunsEvaluation(p=p, run_fn=run_exp, gamma=gamma, COST=COST, N=100)
        else: 
            COST = True
            N_sims_values = [
                1,   2,   3,   5,          # Emergence of structure
                8,  12,  16,               # Early stability range
                20,  25,  30,               # Practical medoid stability region
                40,  50,  65,               # Larger-sample variance reduction
                80, 100, 120, 150           # High-data (plateau) regime
            ]
            NsimSweep_FROM_DATA(p=p, run_fn=run_exp, gamma=gamma, COST=COST, runs_per_N=10, N_sims_values=N_sims_values)
