
def select_gamma(p):
    if not bool(p.get("ambiguity", {}).get("fixGamma", False)):     # set fixGamma: 0
        if p.get("method", "lmi") == "lmi":                         # ------|set method: "lmi"
            if p.get("model", "correlated") == "correlated":        # ------|------| set model: "correlated"
                if bool(p.get("ident", {}).get("stabilise", True)): # ------|------|-------| set stabilise: true
                    if bool(p.get("use_set_out_mats", False)):      # ------|------|-------|-------| set use_set_out_mats: true            | runID: Opt&SetOutMats&Stabilise
                        gamma = 0.41640786499873816
                    else:                                           # ------|------|-------|-------| set use_set_out_mats: false           | runID: Opt&Stabilise
                        gamma = 0.6180339887498949
                else:                                               # ------|------|-------| set stabilise: false
                    if bool(p.get("use_set_out_mats", False)):      # ------|------|-------|-------| set use_set_out_mats: true            | runID: Opt&SetOutMats
                        gamma = 0.06888370749726605
                    else:                                           # ------|------|-------|-------| set use_set_out_mats: false           | runID: Opt
                        gamma = 0.9016994374947425
            else:                                                   # ------|------| set model: "independent"
                gamma = p.get("ambiguity", {}).get("gamma", 0.5)
        else:
            gamma = p.get("ambiguity", {}).get("gamma", 0.5)
    else:                                                           # set fixGamma: 1
        gamma = p.get("ambiguity", {}).get("gamma", 0.5)
    
    return gamma
