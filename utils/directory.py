
def _generate_dir(p: dict, ALL=True, FROM_DATA=True):
    m = p.get("ambiguity", {}).get("model", "W2")
    _runID = p.get("directories", {}).get("runID", "temp")
    _type = p.get("plant", {}).get("type", "explicit")
    _method = p.get("method", "lmi")
    _approach = p.get("approach", "DeePC")

    _data = "DDD" if FROM_DATA else "MBD"
    _old = bool(p.get("old_upd", 1))
    _estm = bool(p.get("estm_only", 0))
    _nonConvex = bool(p.get("non_convex", 0))
    _upd = bool(p.get("upd", 0))

    if m == "W2":
        _model = p.get("model", "independent")
    elif m == "2W":
        _model = m + "_" + p.get("model", "independent")
    else:
        _model = m

    if _method=="lmi" and ALL:
        if _upd:
            if _nonConvex: 
                _method = "lmi-nonConvex"       # WFL
            elif _estm:
                _method = "lmi-estm"            # Estm_dro_lmi
            elif not _old: 
                _method = "lmi-YoungSchur"      # Young_Schur_dro_lmi
            else:
                #_method = "lmi-upd"             # Young_dro_lmi or DeePC_dro_lmi (if approach==DeePC)
                if _approach=="DeePC":
                    _method = "lmi-DeePC"
                else: 
                    _method = "lmi-Young"
        else:
            _method = "lmi"

    path_name = f"{_type}_{_model}_{_data}"
    return m, path_name, (_method, _runID, _model)