from .loader import build_runtime_config

_CFG = None

def get_cfg():
    global _CFG
    if _CFG is None:
        _CFG = build_runtime_config()
    return _CFG