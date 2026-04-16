import numpy as np

## Helper functions for the DRO-LMI controllers.

def _pseudo_inv(D, r=1e-6):
    return D.T @ np.linalg.inv(D @ D.T + r * np.eye(D.shape[0]))

def I(n):
    return np.eye(n)

def Z(r, c): 
    return np.zeros((r, c)) 

def negdef(M, eps=1e-5): 
    return (M << -eps * np.eye(M.shape[0]))

def _val(x):
    if x is None:
        return None
    return float(x) if np.isscalar(x) else x


## Helper functions for the DRO-LMI controllers.
def _safe_scalar(v):
    if v is None:
        return None
    try:
        return float(v)
    except Exception:
        return v

def _print_header(title):
    print("\n" + "=" * 70)
    print(title)
    print("=" * 70)

def _print_scale_dict(name, d):
    print(f"\n{name}")
    for k, v in d.items():
        print(f"  {k:<24}: {v}")
