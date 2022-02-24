import numpy as np

def get_finite_f_bound(n, B=1, D=768, delta=1e-8):
    abs_F = (D+1) * (2**32)
    return B*np.sqrt(2 * np.log((2 * abs_F / delta)) / n)

def get_mcallester_bound(n, B=1):
    KL = 1e5
    delta = 1e-5
    return B*np.sqrt((KL + np.log(4*n/delta)) / (2*n-1))

def requirement_from_finite_f_bound(bound, B=1, D=768, delta=1e-8, hid=True):
    if hid:
        H = 20
        num_params = (D+1)*H + (H+1)
    else:
        num_params = D+1
    abs_F = num_params * (2**32)
    return 2 * np.log(2 * abs_F / delta) / ((bound / B)**2)

def n1_requirement(effect, model, hid=False):
    if model.startswith("bert"):
        D = 768
    else:
        D = {"infersent": 4096, "glove": 300, "sbert": 768}[model]
    if effect < 1e-5:
        return np.nan
    return int(requirement_from_finite_f_bound(effect, D=D, hid=hid))