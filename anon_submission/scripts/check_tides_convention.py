"""Check which interval/input pairing the pinned official TIDES code implements.

Compares the official TIDESSSM forward (UNTRAINED, fixed init seed 0; a mechanism
check, not a result) with two independent numpy recurrences that use the SAME
continuous-time model (Lambda, B, C, D, per-mode timescale) and differ only in
the interval convention:

  R (right endpoint / backward hold):  z_k = e^{L d_k} z_{k-1} + L^{-1}(e^{L d_k}-1) B u_k,
                                       d_k = t_k - t_{k-1},  y_k = Re(C z_k) + D u_k
  P (paper Eq. 2 / 9, forward hold):   x_{k+1} = e^{L D_k} x_k + L^{-1}(e^{L D_k}-1) B u_k,
                                       D_k = t_{k+1} - t_k,  y_k = Re(C x_k) + D u_k

Inputs are a single impulse on a fixed irregular grid and on a fixed uniform grid
(no random draws).  Output: results/raw/P1-REAL-01__convention_check.json
"""

import json
import os
import sys

import numpy as np

ROOT = os.path.dirname(os.path.dirname(os.path.abspath(__file__)))
sys.path.insert(0, os.path.join(ROOT, "src"))

from scssm.real import tides_adapter as TA  # noqa: E402


def main():
    torch, tides = TA.load_tides(os.path.join(ROOT, "third_party", "TIDES"))
    import random
    random.seed(0)
    np.random.seed(0)  # TIDES init also draws from numpy's global RNG
    torch.manual_seed(0)
    ssm = tides.TIDESSSM(ssm_size=4, blocks=1, H=1, conj_sym=False, discretization="zoh",
                         lambda_re_mode="lti", lambda_im_mode="lti", bc_mode="lti").double()
    out = {"status": "MECHANISM CHECK with UNTRAINED weights (init seed 0); not a model result"}
    grids = {
        "irregular": [0.2, 0.5, 0.7, 1.4, 1.5, 2.2, 3.0],
        "uniform": [0.25 * (k + 1) for k in range(8)],
    }
    for name, t in grids.items():
        L = len(t)
        u = np.zeros(L)
        j = 2
        u[j] = 1.0
        step = np.array([t[0]] + [t[k] - t[k - 1] for k in range(1, L)])  # as in tides_collate
        x = torch.tensor(u[:, None], dtype=torch.float64)
        s = torch.tensor(step, dtype=torch.float64)
        with torch.no_grad():
            y_code = ssm(x, s)[:, 0].numpy()
            Lam = ssm._get_lambda(x).numpy()
            Bt, Ct = ssm._get_BC(x)
            Bt, Ct = Bt.numpy()[:, 0], Ct.numpy()[0, :]
            Dv = float(ssm.D.detach().numpy()[0])
            scale = np.exp(ssm.log_step.detach().numpy().reshape(-1))
        # the per-mode timescale enters as step = gap * exp(log_step) in both references
        y_R = recur_right_scaled(Lam, Bt, Ct, Dv, u, t, scale)
        y_P = recur_paper_scaled(Lam, Bt, Ct, Dv, u, t, scale)
        out[name] = {
            "times": t, "impulse_index": j,
            "max_abs_code_minus_right": float(np.max(np.abs(y_code - y_R))),
            "max_abs_code_minus_paper": float(np.max(np.abs(y_code - y_P))),
            "code_state_part_at_impulse": float(y_code[j] - Dv * u[j]),
            "paper_state_part_at_impulse": float(y_P[j] - Dv * u[j]),
            "y_code": y_code.tolist(), "y_right": y_R.tolist(), "y_paper": y_P.tolist(),
        }
        # is the code state at k the paper state at k+1 (a pure one-index shift)?
        state_code = y_code - Dv * u
        state_paper = y_P - Dv * u
        out[name]["max_abs_code_state_k_minus_paper_state_k_plus_1"] = float(
            np.max(np.abs(state_code[:-1] - state_paper[1:])))
    path = os.path.join(ROOT, "results", "raw", "P1-REAL-01__convention_check.json")
    with open(path, "w") as f:
        json.dump(out, f, indent=2)
    print(json.dumps({k: {kk: vv for kk, vv in v.items() if kk.startswith(("max", "code_state", "paper_state"))}
                      if isinstance(v, dict) else v for k, v in out.items()}, indent=2))


def recur_right_scaled(Lam, B, C, D, u, t, scale):
    z = np.zeros(Lam.shape, dtype=complex)
    ys = []
    for k in range(len(u)):
        d = (t[k] - (t[k - 1] if k > 0 else 0.0)) * scale
        a = np.exp(Lam * d)
        z = a * z + (a - 1.0) / Lam * B * u[k]
        ys.append(float((C * z).sum().real + D * u[k]))
    return np.array(ys)


def recur_paper_scaled(Lam, B, C, D, u, t, scale):
    x = np.zeros(Lam.shape, dtype=complex)
    ys = []
    for k in range(len(u)):
        ys.append(float((C * x).sum().real + D * u[k]))
        if k + 1 < len(u):
            d = (t[k + 1] - t[k]) * scale
            a = np.exp(Lam * d)
            x = a * x + (a - 1.0) / Lam * B * u[k]
    return np.array(ys)


if __name__ == "__main__":
    main()
