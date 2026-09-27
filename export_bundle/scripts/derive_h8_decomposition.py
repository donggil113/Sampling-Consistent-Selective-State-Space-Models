"""Signed decomposition of the H8 pooled-readout change (configs/p1_real_02_h8_decomp.json).

P1-REAL-02 did not store per-token outputs, so this script runs ONE frozen native forward
per existing checkpoint on C0 and H8 (no training, no timing), checks that it reproduces
the stored per-trajectory p_mean, p_time and p_C0, and then computes

  p_mean(H8) - p(C0) = W + M,   W = sum_j (a_j - w_j) b_j = (7/18)(A - B),   M = sum_j a_j (z_j - b_j)
  p_time(H8) - p(C0) = Q,       Q = sum_j w_j (z_j - b_j)

with b_j the C0 output of token j's base interval, z_j the H8 output, a_j = 1/L, w_j = dt_j/T.
M and Q are split into tokens that end a base interval and the remaining (intra) tokens.
These signed terms are not an additive decomposition of the absolute-value estimand D_s.
"""

import gzip
import json
import os
import sys
import time

ROOT = os.path.dirname(os.path.dirname(os.path.abspath(__file__)))
sys.path.insert(0, os.path.join(ROOT, "src"))
sys.path.insert(0, os.path.join(ROOT, "scripts"))

from run_p1_real_01 import load_cfg, seed_all, setup, sha256_file, test_conditions  # noqa: E402
from run_p1_real_02 import OUT, PILOT_CKPT, PILOT_TRAIN, eval_mode_checks, load_testset  # noqa: E402
from scssm.real import protocol as P  # noqa: E402
from scssm.real import stats as S  # noqa: E402

CFG = os.path.join(ROOT, "configs", "p1_real_02_h8_decomp.json")
DOUT = os.path.join(OUT, "h8_decomposition")
ROWS = [("seed1", 1), ("seed2", 2), ("seed3", 3), ("seed4", 4), ("dev_seed0", 0)]
TOL = 1e-9


def rows_meta(row):
    if row == "dev_seed0":
        return json.load(open(PILOT_TRAIN)), PILOT_CKPT
    d = os.path.join(OUT, row)
    return json.load(open(os.path.join(d, "train.json"))), os.path.join(d, "checkpoint.pt")


def terms(g0, gh, b, z, T):
    """Signed decomposition for one trajectory (all in label units)."""
    L = len(z)
    dts = [gh.times[k] - gh.times[k - 1] for k in range(1, len(gh.times))]
    # base interval of every H8 token (1-based) and whether it ends that interval
    k_of, is_end = [], []
    for k in range(1, len(gh.base_index)):
        lo, hi = gh.base_index[k - 1], gh.base_index[k]
        for p in range(lo + 1, hi + 1):
            k_of.append(k)
            is_end.append(p == hi)
    assert len(k_of) == L
    bj = [b[k - 1] for k in k_of]
    a = 1.0 / L
    w = [d / T for d in dts]
    half = len(b) // 2
    A = sum(b[:half]) / half
    B = sum(b[half:]) / (len(b) - half)
    W = sum((a - wj) * x for wj, x in zip(w, bj))
    dev = [zj - x for zj, x in zip(z, bj)]
    M_end = sum(a * d for d, e in zip(dev, is_end) if e)
    M_intra = sum(a * d for d, e in zip(dev, is_end) if not e)
    Q_end = sum(wj * d for wj, d, e in zip(w, dev, is_end) if e)
    Q_intra = sum(wj * d for wj, d, e in zip(w, dev, is_end) if not e)
    p_c0 = sum(b) / len(b)
    p_mean = sum(z) / L
    p_time = sum(zj * d for zj, d in zip(z, dts)) / T
    return {"A": A, "B": B, "W": W, "W_identity": 7.0 / 18.0 * (A - B),
            "M": M_end + M_intra, "M_end": M_end, "M_intra": M_intra,
            "Q": Q_end + Q_intra, "Q_end": Q_end, "Q_intra": Q_intra,
            "p_C0": p_c0, "p_mean": p_mean, "p_time": p_time,
            "n_tokens": L, "n_end_tokens": sum(is_end),
            "sum_w_b_minus_pC0": sum(wj * x for wj, x in zip(w, bj)) - p_c0}


def main():
    cfg, _ = load_cfg()
    dc = json.load(open(CFG))
    TA, torch, tides = setup(cfg)
    t_wall, t_cpu = time.perf_counter(), time.process_time()
    rows, thash = load_testset()
    T, K = cfg["task"]["training_grid"]["T"], cfg["task"]["training_grid"]["K"]
    dt_train = T / K
    ids = [r["id"] for r in rows]
    conds = {i: test_conditions(P.trajectory_path(i), T, K, i) for i in ids}
    idx = S.draws(len(ids), 2000, 0)
    os.makedirs(DOUT, exist_ok=True)
    summary = {"config": os.path.relpath(CFG, ROOT), "testset_sha256": thash, "rows": {}}
    for row, seed in ROWS:
        tj, ckpt = rows_meta(row)
        if sha256_file(ckpt) != tj["checkpoint_sha256"]:
            raise SystemExit(f"checkpoint hash mismatch for {row}")
        x_stats, y_stats = tuple(tj["x_stats"]), tuple(tj["y_stats"])
        mu, sd = y_stats
        seed_all(torch, 0)
        model = TA.build_model(torch, tides, cfg)
        model.load_state_dict(torch.load(ckpt))
        model.eval()
        checks = eval_mode_checks(torch, model)
        out = {}
        with torch.inference_mode():
            for name in ("C0", "H8"):
                g_list = [conds[i][name] for i in ids]
                u, s, _ = TA.batch_from_grids(torch, g_list, None, dt_train, x_stats, y_stats)
                pred = TA.forward(torch, model, u, s).tolist()
                out[name] = [[p * sd + mu for p in r] for r in pred]
        stored = {}
        with open(os.path.join(OUT, row, "eval.jsonl")) as f:
            for line in f:
                r = json.loads(line)
                if r["condition"] == "H8":
                    stored[r["id"]] = r
        recs, max_mis, max_id = [], 0.0, 0.0
        for n_i, i in enumerate(ids):
            t = terms(conds[i]["C0"], conds[i]["H8"], out["C0"][n_i], out["H8"][n_i], T)
            st = stored[i]
            for key in ("p_mean", "p_time", "p_C0"):
                max_mis = max(max_mis, abs(t[key] - st[key]))
            max_id = max(max_id, abs(t["W"] - t["W_identity"]), abs(t["sum_w_b_minus_pC0"]),
                         abs(t["p_mean"] - t["p_C0"] - t["W"] - t["M"]), abs(t["p_time"] - t["p_C0"] - t["Q"]))
            t.update({"id": i, "pooled_label": rows[n_i]["pooled_label"]})
            recs.append(t)
        status = "COMPLETED" if max_mis <= TOL else "INVALID (forward does not reproduce stored eval.jsonl)"
        with gzip.open(os.path.join(DOUT, f"{row}_tokens.json.gz"), "wt") as f:
            json.dump({"ids": ids, "C0": out["C0"], "H8": out["H8"], "units": "label units"}, f)
        with open(os.path.join(DOUT, f"{row}_terms.jsonl"), "w") as f:
            for r in recs:
                f.write(json.dumps(r) + "\n")
        res = {"seed": seed, "role": "development (seed 0)" if row == "dev_seed0" else "training seed",
               "status": status, "checkpoint_sha256": tj["checkpoint_sha256"], "eval_mode_checks": checks,
               "max_abs_mismatch_vs_stored": max_mis, "max_abs_identity_residual": max_id}
        if status == "COMPLETED":
            for key in ("W", "M", "M_end", "M_intra", "Q", "Q_end", "Q_intra"):
                e, c = S.mean_ci([r[key] for r in recs], idx)
                res[key] = {"mean": e, "ci95": c, "mean_abs": sum(abs(r[key]) for r in recs) / len(recs)}
            dm = [r["p_mean"] - r["p_C0"] for r in recs]
            e, c = S.mean_ci(dm, idx)
            res["pmean_minus_pC0"] = {"mean": e, "ci95": c, "mean_abs": sum(abs(x) for x in dm) / len(dm)}
            res["A_minus_B_mean"] = sum(r["A"] - r["B"] for r in recs) / len(recs)
            res["frac_sign_W_eq_sign_change"] = sum(1 for r, x in zip(recs, dm) if (r["W"] > 0) == (x > 0)) / len(recs)
            res["pooled_error"] = {
                "token_mean": sum(abs(r["p_mean"] - r["pooled_label"]) for r in recs) / len(recs),
                "time_weighted": sum(abs(r["p_time"] - r["pooled_label"]) for r in recs) / len(recs),
                "C0": sum(abs(r["p_C0"] - r["pooled_label"]) for r in recs) / len(recs),
                "replicated_baseline_token_mean": sum(abs(r["p_C0"] + r["W"] - r["pooled_label"]) for r in recs) / len(recs)}
        summary["rows"][row] = res
        print(row, status, f"mismatch={max_mis:.2e}", f"identity={max_id:.2e}")
    summary["wall_seconds"] = round(time.perf_counter() - t_wall, 3)
    summary["cpu_seconds"] = round(time.process_time() - t_cpu, 3)
    summary["threads"] = torch.get_num_threads()
    with open(os.path.join(DOUT, "summary.json"), "w") as f:
        json.dump(summary, f, indent=2)
    print(json.dumps({"wall_seconds": summary["wall_seconds"], "cpu_seconds": summary["cpu_seconds"]}))


if __name__ == "__main__":
    main()
