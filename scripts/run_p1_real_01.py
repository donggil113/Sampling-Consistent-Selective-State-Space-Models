"""P1-REAL-01 runner: native-grid vs training-grid resampling vs pooling-only, official TIDES.

Stages (configs/p1_real_01.json, amendments R-A0..R-A9):
  plan   stdlib only: protocol checks, no model.
  train  seed-0 CONDITIONAL PILOT: fixed-step Adam on the training grid; dev (calibration)
         MSE every eval_every steps; the best-dev state is saved, hashed and frozen.
  eval   loads the frozen checkpoint (hash-checked) and only then builds the test
         trajectories; paired comparison of arms on the same test trajectories.

Outputs in results/raw/: P1-REAL-01__train.{json,log}, P1-REAL-01__checkpoint.pt,
P1-REAL-01.jsonl (per trajectory x condition), P1-REAL-01__summary.json.
"""

import argparse
import hashlib
import io
import json
import os
import random
import sys
import time

ROOT = os.path.dirname(os.path.dirname(os.path.abspath(__file__)))
sys.path.insert(0, os.path.join(ROOT, "src"))

from scssm.grids import first_half_mask, split_grid  # noqa: E402
from scssm.metrics import rel_l2  # noqa: E402
from scssm.real import protocol as P  # noqa: E402
from scssm.reference import ct_true  # noqa: E402

CFG = os.path.join(ROOT, "configs", "p1_real_01.json")
FR_CFG = os.path.join(ROOT, "configs", "first_run.json")
RAW = os.path.join(ROOT, "results", "raw")
TIDES_DIR = os.path.join(ROOT, "third_party", "TIDES")
CKPT = os.path.join(RAW, "P1-REAL-01__checkpoint.pt")
TRAIN_JSON = os.path.join(RAW, "P1-REAL-01__train.json")


def load_cfg():
    with open(CFG) as f:
        cfg = json.load(f)
    with open(FR_CFG) as f:
        fr = json.load(f)
    return cfg, fr


def make_split(cfg):
    s = cfg["task"]["split"]
    n_tr = s["train"][1] - s["train"][0] + 1
    n_dv = s["dev"][1] - s["dev"][0] + 1
    n_te = s["test"][1] - s["test"][0] + 1
    return P.split_ids(n_tr, n_dv, n_te)


def labels_for(params, path, times):
    return ct_true(params, path, times, n_sub=128)["y"]


def sha256_file(p):
    with open(p, "rb") as f:
        return hashlib.sha256(f.read()).hexdigest()


def git(*args):
    import subprocess
    try:
        return subprocess.check_output(["git", *args], cwd=ROOT, text=True).strip()
    except Exception as e:  # pragma: no cover
        return f"UNAVAILABLE({e})"


# ----------------------------------------------------------------------------- plan

def stage_plan(cfg, fr):
    t0 = time.time()
    sp = make_split(cfg)
    T, K = cfg["task"]["training_grid"]["T"], cfg["task"]["training_grid"]["K"]
    overlap = set(sp.train) & set(sp.dev) | set(sp.train) & set(sp.test) | set(sp.dev) & set(sp.test)
    checks = {"n_train": len(sp.train), "n_dev": len(sp.dev), "n_test": len(sp.test), "id_overlap": len(overlap)}
    tid = sp.test[0]
    path = P.trajectory_path(tid)
    conds = test_conditions(path, T, K, tid)
    base = conds["C0"].times
    checks["base_times_common_refinements"] = all(
        [g.times[i] for i in g.base_index] == list(base) for n, g in conds.items() if n != "J1")
    checks["locf_on_refinements_equals_C0"] = all(
        list(P.resample_locf(g, base)[0]) == list(conds["C0"].values) for n, g in conds.items() if n != "J1")
    checks["seconds"] = round(time.time() - t0, 3)
    print(json.dumps(checks, indent=2))
    with open(os.path.join(RAW, "P1-REAL-01__plan_check.json"), "w") as f:
        json.dump({"status": "PLAN_ONLY: protocol checks, no model was run", "checks": checks}, f, indent=2)


def test_conditions(path, T, K, tid):
    g0 = P.training_grid(path, T, K)
    conds = {"C0": g0}
    for m in (2, 4, 8):
        conds[f"S{m}"] = split_grid(g0, m)
    conds["H8"] = split_grid(g0, 8, first_half_mask(g0))
    conds["J1"] = P.irregular_grid(path, T, K, random.Random(2_000_000 + tid))
    return conds


# ----------------------------------------------------------------------------- torch setup

def setup(cfg):
    from scssm.real import tides_adapter as TA
    torch, tides = TA.load_tides(TIDES_DIR)
    torch.set_num_threads(cfg["budget"]["cpu_threads"])
    return TA, torch, tides


def seed_all(torch, seed):
    import numpy as np
    random.seed(seed)
    np.random.seed(seed)  # TIDES init draws from numpy's global RNG
    torch.manual_seed(seed)


def data_for(ids, params, T, K):
    grids = [P.training_grid(P.trajectory_path(i), T, K) for i in ids]
    labs = [labels_for(params, P.trajectory_path(i), g.times)[1:] for i, g in zip(ids, grids)]
    return grids, labs


def mse_orig(pred_norm, lab, y_stats):
    mu, sd = y_stats
    return sum(((p * sd + mu) - v) ** 2 for p, v in zip(pred_norm, lab)) / len(lab)


# ----------------------------------------------------------------------------- train

def stage_train(cfg, fr):
    TA, torch, tides = setup(cfg)
    t_start = time.time()
    params = P.teacher(fr)
    sp = make_split(cfg)
    T, K = cfg["task"]["training_grid"]["T"], cfg["task"]["training_grid"]["K"]
    dt_train = T / K
    tr = cfg["training"]
    cap = cfg["budget"]["train_wall_clock_seconds"]
    log_lines = []

    def log(msg):
        line = f"[{time.time() - t_start:8.2f}s] {msg}"
        print(line, flush=True)
        log_lines.append(line)

    log(f"git HEAD {git('rev-parse', 'HEAD')} dirty={bool(git('status', '--porcelain'))}; torch {torch.__version__}")
    tr_grids, tr_labs = data_for(sp.train, params, T, K)
    dv_grids, dv_labs = data_for(sp.dev, params, T, K)
    log(f"labels ready: train {len(tr_grids)}, dev {len(dv_grids)}")
    x_stats = P.norm_stats([v for g in tr_grids for v in g.values])
    y_stats = P.norm_stats([v for lab in tr_labs for v in lab])
    seed_all(torch, tr["seed"])
    model = TA.build_model(torch, tides, cfg)
    opt = torch.optim.Adam(model.parameters(), lr=tr["lr"])
    rng = random.Random(tr["seed"])
    u_dv, s_dv, _ = TA.batch_from_grids(torch, dv_grids, None, dt_train, x_stats, y_stats)

    def dev_mse():
        model.eval()
        with torch.no_grad():
            pr = TA.forward(torch, model, u_dv, s_dv).tolist()
        model.train()
        return sum(mse_orig(p, lab, y_stats) for p, lab in zip(pr, dv_labs)) / len(dv_labs)

    curve = [{"step": 0, "dev_mse": dev_mse()}]
    best = {"step": 0, "dev_mse": curve[0]["dev_mse"], "state": {k: v.clone() for k, v in model.state_dict().items()}}
    model.train()
    t_train = time.time()
    status = "COMPLETED"
    for step in range(1, tr["steps"] + 1):
        idx = [rng.randrange(len(tr_grids)) for _ in range(tr["batch_size"])]
        u, s, y = TA.batch_from_grids(torch, [tr_grids[i] for i in idx], [tr_labs[i] for i in idx],
                                      dt_train, x_stats, y_stats)
        loss = torch.mean((TA.forward(torch, model, u, s) - y) ** 2)
        opt.zero_grad()
        loss.backward()
        opt.step()
        if step % tr["eval_every"] == 0:
            dm = dev_mse()
            curve.append({"step": step, "train_loss_norm": float(loss.detach()), "dev_mse": dm})
            if dm < best["dev_mse"]:
                best = {"step": step, "dev_mse": dm, "state": {k: v.clone() for k, v in model.state_dict().items()}}
            log(f"step {step} train_loss(norm) {float(loss.detach()):.5f} dev_mse {dm:.6f} best@{best['step']}")
        if time.time() - t_start > cap:
            status = "NOT_RUN(TIMEOUT)"
            log(f"ABORT at step {step}: cap {cap}s exceeded; no checkpoint is frozen")
            break
    train_seconds = time.time() - t_train
    out = {"status": status, "pilot": "CONDITIONAL PILOT: seed 0 only (R-A6)", "torch": torch.__version__,
           "threads": torch.get_num_threads(), "curve": curve, "x_stats": x_stats, "y_stats": y_stats,
           "train_loop_seconds": round(train_seconds, 2), "total_seconds": round(time.time() - t_start, 2),
           "git_head": git("rev-parse", "HEAD")}
    if status == "COMPLETED":
        buf = io.BytesIO()
        torch.save(best["state"], buf)
        with open(CKPT, "wb") as f:
            f.write(buf.getvalue())
        out.update({"selected_step": best["step"], "selected_dev_mse": best["dev_mse"],
                    "checkpoint": os.path.relpath(CKPT, ROOT), "checkpoint_sha256": sha256_file(CKPT)})
        log(f"FROZEN checkpoint step {best['step']} dev_mse {best['dev_mse']:.6f} sha256 {out['checkpoint_sha256'][:16]}")
    with open(TRAIN_JSON, "w") as f:
        json.dump(out, f, indent=2)
    with open(os.path.join(RAW, "P1-REAL-01__train.log"), "w") as f:
        f.write("\n".join(log_lines) + "\n")


# ----------------------------------------------------------------------------- eval

_BOOT = {}


def bootstrap_ci(values, n_boot=2000, seed=0):
    """95% percentile CI of the mean; the same resampled trajectory indices are reused for
    every metric of the same length (paired resampling across metrics and arms)."""
    n = len(values)
    if (n, n_boot, seed) not in _BOOT:
        rng = random.Random(seed)
        _BOOT[(n, n_boot, seed)] = [[rng.randrange(n) for _ in range(n)] for _ in range(n_boot)]
    means = sorted(sum(values[j] for j in idx) / n for idx in _BOOT[(n, n_boot, seed)])
    return [means[int(0.025 * n_boot)], means[int(0.975 * n_boot) - 1]]


def mean(v):
    return sum(v) / len(v)


def stage_eval(cfg, fr):
    TA, torch, tides = setup(cfg)
    t_start = time.time()
    with open(TRAIN_JSON) as f:
        trn = json.load(f)
    if trn.get("status") != "COMPLETED":
        raise SystemExit("NOT_RUN: training did not complete; untrained evaluation is not a trained-model result")
    if sha256_file(CKPT) != trn["checkpoint_sha256"]:
        raise SystemExit("checkpoint hash mismatch: the frozen checkpoint was modified")
    params = P.teacher(fr)
    sp = make_split(cfg)
    T, K = cfg["task"]["training_grid"]["T"], cfg["task"]["training_grid"]["K"]
    dt_train = T / K
    x_stats, y_stats = tuple(trn["x_stats"]), tuple(trn["y_stats"])
    seed_all(torch, 0)
    model = TA.build_model(torch, tides, cfg)
    model.load_state_dict(torch.load(CKPT))
    model.eval()
    # test trajectories are generated only now, after the checkpoint is frozen
    tids = list(sp.test)
    paths = {i: P.trajectory_path(i) for i in tids}
    conds = {i: test_conditions(paths[i], T, K, i) for i in tids}
    base = conds[tids[0]]["C0"].times
    truth = {i: ct_true(params, paths[i], base, n_sub=128) for i in tids}
    labels = {i: truth[i]["y"][1:] for i in tids}
    pooled_label = {i: truth[i]["integral"] / T for i in tids}
    names = ["C0", "S2", "S4", "S8", "H8", "J1"]
    preds, runtime = {}, {}
    mu, sd = y_stats
    for name in names:
        g_list = [conds[i][name] for i in tids]
        # native: the condition's grid with physical step sizes
        u, s, _ = TA.batch_from_grids(torch, g_list, None, dt_train, x_stats, y_stats)
        out, secs = TA.predict(torch, model, u, s)
        preds[(name, "native")] = [[p * sd + mu for p in row] for row in out.tolist()]
        runtime[(name, "native")] = {"seconds_batch": secs, "tokens_per_sequence": len(g_list[0].values)}
        # resampled_locf: observed values only (R-A3), training grid, unit steps
        feed = [P.Grid(tuple(base), tuple(P.resample_locf(g, base)[0]), tuple(range(len(base)))) for g in g_list]
        u, s, _ = TA.batch_from_grids(torch, feed, None, dt_train, x_stats, y_stats)
        out, secs = TA.predict(torch, model, u, s)
        preds[(name, "resampled_locf")] = [[p * sd + mu for p in row] for row in out.tolist()]
        runtime[(name, "resampled_locf")] = {"seconds_batch": secs, "tokens_per_sequence": len(base) - 1}
    recs = []
    per = {}
    for n_i, i in enumerate(tids):
        for name in names:
            g = conds[i][name]
            yn = preds[(name, "native")][n_i]
            yr = preds[(name, "resampled_locf")][n_i]
            if name == "J1":
                at_base = []
                for t in base[1:]:
                    j = max([k for k in range(1, len(g.times)) if g.times[k] <= t + 1e-12] or [1])
                    at_base.append(yn[j - 1])
            else:
                at_base = [yn[k - 1] for k in g.base_index[1:]]
            dts = [g.times[k] - g.times[k - 1] for k in range(1, len(g.times))]
            per[(i, name)] = {
                "native_base": at_base, "resampled_base": yr,
                "pool_native_mean": sum(yn) / len(yn),
                "pool_native_time": sum(y * d for y, d in zip(yn, dts)) / T,
                "pool_resampled_mean": sum(yr) / len(yr),
            }
        c0 = per[(i, "C0")]
        for name in names:
            r = per[(i, name)]
            lab = labels[i]
            rec = {"traj": i, "condition": name,
                   "mse_native": sum((a - b) ** 2 for a, b in zip(r["native_base"], lab)) / len(lab),
                   "mse_resampled": sum((a - b) ** 2 for a, b in zip(r["resampled_base"], lab)) / len(lab),
                   "disc_native": rel_l2(r["native_base"], c0["native_base"]),
                   "disc_resampled": rel_l2(r["resampled_base"], c0["resampled_base"]),
                   "pooled_label": pooled_label[i]}
            for key in ("pool_native_mean", "pool_native_time", "pool_resampled_mean"):
                rec[key] = r[key]
                rec[key + "_abs_err"] = abs(r[key] - pooled_label[i])
                rec[key + "_abs_change_vs_C0"] = abs(r[key] - c0[key])
            recs.append(rec)
    with open(os.path.join(RAW, "P1-REAL-01.jsonl"), "w") as f:
        for r in recs:
            f.write(json.dumps(r) + "\n")

    summary = {"status": "CONDITIONAL PILOT (seed 0; R-A6)", "checkpoint_sha256": trn["checkpoint_sha256"],
               "selected_step": trn["selected_step"], "n_test": len(tids), "conditions": {}}
    for name in names:
        rs = [r for r in recs if r["condition"] == name]
        diff = [r["mse_native"] - r["mse_resampled"] for r in rs]
        m_res = mean([r["mse_resampled"] for r in rs])
        delta = 0.05 * m_res
        ci = bootstrap_ci(diff)
        if -delta < ci[0] and ci[1] < delta:
            verdict = "NO PRACTICAL DIFFERENCE (CI inside +-5% margin)"
        elif ci[0] > 0:
            verdict = "NATIVE WORSE (CI > 0)" + ("" if ci[0] > delta else "; not shown beyond margin")
        elif ci[1] < 0:
            verdict = "NATIVE BETTER (CI < 0)" + ("" if ci[1] < -delta else "; not shown beyond margin")
        else:
            verdict = "INCONCLUSIVE (CI includes 0 and exceeds the margin)"
        c = {
            "mse_native_mean": mean([r["mse_native"] for r in rs]), "mse_native_ci": bootstrap_ci([r["mse_native"] for r in rs]),
            "mse_resampled_mean": m_res, "mse_resampled_ci": bootstrap_ci([r["mse_resampled"] for r in rs]),
            "paired_diff_mean": mean(diff), "paired_diff_ci": ci, "margin_delta": delta, "equivalence_verdict": verdict,
            "disc_native_mean": mean([r["disc_native"] for r in rs]), "disc_native_ci": bootstrap_ci([r["disc_native"] for r in rs]),
            "disc_native_max": max(r["disc_native"] for r in rs),
            "disc_resampled_mean": mean([r["disc_resampled"] for r in rs]), "disc_resampled_max": max(r["disc_resampled"] for r in rs),
            "runtime_native": runtime[(name, "native")], "runtime_resampled": runtime[(name, "resampled_locf")],
        }
        for key in ("pool_native_mean", "pool_native_time", "pool_resampled_mean"):
            c[key + "_abs_err_mean"] = mean([r[key + "_abs_err"] for r in rs])
            c[key + "_abs_err_ci"] = bootstrap_ci([r[key + "_abs_err"] for r in rs])
            c[key + "_abs_change_mean"] = mean([r[key + "_abs_change_vs_C0"] for r in rs])
            c[key + "_abs_change_ci"] = bootstrap_ci([r[key + "_abs_change_vs_C0"] for r in rs])
        pdiff = [r["pool_native_mean_abs_change_vs_C0"] - r["pool_native_time_abs_change_vs_C0"] for r in rs]
        c["pool_mean_minus_time_change_mean"] = mean(pdiff)
        c["pool_mean_minus_time_change_ci"] = bootstrap_ci(pdiff)
        summary["conditions"][name] = c
    s8 = summary["conditions"]["S8"]
    h8 = summary["conditions"]["H8"]
    summary["predictions"] = {
        "R-P1": {"statement": "native split discrepancy at S8 exceeds 1e-4",
                 "mean": s8["disc_native_mean"], "ci": s8["disc_native_ci"],
                 "status": "SUPPORTED" if s8["disc_native_ci"][0] > 1e-4 else "NOT SUPPORTED"},
        "R-P2": {"statement": "resampled split discrepancy at S2/S4/S8 <= 1e-6",
                 "max": max(summary["conditions"][n]["disc_resampled_max"] for n in ("S2", "S4", "S8")),
                 "status": "SUPPORTED" if max(summary["conditions"][n]["disc_resampled_max"] for n in ("S2", "S4", "S8")) <= 1e-6 else "NOT SUPPORTED"},
        "R-P4": {"statement": "native runtime grows with tokens; resampled constant in m",
                 "native_seconds": {n: summary["conditions"][n]["runtime_native"]["seconds_batch"] for n in ("C0", "S2", "S4", "S8")},
                 "resampled_seconds": {n: summary["conditions"][n]["runtime_resampled"]["seconds_batch"] for n in ("C0", "S2", "S4", "S8")},
                 "status": "DESCRIPTIVE (single timing run per cell; median of 3 calls)"},
        "R-P5": {"statement": "on H8, sample-mean pooled change exceeds time-weighted pooled change",
                 "mean": h8["pool_mean_minus_time_change_mean"], "ci": h8["pool_mean_minus_time_change_ci"],
                 "status": "SUPPORTED" if h8["pool_mean_minus_time_change_ci"][0] > 0 else "NOT SUPPORTED"},
    }
    summary["eval_seconds"] = round(time.time() - t_start, 2)
    with open(os.path.join(RAW, "P1-REAL-01__summary.json"), "w") as f:
        json.dump(summary, f, indent=2)
    print(json.dumps({k: v for k, v in summary.items() if k != "conditions"}, indent=2))
    for n in names:
        c = summary["conditions"][n]
        print(n, f"mse nat {c['mse_native_mean']:.4g} res {c['mse_resampled_mean']:.4g} diff {c['paired_diff_mean']:.3g} "
                 f"CI {c['paired_diff_ci'][0]:.3g},{c['paired_diff_ci'][1]:.3g} | {c['equivalence_verdict']} | "
                 f"disc nat {c['disc_native_mean']:.3g} res {c['disc_resampled_mean']:.3g}")


def main():
    ap = argparse.ArgumentParser()
    ap.add_argument("--stage", choices=["plan", "train", "eval"], required=True)
    a = ap.parse_args()
    cfg, fr = load_cfg()
    if a.stage == "plan":
        stage_plan(cfg, fr)
    elif a.stage == "train":
        stage_train(cfg, fr)
    else:
        stage_eval(cfg, fr)


if __name__ == "__main__":
    main()
