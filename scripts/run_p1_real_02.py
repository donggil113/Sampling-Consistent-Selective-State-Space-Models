"""P1-REAL-02: fixed-protocol seed repetition (configs/p1_real_02.json).

Stages (run in this order; each writes under results/raw/P1-REAL-02/):
  data       generate train/dev inputs+labels once, hash, compare stats with the seed-0 pilot
  eqcheck    retrain seed 0 with this runner; compare tensors with the frozen pilot checkpoint
  train      --seed s  (s in 1..4): identical protocol to the pilot, calibration-selected checkpoint
  testset    generate the NEW shared test set (ids 10000-10255) and its labels, hash it
  eval       --seed s | --dev : paired native / resampled / pooling-only evaluation, timing, checks
  aggregate  per-seed table, cross-seed range, replication verdicts; seed-0 pilot re-summary (derived)
"""

import argparse
import copy
import hashlib
import io
import json
import os
import random
import sys
import time

ROOT = os.path.dirname(os.path.dirname(os.path.abspath(__file__)))
sys.path.insert(0, os.path.join(ROOT, "src"))
sys.path.insert(0, os.path.join(ROOT, "scripts"))

from run_p1_real_01 import (  # noqa: E402  (reuse the pilot's helpers unchanged)
    RAW, data_for, git, labels_for, load_cfg, make_split, mse_orig, seed_all, setup, sha256_file, test_conditions)
from scssm.metrics import rel_l2  # noqa: E402
from scssm.real import protocol as P  # noqa: E402
from scssm.real import stats as S  # noqa: E402
from scssm.reference import ct_true  # noqa: E402

CFG2 = os.path.join(ROOT, "configs", "p1_real_02.json")
OUT = os.path.join(RAW, "P1-REAL-02")
PILOT_TRAIN = os.path.join(RAW, "P1-REAL-01__train.json")
PILOT_CKPT = os.path.join(RAW, "P1-REAL-01__checkpoint.pt")
CONDS = ["C0", "S2", "S4", "S8", "H8", "J1"]


def cfg2():
    with open(CFG2) as f:
        return json.load(f)


def clock():
    return time.perf_counter(), time.process_time()


def since(t):
    return round(time.perf_counter() - t[0], 3), round(time.process_time() - t[1], 3)


def canon_hash(obj):
    return hashlib.sha256(json.dumps(obj, sort_keys=True).encode()).hexdigest()


def wjson(path, obj):
    os.makedirs(os.path.dirname(path), exist_ok=True)
    with open(path, "w") as f:
        json.dump(obj, f, indent=2)


# ----------------------------------------------------------------------------- data

def stage_data():
    cfg, fr = load_cfg()
    t = clock()
    params = P.teacher(fr)
    sp = make_split(cfg)
    T, K = cfg["task"]["training_grid"]["T"], cfg["task"]["training_grid"]["K"]
    tr_g, tr_l = data_for(sp.train, params, T, K)
    dv_g, dv_l = data_for(sp.dev, params, T, K)
    x_stats = P.norm_stats([v for g in tr_g for v in g.values])
    y_stats = P.norm_stats([v for lab in tr_l for v in lab])
    pilot = json.load(open(PILOT_TRAIN))
    cache = {
        "train": [{"id": i, "times": list(g.times), "values": list(g.values), "labels": list(lab)}
                  for i, g, lab in zip(sp.train, tr_g, tr_l)],
        "dev": [{"id": i, "times": list(g.times), "values": list(g.values), "labels": list(lab)}
                for i, g, lab in zip(sp.dev, dv_g, dv_l)],
    }
    h = canon_hash(cache)
    wjson(os.path.join(OUT, "data_cache.json"), cache)
    wall, cpu = since(t)
    rep = {"sha256": h, "x_stats": list(x_stats), "y_stats": list(y_stats),
           "equal_to_pilot_stats": list(x_stats) == pilot["x_stats"] and list(y_stats) == pilot["y_stats"],
           "n_train": len(tr_g), "n_dev": len(dv_g), "wall_seconds": wall, "cpu_seconds": cpu}
    wjson(os.path.join(OUT, "data_hash.json"), rep)
    print(json.dumps(rep, indent=2))


def load_data():
    rep = json.load(open(os.path.join(OUT, "data_hash.json")))
    cache = json.load(open(os.path.join(OUT, "data_cache.json")))
    if canon_hash(cache) != rep["sha256"]:
        raise SystemExit("data cache hash mismatch")
    if not rep["equal_to_pilot_stats"]:
        raise SystemExit("data stats differ from the seed-0 pilot; protocol not identical")

    def grids(rows):
        return ([P.Grid(tuple(r["times"]), tuple(r["values"]), tuple(range(len(r["times"])))) for r in rows],
                [r["labels"] for r in rows])
    tr_g, tr_l = grids(cache["train"])
    dv_g, dv_l = grids(cache["dev"])
    return tr_g, tr_l, dv_g, dv_l, tuple(rep["x_stats"]), tuple(rep["y_stats"]), rep["sha256"]


# ----------------------------------------------------------------------------- checks

def eval_mode_checks(torch, model):
    mods = list(model.modules())
    bns = [m for m in mods if isinstance(m, torch.nn.modules.batchnorm._BatchNorm)]
    dps = [m for m in mods if isinstance(m, torch.nn.Dropout)]
    return {"all_modules_eval": all(not m.training for m in mods),
            "dropout_p": sorted({float(m.p) for m in dps}),
            "batchnorm_running_stats": all(m.track_running_stats and m.running_mean is not None for m in bns),
            "n_batchnorm": len(bns)}


# ----------------------------------------------------------------------------- train (identical loop to the pilot)

def train_one(seed, out_dir):
    cfg, fr = load_cfg()
    c2 = cfg2()
    TA, torch, tides = setup(cfg)
    t_all = clock()
    tr_g, tr_l, dv_g, dv_l, x_stats, y_stats, dhash = load_data()
    T, K = cfg["task"]["training_grid"]["T"], cfg["task"]["training_grid"]["K"]
    dt_train = T / K
    tr = cfg["training"]
    cap = c2["budget"]["train_wall_clock_seconds_per_seed"]
    log_lines = []

    def log(msg):
        line = f"[{time.perf_counter() - t_all[0]:8.2f}s] {msg}"
        print(line, flush=True)
        log_lines.append(line)

    log(f"seed {seed}; git HEAD {git('rev-parse', 'HEAD')} dirty={bool(git('status', '--porcelain'))}; data {dhash[:16]}")
    seed_all(torch, seed)
    model = TA.build_model(torch, tides, cfg)
    opt = torch.optim.Adam(model.parameters(), lr=tr["lr"])
    rng = random.Random(seed)
    u_dv, s_dv, _ = TA.batch_from_grids(torch, dv_g, None, dt_train, x_stats, y_stats)
    checks = []

    def dev_mse():
        model.eval()
        chk = eval_mode_checks(torch, model)
        with torch.no_grad():
            pr = TA.forward(torch, model, u_dv, s_dv).tolist()
        model.train()
        checks.append(chk)
        return sum(mse_orig(p, lab, y_stats) for p, lab in zip(pr, dv_l)) / len(dv_l)

    curve = [{"step": 0, "dev_mse": dev_mse()}]
    best = {"step": 0, "dev_mse": curve[0]["dev_mse"], "state": {k: v.clone() for k, v in model.state_dict().items()}}
    model.train()
    t_loop = clock()
    status = "COMPLETED"
    min_step = float("inf")
    for step in range(1, tr["steps"] + 1):
        idx = [rng.randrange(len(tr_g)) for _ in range(tr["batch_size"])]
        u, s, y = TA.batch_from_grids(torch, [tr_g[i] for i in idx], [tr_l[i] for i in idx], dt_train, x_stats, y_stats)
        min_step = min(min_step, float(s.min()))
        loss = torch.mean((TA.forward(torch, model, u, s) - y) ** 2)
        if not torch.isfinite(loss):
            status = f"FAIL(non-finite loss at step {step})"
            log(status)
            break
        opt.zero_grad()
        loss.backward()
        opt.step()
        if step % tr["eval_every"] == 0:
            dm = dev_mse()
            curve.append({"step": step, "train_loss_norm": float(loss.detach()), "dev_mse": dm})
            if dm < best["dev_mse"]:
                best = {"step": step, "dev_mse": dm, "state": {k: v.clone() for k, v in model.state_dict().items()}}
            log(f"step {step} train_loss(norm) {float(loss.detach()):.5f} dev_mse {dm:.6f} best@{best['step']}")
        if time.perf_counter() - t_all[0] > cap:
            status = "NOT_RUN(TIMEOUT)"
            log(f"ABORT at step {step}: cap {cap}s exceeded")
            break
    loop_wall, loop_cpu = since(t_loop)
    ckpt = os.path.join(out_dir, "checkpoint.pt")
    out = {"seed": seed, "status": status, "data_sha256": dhash, "torch": torch.__version__,
           "threads": torch.get_num_threads(), "curve": curve, "x_stats": list(x_stats), "y_stats": list(y_stats),
           "eval_mode_checks_all_ok": all(c["all_modules_eval"] and c["dropout_p"] in ([0.0], []) and c["batchnorm_running_stats"]
                                          for c in checks),
           "eval_mode_check_example": checks[-1], "min_train_step_scale": min_step,
           "train_loop_wall_seconds": loop_wall, "train_loop_cpu_seconds": loop_cpu, "git_head": git("rev-parse", "HEAD")}
    os.makedirs(out_dir, exist_ok=True)
    if status == "COMPLETED":
        buf = io.BytesIO()
        torch.save(best["state"], buf)
        with open(ckpt, "wb") as f:
            f.write(buf.getvalue())
        out.update({"selected_step": best["step"], "selected_dev_mse": best["dev_mse"],
                    "checkpoint_sha256": sha256_file(ckpt)})
        log(f"FROZEN seed {seed} step {best['step']} dev_mse {best['dev_mse']:.6f}")
    tw, tc = since(t_all)
    out["total_wall_seconds"], out["total_cpu_seconds"] = tw, tc
    wjson(os.path.join(out_dir, "train.json"), out)
    with open(os.path.join(out_dir, "train.log"), "w") as f:
        f.write("\n".join(log_lines) + "\n")
    return out, best


def stage_eqcheck():
    cfg, _ = load_cfg()
    _, torch, _ = setup(cfg)
    out, best = train_one(0, os.path.join(OUT, "eqcheck_seed0"))
    pilot = torch.load(PILOT_CKPT)
    ours = torch.load(os.path.join(OUT, "eqcheck_seed0", "checkpoint.pt"))
    diffs = {k: float((pilot[k].double() - ours[k].double()).abs().max()) if pilot[k].numel() else 0.0 for k in pilot}
    pj = json.load(open(PILOT_TRAIN))
    rep = {"what": "engineering check: seed 0 retrained with the P1-REAL-02 runner vs the frozen pilot checkpoint",
           "same_keys": sorted(pilot) == sorted(ours), "max_abs_tensor_diff": max(diffs.values()),
           "all_tensors_equal": all(v == 0.0 for v in diffs.values()),
           "selected_step_runner": out.get("selected_step"), "selected_step_pilot": pj["selected_step"],
           "dev_curve_equal": [c["dev_mse"] for c in out["curve"]] == [c["dev_mse"] for c in pj["curve"]]}
    wjson(os.path.join(OUT, "eqcheck.json"), rep)
    print(json.dumps(rep, indent=2))


# ----------------------------------------------------------------------------- test set

def test_ids():
    lo, hi = cfg2()["test_set"]["ids"]
    return list(range(lo, hi + 1))


def stage_testset():
    cfg, fr = load_cfg()
    t = clock()
    params = P.teacher(fr)
    T, K = cfg["task"]["training_grid"]["T"], cfg["task"]["training_grid"]["K"]
    rows = []
    for i in test_ids():
        path = P.trajectory_path(i)
        conds = test_conditions(path, T, K, i)
        tr = ct_true(params, path, conds["C0"].times, n_sub=128)
        rows.append({"id": i, "labels": tr["y"][1:], "pooled_label": tr["integral"] / T,
                     "grids_sha256": canon_hash({n: [list(g.times), list(g.values)] for n, g in conds.items()})})
    h = canon_hash(rows)
    wjson(os.path.join(OUT, "testset_cache.json"), {"sha256": h, "rows": rows})
    wall, cpu = since(t)
    rep = {"sha256": h, "n": len(rows), "ids": [rows[0]["id"], rows[-1]["id"]], "wall_seconds": wall, "cpu_seconds": cpu}
    wjson(os.path.join(OUT, "testset_hash.json"), rep)
    print(json.dumps(rep, indent=2))


def load_testset():
    d = json.load(open(os.path.join(OUT, "testset_cache.json")))
    if canon_hash(d["rows"]) != d["sha256"]:
        raise SystemExit("test set hash mismatch")
    return d["rows"], d["sha256"]


# ----------------------------------------------------------------------------- eval

def stage_eval(seed, dev=False):
    cfg, _ = load_cfg()
    c2 = cfg2()
    TA, torch, tides = setup(cfg)
    t_all = clock()
    if dev:
        tj = json.load(open(PILOT_TRAIN))
        ckpt, label, out_dir = PILOT_CKPT, "development (seed 0 pilot checkpoint)", os.path.join(OUT, "dev_seed0")
    else:
        out_dir = os.path.join(OUT, f"seed{seed}")
        tj = json.load(open(os.path.join(out_dir, "train.json")))
        ckpt, label = os.path.join(out_dir, "checkpoint.pt"), f"training seed {seed}"
    if tj["status"] != "COMPLETED":
        wjson(os.path.join(out_dir, "eval_summary.json"), {"seed": seed, "status": f"NOT_EVALUATED ({tj['status']})"})
        return
    if sha256_file(ckpt) != tj["checkpoint_sha256"]:
        raise SystemExit("checkpoint hash mismatch")
    rows, thash = load_testset()
    T, K = cfg["task"]["training_grid"]["T"], cfg["task"]["training_grid"]["K"]
    dt_train = T / K
    x_stats, y_stats = tuple(tj["x_stats"]), tuple(tj["y_stats"])
    mu, sd = y_stats
    seed_all(torch, 0)
    model = TA.build_model(torch, tides, cfg)
    model.load_state_dict(torch.load(ckpt))
    model.eval()
    checks = eval_mode_checks(torch, model)
    m64 = copy.deepcopy(model).double().eval()
    ids = [r["id"] for r in rows]
    conds = {i: test_conditions(P.trajectory_path(i), T, K, i) for i in ids}
    base = conds[ids[0]]["C0"].times
    n = len(ids)

    def readout_native(g, pred):
        if all(g.times[i] == b for i, b in zip(g.base_index, base)) and len(g.base_index) == len(base):
            return [pred[k - 1] for k in g.base_index[1:]]
        out = []
        for t in base[1:]:
            j = max([k for k in range(1, len(g.times)) if g.times[k] <= t + 1e-12] or [1])
            out.append(pred[j - 1])
        return out

    def run_native(g_list, dtype=None):
        u, s, _ = TA.batch_from_grids(torch, g_list, None, dt_train, x_stats, y_stats)
        mdl = model
        if dtype == "float64":
            u, s, mdl = u.double(), s.double(), m64
        out = TA.forward(torch, mdl, u, s).tolist()
        return [[p * sd + mu for p in row] for row in out], float(s.min()), u.shape

    def run_resampled(g_list):
        feed = [P.Grid(tuple(base), tuple(P.resample_locf(g, base)[0]), tuple(range(len(base)))) for g in g_list]
        u, s, _ = TA.batch_from_grids(torch, feed, None, dt_train, x_stats, y_stats)
        out = TA.forward(torch, model, u, s).tolist()
        return [[p * sd + mu for p in row] for row in out], u.shape

    tm = c2["timing_protocol"]
    per = {}
    timing = {}
    shape_info = {}
    with torch.inference_mode():
        for name in CONDS:
            g_list = [conds[i][name] for i in ids]
            lens = {len(g.values) for g in g_list}
            assert len(lens) == 1, "unequal sequence lengths in a batch (no padding/mask is used)"
            pred_n, min_step, shp = run_native(g_list)
            pred_r, shp_r = run_resampled(g_list)
            shape_info[name] = {"native_batch_len": list(shp), "resampled_batch_len": list(shp_r), "min_step_scale_native": min_step}
            per[name] = {"native": pred_n, "resampled": pred_r}
            if name in ("C0", "S8"):
                per[name]["native64"] = run_native(g_list, "float64")[0]
            # timing (end-to-end and forward-only), warm-up then repeats; not independent samples
            u_n, s_n, _ = TA.batch_from_grids(torch, g_list, None, dt_train, x_stats, y_stats)
            feed = [P.Grid(tuple(base), tuple(P.resample_locf(g, base)[0]), tuple(range(len(base)))) for g in g_list]
            u_r, s_r, _ = TA.batch_from_grids(torch, feed, None, dt_train, x_stats, y_stats)
            jobs = {
                "native_end_to_end": lambda: [readout_native(g, p) for g, p in zip(g_list, run_native(g_list)[0])],
                "resampled_end_to_end": lambda: run_resampled(g_list)[0],
                "native_forward_only": lambda: TA.forward(torch, model, u_n, s_n),
                "resampled_forward_only": lambda: TA.forward(torch, model, u_r, s_r),
            }
            timing[name] = {}
            for jn, fn in jobs.items():
                for _ in range(tm["warm_up_calls"]):
                    fn()
                walls, cpus = [], []
                for _ in range(tm["timed_repeats"]):
                    t = clock()
                    fn()
                    w, c = since(t)
                    walls.append(w)
                    cpus.append(c)
                walls.sort()
                cpus.sort()
                timing[name][jn] = {"median_wall_s": walls[len(walls) // 2], "min_wall_s": walls[0],
                                    "median_cpu_s": cpus[len(cpus) // 2]}
            timing[name]["tokens_native"] = shp[1]
            timing[name]["tokens_resampled"] = shp_r[1]
    # per-trajectory records
    recs = []
    for n_i, i in enumerate(ids):
        lab = rows[n_i]["labels"]
        yb = rows[n_i]["pooled_label"]
        g0 = conds[i]["C0"]
        c0n = readout_native(g0, per["C0"]["native"][n_i])
        p_c0 = sum(per["C0"]["native"][n_i]) / len(per["C0"]["native"][n_i])
        for name in CONDS:
            g = conds[i][name]
            pn = per[name]["native"][n_i]
            pr = per[name]["resampled"][n_i]
            nb = readout_native(g, pn)
            dts = [g.times[k] - g.times[k - 1] for k in range(1, len(g.times))]
            p_mean = sum(pn) / len(pn)
            p_time = sum(y * d for y, d in zip(pn, dts)) / T
            p_res = sum(pr) / len(pr)
            rec = {"id": i, "condition": name,
                   "loss_native": sum((a - b) ** 2 for a, b in zip(nb, lab)) / len(lab),
                   "loss_resampled": sum((a - b) ** 2 for a, b in zip(pr, lab)) / len(lab),
                   "disc_native": rel_l2(nb, c0n),
                   "disc_resampled": rel_l2(pr, per["C0"]["resampled"][n_i]),
                   "p_mean": p_mean, "p_time": p_time, "p_res": p_res, "p_C0": p_c0, "pooled_label": yb}
            if name == "S8":
                c064 = readout_native(g0, per["C0"]["native64"][n_i])
                rec["disc_native_float64"] = rel_l2(readout_native(g, per["S8"]["native64"][n_i]), c064)
            recs.append(rec)
    os.makedirs(out_dir, exist_ok=True)
    with open(os.path.join(out_dir, "eval.jsonl"), "w") as f:
        for r in recs:
            f.write(json.dumps(r) + "\n")
    summary = summarize(recs, n)
    ew, ec = since(t_all)
    summary.update({"seed": seed, "label": label, "status": "COMPLETED", "checkpoint_sha256": tj["checkpoint_sha256"],
                    "selected_step": tj["selected_step"], "selected_dev_mse": tj["selected_dev_mse"],
                    "testset_sha256": thash, "eval_mode_checks": checks, "batch_shapes": shape_info, "timing": timing,
                    "threads": torch.get_num_threads(), "eval_wall_seconds": ew, "eval_cpu_seconds": ec})
    wjson(os.path.join(out_dir, "eval_summary.json"), summary)
    print(json.dumps({k: summary[k] for k in ("seed", "primary_D", "S8_disc_float32", "S8_disc_float64")}, indent=2))


def summarize(recs, n):
    idx = S.draws(n, 2000, 0)
    by = {c: [r for r in recs if r["condition"] == c] for c in CONDS}
    out = {}
    h8 = by["H8"]
    d = [abs(r["p_mean"] - r["p_C0"]) - abs(r["p_time"] - r["p_C0"]) for r in h8]
    est, ci = S.mean_ci(d, idx)
    out["primary_D"] = {"estimate": est, "ci95": ci, "units": "label units"}
    for key in ("disc_native", "disc_native_float64"):
        v = [r[key] for r in by["S8"]]
        e, c = S.mean_ci(v, idx)
        out["S8_disc_" + ("float64" if key.endswith("64") else "float32")] = {"estimate": e, "ci95": c, "max": max(v)}
    out["disc_resampled_max_refinements"] = max(r["disc_resampled"] for c in ("S2", "S4", "S8", "H8") for r in by[c])
    out["loss"] = {}
    for c in CONDS:
        ln = [r["loss_native"] for r in by[c]]
        lr = [r["loss_resampled"] for r in by[c]]
        r_, rci, ad, adci = S.ratio_ci(ln, lr, idx)
        out["loss"][c] = {"mean_native": sum(ln) / n, "mean_resampled": sum(lr) / n, "r": r_, "r_ci95": rci,
                          "abs_diff": ad, "abs_diff_ci95": adci, "verdict": S.margin_verdict(rci) if c != "C0" else "identical inputs (check)"}
        out["loss"][c]["disc_native_mean"] = sum(x["disc_native"] for x in by[c]) / n
    out["pooled"] = {}
    for c in ("C0", "S8", "H8", "J1"):
        out["pooled"][c] = {arm: S.mean_ci([abs(r[key] - r["pooled_label"]) for r in by[c]], idx)[0]
                            for arm, key in (("mean", "p_mean"), ("time", "p_time"), ("resampled", "p_res"))}
        out["pooled"][c]["change_mean_vs_C0"] = sum(abs(r["p_mean"] - r["p_C0"]) for r in by[c]) / n
        out["pooled"][c]["change_time_vs_C0"] = sum(abs(r["p_time"] - r["p_C0"]) for r in by[c]) / n
    e_err = [abs(r["p_mean"] - r["pooled_label"]) - abs(r["p_time"] - r["pooled_label"]) for r in h8]
    est, ci = S.mean_ci(e_err, idx)
    out["E_H8_error_diff"] = {"estimate": est, "ci95": ci, "units": "label units"}
    return out


# ----------------------------------------------------------------------------- aggregate

def stage_aggregate():
    c2 = cfg2()
    seeds = c2["seeds"]["training_seeds"]
    rows = {}
    for s in seeds:
        p = os.path.join(OUT, f"seed{s}", "eval_summary.json")
        rows[s] = json.load(open(p)) if os.path.exists(p) else {"seed": s, "status": "NOT_RUN"}
    dev = os.path.join(OUT, "dev_seed0", "eval_summary.json")
    dev_row = json.load(open(dev)) if os.path.exists(dev) else {"status": "NOT_RUN"}
    ok = [s for s in seeds if rows[s].get("status") == "COMPLETED"]

    def rng(vals):
        return {"min": min(vals), "max": max(vals), "mean": sum(vals) / len(vals), "n_seeds": len(vals)} if vals else None

    agg = {
        "status": f"{len(ok)} of {len(seeds)} training seeds completed",
        "primary_D_per_seed": {s: rows[s]["primary_D"] for s in ok},
        "primary_D_across_seeds": rng([rows[s]["primary_D"]["estimate"] for s in ok]),
        "primary_replicated_within_task": (len(ok) == len(seeds) and all(rows[s]["primary_D"]["ci95"][0] > 0 for s in ok)),
        "S8_disc_per_seed": {s: {"float32": rows[s]["S8_disc_float32"], "float64": rows[s]["S8_disc_float64"]} for s in ok},
        "S8_above_float32_floor_all_seeds": all(rows[s]["S8_disc_float32"]["ci95"][0] > 1e-4 for s in ok) if ok else None,
        "loss_contrast_per_seed": {s: {c: rows[s]["loss"][c] for c in ("S2", "S4", "S8", "H8", "J1")} for s in ok},
        "development_row_seed0_new_testset": dev_row if dev_row.get("status") == "COMPLETED" else dev_row,
        "note": "crossed design: all rows share the same 256 test trajectories; no pooled CI across seeds",
    }
    # derived re-summary of the seed-0 pilot on its OLD test set with the r_s estimator (exploratory, derived file)
    pil = [json.loads(line) for line in open(os.path.join(RAW, "P1-REAL-01.jsonl"))]
    idx = S.draws(256, 2000, 0)
    pil_r = {}
    for c in ("S2", "S4", "S8", "H8", "J1"):
        rs = [r for r in pil if r["condition"] == c]
        r_, rci, ad, adci = S.ratio_ci([r["mse_native"] for r in rs], [r["mse_resampled"] for r in rs], idx)
        pil_r[c] = {"r": r_, "r_ci95": rci, "abs_diff": ad, "verdict": S.margin_verdict(rci)}
    agg["seed0_pilot_old_testset_r_rederived"] = {"note": "DERIVED from results/raw/P1-REAL-01.jsonl with the P1-REAL-02 ratio estimator; old test set (ids 640-895); development data", "loss": pil_r}
    wjson(os.path.join(OUT, "aggregate.json"), agg)
    print(json.dumps({k: agg[k] for k in ("status", "primary_D_across_seeds", "primary_replicated_within_task",
                                          "S8_above_float32_floor_all_seeds")}, indent=2))


def main():
    ap = argparse.ArgumentParser()
    ap.add_argument("--stage", required=True, choices=["data", "eqcheck", "train", "testset", "eval", "aggregate"])
    ap.add_argument("--seed", type=int)
    ap.add_argument("--dev", action="store_true")
    a = ap.parse_args()
    if a.stage == "data":
        stage_data()
    elif a.stage == "eqcheck":
        stage_eqcheck()
    elif a.stage == "train":
        if a.seed not in cfg2()["seeds"]["training_seeds"]:
            raise SystemExit("seed not in the fixed training seeds")
        train_one(a.seed, os.path.join(OUT, f"seed{a.seed}"))
    elif a.stage == "testset":
        stage_testset()
    elif a.stage == "eval":
        stage_eval(a.seed, dev=a.dev)
    else:
        stage_aggregate()


if __name__ == "__main__":
    main()
