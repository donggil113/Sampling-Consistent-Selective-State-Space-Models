"""P1-REAL-01 runner: native-grid vs training-grid resampling with official TIDES.

Stages:
  plan   stdlib only: builds the trajectory split and test-condition grids for a
         few test IDs and checks the protocol invariants. Runs no model.
  smoke  (torch) timing of a few training steps and one eval batch. NOT_RUN here.
  train  (torch) fixed-step training under an approved cap. NOT_RUN here.
  eval   (torch) metrics for all test conditions and modes. NOT_RUN here.

See configs/p1_real_01.json for the fixed design and the approval items.
"""

import argparse
import json
import os
import random
import sys
import time

ROOT = os.path.dirname(os.path.dirname(os.path.abspath(__file__)))
sys.path.insert(0, os.path.join(ROOT, "src"))

from scssm.metrics import rel_l2  # noqa: E402
from scssm.real import protocol as P  # noqa: E402
from scssm.reference import ct_true  # noqa: E402

CFG = os.path.join(ROOT, "configs", "p1_real_01.json")
FR_CFG = os.path.join(ROOT, "configs", "first_run.json")
RAW = os.path.join(ROOT, "results", "raw")
TIDES_DIR = os.path.join(ROOT, "third_party", "TIDES")


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


def stage_plan(cfg, fr):
    t0 = time.time()
    sp = make_split(cfg)
    T, K = cfg["task"]["training_grid"]["T"], cfg["task"]["training_grid"]["K"]
    overlap = set(sp.train) & set(sp.dev) | set(sp.train) & set(sp.test) | set(sp.dev) & set(sp.test)
    checks = {"n_train": len(sp.train), "n_dev": len(sp.dev), "n_test": len(sp.test), "id_overlap": len(overlap)}
    tid = sp.test[0]
    path = P.trajectory_path(tid)
    conds = P.test_conditions(path, T, K, [1, 2, 4, 8])
    base = conds["C0"].times
    checks["base_times_common"] = all(
        [g.times[i] for i in g.base_index] == list(base) for g in conds.values())
    checks["split_values_repeat_held"] = all(
        conds[f"S{m}"].values[conds[f"S{m}"].base_index[k] - 1] == conds["C0"].values[k - 1]
        for m in (2, 4, 8) for k in range(1, K + 1))
    loc, early = P.resample_locf(conds["S8"], base)
    checks["locf_on_split_equals_C0"] = list(loc) == list(conds["C0"].values)
    J = P.irregular_grid(path, T, K, random.Random(2_000_000 + tid))
    _, early_j = P.resample_locf(J, base)
    checks["locf_irregular_early_targets"] = early_j
    checks["step_scale_training_grid"] = set(round(s, 12) for s in P.step_scales(conds["C0"], T / K))
    checks["step_scale_training_grid"] = sorted(checks["step_scale_training_grid"])
    checks["seconds"] = round(time.time() - t0, 3)
    print(json.dumps(checks, indent=2))
    os.makedirs(RAW, exist_ok=True)
    with open(os.path.join(RAW, "P1-REAL-01__plan_check.json"), "w") as f:
        json.dump({"status": "PLAN_ONLY: protocol checks, no model was run", "checks": checks}, f, indent=2)
    return checks


def labels_for(params, path, times):
    return ct_true(params, path, times, n_sub=128)["y"]


def stage_torch(stage, cfg, fr, max_seconds):
    from scssm.real import tides_adapter as TA
    torch, tides = TA.load_tides(TIDES_DIR)  # raises NOT_RUN if torch or the checkout is missing
    torch.set_num_threads(cfg["budget"]["cpu_threads"])
    params = P.teacher(fr)
    sp = make_split(cfg)
    T, K = cfg["task"]["training_grid"]["T"], cfg["task"]["training_grid"]["K"]
    dt_train = T / K
    # training data: training grid only, labels at training times
    tr_grids = [P.training_grid(P.trajectory_path(i), T, K) for i in sp.train]
    tr_labels = [labels_for(params, P.trajectory_path(i), g.times)[1:] for i, g in zip(sp.train, tr_grids)]
    x_stats = P.norm_stats([v for g in tr_grids for v in g.values])
    y_stats = P.norm_stats([v for lab in tr_labels for v in lab])
    model = TA.build_model(torch, tides, cfg)
    ckpt = os.path.join(RAW, "P1-REAL-01__model.pt")
    log_lines = []

    def log(msg):
        print(msg, flush=True)
        log_lines.append(msg)

    bs = cfg["training"]["batch_size"]
    rng = random.Random(cfg["training"]["seed"])

    def batches(step):
        idx = [rng.randrange(len(tr_grids)) for _ in range(bs)]
        return TA.batch_from_grids(torch, [tr_grids[i] for i in idx], [tr_labels[i] for i in idx],
                                   dt_train, x_stats, y_stats)

    if stage == "smoke":
        cfg = dict(cfg)
        cfg["training"] = dict(cfg["training"], steps=20, log_every=5)
        cfg["budget"] = dict(cfg["budget"], train_wall_clock_seconds=cfg["budget"]["timing_smoke_wall_clock_seconds"])
        t0 = time.time()
        TA.train(torch, model, batches, cfg, log)
        per_step = (time.time() - t0) / 20
        log(f"SMOKE: {per_step:.3f}s/step -> projected {per_step * 2000 / 60:.1f} min for 2000 steps (UNTRAINED model; no results)")
        return
    if stage == "train":
        if max_seconds is None:
            raise SystemExit("NOT_RUN: --max-seconds (approved cap) is required")
        cfg["budget"] = dict(cfg["budget"], train_wall_clock_seconds=max_seconds)
        ok = TA.train(torch, model, batches, cfg, log)
        if ok:
            torch.save(model.state_dict(), ckpt)
        with open(os.path.join(RAW, "P1-REAL-01__train.log"), "w") as f:
            f.write("\n".join(log_lines) + "\n")
        return
    # eval
    if not os.path.exists(ckpt):
        raise SystemExit("NOT_RUN: no trained checkpoint; evaluation of untrained weights is not a trained-model result")
    model.load_state_dict(torch.load(ckpt))
    recs = []
    for tid in sp.test:
        path = P.trajectory_path(tid)
        conds = P.test_conditions(path, T, K, [1, 2, 4, 8])
        rng_j = random.Random(2_000_000 + tid)
        for rho in (0.5, 1, 2):
            conds[f"J{rho}"] = P.irregular_grid(path, T, max(2, int(rho * K)), rng_j)
        base = conds["C0"].times
        y_true = labels_for(params, path, base)[1:]
        outs = {}
        for name, g in conds.items():
            for mode in ("native", "resampled_locf", "resampled_linear"):
                if mode == "native":
                    feed = g
                else:
                    vals = P.resample_locf(g, base)[0] if mode == "resampled_locf" else P.resample_linear(g, base)
                    feed = P.Grid(tuple(base), tuple(vals), tuple(range(len(base))))
                u, s, _ = TA.batch_from_grids(torch, [feed], None, dt_train, x_stats, y_stats)
                pred, secs = TA.predict(torch, model, u, s)
                pred = [p * y_stats[1] + y_stats[0] for p in pred[0].tolist()]
                if mode == "native" and name.startswith("J"):
                    at_base = []
                    for t in base[1:]:
                        j = max([k for k in range(1, len(feed.times)) if feed.times[k] <= t + 1e-12] or [1])
                        at_base.append(pred[j - 1])
                elif mode == "native":
                    at_base = [pred[i - 1] for i in feed.base_index[1:]]
                else:
                    at_base = pred
                outs[(name, mode)] = at_base
                recs.append({"traj": tid, "condition": name, "mode": mode, "tokens": len(feed.values),
                             "seconds": secs, "mse": sum((a - b) ** 2 for a, b in zip(at_base, y_true)) / len(y_true)})
        for r in recs[-len(outs):]:
            r["discrepancy_vs_native_C0"] = rel_l2(outs[(r["condition"], r["mode"])], outs[("C0", "native")])
    with open(os.path.join(RAW, "P1-REAL-01.jsonl"), "w") as f:
        for r in recs:
            f.write(json.dumps(r) + "\n")


def main():
    ap = argparse.ArgumentParser()
    ap.add_argument("--stage", choices=["plan", "smoke", "train", "eval"], required=True)
    ap.add_argument("--max-seconds", type=float, default=None)
    a = ap.parse_args()
    cfg, fr = load_cfg()
    if a.stage == "plan":
        stage_plan(cfg, fr)
    else:
        stage_torch(a.stage, cfg, fr, a.max_seconds)


if __name__ == "__main__":
    main()
