"""P1-HAR-01: one trained official TIDES classifier on UCI HAR inertial windows, evaluated under
an artificial same-held-path refinement (configs/p1_har_01.json; read that file first).

Stages (run in this order; each appends to results/raw/P1-HAR-01/cost_ledger.json and refuses
to start when the cumulative wall or process-CPU budget would be exceeded):
  data      parse the nine channel files of both official splits, hash, cache locally (gitignored),
            fit normalization on fit subjects, write data_hash.json
  smoke     train-only timing (20 updates + 1 calibration pass, <= 60 s wall) with a model that is
            discarded; writes smoke.json with the feasible-update decision of the config rule
  train     one model seed; calibration-selected checkpoint saved with sha256 before any test read
  eval      test subjects: native token-mean / native time-weighted / resampled arms on C0, S8, H8
            from the same checkpoint; per-window records; timing; C0 token logits stored
  control   time-weighted moment features + multinomial linear classifier (invariance control)
  aggregate subject-mean-first statistics, subject-level bootstrap, verdicts -> aggregate.json
"""

import argparse
import gzip
import hashlib
import io
import json
import math
import os
import random
import sys
import time

ROOT = os.path.dirname(os.path.dirname(os.path.abspath(__file__)))
sys.path.insert(0, os.path.join(ROOT, "src"))
sys.path.insert(0, os.path.join(ROOT, "scripts"))

from run_p1_real_01 import TIDES_DIR, git, seed_all, sha256_file  # noqa: E402
from run_p1_real_02 import eval_mode_checks  # noqa: E402
from scssm.real import stats as S  # noqa: E402

CFG = os.path.join(ROOT, "configs", "p1_har_01.json")
OUT = os.path.join(ROOT, "results", "raw", "P1-HAR-01")
DATA = os.path.join(ROOT, "data", "har", "extracted", "UCI HAR Dataset")
DERIVED = os.path.join(ROOT, "data", "har", "derived")
LEDGER = os.path.join(OUT, "cost_ledger.json")
CONDS = ("C0", "S8", "H8")
ARMS = ("native_mean", "native_time", "resampled")


def cfg():
    with open(CFG) as f:
        return json.load(f)


def clock():
    return time.perf_counter(), time.process_time()


def since(t):
    return round(time.perf_counter() - t[0], 3), round(time.process_time() - t[1], 3)


def wjson(path, obj):
    os.makedirs(os.path.dirname(path), exist_ok=True)
    with open(path, "w") as f:
        json.dump(obj, f, indent=2)


def utc():
    return time.strftime("%Y-%m-%dT%H:%M:%SZ", time.gmtime())


# ----------------------------------------------------------------------------- budget ledger

def ledger_totals():
    if not os.path.exists(LEDGER):
        return 0.0, 0.0, []
    rows = json.load(open(LEDGER))
    return sum(r["wall_seconds"] for r in rows), sum(r["cpu_seconds"] for r in rows), rows


def ledger_guard(stage, projected_wall, projected_cpu):
    """Refuse to start a stage when the cumulative ledger plus the stage's projected cost would exceed a cap."""
    c = cfg()["budget"]
    w, p, _ = ledger_totals()
    if w + projected_wall > c["total_wall_seconds"] or p + projected_cpu > c["total_process_cpu_seconds"]:
        ledger_add(stage, 0.0, 0.0, f"BLOCKED: cumulative {w:.0f}s wall / {p:.0f}s cpu plus projected {projected_wall:.0f}s / {projected_cpu:.0f}s would exceed the cap")
        raise SystemExit(f"BLOCKED: budget (wall {w:.0f}+{projected_wall:.0f}s, cpu {p:.0f}+{projected_cpu:.0f}s)")


def config_sha():
    return sha256_file(CFG)


def provenance():
    return {"git_head": git("rev-parse", "HEAD"), "git_dirty": bool(git("status", "--porcelain")),
            "runner_sha256": sha256_file(os.path.abspath(__file__)), "config_sha256": config_sha()}


def run_stage(name, fn):
    """Run a stage body under a timer that starts before any import or hashing; a crash is recorded as a FAIL ledger row."""
    t0 = clock()
    started = utc()
    try:
        fn(t0, started)
    except SystemExit:
        raise
    except BaseException as e:  # noqa: BLE001  (every failure must leave a ledger row)
        w, c_ = since(t0)
        ledger_add(name, w, c_, f"FAIL({type(e).__name__}: {str(e)[:200]})", started=started)
        raise


def check_calibration_rule(c):
    """Re-derive the ID-only calibration rule (ascending hex digest) and compare with the frozen lists."""
    tr = c["data"]["train_subjects"]
    ranked = sorted(tr, key=lambda s_: hashlib.sha256(f"P1-HAR-01:calibration:{s_}".encode()).hexdigest())
    assert sorted(ranked[:4]) == c["data"]["calibration_subjects"], "calibration subjects differ from the rule"
    assert sorted(set(tr) - set(ranked[:4])) == c["data"]["fit_subjects"], "fit subjects are not the complement"


def refuse_if_test_opened(stage):
    lock = os.path.join(OUT, "test_opened.json")
    if os.path.exists(lock):
        if stage == "eval" and json.load(open(lock)).get("eval_status") != "COMPLETED":
            return  # a crashed evaluation of the SAME frozen checkpoint may be repeated; the ledger keeps the failed attempt
        ledger_add(stage, 0.0, 0.0, "REFUSED: test already opened (no replacement run)")
        raise SystemExit("REFUSED: test already opened")
    if stage == "train" and os.path.exists(os.path.join(OUT, "checkpoint.pt")):
        ledger_add(stage, 0.0, 0.0, "REFUSED: checkpoint.pt already exists (no replacement run)")
        raise SystemExit("REFUSED: checkpoint exists")


def ledger_add(stage, wall, cpu_s, status, extra=None, started=None):
    _, _, rows = ledger_totals()
    row = {"stage": stage, "started_utc": started or utc(), "finished_utc": utc(), "wall_seconds": round(wall, 3), "cpu_seconds": round(cpu_s, 3),
           "threads": 2, "status": status}
    if extra:
        row.update(extra)
    rows.append(row)
    w = sum(r["wall_seconds"] for r in rows)
    p = sum(r["cpu_seconds"] for r in rows)
    wjson(LEDGER, rows)
    print(f"[ledger] {stage}: wall {wall:.1f}s cpu {cpu_s:.1f}s -> cumulative wall {w:.1f}s cpu {p:.1f}s ({status})")


# ----------------------------------------------------------------------------- data

def load_split(split, channels):
    import numpy as np
    X = []
    for c in channels:
        p = os.path.join(DATA, split, "Inertial Signals", f"{c}_{split}.txt")
        X.append(np.loadtxt(p, dtype=np.float64))
    X = np.stack(X, axis=-1).astype(np.float32)  # (N, 128, 9)
    subj = np.loadtxt(os.path.join(DATA, split, f"subject_{split}.txt"), dtype=np.int64)
    y = np.loadtxt(os.path.join(DATA, split, f"y_{split}.txt"), dtype=np.int64) - 1
    return X, subj, y


def arr_sha(a):
    import numpy as np
    return hashlib.sha256(np.ascontiguousarray(a).tobytes()).hexdigest()


def stage_data():
    import numpy as np
    c = cfg()
    refuse_if_test_opened("data")
    ledger_guard("data", 30, 60)
    t = clock()
    ch = c["data"]["channels"]
    Xtr, str_, ytr = load_split("train", ch)
    Xte, ste, yte = load_split("test", ch)
    assert Xtr.shape == (7352, 128, 9) and Xte.shape == (2947, 128, 9), (Xtr.shape, Xte.shape)
    assert sorted(set(str_.tolist())) == c["data"]["train_subjects"]
    assert sorted(set(ste.tolist())) == c["data"]["test_subjects"]
    assert set(ytr.tolist()) == set(range(6)) and set(yte.tolist()) == set(range(6))
    fit = np.isin(str_, c["data"]["fit_subjects"])
    cal = np.isin(str_, c["data"]["calibration_subjects"])
    assert not (fit & cal).any() and (fit | cal).all()
    mu = Xtr[fit].reshape(-1, 9).mean(axis=0)
    sd = Xtr[fit].reshape(-1, 9).std(axis=0)
    assert (sd > 0).all()
    os.makedirs(DERIVED, exist_ok=True)
    np.savez(os.path.join(DERIVED, "cache.npz"), Xtr=Xtr, str_=str_, ytr=ytr, Xte=Xte, ste=ste, yte=yte, mu=mu, sd=sd)
    rep = {"train_shape": list(Xtr.shape), "test_shape": list(Xte.shape),
           "sha256": {"Xtr": arr_sha(Xtr), "ytr": arr_sha(ytr), "subject_train": arr_sha(str_),
                      "Xte": arr_sha(Xte), "yte": arr_sha(yte), "subject_test": arr_sha(ste)},
           "n_fit_windows": int(fit.sum()), "n_calibration_windows": int(cal.sum()), "n_test_windows": int(len(yte)),
           "fit_subjects": c["data"]["fit_subjects"], "calibration_subjects": c["data"]["calibration_subjects"],
           "normalization_fit_only": {"channel": ch, "mean": mu.tolist(), "sd": sd.tolist()},
           "finite": bool(np.isfinite(Xtr).all() and np.isfinite(Xte).all()),
           "channel_value_ranges_train": {cn: [float(Xtr[..., i].min()), float(Xtr[..., i].max())] for i, cn in enumerate(ch)}}
    wall, cpu_s = since(t)
    rep.update({"wall_seconds": wall, "cpu_seconds": cpu_s})
    old = os.path.join(OUT, "data_hash.json")
    if os.path.exists(old) and json.load(open(old))["sha256"] != rep["sha256"]:
        raise SystemExit("data_hash.json exists with different hashes; refusing to overwrite")
    wjson(old, rep)
    ledger_add("data", wall, cpu_s, "COMPLETED")
    print(json.dumps({k: rep[k] for k in ("train_shape", "test_shape", "n_fit_windows", "n_calibration_windows", "n_test_windows")}))


def load_cache(need_test=False):
    """Smoke and train verify only the train arrays; the test arrays are hashed (and used) by eval and control only."""
    import numpy as np
    d = np.load(os.path.join(DERIVED, "cache.npz"))
    rep = json.load(open(os.path.join(OUT, "data_hash.json")))
    if arr_sha(d["Xtr"]) != rep["sha256"]["Xtr"] or arr_sha(d["ytr"]) != rep["sha256"]["ytr"]:
        raise SystemExit("data cache hash mismatch (train)")
    if need_test and (arr_sha(d["Xte"]) != rep["sha256"]["Xte"] or arr_sha(d["yte"]) != rep["sha256"]["yte"]):
        raise SystemExit("data cache hash mismatch (test)")
    return d, rep


# ----------------------------------------------------------------------------- grids

def grid(cond, dt_train=0.02, K=128):
    """Token times, step scales, base-interval index per token, end flags, time weights."""
    m_of = {"C0": [1] * K, "S8": [8] * K, "H8": [8] * (K // 2) + [1] * (K // 2)}[cond]
    times, scale, base, is_end = [0.0], [], [], []
    for k in range(K):
        t0, t1 = k * dt_train, (k + 1) * dt_train
        m = m_of[k]
        for j in range(1, m + 1):
            s = t1 if j == m else t0 + (t1 - t0) * j / m
            times.append(s)
            scale.append(1.0 / m)
            base.append(k)
            is_end.append(j == m)
    T = K * dt_train
    w_time = [sc * dt_train / T for sc in scale]
    return {"cond": cond, "times": times, "scale": scale, "base": base, "is_end": is_end, "w_time": w_time, "L": len(scale), "T": T}


def native_batch(torch, X, g, mu, sd):
    """X: (B, 128, 9) raw values -> normalized (B, L, 9) with held values repeated, (B, L) step scales."""
    import numpy as np
    base = np.asarray(g["base"])
    u = (X[:, base, :] - mu) / sd
    s = np.tile(np.asarray(g["scale"], dtype=np.float32), (X.shape[0], 1))
    return torch.tensor(u, dtype=torch.float32), torch.tensor(s, dtype=torch.float32)


def locf_resample(X_cond_values, g, K=128, dt_train=0.02):
    """LOCF from the condition's observed (time, value) pairs onto the training grid, using only
    observed times and values. Returns (B, K, 9) raw values."""
    import numpy as np
    obs_t = np.asarray(g["times"][1:])
    idx = []
    for k in range(1, K + 1):
        t = k * dt_train
        j = int(np.searchsorted(obs_t, t + 1e-9, side="right")) - 1
        assert j >= 0, ("no observation at or before", t)
        idx.append(j)
    assert X_cond_values.shape[1] == len(obs_t)
    return X_cond_values[:, idx, :]


# ----------------------------------------------------------------------------- model

def build(torch, tides, c):
    kw = dict(c["model"]["kwargs"])
    model = tides.TIDESClassifier(**kw)
    assert isinstance(model.head, torch.nn.Linear), "head must be affine for the pooling identity"
    return model


def setup(c):
    from scssm.real import tides_adapter as TA
    torch, tides = TA.load_tides(TIDES_DIR)
    torch.set_num_threads(c["budget"]["cpu_threads"])
    return torch, tides


def features(torch, model, u, s, bs=256):
    outs = []
    for i in range(0, u.shape[0], bs):
        outs.append(model.backbone(u[i:i + bs], step_scale=s[i:i + bs]))
    return torch.cat(outs, 0)


def ce_mean(torch, logits, y):
    return float(torch.nn.functional.cross_entropy(logits, y).item())


def calibration_pass(torch, model, u, s, y, bs=256):
    model.eval()
    chk = eval_mode_checks(torch, model)
    with torch.inference_mode():
        logits = torch.cat([model(u[i:i + bs], step_scale=s[i:i + bs]) for i in range(0, u.shape[0], bs)], 0)
        ce = ce_mean(torch, logits, y)
        acc = float((logits.argmax(1) == y).float().mean().item())
    model.train()
    return ce, acc, chk


def train_loop(torch, tides, c, d, updates, log, cal_every=100, time_cap=None, smoke=False):
    """Shared by smoke and train. Returns dict with model state, curve, timings."""
    import numpy as np
    seed = c["training"]["model_seed"]
    fit = np.isin(d["str_"], c["data"]["fit_subjects"])
    cal = np.isin(d["str_"], c["data"]["calibration_subjects"])
    Xf, yf = d["Xtr"][fit], d["ytr"][fit]
    Xc, yc = d["Xtr"][cal], d["ytr"][cal]
    g0 = grid("C0")
    u_c, s_c = native_batch(torch, Xc, g0, d["mu"], d["sd"])
    y_c = torch.tensor(yc)
    seed_all(torch, seed)
    model = build(torch, tides, c)
    opt = torch.optim.Adam(model.parameters(), lr=c["training"]["lr"])
    rng = random.Random(seed)
    bsz = 32
    checks, upd_times, cal_times = [], [], []
    t_cal = clock()
    ce0, acc0, chk = calibration_pass(torch, model, u_c, s_c, y_c)
    cal_times.append(since(t_cal)[0])
    checks.append(chk)
    curve = [{"update": 0, "cal_ce": ce0, "cal_acc": acc0}]
    best = {"update": 0, "cal_ce": ce0, "cal_acc": acc0, "state": {k: v.clone() for k, v in model.state_dict().items()}}
    model.train()
    status = "COMPLETED"
    t_loop = clock()
    min_scale = float("inf")
    for step in range(1, updates + 1):
        t_u = time.perf_counter()
        idx = [rng.randrange(len(Xf)) for _ in range(bsz)]
        u, s = native_batch(torch, Xf[idx], g0, d["mu"], d["sd"])
        y = torch.tensor(yf[idx])
        min_scale = min(min_scale, float(s.min()))
        loss = torch.nn.functional.cross_entropy(model(u, step_scale=s), y)
        if not torch.isfinite(loss):
            status = f"FAIL(non-finite loss at update {step})"
            log(status)
            break
        opt.zero_grad()
        loss.backward()
        opt.step()
        upd_times.append(time.perf_counter() - t_u)
        if step % cal_every == 0 or (smoke and step == updates):
            t_cal = clock()
            ce, acc, chk = calibration_pass(torch, model, u_c, s_c, y_c)
            cal_times.append(since(t_cal)[0])
            checks.append(chk)
            curve.append({"update": step, "train_ce": float(loss.detach()), "cal_ce": ce, "cal_acc": acc})
            if ce < best["cal_ce"]:
                best = {"update": step, "cal_ce": ce, "cal_acc": acc, "state": {k: v.clone() for k, v in model.state_dict().items()}}
            log(f"update {step} train_ce {float(loss.detach()):.4f} cal_ce {ce:.4f} cal_acc {acc:.3f} best@{best['update']}")
        if time_cap is not None and time.perf_counter() - t_loop[0] > time_cap:
            status = "INCOMPLETE_BUDGET(train)"
            log(f"STOP at update {step}: wall allowance {time_cap:.0f}s exceeded; best-so-far state kept, test not opened")
            break
    loop_wall, loop_cpu = since(t_loop)
    checks_ok = all(ch["all_modules_eval"] and ch["dropout_p"] in ([0.0], []) and ch["batchnorm_running_stats"] for ch in checks)
    if not checks_ok and status == "COMPLETED":
        status = "FAIL(eval-mode check)"
    return {"status": status, "best": best, "curve": curve, "upd_times": upd_times, "cal_times": cal_times,
            "loop_wall": loop_wall, "loop_cpu": loop_cpu, "checks_ok": checks_ok, "check_example": checks[-1],
            "min_train_step_scale": min_scale, "n_fit": int(len(Xf)), "n_cal": int(len(Xc)), "n_params": sum(p.numel() for p in model.parameters())}


def stage_smoke():
    import numpy as np
    c = cfg()
    refuse_if_test_opened("smoke")
    ledger_guard("smoke", c["budget"]["smoke_wall_seconds_max"], 2 * c["budget"]["smoke_wall_seconds_max"])
    check_calibration_rule(c)
    torch, tides = setup(c)
    d, _ = load_cache(need_test=False)
    t = clock()
    lines = []

    def log(m):
        lines.append(m)
        print(m, flush=True)
    r = train_loop(torch, tides, c, d, updates=20, log=log, cal_every=10**9, time_cap=c["budget"]["smoke_wall_seconds_max"] - 15, smoke=True)
    # one inference batch of 256 fit windows on the S8 grid (L = 1024) for a long-sequence per-token rate
    seed_all(torch, c["training"]["model_seed"])
    probe = build(torch, tides, c)
    probe.eval()
    gS = grid("S8")
    fit_idx = np.where(np.isin(d["str_"], c["data"]["fit_subjects"]))[0][:256]
    uS, sS = native_batch(torch, d["Xtr"][fit_idx], gS, d["mu"], d["sd"])
    with torch.inference_mode():
        probe(uS[:8], step_scale=sS[:8])
        tS = clock()
        probe(uS, step_scale=sS)
        s8_wall, _ = since(tS)
    wall, cpu_s = since(t)
    smoke_ok = r["status"] == "COMPLETED" and len(r["upd_times"]) == 20
    s_upd = sum(r["upd_times"][2:]) / 18 if smoke_ok else None
    s_cal = sum(r["cal_times"]) / len(r["cal_times"])
    n_cal = r["n_cal"]
    rate_cal = s_cal / (n_cal * 128)
    rate_s8 = s8_wall / (256 * 1024)
    rate = max(rate_cal, rate_s8)
    cpu_ratio = cpu_s / max(wall, 1e-9)
    n_test = 2947
    U = c["training"]["updates_planned"]
    tokens_main = n_test * (128 + 1024 + 576) + n_test * 128 + 3 * n_test * 128
    tokens_timing = 7 * 256 * sum(2 * grid(cn)["L"] + 2 * 128 for cn in CONDS)
    proj_eval = 1.5 * rate * (tokens_main + tokens_timing)

    def proj_train(u):
        return (u * s_upd + (u / 100 + 1) * s_cal) if smoke_ok else 0.0
    B = c["budget"]
    w_used, p_used, _ = ledger_totals()

    def ok(u):
        w = w_used + wall + proj_train(u) + proj_eval + 300
        p = p_used + cpu_s + cpu_ratio * (proj_train(u) + proj_eval) + 600
        return w <= B["total_wall_seconds"] and p <= B["total_process_cpu_seconds"]
    if smoke_ok:
        feasible = U if ok(U) else max((u for u in range(0, U, 100) if ok(u)), default=0)
        status = "OK" if feasible >= 500 else "INCOMPLETE_BUDGET"
    else:
        feasible, status = 0, f"FAIL(smoke: {r['status']}, {len(r['upd_times'])} updates)"
    decision = {"updates": feasible, "status": status, "train_wall_allowance_seconds": round(proj_train(feasible) + 300, 1) if smoke_ok else 0,
                "projected_eval_wall_seconds": round(proj_eval, 1), "projected_eval_cpu_seconds": round(cpu_ratio * proj_eval, 1)}
    rep = {"smoke_updates": 20, "status": r["status"], "wall_seconds": wall, "cpu_seconds": cpu_s, "cpu_to_wall_ratio": round(cpu_ratio, 3),
           "s_per_update_mean_3_to_20": s_upd, "s_per_calibration_pass_mean_of_2": s_cal, "n_fit": r["n_fit"], "n_cal": n_cal,
           "s_per_token_calibration_L128": rate_cal, "s_per_token_S8_batch256_L1024": rate_s8, "rate_used": rate,
           "n_parameters": r["n_params"], "projected_train_seconds_2000": round(proj_train(U), 1) if smoke_ok else None,
           "projected_eval_tokens": {"main": tokens_main, "timing_block": tokens_timing}, "projected_eval_seconds": round(proj_eval, 1),
           "ledger_before": {"wall": w_used, "cpu": p_used}, "decision": decision, "model_discarded": True,
           "config_sha256": config_sha(), "rule": c["training"]["feasible_budget_rule"]}
    wjson(os.path.join(OUT, "smoke.json"), rep)
    ledger_add("smoke", wall, cpu_s, r["status"], {"decision": decision})
    print(json.dumps({k: rep[k] for k in ("s_per_update_mean_3_to_20", "s_per_calibration_pass_mean_of_2", "rate_used", "projected_train_seconds_2000", "projected_eval_seconds", "decision")}, indent=1))


def stage_train():
    c = cfg()
    refuse_if_test_opened("train")
    sm = json.load(open(os.path.join(OUT, "smoke.json")))
    dec = sm["decision"]
    if sm["config_sha256"] != config_sha():
        raise SystemExit("config changed since the smoke decision")
    if dec["status"] != "OK":
        ledger_add("train", 0.0, 0.0, f"NOT_RUN({dec['status']} per smoke decision)")
        raise SystemExit(f"{dec['status']}: not training")
    ledger_guard("train", dec["train_wall_allowance_seconds"], sm["cpu_to_wall_ratio"] * dec["train_wall_allowance_seconds"] + 60)
    check_calibration_rule(c)
    torch, tides = setup(c)
    d, dh = load_cache(need_test=False)
    t = clock()
    lines = [f"git HEAD {git('rev-parse', 'HEAD')} dirty={bool(git('status', '--porcelain'))}; updates {dec['updates']}; allowance {dec['train_wall_allowance_seconds']}s"]
    print(lines[0])

    def log(m):
        lines.append(f"[{time.perf_counter() - t[0]:8.2f}s] {m}")
        print(lines[-1], flush=True)
    r = train_loop(torch, tides, c, d, updates=dec["updates"], log=log, cal_every=100, time_cap=dec["train_wall_allowance_seconds"])
    wall, cpu_s = since(t)
    ckpt = os.path.join(OUT, "checkpoint.pt")
    buf = io.BytesIO()
    torch.save(r["best"]["state"], buf)
    with open(ckpt, "wb") as f:
        f.write(buf.getvalue())
    sha = sha256_file(ckpt)
    with open(ckpt + ".sha256", "w") as f:
        f.write(f"{sha}  checkpoint.pt\n")
    out = {"status": r["status"], "model_seed": c["training"]["model_seed"], "updates_run": len(r["upd_times"]), "updates_planned": dec["updates"],
           "selected_update": r["best"]["update"], "selected_cal_ce": r["best"]["cal_ce"], "selected_cal_acc": r["best"]["cal_acc"],
           "curve": r["curve"], "eval_mode_checks_all_ok": r["checks_ok"], "eval_mode_check_example": r["check_example"],
           "min_train_step_scale": r["min_train_step_scale"], "n_parameters": r["n_params"], "n_fit_windows": r["n_fit"], "n_cal_windows": r["n_cal"],
           "train_loop_wall_seconds": r["loop_wall"], "train_loop_cpu_seconds": r["loop_cpu"], "stage_wall_seconds": wall, "stage_cpu_seconds": cpu_s,
           "checkpoint_sha256": sha, "data_sha256": dh["sha256"], "torch": torch.__version__, "threads": torch.get_num_threads(),
           "git_head": git("rev-parse", "HEAD"), "config_sha256": config_sha(), "test_opened": False}
    wjson(os.path.join(OUT, "train.json"), out)
    with open(os.path.join(OUT, "train.log"), "w") as f:
        f.write("\n".join(lines) + "\n")
    ledger_add("train", wall, cpu_s, r["status"], {"updates_run": len(r["upd_times"]), "selected_update": r["best"]["update"]})
    print(json.dumps({k: out[k] for k in ("status", "updates_run", "selected_update", "selected_cal_ce", "selected_cal_acc", "checkpoint_sha256")}, indent=1))


# ----------------------------------------------------------------------------- eval

def stage_eval():
    import numpy as np
    c = cfg()
    refuse_if_test_opened("eval")
    tj = json.load(open(os.path.join(OUT, "train.json")))
    sm = json.load(open(os.path.join(OUT, "smoke.json")))
    if tj["config_sha256"] != config_sha():
        raise SystemExit("config changed since training")
    if tj["status"] != "COMPLETED":
        ledger_add("eval", 0.0, 0.0, f"NOT_RUN(train status {tj['status']})")
        raise SystemExit("test not opened")
    ledger_guard("eval", sm["decision"]["projected_eval_wall_seconds"], sm["decision"]["projected_eval_cpu_seconds"])
    check_calibration_rule(c)
    torch, tides = setup(c)
    ckpt = os.path.join(OUT, "checkpoint.pt")
    if sha256_file(ckpt) != tj["checkpoint_sha256"]:
        raise SystemExit("checkpoint hash mismatch")
    t = clock()
    seed_all(torch, 0)
    model = build(torch, tides, c)
    model.load_state_dict(torch.load(ckpt))
    model.eval()
    checks = eval_mode_checks(torch, model)
    if not (checks["all_modules_eval"] and checks["dropout_p"] in ([0.0], []) and checks["batchnorm_running_stats"]):
        ledger_add("eval", *since(t), "FAIL(eval-mode check)")
        raise SystemExit("FAIL(eval-mode check)")
    # the test arrays are read (and their hash verified) only from here on
    d, dh = load_cache(need_test=True)
    wjson(os.path.join(OUT, "test_opened.json"), {"utc": utc(), "checkpoint_sha256": tj["checkpoint_sha256"], "config_sha256": config_sha(), "eval_status": "in_progress"})
    tj["test_opened"] = True
    wjson(os.path.join(OUT, "train.json"), tj)
    wall_cap = ledger_totals()[0]
    X, subj, y = d["Xte"], d["ste"], d["yte"]
    mu, sd = d["mu"], d["sd"]
    N = X.shape[0]
    y_t = torch.tensor(y)
    gs = {cn: grid(cn) for cn in CONDS}
    tok = {}       # per-condition per-token logits (N, L, 6) float32 numpy
    pooled = {}    # (cond, arm) -> (N, 6) numpy logits
    resample_check = {}
    with torch.inference_mode():
        W_head, b_head = model.head.weight, model.head.bias
        for cn in CONDS:
            g = gs[cn]
            u, s = native_batch(torch, X, g, mu, sd)
            assert float(s.min()) > 0
            h = features(torch, model, u, s)                       # (N, L, 16) features from the official backbone
            z = (h @ W_head.T + b_head)                             # (N, L, 6) per-token logits (affine head), stored for re-aggregation
            tok[cn] = z.numpy().astype(np.float32)
            if not np.isfinite(tok[cn]).all():
                ledger_add("eval", *since(t), f"FAIL(non-finite output on {cn})")
                raise SystemExit(f"FAIL(non-finite output on {cn})")
            w = torch.tensor(g["w_time"], dtype=torch.float32)
            # arms pool FEATURES at the official pooling location and apply the same affine head
            pooled[(cn, "native_mean")] = model.head(h.mean(1)).numpy().astype(np.float64)
            pooled[(cn, "native_time")] = model.head(torch.einsum("nlc,l->nc", h, w)).numpy().astype(np.float64)
            resample_check[f"{cn}_head_of_mean_vs_mean_of_token_logits_max_abs"] = float(np.abs(pooled[(cn, "native_mean")] - z.mean(1).numpy()).max())
            if cn == "C0":
                direct = torch.cat([model(u[i:i + 256], step_scale=s[i:i + 256]) for i in range(0, N, 256)], 0).numpy()
                resample_check["C0_direct_forward_vs_native_mean_max_abs"] = float(np.abs(direct - pooled[(cn, "native_mean")]).max())
                resample_check["C0_native_mean_vs_native_time_max_abs"] = float(np.abs(pooled[(cn, "native_mean")] - pooled[(cn, "native_time")]).max())
            Xr = locf_resample(X[:, g["base"], :], g)
            resample_check[f"{cn}_resampled_input_vs_C0_max_abs"] = float(np.abs(Xr - X).max())
            ur, sr = native_batch(torch, Xr, gs["C0"], mu, sd)
            zr = torch.cat([model(ur[i:i + 256], step_scale=sr[i:i + 256]) for i in range(0, N, 256)], 0).numpy().astype(np.float64)
            pooled[(cn, "resampled")] = zr
            resample_check[f"{cn}_resampled_logits_vs_C0_native_mean_max_abs"] = float(np.abs(zr - pooled[("C0", "native_mean")]).max()) if ("C0", "native_mean") in pooled else None
        # timing on the first 256 test windows (skipped, not run, if it would exceed the wall cap)
        timing = {}
        elapsed = since(t)[0]
        proj_timing = sm["decision"]["projected_eval_wall_seconds"] * sm["projected_eval_tokens"]["timing_block"] / (sm["projected_eval_tokens"]["main"] + sm["projected_eval_tokens"]["timing_block"])
        if wall_cap + elapsed + proj_timing > c["budget"]["total_wall_seconds"]:
            timing = {"status": f"SKIPPED (budget): elapsed {elapsed:.0f}s + projected {proj_timing:.0f}s would exceed the cap"}
        for cn in (CONDS if "status" not in timing else ()):
            g = gs[cn]
            Xb = X[:256]
            ub, sb = native_batch(torch, Xb, g, mu, sd)
            Xrb = locf_resample(Xb[:, g["base"], :], g)
            urb, srb = native_batch(torch, Xrb, gs["C0"], mu, sd)
            wt = torch.tensor(g["w_time"], dtype=torch.float32)

            def native_e2e():
                uu, ss = native_batch(torch, Xb, g, mu, sd)
                hh = model.backbone(uu, step_scale=ss)
                return model.head(hh.mean(1)), model.head(torch.einsum("nlc,l->nc", hh, wt))

            def res_e2e():
                xr = locf_resample(Xb[:, g["base"], :], g)
                uu, ss = native_batch(torch, xr, gs["C0"], mu, sd)
                return model(uu, step_scale=ss)
            jobs = {"native_end_to_end": native_e2e, "resampled_end_to_end": res_e2e,
                    "native_forward_only": lambda: model(ub, step_scale=sb), "resampled_forward_only": lambda: model(urb, step_scale=srb)}
            direct256 = model(ub, step_scale=sb).numpy().astype(np.float64)
            resample_check[f"{cn}_direct_forward_vs_native_mean_first256_max_abs"] = float(np.abs(direct256 - pooled[(cn, "native_mean")][:256]).max())
            timing[cn] = {"tokens_native": g["L"], "tokens_resampled": 128, "batch": 256}
            for jn, fn in jobs.items():
                for _ in range(2):
                    fn()
                walls, cpus = [], []
                for _ in range(5):
                    tt = clock()
                    fn()
                    ww, cc = since(tt)
                    walls.append(ww)
                    cpus.append(cc)
                walls.sort()
                cpus.sort()
                timing[cn][jn] = {"median_wall_s": walls[2], "min_wall_s": walls[0], "median_cpu_s": cpus[2]}
    # per-window records
    def lsm(z):
        z = z - z.max(axis=1, keepdims=True)
        return z - np.log(np.exp(z).sum(axis=1, keepdims=True))
    base_C0 = pooled[("C0", "native_mean")]
    p_C0 = np.exp(lsm(base_C0))
    zc_C0 = base_C0 - base_C0.mean(axis=1, keepdims=True)
    zc_norm = np.linalg.norm(zc_C0, axis=1)
    excluded = int((zc_norm < 1e-6).sum())
    recs = []
    gH = gs["H8"]
    aH = 1.0 / gH["L"]
    wH = np.asarray(gH["w_time"])
    baseH = np.asarray(gH["base"])
    endH = np.asarray(gH["is_end"])
    bC0 = tok["C0"]                                   # (N, 128, 6)
    zH = tok["H8"]                                    # (N, 576, 6)
    bj = bC0[:, baseH, :]                             # inherited baseline per H8 token
    Wd = ((aH - wH)[None, :, None] * bj).sum(1)
    Md = (aH * (zH - bj)).sum(1)
    Md_end = (aH * (zH - bj) * endH[None, :, None]).sum(1)
    Qd = (wH[None, :, None] * (zH - bj)).sum(1)
    A = bC0[:, :64, :].mean(1)
    Bm = bC0[:, 64:, :].mean(1)
    W_identity = 7.0 / 18.0 * (A - Bm)
    ident = {"W_vs_7_18_AB_max_abs": float(np.abs(Wd - W_identity).max()),
             "pmean_minus_pC0_vs_W_plus_M_max_abs": float(np.abs(pooled[("H8", "native_mean")] - base_C0 - Wd - Md).max()),
             "ptime_minus_pC0_vs_Q_max_abs": float(np.abs(pooled[("H8", "native_time")] - base_C0 - Qd).max())}
    for i in range(N):
        r = {"id": int(i), "subject": int(subj[i]), "label": int(y[i]), "excluded_centered_logit_distance": bool(zc_norm[i] < 1e-6)}
        for cn in CONDS:
            for arm in ARMS:
                z = pooled[(cn, arm)][i]
                l = lsm(z[None, :])[0]
                p = np.exp(l)
                zc = z - z.mean()
                r[f"{cn}/{arm}"] = {"logits": [float(v) for v in z], "ce": float(-l[y[i]]), "pred": int(z.argmax()),
                                    "tv_vs_C0": float(0.5 * np.abs(p - p_C0[i]).sum()),
                                    "centered_logit_rel_vs_C0": (float(np.linalg.norm(zc - zc_C0[i]) / zc_norm[i]) if zc_norm[i] >= 1e-6 else None),
                                    "flip_vs_C0": bool(z.argmax() != base_C0[i].argmax())}
        r["H8_decomposition"] = {"W": Wd[i].tolist(), "M": Md[i].tolist(), "M_end": Md_end[i].tolist(), "Q": Qd[i].tolist(),
                                 "A_minus_B": (A[i] - Bm[i]).tolist()}
        recs.append(r)
    with open(os.path.join(OUT, "eval.jsonl"), "w") as f:
        for r in recs:
            f.write(json.dumps(r) + "\n")
    # token logits: C0 in git (float16, gz); S8/H8 local only
    with gzip.open(os.path.join(OUT, "C0_token_logits.f16.npy.gz"), "wb") as f:
        np.save(f, tok["C0"].astype(np.float16))
    os.makedirs(DERIVED, exist_ok=True)
    for cn in ("S8", "H8"):
        np.save(os.path.join(DERIVED, f"{cn}_token_logits.f16.npy"), tok[cn].astype(np.float16))
    sizes_mb = {f: round(os.path.getsize(os.path.join(OUT, f)) / 1e6, 3) for f in ("eval.jsonl", "C0_token_logits.f16.npy.gz")}
    if sum(sizes_mb.values()) > c["stored_outputs"]["size_cap_mb_in_git"]:
        raise SystemExit(f"stored outputs exceed the size cap: {sizes_mb}")
    wall, cpu_s = since(t)
    summary = {"status": "COMPLETED", "checkpoint_sha256": tj["checkpoint_sha256"], "selected_update": tj["selected_update"],
               "n_test_windows": int(N), "test_subjects": sorted(set(subj.tolist())), "eval_mode_checks": checks,
               "numerical_checks": resample_check, "identity_checks": ident, "excluded_windows_centered_logit": excluded,
               "timing": timing, "threads": torch.get_num_threads(), "eval_wall_seconds": wall, "eval_cpu_seconds": cpu_s,
               "stored_sizes_mb": sizes_mb, "provenance": provenance(),
               "stored": {"eval_jsonl": "per-window pooled logits (float64), CE, predictions, distances, H8 decomposition",
                          "C0_token_logits": "results (float16 gz)", "S8_H8_token_logits": "data/har/derived (local only, float16)"}}
    wjson(os.path.join(OUT, "eval_summary.json"), summary)
    lock = json.load(open(os.path.join(OUT, "test_opened.json")))
    lock["eval_status"] = "COMPLETED"
    wjson(os.path.join(OUT, "test_opened.json"), lock)
    ledger_add("eval", wall, cpu_s, "COMPLETED")
    print(json.dumps({"numerical_checks": resample_check, "identity_checks": ident, "excluded": excluded, "wall": wall, "cpu": cpu_s}, indent=1))


# ----------------------------------------------------------------------------- control

def stage_control():
    import numpy as np
    c = cfg()
    ledger_guard("control", 120, 240)
    torch, _ = setup(c)
    d, _ = load_cache(need_test=True)
    t = clock()

    def feats(X, g):
        w = np.asarray(g["w_time"], dtype=np.float64)
        V = X[:, np.asarray(g["base"]), :].astype(np.float64)
        m1 = (w[None, :, None] * V).sum(1)
        m2 = (w[None, :, None] * V * V).sum(1)
        return np.concatenate([m1, m2], axis=1)
    fit = np.isin(d["str_"], c["data"]["fit_subjects"])
    g0 = grid("C0")
    Ftr = feats(d["Xtr"][fit], g0)
    fm, fs = Ftr.mean(0), Ftr.std(0)
    fs[fs == 0] = 1.0
    torch.manual_seed(0)
    lin = torch.nn.Linear(18, 6)
    opt = torch.optim.Adam(lin.parameters(), lr=0.01)
    xt = torch.tensor((Ftr - fm) / fs, dtype=torch.float32)
    yt = torch.tensor(d["ytr"][fit])
    for _ in range(1000):
        loss = torch.nn.functional.cross_entropy(lin(xt), yt)
        opt.zero_grad()
        loss.backward()
        opt.step()
    res = {"train_ce_final": float(loss.item()), "n_fit": int(fit.sum())}
    yte = torch.tensor(d["yte"])
    F0 = feats(d["Xte"], g0)
    per_cond = {}
    with torch.inference_mode():
        for cn in CONDS:
            F = feats(d["Xte"], grid(cn))
            logits = lin(torch.tensor((F - fm) / fs, dtype=torch.float32))
            ce_w = torch.nn.functional.cross_entropy(logits, yte, reduction="none").numpy()
            acc_w = (logits.argmax(1) == yte).numpy().astype(float)
            subs = sorted(set(d["ste"].tolist()))
            if cn == "C0":
                ce_C0, logits_C0 = ce_w.copy(), logits.numpy().copy()
            per_cond[cn] = {"max_abs_feature_diff_vs_C0": float(np.abs(F - F0).max()),
                            "max_abs_logit_diff_vs_C0": float(np.abs(logits.numpy() - logits_C0).max()),
                            "max_abs_ce_diff_vs_C0": float(np.abs(ce_w - ce_C0).max()),
                            "ce_diff_vs_C0_subject_means": {str(s_): float((ce_w - ce_C0)[d["ste"] == s_].mean()) for s_ in subs},
                            "subject_mean_ce": {str(s_): float(ce_w[d["ste"] == s_].mean()) for s_ in subs},
                            "subject_mean_acc": {str(s_): float(acc_w[d["ste"] == s_].mean()) for s_ in subs}}
            per_cond[cn]["ce_diff_vs_C0_equal_weight_subjects"] = float(np.mean(list(per_cond[cn]["ce_diff_vs_C0_subject_means"].values())))
            per_cond[cn]["ce_equal_weight_subjects"] = float(np.mean(list(per_cond[cn]["subject_mean_ce"].values())))
            per_cond[cn]["acc_equal_weight_subjects"] = float(np.mean(list(per_cond[cn]["subject_mean_acc"].values())))
    wall, cpu_s = since(t)
    res.update({"conditions": per_cond, "wall_seconds": wall, "cpu_seconds": cpu_s, "status": "COMPLETED", "provenance": provenance(),
                "role": "representation-invariance control; not a task ceiling, SOTA or capacity-matched baseline"})
    wjson(os.path.join(OUT, "control.json"), res)
    ledger_add("control", wall, cpu_s, "COMPLETED")
    print(json.dumps({cn: {k: per_cond[cn][k] for k in ("max_abs_feature_diff_vs_C0", "ce_equal_weight_subjects", "acc_equal_weight_subjects")} for cn in CONDS}, indent=1))


# ----------------------------------------------------------------------------- aggregate

def stage_aggregate():
    import numpy as np
    c = cfg()
    ledger_guard("aggregate", 120, 240)
    t = clock()
    recs = [json.loads(l) for l in open(os.path.join(OUT, "eval.jsonl"))]
    ev = json.load(open(os.path.join(OUT, "eval_summary.json")))
    tr = json.load(open(os.path.join(OUT, "train.json")))
    subs = sorted(set(r["subject"] for r in recs))
    assert len(recs) == 2947, len(recs)
    assert subs == c["data"]["test_subjects"], subs
    assert ev["status"] == "COMPLETED" and ev["checkpoint_sha256"] == tr["checkpoint_sha256"] == sha256_file(os.path.join(OUT, "checkpoint.pt"))
    finite = all(math.isfinite(v) for r in recs for cn in CONDS for arm in ARMS
                 for v in ([r[f"{cn}/{arm}"]["ce"], r[f"{cn}/{arm}"]["tv_vs_C0"]] + r[f"{cn}/{arm}"]["logits"]))
    if not finite:
        wjson(os.path.join(OUT, "aggregate.json"), {"status": "FAIL(non-finite evaluation)"})
        ledger_add("aggregate", *since(t), "FAIL(non-finite evaluation)")
        raise SystemExit("FAIL(non-finite evaluation)")
    by = {s_: [r for r in recs if r["subject"] == s_] for s_ in subs}
    n_draws, boot_seed = 2000, 0
    idx = S.draws(len(subs), n_draws, boot_seed)

    def subj_means(fn):
        return [float(np.mean([fn(r) for r in by[s_]])) for s_ in subs]

    def summarize(vals):
        assert all(math.isfinite(v) for v in vals), "non-finite subject mean"
        est, ci = S.mean_ci(vals, idx)
        return {"per_subject": dict(zip(map(str, subs), vals)), "mean_over_subjects": est, "ci95_subject_bootstrap": ci,
                "n_positive": int(sum(v > 0 for v in vals)), "n_negative": int(sum(v < 0 for v in vals)),
                "min": min(vals), "max": max(vals), "degenerate_interval": bool(ci[0] == ci[1])}
    out = {"status": "COMPLETED", "n_subjects": len(subs), "n_windows": len(recs), "subjects": subs, "windows_per_subject": {str(s_): len(by[s_]) for s_ in subs},
           "bootstrap": {"draws": n_draws, "seed": boot_seed, "unit": "subject"},
           "note": f"all quantities are subject means first, then equal-weight means over the {len(subs)} test subjects; intervals are subject-level paired percentile bootstraps ({n_draws} draws, seed {boot_seed}) and are coarse with {len(subs)} clusters"}
    out["primary_D"] = summarize(subj_means(lambda r: r["H8/native_mean"]["ce"] - r["H8/native_time"]["ce"]))
    out["primary_D"]["definition"] = "d_s = mean_windows[CE(native_mean,H8) - CE(native_time,H8)]; D = mean over subjects (nats); D > 0: token mean worse"
    out["ce"] = {f"{cn}/{arm}": summarize(subj_means(lambda r, cn=cn, arm=arm: r[f"{cn}/{arm}"]["ce"])) for cn in CONDS for arm in ARMS}
    out["accuracy"] = {f"{cn}/{arm}": summarize(subj_means(lambda r, cn=cn, arm=arm: float(r[f"{cn}/{arm}"]["pred"] == r["label"]))) for cn in CONDS for arm in ARMS}
    out["risk_change_vs_C0"] = {f"{cn}/{arm}": summarize(subj_means(lambda r, cn=cn, arm=arm: r[f"{cn}/{arm}"]["ce"] - r["C0/native_mean"]["ce"]))
                                for cn in ("S8", "H8") for arm in ARMS}
    out["flip_vs_C0"] = {f"{cn}/{arm}": summarize(subj_means(lambda r, cn=cn, arm=arm: float(r[f"{cn}/{arm}"]["flip_vs_C0"]))) for cn in ("S8", "H8") for arm in ARMS}
    out["tv_vs_C0"] = {f"{cn}/{arm}": summarize(subj_means(lambda r, cn=cn, arm=arm: r[f"{cn}/{arm}"]["tv_vs_C0"])) for cn in ("S8", "H8") for arm in ARMS}

    def cl_mean(r, key):
        v = r[key]["centered_logit_rel_vs_C0"]
        return v
    excl_by = {str(s_): sum(1 for r in by[s_] if r["excluded_centered_logit_distance"]) for s_ in subs}
    out["centered_logit_rel_vs_C0"] = {"excluded_windows": sum(excl_by.values()), "excluded_windows_per_subject": excl_by,
                                       "retained_windows_per_subject": {str(s_): len(by[s_]) - excl_by[str(s_)] for s_ in subs}}
    if any(v == 0 for v in out["centered_logit_rel_vs_C0"]["retained_windows_per_subject"].values()):
        out["centered_logit_rel_vs_C0"]["status"] = "NA (a subject retained no window; estimand not computed)"
    else:
        for cn in ("S8", "H8"):
            for arm in ARMS:
                vals = [float(np.mean([r[f"{cn}/{arm}"]["centered_logit_rel_vs_C0"] for r in by[s_] if not r["excluded_centered_logit_distance"]])) for s_ in subs]
                out["centered_logit_rel_vs_C0"][f"{cn}/{arm}"] = summarize(vals)
    dec = {}
    dec["note"] = "W, M, Q, A_minus_B are pre-registered; M_end (M restricted to the 128 interval-end tokens) is a pre-registered secondary split of M"
    for key in ("W", "M", "M_end", "Q", "A_minus_B"):
        dec[key + "_norm"] = summarize(subj_means(lambda r, key=key: float(np.linalg.norm(r["H8_decomposition"][key]))))
        dec[key + "_vector_mean_over_subjects"] = np.mean([np.mean([r["H8_decomposition"][key] for r in by[s_]], axis=0) for s_ in subs], axis=0).tolist()
    dec["pmean_minus_pC0_norm"] = summarize(subj_means(lambda r: float(np.linalg.norm(np.asarray(r["H8/native_mean"]["logits"]) - np.asarray(r["C0/native_mean"]["logits"])))))
    dec["ptime_minus_pC0_norm"] = summarize(subj_means(lambda r: float(np.linalg.norm(np.asarray(r["H8/native_time"]["logits"]) - np.asarray(r["C0/native_mean"]["logits"])))))
    out["H8_logit_decomposition"] = dec
    out["numerical_checks"] = ev["numerical_checks"]
    out["eval_provenance"] = ev.get("provenance", "not recorded by the eval code version that ran (runner as of commit <commit>; see run_manifest.json)")
    out["identity_checks"] = ev["identity_checks"]
    out["timing"] = ev["timing"]
    if tr["config_sha256"] != config_sha():
        raise SystemExit("config changed since training")
    out["provenance"] = provenance()
    out["checkpoint"] = {"sha256": tr["checkpoint_sha256"], "selected_update": tr["selected_update"], "selected_cal_ce": tr["selected_cal_ce"],
                         "selected_cal_acc": tr["selected_cal_acc"], "updates_run": tr["updates_run"]}
    if os.path.exists(os.path.join(OUT, "control.json")):
        out["control"] = json.load(open(os.path.join(OUT, "control.json")))["conditions"]
    pd = out["primary_D"]
    lo, hi, D = pd["ci95_subject_bootstrap"][0], pd["ci95_subject_bootstrap"][1], pd["mean_over_subjects"]
    assert all(math.isfinite(v) for v in (lo, hi, D))
    if pd["degenerate_interval"]:
        primary = "degenerate subject-level interval (all subject values equal): no inference"
    elif lo > 0 and D > 0:
        primary = "D > 0 with the subject-level interval excluding 0"
    elif hi < 0 and D < 0:
        primary = "D < 0 with the subject-level interval excluding 0"
    else:
        primary = "interval includes 0"
    out["verdict"] = {"primary": primary, "signs": f"{pd['n_positive']} of {len(subs)} subjects positive, {pd['n_negative']} negative",
                      "scope": f"one dataset, one model, one seed; conditional on this checkpoint; {len(subs)} subject clusters"}
    wall, cpu_s = since(t)
    out["aggregate_wall_seconds"], out["aggregate_cpu_seconds"] = wall, cpu_s
    wjson(os.path.join(OUT, "aggregate.json"), out)
    ledger_add("aggregate", wall, cpu_s, "COMPLETED")
    tot_w, tot_p, _ = ledger_totals()
    print(json.dumps({"primary_D": {k: pd[k] for k in ("mean_over_subjects", "ci95_subject_bootstrap", "n_positive", "min", "max")},
                      "verdict": out["verdict"], "ledger_total": {"wall": tot_w, "cpu": tot_p}}, indent=1))


if __name__ == "__main__":
    ap = argparse.ArgumentParser()
    ap.add_argument("--stage", required=True, choices=["data", "smoke", "train", "eval", "control", "aggregate"])
    a = ap.parse_args()
    {"data": stage_data, "smoke": stage_smoke, "train": stage_train, "eval": stage_eval,
     "control": stage_control, "aggregate": stage_aggregate}[a.stage]()
