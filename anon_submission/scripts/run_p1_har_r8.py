"""R8-P1-HAR-REPLICATION: two extra model seeds under the frozen P1-HAR-01 protocol
(configs/auto_run_r8.json; read it first). Minimal extension of scripts/run_p1_har_01.py: the
data, splits, normalization, grids, model, optimizer, update count and checkpoint rule are the
P1-HAR-01 helpers unchanged; only the model seed differs (101, 102).

Stages: train:<seed>  eval:<seed>  aggregate  all
  train  : train_loop() of P1-HAR-01 with model_seed = <seed>; checkpoint saved and hashed
  eval   : one forward per condition (C0, S8, H8) in batches of 256; pooled logits, the S8
           common-end-point outputs and the H8 per-token decomposition are taken from the same
           passes and stored in float64; the resampled arm's input is checked equal to C0 and its
           logits are the C0 logits (no forward spent); no timing block
  aggregate : subject-first summaries per run, per-run subject bootstrap; the original run is
           copied from P1-HAR-01 / P1-HAR-CT-01 for the comparison table (never pooled)
A stage is skipped when its outputs exist with matching hashes; a lock file prevents a duplicate
process; every attempt (including failures) is a ledger row with wall and process CPU.
"""

import argparse
import copy
import hashlib
import io
import json
import math
import os
import sys
import time

T_START = (time.perf_counter(), time.process_time())  # before any heavy import
STARTED_UTC = time.strftime("%Y-%m-%dT%H:%M:%SZ", time.gmtime())

ROOT = os.path.dirname(os.path.dirname(os.path.abspath(__file__)))
sys.path.insert(0, os.path.join(ROOT, "src"))
sys.path.insert(0, os.path.join(ROOT, "scripts"))

from run_p1_har_01 import (OUT as HAR_OUT, grid, native_batch, locf_resample, build, setup, load_cache,  # noqa: E402
                           train_loop, eval_mode_checks, config_sha as p1_config_sha, CONDS)
from run_p1_real_01 import sha256_file, git  # noqa: E402
from scssm.real import stats as S  # noqa: E402

CFG = os.path.join(ROOT, "configs", "auto_run_r8.json")
P1CFG = os.path.join(ROOT, "configs", "p1_har_01.json")
OUT = os.path.join(ROOT, "results", "raw", "R8-P1-HAR-REPLICATION")
LEDGER = os.path.join(OUT, "cost_ledger.json")
P1_CFG_SHA = "e603cf75e6472a20547afe22b6056f8e940de93120bc9fd2e6f66806a7dc44fc"
ARMS5 = [("C0", "native_mean"), ("S8", "native_mean"), ("S8", "native_time"), ("H8", "native_mean"), ("H8", "native_time")]


def cfg():
    return json.load(open(CFG))


def since(t=None):
    t = t or T_START
    return round(time.perf_counter() - t[0], 3), round(time.process_time() - t[1], 3)


def clock():
    return time.perf_counter(), time.process_time()


def wjson(path, obj):
    os.makedirs(os.path.dirname(path), exist_ok=True)
    with open(path, "w") as f:
        json.dump(obj, f, indent=2)


def utc():
    return time.strftime("%Y-%m-%dT%H:%M:%SZ", time.gmtime())


# ----------------------------------------------------------------------------- ledger, guard, lock

def ledger_totals():
    rows = json.load(open(LEDGER)) if os.path.exists(LEDGER) else []
    return sum(r["wall_seconds"] for r in rows), sum(r["cpu_seconds"] for r in rows), rows


def ledger_add(stage, t0, started, status, extra=None):
    _, _, rows = ledger_totals()
    w, c_ = since(t0)
    row = {"stage": stage, "started_utc": started, "finished_utc": utc(), "wall_seconds": w, "cpu_seconds": c_,
           "threads": 2, "status": status, "note": "timer from stage start; the process-level timer (before torch import) is reported by the 'process' row"}
    if extra:
        row.update(extra)
    rows.append(row)
    wjson(LEDGER, rows)
    print(f"[ledger] {stage}: wall {w:.1f}s cpu {c_:.1f}s -> {status}", flush=True)


def guard(stage, proj_wall, proj_cpu):
    B = cfg()["budget"]
    w, c_, _ = ledger_totals()
    if w + proj_wall > B["wall_seconds_total"] or c_ + proj_cpu > B["process_cpu_seconds_total"]:
        raise SystemExit(f"BLOCKED({stage}): ledger {w:.0f}s/{c_:.0f}s + projection {proj_wall}/{proj_cpu} exceeds {B['wall_seconds_total']}/{B['process_cpu_seconds_total']}")


class Lock:
    def __init__(self, name):
        self.p = os.path.join(OUT, f"{name}.lock")

    def __enter__(self):
        os.makedirs(OUT, exist_ok=True)
        if os.path.exists(self.p):
            pid = int(open(self.p).read().strip() or 0)
            alive = pid and os.path.exists(f"/proc/{pid}")
            if alive:
                raise SystemExit(f"REFUSED: {self.p} held by live pid {pid}")
        open(self.p, "w").write(str(os.getpid()))
        return self

    def __exit__(self, *a):
        if os.path.exists(self.p):
            os.remove(self.p)


def runner_sha():
    return sha256_file(os.path.abspath(__file__))


def provenance():
    return {"git_head": git("rev-parse", "HEAD"), "git_dirty": bool(git("status", "--porcelain")), "runner_sha256": runner_sha(),
            "auto_run_config_sha256": sha256_file(CFG), "p1_har_01_config_sha256": p1_config_sha()}


def p1cfg_with_seed(seed):
    assert p1_config_sha() == P1_CFG_SHA, "configs/p1_har_01.json changed"
    c = json.load(open(P1CFG))
    c = copy.deepcopy(c)
    c["training"]["model_seed"] = int(seed)
    return c


# ----------------------------------------------------------------------------- train

def stage_train(seed):
    name = f"train:{seed}"
    sd = os.path.join(OUT, f"seed{seed}")
    tj = os.path.join(sd, "train.json")
    if os.path.exists(tj):
        t = json.load(open(tj))
        if t.get("status") == "COMPLETED" and t.get("p1_har_01_config_sha256") == P1_CFG_SHA and os.path.exists(os.path.join(sd, "checkpoint.pt")) \
                and sha256_file(os.path.join(sd, "checkpoint.pt")) == t.get("checkpoint_sha256"):
            print(f"[skip] {name}: completed (checkpoint {t['checkpoint_sha256'][:12]})")
            return
    guard(name, 300, 600)
    with Lock(name):
        t0, started = clock(), utc()
        try:
            c = p1cfg_with_seed(seed)
            torch, tides = setup(c)
            d, dh = load_cache(need_test=False)
            lines = [f"R8 seed {seed}; git HEAD {git('rev-parse', 'HEAD')}; p1 config {P1_CFG_SHA[:12]}; data {dh['sha256']['Xtr'][:12]}"]

            def log(m):
                lines.append(f"[{time.perf_counter() - t0[0]:8.2f}s] {m}")
                print(lines[-1], flush=True)
            r = train_loop(torch, tides, c, d, updates=c["training"]["updates_planned"], log=log, cal_every=100, time_cap=300)
            os.makedirs(sd, exist_ok=True)
            ck = os.path.join(sd, "checkpoint.pt")
            buf = io.BytesIO()
            torch.save(r["best"]["state"], buf)
            open(ck, "wb").write(buf.getvalue())
            sha = sha256_file(ck)
            open(ck + ".sha256", "w").write(f"{sha}  checkpoint.pt\n")
            out = {"status": r["status"], "model_seed": seed, "updates_run": len(r["upd_times"]), "updates_planned": c["training"]["updates_planned"],
                   "selected_update": r["best"]["update"], "selected_cal_ce": r["best"]["cal_ce"], "selected_cal_acc": r["best"]["cal_acc"],
                   "curve": r["curve"], "eval_mode_checks_all_ok": r["checks_ok"], "min_train_step_scale": r["min_train_step_scale"],
                   "n_parameters": r["n_params"], "n_fit_windows": r["n_fit"], "n_cal_windows": r["n_cal"],
                   "train_loop_wall_seconds": r["loop_wall"], "train_loop_cpu_seconds": r["loop_cpu"],
                   "checkpoint_sha256": sha, "data_sha256": dh["sha256"], "p1_har_01_config_sha256": P1_CFG_SHA,
                   "torch": torch.__version__, "threads": torch.get_num_threads(), "provenance": provenance(), "test_opened": False}
            wjson(tj, out)
            open(os.path.join(sd, "train.log"), "w").write("\n".join(lines) + "\n")
            ledger_add(name, t0, started, r["status"], {"selected_update": r["best"]["update"], "updates_run": len(r["upd_times"])})
            if r["status"] != "COMPLETED":
                raise SystemExit(f"{name}: {r['status']}")
        except SystemExit:
            raise
        except BaseException as e:  # noqa: BLE001
            ledger_add(name, t0, started, f"FAIL({type(e).__name__}: {str(e)[:200]})")
            raise


# ----------------------------------------------------------------------------- eval

def lsm(z):
    import numpy as np
    z = z - z.max(axis=-1, keepdims=True)
    return z - np.log(np.exp(z).sum(axis=-1, keepdims=True))


def center(z):
    return z - z.mean(axis=-1, keepdims=True)


def stage_eval(seed):
    import numpy as np
    name = f"eval:{seed}"
    sd = os.path.join(OUT, f"seed{seed}")
    tj = json.load(open(os.path.join(sd, "train.json")))
    es = os.path.join(sd, "eval_summary.json")
    if os.path.exists(es):
        e = json.load(open(es))
        if e.get("status") == "COMPLETED" and e.get("checkpoint_sha256") == tj["checkpoint_sha256"]:
            print(f"[skip] {name}: completed")
            return
    if tj["status"] != "COMPLETED":
        raise SystemExit(f"{name}: training not completed ({tj['status']})")
    guard(name, 200, 400)
    with Lock(name):
        t0, started = clock(), utc()
        try:
            c = p1cfg_with_seed(seed)
            torch, tides = setup(c)
            ck = os.path.join(sd, "checkpoint.pt")
            assert sha256_file(ck) == tj["checkpoint_sha256"], "checkpoint hash"
            d, dh = load_cache(need_test=True)
            tj["test_opened"] = True
            wjson(os.path.join(sd, "train.json"), tj)
            model = build(torch, tides, c)
            model.load_state_dict(torch.load(ck))
            model.eval()
            chk = eval_mode_checks(torch, model)
            assert chk["all_modules_eval"] and chk["dropout_p"] in ([0.0], []) and chk["batchnorm_running_stats"], "eval-mode check"
            assert isinstance(model.head, torch.nn.Linear)
            X, subj, y, mu, sdv = d["Xte"], d["ste"], d["yte"], d["mu"], d["sd"]
            N = X.shape[0]
            gs = {cn: grid(cn) for cn in CONDS}
            end_idx = [j for j, e in enumerate(gs["S8"]["is_end"]) if e]
            assert len(end_idx) == 128 and all(j == 8 * k + 7 for k, j in enumerate(end_idx))
            ends = torch.tensor(end_idx)
            wH = torch.tensor(gs["H8"]["w_time"], dtype=torch.float32)
            wS = torch.tensor(gs["S8"]["w_time"], dtype=torch.float32)
            baseH = np.asarray(gs["H8"]["base"]); endH = np.asarray(gs["H8"]["is_end"]); aH = 1.0 / gs["H8"]["L"]; wHn = np.asarray(gs["H8"]["w_time"])
            Z = {k: np.zeros((N, 6)) for k in ["C0/native_mean", "S8/native_mean", "S8/native_time", "H8/native_mean", "H8/native_time", "S8/z_end"]}
            Wd, Md, Mend, Qd = (np.zeros((N, 6)) for _ in range(4))
            feat_end = np.zeros(N)
            res_in = {"S8": 0.0, "H8": 0.0}
            with torch.inference_mode():
                for i in range(0, N, 256):
                    xb = X[i:i + 256]
                    sl = slice(i, i + len(xb))
                    u0, s0 = native_batch(torch, xb, gs["C0"], mu, sdv)
                    h0 = model.backbone(u0, step_scale=s0)
                    m0 = h0.mean(1)
                    Z["C0/native_mean"][sl] = model.head(m0).numpy().astype(np.float64)
                    tok0 = (h0 @ model.head.weight.T + model.head.bias).numpy().astype(np.float64)   # (b,128,6) C0 token logits
                    u8, s8 = native_batch(torch, xb, gs["S8"], mu, sdv)
                    h8 = model.backbone(u8, step_scale=s8)
                    Z["S8/native_mean"][sl] = model.head(h8.mean(1)).numpy().astype(np.float64)
                    Z["S8/native_time"][sl] = model.head(torch.einsum("nlc,l->nc", h8, wS)).numpy().astype(np.float64)
                    m_end = h8[:, ends, :].mean(1)
                    Z["S8/z_end"][sl] = model.head(m_end).numpy().astype(np.float64)
                    feat_end[sl] = ((h8[:, ends, :] - h0).norm(dim=2).mean(1) / h0.norm(dim=2).mean(1)).numpy()
                    uh, sh = native_batch(torch, xb, gs["H8"], mu, sdv)
                    hh = model.backbone(uh, step_scale=sh)
                    Z["H8/native_mean"][sl] = model.head(hh.mean(1)).numpy().astype(np.float64)
                    Z["H8/native_time"][sl] = model.head(torch.einsum("nlc,l->nc", hh, wH)).numpy().astype(np.float64)
                    tokH = (hh @ model.head.weight.T + model.head.bias).numpy().astype(np.float64)    # (b,576,6)
                    bj = tok0[:, baseH, :]
                    Wd[sl] = ((aH - wHn)[None, :, None] * bj).sum(1)
                    Md[sl] = (aH * (tokH - bj)).sum(1)
                    Mend[sl] = (aH * (tokH - bj) * endH[None, :, None]).sum(1)
                    Qd[sl] = (wHn[None, :, None] * (tokH - bj)).sum(1)
                    for cn in ("S8", "H8"):
                        xr = locf_resample(xb[:, np.asarray(gs[cn]["base"]), :], gs[cn])
                        res_in[cn] = max(res_in[cn], float(np.abs(xr - xb).max()))
                    for k in Z:
                        assert np.isfinite(Z[k][sl]).all(), f"non-finite logits in {k}"
            # per-window records
            z0 = Z["C0/native_mean"]; P0 = center(z0); n0 = np.linalg.norm(P0, axis=1); p0 = np.exp(lsm(z0))
            excluded = int((n0 < 1e-6).sum())
            A = tok0 = None  # noqa: F841 (not kept)
            recs = []
            dE = Z["S8/z_end"] - z0; dA = Z["S8/native_mean"] - z0; dX = Z["S8/native_mean"] - Z["S8/z_end"]
            cE, cA, cX = center(dE), center(dA), center(dX)
            for i in range(N):
                r = {"id": int(i), "subject": int(subj[i]), "label": int(y[i]), "excluded_centered_logit_distance": bool(n0[i] < 1e-6)}
                for cn, arm in ARMS5:
                    z = Z[f"{cn}/{arm}"][i]
                    l = lsm(z[None, :])[0]
                    r[f"{cn}/{arm}"] = {"logits": z.tolist(), "ce": float(-l[y[i]]), "pred": int(z.argmax()),
                                        "tv_vs_C0": float(0.5 * np.abs(np.exp(l) - p0[i]).sum()),
                                        "centered_logit_rel_vs_C0": (float(np.linalg.norm(center(z) - P0[i]) / n0[i]) if n0[i] >= 1e-6 else None),
                                        "flip_vs_C0": bool(z.argmax() != z0[i].argmax())}
                zE = Z["S8/z_end"][i]; lE = lsm(zE[None, :])[0]
                r["S8/common_time"] = {"z_end": zE.tolist(), "ce_z_end": float(-lE[y[i]]), "flip_end_vs_z0": bool(zE.argmax() != z0[i].argmax()),
                                       "c_rel_all": float(np.linalg.norm(cA[i]) / n0[i]), "c_rel_end": float(np.linalg.norm(cE[i]) / n0[i]),
                                       "c_rel_extra": float(np.linalg.norm(cX[i]) / n0[i]), "c_inner_end_extra": float(cE[i] @ cX[i]),
                                       "feat_per_endpoint_rel_change": float(feat_end[i])}
                r["H8_decomposition"] = {"W": Wd[i].tolist(), "M": Md[i].tolist(), "M_end": Mend[i].tolist(), "Q": Qd[i].tolist()}
                recs.append(r)
            ident = {"H8_pmean_minus_pC0_vs_W_plus_M_max_abs": float(np.abs(Z["H8/native_mean"] - z0 - Wd - Md).max()),
                     "H8_ptime_minus_pC0_vs_Q_max_abs": float(np.abs(Z["H8/native_time"] - z0 - Qd).max()),
                     "S8_native_mean_vs_native_time_max_abs": float(np.abs(Z["S8/native_mean"] - Z["S8/native_time"]).max()),
                     "resampled_input_vs_C0_max_abs": res_in}
            with open(os.path.join(sd, "eval.jsonl"), "w") as f:
                for r in recs:
                    f.write(json.dumps(r) + "\n")
            summ = {"status": "COMPLETED", "seed": seed, "checkpoint_sha256": tj["checkpoint_sha256"], "selected_update": tj["selected_update"],
                    "n_test_windows": int(N), "test_subjects": sorted(set(subj.tolist())), "test_subjects_reused_from_P1_HAR_01": True,
                    "eval_mode_checks": chk, "identity_checks": ident, "excluded_windows_centered_logit": excluded,
                    "resampled_arm": "input identical to C0 on S8 and H8 (max abs diff above); logits taken as the C0 logits (P1-HAR-01 verified bit-identical outputs for identical input); no separate forward",
                    "timing": "not re-measured (P1-HAR-01 timing applies)", "threads": torch.get_num_threads(), "provenance": provenance(),
                    "eval_wall_seconds": since(t0)[0], "eval_cpu_seconds": since(t0)[1]}
            wjson(es, summ)
            ledger_add(name, t0, started, "COMPLETED")
        except SystemExit:
            raise
        except BaseException as e:  # noqa: BLE001
            ledger_add(name, t0, started, f"FAIL({type(e).__name__}: {str(e)[:200]})")
            raise


# ----------------------------------------------------------------------------- aggregate

def summarize_run(recs, subs, idx):
    import numpy as np

    def sm(fn):
        return [float(np.mean([fn(r) for r in recs if r["subject"] == s_])) for s_ in subs]

    def summ(vals):
        assert all(math.isfinite(v) for v in vals)
        est, ci = S.mean_ci(vals, idx)
        return {"per_subject": dict(zip(map(str, subs), vals)), "mean_over_subjects": est, "ci95_subject_bootstrap": ci,
                "n_positive": int(sum(v > 0 for v in vals)), "n_negative": int(sum(v < 0 for v in vals)), "min": min(vals), "max": max(vals)}
    out = {"primary_D": summ(sm(lambda r: r["H8/native_mean"]["ce"] - r["H8/native_time"]["ce"]))}
    out["ce"] = {k: summ(sm(lambda r, k=k: r[k]["ce"])) for k in ("C0/native_mean", "S8/native_mean", "H8/native_mean", "H8/native_time")}
    out["accuracy"] = {k: summ(sm(lambda r, k=k: float(r[k]["pred"] == r["label"]))) for k in ("C0/native_mean", "S8/native_mean", "H8/native_mean", "H8/native_time")}
    out["risk_change_vs_C0"] = {k: summ(sm(lambda r, k=k: r[k]["ce"] - r["C0/native_mean"]["ce"])) for k in ("S8/native_mean", "H8/native_mean", "H8/native_time")}
    out["flip_vs_C0"] = {k: summ(sm(lambda r, k=k: float(r[k]["flip_vs_C0"]))) for k in ("S8/native_mean", "H8/native_mean", "H8/native_time")}
    out["tv_vs_C0"] = {k: summ(sm(lambda r, k=k: r[k]["tv_vs_C0"])) for k in ("S8/native_mean", "H8/native_mean", "H8/native_time")}
    out["centered_logit_rel_vs_C0"] = {k: summ(sm(lambda r, k=k: r[k]["centered_logit_rel_vs_C0"])) for k in ("S8/native_mean", "H8/native_mean", "H8/native_time")}
    out["H8_logit_decomposition"] = {key + "_norm": summ(sm(lambda r, key=key: float(np.linalg.norm(r["H8_decomposition"][key])))) for key in ("W", "M", "M_end", "Q")}
    out["H8_logit_decomposition"]["pmean_minus_pC0_norm"] = summ(sm(lambda r: float(np.linalg.norm(np.asarray(r["H8/native_mean"]["logits"]) - np.asarray(r["C0/native_mean"]["logits"])))))
    out["S8_common_time"] = {k: summ(sm(lambda r, k=k: float(r["S8/common_time"][k]))) for k in ("c_rel_all", "c_rel_end", "c_rel_extra", "c_inner_end_extra", "feat_per_endpoint_rel_change", "ce_z_end", "flip_end_vs_z0")}
    return out


def stage_aggregate():
    import numpy as np
    name = "aggregate"
    guard(name, 60, 120)
    t0, started = clock(), utc()
    c = cfg()
    runs = {}
    for run in c["runs"]:
        s_ = run["seed"]
        sd = os.path.join(OUT, f"seed{s_}")
        es = json.load(open(os.path.join(sd, "eval_summary.json")))
        tj = json.load(open(os.path.join(sd, "train.json")))
        assert es["status"] == "COMPLETED" and es["checkpoint_sha256"] == tj["checkpoint_sha256"] == sha256_file(os.path.join(sd, "checkpoint.pt"))
        recs = [json.loads(l) for l in open(os.path.join(sd, "eval.jsonl"))]
        subs = sorted(set(r["subject"] for r in recs))
        assert len(recs) == 2947 and subs == json.load(open(P1CFG))["data"]["test_subjects"]
        idx = S.draws(len(subs), 2000, 0)
        runs[f"seed{s_}"] = {"seed": s_, "checkpoint_sha256": tj["checkpoint_sha256"], "selected_update": tj["selected_update"],
                             "selected_cal_ce": tj["selected_cal_ce"], "selected_cal_acc": tj["selected_cal_acc"],
                             "excluded_windows": es["excluded_windows_centered_logit"], "identity_checks": es["identity_checks"], **summarize_run(recs, subs, idx)}
    # the original run, copied for the comparison table (not recomputed, not pooled)
    A1 = json.load(open(os.path.join(HAR_OUT, "aggregate.json")))
    CT = json.load(open(os.path.join(ROOT, "results", "raw", "P1-HAR-CT-01", "summary.json")))
    orig = {"seed": 0, "source": "P1-HAR-01 aggregate.json + P1-HAR-CT-01 summary.json (copied)", "checkpoint_sha256": A1["checkpoint"]["sha256"],
            "selected_update": A1["checkpoint"]["selected_update"], "selected_cal_ce": A1["checkpoint"]["selected_cal_ce"], "selected_cal_acc": A1["checkpoint"]["selected_cal_acc"],
            "primary_D": A1["primary_D"], "ce": {k: A1["ce"][k] for k in ("C0/native_mean", "S8/native_mean", "H8/native_mean", "H8/native_time")},
            "accuracy": {k: A1["accuracy"][k] for k in ("C0/native_mean", "S8/native_mean", "H8/native_mean", "H8/native_time")},
            "risk_change_vs_C0": {k: A1["risk_change_vs_C0"][k] for k in ("S8/native_mean", "H8/native_mean", "H8/native_time")},
            "flip_vs_C0": {k: A1["flip_vs_C0"][k] for k in ("S8/native_mean", "H8/native_mean", "H8/native_time")},
            "tv_vs_C0": {k: A1["tv_vs_C0"][k] for k in ("S8/native_mean", "H8/native_mean", "H8/native_time")},
            "centered_logit_rel_vs_C0": {k: A1["centered_logit_rel_vs_C0"][k] for k in ("S8/native_mean", "H8/native_mean", "H8/native_time")},
            "H8_logit_decomposition": {k: A1["H8_logit_decomposition"][k] for k in ("W_norm", "M_norm", "M_end_norm", "Q_norm", "pmean_minus_pC0_norm")},
            "S8_common_time": {k: {"per_subject": CT["per_subject"][k], "mean_over_subjects": CT["subject_equal_weight_mean"][k], "min": CT["subject_min"][k], "max": CT["subject_max"][k]}
                               for k in ("c_rel_all", "c_rel_end", "c_rel_extra", "c_inner_end_extra", "feat_per_endpoint_rel_change", "ce_z_end", "flip_end_vs_z0")}}
    out = {"status": "COMPLETED", "label": "initialization sensitivity under the frozen P1-HAR-01 protocol; same data and the same 9 test subjects; runs are separate rows, never pooled",
           "runs": {"seed0_original": orig, **runs}, "bootstrap": {"draws": 2000, "seed": 0, "unit": "subject", "per_run": True},
           "provenance": provenance(), "aggregate_wall_seconds": since(t0)[0]}
    wjson(os.path.join(OUT, "aggregate.json"), out)
    ledger_add(name, t0, started, "COMPLETED")
    # bookkeeping fields of the auto-run config (the only post-execution change to it)
    ac = cfg()
    ac["completed_stages"] = {f"train:{r['seed']}": sha256_file(os.path.join(OUT, f"seed{r['seed']}", "train.json")) for r in c["runs"]}
    ac["completed_stages"].update({f"eval:{r['seed']}": sha256_file(os.path.join(OUT, f"seed{r['seed']}", "eval_summary.json")) for r in c["runs"]})
    ac["completed_stages"]["aggregate"] = sha256_file(os.path.join(OUT, "aggregate.json"))
    ac["next_command"] = "none: the finite stage list is complete; no further stage is queued"
    ac["status"] = "COMPLETED (completed_stages and next_command were filled in after execution; everything else is the pre-execution content)"
    json.dump(ac, open(CFG, "w"), indent=2, ensure_ascii=False)
    for s_, r in runs.items():
        print(s_, "D", round(r["primary_D"]["mean_over_subjects"], 4), r["primary_D"]["ci95_subject_bootstrap"], "+", r["primary_D"]["n_positive"],
              "| S8 cl", round(r["centered_logit_rel_vs_C0"]["S8/native_mean"]["mean_over_subjects"], 4), "common", round(r["S8_common_time"]["c_rel_end"]["mean_over_subjects"], 4),
              "extra", round(r["S8_common_time"]["c_rel_extra"]["mean_over_subjects"], 4), "| W", round(r["H8_logit_decomposition"]["W_norm"]["mean_over_subjects"], 3),
              "M", round(r["H8_logit_decomposition"]["M_norm"]["mean_over_subjects"], 3))


if __name__ == "__main__":
    ap = argparse.ArgumentParser()
    ap.add_argument("--stage", required=True, help="train:<seed> | eval:<seed> | aggregate | all")
    a = ap.parse_args()
    c = cfg()
    allowed = set(c["allowed_stages"])
    stages = c["allowed_stages"] if a.stage == "all" else [a.stage]
    for st in stages:
        assert st in allowed, f"stage {st} not in the allowed list"
        if st.startswith("train:"):
            stage_train(int(st.split(":")[1]))
        elif st.startswith("eval:"):
            stage_eval(int(st.split(":")[1]))
        else:
            stage_aggregate()
    w, c_ = since()
    _, _, rows = ledger_totals()
    rows.append({"stage": "process", "started_utc": STARTED_UTC, "finished_utc": utc(), "wall_seconds": w, "cpu_seconds": c_, "threads": 2,
                 "status": f"process total for stages {stages} (timer before torch import; overlaps the stage rows, not additive)"})
    wjson(LEDGER, rows)
