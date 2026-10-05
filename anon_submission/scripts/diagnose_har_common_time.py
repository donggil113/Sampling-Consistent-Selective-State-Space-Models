"""P1-HAR-CT-01: common-time vs extra-sampling decomposition of the pooled S8 change of the
P1-HAR-01 checkpoint (configs/p1_har_ct_01.json; EXPLORATORY_FIXED_CHECKPOINT, post hoc).

Stage A re-aggregates the STORED float16 per-token logits and records their reproduction gap.
Stage B runs ONE float32 streaming forward of C0 and S8 over the 2947 test windows (batches of
256, no intermediate tensors stored) and computes, per window,
    z0 = A(mean_k H0[k]),   z_end = A(mean_k H8[end(k)]),   z_all = A(mean_j H8[j]),
    z_all - z0 = (z_end - z0) + (z_all - z_end),
with A the official affine head, plus class-centered versions, norms, the cross inner
product, common-end-point feature changes, CE/TV/flip, and subject-first summaries. No new
significance test, primary estimand or subject exclusion. The fp64 check runs only under
the pre-specified trigger rule.
"""

import hashlib
import json
import math
import os
import sys
import time

T_START = (time.perf_counter(), time.process_time())  # before any heavy import or hashing
STARTED_UTC = time.strftime("%Y-%m-%dT%H:%M:%SZ", time.gmtime())

ROOT = os.path.dirname(os.path.dirname(os.path.abspath(__file__)))
sys.path.insert(0, os.path.join(ROOT, "src"))
sys.path.insert(0, os.path.join(ROOT, "scripts"))

from run_p1_har_01 import OUT as HAR_OUT, DERIVED, grid, native_batch, build, setup, load_cache, eval_mode_checks  # noqa: E402
from run_p1_real_01 import sha256_file, git  # noqa: E402

CFG = os.path.join(ROOT, "configs", "p1_har_ct_01.json")
OUT = os.path.join(ROOT, "results", "raw", "P1-HAR-CT-01")
LEDGER = os.path.join(OUT, "cost_ledger.json")
C = 6


def since():
    return round(time.perf_counter() - T_START[0], 3), round(time.process_time() - T_START[1], 3)


def wjson(path, obj):
    os.makedirs(os.path.dirname(path), exist_ok=True)
    with open(path, "w") as f:
        json.dump(obj, f, indent=2)


def ledger_add(status, extra=None):
    rows = json.load(open(LEDGER)) if os.path.exists(LEDGER) else []
    w, c = since()
    row = {"stage": "diagnostic", "started_utc": STARTED_UTC, "finished_utc": time.strftime("%Y-%m-%dT%H:%M:%SZ", time.gmtime()),
           "wall_seconds": w, "cpu_seconds": c, "threads": 2, "status": status,
           "note": "timer started before importing torch and before hashing"}
    if extra:
        row.update(extra)
    rows.append(row)
    wjson(LEDGER, rows)
    print(f"[ledger] {status}: wall {w:.1f}s cpu {c:.1f}s")


def lsm(z):
    import numpy as np
    z = z - z.max(axis=-1, keepdims=True)
    return z - np.log(np.exp(z).sum(axis=-1, keepdims=True))


def main():
    import numpy as np
    cfg = json.load(open(CFG))
    B = cfg["budget"]
    # ------------------------------------------------------------------ hashes of the frozen inputs
    ck = os.path.join(HAR_OUT, "checkpoint.pt")
    assert sha256_file(ck) == "e3b414f5bdf55541d84a2f3abaf5a1c04faddfd6e2fdfd38dd79a20be049910d", "checkpoint hash"
    assert sha256_file(os.path.join(ROOT, "configs", "p1_har_01.json")) == "e603cf75e6472a20547afe22b6056f8e940de93120bc9fd2e6f66806a7dc44fc", "config hash"
    c1 = json.load(open(os.path.join(ROOT, "configs", "p1_har_01.json")))
    recs = [json.loads(l) for l in open(os.path.join(HAR_OUT, "eval.jsonl"))]
    assert len(recs) == 2947
    ids = np.array([r["id"] for r in recs])
    assert (ids == np.arange(2947)).all()
    subj = np.array([r["subject"] for r in recs])
    y = np.array([r["label"] for r in recs])
    z0_stored = np.array([r["C0/native_mean"]["logits"] for r in recs], dtype=np.float64)
    zall_stored = np.array([r["S8/native_mean"]["logits"] for r in recs], dtype=np.float64)
    subs = sorted(set(subj.tolist()))
    assert subs == c1["data"]["test_subjects"]
    # ------------------------------------------------------------------ endpoint map (checked)
    g0, g8 = grid("C0"), grid("S8")
    end_idx = [j for j, e in enumerate(g8["is_end"]) if e]
    assert len(end_idx) == 128 and len(g8["is_end"]) == 1024
    for k, j in enumerate(end_idx):
        assert j == 8 * k + 7 and g8["base"][j] == k, (k, j)
        assert abs(g8["times"][j + 1] - (k + 1) * 0.02) < 1e-12 and abs(g0["times"][k + 1] - (k + 1) * 0.02) < 1e-12
    # ------------------------------------------------------------------ fp64 selection rule (ids fixed now)
    fp64_ids = []
    for s_ in subs:
        wins = [int(i) for i in ids[subj == s_]]
        wins.sort(key=lambda i: hashlib.sha256(f"P1-HAR-CT-01:fp64:{i}".encode()).hexdigest())
        fp64_ids += wins[:2]
    # ------------------------------------------------------------------ stage A: stored float16 logits
    import gzip
    with gzip.open(os.path.join(HAR_OUT, "C0_token_logits.f16.npy.gz"), "rb") as f:
        c0tok = np.load(f).astype(np.float64)
    s8path = os.path.join(DERIVED, "S8_token_logits.f16.npy")
    stageA = {"C0_token_logits": "results (float16)", "S8_token_logits": s8path if os.path.exists(s8path) else "MISSING (local file not present)"}
    zA0 = c0tok.mean(1)
    stageA["C0_max_abs_vs_stored_pooled"] = float(np.abs(zA0 - z0_stored).max())
    if os.path.exists(s8path):
        s8tok = np.load(s8path).astype(np.float64)
        zA_all = s8tok.mean(1)
        zA_end = s8tok[:, end_idx, :].mean(1)
        stageA["S8_all_max_abs_vs_stored_pooled"] = float(np.abs(zA_all - zall_stored).max())
        stageA["S8_common_time_change_centered_rel_mean_window"] = float(np.mean(np.linalg.norm(center(zA_end - zA0), axis=1) / np.linalg.norm(center(zA0), axis=1)))
        stageA["note"] = "float16 storage limits these to about 1e-3 absolute; Stage B (float32 forward) gives the reported values"
        del s8tok
    wjson(os.path.join(OUT, "stageA_float16_reproduction.json"), stageA)
    print("stage A", json.dumps({k: v for k, v in stageA.items() if "max_abs" in k}))
    # ------------------------------------------------------------------ stage B: one float32 streaming forward
    w_now, c_now = since()
    proj = 45.0
    if w_now + 3 * proj > B["wall_seconds_total"] or c_now + 6 * proj > B["process_cpu_seconds_total"]:
        ledger_add("BLOCKED(projection would exceed the cap)")
        raise SystemExit("blocked")
    torch, tides = setup(c1)
    d, _ = load_cache(need_test=True)
    X, mu, sd = d["Xte"], d["mu"], d["sd"]
    model = build(torch, tides, c1)
    model.load_state_dict(torch.load(ck))
    model.eval()
    chk = eval_mode_checks(torch, model)
    assert chk["all_modules_eval"] and chk["dropout_p"] in ([0.0], []) and chk["batchnorm_running_stats"]
    assert isinstance(model.head, torch.nn.Linear)
    N = X.shape[0]
    Z0 = np.zeros((N, C)); ZE = np.zeros((N, C)); ZA = np.zeros((N, C))
    fpool = np.zeros(N); fpt = np.zeros(N); fp0 = np.zeros(N)
    ends = torch.tensor(end_idx)
    with torch.inference_mode():
        for i in range(0, N, 256):
            xb = X[i:i + 256]
            u0, s0 = native_batch(torch, xb, g0, mu, sd)
            u8, s8 = native_batch(torch, xb, g8, mu, sd)
            h0 = model.backbone(u0, step_scale=s0)                     # (b, 128, 16)
            h8 = model.backbone(u8, step_scale=s8)                     # (b, 1024, 16)
            m0, m_end, m_all = h0.mean(1), h8[:, ends, :].mean(1), h8.mean(1)
            Z0[i:i + 256] = model.head(m0).numpy().astype(np.float64)
            ZE[i:i + 256] = model.head(m_end).numpy().astype(np.float64)
            ZA[i:i + 256] = model.head(m_all).numpy().astype(np.float64)
            fpool[i:i + 256] = (m_end - m0).norm(dim=1).numpy()
            fpt[i:i + 256] = (h8[:, ends, :] - h0).norm(dim=2).mean(1).numpy()
            fp0[i:i + 256] = h0.norm(dim=2).mean(1).numpy()
    repro = {"z0_max_abs_vs_stored_C0_native_mean": float(np.abs(Z0 - z0_stored).max()),
             "z_all_max_abs_vs_stored_S8_native_mean": float(np.abs(ZA - zall_stored).max())}
    print("stage B reproduction", repro)
    # ------------------------------------------------------------------ decomposition (vector level)
    d_all, d_end, d_extra = ZA - Z0, ZE - Z0, ZA - ZE
    resid = d_all - d_end - d_extra
    P0, PE, PA = center(Z0), center(ZE), center(ZA)
    cd_all, cd_end, cd_extra = PA - P0, PE - P0, PA - PE
    n0 = np.linalg.norm(P0, axis=1)
    rows = []
    for i in range(N):
        rows.append({"id": int(i), "subject": int(subj[i]), "label": int(y[i]),
                     "z0": Z0[i].tolist(), "z_end": ZE[i].tolist(), "z_all": ZA[i].tolist(),
                     "identity_residual_max_abs": float(np.abs(resid[i]).max()),
                     "norm_all": float(np.linalg.norm(d_all[i])), "norm_end": float(np.linalg.norm(d_end[i])), "norm_extra": float(np.linalg.norm(d_extra[i])),
                     "c_norm_all": float(np.linalg.norm(cd_all[i])), "c_norm_end": float(np.linalg.norm(cd_end[i])), "c_norm_extra": float(np.linalg.norm(cd_extra[i])),
                     "c_norm_z0": float(n0[i]),
                     "c_rel_all": float(np.linalg.norm(cd_all[i]) / n0[i]), "c_rel_end": float(np.linalg.norm(cd_end[i]) / n0[i]), "c_rel_extra": float(np.linalg.norm(cd_extra[i]) / n0[i]),
                     "c_inner_end_extra": float(cd_end[i] @ cd_extra[i]), "inner_end_extra": float(d_end[i] @ d_extra[i]),
                     "feat_pooled_end_change": float(fpool[i]), "feat_per_endpoint_rel_change": float(fpt[i] / fp0[i]),
                     "ce_z0": float(-lsm(Z0[i])[y[i]]), "ce_z_end": float(-lsm(ZE[i])[y[i]]), "ce_z_all": float(-lsm(ZA[i])[y[i]]),
                     "tv_end_vs_z0": float(0.5 * np.abs(np.exp(lsm(ZE[i])) - np.exp(lsm(Z0[i]))).sum()),
                     "tv_all_vs_z0": float(0.5 * np.abs(np.exp(lsm(ZA[i])) - np.exp(lsm(Z0[i]))).sum()),
                     "flip_end_vs_z0": bool(ZE[i].argmax() != Z0[i].argmax()), "flip_all_vs_z0": bool(ZA[i].argmax() != Z0[i].argmax())})
    with open(os.path.join(OUT, "per_window.jsonl"), "w") as f:
        for r in rows:
            f.write(json.dumps(r) + "\n")
    keys = ["norm_all", "norm_end", "norm_extra", "c_norm_all", "c_norm_end", "c_norm_extra", "c_norm_z0", "c_rel_all", "c_rel_end", "c_rel_extra",
            "c_inner_end_extra", "inner_end_extra", "feat_pooled_end_change", "feat_per_endpoint_rel_change",
            "ce_z0", "ce_z_end", "ce_z_all", "tv_end_vs_z0", "tv_all_vs_z0", "flip_end_vs_z0", "flip_all_vs_z0", "identity_residual_max_abs"]
    summary = {"status": "COMPLETED", "label": "EXPLORATORY_FIXED_CHECKPOINT (post hoc design)", "config_sha256": sha256_file(CFG),
               "checkpoint_sha256": sha256_file(ck), "original_config_sha256": sha256_file(os.path.join(ROOT, "configs", "p1_har_01.json")),
               "runner_sha256": sha256_file(os.path.abspath(__file__)), "git_head": git("rev-parse", "HEAD"), "git_dirty": bool(git("status", "--porcelain")),
               "n_windows": int(N), "subjects": subs, "endpoint_map": "S8 token 8k+7 (0-based) <-> C0 token k; 128 end points verified against time stamps",
               "eval_mode_checks": chk, "stage_A": stageA, "stage_B_reproduction": repro,
               "identity_residual_max_abs_all_windows": float(np.abs(resid).max()),
               "per_subject": {}, "subject_equal_weight_mean": {}, "subject_min": {}, "subject_max": {}, "window_pooled_mean_for_reference": {}}
    for k in keys:
        vals = {str(s_): float(np.mean([float(r[k]) for r in rows if r["subject"] == s_])) for s_ in subs}
        summary["per_subject"][k] = vals
        summary["subject_equal_weight_mean"][k] = float(np.mean(list(vals.values())))
        summary["subject_min"][k] = float(min(vals.values()))
        summary["subject_max"][k] = float(max(vals.values()))
        summary["window_pooled_mean_for_reference"][k] = float(np.mean([float(r[k]) for r in rows]))
    # subject-mean vectors of the centered terms (signed), for the record
    summary["centered_term_subject_mean_vectors"] = {
        "d_end": np.mean([cd_end[subj == s_].mean(0) for s_ in subs], axis=0).tolist(),
        "d_extra": np.mean([cd_extra[subj == s_].mean(0) for s_ in subs], axis=0).tolist(),
        "d_all": np.mean([cd_all[subj == s_].mean(0) for s_ in subs], axis=0).tolist()}
    # ------------------------------------------------------------------ fp64 trigger rule
    rel_end = summary["subject_equal_weight_mean"]["c_rel_end"]
    trig = rel_end < 1e-3 or summary["identity_residual_max_abs_all_windows"] > 1e-4
    summary["fp64_check"] = {"selected_window_ids": fp64_ids, "trigger_rule": cfg["fp64_check"]["trigger_rule_fixed_before_execution"],
                             "observed_subject_mean_c_rel_end": rel_end, "observed_identity_residual_max": summary["identity_residual_max_abs_all_windows"],
                             "status": "RUN" if trig else "NOT_NEEDED"}
    if trig:
        m64 = build(torch, tides, c1)
        m64.load_state_dict(torch.load(ck))
        m64 = m64.double().eval()
        sel = np.array(fp64_ids)
        out64 = {}
        with torch.inference_mode():
            u0, s0 = native_batch(torch, X[sel], g0, mu, sd)
            u8, s8 = native_batch(torch, X[sel], g8, mu, sd)
            h0 = m64.backbone(u0.double(), step_scale=s0.double())
            h8 = m64.backbone(u8.double(), step_scale=s8.double())
            z0_64 = m64.head(h0.mean(1)).numpy(); ze_64 = m64.head(h8[:, ends, :].mean(1)).numpy(); za_64 = m64.head(h8.mean(1)).numpy()
        out64 = {"ids": fp64_ids, "max_abs_z0_fp64_vs_fp32": float(np.abs(z0_64 - Z0[sel]).max()), "max_abs_z_end_fp64_vs_fp32": float(np.abs(ze_64 - ZE[sel]).max()),
                 "max_abs_z_all_fp64_vs_fp32": float(np.abs(za_64 - ZA[sel]).max()),
                 "c_rel_end_fp64_mean": float(np.mean(np.linalg.norm(center(ze_64 - z0_64), axis=1) / np.linalg.norm(center(z0_64), axis=1))),
                 "c_rel_end_fp32_mean_same_windows": float(np.mean([rows[i]["c_rel_end"] for i in sel]))}
        wjson(os.path.join(OUT, "fp64_check.json"), out64)
        summary["fp64_check"].update(out64)
    w, c_ = since()
    summary["wall_seconds"], summary["cpu_seconds"], summary["threads"] = w, c_, torch.get_num_threads()
    wjson(os.path.join(OUT, "summary.json"), summary)
    ledger_add("COMPLETED", {"fp64": summary["fp64_check"]["status"]})
    m = summary["subject_equal_weight_mean"]
    print(json.dumps({k: round(m[k], 5) for k in ("c_rel_all", "c_rel_end", "c_rel_extra", "c_inner_end_extra", "feat_per_endpoint_rel_change", "ce_z0", "ce_z_end", "ce_z_all", "flip_end_vs_z0", "flip_all_vs_z0")}, indent=1))


def center(z):
    return z - z.mean(axis=-1, keepdims=True)


if __name__ == "__main__":
    try:
        main()
    except SystemExit:
        raise
    except BaseException as e:  # noqa: BLE001
        ledger_add(f"FAIL({type(e).__name__}: {str(e)[:200]})")
        raise
