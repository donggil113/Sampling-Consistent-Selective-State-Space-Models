"""Run P1-COMP-01 (configs/p1_comp_01.json) and append its manifest entry.

Outputs: results/raw/P1-COMP-01.jsonl, results/raw/P1-COMP-01__hypotheses.json,
         results/raw/P1-COMP-01__run.log, run_manifest.json["subsequent_runs"]
"""

import json
import os
import sys
import time

ROOT = os.path.dirname(os.path.dirname(os.path.abspath(__file__)))
sys.path.insert(0, os.path.join(ROOT, "src"))
sys.path.insert(0, os.path.join(ROOT, "scripts"))

from run_first_run import git, sha256  # noqa: E402  (reuse FIRST_RUN helpers)
from scssm.cascade import run_coupled_exact, run_coupled_rk4, run_layerwise  # noqa: E402
from scssm.experiments import make_unit  # noqa: E402
from scssm.grids import split_grid  # noqa: E402
from scssm.metrics import at_index, loglog_slope, rel_l2  # noqa: E402

CFG = os.path.join(ROOT, "configs", "p1_comp_01.json")
FR_CFG = os.path.join(ROOT, "configs", "first_run.json")
RAW = os.path.join(ROOT, "results", "raw")


def main():
    dirty = git("status", "--porcelain", "--untracked-files=all")
    try:
        os.sched_setaffinity(0, set(sorted(os.sched_getaffinity(0))[:2]))
        affinity = sorted(os.sched_getaffinity(0))
    except (AttributeError, OSError) as e:
        affinity = f"UNAVAILABLE({e})"
    cfg = json.load(open(CFG))
    fr = json.load(open(FR_CFG))
    t0 = time.time()
    log_lines = []

    def log(msg):
        line = f"[{time.time() - t0:8.3f}s] {msg}"
        print(line, flush=True)
        log_lines.append(line)

    log(f"git HEAD {git('rev-parse', 'HEAD')} dirty={bool(dirty)}; affinity {affinity}")
    _, _, base = make_unit(0, fr, "uniform", False)  # FIRST_RUN unit 0: existing path and grid
    ms = cfg["input"]["m_values"]
    recs = []
    for case, p in cfg["system"]["cases"].items():
        a, b, c, d = p["a"], p["b"], p["c"], p["d"]
        rk1, rk2 = run_coupled_rk4(base.times, base.values, a, b, c, d, 4096)
        ex_by_m, lw_by_m = {}, {}
        for m in ms:
            G = split_grid(base, m)
            e1, e2 = run_coupled_exact(G.times, G.values, a, b, c, d)
            l1, l2 = run_layerwise(G.times, G.values, a, b, c, d)
            ex_by_m[m] = (at_index(e1, G.base_index), at_index(e2, G.base_index))
            lw_by_m[m] = (at_index(l1, G.base_index), at_index(l2, G.base_index))
        for m in ms:
            e1, e2 = ex_by_m[m]
            l1, l2 = lw_by_m[m]
            recs.append({
                "exp": "P1-COMP-01", "case": case, "a": a, "b": b, "c": c, "d": d, "m": m,
                "coupled_invariance": rel_l2(e2, ex_by_m[1][1]),
                "coupled_vs_rk4": rel_l2(e2, rk2),
                "coupled_vs_rk4_h1": rel_l2(e1, rk1),
                "layer1_identity": rel_l2(l1, e1),
                "layerwise_error": rel_l2(l2, e2),
                "layerwise_change": rel_l2(l2, lw_by_m[1][1]),
                "h2_exact_base": e2, "h2_layerwise_base": l2,
            })
    wall = time.time() - t0

    H = []
    for case in cfg["system"]["cases"]:
        rc = [r for r in recs if r["case"] == case]
        inv = max(r["coupled_invariance"] for r in rc if r["m"] > 1)
        rk = max(r["coupled_vs_rk4"] for r in rc)
        l1 = max(r["layer1_identity"] for r in rc)
        errs = [next(r["layerwise_error"] for r in rc if r["m"] == m) for m in ms]
        slope = loglog_slope(ms, errs)
        for hid, ok, ev in (
            ("COMP-H1", inv <= 1e-12, {"max_coupled_invariance": inv}),
            ("COMP-H2", rk <= 1e-9, {"max_coupled_vs_rk4": rk}),
            ("COMP-H3", l1 <= 1e-12, {"max_layer1_identity": l1}),
            ("COMP-H4", slope is not None and -1.2 <= slope <= -0.8, {"slope": slope, "errors": dict(zip(ms, errs))}),
            ("COMP-H5", errs[0] >= 1e-3, {"layerwise_error_m1": errs[0]}),
        ):
            H.append({"id": hid, "case": case, "status": "PASS" if ok else "FAIL", "evidence": ev})
    for h in H:
        log(f"{h['id']} [{h['case']}]: {h['status']} {json.dumps(h['evidence'])}")
    within = wall <= cfg["budget"]["wall_clock_seconds_total"]
    log(f"wall clock {wall:.3f}s (budget {cfg['budget']['wall_clock_seconds_total']}s) within={within}")

    os.makedirs(RAW, exist_ok=True)
    out = os.path.join(RAW, "P1-COMP-01.jsonl")
    with open(out, "w") as f:
        for r in recs:
            f.write(json.dumps(r) + "\n")
    hyp = os.path.join(RAW, "P1-COMP-01__hypotheses.json")
    with open(hyp, "w") as f:
        json.dump({"hypotheses": H, "kind": "STANDARD-PROPERTY CHECK"}, f, indent=2)
    logp = os.path.join(RAW, "P1-COMP-01__run.log")
    with open(logp, "w") as f:
        f.write("\n".join(log_lines) + "\n")

    man_path = os.path.join(ROOT, "run_manifest.json")
    man = json.load(open(man_path))
    entry = {
        "run_id": cfg["run_id"],
        "started_utc": time.strftime("%Y-%m-%dT%H:%M:%SZ", time.gmtime(t0)),
        "wall_clock_seconds": round(wall, 3),
        "within_budget": within,
        "git_head": git("rev-parse", "HEAD"),
        "git_dirty_at_start": bool(dirty),
        "git_status_at_start": dirty.splitlines(),
        "config": {"path": "configs/p1_comp_01.json", "sha256": sha256(CFG)},
        "code_sha256": {os.path.relpath(p, ROOT): sha256(p) for p in (
            os.path.join(ROOT, "src", "scssm", "cascade.py"), os.path.join(ROOT, "src", "scssm", "model.py"),
            os.path.abspath(__file__))},
        "cpu_affinity": affinity,
        "outputs_sha256": {os.path.relpath(p, ROOT): sha256(p) for p in (out, hyp)},
    }
    man.setdefault("subsequent_runs", [])
    man["subsequent_runs"] = [r for r in man["subsequent_runs"] if r.get("run_id") != cfg["run_id"]] + [entry]
    with open(man_path, "w") as f:
        json.dump(man, f, indent=2)


if __name__ == "__main__":
    main()
