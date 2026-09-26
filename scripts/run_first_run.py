"""Run FIRST_RUN under the pre-registered CPU budget and write raw results.

Usage: python3 scripts/run_first_run.py
Outputs: results/raw/*.jsonl, results/raw/hypotheses.json, results/raw/run.log,
         run_manifest.json
"""

import hashlib
import json
import os
import platform
import subprocess
import sys
import time

ROOT = os.path.dirname(os.path.dirname(os.path.abspath(__file__)))
sys.path.insert(0, os.path.join(ROOT, "src"))

for var in ("OMP_NUM_THREADS", "OPENBLAS_NUM_THREADS", "MKL_NUM_THREADS"):
    os.environ[var] = "2"

from scssm import experiments as ex  # noqa: E402
from scssm.hypotheses import evaluate  # noqa: E402

CFG_PATH = os.path.join(ROOT, "configs", "first_run.json")
RAW = os.path.join(ROOT, "results", "raw")


def sha256(path):
    with open(path, "rb") as f:
        return hashlib.sha256(f.read()).hexdigest()


def git(*args):
    try:
        return subprocess.check_output(["git", *args], cwd=ROOT, text=True).strip()
    except Exception as e:  # pragma: no cover
        return f"UNAVAILABLE({e})"


def main():
    try:
        os.sched_setaffinity(0, set(sorted(os.sched_getaffinity(0))[:2]))
        affinity = sorted(os.sched_getaffinity(0))
    except (AttributeError, OSError) as e:
        affinity = f"UNAVAILABLE({e})"
    with open(CFG_PATH) as f:
        cfg = json.load(f)
    budget = cfg["budget"]["wall_clock_seconds_total"]
    lo, hi = cfg["units"]["primary_seeds"]
    primary = list(range(lo, hi + 1))
    secondary = list(range(0, 8))
    # check the tree BEFORE this script creates any output file
    dirty = git("status", "--porcelain", "--untracked-files=all")
    os.makedirs(RAW, exist_ok=True)
    log_path = os.path.join(RAW, "run.log")
    logf = open(log_path, "w")
    t_start = time.time()

    def log(msg):
        line = f"[{time.time() - t_start:8.3f}s] {msg}"
        print(line, flush=True)
        logf.write(line + "\n")
        logf.flush()

    log(f"git HEAD {git('rev-parse', 'HEAD')} dirty={bool(dirty)} {dirty.splitlines() if dirty else ''}")
    log(f"cpu affinity {affinity}; python {sys.version.split()[0]}; budget {budget}s")

    plan = [
        ("FR-E1-SPLIT", "uniform-real", lambda: ex.run_e1(primary, cfg, "uniform", False)),
        ("FR-E2-NEWOBS", "uniform-real", lambda: ex.run_e2(primary, cfg, "uniform", False)),
        ("FR-E3-READOUT", "uniform-real", lambda: ex.run_e3(primary, cfg, "uniform", False)),
        ("FR-E4-NUMERICS", "scalar", lambda: ex.run_e4(cfg)),
        ("FR-E1-SPLIT", "jittered-real", lambda: ex.run_e1(secondary, cfg, "jittered", False)),
        ("FR-E1-SPLIT", "uniform-complex", lambda: ex.run_e1(secondary, cfg, "uniform", True)),
        ("FR-E2-NEWOBS", "jittered-real", lambda: ex.run_e2(secondary, cfg, "jittered", False)),
        ("FR-E2-NEWOBS", "uniform-complex", lambda: ex.run_e2(secondary, cfg, "uniform", True)),
        ("FR-E3-READOUT", "jittered-real", lambda: ex.run_e3(secondary, cfg, "jittered", False)),
    ]
    results = {}
    status = []
    for exp_id, tag, fn in plan:
        elapsed = time.time() - t_start
        key = f"{exp_id}__{tag}"
        if elapsed > budget:
            log(f"SKIP {key}: budget exhausted ({elapsed:.1f}s > {budget}s) -> NOT_RUN(TIMEOUT)")
            status.append({"experiment": exp_id, "config": tag, "status": "NOT_RUN(TIMEOUT)"})
            continue
        t0 = time.time()
        try:
            recs = fn()
        except Exception as e:  # record failures verbatim, never as PASS
            log(f"FAIL {key}: {type(e).__name__}: {e}")
            status.append({"experiment": exp_id, "config": tag, "status": f"FAIL({type(e).__name__})"})
            continue
        dt = time.time() - t0
        results.setdefault(exp_id, []).extend(recs)
        out = os.path.join(RAW, f"{key}.jsonl")
        with open(out, "w") as f:
            for r in recs:
                f.write(json.dumps(r) + "\n")
        over = (time.time() - t_start) > budget
        st = "COMPLETED_OVER_BUDGET" if over else "COMPLETED"
        status.append({"experiment": exp_id, "config": tag, "status": st, "seconds": round(dt, 3),
                       "n_records": len(recs), "file": os.path.relpath(out, ROOT)})
        log(f"{st} {key}: {len(recs)} records in {dt:.2f}s")

    hyp = evaluate(results.get("FR-E1-SPLIT"), results.get("FR-E2-NEWOBS"), results.get("FR-E3-READOUT"))
    hyp_path = os.path.join(RAW, "hypotheses.json")
    with open(hyp_path, "w") as f:
        json.dump(hyp, f, indent=2)
    for h in hyp["hypotheses"]:
        log(f"{h['id']}: {h['status']}  {json.dumps(h.get('evidence', h.get('reason')))}")
    log(f"S2: {hyp['stop_condition_S2']}")
    wall = time.time() - t_start
    log(f"total wall clock {wall:.2f}s (budget {budget}s)")
    logf.close()

    files = sorted(os.path.join(RAW, x) for x in os.listdir(RAW))
    manifest = {
        "run_id": cfg["run_id"],
        "started_utc": time.strftime("%Y-%m-%dT%H:%M:%SZ", time.gmtime(t_start)),
        "wall_clock_seconds": round(wall, 3),
        "budget_seconds": budget,
        "within_budget": wall <= budget,
        "git_head": git("rev-parse", "HEAD"),
        "git_dirty_at_start": bool(dirty),
        "git_status_at_start": dirty.splitlines(),
        "config": {"path": "configs/first_run.json", "sha256": sha256(CFG_PATH)},
        "code_sha256": {os.path.relpath(p, ROOT): sha256(p) for p in sorted(
            [os.path.join(ROOT, "src", "scssm", x) for x in os.listdir(os.path.join(ROOT, "src", "scssm")) if x.endswith(".py")]
            + [os.path.abspath(__file__)])},
        "environment": {
            "python": sys.version, "implementation": platform.python_implementation(),
            "platform": platform.platform(), "cpu_affinity": affinity,
            "thread_env": {v: os.environ.get(v) for v in ("OMP_NUM_THREADS", "OPENBLAS_NUM_THREADS", "MKL_NUM_THREADS")},
            "third_party_packages": "none (stdlib only: math, cmath, random, decimal, struct, json)",
            "gpu": "none used",
        },
        "data": "synthetic only (random sine-sum input paths generated from integer seeds); no external data, models or downloads",
        "seeds": {"primary": primary, "secondary": secondary},
        "experiments": status,
        "hypotheses_file": os.path.relpath(hyp_path, ROOT),
        "stop_condition_S2": hyp["stop_condition_S2"],
        "outputs_sha256": {os.path.relpath(p, ROOT): sha256(p) for p in files},
    }
    with open(os.path.join(ROOT, "run_manifest.json"), "w") as f:
        json.dump(manifest, f, indent=2)


if __name__ == "__main__":
    main()
