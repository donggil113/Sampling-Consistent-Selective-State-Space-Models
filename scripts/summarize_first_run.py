"""Summarize results/raw/*.jsonl into results/first_run_summary.{md,json}.

Pure aggregation of existing raw records; runs no experiment.
"""

import json
import os
import sys

ROOT = os.path.dirname(os.path.dirname(os.path.abspath(__file__)))
sys.path.insert(0, os.path.join(ROOT, "src"))

from scssm.metrics import loglog_slope, median  # noqa: E402
from scssm.model import VARIANTS  # noqa: E402

RAW = os.path.join(ROOT, "results", "raw")
MS = [1, 2, 4, 8]


def load(name):
    p = os.path.join(RAW, name + ".jsonl")
    if not os.path.exists(p):
        return None
    with open(p) as f:
        return [json.loads(l) for l in f]


def fmt(x):
    if x is None:
        return "n/a"
    if x == 0:
        return "0"
    return f"{x:.2e}"


def med(recs, key):
    return median([r[key] for r in recs if r.get(key) is not None])


def mx(recs, key):
    v = [r[key] for r in recs if r.get(key) is not None]
    return max(v) if v else None


def per_seed_slopes(recs, key, ms):
    by = {}
    for r in recs:
        by.setdefault(r["seed"], {})[r["m"]] = r[key]
    return [loglog_slope(ms, [row[m] for m in ms]) for row in by.values() if all(m in row for m in ms)]


def e1_table(recs, tag):
    rows = [f"### FR-E1-SPLIT [{tag}] (n_units={len({r['seed'] for r in recs})})", "",
            "Output change at common base times vs m=1 (change) and vs the exact held-path solution (artifact); median [max] over units.", "",
            "| variant | m=2 change | m=4 change | m=8 change | m=1 artifact | m=8 artifact | artifact slope (median) |",
            "|---|---|---|---|---|---|---|"]
    out = {}
    for v in VARIANTS:
        rv = [r for r in recs if r.get("variant") == v]
        cells = []
        for m in (2, 4, 8):
            rm = [r for r in rv if r["m"] == m]
            cells.append(f"{fmt(med(rm, 'change_y'))} [{fmt(mx(rm, 'change_y'))}]")
        a1 = [r for r in rv if r["m"] == 1]
        a8 = [r for r in rv if r["m"] == 8]
        sl = median(per_seed_slopes(rv, "artifact_y", MS))
        rows.append(f"| {v} | " + " | ".join(cells) +
                    f" | {fmt(med(a1, 'artifact_y'))} | {fmt(med(a8, 'artifact_y'))} [{fmt(mx(a8, 'artifact_y'))}] | "
                    f"{'n/a' if sl is None else f'{sl:.2f}'} |")
        out[v] = {"median_change_y": {m: med([r for r in rv if r["m"] == m], "change_y") for m in MS},
                  "median_artifact_y": {m: med([r for r in rv if r["m"] == m], "artifact_y") for m in MS},
                  "median_artifact_slope": sl}
    return rows, out


def e2_table(recs, refs, tag):
    rows = [f"### FR-E2-NEWOBS [{tag}] (n_units={len({r['seed'] for r in recs})})", "",
            "Nested refinement with new observations. artifact = rel_l2(model(G_m), CT model on held path of G_m); "
            "path = change of the CT held-path solution itself (genuine new information); err = rel_l2 to CT truth on the continuous path. Medians over units.", "",
            "| variant | m | total change | path change | artifact | nd artifact delta / (nd path + nd artifact delta) | err to truth |",
            "|---|---|---|---|---|---|---|"]
    out = {}
    for v in VARIANTS:
        rv = [r for r in recs if r.get("variant") == v]
        out[v] = {}
        for m in MS:
            rm = [r for r in rv if r["m"] == m]
            shares = [r["nd_artifact_delta"] / (r["nd_path"] + r["nd_artifact_delta"])
                      for r in rm if (r["nd_path"] + r["nd_artifact_delta"]) > 0]
            share = median(shares)
            rows.append(f"| {v} | {m} | {fmt(med(rm, 'total_change'))} | {fmt(med(rm, 'path_change'))} | "
                        f"{fmt(med(rm, 'artifact'))} [{fmt(mx(rm, 'artifact'))}] | {'n/a' if share is None else f'{share:.3f}'} | "
                        f"{fmt(med(rm, 'err_to_truth'))} |")
            out[v][m] = {"total": med(rm, "total_change"), "path": med(rm, "path_change"), "artifact": med(rm, "artifact"),
                         "artifact_share": share, "err_to_truth": med(rm, "err_to_truth")}
        out[v]["err_to_truth_slope"] = median(per_seed_slopes(rv, "err_to_truth", MS))
    rows.append("")
    rows.append("CT held-path reference error to truth (input-reconstruction error only), median: " +
                ", ".join(f"m={m}: {fmt(med([r for r in refs if r['m'] == m], 'ct_held_err_to_truth'))}" for m in MS))
    return rows, out


def e3_tables(recs, tag):
    rows, out = [], {}
    ra = [r for r in recs if r["scenario"] == "a_split_nonuniform"]
    rb = [r for r in recs if r["scenario"] == "b_newobs_nonuniform"]
    names = ["last", "sample_sum", "sample_mean", "time_riemann", "time_trapz", "time_exact"]
    rows += [f"### FR-E3-READOUT (a) split first half only, held path unchanged [{tag}]", "",
             "Scale-normalized readout change |r_m - r_1| / scale (sample_sum scale x K_1), median over units at m=8 [max].", "",
             "| variant | " + " | ".join(names) + " |", "|---|" + "---|" * len(names)]
    out["a"] = {}
    for v in VARIANTS:
        r8 = [r for r in ra if r["variant"] == v and r["m"] == 8]
        cells = [f"{fmt(med(r8, 'change_' + n))} [{fmt(mx(r8, 'change_' + n))}]" if r8 and r8[0].get("change_" + n) is not None
                 else "NOT_APPLICABLE" for n in names]
        rows.append(f"| {v} | " + " | ".join(cells) + " |")
        out["a"][v] = {n: med(r8, "change_" + n) for n in names}
    rows += ["", "Error to the hold-exact time average of the unchanged held path (scale-normalized), median at m=1 -> m=8:", "",
             "| variant | sample_mean | time_riemann | time_trapz | time_exact |", "|---|---|---|---|---|"]
    for v in VARIANTS:
        cells = []
        for n in ("sample_mean", "time_riemann", "time_trapz", "time_exact"):
            k = "err_heldavg_" + n
            r1 = [r for r in ra if r["variant"] == v and r["m"] == 1 and r.get(k) is not None]
            r8 = [r for r in ra if r["variant"] == v and r["m"] == 8 and r.get(k) is not None]
            cells.append(f"{fmt(med(r1, k))} -> {fmt(med(r8, k))}" if r1 else "NOT_APPLICABLE")
        rows.append(f"| {v} | " + " | ".join(cells) + " |")
    rows += ["", f"### FR-E3-READOUT (b) new observations in first half only [{tag}]", "",
             "Scale-normalized error to the continuous-time truth (time average for sample_mean/time_*; y(T) for last), median at m=1 -> m=8.", "",
             "| variant | last | sample_mean | time_riemann | time_trapz | time_exact | sample_sum change m=8 |",
             "|---|---|---|---|---|---|---|"]
    out["b"] = {}
    for v in VARIANTS:
        cells = []
        out["b"][v] = {}
        for n in ("last", "sample_mean", "time_riemann", "time_trapz", "time_exact"):
            k = "err_truth_" + n
            r1 = [r for r in rb if r["variant"] == v and r["m"] == 1 and r.get(k) is not None]
            r8 = [r for r in rb if r["variant"] == v and r["m"] == 8 and r.get(k) is not None]
            cells.append(f"{fmt(med(r1, k))} -> {fmt(med(r8, k))}" if r1 else "NOT_APPLICABLE")
            out["b"][v][n] = {"m1": med(r1, k) if r1 else None, "m8": med(r8, k) if r8 else None}
        r8 = [r for r in rb if r["variant"] == v and r["m"] == 8]
        cells.append(fmt(med(r8, "change_sample_sum")))
        rows.append(f"| {v} | " + " | ".join(cells) + " |")
    return rows, out


def e4_tables(recs):
    rows = ["### FR-E4-NUMERICS (failure cases and numerical error logs)", ""]
    n1 = [r for r in recs if r["case"] == "N1_cancellation"]
    rows += ["N1: relative error of the ZOH input coefficient (e^z - 1)/lambda vs 50-digit Decimal reference.", "",
             "| z | naive exp(z)-1 | expm1 |", "|---|---|---|"]
    rows += [f"| {r['z']:.0e} | {fmt(r['rel_err_naive'])} | {fmt(r['rel_err_expm1'])} |" for r in n1]
    for case, title in (("N1b_split_drift_fp64", "N1b: float64, split [0,1] into m held sub-steps; rel. error of h(1) (exact 0.5)"),
                        ("N4_split_drift_fp32_emulated", "N4: EMULATED float32 (every primitive rounded), same test")):
        rr = [r for r in recs if r["case"] == case]
        ms = sorted({r["m"] for r in rr})
        rows += ["", title + ".", "", "| m | expm1 | naive |", "|---|---|---|"]
        for m in ms:
            e = {r["formula"]: r["rel_err_h1"] for r in rr if r["m"] == m}
            rows.append(f"| {m} | {fmt(e.get('expm1'))} | {fmt(e.get('naive'))} |")
    rr = [r for r in recs if r["case"] == "N2_stiff"]
    rows += ["", "N2: stiff scalar mode (g=ln2, dt=0.5 base), output artifact vs exact ZOH; Abar for the bilinear step.", "",
             "| lambda | m | bilinear Abar | bilinear artifact | eulerB artifact |", "|---|---|---|---|---|"]
    for lam in sorted({r["lambda"] for r in rr}, reverse=True):
        for m in sorted({r["m"] for r in rr}):
            b = [r for r in rr if r["lambda"] == lam and r["m"] == m and r["variant"] == "bilinear_dt"][0]
            e = [r for r in rr if r["lambda"] == lam and r["m"] == m and r["variant"] == "eulerB_dt"][0]
            rows.append(f"| {lam:g} | {m} | {b['abar']:+.3f} | {fmt(b['artifact_y'])} | {fmt(e['artifact_y'])} |")
    rr = [r for r in recs if r["case"] == "N3_coarse_steady_state"]
    rows += ["", "N3: steady state under held u=1 (exact h*=1), relative error.", "",
             "| dt | zoh_dt | eulerB_dt | bilinear_dt |", "|---|---|---|---|"]
    for dt in sorted({r["dt"] for r in rr}):
        e = {r["variant"]: r["rel_err"] for r in rr if r["dt"] == dt}
        rows.append(f"| {dt:g} | {fmt(e['zoh_dt'])} | {fmt(e['eulerB_dt'])} | {fmt(e['bilinear_dt'])} |")
    return rows


def factorial(e3):
    """Median scale-normalized change at m=8 under (a) nonuniform split, one fix at a time."""
    ra = [r for r in e3 if r["config"] == "uniform-real" and r["scenario"] == "a_split_nonuniform" and r["m"] == 8]

    def cell(v, n):
        return med([r for r in ra if r["variant"] == v], "change_" + n)

    combos = [
        ("no fix (Mamba-style toy recipe: Delta=tau*g, Euler B, sample mean)", "eulerB_nodt", "sample_mean"),
        ("+ time-weighted readout only (trapz)", "eulerB_nodt", "time_trapz"),
        ("+ exact ZOH B only", "zoh_nodt", "sample_mean"),
        ("+ Delta=dt*g only", "eulerB_dt", "sample_mean"),
        ("+ Delta=dt*g + time trapz", "eulerB_dt", "time_trapz"),
        ("+ Delta=dt*g + exact ZOH (sample mean)", "zoh_dt", "sample_mean"),
        ("+ Delta=dt*g + exact ZOH + time trapz", "zoh_dt", "time_trapz"),
        ("all fixes: Delta=dt*g + exact ZOH + time-exact readout", "zoh_dt", "time_exact"),
    ]
    rows = ["### Decomposition by one-fix-at-a-time (FR-E3(a) split first half, m=8, uniform-real, median over 32 units)", "",
            "| configuration | variant | readout | median scale-normalized change |", "|---|---|---|---|"]
    out = []
    for name, v, n in combos:
        c = cell(v, n)
        rows.append(f"| {name} | {v} | {n} | {fmt(c)} |")
        out.append({"name": name, "variant": v, "readout": n, "median_change_m8": c})
    return rows, out


def main():
    md = ["# FIRST_RUN summary (generated by scripts/summarize_first_run.py from results/raw)", "",
          "All numbers are from the toy described in configs/first_run.json. They are numerical checks, not proofs, "
          "and say nothing directly about trained Mamba/S4/S5 models.", ""]
    js = {}
    hyp = json.load(open(os.path.join(RAW, "hypotheses.json")))
    md += ["## Pre-registered hypotheses (primary config: uniform-real, 32 units)", "",
           "| id | status | kind |", "|---|---|---|"]
    md += [f"| {h['id']} | {h['status']} | {h.get('kind', h.get('reason'))} |" for h in hyp["hypotheses"]]
    md += ["", f"Stop condition S2: {hyp['stop_condition_S2']}", ""]
    js["hypotheses"] = hyp

    for tag in ("uniform-real", "jittered-real", "uniform-complex"):
        e1 = load(f"FR-E1-SPLIT__{tag}")
        if e1 is None:
            md += [f"### FR-E1-SPLIT [{tag}]: NOT_RUN", ""]
            continue
        refs = [r for r in e1 if r["kind"] == "reference_check"]
        rows, out = e1_table([r for r in e1 if r["kind"] == "variant"], tag)
        md += rows + ["", f"exact ZOH vs independent RK4 on the base held path: max rel_l2(y) = "
                      f"{fmt(max(r['exact_zoh_vs_rk4_rel_l2_y'] for r in refs))}", ""]
        js[f"E1_{tag}"] = out
    for tag in ("uniform-real", "jittered-real", "uniform-complex"):
        e2 = load(f"FR-E2-NEWOBS__{tag}")
        if e2 is None:
            md += [f"### FR-E2-NEWOBS [{tag}]: NOT_RUN", ""]
            continue
        rows, out = e2_table([r for r in e2 if r["kind"] == "variant"], [r for r in e2 if r["kind"] == "reference"], tag)
        md += rows + [""]
        js[f"E2_{tag}"] = out
    for tag in ("uniform-real", "jittered-real"):
        e3 = load(f"FR-E3-READOUT__{tag}")
        if e3 is None:
            md += [f"### FR-E3-READOUT [{tag}]: NOT_RUN", ""]
            continue
        rows, out = e3_tables(e3, tag)
        md += rows + [""]
        js[f"E3_{tag}"] = out
    e3 = load("FR-E3-READOUT__uniform-real")
    if e3:
        rows, out = factorial(e3)
        md += rows + [""]
        js["factorial"] = out
    e4 = load("FR-E4-NUMERICS__scalar")
    md += e4_tables(e4) if e4 else ["### FR-E4-NUMERICS: NOT_RUN"]
    with open(os.path.join(ROOT, "results", "first_run_summary.md"), "w") as f:
        f.write("\n".join(md) + "\n")
    with open(os.path.join(ROOT, "results", "first_run_summary.json"), "w") as f:
        json.dump(js, f, indent=1, default=str)


if __name__ == "__main__":
    main()
