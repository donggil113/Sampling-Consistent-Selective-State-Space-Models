"""Within-seed paired bootstrap helpers for P1-REAL-02 (stdlib only)."""

import random


def draws(n, n_boot=2000, seed=0):
    """Fixed trajectory index draws, reused for every estimand and seed."""
    rng = random.Random(seed)
    return [[rng.randrange(n) for _ in range(n)] for _ in range(n_boot)]


def _pct(sorted_vals):
    b = len(sorted_vals)
    return [sorted_vals[int(0.025 * b)], sorted_vals[int(0.975 * b) - 1]]


def mean_ci(values, idx_draws):
    n = len(values)
    est = sum(values) / n
    means = sorted(sum(values[j] for j in idx) / n for idx in idx_draws)
    return est, _pct(means)


def ratio_ci(num_vals, den_vals, idx_draws, eps=1e-12):
    """r = (mean(num) - mean(den)) / mean(den); numerator and denominator recomputed per draw.
    Returns (r, ci, abs_diff, abs_ci); r and ci are None (NA) if the denominator is below eps."""
    n = len(num_vals)
    mn, md = sum(num_vals) / n, sum(den_vals) / n
    diffs, rs = [], []
    for idx in idx_draws:
        a = sum(num_vals[j] for j in idx) / n
        b = sum(den_vals[j] for j in idx) / n
        diffs.append(a - b)
        rs.append(None if b < eps else (a - b) / b)
    abs_ci = _pct(sorted(diffs))
    if md < eps or any(r is None for r in rs):
        return None, None, mn - md, abs_ci
    return (mn - md) / md, _pct(sorted(rs)), mn - md, abs_ci


def margin_verdict(ci, margin=0.05):
    if ci is None:
        return "NA (denominator too small)"
    lo, hi = ci
    if -margin < lo and hi < margin:
        return "no practical difference (CI inside +-5%)"
    if hi < 0:
        return "native lower" + (", beyond margin" if hi < -margin else "; CI not beyond margin")
    if lo > 0:
        return "native higher" + (", beyond margin" if lo > margin else "; CI not beyond margin")
    return "inconclusive"
