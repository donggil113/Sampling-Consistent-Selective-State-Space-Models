"""Evaluation protocol for P1-REAL-01 (stdlib only; no model code).

Order of operations is fixed and tested:
  1. draw trajectories by ID (input path from its own seed);
  2. split trajectory IDs into train / dev / test;
  3. only then build grids for each split independently.
Labels are the continuous-time teacher output at physical query times; the
horizon is the physical interval [0, T]; normalization statistics come from the
training split on the training grid and are reused unchanged for every grid.
"""

import math
import random
from dataclasses import dataclass

from ..experiments import make_unit
from ..grids import Grid, base_grid, base_times, refine_grid, sample_path, split_grid  # noqa: F401

TRAJ_SEED_OFFSET = 1_000_000  # disjoint from FIRST_RUN seeds 0..31 and the smoke seed 999


@dataclass(frozen=True)
class Split:
    train: tuple
    dev: tuple
    test: tuple


def split_ids(n_train, n_dev, n_test):
    ids = list(range(n_train + n_dev + n_test))
    return Split(tuple(ids[:n_train]), tuple(ids[n_train:n_train + n_dev]), tuple(ids[n_train + n_dev:]))


def trajectory_path(traj_id):
    return sample_path(random.Random(TRAJ_SEED_OFFSET + traj_id))


def teacher(first_run_cfg):
    """Fixed teacher: the FIRST_RUN continuous-time toy with unit-0 parameters."""
    params, _, _ = make_unit(0, first_run_cfg, "uniform", False)
    return params


def training_grid(path, T, K):
    return base_grid(base_times(T, K), path)


def test_conditions(path, T, K, ms):
    """All test grids for one (already split) trajectory. Base times are common."""
    g0 = training_grid(path, T, K)
    conds = {"C0": g0}
    for m in ms:
        if m == 1:
            continue
        conds[f"S{m}"] = split_grid(g0, m)          # same held path, repeated values
        conds[f"N{m}"] = refine_grid(g0, m, path)   # new external observations
    return conds


def irregular_grid(path, T, n_obs, rng):
    """Random observation times (no training times guaranteed): for resampling tests."""
    ts = sorted(rng.uniform(0.0, T) for _ in range(n_obs - 1))
    times = tuple([0.0] + ts + [T])
    return base_grid(times, path)


def resample_locf(grid, target_times):
    """Causal resampling: last observation carried forward (no future leakage).

    Returns the value at each target time t_k (k >= 1) from the latest
    observation time <= t_k; before the first observation the first value is used
    and flagged.
    """
    out, early = [], 0
    j = 0
    obs_t = grid.times[1:]
    for t in target_times[1:]:
        while j + 1 < len(obs_t) and obs_t[j + 1] <= t + 1e-12:
            j += 1
        if obs_t[j] > t + 1e-12:
            early += 1
        out.append(grid.values[j])
    return out, early


def resample_linear(grid, target_times):
    """NON-causal linear interpolation (uses the next observation); secondary only."""
    ts, vs = grid.times[1:], grid.values
    out = []
    for t in target_times[1:]:
        if t <= ts[0]:
            out.append(vs[0])
            continue
        if t >= ts[-1]:
            out.append(vs[-1])
            continue
        j = max(i for i in range(len(ts)) if ts[i] <= t)
        w = (t - ts[j]) / (ts[j + 1] - ts[j])
        out.append((1 - w) * vs[j] + w * vs[j + 1])
    return out


def step_scales(grid, dt_train):
    """Per-step physical gaps in units of the training step (1.0 on the training grid)."""
    return [(grid.times[k] - grid.times[k - 1]) / dt_train for k in range(1, len(grid.times))]


def norm_stats(values):
    n = len(values)
    mu = sum(values) / n
    sd = math.sqrt(sum((v - mu) ** 2 for v in values) / n)
    return mu, (sd if sd > 0 else 1.0)
