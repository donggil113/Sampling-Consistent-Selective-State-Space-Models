"""Unit tests for the toy model.  Numerical checks here are NOT proofs."""

import cmath
import math
import random
import unittest

from tests import _path  # noqa: F401
from scssm.grids import base_grid, base_times, refine_grid, sample_path, split_grid
from scssm.metrics import at_index, readouts, rel_l2
from scssm.model import VARIANTS, ToyParams, cexpm1, phi1, phi1m1, run_scan, sample_params, softplus, step_coeffs
from scssm.reference import ct_held


def scalar(lam, w=0.0, beta=0.0, b=1.0, c=1.0):
    return ToyParams((complex(lam, 0.0),), w, beta, (complex(b, 0.0),), (0j,), (complex(c, 0.0),), (0j,))


class TestPrimitives(unittest.TestCase):
    def test_softplus_positive_and_stable(self):
        for x in (-50.0, -5.0, 0.0, 5.0, 50.0):
            self.assertGreater(softplus(x), 0.0)
        self.assertAlmostEqual(softplus(0.0), math.log(2.0), places=15)

    def test_cexpm1_matches_exp_minus_one(self):
        for z in (0.3 + 0.7j, -2.0 + 3.0j, 0.01 - 0.02j):
            ref = cmath.exp(z) - 1.0
            self.assertLess(abs(cexpm1(z) - ref), 1e-14)
        # small-argument accuracy against the series z + z^2/2
        z = 1e-10 + 2e-10j
        self.assertLess(abs(cexpm1(z) - (z + z * z / 2)) / abs(z), 1e-15)

    def test_phi_series_continuity(self):
        for z in (0.0999 + 0j, 0.1001 + 0j, -0.0999 + 0.01j, -0.1001 + 0.01j):
            direct = (cmath.exp(z) - 1.0) / z
            self.assertLess(abs(phi1(z) - direct), 1e-14)
            self.assertLess(abs(phi1m1(z) - (direct - 1.0)), 1e-14)
        self.assertEqual(phi1(0j), 1.0)


class TestSteps(unittest.TestCase):
    def test_zoh_step_closed_form(self):
        p = scalar(-1.3, w=0.7, beta=0.1, b=0.9)
        u, dt = 0.4, 0.37
        a, bb, delta = step_coeffs(p, u, dt, "zoh_dt", 1.0)
        g = softplus(0.7 * u + 0.1)
        self.assertAlmostEqual(delta, dt * g, places=15)
        self.assertAlmostEqual(a[0].real, math.exp(-1.3 * dt * g), places=15)
        self.assertAlmostEqual(bb[0].real, (math.exp(-1.3 * dt * g) - 1.0) / -1.3 * 0.9, places=14)

    def test_stability_all_variants(self):
        rng = random.Random(3)
        for _ in range(20):
            p = sample_params(rng, 4, complex_modes=True)
            for v in VARIANTS:
                for dt in (1e-3, 0.5, 20.0):
                    a, _, _ = step_coeffs(p, rng.uniform(-3, 3), dt, v, 0.5)
                    for x in a:
                        self.assertLess(abs(x), 1.0)

    def test_nodt_equals_dt_on_uniform_base(self):
        rng = random.Random(0)
        p = sample_params(rng)
        path = sample_path(rng)
        G = base_grid(base_times(8.0, 16), path)
        tau = G.T / G.n_steps
        for s in ("zoh", "eulerB", "bilinear"):
            y_dt = run_scan(p, G.times, G.values, f"{s}_dt", tau)["y"]
            y_nd = run_scan(p, G.times, G.values, f"{s}_nodt", tau)["y"]
            self.assertLess(rel_l2(y_nd, y_dt), 1e-13)


class TestSplitAndRefine(unittest.TestCase):
    def setUp(self):
        rng = random.Random(1)
        self.p = sample_params(rng, 4, complex_modes=True)
        self.path = sample_path(rng)
        self.base = base_grid(base_times(8.0, 16, "jittered", rng=random.Random(9)), self.path)

    def test_split_preserves_held_path_and_base_times(self):
        for m in (1, 2, 4, 8):
            G = split_grid(self.base, m)
            self.assertEqual(at_index(G.times, G.base_index), list(self.base.times))
            for k in range(1, len(G.times)):
                # the held value on every sub-interval equals the base held value
                t_mid = 0.5 * (G.times[k - 1] + G.times[k])
                j = next(i for i in range(1, len(self.base.times)) if t_mid <= self.base.times[i])
                self.assertEqual(G.values[k - 1], self.base.values[j - 1])

    def test_refine_is_nested_with_new_observations(self):
        G = refine_grid(self.base, 4, self.path)
        self.assertEqual(at_index(G.times, G.base_index), list(self.base.times))
        for t, u in zip(G.times[1:], G.values):
            self.assertAlmostEqual(u, self.path(t), places=14)

    def test_exact_zoh_is_split_invariant(self):
        tau = self.base.T / self.base.n_steps
        y1 = run_scan(self.p, self.base.times, self.base.values, "zoh_dt", tau)["y"]
        for m in (2, 4, 8):
            G = split_grid(self.base, m)
            ym = at_index(run_scan(self.p, G.times, G.values, "zoh_dt", tau)["y"], G.base_index)
            self.assertLess(rel_l2(ym, y1), 1e-13)

    def test_time_exact_readout_is_split_invariant_nonuniform(self):
        tau = self.base.T / self.base.n_steps
        mask = [i % 3 == 0 for i in range(self.base.n_steps)]
        o1 = run_scan(self.p, self.base.times, self.base.values, "zoh_dt", tau)
        r1 = readouts(self.base.times, o1["y"], o1["integral"])
        G = split_grid(self.base, 8, mask)
        o8 = run_scan(self.p, G.times, G.values, "zoh_dt", tau)
        r8 = readouts(G.times, o8["y"], o8["integral"])
        self.assertLess(abs(r8["time_exact"] - r1["time_exact"]), 1e-13 * max(1.0, abs(r1["time_exact"])))
        self.assertGreater(abs(r8["sample_sum"] - r1["sample_sum"]), 1e-6)

    def test_exact_zoh_matches_independent_rk4(self):
        tau = self.base.T / self.base.n_steps
        o = run_scan(self.p, self.base.times, self.base.values, "zoh_dt", tau)
        ref = ct_held(self.p, self.base)
        self.assertLess(rel_l2(o["y"], ref["y"]), 1e-9)
        self.assertLess(abs(o["integral"] - ref["integral"]) / max(abs(ref["integral"]), 1e-12), 1e-9)

    def test_eulerB_first_order_numerically(self):
        # numerical rate check on one unit; not a proof of the order
        tau = self.base.T / self.base.n_steps
        exact = run_scan(self.p, self.base.times, self.base.values, "zoh_dt", tau)["y"]
        errs = []
        for m in (8, 16, 32):
            G = split_grid(self.base, m)
            errs.append(rel_l2(at_index(run_scan(self.p, G.times, G.values, "eulerB_dt", tau)["y"], G.base_index), exact))
        self.assertAlmostEqual(math.log2(errs[1] / errs[2]), 1.0, delta=0.15)


class TestReadouts(unittest.TestCase):
    def test_readouts_on_constant_sequence(self):
        times = [0.0, 1.0, 3.0, 4.0]
        ys = [0.0, 2.0, 2.0, 2.0]
        r = readouts(times, ys)
        self.assertAlmostEqual(r["sample_mean"], 2.0)
        self.assertAlmostEqual(r["sample_sum"], 6.0)
        self.assertAlmostEqual(r["time_riemann"], 2.0)
        self.assertAlmostEqual(r["time_trapz"], (1.0 + 4.0 + 2.0) / 4.0)
        self.assertIsNone(r["time_exact"])


if __name__ == "__main__":
    unittest.main()
