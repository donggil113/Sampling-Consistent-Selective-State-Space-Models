"""Unit tests for the P1-COMP-01 cascade (numerical checks, not proofs)."""

import math
import unittest

from tests import _path  # noqa: F401
from scssm.cascade import coupled_step, run_coupled_exact, run_coupled_rk4, run_layerwise


class TestCascade(unittest.TestCase):
    def test_equal_rates_is_continuous_limit(self):
        base = coupled_step(0.3, -0.2, 0.7, 0.5, 1.0, 1.0, 1.0, 1.0)
        near = coupled_step(0.3, -0.2, 0.7, 0.5, 1.0, 1.0, 1.0 + 1e-7, 1.0)
        self.assertLess(abs(base[1] - near[1]), 1e-7)
        # a = c closed form: psi = tau e^{-a tau}
        a, tau = 1.0, 0.5
        h1, h2 = coupled_step(1.0, 0.0, 0.0, tau, a, 1.0, a, 1.0)
        self.assertAlmostEqual(h2, tau * math.exp(-a * tau), places=15)

    def test_exact_matches_rk4_single_interval(self):
        for c in (0.5, 1.0, 3.0):
            ex = run_coupled_exact([0.0, 0.8], [1.3], 1.0, 0.9, c, 1.1)
            rk = run_coupled_rk4([0.0, 0.8], [1.3], 1.0, 0.9, c, 1.1, 2000)
            self.assertLess(abs(ex[1][-1] - rk[1][-1]), 1e-12)

    def test_steady_state_layerwise_is_exact(self):
        # constant input for a long time: h1 is constant, so holding it is exact
        times = [0.1 * k for k in range(801)]
        vals = [1.0] * 800
        ex = run_coupled_exact(times, vals, 1.0, 1.0, 0.5, 1.0)
        lw = run_layerwise(times, vals, 1.0, 1.0, 0.5, 1.0)
        self.assertAlmostEqual(ex[1][-1], 2.0, places=10)
        self.assertAlmostEqual(lw[1][-1], 2.0, places=10)

    def test_layer1_identical(self):
        times = [0.0, 0.3, 0.9, 1.0]
        vals = [0.5, -1.0, 2.0]
        ex = run_coupled_exact(times, vals, 1.0, 1.0, 0.5, 1.0)
        lw = run_layerwise(times, vals, 1.0, 1.0, 0.5, 1.0)
        for x, y in zip(ex[0], lw[0]):
            self.assertAlmostEqual(x, y, places=15)


if __name__ == "__main__":
    unittest.main()
