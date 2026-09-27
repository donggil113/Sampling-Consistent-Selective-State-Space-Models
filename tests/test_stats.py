"""Tests for the P1-REAL-02 bootstrap helpers (numerical checks, not proofs)."""

import unittest

from tests import _path  # noqa: F401
from scssm.real import stats as S


class TestStats(unittest.TestCase):
    def test_draws_are_fixed(self):
        self.assertEqual(S.draws(10, 5, 0), S.draws(10, 5, 0))

    def test_ratio_recomputes_both_parts(self):
        num = [1.1, 0.9, 1.2, 1.0]
        den = [1.0, 1.0, 1.0, 1.0]
        r, ci, d, dci = S.ratio_ci(num, den, S.draws(4, 200, 1))
        self.assertAlmostEqual(r, 0.05)
        self.assertAlmostEqual(d, 0.05)
        self.assertLessEqual(ci[0], r)
        self.assertGreaterEqual(ci[1], r)

    def test_small_denominator_is_na(self):
        r, ci, d, _ = S.ratio_ci([1.0, 2.0], [0.0, 0.0], S.draws(2, 10, 0))
        self.assertIsNone(r)
        self.assertIsNone(ci)
        self.assertEqual(d, 1.5)

    def test_margin_verdicts(self):
        self.assertTrue(S.margin_verdict([-0.01, 0.02]).startswith("no practical"))
        self.assertEqual(S.margin_verdict([-0.08, -0.06]), "native lower, beyond margin")
        self.assertEqual(S.margin_verdict([-0.07, -0.01]), "native lower; CI not beyond margin")
        self.assertEqual(S.margin_verdict([-0.07, 0.01]), "inconclusive")
        self.assertTrue(S.margin_verdict(None).startswith("NA"))


if __name__ == "__main__":
    unittest.main()
