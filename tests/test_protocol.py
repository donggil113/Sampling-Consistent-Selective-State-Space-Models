"""Tests for the P1-REAL-01 evaluation protocol (stdlib parts only)."""

import random
import unittest

from tests import _path  # noqa: F401
from scssm.real import protocol as P


class TestProtocol(unittest.TestCase):
    def test_split_is_disjoint_and_complete(self):
        sp = P.split_ids(512, 128, 256)
        ids = set(sp.train) | set(sp.dev) | set(sp.test)
        self.assertEqual(len(ids), 896)
        self.assertFalse(set(sp.train) & set(sp.test))
        self.assertFalse(set(sp.dev) & set(sp.test))

    def test_trajectory_depends_only_on_its_id(self):
        self.assertEqual(P.trajectory_path(700), P.trajectory_path(700))
        self.assertNotEqual(P.trajectory_path(700), P.trajectory_path(701))

    def test_conditions_share_base_times_and_labels(self):
        path = P.trajectory_path(640)
        conds = P.test_conditions(path, 8.0, 32, [1, 2, 4, 8])
        base = conds["C0"].times
        for g in conds.values():
            self.assertEqual([g.times[i] for i in g.base_index], list(base))

    def test_locf_is_causal(self):
        path = P.trajectory_path(641)
        J = P.irregular_grid(path, 8.0, 16, random.Random(3))
        target = [0.25 * k for k in range(33)]
        vals, _ = P.resample_locf(J, target)
        for t, v in zip(target[1:], vals):
            # the chosen value must come from an observation at or before t (or the first one)
            obs = [(ti, vi) for ti, vi in zip(J.times[1:], J.values) if ti <= t + 1e-12]
            self.assertEqual(v, obs[-1][1] if obs else J.values[0])

    def test_locf_on_split_grid_recovers_training_grid(self):
        path = P.trajectory_path(642)
        conds = P.test_conditions(path, 8.0, 32, [1, 8])
        vals, early = P.resample_locf(conds["S8"], conds["C0"].times)
        self.assertEqual(list(vals), list(conds["C0"].values))
        self.assertEqual(early, 0)

    def test_step_scale_is_one_on_training_grid(self):
        g = P.training_grid(P.trajectory_path(643), 8.0, 32)
        for s in P.step_scales(g, 0.25):
            self.assertAlmostEqual(s, 1.0, places=12)


if __name__ == "__main__":
    unittest.main()
