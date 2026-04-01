from __future__ import annotations

import unittest

from code.equilibrium_analysis import dfe, positive_equilibrium
from code.ivfs_config import FITTED_BETA0, FITTED_GAMMA0


class ThresholdTests(unittest.TestCase):
    def test_positive_equilibrium_tracks_r0_threshold(self) -> None:
        low_alpha = 0.2
        high_alpha = 0.9

        _i_low, _u_low, r0_low = dfe(low_alpha, FITTED_BETA0, FITTED_GAMMA0)
        _i_high, _u_high, r0_high = dfe(high_alpha, FITTED_BETA0, FITTED_GAMMA0)

        self.assertLess(r0_low, 1.0)
        self.assertGreater(r0_high, 1.0)
        self.assertIsNone(positive_equilibrium(low_alpha, FITTED_BETA0, FITTED_GAMMA0))
        self.assertIsNotNone(positive_equilibrium(high_alpha, FITTED_BETA0, FITTED_GAMMA0))


if __name__ == '__main__':
    unittest.main()