from __future__ import annotations

import unittest

from code.equilibrium_analysis import A_LOSS, dfe, inflow
from code.ivfs_config import FITTED_BETA0, FITTED_GAMMA0


class DfeTests(unittest.TestCase):
    def test_dfe_matches_closed_form(self) -> None:
        alpha = 0.5
        i_star, u_star, r0 = dfe(alpha, FITTED_BETA0, FITTED_GAMMA0)

        self.assertGreater(u_star, 0.0)
        self.assertAlmostEqual(i_star, inflow(u_star) / A_LOSS, places=10)
        self.assertAlmostEqual(r0, alpha * FITTED_BETA0 * i_star / (FITTED_GAMMA0 + 0.01), places=10)


if __name__ == '__main__':
    unittest.main()