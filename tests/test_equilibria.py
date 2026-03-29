from __future__ import annotations

import unittest

import numpy as np

from code.equilibrium_analysis import jacobian, positive_equilibrium, scalar_balance
from code.ivfs_config import FITTED_BETA0, FITTED_GAMMA0


class EquilibriumTests(unittest.TestCase):
    def test_positive_equilibrium_satisfies_scalar_balance(self) -> None:
        state = positive_equilibrium(0.9, FITTED_BETA0, FITTED_GAMMA0)
        self.assertIsNotNone(state)
        assert state is not None

        self.assertTrue(np.all(state >= 0.0))
        self.assertAlmostEqual(scalar_balance(float(state[4]), 0.9, FITTED_BETA0, FITTED_GAMMA0), 0.0, places=8)

    def test_jacobian_has_expected_shape(self) -> None:
        state = positive_equilibrium(0.9, FITTED_BETA0, FITTED_GAMMA0)
        assert state is not None
        jac = jacobian(state, 0.9, FITTED_BETA0, FITTED_GAMMA0)
        self.assertEqual(jac.shape, (6, 6))
        self.assertTrue(np.all(np.isfinite(jac)))


if __name__ == '__main__':
    unittest.main()