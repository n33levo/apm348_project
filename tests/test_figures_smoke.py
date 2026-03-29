from __future__ import annotations

import unittest

import numpy as np

from code.ivfs_config import PHI_SENS_FIGURE_PATH
from code.ivfs_figures import make_phi_sensitivity_figure
from code.model_structure_figure import FIGURE_PATH as STRUCTURE_FIGURE_PATH, main as make_structure_figure


class FigureSmokeTests(unittest.TestCase):
    def test_phi_sensitivity_figure_renders(self) -> None:
        phi_grid = np.linspace(0.03, 0.08, 5)
        phi_results = {
            'Engagement-First (alpha=0.9)': {'tau_star': np.linspace(0.2, 0.4, 5), 'U_star': np.linspace(10.0, 8.0, 5)},
            'Moderate (alpha=0.5)': {'tau_star': np.linspace(0.1, 0.25, 5), 'U_star': np.linspace(12.0, 10.0, 5)},
            'Health-First (alpha=0.2)': {'tau_star': np.linspace(0.0, 0.05, 5), 'U_star': np.linspace(14.0, 13.0, 5)},
        }
        make_phi_sensitivity_figure(phi_grid, phi_results, current_phi=0.056)
        self.assertTrue(PHI_SENS_FIGURE_PATH.exists())
        self.assertGreater(PHI_SENS_FIGURE_PATH.stat().st_size, 0)

    def test_structure_figure_renders(self) -> None:
        make_structure_figure()
        self.assertTrue(STRUCTURE_FIGURE_PATH.exists())
        self.assertGreater(STRUCTURE_FIGURE_PATH.stat().st_size, 0)


if __name__ == '__main__':
    unittest.main()