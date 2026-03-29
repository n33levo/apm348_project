from __future__ import annotations

import tempfile
import unittest
from pathlib import Path
from unittest.mock import patch

import numpy as np

import code.ivfs_figures as ivfs_figures
import code.model_structure_figure as model_structure_figure


class FigureSmokeTests(unittest.TestCase):
    def test_phi_sensitivity_figure_renders(self) -> None:
        phi_grid = np.linspace(0.03, 0.08, 5)
        phi_results = {
            'Engagement-First (alpha=0.9)': {'tau_star': np.linspace(0.2, 0.4, 5), 'U_star': np.linspace(10.0, 8.0, 5)},
            'Moderate (alpha=0.5)': {'tau_star': np.linspace(0.1, 0.25, 5), 'U_star': np.linspace(12.0, 10.0, 5)},
            'Health-First (alpha=0.2)': {'tau_star': np.linspace(0.0, 0.05, 5), 'U_star': np.linspace(14.0, 13.0, 5)},
        }
        with tempfile.TemporaryDirectory() as tmpdir:
            out_path = Path(tmpdir) / 'phi_sensitivity.png'
            with patch.object(ivfs_figures, 'PHI_SENS_FIGURE_PATH', out_path):
                ivfs_figures.make_phi_sensitivity_figure(phi_grid, phi_results, current_phi=0.056)
            self.assertTrue(out_path.exists())
            self.assertGreater(out_path.stat().st_size, 0)

    def test_structure_figure_renders(self) -> None:
        with tempfile.TemporaryDirectory() as tmpdir:
            out_path = Path(tmpdir) / 'ivfs_structure_diagram.png'
            with patch.object(model_structure_figure, 'FIGURE_PATH', out_path):
                model_structure_figure.main()
            self.assertTrue(out_path.exists())
            self.assertGreater(out_path.stat().st_size, 0)


if __name__ == '__main__':
    unittest.main()