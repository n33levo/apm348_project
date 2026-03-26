from __future__ import annotations

from common import ASSETS_DIR, HIGGS_GZ, HIGGS_TXT, HIGGS_URL, ensure_layout


FIGURE_PATH = ASSETS_DIR / 'apm348_results.png'
DIAG_FIGURE_PATH = ASSETS_DIR / 'apm348_calibration_diagnostics.png'
PHI_SENS_FIGURE_PATH = ASSETS_DIR / 'phi_sensitivity.png'
TAU_COMPARE_FIGURE_PATH = ASSETS_DIR / 'tau_proxy_comparison.png'
PROFILE_FIGURE_PATH = ASSETS_DIR / 'profile_likelihood.png'

# --- model params (shared across files) ---
KAPPA = 0.8
ETA = 0.3
PHI = 0.056
PSI = 0.1
RHO = 0.06
LAMBDA_U = 0.02
NU = 1.0
MU_C = 0.01
DELTA = 0.05
W = 10.0
SCENARIO_ALPHAS = {
    'Engagement-First (alpha=0.9)': 0.9,
    'Moderate (alpha=0.5)': 0.5,
    'Health-First (alpha=0.2)': 0.2,
}

# --- calibration / plotting constants ---
FIT_WINDOW_HOURS = 100
DISPLAY_WINDOW_HOURS = 50
SPIKE_WINDOW_HOURS = 35
TAIL_START_HOURS = 40
SCENARIO_DISPLAY_HOURS = 200
SMOOTH_WINDOW = 3

IVF_PARAM_BOUNDS = (
    (0.2, 4.0),
    (0.05, 1.0),
    (0.0, 0.2),
    (0.001, 0.2),
    (1e-5, 0.1),
)

TAU_PARAM_BOUNDS = (
    (0.005, 1.0),
    (0.01, 1.0),
)

TAU_PROXY_WEIGHTS = {
    're': 0.55,
    'ratio': 0.30,
    'mt': 0.15,
}

TAU_DECAY_MIN_POINTS = 6
