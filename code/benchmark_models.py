from __future__ import annotations

"""Quick benchmark: fit a simple SIR curve to the same Higgs window and compare it to IVF."""

import numpy as np
from scipy.integrate import odeint
from scipy.optimize import minimize

from common import ASSETS_DIR, HIGGS_TXT
from ivfs_validation import build_hourly_curve, ensure_dataset, fit_basic_ivf, parse_activity_file

import matplotlib

matplotlib.use('Agg')
import matplotlib.pyplot as plt


FIGURE_PATH = ASSETS_DIR / 'benchmark_curve_compare.png'


def sir_ode(y, t, beta, gamma):
    s_val, i_val, r_val = y
    ds = -beta * s_val * i_val
    di = beta * s_val * i_val - gamma * i_val
    dr = gamma * i_val
    return [ds, di, dr]


def fit_basic_sir(v_empirical_norm: np.ndarray) -> tuple[float, float, float, np.ndarray]:
    t_data = np.arange(len(v_empirical_norm), dtype=float)

    def loss(params):
        beta, gamma = np.abs(params)
        if beta < 1e-4 or gamma < 1e-4:
            return 1e9
        sol = odeint(sir_ode, [0.995, 0.005, 0.0], t_data, args=(beta, gamma), mxstep=10000)
        i_sim = sol[:, 1]
        imax = float(np.max(i_sim))
        if imax <= 1e-12:
            return 1e9
        i_sim_norm = i_sim / imax
        return float(np.sum((i_sim_norm - v_empirical_norm) ** 2))

    result = minimize(
        loss,
        x0=np.array([0.8, 0.1]),
        method='Nelder-Mead',
        options={'maxiter': 10000, 'xatol': 1e-8, 'fatol': 1e-8},
    )

    beta, gamma = np.abs(result.x)
    sol = odeint(sir_ode, [0.995, 0.005, 0.0], t_data, args=(beta, gamma), mxstep=10000)
    i_fit = sol[:, 1]
    i_fit_norm = i_fit / (np.max(i_fit) + 1e-12)
    return float(beta), float(gamma), float(result.fun), i_fit_norm


def make_plot(empirical_norm: np.ndarray, ivf_fit: np.ndarray, sir_fit: np.ndarray) -> None:
    plt.style.use('ggplot')
    fig, ax = plt.subplots(figsize=(8, 4.5))
    x = np.arange(len(empirical_norm))
    ax.scatter(x, empirical_norm, s=14, alpha=0.6, color='#616161', label='Empirical Higgs RT')
    ax.plot(x, ivf_fit, linewidth=2.2, color='#C62828', label='Content-centric IVF')
    ax.plot(x, sir_fit, linewidth=2.2, color='#1565C0', label='User-centric SIR')
    ax.set_title('Benchmark: Higgs RT Fit')
    ax.set_xlabel('Hour index in 100-hour window')
    ax.set_ylabel('Normalized activity')
    ax.legend()
    fig.tight_layout()
    fig.savefig(FIGURE_PATH, dpi=300, bbox_inches='tight')
    plt.close(fig)


def main() -> None:
    ensure_dataset()
    rt_timestamps, _ = parse_activity_file(HIGGS_TXT)
    _, _, _, window_counts = build_hourly_curve(rt_timestamps)
    empirical_norm = window_counts / np.max(window_counts)

    beta_ivf, gamma_ivf, sse_ivf, ivf_fit = fit_basic_ivf(empirical_norm)
    beta_sir, gamma_sir, sse_sir, sir_fit = fit_basic_sir(empirical_norm)
    make_plot(empirical_norm, ivf_fit, sir_fit)

    print('Benchmark against a simpler SIR model')
    print(f'IVF: beta={beta_ivf:.6f}, gamma={gamma_ivf:.6f}, SSE={sse_ivf:.6f}')
    print(f'SIR: beta={beta_sir:.6f}, gamma={gamma_sir:.6f}, SSE={sse_sir:.6f}')
    print(f'Saved figure: {FIGURE_PATH}')
    print('Takeaway: the point of IVFS is richer structure, not just winning a one-curve fit contest.')


if __name__ == '__main__':
    main()
