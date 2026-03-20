from __future__ import annotations

"""Quick benchmark: fit a simple SIR curve to the same Higgs window and compare it to IVF."""

import numpy as np
from scipy.integrate import odeint
from scipy.optimize import minimize

from common import ASSETS_DIR, HIGGS_TXT
from ivfs_validation import (build_hourly_curve, ensure_dataset, fit_basic_ivf,
                              full_ivfs_ode, parse_activity_file, run_scenarios)

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


def make_plot(empirical_norm: np.ndarray, ivf_fit: np.ndarray, sir_fit: np.ndarray,
              sse_ivf: float, sse_sir: float,
              t_scenario, scenario_results) -> None:
    plt.style.use('ggplot')
    fig, axs = plt.subplots(1, 3, figsize=(18, 5))

    # Panel 1 — they end up almost identical on a single spike, and that's fine
    x = np.arange(len(empirical_norm))
    corr = float(np.corrcoef(ivf_fit, sir_fit)[0, 1])
    axs[0].scatter(x, empirical_norm, s=14, alpha=0.55, color='#616161', label='Empirical Higgs RT')
    axs[0].plot(x, ivf_fit, linewidth=2.2, color='#C62828', label=f'IVF (SSE={sse_ivf:.2f})')
    axs[0].plot(x, sir_fit, linewidth=2.2, color='#1565C0', linestyle='--', label=f'SIR (SSE={sse_sir:.2f})')
    axs[0].set_title('(a) Single-Spike Fit: IVF \u2248 SIR')
    axs[0].set_xlabel('Hour index in 100-hour window')
    axs[0].set_ylabel('Normalized activity')
    axs[0].annotate(f'r = {corr:.4f}', xy=(0.55, 0.85), xycoords='axes fraction',
                    fontsize=10, bbox=dict(boxstyle='round,pad=0.3', fc='white', alpha=0.8))
    axs[0].legend(fontsize=8)

    colors = {
        'Engagement-First (alpha=0.9)': '#C62828',
        'Moderate (alpha=0.5)': '#1565C0',
        'Health-First (alpha=0.2)': '#2E7D32',
    }

    # Panel 2 — toxicity over time; SIR just can't do this
    for label, data in scenario_results.items():
        axs[1].plot(t_scenario, data['solution'][:, 3], linewidth=2.0,
                    color=colors[label], label=label)
    axs[1].axhline(0, color='gray', linewidth=0.8, linestyle=':')
    axs[1].set_title('(b) Toxicity τ(t) (IVFS only)')
    axs[1].set_xlabel('Time')
    axs[1].set_ylabel('Aggregate toxicity \u03C4')
    axs[1].legend(fontsize=8)

    # Panel 3 — user retention over time; again, not something SIR tracks
    for label, data in scenario_results.items():
        axs[2].plot(t_scenario, data['solution'][:, 4], linewidth=2.0,
                    color=colors[label], label=label)
    axs[2].set_title('(c) User Retention U(t) (IVFS only)')
    axs[2].set_xlabel('Time')
    axs[2].set_ylabel('Active users U')
    axs[2].legend(fontsize=8)

    fig.suptitle('IVF vs SIR Comparison', fontsize=13, fontweight='bold', y=1.01)
    fig.tight_layout(pad=2)
    fig.savefig(FIGURE_PATH, dpi=300, bbox_inches='tight')
    plt.close(fig)


def main() -> None:
    ensure_dataset()
    rt_timestamps, _ = parse_activity_file(HIGGS_TXT)
    _, _, _, window_counts = build_hourly_curve(rt_timestamps)
    empirical_norm = window_counts / np.max(window_counts)

    beta_ivf, gamma_ivf, lambda_bg, sse_ivf, ivf_fit = fit_basic_ivf(empirical_norm)
    beta_sir, gamma_sir, sse_sir, sir_fit = fit_basic_sir(empirical_norm)

    # Run IVFS scenarios — this is the stuff SIR can't capture.
    t_scenario, scenario_results = run_scenarios(beta_ivf, gamma_ivf)

    make_plot(empirical_norm, ivf_fit, sir_fit, sse_ivf, sse_sir,
              t_scenario, scenario_results)

    corr = float(np.corrcoef(ivf_fit, sir_fit)[0, 1])
    print('Benchmark: IVF vs SIR on a single Higgs spike')
    print(f'IVF: beta={beta_ivf:.6f}, gamma={gamma_ivf:.6f}, lambda_bg={lambda_bg:.6f}, SSE={sse_ivf:.4f}')
    print(f'SIR: beta={beta_sir:.6f}, gamma={gamma_sir:.6f}, SSE={sse_sir:.4f}')
    print(f'Curve correlation: r={corr:.6f}')
    print(f'Saved figure: {FIGURE_PATH}')
    print()
    print('Takeaway: on a single normalized spike, IVF and SIR look basically the same.')
    print('IVFS earns its keep with the toxicity and user-retention dynamics')
    print('that SIR just doesn\'t have (panels b and c).')


if __name__ == '__main__':
    main()
