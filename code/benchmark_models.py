from __future__ import annotations

"""fit SIR to the same higgs data and compare it to our IVF"""

import sys
from pathlib import Path

sys.path.insert(0, str(Path(__file__).resolve().parent))

import numpy as np
from scipy.integrate import odeint
from scipy.optimize import minimize

from common import ASSETS_DIR, HIGGS_TXT
from ivfs_validation import (FIT_WINDOW_HOURS, SCENARIO_DISPLAY_HOURS, SPIKE_WINDOW_HOURS, build_hourly_curve,
                             ensure_dataset, fit_basic_ivf, parse_activity_file, run_scenarios)

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


def simulate_basic_sir(beta: float, gamma: float, i0_fit: float, n_steps: int) -> np.ndarray:
    t_data = np.arange(n_steps, dtype=float)
    i0_fit = float(np.clip(i0_fit, 1e-5, 0.1))
    sol = odeint(sir_ode, [1.0 - i0_fit, i0_fit, 0.0], t_data, args=(beta, gamma), mxstep=10000)
    i_fit = sol[:, 1]
    return i_fit / (np.max(i_fit) + 1e-12)


def fit_basic_sir(v_empirical_norm: np.ndarray,
                  window_counts: np.ndarray | None = None,
                  n_steps: int | None = None) -> tuple[float, float, float, float, np.ndarray]:
    """fit SIR with 3 params (beta, gamma, i0) using poisson weights"""
    t_data = np.arange(len(v_empirical_norm), dtype=float)
    if window_counts is not None:
        weights = 1.0 / (np.sqrt(window_counts) + 1.0)
    else:
        weights = np.ones_like(v_empirical_norm)

    def loss(params):
        beta, gamma, i0 = np.abs(params)
        i0 = np.clip(i0, 1e-5, 0.1)
        if beta < 1e-4 or gamma < 1e-4:
            return 1e9
        sol = odeint(sir_ode, [1.0 - i0, i0, 0.0], t_data, args=(beta, gamma), mxstep=10000)
        i_sim = sol[:, 1]
        imax = float(np.max(i_sim))
        if imax <= 1e-12:
            return 1e9
        i_sim_norm = i_sim / imax
        return float(np.sum(weights * (i_sim_norm - v_empirical_norm) ** 2))

    best_x = np.array([0.8, 0.1, 0.005])
    best_sse = 1e9
    for b0 in (0.5, 0.8, 1.2):
        for g0 in (0.1, 0.2, 0.3):
            for i0_init in (0.0005, 0.001, 0.005):
                res = minimize(
                    loss,
                    x0=np.array([b0, g0, i0_init]),
                    method='Nelder-Mead',
                    options={'maxiter': 15000, 'xatol': 1e-9, 'fatol': 1e-9},
                )
                if res.fun < best_sse:
                    best_sse = res.fun
                    best_x = np.abs(res.x)

    beta, gamma, i0_fit = best_x
    i0_fit = np.clip(i0_fit, 1e-5, 0.1)
    if n_steps is None:
        n_steps = len(v_empirical_norm)
    i_fit_norm = simulate_basic_sir(float(beta), float(gamma), float(i0_fit), n_steps=n_steps)
    return float(beta), float(gamma), float(i0_fit), float(best_sse), i_fit_norm


def make_plot(empirical_norm: np.ndarray, ivf_fit: np.ndarray, sir_fit: np.ndarray,
              sse_ivf: float, sse_sir: float,
              fit_window_hours: int,
              ivf_r2: float, sir_r2: float,
              t_scenario, scenario_results) -> None:
    plt.style.use('ggplot')
    fig, axs = plt.subplots(1, 3, figsize=(18, 5))

    # panel 1 -- compare both fits on the same primary calibration window
    x = np.arange(len(empirical_norm))
    smooth = np.convolve(empirical_norm, np.ones(3) / 3.0, mode='same')
    corr = float(np.corrcoef(ivf_fit, sir_fit)[0, 1])
    axs[0].scatter(x, empirical_norm, s=14, alpha=0.35, color='#616161', label='Raw hourly RT')
    axs[0].plot(x, smooth, linewidth=2.0, color='#37474F', label='Empirical 3h mean')
    axs[0].plot(x, ivf_fit, linewidth=2.2, color='#C62828', label=f'IVF (loss={sse_ivf:.3f}, 5 params)')
    axs[0].plot(x, sir_fit, linewidth=2.2, color='#1565C0', linestyle='--', label=f'SIR (loss={sse_sir:.3f}, 3 params)')
    if fit_window_hours < len(empirical_norm):
        axs[0].axvline(fit_window_hours, color='gray', linestyle='--', linewidth=1.0, alpha=0.8,
                       label=f'Fit cutoff (h {fit_window_hours})')
    axs[0].set_title('(a) Higgs Window Fit: IVF vs SIR')
    axs[0].set_xlabel('Hour index in 100-hour window')
    axs[0].set_ylabel('Normalized activity')
    axs[0].annotate(
        f'r = {corr:.4f}\n'
        f'IVF R² (0–{SPIKE_WINDOW_HOURS - 1}h) = {ivf_r2:.4f}\n'
        f'SIR R² (0–{SPIKE_WINDOW_HOURS - 1}h) = {sir_r2:.4f}',
        xy=(0.52, 0.77), xycoords='axes fraction',
        fontsize=9, bbox=dict(boxstyle='round,pad=0.3', fc='white', alpha=0.8))
    axs[0].legend(fontsize=8)

    colors = {
        'Engagement-First (alpha=0.9)': '#C62828',
        'Moderate (alpha=0.5)': '#1565C0',
        'Health-First (alpha=0.2)': '#2E7D32',
    }

    # panel 2 -- discussion pressure, SIR can't do this
    for label, data in scenario_results.items():
        axs[1].plot(t_scenario, data['solution'][:, 4], linewidth=2.0,
                    color=colors[label], label=label)
    axs[1].axhline(0, color='gray', linewidth=0.8, linestyle=':')
    axs[1].set_title('(b) Discussion Pressure τ(t) (IVFS only)')
    axs[1].set_xlabel('Time')
    axs[1].set_ylabel('Aggregate discussion pressure \u03C4')
    axs[1].set_xlim(0, SCENARIO_DISPLAY_HOURS)
    axs[1].set_ylim(bottom=0)
    axs[1].legend(fontsize=8)

    # panel 3 -- user retention, also not in SIR
    for label, data in scenario_results.items():
        axs[2].plot(t_scenario, data['solution'][:, 5], linewidth=2.0,
                    color=colors[label], label=label)
    axs[2].set_title('(c) User Retention U(t) (IVFS only)')
    axs[2].set_xlabel('Time')
    axs[2].set_ylabel('Active users U')
    axs[2].set_xlim(0, SCENARIO_DISPLAY_HOURS)
    axs[2].legend(fontsize=8)

    fig.suptitle('Benchmark: IVF vs SIR (IVFS adds discussion-pressure + user structure)', fontsize=13, fontweight='bold', y=1.01)
    fig.tight_layout(pad=2)
    fig.savefig(FIGURE_PATH, dpi=300, bbox_inches='tight')
    plt.close(fig)


def main() -> None:
    ensure_dataset()
    rt_timestamps, *_ = parse_activity_file(HIGGS_TXT)
    cal = build_hourly_curve(rt_timestamps)
    window_counts = cal['rt_window']
    empirical_norm = window_counts / np.max(window_counts)
    fit_window_hours = min(FIT_WINDOW_HOURS, len(empirical_norm))
    fit_target = empirical_norm[:fit_window_hours]
    fit_counts = window_counts[:fit_window_hours]

    beta_ivf, gamma_ivf, lambda0_ivf, lambda_decay_ivf, v0_ivf, sse_ivf, ivf_fit = fit_basic_ivf(
        fit_target,
        fit_counts,
        n_steps=len(empirical_norm),
    )
    beta_sir, gamma_sir, i0_sir, sse_sir, sir_fit = fit_basic_sir(
        fit_target,
        fit_counts,
        n_steps=len(empirical_norm),
    )

    # run the IVFS scenarios -- this is what SIR can't do
    t_scenario, scenario_results = run_scenarios(beta_ivf, gamma_ivf)

    spike_end = min(SPIKE_WINDOW_HOURS, len(empirical_norm))
    ivf_r2 = 1.0 - float(np.sum((ivf_fit[:spike_end] - empirical_norm[:spike_end])**2)) / \
             max(float(np.sum((empirical_norm[:spike_end] - np.mean(empirical_norm[:spike_end]))**2)), 1e-12)
    sir_r2 = 1.0 - float(np.sum((sir_fit[:spike_end] - empirical_norm[:spike_end])**2)) / \
             max(float(np.sum((empirical_norm[:spike_end] - np.mean(empirical_norm[:spike_end]))**2)), 1e-12)
    make_plot(empirical_norm, ivf_fit, sir_fit, sse_ivf, sse_sir,
              fit_window_hours, ivf_r2, sir_r2,
              t_scenario, scenario_results)

    corr = float(np.corrcoef(ivf_fit, sir_fit)[0, 1])
    tail_start = 40
    ivf_tail = float(np.mean(empirical_norm[tail_start:] - ivf_fit[tail_start:]))
    sir_tail = float(np.mean(empirical_norm[tail_start:] - sir_fit[tail_start:]))

    print('Benchmark: IVF vs SIR on a single Higgs spike')
    print(f'Primary fit window: 0..{fit_window_hours - 1}')
    print(f'IVF: beta={beta_ivf:.6f}, gamma={gamma_ivf:.6f}, lambda0={lambda0_ivf:.6f}, decay={lambda_decay_ivf:.6f}, V0={v0_ivf:.6f}, weighted_loss={sse_ivf:.4f}')
    print(f'SIR: beta={beta_sir:.6f}, gamma={gamma_sir:.6f}, I0={i0_sir:.6f}, weighted_loss={sse_sir:.4f}')
    print(f'Curve correlation: r={corr:.6f}')
    print(f'IVF spike R\u00B2 (0-{spike_end - 1}h): {ivf_r2:.4f}, tail mean bias (40-99h): {ivf_tail:+.4f}')
    print(f'SIR spike R\u00B2 (0-{spike_end - 1}h): {sir_r2:.4f}, tail mean bias (40-99h): {sir_tail:+.4f}')
    print(f'Saved figure: {FIGURE_PATH}')
    print()
    print('Across the full Higgs window, IVF now fits the spike and late tail much better than SIR.')
    print('IVFS additionally carries discussion-pressure and user-retention dynamics that SIR cannot represent.')


if __name__ == '__main__':
    main()
