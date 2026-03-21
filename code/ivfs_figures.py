from __future__ import annotations

import numpy as np

import matplotlib

matplotlib.use('Agg')
import matplotlib.pyplot as plt

from ivfs_calibration import moving_average
from ivfs_config import (DIAG_FIGURE_PATH, DISPLAY_WINDOW_HOURS, FIGURE_PATH, PHI_SENS_FIGURE_PATH,
                         PROFILE_FIGURE_PATH, SCENARIO_ALPHAS, SCENARIO_DISPLAY_HOURS, SMOOTH_WINDOW,
                         SPIKE_WINDOW_HOURS, TAIL_START_HOURS, TAU_COMPARE_FIGURE_PATH)


def make_calibration_figure(empirical_norm, fitted_norm, window_counts,
                            re_window, reply_proxy,
                            fit_window_hours,
                            weighted_loss, r2_fit, r2_full, r2_spike, tail_bias):
    """2x2 calibration diagnostics -- compare what was fit vs what is only extrapolated"""
    plt.style.use('ggplot')
    fig, axs = plt.subplots(2, 2, figsize=(14, 10))
    t_data = np.arange(len(empirical_norm))
    fit_end = min(int(fit_window_hours), len(empirical_norm))
    zoom_end = min(DISPLAY_WINDOW_HOURS, len(empirical_norm))
    spike_end = min(SPIKE_WINDOW_HOURS, len(empirical_norm))
    tail_start = min(TAIL_START_HOURS, len(empirical_norm))
    empirical_smooth = moving_average(empirical_norm, SMOOTH_WINDOW)
    sigma_band = 2.0 * np.sqrt(window_counts) / (np.max(window_counts) + 1e-12)
    lower_band = np.clip(empirical_norm - sigma_band, 0.0, None)
    upper_band = empirical_norm + sigma_band

    zoom_empirical = empirical_norm[:zoom_end]
    zoom_fitted = fitted_norm[:zoom_end]
    zoom_smooth = empirical_smooth[:zoom_end]
    zoom_ss_tot = float(np.sum((zoom_empirical - np.mean(zoom_empirical)) ** 2))
    r2_zoom = 1.0 - float(np.sum((zoom_fitted - zoom_empirical) ** 2)) / zoom_ss_tot if zoom_ss_tot > 0 else 0.0
    smooth_ss_tot = float(np.sum((zoom_smooth - np.mean(zoom_smooth)) ** 2))
    if smooth_ss_tot > 0:
        r2_vs_smooth = 1.0 - float(np.sum((zoom_fitted - zoom_smooth) ** 2)) / smooth_ss_tot
    else:
        r2_vs_smooth = 0.0

    axs[0, 0].fill_between(t_data, lower_band, upper_band, color='#B0BEC5', alpha=0.25,
                           label='Approx. ±2σ count noise')
    axs[0, 0].scatter(t_data, empirical_norm, s=14, alpha=0.35, color='#616161',
                      label='Raw hourly RT')
    axs[0, 0].plot(t_data, empirical_smooth, color='#37474F', linewidth=2.0,
                   label=f'Empirical {SMOOTH_WINDOW}h mean')
    axs[0, 0].plot(t_data, fitted_norm, color='#C62828', linewidth=2.3, label='Fitted IVF')
    if fit_end < len(empirical_norm):
        axs[0, 0].axvline(fit_end, color='gray', linestyle='--', linewidth=1.0, alpha=0.8,
                          label=f'Fit cutoff (h {fit_end})')
    axs[0, 0].set_xlim(0, zoom_end)
    axs[0, 0].set_ylim(bottom=0)
    axs[0, 0].set_title(f'(a) Spike Region Zoom (0–{zoom_end - 1} h)')
    axs[0, 0].set_xlabel('Hour index')
    axs[0, 0].set_ylabel('Normalized viral volume')
    axs[0, 0].annotate(
        f'Weighted loss = {weighted_loss:.4f}\n'
        f'R² (zoom 0–{zoom_end - 1}h) = {r2_zoom:.4f}\n'
        f'R² vs {SMOOTH_WINDOW}h mean (0–{zoom_end - 1}h) = {r2_vs_smooth:.4f}',
        xy=(0.41, 0.72), xycoords='axes fraction', fontsize=9,
        bbox=dict(boxstyle='round,pad=0.3', fc='white', alpha=0.8))
    axs[0, 0].legend(fontsize=8)

    axs[0, 1].scatter(t_data, empirical_norm, s=14, alpha=0.35, color='#616161',
                      label='Raw hourly RT')
    axs[0, 1].plot(t_data, empirical_smooth, color='#37474F', linewidth=2.0,
                   label=f'Empirical {SMOOTH_WINDOW}h mean')
    axs[0, 1].plot(t_data, fitted_norm, color='#C62828', linewidth=2.1, label='Fitted IVF')
    axs[0, 1].axvline(spike_end, color='gray', linestyle=':', linewidth=1.0, alpha=0.7,
                      label=f'Spike cutoff (h {spike_end})')
    axs[0, 1].axvline(tail_start, color='gray', linestyle='--', linewidth=1.0, alpha=0.6,
                      label=f'Tail window start (h {tail_start})')
    axs[0, 1].set_title('(b) Full 100-Hour Window Calibration')
    axs[0, 1].set_xlabel('Hour index')
    axs[0, 1].set_ylabel('Normalized viral volume')
    axs[0, 1].annotate(
        f'R² (spike 0–{spike_end - 1}h) = {r2_spike:.4f}\n'
        f'R² (full 0–{len(empirical_norm) - 1}h) = {r2_full:.4f}\n'
        f'Tail mean bias (h {tail_start}–{len(empirical_norm) - 1}) = {tail_bias:+.4f}',
        xy=(0.42, 0.70), xycoords='axes fraction', fontsize=9,
        bbox=dict(boxstyle='round,pad=0.3', fc='white', alpha=0.8))
    axs[0, 1].legend(fontsize=8)

    if re_window is not None and reply_proxy is not None:
        t_re = np.arange(len(re_window))
        re_norm = re_window / (np.max(re_window) + 1e-12)
        proxy_norm = reply_proxy / (np.max(reply_proxy) + 1e-12)
        rt_reference = empirical_smooth / (np.max(empirical_smooth) + 1e-12)
        re_corr = float(np.corrcoef(re_window[:fit_end], window_counts[:fit_end])[0, 1])

        axs[1, 0].bar(t_re, re_norm, alpha=0.35, color='#7B1FA2',
                      label='RE events (normalized)')
        axs[1, 0].plot(t_re, proxy_norm, color='#4A148C', linewidth=2.0,
                       label='Reply proxy (smoothed RE activity)')
        axs[1, 0].plot(t_data, rt_reference, color='#455A64', linewidth=1.8, linestyle='--',
                       label='RT reference (smoothed)')
        axs[1, 0].set_xlim(0, zoom_end)
        axs[1, 0].set_ylim(bottom=0)
        axs[1, 0].set_title('(c) Reply-Pressure Proxy (Higgs RE Activity)')
        axs[1, 0].set_xlabel('Hour index')
        axs[1, 0].set_ylabel('Normalized reply activity')
        re_total = int(np.sum(re_window))
        axs[1, 0].annotate(
            f'RE events in window: {re_total:,}\n'
            f'corr(RE, RT) over fit window = {re_corr:.4f}',
            xy=(0.04, 0.84), xycoords='axes fraction', fontsize=9,
            bbox=dict(boxstyle='round,pad=0.3', fc='white', alpha=0.8))
        axs[1, 0].legend(fontsize=8, loc='lower right')
    else:
        axs[1, 0].text(0.5, 0.5, 'No RE data available', transform=axs[1, 0].transAxes,
                       ha='center', va='center', fontsize=12, color='gray')
        axs[1, 0].set_title('(c) Reply-Pressure Proxy')

    residuals = fitted_norm - empirical_norm
    bar_colors = ['#C62828' if r > 0 else '#1565C0' for r in residuals]
    axs[1, 1].bar(t_data, residuals, color=bar_colors, alpha=0.6)
    axs[1, 1].plot(t_data, moving_average(residuals, SMOOTH_WINDOW), color='#263238',
                   linewidth=1.8, label=f'{SMOOTH_WINDOW}h residual mean')
    axs[1, 1].axhline(0, color='black', linewidth=0.8)
    axs[1, 1].axvline(spike_end, color='gray', linestyle=':', linewidth=1.0, alpha=0.7,
                      label=f'Spike cutoff (h {spike_end})')
    axs[1, 1].axvline(tail_start, color='gray', linestyle='--', linewidth=1.0, alpha=0.6,
                      label=f'Tail window start (h {tail_start})')
    axs[1, 1].set_title('(d) Residuals (Model − Raw Data)')
    axs[1, 1].set_xlabel('Hour index')
    axs[1, 1].set_ylabel('Residual')
    axs[1, 1].legend(fontsize=8)

    fig.suptitle('APM348 IVFS Calibration Diagnostics', fontsize=14, fontweight='bold', y=1.01)
    fig.tight_layout(pad=2)
    fig.savefig(DIAG_FIGURE_PATH, dpi=300, bbox_inches='tight')
    plt.close(fig)


def make_tau_comparison_figure(reply_proxy: np.ndarray,
                               re_window: np.ndarray | None,
                               tau_configs: dict[str, dict]) -> None:
    """compare the RE proxy against the current, Higgs-fit, and external tau setups"""
    plt.style.use('ggplot')
    fig, axs = plt.subplots(1, 2, figsize=(14, 5.5))
    t_data = np.arange(len(reply_proxy))

    if re_window is not None:
        re_norm = re_window / (np.max(re_window) + 1e-12)
        axs[0].bar(t_data, re_norm, alpha=0.25, color='#9575CD', label='RE events (normalized)')
    axs[0].plot(t_data, reply_proxy, color='black', linewidth=2.3, label='Empirical reply-pressure proxy')
    for cfg in tau_configs.values():
        axs[0].plot(t_data, cfg['tau_curve'], linewidth=2.0, color=cfg['color'], label=cfg['label'])
    axs[0].set_title('(a) Higgs Reply-Pressure Proxy Fit')
    axs[0].set_xlabel('Hour index in 100-hour window')
    axs[0].set_ylabel('Normalized discussion pressure')
    axs[0].set_ylim(bottom=0)
    axs[0].legend(fontsize=8)

    scenario_order = list(SCENARIO_ALPHAS.keys())[::-1]
    x = np.arange(len(scenario_order), dtype=float)
    width = 0.24
    offsets = np.linspace(-width, width, len(tau_configs))
    for offset, cfg in zip(offsets, tau_configs.values()):
        tau_vals = [max(0.0, float(cfg['scenario_results'][label]['tau_star'])) for label in scenario_order]
        axs[1].bar(x + offset, tau_vals, width=width, color=cfg['color'], alpha=0.9, label=cfg['label'])
    axs[1].set_xticks(x)
    axs[1].set_xticklabels(['Health-First', 'Moderate', 'Engagement-First'])
    axs[1].set_title('(b) Long-Run Discussion Pressure $\\tau^*$ by Policy')
    axs[1].set_xlabel('Policy scenario')
    axs[1].set_ylabel('$\\tau^*$')
    axs[1].set_ylim(bottom=0)
    axs[1].legend(fontsize=8)

    fig.suptitle('APM348 Tau-Side Comparison (Higgs Proxy + External Reference)', fontsize=13, fontweight='bold', y=1.02)
    fig.tight_layout(pad=2)
    fig.savefig(TAU_COMPARE_FIGURE_PATH, dpi=300, bbox_inches='tight')
    plt.close(fig)


def make_results_figure(t_scenario, scenario_results,
                        alphas, tau_star, v_star_cont, u_star_cont, alpha_r0):
    """2x3 policy results -- no calibration panel, purely scenario output"""
    plt.style.use('ggplot')
    fig, axs = plt.subplots(2, 3, figsize=(18, 10))

    colors = {
        'Engagement-First (alpha=0.9)': '#C62828',
        'Moderate (alpha=0.5)': '#1565C0',
        'Health-First (alpha=0.2)': '#2E7D32',
    }

    for label, data in scenario_results.items():
        axs[0, 0].plot(t_scenario, data['solution'][:, 1], linewidth=2.0,
                       color=colors[label], label=label)
    axs[0, 0].set_title('(a) Viral Volume V(t) by Policy')
    axs[0, 0].set_xlabel('Time')
    axs[0, 0].set_ylabel('Viral volume V')
    axs[0, 0].set_xlim(0, SCENARIO_DISPLAY_HOURS)
    axs[0, 0].set_ylim(bottom=0)
    axs[0, 0].legend(fontsize=8)

    for label, data in scenario_results.items():
        axs[0, 1].plot(t_scenario, data['solution'][:, 4], linewidth=2.0,
                       color=colors[label], label=label)
    axs[0, 1].set_title('(b) Discussion Pressure τ(t) by Policy')
    axs[0, 1].set_xlabel('Time')
    axs[0, 1].set_ylabel('Aggregate discussion pressure τ')
    axs[0, 1].set_xlim(0, SCENARIO_DISPLAY_HOURS)
    axs[0, 1].set_ylim(bottom=0)
    axs[0, 1].legend(fontsize=8)

    for label, data in scenario_results.items():
        axs[0, 2].plot(t_scenario, data['solution'][:, 5], linewidth=2.0,
                       color=colors[label], label=label)
    axs[0, 2].set_title('(c) Active Users U(t) by Policy')
    axs[0, 2].set_xlabel('Time')
    axs[0, 2].set_ylabel('Active users U')
    axs[0, 2].set_xlim(0, SCENARIO_DISPLAY_HOURS)
    axs[0, 2].legend(fontsize=8)

    axs[1, 0].plot(alphas, tau_star, color='#6A1B9A', linewidth=2.2)
    axs[1, 0].axvline(alpha_r0, color='black', linestyle='--', linewidth=1.5,
                      label=f'R₀=1 at α≈{alpha_r0:.2f}')
    axs[1, 0].set_title('(d) Equilibrium Discussion Pressure τ* vs α')
    axs[1, 0].set_xlabel('Amplification α')
    axs[1, 0].set_ylabel('τ*')
    axs[1, 0].legend(fontsize=8)

    axs[1, 1].plot(alphas, u_star_cont, color='#00695C', linewidth=2.2)
    axs[1, 1].axvline(alpha_r0, color='black', linestyle='--', linewidth=1.5,
                      label=f'R₀=1 at α≈{alpha_r0:.2f}')
    axs[1, 1].set_title('(e) Equilibrium Users U* vs α')
    axs[1, 1].set_xlabel('Amplification α')
    axs[1, 1].set_ylabel('U*')
    axs[1, 1].legend(fontsize=8)

    e_star = alphas * v_star_cont * u_star_cont
    axs[1, 2].plot(alphas, e_star, color='#E65100', linewidth=2.2)
    axs[1, 2].axvline(alpha_r0, color='black', linestyle='--', linewidth=1.5,
                      label=f'R₀=1 at α≈{alpha_r0:.2f}')
    peak_idx = int(np.argmax(e_star))
    axs[1, 2].axvline(alphas[peak_idx], color='#E65100', linestyle=':', linewidth=1.5,
                      label=f'Peak E* at α≈{alphas[peak_idx]:.2f}')
    axs[1, 2].set_title('(f) Engagement E* = α·V*·U* vs α')
    axs[1, 2].set_xlabel('Amplification α')
    axs[1, 2].set_ylabel('E*')
    axs[1, 2].legend(fontsize=8)

    fig.suptitle('APM348 IVFS Policy Scenarios', fontsize=14, fontweight='bold', y=1.01)
    fig.tight_layout(pad=2)
    fig.savefig(FIGURE_PATH, dpi=300, bbox_inches='tight')
    plt.close(fig)


def make_phi_sensitivity_figure(phi_grid: np.ndarray,
                                phi_results: dict[str, dict[str, np.ndarray | float]],
                                current_phi: float,
                                external_phi: float | None = None,
                                external_source: str | None = None) -> None:
    """save a small robustness figure for the toxicity coupling PHI"""
    plt.style.use('ggplot')
    fig, axs = plt.subplots(1, 2, figsize=(12.5, 4.8))

    colors = {
        'Engagement-First (alpha=0.9)': '#C62828',
        'Moderate (alpha=0.5)': '#1565C0',
        'Health-First (alpha=0.2)': '#2E7D32',
    }

    for label, data in phi_results.items():
        axs[0].plot(phi_grid, data['tau_star'], linewidth=2.1, color=colors[label], label=label)
        axs[1].plot(phi_grid, data['U_star'], linewidth=2.1, color=colors[label], label=label)

    for ax in axs:
        ax.axvline(current_phi, color='black', linestyle='--', linewidth=1.2,
                   label=f'Current PHI={current_phi:.3f}')
        if external_phi is not None:
            ext_label = f'External PHI≈{external_phi:.3f}'
            if external_source:
                ext_label = f'{external_source} PHI≈{external_phi:.3f}'
            ax.axvline(external_phi, color='#6A1B9A', linestyle=':', linewidth=1.4, label=ext_label)
        ax.set_xlabel('Toxicity coupling PHI')

    axs[0].set_title('(a) Equilibrium Discussion Pressure $\\tau^*$ vs PHI')
    axs[0].set_ylabel('$\\tau^*$')
    axs[0].set_ylim(bottom=0)
    axs[0].legend(fontsize=8)

    axs[1].set_title('(b) Equilibrium Users $U^*$ vs PHI')
    axs[1].set_ylabel('$U^*$')
    axs[1].legend(fontsize=8)

    fig.suptitle('APM348 PHI Sensitivity (Discussion-Pressure Block)', fontsize=13, fontweight='bold', y=1.02)
    fig.tight_layout(pad=2)
    fig.savefig(PHI_SENS_FIGURE_PATH, dpi=300, bbox_inches='tight')
    plt.close(fig)


def plot_profile(beta_grid, profile_beta, gamma_grid, profile_gamma,
                 beta0_best, gamma0_best) -> None:
    """save the profile likelihood plots"""
    plt.style.use('ggplot')
    fig, (ax1, ax2) = plt.subplots(1, 2, figsize=(12, 4.5))
    beta_excess = profile_beta - float(np.min(profile_beta))
    gamma_excess = profile_gamma - float(np.min(profile_gamma))

    ax1.plot(beta_grid, beta_excess, 'o-', markersize=3, color='#C62828')
    ax1.axvline(beta0_best, color='black', linestyle='--', linewidth=1.2,
                label=f'β₀={beta0_best:.3f}')
    ax1.set_xlabel('β₀ (fixed)')
    ax1.set_ylabel('Excess profile weighted loss')
    ax1.set_title('Local Profile Likelihood: β₀')
    ax1.legend(fontsize=9)

    ax2.plot(gamma_grid, gamma_excess, 'o-', markersize=3, color='#1565C0')
    ax2.axvline(gamma0_best, color='black', linestyle='--', linewidth=1.2,
                label=f'γ₀={gamma0_best:.3f}')
    ax2.set_xlabel('γ₀ (fixed)')
    ax2.set_ylabel('Excess profile weighted loss')
    ax2.set_title('Local Profile Likelihood: γ₀')
    ax2.legend(fontsize=9)

    fig.tight_layout()
    fig.savefig(PROFILE_FIGURE_PATH, dpi=300, bbox_inches='tight')
    plt.close(fig)
    print(f'Saved profile likelihood figure: {PROFILE_FIGURE_PATH}')
