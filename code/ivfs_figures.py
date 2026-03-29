from __future__ import annotations

import numpy as np

import matplotlib

matplotlib.use('Agg')
import matplotlib.pyplot as plt

from .ivfs_calibration import moving_average
from .ivfs_config import (DIAG_FIGURE_PATH, DISPLAY_WINDOW_HOURS, FIGURE_PATH, PHI_SENS_FIGURE_PATH,
                          PROFILE_FIGURE_PATH, SCENARIO_ALPHAS, SCENARIO_DISPLAY_HOURS, SMOOTH_WINDOW,
                          SPIKE_WINDOW_HOURS, TAIL_START_HOURS, TAU_COMPARE_FIGURE_PATH)
from .plot_style import (BAND_COLOR, ENGAGEMENT_COLOR, FIT_COLOR, LEGEND_FONT_SIZE,
                         OBSERVED_COLOR, OBSERVED_MARKER_SIZE, PRESSURE_COLOR,
                         REFERENCE_COLOR, SCENARIO_COLORS, SHADE_COLOR, SMOOTH_COLOR,
                         SUPTITLE_FONT_SIZE, THRESHOLD_COLOR, USER_COLOR, add_metric_box,
                         add_threshold_shading, apply_plot_style, finish_axes)


def make_calibration_figure(empirical_norm, fitted_norm, window_counts,
                            re_window, mt_window, tau_proxy,
                            re_proxy, ratio_proxy,
                            selected_proxy, selected_proxy_label,
                            fit_window_hours,
                            weighted_loss, r2_fit, r2_full, r2_spike, tail_bias,
                            fit_band: tuple[np.ndarray, np.ndarray] | None = None):
    """Plot the fit, proxy evidence, and residual structure for the Higgs calibration window."""
    apply_plot_style()
    fig, axs = plt.subplots(2, 2, figsize=(14.5, 9.8))
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

    if fit_band is not None:
        band_low, band_high = fit_band
        axs[0, 0].fill_between(t_data, band_low, band_high, color=BAND_COLOR, alpha=0.28,
                               label='Bootstrap 90% fit band')
    axs[0, 0].fill_between(t_data, lower_band, upper_band, color=SHADE_COLOR, alpha=0.45,
                           label='Approx. count-noise envelope')
    axs[0, 0].scatter(t_data, empirical_norm, s=OBSERVED_MARKER_SIZE, alpha=0.55, color=OBSERVED_COLOR,
                      label='Observed hourly retweets', zorder=3)
    axs[0, 0].plot(t_data, empirical_smooth, color=SMOOTH_COLOR, linewidth=2.0,
                   label=f'Observed {SMOOTH_WINDOW}h mean')
    axs[0, 0].plot(t_data, fitted_norm, color=FIT_COLOR, linewidth=2.4, label='Fitted IVF trajectory')
    if fit_end < len(empirical_norm):
        axs[0, 0].axvline(fit_end, color=THRESHOLD_COLOR, linestyle='--', linewidth=1.1, alpha=0.8,
                          label=f'Fit cutoff (h {fit_end})')
    axs[0, 0].set_xlim(0, zoom_end)
    axs[0, 0].set_ylim(bottom=0)
    axs[0, 0].set_title(f'(a) Main spike fit on hours 0 to {zoom_end - 1}')
    finish_axes(axs[0, 0], 'Hours since calibration window start', 'Normalized retweet volume')
    add_metric_box(
        axs[0, 0],
        [
            f'Weighted SSE = {weighted_loss:.4f}',
            f'R² on fit window = {r2_fit:.4f}',
            f'R² on 0 to {zoom_end - 1} h = {r2_zoom:.4f}',
            f'Fit window = 0 to {fit_end - 1} h',
        ],
    )
    axs[0, 0].legend(loc='lower right', fontsize=LEGEND_FONT_SIZE)

    if fit_band is not None:
        axs[0, 1].fill_between(t_data, band_low, band_high, color=BAND_COLOR, alpha=0.28,
                               label='Bootstrap 90% fit band')
    axs[0, 1].scatter(t_data, empirical_norm, s=OBSERVED_MARKER_SIZE, alpha=0.55, color=OBSERVED_COLOR,
                      label='Observed hourly retweets', zorder=3)
    axs[0, 1].plot(t_data, empirical_smooth, color=SMOOTH_COLOR, linewidth=2.0,
                   label=f'Observed {SMOOTH_WINDOW}h mean')
    axs[0, 1].plot(t_data, fitted_norm, color=FIT_COLOR, linewidth=2.3, label='Fitted IVF trajectory')
    axs[0, 1].axvline(spike_end, color=REFERENCE_COLOR, linestyle=':', linewidth=1.1, alpha=0.8,
                      label=f'Spike cutoff (h {spike_end})')
    axs[0, 1].axvline(tail_start, color=THRESHOLD_COLOR, linestyle='--', linewidth=1.1, alpha=0.7,
                      label=f'Tail window start (h {tail_start})')
    axs[0, 1].set_title('(b) Full calibration window and tail behaviour')
    finish_axes(axs[0, 1], 'Hours since calibration window start', 'Normalized retweet volume')
    add_metric_box(
        axs[0, 1],
        [
            f'R² on spike window = {r2_spike:.4f}',
            f'R² on full window = {r2_full:.4f}',
            f'Tail mean bias = {tail_bias:+.4f}',
            f'R² vs smoothed series = {r2_vs_smooth:.4f}',
        ],
    )
    axs[0, 1].legend(loc='lower right', fontsize=LEGEND_FONT_SIZE)

    if re_window is not None and selected_proxy is not None:
        t_re = np.arange(len(re_window))
        re_norm = re_window / (np.max(re_window) + 1e-12)
        selected_proxy_norm = selected_proxy / (np.max(selected_proxy) + 1e-12)
        rt_reference = empirical_smooth / (np.max(empirical_smooth) + 1e-12)
        re_corr = float(np.corrcoef(re_window[:fit_end], window_counts[:fit_end])[0, 1])
        selected_key = str(selected_proxy_label).strip().lower()

        axs[1, 0].bar(t_re, re_norm, alpha=0.35, color=PRESSURE_COLOR,
                      label='Observed replies (normalized)')
        if mt_window is not None:
            mt_norm = mt_window / (np.max(mt_window) + 1e-12)
            axs[1, 0].plot(t_re, mt_norm, color='#0F766E', linewidth=1.7,
                           label='Mentions (normalized)')
        if re_proxy is not None and selected_key != 're proxy':
            axs[1, 0].plot(t_re, re_proxy, color='#6D28D9', linewidth=1.8,
                           label='RE proxy')
        if ratio_proxy is not None and selected_key != 're/(rt+1) proxy':
            axs[1, 0].plot(t_re, ratio_proxy, color='#EA580C', linewidth=1.8,
                           label='RE/(RT+1) proxy')
        if tau_proxy is not None and selected_key != 'composite proxy':
            axs[1, 0].plot(t_re, tau_proxy, color='#6B4F3A', linewidth=1.8,
                           label='Composite proxy')
        axs[1, 0].plot(t_re, selected_proxy_norm, color=FIT_COLOR, linewidth=2.4,
                       label=f'Selected proxy ({selected_proxy_label})')
        axs[1, 0].plot(t_data, rt_reference, color=REFERENCE_COLOR, linewidth=1.8, linestyle='--',
                       label='Smoothed retweet reference')
        axs[1, 0].set_xlim(0, zoom_end)
        axs[1, 0].set_ylim(bottom=0)
        axs[1, 0].set_title('(c) Same-dataset toxicity proxy evidence')
        finish_axes(axs[1, 0], 'Hours since calibration window start', 'Normalized proxy level')
        re_total = int(np.sum(re_window))
        add_metric_box(
            axs[1, 0],
            [
                f'Replies in window = {re_total:,}',
                f'corr(RE, RT) on fit window = {re_corr:.4f}',
                f'Selected target = {selected_proxy_label}',
            ],
            x=0.03,
            y=0.42,
        )
        axs[1, 0].legend(loc='upper right', fontsize=LEGEND_FONT_SIZE)
    else:
        axs[1, 0].text(0.5, 0.5, 'No RE data available', transform=axs[1, 0].transAxes,
                       ha='center', va='center', fontsize=12, color='gray')
        axs[1, 0].set_title('(c) Reply-driven toxicity proxy')

    residuals = fitted_norm - empirical_norm
    bar_colors = [FIT_COLOR if r > 0 else '#1565C0' for r in residuals]
    axs[1, 1].bar(t_data, residuals, color=bar_colors, alpha=0.6)
    axs[1, 1].plot(t_data, moving_average(residuals, SMOOTH_WINDOW), color=SMOOTH_COLOR,
                   linewidth=1.8, label=f'{SMOOTH_WINDOW}h residual mean')
    axs[1, 1].axhline(0, color=THRESHOLD_COLOR, linewidth=0.9)
    axs[1, 1].axvline(spike_end, color=REFERENCE_COLOR, linestyle=':', linewidth=1.1, alpha=0.8,
                      label=f'Spike cutoff (h {spike_end})')
    axs[1, 1].axvline(tail_start, color=THRESHOLD_COLOR, linestyle='--', linewidth=1.1, alpha=0.7,
                      label=f'Tail window start (h {tail_start})')
    axs[1, 1].set_title('(d) Residual structure across the full window')
    finish_axes(axs[1, 1], 'Hours since calibration window start', 'Model minus observed volume')
    axs[1, 1].legend(loc='upper right', fontsize=LEGEND_FONT_SIZE)

    fig.suptitle('Spread calibration on the Higgs retweet window', fontsize=SUPTITLE_FONT_SIZE, fontweight='bold', y=1.01)
    fig.tight_layout(pad=2)
    fig.savefig(DIAG_FIGURE_PATH, dpi=300, bbox_inches='tight')
    plt.close(fig)


def make_tau_comparison_figure(selected_proxy: np.ndarray,
                               selected_proxy_label: str,
                               re_window: np.ndarray | None,
                               tau_configs: dict[str, dict]) -> None:
    """Compare the selected proxy against baseline, same-dataset, and external tau setups"""
    apply_plot_style()
    fig, axs = plt.subplots(1, 2, figsize=(14, 5.5))
    t_data = np.arange(len(selected_proxy))

    if re_window is not None:
        re_norm = re_window / (np.max(re_window) + 1e-12)
        axs[0].bar(t_data, re_norm, alpha=0.25, color=PRESSURE_COLOR, label='Observed replies (normalized)')
    axs[0].plot(t_data, selected_proxy, color=FIT_COLOR, linewidth=2.4,
                label=f'Selected proxy ({selected_proxy_label})')
    for cfg in tau_configs.values():
        axs[0].plot(t_data, cfg['tau_curve'], linewidth=2.0, color=cfg['color'], label=cfg['label'])
    axs[0].set_title('(a) Same-dataset proxy versus fitted toxicity curves')
    finish_axes(axs[0], 'Hours since calibration window start', 'Normalized latent toxicity')
    axs[0].set_ylim(bottom=0)
    axs[0].legend(loc='upper right', fontsize=LEGEND_FONT_SIZE)

    scenario_order = list(SCENARIO_ALPHAS.keys())[::-1]
    x = np.arange(len(scenario_order), dtype=float)
    bar_items = list(tau_configs.items())
    if 'external' in tau_configs and 'reference_constrained' in tau_configs:
        bar_items = [(key, cfg) for key, cfg in tau_configs.items() if key != 'reference_constrained']
    width = min(0.18, 0.72 / max(len(bar_items), 1))
    offsets = (np.arange(len(bar_items), dtype=float) - 0.5 * (len(bar_items) - 1)) * width
    for offset, (_key, cfg) in zip(offsets, bar_items):
        tau_vals = [max(0.0, float(cfg['scenario_results'][label]['tau_star'])) for label in scenario_order]
        axs[1].bar(x + offset, tau_vals, width=width, color=cfg['color'], alpha=0.9, label=cfg['label'])
        zero_mask = [val < 1e-8 for val in tau_vals]
        if any(zero_mask):
            zero_x = [xpos + offset for xpos, is_zero in zip(x, zero_mask) if is_zero]
            axs[1].plot(
                zero_x,
                [0.0] * len(zero_x),
                linestyle='None',
                marker='o',
                markersize=4,
                markerfacecolor='white',
                markeredgewidth=1.2,
                markeredgecolor=cfg['color'],
                zorder=5,
            )
    axs[1].set_xticks(x)
    axs[1].set_xticklabels(['Health-First', 'Moderate', 'Engagement-First'])
    axs[1].set_title('(b) Long-run toxicity by policy under each calibration')
    finish_axes(axs[1], 'Policy scenario', r'Equilibrium toxicity $\tau^*$')
    axs[1].set_ylim(bottom=0)
    axs[1].annotate(
        'Health-First is subcritical in all\nsetups, so $\\tau^* \\approx 0$.',
        xy=(x[0], 0.0), xycoords='data',
        xytext=(x[0] - 0.45, 0.028), textcoords='data',
        fontsize=8,
        ha='left',
        bbox=dict(boxstyle='round,pad=0.2', fc='white', alpha=0.8),
        arrowprops=dict(arrowstyle='-', color='gray', linewidth=0.8),
    )
    if 'external' in tau_configs and 'reference_constrained' in tau_configs:
        axs[1].text(
            0.56, 0.96,
            'Ref.-constrained is omitted in panel (b):\nit shares the same long-run $\\tau^*$ as external\n(same $\\phi/\\psi$) and differs only in panel (a).',
            transform=axs[1].transAxes, fontsize=7, ha='center', va='top',
            bbox=dict(boxstyle='round,pad=0.3', fc='lightyellow', alpha=0.85),
        )
    axs[1].legend(fontsize=LEGEND_FONT_SIZE, loc='upper left', bbox_to_anchor=(0.0, 0.86))

    fig.suptitle('Toxicity calibration evidence from Higgs replies and the Jigsaw reference', fontsize=SUPTITLE_FONT_SIZE, fontweight='bold', y=1.02)
    fig.tight_layout(pad=2)
    fig.savefig(TAU_COMPARE_FIGURE_PATH, dpi=300, bbox_inches='tight')
    plt.close(fig)


def make_results_figure(t_scenario, scenario_results,
                        alphas, tau_star, v_star_cont, u_star_cont, alpha_r0):
    """2x3 policy results -- no calibration panel, purely scenario output"""
    apply_plot_style()
    fig, axs = plt.subplots(2, 3, figsize=(18, 10))

    for label, data in scenario_results.items():
        axs[0, 0].plot(t_scenario, data['solution'][:, 1], linewidth=2.0,
                       color=SCENARIO_COLORS[label], label=label)
    axs[0, 0].set_title('(a) Viral volume trajectories by policy')
    finish_axes(axs[0, 0], 'Time since scenario start (hours)', 'Viral volume $V(t)$')
    axs[0, 0].set_xlim(0, SCENARIO_DISPLAY_HOURS)
    axs[0, 0].set_ylim(bottom=0)
    axs[0, 0].legend(loc='upper right', fontsize=LEGEND_FONT_SIZE)

    for label, data in scenario_results.items():
        axs[0, 1].plot(t_scenario, data['solution'][:, 4], linewidth=2.0,
                       color=SCENARIO_COLORS[label], label=label)
    axs[0, 1].set_title('(b) Toxicity trajectories by policy')
    finish_axes(axs[0, 1], 'Time since scenario start (hours)', r'Latent toxicity $\tau(t)$')
    axs[0, 1].set_xlim(0, SCENARIO_DISPLAY_HOURS)
    axs[0, 1].set_ylim(bottom=0)
    axs[0, 1].legend(loc='upper right', fontsize=LEGEND_FONT_SIZE)

    for label, data in scenario_results.items():
        axs[0, 2].plot(t_scenario, data['solution'][:, 5], linewidth=2.0,
                       color=SCENARIO_COLORS[label], label=label)
    axs[0, 2].set_title('(c) User retention trajectories by policy')
    finish_axes(axs[0, 2], 'Time since scenario start (hours)', 'Active users $U(t)$')
    axs[0, 2].set_xlim(0, SCENARIO_DISPLAY_HOURS)
    axs[0, 2].legend(loc='upper right', fontsize=LEGEND_FONT_SIZE)

    axs[1, 0].plot(alphas, tau_star, color=PRESSURE_COLOR, linewidth=2.3)
    axs[1, 0].axvline(alpha_r0, color=THRESHOLD_COLOR, linestyle='--', linewidth=1.5,
                      label=f'R₀=1 at α≈{alpha_r0:.2f}')
    axs[1, 0].set_title('(d) Threshold in long-run toxicity')
    finish_axes(axs[1, 0], r'Amplification $\alpha$', r'Equilibrium toxicity $\tau^*$')
    add_threshold_shading(axs[1, 0], alpha_r0)
    axs[1, 0].legend(loc='upper left', fontsize=LEGEND_FONT_SIZE)

    axs[1, 1].plot(alphas, u_star_cont, color=USER_COLOR, linewidth=2.3)
    axs[1, 1].axvline(alpha_r0, color=THRESHOLD_COLOR, linestyle='--', linewidth=1.5,
                      label=f'R₀=1 at α≈{alpha_r0:.2f}')
    axs[1, 1].set_title('(e) Retained users fall above the threshold')
    finish_axes(axs[1, 1], r'Amplification $\alpha$', 'Equilibrium users $U^*$')
    add_threshold_shading(axs[1, 1], alpha_r0)
    axs[1, 1].legend(loc='upper right', fontsize=LEGEND_FONT_SIZE)

    e_star = alphas * v_star_cont * u_star_cont
    axs[1, 2].plot(alphas, e_star, color=ENGAGEMENT_COLOR, linewidth=2.3)
    axs[1, 2].axvline(alpha_r0, color=THRESHOLD_COLOR, linestyle='--', linewidth=1.5,
                      label=f'R₀=1 at α≈{alpha_r0:.2f}')
    peak_idx = int(np.argmax(e_star))
    axs[1, 2].axvline(alphas[peak_idx], color=ENGAGEMENT_COLOR, linestyle=':', linewidth=1.5,
                      label=f'Peak E* at α≈{alphas[peak_idx]:.2f}')
    axs[1, 2].set_title('(f) Engagement rises while health deteriorates')
    finish_axes(axs[1, 2], r'Amplification $\alpha$', r'Equilibrium engagement $E^* = \alpha V^* U^*$')
    add_threshold_shading(axs[1, 2], alpha_r0)
    axs[1, 2].legend(loc='upper left', fontsize=LEGEND_FONT_SIZE)

    fig.suptitle('Amplification creates a threshold and a virality-health tradeoff', fontsize=SUPTITLE_FONT_SIZE, fontweight='bold', y=1.01)
    fig.tight_layout(pad=2)
    fig.savefig(FIGURE_PATH, dpi=300, bbox_inches='tight')
    plt.close(fig)


def make_phi_sensitivity_figure(phi_grid: np.ndarray,
                                phi_results: dict[str, dict[str, np.ndarray | float]],
                                current_phi: float,
                                external_phi: float | None = None,
                                external_source: str | None = None) -> None:
    """Save a small robustness figure for the toxicity coupling PHI"""
    apply_plot_style()
    fig, axs = plt.subplots(1, 2, figsize=(12.5, 4.8))

    for label, data in phi_results.items():
        axs[0].plot(phi_grid, data['tau_star'], linewidth=2.1, color=SCENARIO_COLORS[label], label=label)
        axs[1].plot(phi_grid, data['U_star'], linewidth=2.1, color=SCENARIO_COLORS[label], label=label)

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
    finish_axes(axs[0], r'Toxicity forcing $\phi$', r'Equilibrium toxicity $\tau^*$')
    axs[0].set_ylim(bottom=0)
    axs[0].legend(fontsize=LEGEND_FONT_SIZE)

    axs[1].set_title('(b) Equilibrium Users $U^*$ vs PHI')
    finish_axes(axs[1], r'Toxicity forcing $\phi$', 'Equilibrium users $U^*$')
    axs[1].legend(fontsize=LEGEND_FONT_SIZE)

    fig.suptitle(r'Policy ordering is robust to the toxicity forcing parameter $\phi$', fontsize=SUPTITLE_FONT_SIZE, fontweight='bold', y=1.02)
    fig.tight_layout(pad=2)
    fig.savefig(PHI_SENS_FIGURE_PATH, dpi=300, bbox_inches='tight')
    plt.close(fig)


def plot_profile(beta_grid, profile_beta, gamma_grid, profile_gamma,
                 beta0_best, gamma0_best) -> None:
    """Save the profile likelihood plots"""
    apply_plot_style()
    fig, (ax1, ax2) = plt.subplots(1, 2, figsize=(12, 4.5))
    beta_excess = profile_beta - float(np.min(profile_beta))
    gamma_excess = profile_gamma - float(np.min(profile_gamma))

    ax1.plot(beta_grid, beta_excess, 'o-', markersize=4, color=FIT_COLOR)
    ax1.axvline(beta0_best, color=THRESHOLD_COLOR, linestyle='--', linewidth=1.2,
                label=f'β₀={beta0_best:.3f}')
    finish_axes(ax1, r'Fixed $\beta_0$', 'Excess weighted loss')
    ax1.set_title('Local Profile Likelihood: β₀')
    ax1.legend(fontsize=LEGEND_FONT_SIZE)

    ax2.plot(gamma_grid, gamma_excess, 'o-', markersize=4, color='#1565C0')
    ax2.axvline(gamma0_best, color=THRESHOLD_COLOR, linestyle='--', linewidth=1.2,
                label=f'γ₀={gamma0_best:.3f}')
    finish_axes(ax2, r'Fixed $\gamma_0$', 'Excess weighted loss')
    ax2.set_title('Local Profile Likelihood: γ₀')
    ax2.legend(fontsize=LEGEND_FONT_SIZE)

    fig.tight_layout()
    fig.savefig(PROFILE_FIGURE_PATH, dpi=300, bbox_inches='tight')
    plt.close(fig)
    print(f'Saved profile likelihood figure: {PROFILE_FIGURE_PATH}')
