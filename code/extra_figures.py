"""
Generate supplementary figures for the early report sections:
  1. Higgs raw-data overview (background, section 1)
  2. Sensitivity bar chart (analysis, section 6)
  3. Bifurcation schematic (analysis, section 6)
"""
from __future__ import annotations

import numpy as np
import matplotlib
matplotlib.use('Agg')
import matplotlib.pyplot as plt

from .common import ASSETS_DIR, ensure_layout
from .equilibrium_analysis import positive_equilibrium
from .ivfs_calibration import ensure_dataset, parse_activity_file
from .ivfs_config import DELTA, FITTED_BETA0, FITTED_GAMMA0, LAMBDA_U, MU_C, NU, RHO, SMOOTH_WINDOW
from .ivfs_dynamics import run_sensitivity
from .plot_style import (ENGAGEMENT_COLOR, FIT_COLOR, PRESSURE_COLOR, REFERENCE_COLOR, SUPTITLE_FONT_SIZE,
                         THRESHOLD_COLOR, USER_COLOR, add_threshold_shading,
                         add_top_padding, apply_plot_style, finish_axes)

ensure_layout()

# Load Higgs activity data
def _load_higgs_activity():
    """Return hourly RT, MT, RE counts over the full 168-hour window"""
    dataset_path = ensure_dataset()
    rt_timestamps, re_timestamps, mt_timestamps, _ = parse_activity_file(dataset_path)
    t0 = int(min(rt_timestamps.min(), re_timestamps.min(initial=rt_timestamps.min()), mt_timestamps.min(initial=rt_timestamps.min())))
    rt_hours = ((rt_timestamps - t0) / 3600.0).astype(int)
    re_hours = ((re_timestamps - t0) / 3600.0).astype(int)
    mt_hours = ((mt_timestamps - t0) / 3600.0).astype(int)
    max_h = int(max(rt_hours.max(initial=0), re_hours.max(initial=0), mt_hours.max(initial=0))) + 1

    rt = np.zeros(max_h)
    re = np.zeros(max_h)
    mt = np.zeros(max_h)

    for h_val in rt_hours:
        if h_val < max_h:
            rt[h_val] += 1
    for h_val in re_hours:
        if h_val < max_h:
            re[h_val] += 1
    for h_val in mt_hours:
        if h_val < max_h:
            mt[h_val] += 1
    return np.arange(max_h), rt, re, mt


# Figure 1: Higgs raw-data overview
def make_higgs_overview():
    hrs, rt, re, mt = _load_higgs_activity()

    apply_plot_style()
    fig, axs = plt.subplots(3, 1, figsize=(12.8, 8.8), sharex=True)

    # smooth
    kern = np.ones(SMOOTH_WINDOW) / SMOOTH_WINDOW
    rt_s = np.convolve(rt, kern, 'same')
    re_s = np.convolve(re, kern, 'same')
    mt_s = np.convolve(mt, kern, 'same')

    axs[0].fill_between(hrs, rt_s, alpha=0.35, color=FIT_COLOR)
    axs[0].plot(hrs, rt_s, color=FIT_COLOR, linewidth=1.6)
    axs[0].set_ylabel('Retweets / hour')
    axs[0].set_title('(a) Retweet Activity')
    add_top_padding(axs[0], fraction=0.18, keep_bottom=0.0)
    axs[0].annotate('Higgs boson\nannouncement',
                    xy=(hrs[int(np.argmax(rt_s))], rt_s.max()),
                    xytext=(hrs[int(np.argmax(rt_s))] + 18, rt_s.max() * 0.75),
                    arrowprops=dict(arrowstyle='->', color='black'),
                    fontsize=9, ha='left')

    axs[1].fill_between(hrs, re_s, alpha=0.35, color=PRESSURE_COLOR)
    axs[1].plot(hrs, re_s, color=PRESSURE_COLOR, linewidth=1.6)
    axs[1].set_ylabel('Replies / hour')
    axs[1].set_title('(b) Reply Activity (Proxy for Toxicity)')
    add_top_padding(axs[1], fraction=0.12, keep_bottom=0.0)

    axs[2].fill_between(hrs, mt_s, alpha=0.35, color=USER_COLOR)
    axs[2].plot(hrs, mt_s, color=USER_COLOR, linewidth=1.6)
    axs[2].set_ylabel('Mentions / hour')
    axs[2].set_title('(c) Mention Activity')
    axs[2].set_xlabel('Hours since start of observation window')
    add_top_padding(axs[2], fraction=0.12, keep_bottom=0.0)

    fig.suptitle('Higgs activity overview at one-hour resolution',
                 fontsize=SUPTITLE_FONT_SIZE, fontweight='bold', y=1.01)
    fig.tight_layout(pad=1.9, h_pad=1.6)
    out = ASSETS_DIR / 'higgs_overview.png'
    fig.savefig(out, dpi=300, bbox_inches='tight')
    plt.close(fig)
    print(f'[extra_figures] saved {out}')


# Figure 2: sensitivity bar chart
def make_sensitivity_bar():
    """Plot S(R0, p) bar chart for the key parameters"""
    _v_base, _tau_base, _u_base, sens_rows = run_sensitivity(FITTED_BETA0, FITTED_GAMMA0)
    order = ['alpha', 'beta0', 'gamma0', 'kappa', 'eta', 'phi', 'psi', 'w']
    labels = ['α', 'β₀', 'γ₀', 'κ', 'η', 'ϕ', 'ψ', 'w']
    sens_map = {name: (s_v, s_tau, s_u) for name, s_v, s_tau, s_u in sens_rows}
    s_Vstar = [sens_map[name][0] for name in order]
    s_tstar = [sens_map[name][1] for name in order]
    s_Ustar = [sens_map[name][2] for name in order]

    apply_plot_style()
    fig, axs = plt.subplots(1, 3, figsize=(15.8, 5.8))
    x = np.arange(len(labels))
    width = 0.55

    colors_V = ['#d62728' if v < 0 else '#2ca02c' for v in s_Vstar]
    colors_t = ['#d62728' if v < 0 else '#2ca02c' for v in s_tstar]
    colors_U = ['#d62728' if v < 0 else '#2ca02c' for v in s_Ustar]

    axs[0].bar(x, s_Vstar, width, color=colors_V, edgecolor='black', linewidth=0.5)
    axs[0].set_xticks(x)
    axs[0].set_xticklabels(labels, fontsize=10)
    axs[0].set_ylabel('$S(V^*, p)$')
    axs[0].set_title('(a) Sensitivity of Equilibrium Viral Volume $V^*$')
    axs[0].axhline(0, color='black', linewidth=0.5)

    axs[1].bar(x, s_tstar, width, color=colors_t, edgecolor='black', linewidth=0.5)
    axs[1].set_xticks(x)
    axs[1].set_xticklabels(labels, fontsize=10)
    axs[1].set_ylabel(r'$S(\tau^*, p)$')
    axs[1].set_title(r'(b) Sensitivity of Equilibrium Pressure $\tau^*$')
    axs[1].axhline(0, color='black', linewidth=0.5)

    axs[2].bar(x, s_Ustar, width, color=colors_U, edgecolor='black', linewidth=0.5)
    axs[2].set_xticks(x)
    axs[2].set_xticklabels(labels, fontsize=10)
    axs[2].set_ylabel('$S(U^*, p)$')
    axs[2].set_title('(c) Sensitivity of Equilibrium Users $U^*$')
    axs[2].axhline(0, color='black', linewidth=0.5)

    fig.suptitle('Moderate-policy sensitivity ranking at the equilibrium point',
                 fontsize=SUPTITLE_FONT_SIZE, fontweight='bold', y=1.01)
    fig.tight_layout(pad=1.9, w_pad=2.0)
    out = ASSETS_DIR / 'sensitivity_bars.png'
    fig.savefig(out, dpi=300, bbox_inches='tight')
    plt.close(fig)
    print(f'[extra_figures] saved {out}')


# Figure 3: R0 vs alpha bifurcation diagram
def make_r0_bifurcation():
    """Show R0 and V* as functions of alpha, marking the critical threshold"""
    gamma0 = FITTED_GAMMA0
    beta0 = FITTED_BETA0

    U_dfe = NU / LAMBDA_U
    I_dfe = (RHO * U_dfe) / ((1.0 + U_dfe) * (DELTA + MU_C))

    alphas = np.linspace(0, 1.2, 500)
    R0 = alphas * beta0 * I_dfe / (gamma0 + MU_C)
    alpha_crit = (gamma0 + MU_C) / (beta0 * I_dfe)

    # Use the positive-equilibrium calculation directly instead of a long ODE sweep.
    V_star = np.zeros_like(alphas)
    for i, a in enumerate(alphas):
        if a <= 0.0:
            V_star[i] = 0.0
            continue
        eq_state = positive_equilibrium(float(a), beta0, gamma0)
        V_star[i] = 0.0 if eq_state is None else float(eq_state[1])

    apply_plot_style()
    fig, (ax1, ax2) = plt.subplots(1, 2, figsize=(16.4, 7.2))

    ax1.plot(alphas, R0, color=FIT_COLOR, linewidth=2.2)
    ax1.axhline(1, color=REFERENCE_COLOR, linestyle='--', linewidth=1)
    ax1.axvline(alpha_crit, color=THRESHOLD_COLOR, linestyle=':', linewidth=1.5,
                )
    finish_axes(ax1, 'Amplification $\\alpha$', '$\\mathcal{R}_0$')
    ax1.set_title('(a) Basic Reproduction Number vs Amplification')
    ax1.set_xlim(0, 1.2)
    add_threshold_shading(ax1, alpha_crit)
    add_top_padding(ax1, fraction=0.18, keep_bottom=0.0)
    ax1.text(1.17, 1.04, '$\\mathcal{R}_0 = 1$', color=REFERENCE_COLOR,
             fontsize=10, ha='right', va='bottom')
    ax1.text(alpha_crit + 0.015, 4.45, f'$\\alpha_{{\\mathrm{{crit}}}} \\approx {alpha_crit:.4f}$',
             color=THRESHOLD_COLOR, fontsize=10, ha='left', va='top', rotation=0,
             bbox=dict(boxstyle='round,pad=0.18', fc='white', alpha=0.92, ec='none'))

    ax2.plot(alphas, V_star, color=ENGAGEMENT_COLOR, linewidth=2.2)
    ax2.axvline(alpha_crit, color=THRESHOLD_COLOR, linestyle=':', linewidth=1.5)
    finish_axes(ax2, 'Amplification $\\alpha$', '$V^*$')
    ax2.set_title('(b) Equilibrium Viral Volume vs Amplification')
    ax2.set_xlim(0, 1.2)
    ax2.fill_between(alphas, 0, V_star, alpha=0.2, color=ENGAGEMENT_COLOR)
    add_threshold_shading(ax2, alpha_crit)
    add_top_padding(ax2, fraction=0.18, keep_bottom=0.0)
    ax2.text(alpha_crit + 0.02, 0.158, f'$\\alpha_{{\\mathrm{{crit}}}} \\approx {alpha_crit:.4f}$',
             color=THRESHOLD_COLOR, fontsize=10, ha='left', va='top',
             bbox=dict(boxstyle='round,pad=0.18', fc='white', alpha=0.92, ec='none'))
    ax2.text(0.12, 0.64, 'DFE stable\n($V^* = 0$)', transform=ax2.transAxes,
             fontsize=11, ha='center', va='center',
             bbox=dict(boxstyle='round,pad=0.28', fc='white', alpha=0.92))
    # Arrow annotation pointing from label box to the equilibrium curve
    arrow_alpha = 0.78
    arrow_v = float(np.interp(arrow_alpha, alphas, V_star))
    ax2.annotate(
        'Positive-equilibrium\nbranch',
        xy=(arrow_alpha, arrow_v),
        xytext=(0.95, 0.145),
        fontsize=11,
        ha='center', va='top',
        bbox=dict(boxstyle='round,pad=0.28', fc='white', alpha=0.92),
        arrowprops=dict(arrowstyle='->', color=ENGAGEMENT_COLOR, lw=1.4),
    )

    fig.suptitle('The calibrated model crosses a sharp amplification threshold',
                 fontsize=SUPTITLE_FONT_SIZE, fontweight='bold', y=1.01)
    fig.tight_layout(rect=(0, 0, 1, 0.97), pad=2.2, w_pad=2.8)
    out = ASSETS_DIR / 'r0_bifurcation.png'
    fig.savefig(out, dpi=300, bbox_inches='tight')
    plt.close(fig)
    print(f'[extra_figures] saved {out}')


if __name__ == '__main__':
    make_higgs_overview()
    make_sensitivity_bar()
    make_r0_bifurcation()
    print('[extra_figures] all done')
