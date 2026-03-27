"""
Generate supplementary figures for early report sections:
  1. Higgs raw-data overview (Background, Section 1)
  2. R_0 sensitivity bar chart  (Analysis, Section 6)
  3. Bifurcation schematic       (Analysis, Section 6)
"""
from __future__ import annotations

import numpy as np
import matplotlib
matplotlib.use('Agg')
import matplotlib.pyplot as plt

from common import ASSETS_DIR, HIGGS_TXT, ensure_layout
from ivfs_config import (
    KAPPA, ETA, PHI, PSI, RHO, LAMBDA_U, NU, MU_C, DELTA, W,
    SMOOTH_WINDOW,
)

ensure_layout()

# ── helper: load Higgs activity data ──────────────────────────
def _load_higgs_activity():
    """Return hourly RT, MT, RE counts over the full 168-hour window."""
    timestamps = []
    kinds = []
    with open(HIGGS_TXT, 'r') as f:
        for line in f:
            parts = line.strip().split()
            if len(parts) >= 4:
                timestamps.append(int(parts[2]))
                kinds.append(parts[3])
    
    ts = np.array(timestamps)
    t0 = ts.min()
    hours = (ts - t0) / 3600.0
    max_h = int(np.ceil(hours.max())) + 1

    rt = np.zeros(max_h)
    re = np.zeros(max_h)
    mt = np.zeros(max_h)

    for h_val, k in zip(hours.astype(int), kinds):
        if h_val < max_h:
            if   k == 'RT': rt[h_val] += 1
            elif k == 'RE': re[h_val] += 1
            elif k == 'MT': mt[h_val] += 1
    return np.arange(max_h), rt, re, mt


# ━━━━ Figure 1: Higgs Raw-Data Overview ━━━━━━━━━━━━━━━━━━━━━━
def make_higgs_overview():
    hrs, rt, re, mt = _load_higgs_activity()

    plt.style.use('ggplot')
    fig, axs = plt.subplots(3, 1, figsize=(11, 7.5), sharex=True)

    # smooth
    kern = np.ones(SMOOTH_WINDOW) / SMOOTH_WINDOW
    rt_s = np.convolve(rt, kern, 'same')
    re_s = np.convolve(re, kern, 'same')
    mt_s = np.convolve(mt, kern, 'same')

    axs[0].fill_between(hrs, rt_s, alpha=0.35, color='#1f77b4')
    axs[0].plot(hrs, rt_s, color='#1f77b4', linewidth=1.2)
    axs[0].set_ylabel('Retweets / hour')
    axs[0].set_title('(a) Retweet Activity')
    axs[0].annotate('Higgs boson\nannouncement',
                    xy=(hrs[int(np.argmax(rt_s))], rt_s.max()),
                    xytext=(hrs[int(np.argmax(rt_s))] + 18, rt_s.max() * 0.75),
                    arrowprops=dict(arrowstyle='->', color='black'),
                    fontsize=9, ha='left')

    axs[1].fill_between(hrs, re_s, alpha=0.35, color='#ff7f0e')
    axs[1].plot(hrs, re_s, color='#ff7f0e', linewidth=1.2)
    axs[1].set_ylabel('Replies / hour')
    axs[1].set_title('(b) Reply Activity (Proxy for Discussion Pressure)')

    axs[2].fill_between(hrs, mt_s, alpha=0.35, color='#2ca02c')
    axs[2].plot(hrs, mt_s, color='#2ca02c', linewidth=1.2)
    axs[2].set_ylabel('Mentions / hour')
    axs[2].set_title('(c) Mention Activity')
    axs[2].set_xlabel('Hours since start of observation window')

    fig.suptitle('Higgs Twitter Dataset — Activity Overview (168 Hours)',
                 fontsize=13, fontweight='bold', y=1.01)
    fig.tight_layout(pad=1.5)
    out = ASSETS_DIR / 'higgs_overview.png'
    fig.savefig(out, dpi=300, bbox_inches='tight')
    plt.close(fig)
    print(f'[extra_figures] saved {out}')


# ━━━━ Figure 2: R₀ Sensitivity Bar Chart ━━━━━━━━━━━━━━━━━━━━
def make_sensitivity_bar():
    """Bar chart of S(R0, p) for key parameters."""
    # Analytical sensitivity of R0
    gamma0_cal = 0.3095       # calibrated value from report
    beta0_cal  = 0.3184

    s_alpha  = 1.0
    s_beta0  = 1.0
    s_gamma0 = -gamma0_cal / (gamma0_cal + MU_C)
    # S(R0, mu_c) from report derivation  (mu_c enters denominator AND I_DFE)
    # Approximate numerically
    s_mu_c   = -0.032  # small, from the report

    # Numerical sensitivities for V*, tau*, U* at alpha=0.5
    params  = ['α', 'β₀', 'γ₀', 'κ', 'η', 'ϕ', 'ψ', 'w']
    s_Vstar = [1.67,  1.67, -2.59,  0.05, -0.03,  0.0,   0.0,  -0.02]
    s_tstar = [1.67,  1.67, -2.59,  0.05, -0.03,  1.0,  -1.0,  -0.02]
    s_Ustar = [-0.469, -0.469, 0.726, -0.014, 0.008, -0.281, 0.281, -0.275]

    plt.style.use('ggplot')
    fig, axs = plt.subplots(1, 3, figsize=(14, 5))
    x = np.arange(len(params))
    width = 0.55

    colors_V = ['#d62728' if v < 0 else '#2ca02c' for v in s_Vstar]
    colors_t = ['#d62728' if v < 0 else '#2ca02c' for v in s_tstar]
    colors_U = ['#d62728' if v < 0 else '#2ca02c' for v in s_Ustar]

    axs[0].bar(x, s_Vstar, width, color=colors_V, edgecolor='black', linewidth=0.5)
    axs[0].set_xticks(x)
    axs[0].set_xticklabels(params, fontsize=10)
    axs[0].set_ylabel('$S(V^*, p)$')
    axs[0].set_title('(a) Sensitivity of Equilibrium Viral Volume $V^*$')
    axs[0].axhline(0, color='black', linewidth=0.5)

    axs[1].bar(x, s_tstar, width, color=colors_t, edgecolor='black', linewidth=0.5)
    axs[1].set_xticks(x)
    axs[1].set_xticklabels(params, fontsize=10)
    axs[1].set_ylabel(r'$S(\tau^*, p)$')
    axs[1].set_title(r'(b) Sensitivity of Equilibrium Pressure $\tau^*$')
    axs[1].axhline(0, color='black', linewidth=0.5)

    axs[2].bar(x, s_Ustar, width, color=colors_U, edgecolor='black', linewidth=0.5)
    axs[2].set_xticks(x)
    axs[2].set_xticklabels(params, fontsize=10)
    axs[2].set_ylabel('$S(U^*, p)$')
    axs[2].set_title('(c) Sensitivity of Equilibrium Users $U^*$')
    axs[2].axhline(0, color='black', linewidth=0.5)

    fig.suptitle('Normalized Sensitivity Indices at the Moderate Scenario (α = 0.5)',
                 fontsize=13, fontweight='bold', y=1.01)
    fig.tight_layout(pad=1.5)
    out = ASSETS_DIR / 'sensitivity_bars.png'
    fig.savefig(out, dpi=300, bbox_inches='tight')
    plt.close(fig)
    print(f'[extra_figures] saved {out}')


# ━━━━ Figure 3: R₀ vs α bifurcation diagram ━━━━━━━━━━━━━━━━━
def make_r0_bifurcation():
    """Show R0 and V* as functions of alpha, marking the critical threshold."""
    gamma0 = 0.3095
    beta0  = 0.3184

    U_dfe = NU / LAMBDA_U
    I_dfe = (RHO * U_dfe) / ((1.0 + U_dfe) * (DELTA + MU_C))

    alphas = np.linspace(0, 1.2, 500)
    R0 = alphas * beta0 * I_dfe / (gamma0 + MU_C)
    alpha_crit = (gamma0 + MU_C) / (beta0 * I_dfe)

    # Approximate V* above threshold via steady-state numerics
    from ivfs_dynamics import full_ivfs_ode
    from scipy.integrate import odeint

    V_star = np.zeros_like(alphas)
    t_long = np.linspace(0, 8000, 4000)
    for i, a in enumerate(alphas):
        if a <= alpha_crit:
            V_star[i] = 0.0
        else:
            y0 = [I_dfe * 0.95, 0.01, 0.0, 0.0, 0.0, U_dfe]
            sol = odeint(full_ivfs_ode, y0, t_long, args=(a, beta0, gamma0))
            V_star[i] = sol[-1, 1]

    plt.style.use('ggplot')
    fig, (ax1, ax2) = plt.subplots(1, 2, figsize=(12, 4.5))

    ax1.plot(alphas, R0, color='#1f77b4', linewidth=2)
    ax1.axhline(1, color='gray', linestyle='--', linewidth=1, label='$\\mathcal{R}_0 = 1$')
    ax1.axvline(alpha_crit, color='#d62728', linestyle=':', linewidth=1.5,
                label=f'$\\alpha_{{\\mathrm{{crit}}}} \\approx {alpha_crit:.4f}$')
    ax1.set_xlabel('Amplification $\\alpha$')
    ax1.set_ylabel('$\\mathcal{R}_0$')
    ax1.set_title('(a) Basic Reproduction Number vs Amplification')
    ax1.legend(fontsize=9)
    ax1.set_xlim(0, 1.2)

    ax2.plot(alphas, V_star, color='#ff7f0e', linewidth=2)
    ax2.axvline(alpha_crit, color='#d62728', linestyle=':', linewidth=1.5,
                label=f'$\\alpha_{{\\mathrm{{crit}}}} \\approx {alpha_crit:.4f}$')
    ax2.set_xlabel('Amplification $\\alpha$')
    ax2.set_ylabel('$V^*$')
    ax2.set_title('(b) Equilibrium Viral Volume vs Amplification')
    ax2.legend(fontsize=9)
    ax2.set_xlim(0, 1.2)
    ax2.fill_between(alphas, 0, V_star, alpha=0.2, color='#ff7f0e')
    ax2.annotate('DFE stable\n($V^* = 0$)',
                 xy=(alpha_crit * 0.5, 0), xytext=(alpha_crit * 0.4, max(V_star) * 0.3),
                 fontsize=9, ha='center',
                 arrowprops=dict(arrowstyle='->', color='black'))
    ax2.annotate('Endemic\nequilibrium',
                 xy=(alpha_crit * 1.8, V_star[int(len(alphas) * 0.6)]),
                 fontsize=9, ha='center')

    fig.suptitle('Transcritical Bifurcation: Amplification Threshold for Self-Sustaining Virality',
                 fontsize=13, fontweight='bold', y=1.01)
    fig.tight_layout(pad=1.5)
    out = ASSETS_DIR / 'r0_bifurcation.png'
    fig.savefig(out, dpi=300, bbox_inches='tight')
    plt.close(fig)
    print(f'[extra_figures] saved {out}')


if __name__ == '__main__':
    make_higgs_overview()
    make_sensitivity_bar()
    make_r0_bifurcation()
    print('[extra_figures] all done')
