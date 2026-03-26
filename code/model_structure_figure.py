from __future__ import annotations

"""IVFS model structure diagram – polished version."""

import sys
from pathlib import Path

sys.path.insert(0, str(Path(__file__).resolve().parent))

import matplotlib
matplotlib.use('Agg')
import matplotlib.pyplot as plt
from matplotlib.patches import FancyArrowPatch, FancyBboxPatch

from common import ASSETS_DIR, ensure_layout

FIGURE_PATH = ASSETS_DIR / 'ivfs_structure_diagram.png'

# ── colour palette ────────────────────────────────────────────────────────────
_BOX = dict(
    I=('#EEF2FF', '#4338CA'),   # indigo  – ignored/latent
    V=('#FFF7ED', '#C2410C'),   # orange  – viral
    F=('#FDF4FF', '#7E22CE'),   # purple  – fatigued
    S=('#FFF1F2', '#BE123C'),   # rose    – suppressed
    T=('#FEFCE8', '#A16207'),   # amber   – pressure
    U=('#F0FDF4', '#15803D'),   # green   – active users
)
_C = dict(
    spread='#C2410C',     # content spread   (orange)
    suppress='#475569',   # suppression       (slate)
    gen='#A16207',        # V→τ generation    (amber)
    fb='#7C3AED',         # τ feedback        (violet)
    user='#15803D',       # user dynamics     (green)
    recruit='#1D4ED8',    # U recruitment     (blue)
)

# box geometry
_W, _H = 1.68, 1.08


def _box_shadow(ax, cx, cy):
    """draw a soft drop-shadow behind a box."""
    off = 0.07
    ax.add_patch(FancyBboxPatch(
        (cx - _W / 2 + off, cy - _H / 2 - off), _W, _H,
        boxstyle='round,pad=0.03,rounding_size=0.07',
        linewidth=0, facecolor='#94A3B8', alpha=0.18, zorder=2,
    ))


def draw_box(ax, cx, cy, sym, name, fc, ec):
    """draw a labelled compartment box centred at (cx, cy)."""
    _box_shadow(ax, cx, cy)
    ax.add_patch(FancyBboxPatch(
        (cx - _W / 2, cy - _H / 2), _W, _H,
        boxstyle='round,pad=0.03,rounding_size=0.07',
        linewidth=2.2, edgecolor=ec, facecolor=fc, zorder=3,
    ))
    ax.text(cx, cy + 0.15, f'${sym}$',
            ha='center', va='center', fontsize=15, fontweight='bold', color=ec, zorder=4)
    ax.text(cx, cy - 0.2, name,
            ha='center', va='center', fontsize=9, color='#475569', fontstyle='italic', zorder=4)


def arrow(ax, start, end, label, label_xy, color, curve=0.0, lw=1.85, fs=9.0):
    """labelled curved arrow with a white pill background on the label."""
    ax.add_patch(FancyArrowPatch(
        start, end,
        arrowstyle='-|>', mutation_scale=14,
        linewidth=lw, color=color,
        connectionstyle=f'arc3,rad={curve}',
        zorder=5,
    ))
    ax.text(*label_xy, label,
            ha='center', va='center', fontsize=fs, color=color, zorder=6,
            bbox=dict(boxstyle='round,pad=0.25', fc='white', ec='none', alpha=0.92))


def main() -> None:
    ensure_layout()

    fig, ax = plt.subplots(figsize=(14.0, 7.6))
    fig.patch.set_facecolor('white')
    ax.set_facecolor('white')
    ax.set_xlim(0, 14.0)
    ax.set_ylim(0, 7.6)
    ax.axis('off')

    # ── section divider ───────────────────────────────────────────────────────
    ax.plot([7.6, 7.6], [0.7, 7.15], color='#CBD5E1', lw=1.4, ls='--', zorder=1)

    # section headers (with a subtle background pill)
    for txt, cx in [('Content lifecycle compartments', 3.5),
                    ('Pressure and user block', 10.9)]:
        ax.text(cx, 7.22, txt,
                ha='center', va='center', fontsize=11.5, fontweight='bold', color='#1E293B',
                bbox=dict(boxstyle='round,pad=0.35', fc='#F8FAFC', ec='#CBD5E1', lw=1.0))

    # ── boxes ─────────────────────────────────────────────────────────────────
    # cx, cy, symbol, name,  face, edge
    _BOXES = [
        (1.30, 5.2,  'I',       'ignored / latent', *_BOX['I']),
        (3.45, 5.2,  'V',       'viral',             *_BOX['V']),
        (5.70, 5.2,  'F',       'fatigued',          *_BOX['F']),
        (3.45, 2.95, 'S',       'suppressed',        *_BOX['S']),
        (10.0, 5.2,  r'\tau',   'latent pressure',   *_BOX['T']),
        (10.0, 2.95, 'U',       'active users',      *_BOX['U']),
    ]
    for cx, cy, sym, name, fc, ec in _BOXES:
        draw_box(ax, cx, cy, sym, name, fc, ec)

    # ── derived edge coordinates ──────────────────────────────────────────────
    # (using box cx ± _W/2, cy ± _H/2)
    I_r  = (1.30 + _W/2, 5.2)
    V_l  = (3.45 - _W/2, 5.2)
    V_r  = (3.45 + _W/2, 5.2)
    V_b  = (3.45, 5.2 - _H/2)
    F_l  = (5.70 - _W/2, 5.2)
    F_r  = (5.70 + _W/2, 5.2)
    S_t  = (3.45, 2.95 + _H/2)
    T_l  = (10.0 - _W/2, 5.2)
    T_r  = (10.0 + _W/2, 5.2)
    T_b  = (10.0, 5.2 - _H/2)
    U_t  = (10.0, 2.95 + _H/2)
    U_b  = (10.0, 2.95 - _H/2)
    U_l  = (10.0 - _W/2, 2.95)

    # ── content-spread arrows (orange) ────────────────────────────────────────
    arrow(ax, I_r, V_l,
          r'$\beta_{\rm eff}\,IV$', (2.375, 5.74), _C['spread'])
    arrow(ax, V_r, F_l,
          r'$\gamma_{\rm eff}\,V$', (4.575, 5.74), _C['spread'])

    # ── suppression (slate, downward) ─────────────────────────────────────────
    arrow(ax, V_b, S_t,
          r'$\delta I$', (3.92, 4.07), _C['suppress'])

    # ── pressure generation  V→τ  (amber, above) ─────────────────────────────
    arrow(ax, (V_r[0], 5.42), (T_l[0], 5.42),
          r'$\phi V$', (6.72, 5.82), _C['gen'], lw=1.75)

    # ── τ feedback → V  αβ₀  (violet, below) ─────────────────────────────────
    arrow(ax, (T_l[0], 4.98), (V_r[0], 4.98),
          r'$\alpha\beta_0(1+\kappa\tau)$', (6.72, 4.55), _C['fb'], lw=1.75)

    # ── τ feedback → F  γ₀  (violet, mid) ────────────────────────────────────
    arrow(ax, (T_l[0], 5.18), (F_r[0], 5.18),
          r'$\gamma_0(1+\eta\tau)$', (8.0, 5.62), _C['fb'], lw=1.75)

    # ── user dynamics (green) ─────────────────────────────────────────────────
    # τ → U  (churn driven by pressure, downward on right side)
    arrow(ax, (T_r[0] - 0.12, T_b[1]), (T_r[0] - 0.12, U_t[1]),
          r'$\lambda_u(1+w\tau)U$', (11.18, 4.07), _C['user'])
    # ν inflow into U (upward stub from below, left side of column)
    arrow(ax, (T_l[0] + 0.28, 2.0), (T_l[0] + 0.28, U_b[1]),
          r'$\nu$', (9.36, 2.12), _C['user'])

    # ── recruitment  U→V  (blue, long curve below S) ──────────────────────────
    arrow(ax, (U_l[0], 2.78), (V_l[0], 4.95),
          r'$\rho U/(1+U)$', (5.5, 1.52), _C['recruit'], curve=0.28, lw=1.75)

    # ── caption ───────────────────────────────────────────────────────────────
    ax.text(
        7.0, 0.52,
        (r'Deterministic compartmental ODE.  '
         r'$\beta_{\rm eff}=\alpha\beta_0(1\!+\!\kappa\tau)$;  '
         r'latent pressure $\tau$ feeds back into spread, fatigue, and user retention.'),
        ha='center', va='center', fontsize=9.5, color='#64748B', fontstyle='italic',
    )

    fig.tight_layout(pad=0.4)
    fig.savefig(FIGURE_PATH, dpi=300, bbox_inches='tight')
    plt.close(fig)
    print(f'Saved → {FIGURE_PATH}')


if __name__ == '__main__':
    main()
