from __future__ import annotations

"""Draw the IVFS structure diagram with orthogonal routing for the cross-links"""

import matplotlib

matplotlib.use('Agg')
import matplotlib.pyplot as plt
from matplotlib.patches import FancyArrowPatch, FancyBboxPatch

from .common import ASSETS_DIR, ensure_layout
from .plot_style import SUPTITLE_FONT_SIZE, apply_plot_style


FIGURE_PATH = ASSETS_DIR / 'ivfs_structure_diagram.png'

BOX_W = 1.72
BOX_H = 1.08

BOX_STYLE = {
    'I': ('#EEF2FF', '#4F46E5'),
    'V': ('#FFF7ED', '#C2410C'),
    'F': ('#FAF5FF', '#7C3AED'),
    'S': ('#FFF1F2', '#E11D48'),
    'T': ('#FEFCE8', '#A16207'),
    'U': ('#F0FDF4', '#15803D'),
}

CLR = {
    'spread': '#C2410C',
    'suppress': '#475569',
    'gen': '#A16207',
    'fb': '#7C3AED',
    'user': '#15803D',
    'recruit': '#2563EB',
}

# Box centres
C = {
    'I': (2.5, 5.3),
    'V': (5.5, 5.3),
    'F': (8.5, 5.3),
    'S': (2.5, 2.3),
    'T': (11.2, 5.3),
    'U': (11.2, 2.3),
}


def edge(key, side, frac=0.5):
    cx, cy = C[key]
    if side == 'r':
        return (cx + BOX_W / 2, cy - BOX_H / 2 + frac * BOX_H)
    if side == 'l':
        return (cx - BOX_W / 2, cy - BOX_H / 2 + frac * BOX_H)
    if side == 't':
        return (cx - BOX_W / 2 + frac * BOX_W, cy + BOX_H / 2)
    if side == 'b':
        return (cx - BOX_W / 2 + frac * BOX_W, cy - BOX_H / 2)
    raise ValueError(side)


def draw_box(ax, key, subtitle):
    cx, cy = C[key]
    fc, ec = BOX_STYLE[key]
    ax.add_patch(FancyBboxPatch(
        (cx - BOX_W / 2 + 0.06, cy - BOX_H / 2 - 0.06), BOX_W, BOX_H,
        boxstyle='round,pad=0.03,rounding_size=0.06',
        linewidth=0, facecolor='#94A3B8', alpha=0.16, zorder=1))
    ax.add_patch(FancyBboxPatch(
        (cx - BOX_W / 2, cy - BOX_H / 2), BOX_W, BOX_H,
        boxstyle='round,pad=0.03,rounding_size=0.06',
        linewidth=2.2, edgecolor=ec, facecolor=fc, zorder=3))
    sym = r'\tau' if key == 'T' else key
    ax.text(cx, cy + 0.14, f'${sym}$', ha='center', va='center',
            fontsize=18, fontweight='bold', color=ec, zorder=4)
    ax.text(cx, cy - 0.22, subtitle, ha='center', va='center',
            fontsize=10, color='#64748B', style='italic', zorder=4)


def label(ax, xy, text, color, fs=10):
    ax.text(*xy, text, fontsize=fs, color=color,
            ha='center', va='center', zorder=6,
            bbox=dict(boxstyle='round,pad=0.18', fc='white', ec='none', alpha=0.92))


def straight(ax, start, end, txt, txt_xy, color, lw=2.0, fs=10):
    ax.add_patch(FancyArrowPatch(
        start, end, arrowstyle='-|>', mutation_scale=15,
        linewidth=lw, color=color, zorder=5))
    label(ax, txt_xy, txt, color, fs)


def routed(ax, pts, txt, txt_xy, color, lw=2.0, fs=10):
    """Draw an orthogonal multi-segment arrow with the arrowhead on the last segment"""
    for i in range(len(pts) - 2):
        ax.plot([pts[i][0], pts[i+1][0]], [pts[i][1], pts[i+1][1]],
                color=color, linewidth=lw, solid_capstyle='round', zorder=5)
    ax.add_patch(FancyArrowPatch(
        pts[-2], pts[-1], arrowstyle='-|>', mutation_scale=15,
        linewidth=lw, color=color, zorder=5))
    label(ax, txt_xy, txt, color, fs)


def main():
    ensure_layout()
    apply_plot_style()

    fig, ax = plt.subplots(figsize=(14.5, 8.5))
    fig.patch.set_facecolor('white')
    ax.set_facecolor('white')
    ax.set_xlim(0, 14.0)
    ax.set_ylim(0, 8.5)
    ax.axis('off')

    ax.plot([9.8, 9.8], [0.55, 8.0], color='#CBD5E1', lw=1.3, ls='--', zorder=0)
    ax.text(4.5, 8.10, 'Content lifecycle compartments',
            fontsize=SUPTITLE_FONT_SIZE, fontweight='bold', color='#1E293B', ha='center')
    ax.text(11.2, 8.10, 'Toxicity and user block',
            fontsize=SUPTITLE_FONT_SIZE, fontweight='bold', color='#1E293B', ha='center')

    names = {'I': 'ignored content', 'V': 'viral content', 'F': 'fatigued content',
             'S': 'suppressed content', 'T': 'latent toxicity', 'U': 'active users'}
    for key in C:
        draw_box(ax, key, names[key])

    # Straight arrows

    # I -> V
    straight(ax, edge('I', 'r'), edge('V', 'l'),
             r'$\beta_{\mathrm{eff}}IV$', (4.0, 5.88), CLR['spread'])
    # V -> F
    straight(ax, edge('V', 'r'), edge('F', 'l'),
             r'$\gamma_{\mathrm{eff}}V$', (7.0, 5.88), CLR['spread'])
    # I -> S  (Moderation removes content to suppressed)
    straight(ax, edge('I', 'b'), edge('S', 't'),
             r'$\delta I$', (2.8, 3.85), CLR['suppress'])
    # tau -> U
    straight(ax, edge('T', 'b', 0.65), edge('U', 't', 0.65),
             r'$\lambda_u(1+w\tau)U$', (12.15, 3.85), CLR['user'])
    # Nu -> U  (external inflow)
    straight(ax, (C['U'][0], 0.65), edge('U', 'b'),
             r'$\nu$', (C['U'][0] + 0.50, 1.05), CLR['user'])
    # Tau modulating the V->F fatigue rate
    vf_mid = ((C['V'][0] + BOX_W / 2 + C['F'][0] - BOX_W / 2) / 2,
              C['V'][1] - BOX_H / 2)  # bottom of V->F midpoint
    routed(ax,
           [edge('T', 'l'),
            (edge('T', 'l')[0] - 0.3, edge('T', 'l')[1]),
            (edge('T', 'l')[0] - 0.3, vf_mid[1] - 0.6),
            (vf_mid[0], vf_mid[1] - 0.6),
            vf_mid],
           r'$\gamma_0(1+\eta\tau)$', (9.1, 4.15), CLR['fb'])

    # Routed arrows

    # phi*V :  V -> tau  (going above all boxes)
    routed(ax,
           [edge('V', 't'),
            (C['V'][0], 7.2),
            (C['T'][0], 7.2),
            edge('T', 't')],
           r'$\phi V$', (8.35, 7.45), CLR['gen'])

    # alpha*beta_0*(1+kappa*tau) :  tau -> V  (going below the box row)
    tau_bx = edge('T', 'b', 0.35)[0]
    v_bx = edge('V', 'b', 0.65)[0]
    routed(ax,
           [edge('T', 'b', 0.35),
            (tau_bx, 3.8),
            (v_bx, 3.8),
            edge('V', 'b', 0.65)],
           r'$\alpha\beta_0(1+\kappa\tau)$', (8.35, 3.55), CLR['fb'])

    # rho*U/(1+U) :  U -> I  (simple arc through the middle gap)
    ax.annotate('', xy=edge('I', 'b', 0.85), xytext=edge('U', 'l', 0.5),
                arrowprops=dict(arrowstyle='-|>', color=CLR['recruit'], lw=2.0,
                                connectionstyle='arc3,rad=-0.20'),
                zorder=5)
    label(ax, (6.0, 2.15), r'$\rho U/(1+U)$', CLR['recruit'])

    # Caption
    ax.text(
        7.0, 0.25,
        r'Deterministic compartmental ODE.  '
        r'$\beta_{\mathrm{eff}}=\alpha\beta_0(1+\kappa\tau)$;  '
        r'latent toxicity $\tau$ feeds back into spread, fatigue, and user retention.',
        fontsize=10.5, color='#64748B', ha='center', style='italic')

    fig.tight_layout(pad=0.4)
    fig.savefig(FIGURE_PATH, dpi=300, bbox_inches='tight')
    plt.close(fig)
    print(f'Saved -> {FIGURE_PATH}')


if __name__ == '__main__':
    main()
