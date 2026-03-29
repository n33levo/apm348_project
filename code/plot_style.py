from __future__ import annotations

import matplotlib as mpl
import matplotlib.pyplot as plt

FONT_FAMILY = 'DejaVu Sans'
BASE_FONT_SIZE = 11
TITLE_FONT_SIZE = 13
SUPTITLE_FONT_SIZE = 14
LEGEND_FONT_SIZE = 10
LINE_WIDTH = 2.3
GRID_ALPHA = 0.22
OBSERVED_MARKER_SIZE = 24
SMALL_MARKER_SIZE = 16

OBSERVED_COLOR = '#4B5563'
SMOOTH_COLOR = '#1F2937'
FIT_COLOR = '#B42318'
BAND_COLOR = '#FCA5A5'
REFERENCE_COLOR = '#475467'
PRESSURE_COLOR = '#7C3AED'
USER_COLOR = '#15803D'
ENGAGEMENT_COLOR = '#C2410C'
THRESHOLD_COLOR = '#111827'
SHADE_COLOR = '#E5E7EB'

SCENARIO_COLORS = {
    'Engagement-First (alpha=0.9)': '#C62828',
    'Moderate (alpha=0.5)': '#1565C0',
    'Health-First (alpha=0.2)': '#2E7D32',
}


def apply_plot_style() -> None:
    plt.style.use('default')
    mpl.rcParams.update({
        'font.family': FONT_FAMILY,
        'font.size': BASE_FONT_SIZE,
        'axes.titlesize': TITLE_FONT_SIZE,
        'axes.titleweight': 'semibold',
        'axes.labelsize': BASE_FONT_SIZE,
        'axes.grid': True,
        'grid.alpha': GRID_ALPHA,
        'grid.linestyle': '-',
        'grid.linewidth': 0.7,
        'axes.facecolor': 'white',
        'figure.facecolor': 'white',
        'savefig.facecolor': 'white',
        'legend.frameon': True,
        'legend.framealpha': 0.95,
        'legend.edgecolor': '#D0D5DD',
        'legend.fontsize': LEGEND_FONT_SIZE,
        'lines.linewidth': LINE_WIDTH,
        'axes.spines.top': False,
        'axes.spines.right': False,
    })


def finish_axes(ax, xlabel: str, ylabel: str) -> None:
    ax.set_xlabel(xlabel)
    ax.set_ylabel(ylabel)
    ax.grid(True, alpha=GRID_ALPHA)


def add_threshold_shading(ax, threshold: float, label: str = 'Subcritical') -> None:
    xmin, xmax = ax.get_xlim()
    ax.axvspan(xmin, threshold, color=SHADE_COLOR, alpha=0.45, zorder=0)
    if xmax > xmin:
        x_text = xmin + 0.45 * (threshold - xmin)
        y_top = ax.get_ylim()[1]
        ax.text(x_text, 0.92 * y_top, label, ha='center', va='top', color='#475467', fontsize=10)


def add_metric_box(ax, lines: list[str], x: float = 0.03, y: float = 0.97) -> None:
    ax.text(
        x,
        y,
        '\n'.join(lines),
        transform=ax.transAxes,
        ha='left',
        va='top',
        fontsize=9,
        bbox={'boxstyle': 'round,pad=0.35', 'facecolor': 'white', 'edgecolor': '#D0D5DD', 'alpha': 0.95},
    )
