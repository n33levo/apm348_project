from __future__ import annotations

import matplotlib as mpl
import matplotlib.pyplot as plt

FONT_FAMILY = 'DejaVu Sans'
BASE_FONT_SIZE = 12
TITLE_FONT_SIZE = 15
SUPTITLE_FONT_SIZE = 17
LEGEND_FONT_SIZE = 11
LINE_WIDTH = 2.5
GRID_ALPHA = 0.22
OBSERVED_MARKER_SIZE = 30
SMALL_MARKER_SIZE = 18

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
        'xtick.labelsize': BASE_FONT_SIZE,
        'ytick.labelsize': BASE_FONT_SIZE,
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


def add_top_padding(ax, fraction: float = 0.16, keep_bottom: float | None = None) -> None:
    ymin, ymax = ax.get_ylim()
    if keep_bottom is not None:
        ymin = keep_bottom
    span = ymax - ymin
    if span <= 0:
        span = max(abs(ymax), 1.0)
    ax.set_ylim(ymin, ymax + fraction * span)


def add_symmetric_padding(ax, fraction: float = 0.12) -> None:
    ymin, ymax = ax.get_ylim()
    span = ymax - ymin
    if span <= 0:
        span = max(abs(ymax), 1.0)
    pad = fraction * span
    ax.set_ylim(ymin - pad, ymax + pad)


def add_shared_legend(fig, handles, labels,
                      loc: str = 'upper center',
                      bbox_to_anchor: tuple[float, float] = (0.5, 0.99),
                      ncol: int | None = None,
                      fontsize: float | None = None):
    unique_handles = []
    unique_labels = []
    seen: set[str] = set()
    for handle, label in zip(handles, labels):
        if not label or label.startswith('_') or label in seen:
            continue
        seen.add(label)
        unique_handles.append(handle)
        unique_labels.append(label)
    if not unique_handles:
        return None
    if ncol is None:
        ncol = min(4, len(unique_handles))
    return fig.legend(
        unique_handles,
        unique_labels,
        loc=loc,
        bbox_to_anchor=bbox_to_anchor,
        ncol=max(1, ncol),
        fontsize=LEGEND_FONT_SIZE if fontsize is None else fontsize,
        frameon=True,
        framealpha=0.95,
        edgecolor='#D0D5DD',
    )


def add_metric_box(ax, lines: list[str], x: float = 0.03, y: float = 0.97,
                   va: str = 'top', fontsize: float = 9) -> None:
    ax.text(
        x,
        y,
        '\n'.join(lines),
        transform=ax.transAxes,
        ha='left',
        va=va,
        fontsize=fontsize,
        bbox={'boxstyle': 'round,pad=0.35', 'facecolor': 'white', 'edgecolor': '#D0D5DD', 'alpha': 0.95},
    )
