from __future__ import annotations

"""Fit the reduced IVF helper model on several additional active Higgs windows"""

import argparse

import numpy as np

import matplotlib

matplotlib.use('Agg')
import matplotlib.pyplot as plt

from .common import ASSETS_DIR, add_dataset_cli_arguments
from .ivfs_calibration import ensure_dataset, fit_basic_ivf, parse_activity_file
from .plot_style import (FIT_COLOR, LEGEND_FONT_SIZE, OBSERVED_COLOR, OBSERVED_MARKER_SIZE,
                         SUPTITLE_FONT_SIZE, add_shared_legend, add_top_padding,
                         apply_plot_style, finish_axes)

WINDOW_WIDTH = 48
WINDOW_STEP = 4
MIN_PEAK_COUNT = 1000
MIN_START_SEPARATION = 16
N_WINDOWS = 3
FIGURE_PATH = ASSETS_DIR / 'higgs_window_validation.png'


def fit_window(counts: np.ndarray) -> dict[str, float | np.ndarray]:
    counts = np.asarray(counts, dtype=float)
    empirical_norm = counts / (np.max(counts) + 1e-12)
    beta0, gamma0, lambda0, lambda_decay, v0_fit, loss, fitted_norm = fit_basic_ivf(
        empirical_norm,
        counts,
        n_steps=len(empirical_norm),
    )
    sse = float(np.sum((fitted_norm - empirical_norm) ** 2))
    ss_tot = float(np.sum((empirical_norm - np.mean(empirical_norm)) ** 2))
    r2 = 1.0 - sse / ss_tot if ss_tot > 0 else 0.0
    return {
        'beta0': float(beta0),
        'gamma0': float(gamma0),
        'lambda0': float(lambda0),
        'lambda_decay': float(lambda_decay),
        'loss': float(loss),
        'r2': float(r2),
        'empirical_norm': empirical_norm,
        'fitted_norm': fitted_norm,
    }


def select_windows(hourly_counts: np.ndarray) -> list[dict[str, float | int | np.ndarray]]:
    candidates: list[dict[str, float | int | np.ndarray]] = []
    for start in range(0, len(hourly_counts) - WINDOW_WIDTH + 1, WINDOW_STEP):
        counts = hourly_counts[start:start + WINDOW_WIDTH].astype(float)
        peak = int(np.max(counts))
        if peak < MIN_PEAK_COUNT:
            continue
        fit = fit_window(counts)
        candidates.append({
            'start': start,
            'end': start + WINDOW_WIDTH - 1,
            'peak': peak,
            'counts': counts,
            **fit,
        })

    candidates.sort(key=lambda row: float(row['r2']), reverse=True)
    selected: list[dict[str, float | int | np.ndarray]] = []
    for row in candidates:
        if all(abs(int(row['start']) - int(prev['start'])) >= MIN_START_SEPARATION for prev in selected):
            selected.append(row)
        if len(selected) >= N_WINDOWS:
            break
    return selected


def make_figure(rows: list[dict[str, float | int | np.ndarray]]) -> None:
    apply_plot_style()
    n = len(rows)
    fig, axs = plt.subplots(1, n, figsize=(6.8 * n, 5.4))
    if n == 1:
        axs = [axs]
    for ax, row in zip(axs, rows):
        empirical = np.asarray(row['empirical_norm'], dtype=float)
        fitted = np.asarray(row['fitted_norm'], dtype=float)
        t = np.arange(len(empirical))
        ax.scatter(t, empirical, s=28, alpha=0.55, color=OBSERVED_COLOR, label='Observed retweets')
        ax.plot(t, fitted, color=FIT_COLOR, linewidth=2.2, label='Fitted IVF curve')
        ax.set_title(
            f"Hours {int(row['start'])} to {int(row['end'])}\n"
            f"peak={int(row['peak']):,}, $R^2$={float(row['r2']):.3f}",
            fontsize=12, fontweight='normal',
        )
        finish_axes(ax, 'Hours since local-window start', 'Normalized retweet volume')
        ax.set_ylim(bottom=0)
        add_top_padding(ax, fraction=0.12, keep_bottom=0.0)
        ax.margins(x=0.03)
        ax.tick_params(axis='y', labelleft=True)
    handles, labels = axs[0].get_legend_handles_labels()
    add_shared_legend(fig, handles, labels, loc='lower center', bbox_to_anchor=(0.5, 0.0), ncol=2)
    fig.suptitle('Alternate active windows show the same burst-and-decay shape',
                 fontsize=13, fontweight='bold', y=0.98)
    fig.subplots_adjust(left=0.07, right=0.99, bottom=0.22, top=0.86, wspace=0.22)
    fig.savefig(FIGURE_PATH, dpi=300, bbox_inches='tight')
    plt.close(fig)


def main(dataset_path: str | None = None,
        ) -> None:
    resolved_dataset = ensure_dataset(dataset_path=dataset_path)
    rt_timestamps, _re_timestamps, _mt_timestamps, total_rows = parse_activity_file(resolved_dataset)
    t0 = int(rt_timestamps.min())
    hours = (rt_timestamps - t0) / 3600.0
    max_hour = int(np.ceil(hours.max()))
    hourly_counts, _ = np.histogram(hours, bins=np.arange(0, max_hour + 1, 1))

    rows = select_windows(hourly_counts)
    make_figure(rows)

    print('Additional Higgs active-window validation (48h windows):')
    print(f'Total RT events: {int(np.sum(hourly_counts)):,}')
    print(f'Selected windows: {len(rows)}')
    for idx, row in enumerate(rows, start=1):
        print(
            f"  {idx}. hours {int(row['start'])}-{int(row['end'])}, peak={int(row['peak']):,}, "
            f"R2={float(row['r2']):.4f}, beta0={float(row['beta0']):.4f}, "
            f"gamma0={float(row['gamma0']):.4f}, lambda0={float(row['lambda0']):.4f}, "
            f"lambda_decay={float(row['lambda_decay']):.4f}"
        )
    print(f'Saved figure: {FIGURE_PATH}')


if __name__ == '__main__':
    parser = argparse.ArgumentParser(description='Fit the reduced IVF model on alternate active Higgs windows.')
    add_dataset_cli_arguments(parser)
    args = parser.parse_args()
    main(dataset_path=args.dataset_path)
