from __future__ import annotations

"""fit the reduced IVF helper model on several additional active Higgs windows"""

import sys
from pathlib import Path

sys.path.insert(0, str(Path(__file__).resolve().parent))

import numpy as np

import matplotlib

matplotlib.use('Agg')
import matplotlib.pyplot as plt

from common import ASSETS_DIR
from ivfs_calibration import ensure_dataset, fit_basic_ivf, parse_activity_file
from ivfs_config import HIGGS_TXT

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
    plt.style.use('ggplot')
    fig, axs = plt.subplots(1, len(rows), figsize=(4.8 * len(rows), 4.2), sharey=True)
    if len(rows) == 1:
        axs = [axs]
    for ax, row in zip(axs, rows):
        empirical = np.asarray(row['empirical_norm'], dtype=float)
        fitted = np.asarray(row['fitted_norm'], dtype=float)
        t = np.arange(len(empirical))
        ax.scatter(t, empirical, s=18, alpha=0.45, color='#616161', label='Empirical RT')
        ax.plot(t, fitted, color='#C62828', linewidth=2.1, label='Fitted IVF')
        ax.set_title(
            f"h {int(row['start'])}–{int(row['end'])}\n"
            f"peak={int(row['peak']):,}, $R^2$={float(row['r2']):.3f}",
            fontsize=10,
        )
        ax.set_xlabel('Hour in local window')
        ax.set_ylim(bottom=0)
    axs[0].set_ylabel('Normalized RT volume')
    axs[0].legend(fontsize=8, loc='upper right')
    fig.suptitle('Additional Higgs Active-Window Validation for the Reduced IVF Fit', fontsize=13, fontweight='bold', y=1.03)
    fig.tight_layout(pad=1.6)
    fig.savefig(FIGURE_PATH, dpi=300, bbox_inches='tight')
    plt.close(fig)


def main() -> None:
    ensure_dataset()
    rt_timestamps, _re_timestamps, _mt_timestamps, total_rows = parse_activity_file(HIGGS_TXT)
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
    main()
