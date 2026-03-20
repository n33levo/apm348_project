from __future__ import annotations

"""Look for a Jigsaw-style or Ruddit-style toxicity dataset and print a quick schema summary."""

import csv
import sys
from pathlib import Path

sys.path.insert(0, str(Path(__file__).resolve().parent))

import numpy as np

from common import TOXICITY_DIR, ensure_layout

JIGSAW_COLUMNS = ['toxic', 'severe_toxic', 'obscene', 'threat', 'insult', 'identity_hate']
RUDDIT_COLUMN = 'offensiveness_score'


def candidate_files() -> list[Path]:
    ensure_layout()
    files: list[Path] = []
    for path in sorted(TOXICITY_DIR.glob('*')):
        if path.is_file() and path.suffix.lower() in {'.csv', '.tsv'}:
            files.append(path)
    return files


def delimiter_for(path: Path) -> str:
    return '\t' if path.suffix.lower() == '.tsv' else ','


def summarize_jigsaw(path: Path) -> dict[str, float]:
    total = 0
    toxic_any = 0
    label_sums = {col: 0.0 for col in JIGSAW_COLUMNS}

    with path.open('r', newline='', encoding='utf-8') as handle:
        reader = csv.DictReader(handle, delimiter=delimiter_for(path))
        for row in reader:
            total += 1
            values = []
            for col in JIGSAW_COLUMNS:
                value = float(row[col])
                label_sums[col] += value
                values.append(value)
            if max(values) > 0:
                toxic_any += 1

    summary = {'rows': float(total), 'toxic_any_rate': toxic_any / total if total else 0.0}
    for col in JIGSAW_COLUMNS:
        summary[f'mean_{col}'] = label_sums[col] / total if total else 0.0
    return summary


def summarize_ruddit(path: Path) -> dict[str, float]:
    values = []
    with path.open('r', newline='', encoding='utf-8') as handle:
        reader = csv.DictReader(handle, delimiter=delimiter_for(path))
        for row in reader:
            values.append(float(row[RUDDIT_COLUMN]))

    arr = np.array(values, dtype=float)
    if arr.size == 0:
        return {'rows': 0.0}
    return {
        'rows': float(arr.size),
        'score_min': float(np.min(arr)),
        'score_q25': float(np.quantile(arr, 0.25)),
        'score_median': float(np.median(arr)),
        'score_q75': float(np.quantile(arr, 0.75)),
        'score_max': float(np.max(arr)),
        'score_mean': float(np.mean(arr)),
    }


def main() -> None:
    files = candidate_files()
    if not files:
        print('No local toxicity dataset found.')
        print(f'If you want direct toxicity calibration, drop a Jigsaw or Ruddit file into {TOXICITY_DIR}.')
        return

    selected = files[0]
    with selected.open('r', newline='', encoding='utf-8') as handle:
        reader = csv.reader(handle, delimiter=delimiter_for(selected))
        header = next(reader)
    header_set = set(header)

    print(f'Detected dataset: {selected.name}')
    if all(col in header_set for col in JIGSAW_COLUMNS):
        print('Schema: Jigsaw-style toxicity labels')
        summary = summarize_jigsaw(selected)
        for key, value in summary.items():
            print(f'  {key}: {value:.6f}')
        # Derive empirical phi/psi constraint.
        mean_tau = summary['toxic_any_rate']
        print(f'\n  Empirical tau proxy (mean toxicity rate): {mean_tau:.6f}')
        print('  Model constraint: phi/psi = mean_tau / V*(alpha=0.5)')
        try:
            from ivfs_validation import (HIGGS_TXT, build_hourly_curve, ensure_dataset,
                                          fit_basic_ivf, parse_activity_file, run_scenarios)
            ensure_dataset()
            rt, _ = parse_activity_file(HIGGS_TXT)
            _, _, _, wc = build_hourly_curve(rt)
            en = wc / np.max(wc)
            b0, g0, *_ = fit_basic_ivf(en)
            _, sr = run_scenarios(b0, g0)
            v_star = sr['Moderate (alpha=0.5)']['V_star']
            ratio = mean_tau / v_star if v_star > 0 else float('inf')
            print(f'  V* at alpha=0.5: {v_star:.6f}')
            print(f'  => phi/psi = {ratio:.4f}  (current: {0.5 / 0.1:.1f})')
        except Exception as exc:
            print(f'  (could not run IVFS model: {exc})')
    elif RUDDIT_COLUMN in header_set:
        print('Schema: Ruddit-style offensiveness scores')
        for key, value in summarize_ruddit(selected).items():
            print(f'  {key}: {value:.6f}')
    else:
        print('Schema not recognized.')
        print(f'Expected either Jigsaw columns {JIGSAW_COLUMNS} or Ruddit column {RUDDIT_COLUMN}.')


if __name__ == '__main__':
    main()
