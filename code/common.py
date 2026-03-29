"""Shared paths, dataset helpers, and ODE integration wrapper."""

from __future__ import annotations

import argparse
import gzip
import os
import shutil
from pathlib import Path

import numpy as np
from scipy.integrate import solve_ivp

PROJECT_DIR = Path(__file__).resolve().parents[1]
CODE_DIR = PROJECT_DIR / 'code'
DATA_DIR = PROJECT_DIR / 'data'
ASSETS_DIR = PROJECT_DIR / 'assets'
TOXICITY_DIR = DATA_DIR / 'toxicity'

MPLCONFIG_DIR = CODE_DIR / '.mplconfig'
CACHE_DIR = CODE_DIR / '.cache'

os.environ.setdefault('MPLCONFIGDIR', str(MPLCONFIG_DIR))
os.environ.setdefault('XDG_CACHE_HOME', str(CACHE_DIR))

HIGGS_GZ = DATA_DIR / 'higgs-activity_time.txt.gz'
HIGGS_TXT = DATA_DIR / 'higgs-activity_time.txt'


def ensure_layout() -> None:
    """Create the project directories the scripts write to."""
    for path in (DATA_DIR, ASSETS_DIR, TOXICITY_DIR, MPLCONFIG_DIR, CACHE_DIR):
        path.mkdir(parents=True, exist_ok=True)


def resolve_higgs_dataset(
    dataset_path: str | Path | None = None,
) -> Path:
    """Return a local Higgs activity file from the repository or a user path.

    Looks in order: user-supplied path, then data/higgs-activity_time.txt,
    then data/higgs-activity_time.txt.gz (auto-decompresses).
    """
    ensure_layout()

    candidates: list[Path] = []
    if dataset_path is not None:
        candidates.append(Path(dataset_path).expanduser().resolve())
    candidates += [HIGGS_TXT, HIGGS_GZ]

    for path in candidates:
        if not path.exists():
            continue
        if path.suffix == '.gz':
            if not HIGGS_TXT.exists() or HIGGS_TXT.stat().st_mtime < path.stat().st_mtime:
                with gzip.open(path, 'rt', encoding='utf-8') as src, \
                     HIGGS_TXT.open('w', encoding='utf-8') as dst:
                    shutil.copyfileobj(src, dst)
            return HIGGS_TXT
        return path

    raise FileNotFoundError(
        f'Higgs dataset not found locally. Place it at {HIGGS_TXT} or pass --dataset-path.'
    )


def solve_trajectory(
    rhs,
    y0,
    t_eval,
    args=(),
    method: str = 'RK45',
    rtol: float = 1e-8,
    atol: float = 1e-10,
    max_step: float = np.inf,
):
    """Integrate an ODE system on a fixed time grid via solve_ivp.

    The default tolerances (rtol=1e-8, atol=1e-10) are tighter than the
    scipy defaults.  This matters for the IVFS system because the 2 000+
    hour integrations accumulate drift in the tail if the tolerances are
    too loose.  RK45 is a good match here: the system is non-stiff and
    moderate-dimensional (6 states).
    """
    t_arr = np.asarray(t_eval, dtype=float)
    sol = solve_ivp(
        lambda t, y: rhs(t, y, *args),
        (float(t_arr[0]), float(t_arr[-1])),
        np.asarray(y0, dtype=float),
        method=method,
        t_eval=t_arr,
        rtol=rtol,
        atol=atol,
        max_step=max_step,
    )
    if not sol.success:
        raise RuntimeError(f'ODE solver failed: {sol.message}')
    return sol.y.T


def add_dataset_cli_arguments(parser: argparse.ArgumentParser) -> None:
    """Add the standard local-dataset CLI flags."""
    parser.add_argument('--dataset-path', type=str, default=None,
                        help='Path to a local Higgs activity file (.txt or .gz).')
