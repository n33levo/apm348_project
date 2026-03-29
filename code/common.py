from __future__ import annotations

import argparse
import gzip
import hashlib
import os
import shutil
import subprocess
from datetime import datetime, timezone
from pathlib import Path

import json

import numpy as np
from scipy.integrate import solve_ivp

PROJECT_DIR = Path(__file__).resolve().parents[1]
CODE_DIR = PROJECT_DIR / 'code'
DATA_DIR = PROJECT_DIR / 'data'
ASSETS_DIR = PROJECT_DIR / 'assets'
TOXICITY_DIR = DATA_DIR / 'toxicity'
MPLCONFIG_DIR = CODE_DIR / '.mplconfig'
XDG_CACHE_HOME = CODE_DIR / '.cache'

HIGGS_GZ = DATA_DIR / 'higgs-activity_time.txt.gz'
HIGGS_TXT = DATA_DIR / 'higgs-activity_time.txt'
HIGGS_URL = 'http://snap.stanford.edu/data/higgs-activity_time.txt.gz'
HIGGS_SHA256 = '6487c03bb5f0f966447a7a7410c5ba124f7ddf1b3d7ff9c3a94d20891214225b'

# Matplotlib keeps dumping cache files everywhere so I've just shoved them in the repo
os.environ.setdefault('MPLCONFIGDIR', str(MPLCONFIG_DIR))
os.environ.setdefault('XDG_CACHE_HOME', str(XDG_CACHE_HOME))


def ensure_layout() -> None:
    """make sure all the folders we need actually exist"""
    for path in (DATA_DIR, ASSETS_DIR, TOXICITY_DIR, MPLCONFIG_DIR, XDG_CACHE_HOME):
        path.mkdir(parents=True, exist_ok=True)


def file_sha256(path: Path, chunk_size: int = 1024 * 1024) -> str:
    hasher = hashlib.sha256()
    with path.open('rb') as handle:
        while True:
            chunk = handle.read(chunk_size)
            if not chunk:
                break
            hasher.update(chunk)
    return hasher.hexdigest()


def verify_checksum(path: Path, expected_sha256: str | None) -> bool:
    if expected_sha256 is None or not path.exists():
        return True
    return file_sha256(path) == expected_sha256


def resolve_higgs_dataset(dataset_path: str | Path | None = None,
                          allow_download: bool = False,
                          offline: bool = False,
                          expected_sha256: str | None = HIGGS_SHA256) -> Path:
    """Return a usable Higgs dataset path, downloading only when explicitly allowed."""
    ensure_layout()

    candidates: list[Path] = []
    if dataset_path is not None:
        candidates.append(Path(dataset_path).expanduser().resolve())
    candidates.extend([HIGGS_TXT, HIGGS_GZ])

    for candidate in candidates:
        if not candidate.exists():
            continue
        if candidate.suffix == '.gz':
            if offline and not HIGGS_TXT.exists():
                target = HIGGS_TXT
            else:
                target = HIGGS_TXT
            if not target.exists() or target.stat().st_mtime < candidate.stat().st_mtime:
                with gzip.open(candidate, 'rt', encoding='utf-8') as src, target.open('w', encoding='utf-8') as dst:
                    shutil.copyfileobj(src, dst)
            if expected_sha256 is not None and not verify_checksum(target, expected_sha256):
                raise RuntimeError(f'Higgs dataset checksum mismatch for {target}')
            return target
        if expected_sha256 is not None and candidate == HIGGS_TXT and not verify_checksum(candidate, expected_sha256):
            raise RuntimeError(f'Higgs dataset checksum mismatch for {candidate}')
        return candidate

    if offline:
        raise FileNotFoundError(
            f'Higgs dataset not found. Provide a local file with --dataset-path or place it at {HIGGS_TXT}.'
        )

    if not allow_download:
        raise FileNotFoundError(
            'Higgs dataset not found locally and download is disabled. '
            f'Place the raw file at {HIGGS_TXT} or rerun with --allow-download.'
        )

    import urllib.request

    urllib.request.urlretrieve(HIGGS_URL, HIGGS_GZ)
    with gzip.open(HIGGS_GZ, 'rt', encoding='utf-8') as src, HIGGS_TXT.open('w', encoding='utf-8') as dst:
        shutil.copyfileobj(src, dst)
    if expected_sha256 is not None and not verify_checksum(HIGGS_TXT, expected_sha256):
        raise RuntimeError(f'Higgs dataset checksum mismatch for downloaded file {HIGGS_TXT}')
    return HIGGS_TXT


def solve_trajectory(rhs,
                     y0: list[float] | np.ndarray,
                     t_eval: np.ndarray,
                     args: tuple = (),
                     method: str = 'RK45',
                     rtol: float = 1e-8,
                     atol: float = 1e-10,
                     max_step: float = np.inf) -> np.ndarray:
    """Integrate an ODE on a fixed grid and return the state array with shape (n_times, n_states)."""
    t_arr = np.asarray(t_eval, dtype=float)
    sol = solve_ivp(
        lambda t_now, y_now: rhs(t_now, y_now, *args),
        (float(t_arr[0]), float(t_arr[-1])),
        np.asarray(y0, dtype=float),
        method=method,
        t_eval=t_arr,
        rtol=rtol,
        atol=atol,
        max_step=max_step,
    )
    if not sol.success:
        raise RuntimeError(f'ODE solve failed: {sol.message}')
    return sol.y.T


def current_git_commit(project_dir: Path = PROJECT_DIR) -> str | None:
    try:
        result = subprocess.run(
            ['git', 'rev-parse', 'HEAD'],
            cwd=project_dir,
            capture_output=True,
            text=True,
            check=True,
        )
    except Exception:
        return None
    return result.stdout.strip() or None


def utc_timestamp() -> str:
    return datetime.now(timezone.utc).replace(microsecond=0).isoformat()


def write_json(path: Path, payload: dict) -> None:
    path.parent.mkdir(parents=True, exist_ok=True)
    with path.open('w', encoding='utf-8') as handle:
        json.dump(payload, handle, indent=2, sort_keys=True)


def dataset_metadata(path: Path, expected_sha256: str | None = HIGGS_SHA256) -> dict[str, str | int | bool]:
    meta = {
        'path': str(path),
        'exists': path.exists(),
    }
    if path.exists():
        digest = file_sha256(path)
        meta['sha256'] = digest
        meta['bytes'] = path.stat().st_size
        if expected_sha256 is not None:
            meta['checksum_matches_expected'] = digest == expected_sha256
    return meta


def add_dataset_cli_arguments(parser: argparse.ArgumentParser) -> None:
    parser.add_argument('--dataset-path', type=str, default=None,
                        help='Optional path to a local Higgs activity file (.txt or .gz).')
    parser.add_argument('--allow-download', action='store_true',
                        help='Allow downloading the Higgs dataset if it is missing locally.')
    parser.add_argument('--offline', action='store_true',
                        help='Fail instead of downloading if the Higgs dataset is missing locally.')


def build_run_metadata(script_name: str,
                       dataset_path: Path,
                       parameters: dict,
                       solver: dict,
                       outputs: dict,
                       notes: dict | None = None) -> dict:
    return {
        'script': script_name,
        'timestamp_utc': utc_timestamp(),
        'git_commit': current_git_commit(),
        'dataset': dataset_metadata(dataset_path),
        'parameters': parameters,
        'solver': solver,
        'outputs': outputs,
        'notes': notes or {},
    }
