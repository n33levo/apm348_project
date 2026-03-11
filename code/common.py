from __future__ import annotations

import os
from pathlib import Path

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

# Matplotlib loves writing cache files somewhere. I would rather keep that noise inside the repo.
os.environ.setdefault('MPLCONFIGDIR', str(MPLCONFIG_DIR))
os.environ.setdefault('XDG_CACHE_HOME', str(XDG_CACHE_HOME))


def ensure_layout() -> None:
    """Create the folders this repo expects to exist."""
    for path in (DATA_DIR, ASSETS_DIR, TOXICITY_DIR, MPLCONFIG_DIR, XDG_CACHE_HOME):
        path.mkdir(parents=True, exist_ok=True)
