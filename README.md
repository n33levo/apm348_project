Setup

1. Create a virtual environment.
2. Install the requirements.
3. Run the scripts from the repo root.

Commands

python -m venv .venv
source .venv/bin/activate
pip install -r requirements.txt

python code/ivfs_validation.py
python code/equilibrium_analysis.py
python code/benchmark_models.py
python code/toxicity_calibration.py

Notes

- The Higgs dataset lives in `data/`.
- Figures are written to `assets/`.
- If you want to test toxicity calibration, put a Jigsaw or Ruddit CSV/TSV file in `data/toxicity/`.
