# APM348 IVFS Project

## Setup

Create a virtual environment, install the requirements, and run the scripts from the repo root.

```bash
python -m venv .venv
source .venv/bin/activate
pip install -r requirements.txt
```

## Main scripts

Run these from the repository root:

```bash
python code/ivfs_validation.py
python code/equilibrium_analysis.py
python code/benchmark_models.py
python code/toxicity_calibration.py
```

## What each script does

- `code/ivfs_validation.py`
  - main pipeline
  - calibrates the Higgs spread fit
  - runs the IVFS policy scenarios
  - generates the main figures

- `code/equilibrium_analysis.py`
  - checks the disease-free and positive equilibria
  - computes local stability numerically

- `code/benchmark_models.py`
  - compares the reduced IVF spread fit against SIR on the same Higgs window

- `code/toxicity_calibration.py`
  - summarizes an optional external toxicity dataset
  - supports Jigsaw-style and Ruddit-style files

## Data

- Higgs activity data lives in `data/`
- optional toxicity reference files go in `data/toxicity/`

## Outputs

Figures are written to `assets/`, including:

- `apm348_calibration_diagnostics.png`
- `apm348_results.png`
- `benchmark_curve_compare.png`
- `tau_proxy_comparison.png`
- `phi_sensitivity.png`
- `profile_likelihood.png`

## Notes

- the repo uses Higgs as the main spread dataset
- `tau` is currently interpreted as harmful-discussion or reply-pressure, not literal observed text toxicity
- the detailed project direction is documented in `purpose.md` and `nextplans.md`
