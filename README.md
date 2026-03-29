# APM348 IVFS Project

## Setup

Create a virtual environment, install the requirements, and run the scripts from the repo root.

```bash
python -m venv .venv
source .venv/bin/activate
pip install -r requirements.txt
```

The code can be run either as plain scripts or as package modules. The package form is the more reproducible option because all internal imports follow the same path.

## Main scripts

Run these from the repository root:

```bash
python -m code.ivfs_validation --offline
python -m code.equilibrium_analysis --offline
python -m code.benchmark_models --offline
python -m code.toxicity_calibration
```

If the Higgs file is not in the default data directory, pass `--dataset-path /path/to/higgs-activity_time.txt`. Download is disabled by default. Use `--allow-download` only if you explicitly want the code to fetch the public SNAP file.

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
  - summarizes an optional external toxicity reference file supplied by the user
  - supports Jigsaw-style and Ruddit-style CSV schemas
  - writes a metadata record even when no external file is present

- `code/higgs_window_validation.py`
  - refits the reduced IVF model on alternate active Higgs windows
  - checks whether the spread fit is stable away from the main 100 hour window

- `code/equilibrium_analysis.py`
  - computes disease free and positive equilibria
  - checks local stability numerically

- `code/heatmap_figures.py`, `code/extra_figures.py`, `code/model_structure_figure.py`
  - generate supplementary figures used in the report
  - reuse the same fitted parameters and solver settings as the main pipeline

## Data

- The main Higgs activity file should be placed at `data/higgs-activity_time.txt` or `data/higgs-activity_time.txt.gz`.
- The repository code checks the Higgs file against SHA256
  `6487c03bb5f0f966447a7a7410c5ba124f7ddf1b3d7ff9c3a94d20891214225b`.
- Optional external toxicity reference files are not required and are not assumed to be present. If you want to use one, place it in `data/toxicity/`.
- The Jigsaw or Ruddit files are used only as external text toxicity references. They are not the main calibration dataset.

## Outputs

Figures and run metadata are written to `assets/`, including:

- `apm348_calibration_diagnostics.png`
- `apm348_results.png`
- `benchmark_curve_compare.png`
- `tau_proxy_comparison.png`
- `phi_sensitivity.png`
- `profile_likelihood.png`
- `ivfs_run_metadata.json`
- `benchmark_run_metadata.json`
- `equilibrium_run_metadata.json`
- `toxicity_reference_metadata.json`

## Notes

- the repo uses Higgs as the main spread dataset
- throughout the project, `tau` represents latent rage-bait toxicity, namely reply-driven discussion pressure and its feedback into spread and user attrition
- Jigsaw toxicity refers only to an external text label scale, not to the state variable itself
- the ODE integration uses explicit `solve_ivp` settings recorded in metadata files
- the detailed project direction is documented in `purpose.md` and `nextplans.md`

## Tests

Run the lightweight regression suite from the repository root:

```bash
python -m unittest discover -s tests -v
```
