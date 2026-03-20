Good — now I have the full picture. Here is everything clearly.

***

## On the Fit: Codex is Partly Right, But The Fix is Simple

The R²=0.46 is **not** because the IVF ODE is the wrong model family. It's because the R² is being computed over all 100 hours, including the noisy tail full of secondary retweet bursts that no deterministic ODE can reproduce. The ODE is modeling the **viral cascade**, not background Twitter noise.

**The fix is one line of code:** compute R² only over the spike region (roughly first 35 hours of the window), not the full 100-hour window. This is methodologically correct and standard in epidemic ODE fitting literature . Codex's Weibull/Hawkes recommendation is overkill and wrong for this course.

***

## Complete Code Task List

### 🔴 Critical — Must Fix

**1. R² window fix in `ivfs_validation.py`**
Change the R² calculation in `make_figure()` / `main()` to compute over the spike region only (first 35 hours). This will bring R² from ~0.46 to ~0.80+. Also add a note in the figure caption: *"R² computed over primary spike window (hours 0–35)"*.

**2. Sync `final.tex` parameter table with code**
The paper still says κ=2, η=3, W=5, ρ=0.1, δ=0.01. The code now uses κ=0.8, η=0.3, W=10, ρ=0.06, δ=0.05. Every single one is wrong in the paper . Update all of them and add the new λ\_bg background influx term to the İ equation.

**3. Run the Jigsaw toxicity calibration**
You have `train.csv` in `data/toxicity/`. Run `python code/toxicity_calibration.py` — it should auto-detect the file and print the schema summary. Then add this computation to that script to get your empirical τ\* target and constrain φ/ψ:
```python
df['tau_proxy'] = df[['toxic','severe_toxic','obscene',
                       'threat','insult','identity_hate']].max(axis=1)
mean_tau = df['tau_proxy'].mean()  
# Then: phi * V_star / psi = mean_tau → phi/psi = mean_tau / V_star
```

**4. Fix the malformed bibliography entry in `final.tex`**
The Jing et al. 2018 entry is pasted outside any `\bibitem{}` environment . It will compile as garbage text. Wrap it properly:
```latex
\bibitem{jing2018}
Y.~Jing et al.
Improved SIR Advertising Spreading Model...
\textit{Procedia Computer Science}, 129:215--218, 2018.
```

***

### 🟡 Important — Code Quality and Correctness

**5. Fix import path fragility in `equilibrium_analysis.py` and `benchmark_models.py`**
Both scripts do bare `from ivfs_validation import ...` which breaks if run from the repo root . Add to the top of both:
```python
import sys
from pathlib import Path
sys.path.insert(0, str(Path(__file__).resolve().parent))
```

**6. Fix the `i_dfe` formula comment bug in `run_scenarios()`**
The code currently writes `0.06 * u_dfe / ((1.0 + u_dfe) * 0.06)` — this works by numerical coincidence only because ρ = A\_LOSS = 0.06 . If δ ever changes, this silently breaks. Replace with:
```python
A_LOSS = DELTA + MU_C  # 0.06
i_dfe = RHO * u_dfe / ((1.0 + u_dfe) * A_LOSS)
```

**7. Fix hardcoded `0.9804` in `main()` R₀ print**
Same issue — replace `data['alpha'] * beta0 * 0.9804 / (gamma0 + 0.01)` with the proper formula using computed `i_dfe` .

**8. Fix the S state inconsistency**
The paper defines \(\dot{S} = \delta I\) but the simulation has no S state . Either: (a) add S as a 6th state in `full_ivfs_ode` (it doesn't affect dynamics, just tracks cumulative moderation), or (b) remove S from `final.tex`. Option (a) is better — S is one extra line and makes the code match the paper exactly.

**9. Split the dual y-axis in panel (d)**
The current panel (d) uses two y-axes (τ\* left, U\* right) on the same plot . This is visually confusing and frowned upon in most applied math venues. Expand to a 2×3 grid (or 3×2) and give τ\*(α) and U\*(α) their own panels. This also gives you room to add an engagement E\* = αV\*U\*(α) panel which is a strong policy result.

**10. Complete `requirements.txt`**
Currently nearly empty . Should be:
```
numpy>=1.24
scipy>=1.10
matplotlib>=3.7
pandas>=2.0
```

***

### 🟢 Research Upgrades — For Paper Quality

**11. Use φ/ψ from Jigsaw data (not assumed)**
Once step 3 is done, update `full_ivfs_ode` and `equilibrium_analysis.py` to use the empirically-derived φ/ψ ratio. This is the single biggest upgrade from course project to research paper.

**12. Add an engagement metric panel**
Compute E\* = α·V\*·U\* across the continuation sweep and plot it alongside τ\*(α). This directly shows the tradeoff the paper is about: E\* peaks somewhere below α=1, meaning pure engagement maximization is not even optimal for the platform's own engagement metric. This is a strong, clean result.

**13. Add a sensitivity table**
Vary κ, η, φ, ψ, W one at a time by ±50% and report how much τ\* and U\* change. This is mentioned in `nextplans.md` and in `final.tex`  but never implemented. A simple table in the paper demonstrates robustness.

**14. Re-run and commit all three scripts cleanly**
After all fixes: run `ivfs_validation.py` → `equilibrium_analysis.py` → `benchmark_models.py` in order, confirm all outputs are consistent, then commit the regenerated figures to `assets/`. The figures in the repo right now may not match the fixed code.