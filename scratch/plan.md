from codex;

# Repair Higgs Figures Without Pretending Toxicity Is Calibrated

## Summary
- Keep Higgs as the only calibrated dataset in this phase. The official SNAP release is anonymized activity/network data with timestamps, not tweet text, so same-dataset toxicity calibration is not possible from the current source ([SNAP Higgs](https://snap.stanford.edu/data/higgs-twitter.html)).
- Remove Jigsaw-derived toxicity from the main calibrated story. Treat toxicity and user-retention outputs as assumption-driven scenario analysis until a joint dataset is added.
- Do not pivot datasets now. If you later want a single-dataset rebuild, the best online candidate I found is Civil Comments because it exposes text plus `created_date`, `article_id`, `parent_id`, and toxicity labels ([TFDS Civil Comments](https://www.tensorflow.org/datasets/catalog/civil_comments)).

## Implementation Changes
- In [ivfs_validation.py](/Users/neelmac1/Documents/hw/project/project/code/ivfs_validation.py):
  - Split calibration diagnostics from policy results.
  - Add a new figure `assets/apm348_calibration_diagnostics.png` with four panels: Higgs raw points + rolling empirical mean + IVF fit, residuals by hour, tail-bias summary, and IVF-vs-SIR comparison.
  - Keep `assets/apm348_results.png` as policy-only, with panels `(V(t), τ(t), U(t), τ*(α), U*(α), E*(α))`.
  - Remove the current Higgs calibration panel from `apm348_results.png`.
  - Stop labeling the weighted optimization objective as plain `SSE`. Report `weighted_loss`, unweighted full-window `R²`, spike-window `R²`, and empirical-vs-model tail mean.
  - Keep the current Higgs fit family for now (`β0, γ0, λ_bg, V0`), but explicitly mark late-hour secondary bursts as unresolved background rather than fitted signal.
- In [toxicity_calibration.py](/Users/neelmac1/Documents/hw/project/project/code/toxicity_calibration.py):
  - Reclassify the script as an external toxicity reference tool, not a main calibration step.
  - Fix the stale hardcoded `phi/psi` printout by importing live `PHI` and `PSI`.
  - Do not auto-write its output back into the main IVFS parameters.
  - Rename console messaging so it no longer implies Higgs and Jigsaw are jointly calibrated.
- In [benchmark_models.py](/Users/neelmac1/Documents/hw/project/project/code/benchmark_models.py):
  - Keep the benchmark figure, but change the framing from “IVF outperforms SIR” to “IVF is only marginally better on one spike; IVFS adds structure beyond SIR.”
  - Report correlation and tail means directly so the figure matches the actual evidence.

## Test Plan
- Re-run the three scripts and verify that:
  - Higgs calibration stays numerically stable near the current fit (`β0≈1.485`, `γ0≈0.153`) within tolerance.
  - `apm348_calibration_diagnostics.png` reports consistent metrics: weighted loss, full-window `R²`, spike-window `R²`, and tail mean bias.
  - `apm348_results.png` is clearly labeled as scenario output and no longer implies toxicity is Higgs-calibrated.
  - `toxicity_calibration.py` prints the live `PHI/PSI` ratio, not the stale `0.5/0.1`.
  - Scenario ordering still holds: higher `α` gives higher `τ*`, lower `U*`, and boundary-peaking `E*` unless parameters are intentionally changed later.

## Assumptions And Defaults
- `scratch/perplexity.md` is not present in the workspace, so this plan is based on the repo, PDFs, `percopilot.md`, and the attached figures.
- This phase fixes the code and figures only; paper rewriting is deferred.
- Online check also confirmed the toxicity/engagement tradeoff is still a defensible modeling target even if calibration is not yet unified ([Beknazar-Yuzbashev et al., January 19, 2025](https://ssrn.com/abstract=5130929)).


from copilot;

**What I found:**

The core problems are:
1. **Panel (a) shows all 100 hours with no x-axis clip** — the model naturally decays smoothly after the spike peak at ~hour 10, but the real data has secondary bursts and noise in hours 35–100. This creates dozens of scattered gray dots far above the model curve. The R² of 0.69 is computed *only* over hours 0–35, but the *plot* shows all 100 hours — so the figure looks much worse than the number suggests.

2. **The Higgs dataset already has RE (reply) events** — confirmed on the SNAP page. The file contains RT, MT, and RE interactions. RE (replies) are direct one-person-responds-to-another events, which are a natural proxy for discussion intensity / controversy, i.e. exactly what τ should represent. We're sitting on unified calibration data and not using it.

3. **Jigsaw is Wikipedia comments, not Twitter** — completely different content ecosystem, different toxicity rates, different dynamics. This is a legitimate methodological hole.

---

## Plan: Unified Calibration from Higgs RE+RT Events

Right now the model is calibrated in two disconnected pieces — β₀/γ₀ from Higgs RT data, and φ from Jigsaw Wikipedia comments. The Higgs activity file already contains RE (reply) events which serve as a natural τ proxy from the same source. The fix is to use RT as V(t) and RE as τ(t), calibrate everything from one file, and clip the calibration figure to the spike window.

### Steps

1. **Fix panel (a) visually** by adding `set_xlim(0, 50)` and `set_ylim(bottom=0)` to the calibration panel in `make_figure` — show only the spike, not the noisy 65-hour tail that the ODE can't model.

2. **Extract RE events in `parse_activity_file`** in ivfs_validation.py — currently it only keeps `parts[3] == 'RT'`; extend it to also collect `RE` timestamps in a second list and return both.

3. **Build a τ_empirical(t) series** alongside the RT hourly curve in `build_hourly_curve` — bin RE events into the same 100h window and define `tau_empirical = re_counts / (rt_counts + 1)` (reply rate), normalized to [0,1]. This is the proxy for platform toxicity from the Higgs data itself.

4. **Add a `calibrate_phi_psi` function** that fits φ and ψ by minimizing weighted SSE between the IVFS model's τ(t) trajectory (using already-fitted β₀, γ₀) and `tau_empirical(t)`. Replace the Jigsaw-derived `PHI = 0.056` constant with the fitted value. Update toxicity_calibration.py to use this approach and keep Jigsaw only as an optional cross-check.

5. **Add a second calibration panel to the figure** — panel (a) stays as the RT/V fit, add a new panel (or replace panel (b)'s "scenario τ(t)") showing the RE-based τ_empirical overlay with the model's τ(t) trajectory, proving the fit visually.

6. **Update `run_sensitivity` and equilibrium outputs** with the new calibrated φ/ψ values, re-run everything, and verify all 4 scripts still pass.

### Further Considerations

1. **RE event count is ~10% of RT** (32k vs 328k in the Higgs network statistics) — the reply curve will be noisier. You may want to smooth it with a rolling 3-hour window before fitting φ/ψ, or fit only over the spike window (0–35h) where signal is strongest.

2. **The τ proxy semantics**: RE events in the Higgs data include replies to retweeted tweets (not purely "toxic" replies — mostly scientific discussion). This is actually *better* for your model than Jigsaw because τ in your model represents "discussion intensity / engagement fatigue" rather than hate speech per se. The proposal cites Beknazar-Yuzbashev et al. (2025) which treats toxic engagement broadly — this framing fits.

3. **E\* narrative is still broken** — the W-search confirmed E\* is monotonically increasing in α for all W≤80. This is actually fine if the paper's conclusion is reframed as: *"platforms cannot simultaneously maximize engagement and minimize toxicity — α=1 gives the highest E\* but the worst τ\* and U\*."* That's a real and interesting result. No code change needed, just a narrative fix in the paper later.


from codex again; 
# Best Figure-Repair Plan: Honest Higgs Calibration, RE As Auxiliary Proxy, No Mixed Calibration Claim

## Summary
- Adopt the visual/diagnostic fixes from the Copilot suggestion, but do **not** make `RE` the primary toxicity calibration for the model.
- Use Higgs `RT` as the only primary calibration target for the fitted spread model.
- Use Higgs `RE` as a **same-dataset auxiliary reply-pressure proxy** for diagnostics only, because it is real, unified, and worth showing, but it is too semantically weak and too collinear with `RT` to support a clean standalone `φ, ψ` calibration.
- Remove Jigsaw from the main calibrated pipeline and keep it only as an optional external cross-check tool.
- Reframe the results figure so it no longer mixes calibration evidence and policy scenarios in one panel set.

## Key Decisions
- Treat `τ` as “toxicity / discussion pressure” in the codebase comments and figure labels when referring to empirical proxies.
- Reject “fit `φ, ψ` from `RE` and replace Jigsaw entirely” as the primary path.
  Reason: in the Higgs spike window, `RE` is nearly a scaled copy of `RT` rather than an independent signal. Local exploration showed `corr(RT, RE) ≈ 0.995` and `RE/RT` is roughly flat around `0.11`, so it does not identify `τ` dynamics robustly.
- Keep the monotone `E*(α)` result. Do not force an interior optimum. Later paper framing should say the engagement-maximizing platform also produces the worst long-run pressure/user outcomes.
- Show the spike window prominently, but keep a separate diagnostic view of the full 100-hour window so the tail mismatch is visible rather than hidden.

## Implementation Changes
- In [ivfs_validation.py](/Users/neelmac1/Documents/hw/project/project/code/ivfs_validation.py):
  - Replace the positional calibration-return flow with a small structured container for calibration data, carrying `rt_timestamps`, `re_timestamps`, hourly counts, peak hour, window start, `rt_window`, `re_window`, and a smoothed `reply_proxy_window`.
  - Extend `parse_activity_file()` to collect both `RT` and `RE` timestamps.
  - Extend `build_hourly_curve()` into a calibration-window builder that bins both series on the same clock.
  - Define `reply_proxy_window` as a smoothed same-window series using `RE / (RT + 1)` with a centered 3-hour moving average, clipped to `[0, 1]`.
  - Keep `fit_basic_ivf()` fitting only the `RT` spike.
  - Add unweighted diagnostics alongside the weighted fit objective:
    - `weighted_loss`
    - full-window `R²`
    - spike-window `R²` on hours `0–35`
    - tail mean bias on hours `40–99`
  - Split figure generation into:
    - `assets/apm348_calibration_diagnostics.png`
    - `assets/apm348_results.png`
  - Make `apm348_calibration_diagnostics.png` a 2x2 diagnostic figure:
    - `RT` empirical vs IVF fit, clipped to `xlim=(0, 50)` and `ylim(bottom=0)`
    - full 100-hour calibration with residual/tail mismatch visible
    - `RE` reply-pressure proxy over the same window
    - benchmark overlay of IVF vs SIR on the RT spike
  - Make `apm348_results.png` policy-only:
    - `V(t)` by policy
    - `τ(t)` by policy
    - `U(t)` by policy
    - `τ*(α)`
    - `U*(α)`
    - `E*(α)`
  - Remove the calibration scatter from the main policy figure.
  - Change annotation text so the figure no longer calls the weighted objective plain `SSE`.

- In [benchmark_models.py](/Users/neelmac1/Documents/hw/project/project/code/benchmark_models.py):
  - Reword the benchmark title/caption logic to “IVF is only marginally better on one spike; IVFS adds extra state structure.”
  - Report full-window `R²`, weighted loss, correlation, and tail means directly in the console output and benchmark annotation.

- In [toxicity_calibration.py](/Users/neelmac1/Documents/hw/project/project/code/toxicity_calibration.py):
  - Recast the script as an optional external-toxicity summary tool.
  - Import live `PHI` and `PSI` instead of printing the stale hardcoded `0.5 / 0.1`.
  - Remove any implication that Jigsaw calibration feeds automatically into the main IVFS constants.
  - Add console wording that Jigsaw is an external reference distribution, not the main Higgs calibration source.

- In [equilibrium_analysis.py](/Users/neelmac1/Documents/hw/project/project/code/equilibrium_analysis.py):
  - Keep the current equilibrium path unchanged except for importing renamed/shared calibration helpers if needed.
  - Update printed messaging so `τ` is described as scenario-driven rather than empirically toxicity-calibrated.

## Test Plan
- Re-run all four scripts from the repo root.
- Verify Higgs event parsing returns counts consistent with local inspection:
  - `RT ≈ 354,930`
  - `RE ≈ 36,902`
- Verify the RT fit remains near the current calibrated values:
  - `β0 ≈ 1.485`
  - `γ0 ≈ 0.153`
  - weighted loss stays close to the current optimum
- Verify calibration diagnostics report both:
  - full-window fit quality
  - spike-window fit quality
- Verify the clipped RT panel looks visually consistent with the spike-window metric.
- Verify the RE proxy panel is clearly labeled as a reply-pressure proxy, not literal toxicity.
- Verify `apm348_results.png` contains only policy/equilibrium outputs and no empirical calibration panel.
- Verify `toxicity_calibration.py` prints the live `PHI/PSI` ratio and never claims those values came from Higgs unless explicitly running an auxiliary proxy path.
- Verify `E*(α)` still peaks at the boundary and that this is reported as an expected result, not a failure.

## Assumptions And Defaults
- `scratch/perplexity.md` is still not available in the workspace, so this plan is based on the repo, attached figures, PDFs, `percopilot.md`, and direct Higgs inspection.
- RE data from SNAP is accepted as a valid same-dataset interaction proxy, but not as a literal toxicity label ([SNAP Higgs](https://snap.stanford.edu/data/higgs-twitter.html)).
- A future single-dataset empirical rebuild should use a source with timestamps plus toxicity labels, such as Civil Comments/WILDS, rather than forcing Higgs replies to do more than they support.
- Paper-writing changes are deferred; only code, figures, labels, and calibration honesty are in scope for this phase.
