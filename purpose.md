# Purpose

## What this repo is, in normal language

This repo is basically the clean version of the APM348 project after all the earlier drafts, random folders, and test files got stripped away. The core idea is to model how rage-bait style content moves through a platform over time, instead of only modeling users like a standard epidemic paper would do.

So the main question is:

how does platform amplification boost short-term virality, but also make the platform more toxic and possibly push users away over time?

That is the whole point of the project.

## What the proposal is saying

The proposal in `final.tex` and the matching PDF is pitching a content-centric ODE model. The states are:

- I = ignored content
- V = viral content
- F = fatigued content
- S = moderated or suppressed content
- tau = overall toxicity level on the platform
- U = active users

The important twist is that the spread and fatigue rates are not fixed. They depend on toxicity too, so the model uses:

- beta_eff = effective virality rate
- gamma_eff = effective fatigue rate

That matters because it lets the platform environment feed back into the content lifecycle, instead of treating every post like it spreads in the exact same way forever.

## What data we actually have

Right now the real dataset in the repo is the Stanford SNAP Higgs Twitter activity stream.

We already verified the main counts:

- 563,069 total events
- 354,930 RT events
- RT just means retweet
- the main retweet spike peaks at hour 78
- that peak hour has 30,602 retweets

For calibration, we do not use the whole raw series directly. We bin retweets by hour, then take a 100-hour window around the main spike. That gives us a clean viral spike + decay shape to fit against the baseline IVF model.

## What the code already does

### `code/ivfs_validation.py`
This is the main script.

What it does:

- makes sure the Higgs dataset exists in `data/`
- unzips the file if needed
- parses out only RT events
- groups them into hourly counts
- builds the 100-hour calibration window
- fits the baseline IVF model with Nelder-Mead
- runs the full IVFS-type model under 3 policy scenarios
- runs the alpha sweep
- saves the main figure to `assets/apm348_results.png`

So if someone only runs one script in this repo, this is the one.

### `code/equilibrium_analysis.py`
This is the more math-heavy follow-up.

What it does:

- uses the fitted beta0 and gamma0 from the Higgs calibration
- checks the disease-free equilibrium
- computes the positive equilibrium numerically
- checks local stability through the Jacobian eigenvalues

This is useful because it gives us something stronger than just “we ran simulations and the plot looked nice.”

### `code/benchmark_models.py`
This compares our content-centric setup against a simpler SIR benchmark on the same Higgs retweet spike.

That comparison is important because it keeps the claim honest.

The result is basically:

- SIR can fit a one-dimensional viral spike pretty well too
- so the novelty is not “our model crushes SIR on one curve fit”
- the novelty is that our model has content states, moderation, toxicity, and user-retention dynamics that SIR does not have

### `code/toxicity_calibration.py`
This is the placeholder for the next empirical step.

Right now it looks for either:

- Jigsaw-style toxicity labels
- Ruddit-style offensiveness scores

But we do not currently have a toxicity-labeled dataset inside this repo. So the script is ready, but the real direct toxicity calibration has not happened yet.

## What results we already have

From the clean calibration run, the baseline fitted rates are:

- beta0 = 0.818194 per hour
- gamma0 = 0.211753 per hour

From the scenario runs:

- alpha = 0.9 gives higher toxicity and lower user retention
- alpha = 0.5 is in the middle
- alpha = 0.2 gives lower toxicity and better user retention

From the equilibrium check:

- the disease-free equilibrium is unstable in the tested scenarios
- a positive equilibrium exists for alpha = 0.2, 0.5, 0.9
- that positive equilibrium is locally stable in the clean calibrated setup

So the project already has a real calibration step, a real simulation step, and a real equilibrium/stability step.

## What we can honestly claim rn

The strong version we can defend is:

this is a content-centric deterministic ODE framework with preliminary data grounding from a real social media cascade.

The version we should not overclaim is:

this is already a fully data-calibrated proof of bistability for platform toxicity.

That would be too strong rn because:

- only beta0 and gamma0 are directly fit from data
- the toxicity side is still parameterized mostly through assumptions and sensitivity choices
- the clean parameterization supports threshold and local stability results, but not a full dramatic bistability theorem yet

## What the assets are

There are 2 main generated figures in `assets/`:

- `apm348_results.png` = calibration + scenario time series + alpha sweep
- `benchmark_curve_compare.png` = IVF vs SIR fit comparison on the Higgs spike

## Big-picture summary

If I had to explain the whole repo in like 20 seconds, I would say:

this project takes a real viral event dataset, fits the basic spread/fatigue part of a content-lifecycle model, then uses that to study the tradeoff between amplification, toxicity, and user retention on a platform. The course-project version is already in good shape. The paper version would need stronger proofs and better toxicity data.
