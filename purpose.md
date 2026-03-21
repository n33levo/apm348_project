# Purpose

## What this repository is trying to do

This repository studies a simple but important platform question:

how does stronger amplification help content spread in the short run, while also increasing harmful discussion pressure and reducing long-run user retention?

The project is built around a content-based ODE model rather than a standard user-only epidemic model. That is the main conceptual point of the work.

Instead of only tracking people, the model tracks the lifecycle of content itself:

- `I` = ignored content
- `V` = viral content
- `F` = fatigued content
- `S` = suppressed or moderated content
- `tau` = aggregate harmful-discussion pressure
- `U` = active users

The platform amplification parameter `alpha` changes how aggressively content is pushed through the system.

## What the repo is meant to provide

The repository is meant to be:

- a clean research version of the APM348 project
- a reproducible calibration and simulation pipeline
- a base for a stronger applied math paper after the course project

So the repo is not just a folder of plots. It is supposed to show:

- what is empirically grounded
- what is still assumption-driven
- what has already been established mathematically
- what still needs to be proved

## What is empirically grounded right now

The main empirical dataset is the Stanford SNAP Higgs Twitter activity stream.

What Higgs gives us directly:

- retweet activity (`RT`) for spread calibration
- reply activity (`RE`) for a same-dataset discussion-pressure proxy

What Higgs does not give us:

- tweet text
- literal toxicity labels in the Jigsaw sense

Because of that, the project is now more careful about what `tau` means.

The most honest interpretation is:

- `tau` is a harmful-discussion or outrage-pressure variable
- not a directly observed text-toxicity score

That framing fits the data much better and keeps the project closer to the actual rage-bait idea.

## What changed in the calibration story

The project originally had a weaker spread fit. The late tail of the Higgs curve did not match well.

That was fixed by improving the reduced helper calibration model. The current spread fit uses a decaying seed term, which allows the empirical Higgs spike and later decay to be fit much more cleanly.

This matters because:

- the spread fit is now strong enough to act as a baseline
- the benchmark against SIR is clearer
- the threshold results are more believable

At the moment, the spread side is the strongest part of the empirical project.

## What the tau side means now

The tau side is the part that still needs the most care.

Right now the repository compares three tau interpretations:

1. a baseline fixed `phi, psi` choice
2. a Higgs `RE` proxy fit
3. an external Jigsaw-based reference scale

The important point is that the Higgs `RE` fit is now the best same-dataset tau-side experiment in the repo, but it still does not mean `phi` and `psi` are fully and perfectly identified.

So the honest claim is:

- the repo now has a partially grounded tau block
- not a perfectly calibrated toxicity block

## What the mathematical purpose is

The empirical code is only one part of the larger goal.

The bigger mathematical purpose is to turn this from a simulation project into a real applied math paper by proving the core structure of the model:

- nonnegativity of solutions
- a positively invariant region
- a rigorous threshold quantity `R0`
- local and then global stability
- bifurcation behavior at `R0 = 1`

That is the direction most emphasized in `scratch/research.md`, and it is also the part that would matter most for publication beyond the course project.

## What we can honestly claim today

The strongest defensible claim today is:

this repository contains a content-centric IVFS framework with a strong Higgs-based spread calibration, a same-dataset reply-pressure proxy for `tau`, and a clear policy tradeoff between amplification, discussion pressure, and user retention.

The weaker claim that should not be made is:

every parameter in the full model is already cleanly estimated from one perfect dataset.

That is still not true, and the repo should stay honest about that.

## What this repo should become

In the short term, this repo should function as:

- the clean public version of the project
- the reproducible source for the figures and scripts
- the base documentation for GitHub

In the longer term, it should become:

- a math-first paper repository
- with stronger theorem-level results
- broader empirical validation
- and clearer positioning relative to the existing literature

## Bottom line

The purpose of this repository is not just to fit one viral curve.

Its real purpose is to build a mathematically meaningful and empirically grounded framework for studying how platform amplification affects virality, harmful discussion pressure, and user retention, while being honest about what is already established and what still needs to be proved.
