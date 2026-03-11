# Next Plans

## What still needs to be done for the course project first

- make sure the proposal PDF and `final.tex` stay consistent with the clean repo version
- rerun `code/ivfs_validation.py` once before final submission so the figure in `assets/` definitely matches the current code
- rerun `code/equilibrium_analysis.py` and keep the printed equilibrium numbers nearby for the writeup
- decide exactly how much of the math proof is going into the final course report vs just being mentioned as supporting analysis
- write the final project report in a clean structure: intro, literature, model setup, calibration, simulations, equilibrium analysis, limits, conclusion
- be explicit in the final report that only beta0 and gamma0 are data-fit from Higgs
- keep the claim level honest: threshold + local stability is supported, full toxicity calibration is not there yet
- add the figures properly in the final report and label them clearly
- tighten the explanation of beta_eff and gamma_eff so the reader sees why those are not just random extra symbols
- explain RT once in the report so nobody gets confused later
- if time allows, do one sensitivity table or one extra sensitivity figure for kappa, eta, phi, psi, or lambda_u
- proofread the citations in `final.tex`, especially the last SVFR reference, because that one is still pasted in rough form

## Minimum viable finish for the course project

- clean proposal PDF
- working reproducible code
- one real dataset
- one main results figure
- one benchmark figure
- honest discussion of what is calibrated vs assumed
- short but real equilibrium/stability discussion

## If we want to push this into a paper after the course project (we should, in fact even for the project we should actually just present the paper instead)

- add a real toxicity-labeled dataset to `data/toxicity/`
- calibrate the toxicity side directly instead of leaving it mostly assumption-driven
- prove something stronger mathematically than local stability in a few tested scenarios
- try to get either a stronger uniqueness result, a sharper threshold theorem, or a genuine multi-equilibrium / hysteresis result
- benchmark against more than just SIR, ideally also a toxicity-oriented user model like SEIZ or SEIQR-style work
- test the model on more than one cascade instead of one Higgs spike window
- justify parameter choices more systematically instead of hand-setting most of them
- clean up the notation and write the model section more like an applied math paper and less like a project doc
- write a proper results section with tables, not just figure-driven discussion
- add a limitations section that is direct about homogeneous mixing, deterministic structure, and the coarse toxicity variable
- choose a realistic first venue, like a student journal, workshop, or interdisciplinary computational social science venue, before aiming higher

## What is specifically missing for a serious paper version

- direct toxicity calibration
- stronger theorem-level results
- broader empirical validation
- cleaner literature positioning
- more careful benchmarking
- more polished writing and tables

## Best order to do things

1. finish the course project cleanly
2. lock down the final report and figures
3. add toxicity data
4. strengthen the math
5. expand benchmarking
6. only then think about paper submission

## Bottom line

The course-project path is already real and finishable.

The paper path is possible, but it is a second phase, not something to pretend is already done.
