# Next Plans

## Current position

### 1. Stronger Higgs spread calibration

- Technical:
  - the reduced spread calibration now fits the Higgs retweet window using a decaying seed term instead of the older constant background term
  - `beta0` and `gamma0` are now estimated from a much better empirical fit
  - the late-tail mismatch is much smaller than before

- What I mean by that:
  - the spread side of the project is now in good enough shape to stop changing every week
  - this should be treated as the baseline calibration for the next stage of the work

### 2. Tau side is improved but not fully solved

- Technical:
  - `tau` is not directly observed in the Higgs file
  - the repo now uses Higgs `RE` events as a same-dataset proxy for harmful discussion pressure
  - the code compares three tau-side setups:
    - fixed baseline `phi, psi`
    - Higgs `RE` proxy fit
    - external Jigsaw reference scale
  - the Higgs `RE` fit is better than the old baseline, but `phi` and `psi` are still not cleanly identified separately

- What I mean by that:
  - the repo is now more honest and more coherent
  - but we still cannot claim that every tau-side parameter is perfectly calibrated from one dataset

### 3. Main gap is now mathematics, not plotting

- Technical:
  - the code already has calibration, simulation, continuation, sensitivity, and numerical local stability checks
  - what is missing are theorem-level results that would support an applied math paper

- What I mean by that:
  - the next stage should mainly be proofs, not more cosmetic changes to figures

## Main math tasks

### 1. Prove nonnegativity of solutions

- Technical:
  - show that if `I(0), V(0), F(0), S(0), tau(0), U(0) >= 0`, then all six states remain nonnegative for all `t >= 0`
  - this should be done by checking the vector field on each coordinate boundary
  - in practice, check the sign of:
    - `I'(t)` when `I = 0`
    - `V'(t)` when `V = 0`
    - `F'(t)` when `F = 0`
    - `S'(t)` when `S = 0`
    - `tau'(t)` when `tau = 0`
    - `U'(t)` when `U = 0`

- What I mean by that:
  - we need to prove the model never produces impossible negative values
  - this is one of the first things reviewers expect in an applied ODE paper

### 2. Prove existence of a positively invariant bounded region

- Technical:
  - derive bounds showing that the IVFS system has a compact absorbing set
  - likely strategy:
    - derive an upper bound for the content variables, for example some combination like `I + V + F + S`
    - derive a separate upper bound for `U`
    - use the tau equation together with boundedness of `V`
  - conclude that trajectories eventually enter and remain in a bounded feasible region

- What I mean by that:
  - this proves the system stays physically meaningful and does not blow up
  - it is the natural follow-up after proving nonnegativity

### 3. Derive `R0` rigorously

- Technical:
  - use the **van den Driessche-Watmough next-generation matrix method**
  - identify the transmission-relevant subsystem at the disease-free equilibrium
  - in particular, treat the content-spread classes carefully because the model has several infectious-like states, especially `I`, `V`, and `F`
  - decompose the subsystem into `F(x)` and `V(x)`
  - compute the spectral radius
    - `R0 = rho(F V^{-1})`
  - compare the rigorous formula with the threshold expression already used in the code
  - check whether the hand-derived threshold misses any coupling or cross-terms

- What I mean by that:
  - the threshold story should come from a standard theorem-based method, not only from a hand-derived expression
  - once this is done, the paper’s threshold results become much more defensible

### 4. Strengthen local stability into theorem form

- Technical:
  - the code already computes Jacobian eigenvalues numerically at the disease-free equilibrium and positive equilibrium
  - this now needs to be converted into actual propositions or theorems with assumptions stated clearly
  - derive the Jacobian symbolically if possible, or at least present its structure analytically before the numerical confirmation

- What I mean by that:
  - we already have the numerical check
  - now we need to write it properly as mathematics, not just as code output

### 5. Prove global stability

- Technical:
  - write this as a **Lyapunov function proof**, not just a numerical eigenvalue check
  - for the disease-free equilibrium when `R0 < 1`, construct a Lyapunov function showing global asymptotic stability
  - the most natural first attempt is to use the viral state
    - `L_DFE(t) = V(t)`
    - or a closely related positive linear combination
  - for the positive equilibrium when `R0 > 1`, attempt a **Korobeinikov-type log-sum Lyapunov function** of the form
    - `L = sum_i c_i (x_i - x_i^* - x_i^* ln(x_i / x_i^*))`
  - compute `dL/dt` along solutions and show it is nonpositive under the right conditions
  - if the full six-state proof is too hard at first, first prove the disease-free result cleanly and then attempt the endemic result

- What I mean by that:
  - this is the point where the project starts looking like a real applied math paper
  - even one clean global stability theorem would strengthen the paper a lot

### 6. Classify the bifurcation at `R0 = 1`

- Technical:
  - use **center manifold analysis** near the threshold parameter value
  - determine whether the transition at `R0 = 1` is a **forward transcritical bifurcation** or a **backward bifurcation**
  - if possible, compute the sign of the center-manifold coefficients that determine the type
  - connect the theorem directly to the continuation plots, where `tau*` moves from zero to positive as `alpha` crosses the threshold
  - if the bifurcation turns out to be backward, state the **bistability** interpretation clearly

- What I mean by that:
  - the continuation plots suggest a threshold transition, but the paper should prove what kind it is
  - a backward bifurcation would be especially strong because it would suggest hysteresis or **bistability**

### 7. Compute analytic sensitivity indices

- Technical:
  - compute **normalized forward sensitivity indices** of `R0`
  - for each parameter `p`, compute
    - `Upsilon_p^{R0} = (dR0/dp) * (p/R0)`
  - collect them in a compact table and interpret sign and magnitude
  - compare the analytic sensitivity table against the numerical sensitivity already produced by the code

- What I mean by that:
  - we already have numerical sensitivity plots and tables
  - this adds the standard analytic version that many epidemic-model papers include

## Extra math emphasis from the research roadmap

### 1. Global stability should be treated as more important than local checks

- Technical:
  - the current code already checks Jacobian eigenvalues at selected equilibria
  - that is only local stability information
  - the research direction is to go beyond this and prove global asymptotic stability, at least in the disease-free case, using **Lyapunov methods**

- What I mean by that:
  - local Jacobian checks are useful, but by themselves they are not enough for a stronger applied math paper
  - this should be treated as one of the highest-priority theorem tasks

### 2. The bifurcation result could become one of the most publishable parts

- Technical:
  - if the threshold at `R0 = 1` is a clean forward transcritical bifurcation, that is already a complete theorem result
  - if it is a backward bifurcation, then the model may admit **bistability** or hysteresis
  - that would mean the platform could remain trapped in a harmful state even after amplification is reduced below the naive threshold

- What I mean by that:
  - this is not just a side calculation
  - it could become one of the strongest mathematical contributions in the whole paper

### 3. Well-posedness is the foundation for everything else

- Technical:
  - nonnegativity and the invariant region should appear before the main stability theorems
  - the later Lyapunov and bifurcation arguments rest on the assumption that the system evolves inside a meaningful state space

- What I mean by that:
  - this part may look basic, but it is not optional
  - it is the groundwork that makes the rest of the paper look mathematically complete

## Main empirical tasks

### 1. Keep the current Higgs spread calibration fixed

- Technical:
  - use the current reduced IVF Higgs fit as the baseline empirical estimate for `beta0` and `gamma0`
  - do not keep changing the spread calibration while working on the theorem side or the tau side
  - treat the extra calibration helper parameters only as nuisance fitting parameters, not new core IVFS state variables

- What I mean by that:
  - the spread side is good enough for now
  - we should stop moving the target and build the next layer on top of it

### 2. Keep the tau interpretation honest

- Technical:
  - interpret `tau` as aggregate harmful-discussion pressure, outrage pressure, or reply pressure
  - do not present it as directly observed text toxicity from Higgs
  - keep Higgs `RE` as the same-dataset proxy and keep Jigsaw or Ruddit as external reference datasets only

- What I mean by that:
  - the project is really about rage-bait and harmful engagement dynamics
  - the wording in the paper and repo should match what the data can actually support

### 3. Multi-cascade validation later

- Technical:
  - fit the reduced spread model to at least 3 distinct cascades beyond the current Higgs spike
  - compare fitted `beta0`, `gamma0`, threshold behavior, and qualitative policy ordering across cascades
  - use actual diffusion/cascade datasets, not pure toxicity datasets, for this stage

- What I mean by that:
  - right now the spread story is based on one event
  - later we need more than one cascade so the results do not look too event-specific

### 4. Keep external toxicity datasets secondary

- Technical:
  - Jigsaw and Ruddit can still be used to summarize external toxicity levels or imply a scale for `phi/psi`
  - they should not replace Higgs as the main dataset for the repo
  - they should not be described as same-system spread calibration data

- What I mean by that:
  - these datasets are still useful
  - but they are there to support interpretation, not to redefine the whole project

## What should be done first

### Stage 1. Lock the current project state

- Technical:
  - keep the current Higgs spread fit
  - keep the current tau comparison figures
  - keep the documentation honest about what is calibrated and what is still proxy-based

- What I mean by that:
  - do not spend the next phase redoing the same empirical work again

### Stage 2. Finish the minimum serious theorem package

- Technical:
  - prove nonnegativity
  - prove boundedness and the invariant region
  - derive `R0` rigorously
  - restate the equilibrium section around those results

- What I mean by that:
  - this is the most realistic next mathematical step
  - it is also the step that most directly upgrades the project from strong coursework to serious paper groundwork

### Stage 3. Push into deeper stability theory

- Technical:
  - prove global stability for the disease-free equilibrium
  - attempt global stability for the positive equilibrium
  - classify the bifurcation at `R0 = 1`

- What I mean by that:
  - this is the stage where the project starts to gain real theorem-level originality

### Stage 4. Add broader empirical support

- Technical:
  - perform multi-cascade spread validation
  - compare whether the same threshold and policy conclusions survive across multiple events

- What I mean by that:
  - this is the answer to the "one-cascade only" limitation
  - but it should come after the core theorem package is stronger

### Stage 5. Only after that, consider advanced extensions

- Technical:
  - optimal control with time-varying `alpha(t)`
  - network-aware mean-field extension
  - broader benchmarking against richer social contagion models

- What I mean by that:
  - these are valuable, but they are not the immediate next move
  - the project first needs a stronger core before adding harder extensions

## What should not happen next

### 1. Do not replace Higgs as the main dataset right now

- Technical:
  - replacing the main dataset now would force another full recalibration and delay the theorem work

- What I mean by that:
  - we already have a workable empirical base
  - use it instead of resetting the project

### 2. Do not overclaim the tau side

- Technical:
  - `phi` and `psi` are still not cleanly identified as if they came from one fully observed system

- What I mean by that:
  - the paper should present the tau side as partially grounded, not perfectly solved

### 3. Do not spend the next phase mostly making new plots

- Technical:
  - the code already has enough figures to support the current state of the model

- What I mean by that:
  - the real bottleneck is mathematics now, not visualization

## Bottom line

### Main point

- Technical:
  - the project now has a strong enough empirical baseline to support moving into theorem work

- What I mean by that:
  - the next stage should mainly be proving the model properly, then later validating the spread side on multiple cascades
