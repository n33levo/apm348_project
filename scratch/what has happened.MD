# What Has Happened

## Short version

We did **not** throw away the main IVFS model from the proposal.  
The main model in the proposal is still the same idea:

- content states: `I, V, F, S`
- toxicity: `tau`
- active users: `U`
- amplification parameter: `alpha`

What changed was the **calibration step** we use on the Higgs data.

Before, the Higgs calibration curve had a bad tail, so the red line looked wrong after around hour 40.  
To fix that, the reduced calibration model was improved so it could match the Higgs activity curve more honestly.

So there are now really **two layers**:

1. The **main IVFS paper model**  
   This is still the proposal model.

2. The **reduced Higgs fitting model**  
   This is only used to fit the spread curve and estimate `beta0` and `gamma0` better.

That is the main source of confusion.

## What exactly changed

Originally, the reduced Higgs fit used a constant background term.  
That made the fitted red line stay too high in the tail.

Now the Higgs fit uses a **decaying seed term** instead, which means:

- activity can be high early on
- then naturally fade out over time
- the fit matches the real Higgs data much better

Because of that, the fitted spread parameters changed:

- `beta0` changed
- `gamma0` changed
- there are now extra fitting-only parameters for the Higgs calibration curve

These extra fitting-only parameters are **not** new paper-state variables.  
They are just part of the helper model used to fit the data properly.

## Did the paper model change?

My answer is:

**Conceptually, no.**

The proposal model is still the same content-based IVFS system.

**Practically, yes, a little bit in calibration.**

The way we estimate `beta0` and `gamma0` from Higgs is now more advanced than what was written in the proposal.  
So if we write the paper honestly, we should say something like:

> We kept the proposed IVFS system, but improved the reduced calibration model used for fitting the Higgs retweet window.

That is a normal and acceptable change. It is not a complete change of project direction.

## Why the results changed

Since `beta0` and `gamma0` changed, the policy plots changed too.

The biggest effect is this:

- under `alpha = 0.2`, the system is now **subcritical**
- this means `R0 < 1`
- so viral content dies out instead of surviving

That is why the health-first case now goes to zero for `V` and `tau`.

This is not necessarily bad. In fact, it gives a cleaner story:

- engagement-first: more spread, more toxicity, lower long-run user health
- moderate: still spreads, but less extremely
- health-first: suppresses spread below threshold

## Do we need toxicity calibration right now?

This is the part that is still unfinished.

Right now:

- Higgs is good for fitting the **spread/activity** side
- Higgs is **not** giving us a clean toxicity label
- the Jigsaw data gives a toxicity reference, but it is from a different platform/context

So at the moment, toxicity is still better described as **scenario-driven**, not fully calibrated.

That means:

- `beta0` and `gamma0` are empirically calibrated from Higgs
- `phi` and `psi` are still more assumption-based

So yes, the toxicity calibration issue is still real.  
But no, I do **not** think we should panic and switch datasets immediately.

## Should we use a new dataset now?

My opinion is:

**Not yet.**

Right now the smarter move is:

1. keep the Higgs calibration for spread
2. be honest that toxicity is scenario-based
3. add a sensitivity analysis for `phi`
4. write this clearly in the paper

If later we want to make this stronger for publication, then we can look for a single dataset with:

- timestamps
- spread/activity
- toxicity labels or text that can be scored

But changing datasets right now would likely slow everything down a lot.

## What should happen next

I think the next steps should be:

1. Keep the current Higgs calibration, because it is finally fitting well.
2. Add a `phi` sensitivity figure, so we can show the conclusions do not depend on one exact toxicity value.
3. In the paper, clearly separate:
   - the proposed IVFS model
   - the reduced calibration model for Higgs
4. Explain that toxicity is not fully empirically calibrated yet.
5. Only after that, decide whether a new dataset is worth it.

## Final takeaway

So the situation is not:

> everything is broken and we changed the project completely

The real situation is:

> the spread calibration is now much better, the policy story is cleaner, but the toxicity side is still only partially grounded in data

That means we are actually in a better place than before.  
The project is now more honest and more mathematically defensible.

The main thing left is not to rebuild the whole project from scratch.  
The main thing left is to make the toxicity part clearer and more robust.
