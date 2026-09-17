# How the forecast earns trust

Every delivered forecast is a distribution, a backtested error rate, and a labeled list of inputs (README, "The output contract"). This document describes the checks behind each of those, and what they found. All numbers below come from the synthetic validation brands in `model/synthetic.py` (two brands, BFCM 2024 in training, BFCM 2025 held out), not from client data.

## 1. The sampler has to converge before a number is quoted

Client runs use NUTS: 4 chains, 1,000 tuning + 1,000 draws, `target_accept` 0.95. A configuration passes the **convergence gate** only when, across **3 seeds**:

- pooled divergences are under 0.5% of draws,
- the worst rhat over all sampled variables is at most 1.01,
- PyMC raises no effective-sample-size warnings (it warns below 100 effective draws per chain).

Three seeds, because divergence counts are noisy: identical settings produced 80, 52 and 4 divergences on three seeds. A single run cannot tell a fix from luck.

**What is automated today.** Each NUTS run stores the worst rhat per variable and the divergence count in `model_runs.diagnostics_json`. API run responses carry `rhat_flag` and `rhat_flagged_params`, set when any variable's rhat exceeds 1.01 or cannot be computed. The cross-seed gate and the ESS check are run by hand from saved posterior draws (`persist_draws`); the results page does not yet display the flag.

**The gate changed the model.** The first NUTS runs failed badly: 282 and 296 divergences out of 4,000 draws, rhat up to 1.045. Rewriting the hierarchical offsets in non-centered form cut that to 77 and 115. Pooled across seeds the remaining divergences clustered where the brand intercept and media coefficient traded off (posterior correlation -0.85): adstocked spend barely varies day to day, so the media term behaved like a second intercept. Centering media on its training-window mean removed the ridge (correlation +0.01 to +0.07) and brought both brands through the gate: 0.11% and 0.12% pooled divergences, worst rhat 1.005, lowest effective sample size 1,150 (PyMC warns below 400 for 4 chains).

The same seed does not reproduce bit-for-bit across processes, so runs that need later inspection save their draws rather than relying on a rerun.

## 2. Quoted numbers are NUTS-only

ADVI (a fast approximation) is used only for local development. On the earlier, correlated form of the model, ADVI's window-total medians sat 10% (brand_a, $707k vs $645k) and 14% (brand_b, $795k vs $696k) above NUTS, and its 80% upper bound for brand_b was $1.50M against NUTS's $1.01M. After reparameterization the gap shrank to 3-5%, but ADVI still moves between runs. An earlier README example ($809k forecast against $806k actual) matched ADVI's behavior on that model; NUTS put the same median near $690k at every model version tested.

## 3. Validation against a world with known answers

The synthetic generator encodes every effect explicitly: baseline trend, weekday and annual seasonality, event lift with a known depth response, pull-forward dips, adstocked media, and noise. That makes it possible to compare the model with the truth, not only with realized data.

Backtest on the held-out BFCM 2025 window (NUTS, one seed):

| | Daily MAPE | 80% coverage | Median window total | 80% range | Actual | Generator's noise-free path |
|---|---|---|---|---|---|---|
| brand_a | 19% | 90% | $669k | $508k-$894k | $644k | $727k |
| brand_b | 19% | 90% | $721k | $497k-$1.02M | $844k | $831k |

Actuals fall inside the 80% ranges. The medians sit 8-13% below the noise-free path; section 5 explains why.

**Does the model credit lift to the right causes?** `scripts/diag_components.py` breaks log orders around BFCM 2024 into components and compares each with the generator's (change versus adjacent non-event weeks):

| Component | brand_a model / truth | brand_b model / truth |
|---|---|---|
| Event lift | +0.939 / +0.923 | +1.082 / +1.195 |
| Media | +0.035 / +0.055 | +0.027 / +0.062 |
| Annual seasonality | -0.006 / +0.001 | -0.003 / +0.001 |
| Observed data (all components) | +1.013 | +1.124 |

The seasonal term absorbs none of the event. brand_b's event lift sits 0.11 below the generator's mean, but the observed data sit 0.13 below it (day-level demand noise that year), and the model's components sum to within 0.02 of the data: it tracks what happened, not a misfit. Two real misses remain: media is under-credited by 0.02-0.035, and brand_a's pull-forward dip is estimated at -0.024 against a true -0.078.

The same exercise found a bug in the generator itself (orders ignored the BFCM spend push the model could see), which was fixed before the numbers above.

## 4. Disclosing what the data could not teach

Some coefficients cannot be learned from a given brand's history. After every fit the worker compares the posterior of the depth, list-growth and event-type coefficients with its prior: posterior sd divided by prior sd, and how far the mean moved in prior-sd units. A coefficient is marked **prior-driven** when the sd ratio exceeds 0.9 and the shift is under 0.2.

The depth coefficient was prior-driven in all six validation fits (sd ratio 1.00-1.02, shift at most 0.04). The history held one past promotion per brand, at a single depth, so there was nothing to learn from. That matters for brand_b, which plans 40% in 2025 after 30% in 2024: the response to the extra depth comes entirely from the prior (slope 1.8; the generator's true slope is 1.1). When a coefficient is prior-driven and a forecast event's depth differs from every training depth, the results page adds an **evidenced assumption**: "Offer-depth response: from cross-brand and expert priors, not yet learned from your own history."

## 5. One past event: shrinkage is the honest answer

With a single past BFCM, the model cannot tell a brand's durable event lift from that year's noise. The posterior shows it directly: event base and instance noise correlate at -0.94 to -0.95. Rather than read one year's result as permanent, the model shrinks toward its prior and widens the range. In validation that year's noise ran low for both brands, so medians landed 8-13% under the noise-free path while the actuals stayed inside the 80% ranges. A model that ignored this ambiguity would be more confident and more often wrong.

A second observation resolves much of it. Adding BFCM 2023 to the history, with 2024-2025 data held identical (`scripts/diag_two_event.py`):

| | Event-lift sd, 1 event → 2 events | Median vs noise-free path | 80% range |
|---|---|---|---|
| brand_a | 0.161 → 0.073 | -8.0% → -7.2% | $502k-$898k → $588k-$785k |
| brand_b | 0.175 → 0.095 | -12.6% → -7.6% | $509k-$1.05M → $624k-$940k |

Uncertainty about the event lift roughly halves. This result is **provisional**: the two-event run is a single seed with worst rhat 1.013, which fails the gate.

This is the reason to ask for two or more years of history. What counts is past BFCMs in the training window, not calendar length: 23 months ending before BFCM 2025 contains only one. The intake asks for 2+ years; validation currently blocks only histories under one year.

## 6. Evidence tiers: what is in the numbers

Every intake answer is stored with a tier (`data`, `evidenced`, `assumption`) and shown with that label.

- **Data-backed** inputs build the base forecast: order and revenue history, ad spend, declared past and planned promotions.
- **Evidenced assumptions** are in the numbers but not learned from this brand, such as the prior-driven depth response above. The page says so.
- **Unverified assumptions** (client statements such as "no product launch during the window") are listed but not applied. A launch component exists in `model/forecast.py` but is not wired into client runs.

The page also states the revenue definition, derives the window total and P(total > target) from each draw's window total rather than by adding daily quantiles (adding them put brand_b's median 12% too low and the 80% range 22% too wide), and reports expected error from the held-out backtest, with a note on what coverage above or below 80% means.

## Known gaps

- The 3-seed gate and ESS check are manual; the page does not show `rhat_flag`.
- Media is under-credited and brand_a's pull-forward dip under-estimated in validation.
- The two-event result has not passed the gate.
- Each tenant is fit as a single brand; cross-brand pooling is not built.
- Every figure here is from synthetic data. The first client backtests will be the real test.
