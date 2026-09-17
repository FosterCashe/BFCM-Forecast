# DTC Revenue Forecasting (BFCM wedge, general engine)

Bayesian daily revenue forecasting for DTC brands. Black Friday is the
go-to-market wedge; the engine is general: any promo is a row in the events
table with a type, window, offer and depth.

## Architecture principle

LLMs at the edges, statistics at the core. Deterministic PyMC model in the
middle: reproducible, auditable, backtestable. LLMs do exactly two jobs:
parse messy intake text into structured fields (human-confirmed), and narrate
computed numbers. Never in the math path.

## The output contract (liability armor)

Every delivered forecast contains all three, always:
1. Likelihood ranges from the posterior: quantiles, 80% band, P(total > target).
2. Expected error from BACKTESTS on that client's own held-out history
   (daily MAPE, interval coverage). The model never grades itself.
3. Assumptions with evidence tiers: DATA-BACKED / EVIDENCED / UNVERIFIED.
   The base forecast uses data-backed inputs only; assumptions are labeled
   scenario deltas. Misses stay attributable.
You never deliver a number. You deliver a calibrated distribution with a
documented track record and labeled assumptions. See model/output.py.

## What's here

    model/design.py       dataframes -> design matrices (fit + forecast share it)
    model/model.py        the PyMC model (read first; priors documented inline)
    model/forecast.py     posterior -> forecast paths; launches; inventory ceilings
    model/synthetic.py    fake brands with known ground truth
    model/output.py       the report: ranges + backtested error + assumption tiers
    ingest/validate.py    upload checks incl. undeclared-spike detection
    scripts/smoke_backtest.py  end-to-end: fit, forecast held-out BFCM, score, report
    db/schema.sql         Postgres, tenant_id + RLS-ready, evidence tiers stored
    intake/intake_spec.yaml    every wizard question mapped to a model input

## Quickstart

    pip install -r requirements.txt
    python scripts/smoke_backtest.py

Reference numbers use the smoke protocol with NUTS (4 chains, 1000 tune + 1000
draws, target_accept 0.95, one seed) on the synthetic holdout (BFCM 2025 unseen,
brand_b changes depth 30% -> 40%):

    brand    MAPE  80% cov  window median  80% range        actual  generator central
    brand_a  19%   90%      $669k          $508k-$894k      $644k   $727k
    brand_b  19%   90%      $721k          $497k-$1.02M     $844k   $831k

"Generator central" is model/synthetic.py's noise-free path (no instance noise,
orders = mu). Medians sit 8-13% below it, mainly because with one past event per
brand the model cannot separate a durable event lift from that year's noise (both
brands' 2024 events came in below their durable level), and brand_b's fit
under-captures its 2024 lift by ~0.15 on the log scale (not caused by the BFCM
shape prior: strength 8, 3 and flat priors all fit 1.07-1.10 vs 1.19 observed).
brand_b's 40% depth rides on a prior-only depth coefficient steeper than the
generator's, which partly offsets. The script itself runs ADVI for speed; ADVI
numbers vary run to run, so treat them as smoke only.

## Model summary

Orders (NegBin): brand intercept + trend + day-of-week + annual Fourier
+ event lift + pull-forward dips + media + email/SMS sends.
Event magnitude hierarchy: population base + brand offset + depth coefficient
(centered at ~18% lift per 10 pts of depth) + offer-type effect + list-growth
coefficient + anomaly dummy + instance noise. Per-type daily shape (Dirichlet,
informative for BFCM: ramp, BF spike, weekend sag, CM second peak).
Media: geometric adstock (priors by channel kind: prospecting/search/owned)
into Hill saturation, regime dummy for dated account changes; scenario math
with honest uncertainty, not causal truth.
AOV: pre-discount basket estimated (offers grow baskets), discount price
effect imposed as arithmetic via log(1 - depth x discounted_share). Never
estimate known math.
Revenue = orders x AOV from the posterior. Inventory ceilings truncate demand
and report P(sellout). Launches enter as client-assumption lognormals.

## Non-negotiable workflow per client

1. Run validation, resolve every 'ask' finding (undeclared spikes especially).
2. Backtest on THEIR history before delivering anything. Coverage is the product.
3. NUTS for real runs (pm.sample, 4 chains, target_accept=0.95); ADVI is smoke
   only. Store rhat/divergences + backtest scores in model_runs.diagnostics.
   Convergence gate, run with 3 seeds: passes only when pooled divergences are
   under 0.5% of draws, rhat_max <= 1.01, and there are no effective-sample-size
   warnings. Divergence counts swing widely between seeds; never gate on one.
4. pm.sample_prior_predictive before every first fit: eyeball implied dollars.
5. Every client assumption labeled in the output. No silent boosts, ever.

## Upgrade path, in order of value

1. New vs returning split (schema ready). Aligns model with DTC mechanics;
   creative/media effects then act only on the series they can affect.
2. Margin layer (cogs_pct in intake): revenue AND contribution by offer.
3. Cross-client partial pooling at 3+ brands: the compounding moat.
4. Live pacing during the event: actuals vs band, daily. The retention feature.
5. Resist: GP trends, time-varying saturation, macro covariates, intraday.

## Build sequence

Now-Nov: manual engagements ON this pipeline (you are the UI). Dec: wizard from
intake spec + forecast-vs-actuals postmortems. Spring: self-serve beta. CSV
upload until product-market fit; OAuth is a v2 decision.
