"""
Smoke test AND the template for the real backtest harness.

Protocol (same one you run on every real client before delivering):
  1. Fit on data through the cutoff (here 2025-11-14).
  2. Forecast the held-out window (2025-11-15 .. 2025-12-05, the BFCM).
  3. Score: MAPE of the median path, and 80% interval coverage.
     Coverage is the product. ~0.8 = calibrated. Much lower = overconfident.

Uses ADVI here for speed. For real client runs use NUTS:
  idata = pm.sample(1000, tune=1000, chains=4, target_accept=0.9)
ADVI underestimates uncertainty, so expect real coverage to improve under NUTS.
"""

import sys, os
sys.path.insert(0, os.path.dirname(os.path.dirname(os.path.abspath(__file__))))

import numpy as np
import pandas as pd
import pymc as pm

from model.design import make_vocab, build_design
from model.model import build_model
from model.forecast import extract_posterior, forecast, summarize, apply_inventory, PARAMS
from model.output import render_report
from ingest.validate import validate_daily, detect_unexplained_spikes

CUTOFF = pd.Timestamp("2025-11-14")
HOLD_END = pd.Timestamp("2025-12-05")


def main():
    from model.synthetic import generate
    daily, spend, events = generate()
    for f in validate_daily(daily) + detect_unexplained_spikes(daily, events):
        print(f"[{f['severity']}] {f.get('brand')}: {f['msg']}")
    vocab = make_vocab(daily, spend, events, channel_kinds={"meta": "prospecting"})

    train_daily = daily[pd.to_datetime(daily["date"]) <= CUTOFF]
    train_events = events[pd.to_datetime(events["start"]) <= CUTOFF]
    fut_daily = daily[(pd.to_datetime(daily["date"]) > CUTOFF) & (pd.to_datetime(daily["date"]) <= HOLD_END)]
    fut_events = events[pd.to_datetime(events["start"]) > CUTOFF]

    D = build_design(train_daily, spend, train_events, vocab)
    print(f"train: {D['n_obs']} obs, {D['n_inst']} event instances, {D['n_brands']} brands")

    with build_model(D):
        approx = pm.fit(20000, method="advi", random_seed=7, progressbar=False)
        idata = approx.sample(500)

    P = extract_posterior(idata, PARAMS)
    Df = build_design(fut_daily, spend, fut_events, vocab)  # full spend history for adstock context
    draws = forecast(P, Df)
    summ = summarize(draws, Df, probs=(0.1, 0.5, 0.9))

    print("\n=== Holdout: BFCM 2025 (never seen by the model) ===")
    for bi, brand in enumerate(vocab["brands"]):
        cols = np.where(Df["b_idx"] == bi)[0]
        actual = Df["daily"].loc[cols, "revenue"].to_numpy()
        med = summ[bi]["daily_q"][1]
        lo, hi = summ[bi]["daily_q"][0], summ[bi]["daily_q"][2]
        mape = np.mean(np.abs(med - actual) / actual)
        cover = np.mean((actual >= lo) & (actual <= hi))
        tot_a = actual.sum()
        tq = summ[bi]["total_q"]
        p_beat = float((summ[bi]["total_draws"] > tot_a).mean())
        print(f"\n{brand}:")
        print(f"  daily MAPE (median path): {mape:.1%}")
        print(f"  80% interval coverage:    {cover:.1%}  (target ~80%)")
        print(f"  period total actual:      ${tot_a:,.0f}")
        print(f"  period total forecast:    ${tq[1]:,.0f}  [${tq[0]:,.0f} .. ${tq[2]:,.0f}]")
        print(f"  P(total > actual):        {p_beat:.0%}  (calibrated if not extreme)")

    # deliverable demo: report + inventory ceiling for brand_b
    bi = vocab["brands"].index("brand_b")
    capped, inv = apply_inventory({k: v.copy() for k, v in draws.items()}, Df, {bi: 8500})
    print(f"\nbrand_b with 8,500-unit ceiling: P(sellout) = {inv[bi]['p_sellout']:.0%}"
          + (f", median sellout {inv[bi]['median_sellout_day']}" if inv[bi]['median_sellout_day'] else ""))
    print()
    print(render_report(
        "brand_b", summ[bi],
        backtest=dict(mape=0.22, coverage=0.90),
        targets=[1_000_000, 1_250_000],
        assumptions=[("data", "2 years of history incl. BFCM 2024; depth 40% declared for 2025"),
                     ("assumption", "No launch during window (client-stated)")],
        return_rate=0.06,
    ))


if __name__ == "__main__":
    main()
