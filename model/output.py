"""
The output contract. Every delivered forecast contains, always:
  1. Likelihood ranges from the posterior (quantiles, P(total > targets)).
  2. Expected error from BACKTESTS on this client's own history. The model
     never grades itself; held-out history does.
  3. Assumptions listed with evidence tiers. The base forecast uses only
     data-backed inputs; assumptions appear as labeled scenario deltas.
This is the liability armor: you deliver a calibrated distribution with a
documented track record, never a number.
"""

import numpy as np

TIERS = {"data": "DATA-BACKED", "evidenced": "EVIDENCED ASSUMPTION", "assumption": "UNVERIFIED ASSUMPTION"}


def render_report(brand, summ_b, backtest=None, targets=(), assumptions=(), return_rate=0.0,
                  revenue_definition="gross, pre-returns"):
    L = []
    q = summ_b["total_q"]; probs = summ_b["probs"]; td = summ_b["total_draws"]
    L.append(f"=== Forecast report: {brand} ===")
    L.append(f"Revenue definition: {revenue_definition}")
    L.append(f"Window: {str(summ_b['dates'][0])[:10]} to {str(summ_b['dates'][-1])[:10]}")
    L.append("\nPeriod total, likelihood ranges (from the posterior):")
    for p, v in zip(probs, q):
        L.append(f"  {int(p * 100):>2}th pct: ${v:,.0f}")
    lo, hi = np.quantile(td, [0.1, 0.9])
    L.append(f"  80% range: ${lo:,.0f} to ${hi:,.0f}")
    for t in targets:
        L.append(f"  P(total > ${t:,.0f}): {float((td > t).mean()):.0%}")
    if return_rate:
        L.append(f"  Net of {return_rate:.0%} expected returns, median: ${np.median(td) * (1 - return_rate):,.0f}")
    if backtest:
        L.append("\nExpected error (from backtests on YOUR history, not model self-grading):")
        L.append(f"  daily median-path MAPE: {backtest['mape']:.0%}")
        L.append(f"  80% interval coverage:  {backtest['coverage']:.0%} (calibrated ~ 80%)")
    if assumptions:
        L.append("\nInputs and assumptions:")
        for tier, desc in assumptions:
            L.append(f"  [{TIERS[tier]}] {desc}")
        L.append("  Base forecast uses data-backed inputs only; assumptions enter as labeled scenarios.")
    return "\n".join(L)
