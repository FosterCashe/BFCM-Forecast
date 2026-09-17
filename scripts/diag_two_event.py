"""Smoke-backtest protocol with NUTS against the generator's ground truth, for a
given history start. Default start trains on one BFCM (2024); --start 2023-01-05
adds BFCM 2023 while keeping 2024-2025 data identical, so the two runs differ
only by the extra history.

Not yet gate-clean: the two-event configuration's first run (seed 7) had
rhat_max 1.013, which fails the convergence gate. Before quoting any result from
it, run the 3-seed protocol (e.g. --seed 7, 11, 13) and confirm pooled divergences
< 0.5%, rhat_max <= 1.01 and no ESS warnings.

Usage: python scripts/diag_two_event.py [--start 2023-01-05] [--seed 7]
"""
import argparse
import json
import os
import sys

sys.path.insert(0, os.path.dirname(os.path.dirname(os.path.abspath(__file__))))

import arviz as az
import numpy as np
import pandas as pd
import pymc as pm

from model import synthetic as syn
from model.design import DEPTH_CENTER, build_design, make_vocab
from model.forecast import PARAMS, extract_posterior, forecast, summarize
from model.model import build_model
from scripts.diag_truth import BRANDS, central_revenue, check_replica, truth

CUTOFF, HOLD_END = pd.Timestamp("2025-11-14"), pd.Timestamp("2025-12-05")


def main():
    ap = argparse.ArgumentParser()
    ap.add_argument("--start", default=syn.DEFAULT_START)
    ap.add_argument("--seed", type=int, default=7)
    args = ap.parse_args()

    daily, spend, events, eps, comps = truth(start=args.start)
    check_replica(daily, comps)

    dates = pd.to_datetime(daily["date"])
    train_events = events[pd.to_datetime(events["start"]) <= CUTOFF]
    vocab = make_vocab(daily, spend, events, channel_kinds={"meta": "prospecting"})
    D = build_design(daily[dates <= CUTOFF], spend, train_events, vocab)
    with build_model(D) as m:
        idata = pm.sample(draws=1000, tune=1000, chains=4, target_accept=0.95, random_seed=args.seed, progressbar=False)
        free = [rv.name for rv in m.free_RVs]
    rhat = az.rhat(idata)
    out = {
        "start": args.start, "seed": args.seed,
        "train_events": train_events["event_id"].tolist(),
        "divergences": int(idata.sample_stats["diverging"].sum()),
        "rhat_max": max(float(np.nanmax(rhat[n].values)) for n in free),
    }

    Df = build_design(daily[(dates > CUTOFF) & (dates <= HOLD_END)], spend, events[pd.to_datetime(events["start"]) > CUTOFF], vocab)
    draws = forecast(extract_posterior(idata, PARAMS), Df, D)
    summ = summarize(draws, Df, probs=(0.1, 0.5, 0.9))
    post = az.extract(idata)
    for bi, brand in enumerate(vocab["brands"]):
        cols = np.where(Df["b_idx"] == bi)[0]
        actual = Df["daily"].loc[cols, "revenue"].to_numpy()
        lo, med, hi = summ[bi]["daily_q"]
        comp = comps[brand]
        in_window = ((comp["date"] > CUTOFF) & (comp["date"] <= HOLD_END)).to_numpy()
        d25 = BRANDS[brand]["depth"][2025]
        magnitude = (post["event_base"].values + post["event_brand"].values[bi]
                     + post["event_depth_coef"].values * (d25 - DEPTH_CENTER)
                     + post["event_type_effect"].values[0])
        out[brand] = {
            "central_eps0": float(central_revenue(comp, eps_on=False)[in_window].sum()),
            "central_realized_eps": float(central_revenue(comp, eps_on=True)[in_window].sum()),
            "median": float(summ[bi]["total_q"][1]), "p10": float(summ[bi]["total_q"][0]), "p90": float(summ[bi]["total_q"][2]),
            "actual": float(actual.sum()),
            "mape": float(np.mean(np.abs(med - actual) / actual)),
            "coverage": float(np.mean((actual >= lo) & (actual <= hi))),
            "magnitude_2025_mean": float(magnitude.mean()), "magnitude_2025_sd": float(magnitude.std()),
            "magnitude_2025_truth": BRANDS[brand]["bfcm_lift"] + 1.1 * (d25 - 0.30),
            "train_eps": {str(y): round(eps[(brand, y)], 4) for (b, y) in eps if b == brand and pd.Timestamp(syn.BFCM[y][0]) <= CUTOFF},
        }
    print("RESULT " + json.dumps(out))


if __name__ == "__main__":
    main()
