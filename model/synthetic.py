"""
Synthetic DTC brands with known ground truth, for smoke tests and for
validating that the model recovers parameters it should recover.
Two brands, ~2 years of daily data, BFCM 2024 + 2025, one paid channel.
"""

import numpy as np
import pandas as pd

BFCM = {2024: ("2024-11-22", "2024-12-04"), 2025: ("2025-11-21", "2025-12-03")}


def generate(seed=11, end="2025-12-10"):
    rng = np.random.default_rng(seed)
    brands = [
        dict(name="brand_a", base_orders=140, aov=78.0, trend=0.18, bfcm_lift=1.05, depth={2024: 0.25, 2025: 0.25}),
        dict(name="brand_b", base_orders=60, aov=132.0, trend=0.30, bfcm_lift=1.25, depth={2024: 0.30, 2025: 0.40}),
    ]
    daily_rows, spend_rows, event_rows = [], [], []

    for b in brands:
        dates = pd.date_range("2024-01-05", end)
        t = np.arange(len(dates)) / 365.25
        doy = dates.dayofyear.to_numpy()
        dow = dates.dayofweek.to_numpy()
        dow_eff = np.array([0.0, -0.02, -0.03, 0.0, 0.06, 0.02, -0.04])[dow]
        annual = 0.10 * np.sin(2 * np.pi * doy / 365.25 - 2.4)

        # spend: noisy level with a BFCM push
        spend = np.maximum(rng.normal(900 if b["name"] == "brand_a" else 450, 120, len(dates)), 50.0)
        lam_true, K_true, beta_true = 0.45, 1800.0, 0.35
        ad = np.zeros(len(dates))
        for i in range(len(dates)):
            ad[i] = spend[i] + (lam_true * ad[i - 1] if i else 0.0)
        media = beta_true * ad / (ad + K_true)

        lift = np.zeros(len(dates))
        shape = np.array([0.4, 0.5, 0.7, 0.9, 1.1, 1.4, 2.6, 1.0, 0.8, 1.0, 1.3, 2.2, 1.1])
        shape = shape / shape.mean()
        for yr, (s, e) in BFCM.items():
            win = (dates >= s) & (dates <= e)
            idx = np.where(win)[0]
            if len(idx) == 0:
                continue
            spend[idx] *= 1.9  # brands push spend during BFCM
            m = b["bfcm_lift"] + 1.1 * (b["depth"][yr] - 0.30) + rng.normal(0, 0.08)
            lift[idx] = m * shape[: len(idx)]
            event_rows.append(dict(brand=b["name"], event_id=f"{b['name']}_bfcm_{yr}",
                                   event_type="bfcm", start=s, end=e, depth=b["depth"][yr]))

        # mild pull-forward: soft week before, hangover after
        dip = np.zeros(len(dates))
        for yr, (s, e) in BFCM.items():
            s_, e_ = pd.Timestamp(s), pd.Timestamp(e)
            dip = np.where((dates >= s_ - pd.Timedelta(days=7)) & (dates < s_), -0.06, dip)
            dip = np.where((dates > e_) & (dates <= e_ + pd.Timedelta(days=10)), -0.09, dip)
        log_mu = np.log(b["base_orders"]) + b["trend"] * t + dow_eff + annual + lift + dip + media
        mu = np.exp(log_mu)
        alpha = 35.0
        orders = rng.poisson(rng.gamma(alpha, mu / alpha))

        # pre-discount basket grows during sales; price factor is arithmetic
        basket = np.zeros(len(dates)); price = np.zeros(len(dates))
        for yr, (s, e) in BFCM.items():
            win = (dates >= s) & (dates <= e)
            basket = np.where(win, 0.3 * (b["depth"][yr] - 0.30) + 0.03, basket)
            price = np.where(win, np.log(1 - b["depth"][yr]), price)
        aov = np.exp(rng.normal(np.log(b["aov"]) + basket + price, 0.08))
        revenue = orders * aov

        for i, d in enumerate(dates):
            daily_rows.append(dict(brand=b["name"], date=d, orders=int(orders[i]), revenue=float(revenue[i])))
            spend_rows.append(dict(brand=b["name"], date=d, channel="meta", spend=float(spend[i])))

    return pd.DataFrame(daily_rows), pd.DataFrame(spend_rows), pd.DataFrame(event_rows)
