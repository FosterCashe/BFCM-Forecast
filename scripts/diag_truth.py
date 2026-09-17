"""Ground truth for diagnostics: model/synthetic.py's generator, decomposed.

truth() reruns generate() while recording each brand-year's event noise, and
rebuilds the generator's log-scale components per day. check_replica() fails
if the rebuild has drifted from generate() (mean orders/mu near 1, log-AOV
residuals centered), so edit this file whenever the generator's math changes.
"""
import numpy as np
import pandas as pd

from model import synthetic as syn

# Mirrors the brand table inside synthetic.generate().
BRANDS = {
    "brand_a": dict(base_orders=140, aov=78.0, trend=0.18, bfcm_lift=1.05, depth={2023: 0.25, 2024: 0.25, 2025: 0.25}),
    "brand_b": dict(base_orders=60, aov=132.0, trend=0.30, bfcm_lift=1.25, depth={2023: 0.30, 2024: 0.30, 2025: 0.40}),
}
SHAPE = np.array([0.4, 0.5, 0.7, 0.9, 1.1, 1.4, 2.6, 1.0, 0.8, 1.0, 1.3, 2.2, 1.1])
SHAPE = SHAPE / SHAPE.mean()
EPS_SD = 0.08


class _Recorder:
    """Wraps a numpy Generator and records scalar N(0, EPS_SD) draws (the event noise)."""

    def __init__(self, gen, sink):
        self._gen, self._sink = gen, sink

    def normal(self, loc=0.0, scale=1.0, size=None):
        out = self._gen.normal(loc, scale, size)
        if np.isscalar(loc) and loc == 0 and scale == EPS_SD and size is None:
            self._sink.append(float(out))
        return out

    def __getattr__(self, name):
        return getattr(self._gen, name)


def truth(start=syn.DEFAULT_START, end="2025-12-10", seed=11):
    """Returns daily, spend, events (exactly as generate() returns them), the
    realized event noise {(brand, year): eps}, and per-brand component frames."""
    draws = []
    real = np.random.default_rng
    np.random.default_rng = lambda s=None: _Recorder(real(s), draws)
    try:
        daily, spend, events = syn.generate(seed=seed, end=end, start=start)
    finally:
        np.random.default_rng = real
    plain = syn.generate(seed=seed, end=end, start=start)
    assert plain[0].equals(daily) and plain[1].equals(spend) and plain[2].equals(events), "recorded rerun diverged"

    keys = [(r.brand, int(r.start[:4])) for r in events.itertuples()]  # generate() draws noise in this order
    assert len(keys) == len(draws), (keys, draws)
    eps = dict(zip(keys, draws))
    comps = {brand: _components(brand, spend, eps, start, end) for brand in BRANDS}
    return daily, spend, events, eps, comps


def _components(brand, spend, eps, start, end):
    b = BRANDS[brand]
    anchor = pd.Timestamp(syn.DEFAULT_START)
    dates = pd.date_range(start, end)
    t = np.asarray((dates - anchor).days) / 365.25
    sp = spend[spend.brand == brand].sort_values("date")["spend"].to_numpy()
    ad = np.zeros(len(dates))
    for i in range(len(dates)):
        ad[i] = sp[i] + (0.45 * ad[i - 1] if i and dates[i] != anchor else 0.0)
    lift, lift_eps0, dip, basket, price = (np.zeros(len(dates)) for _ in range(5))
    for yr, (s, e) in syn.BFCM.items():
        idx = np.where((dates >= s) & (dates <= e))[0]
        if len(idx) == 0:
            continue
        durable = b["bfcm_lift"] + 1.1 * (b["depth"][yr] - 0.30)
        lift[idx] = (durable + eps[(brand, yr)]) * SHAPE[: len(idx)]
        lift_eps0[idx] = durable * SHAPE[: len(idx)]
    for s, e in syn.BFCM.values():
        s_, e_ = pd.Timestamp(s), pd.Timestamp(e)
        dip = np.where((dates >= s_ - pd.Timedelta(days=7)) & (dates < s_), -0.06, dip)
        dip = np.where((dates > e_) & (dates <= e_ + pd.Timedelta(days=10)), -0.09, dip)
    for yr, (s, e) in syn.BFCM.items():
        win = (dates >= s) & (dates <= e)
        basket = np.where(win, 0.3 * (b["depth"][yr] - 0.30) + 0.03, basket)
        price = np.where(win, np.log(1 - b["depth"][yr]), price)
    return pd.DataFrame({
        "date": dates,
        "trend_intercept": np.log(b["base_orders"]) + b["trend"] * t,
        "dow": np.array([0.0, -0.02, -0.03, 0.0, 0.06, 0.02, -0.04])[dates.dayofweek.to_numpy()],
        "seasonal": 0.10 * np.sin(2 * np.pi * dates.dayofyear.to_numpy() / 365.25 - 2.4),
        "event": lift,
        "event_eps0": lift_eps0,
        "pull_forward": dip,
        "media": 0.35 * ad / (ad + 1800.0),
        "log_aov": np.log(b["aov"]) + basket + price,
    })


def central_revenue(comp, eps_on=True):
    """Noise-free daily revenue: orders = mu, AOV = exp(mean log AOV)."""
    event = comp["event"] if eps_on else comp["event_eps0"]
    log_mu = comp["trend_intercept"] + comp["dow"] + comp["seasonal"] + event + comp["pull_forward"] + comp["media"]
    return np.exp(log_mu + comp["log_aov"])


def check_replica(daily, comps):
    for brand, comp in comps.items():
        obs = daily[daily.brand == brand].sort_values("date").reset_index(drop=True)
        mu = np.exp(comp["trend_intercept"] + comp["dow"] + comp["seasonal"] + comp["event"] + comp["pull_forward"] + comp["media"])
        ratio = float((obs["orders"] / mu).mean())
        resid = np.log(obs["revenue"] / obs["orders"]) - comp["log_aov"]
        print(f"replica check {brand}: mean(orders/mu) {ratio:.4f}, log-AOV residual mean {resid.mean():+.4f} sd {resid.std():.4f}")
        assert abs(ratio - 1) < 0.03 and abs(resid.mean()) < 0.01, f"generator replica drifted for {brand}"
