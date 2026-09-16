"""
Upload validation. Runs at intake, findings shown to the client BEFORE any fit.
Bad data discovered here is a conversation; discovered later it is a wrong
forecast with your name on it. Findings carry a severity so the UI can block,
warn, or just note.
"""

import numpy as np
import pandas as pd


def validate_daily(daily):
    f = []
    d = daily.copy()
    d["date"] = pd.to_datetime(d["date"])
    for b, g in d.groupby("brand"):
        g = g.sort_values("date")
        span = pd.date_range(g["date"].min(), g["date"].max())
        missing = span.difference(g["date"])
        if len(missing):
            f.append(dict(severity="warn", brand=b,
                          msg=f"{len(missing)} missing days (e.g. {missing[0].date()}). Gaps corrupt trend and seasonality."))
        if (g["date"].max() - g["date"].min()).days < 365:
            f.append(dict(severity="block", brand=b,
                          msg="Under 1 year of history. Annual seasonality and event priors cannot anchor."))
        if (g["revenue"] < 0).any() or (g["orders"] < 0).any():
            f.append(dict(severity="block", brand=b, msg="Negative values found. Check refunds handling / export type."))
        aov = g["revenue"] / g["orders"].replace(0, np.nan)
        if aov.max() / max(aov.min(), 1e-9) > 20:
            f.append(dict(severity="warn", brand=b,
                          msg="AOV varies >20x across days. Possible currency mix or wholesale orders in the export."))
    return f


def validate_spend(spend, daily):
    f = []
    s = spend.copy(); s["date"] = pd.to_datetime(s["date"])
    d = daily.copy(); d["date"] = pd.to_datetime(d["date"])
    for b in d["brand"].unique():
        sb, db = s[s["brand"] == b], d[d["brand"] == b]
        if sb.empty:
            f.append(dict(severity="warn", brand=b, msg="No ad spend data. Media component will be disabled."))
            continue
        if sb["date"].min() > db["date"].min() + pd.Timedelta(days=60):
            f.append(dict(severity="warn", brand=b,
                          msg="Spend history starts well after orders history. Early baseline absorbs unattributed media."))
    f.append(dict(severity="note", brand=None,
                  msg="Confirm store timezone at intake: Shopify exports UTC, ad platforms report account-local. "
                      "A day off during a spike week poisons the media terms."))
    return f


def detect_unexplained_spikes(daily, events, ratio=2.2, window=28):
    """Spike days not covered by any declared event (+/- 2 days). Each finding
    becomes a question back to the client: 'what happened on this date?'
    Undeclared promos in history are the #1 silent forecast killer.
    ratio=2.2: below ~2x, ordinary day-to-day NegBin dispersion trips this on
    clean data (verified against model/synthetic.py across several seeds)."""
    f = []
    d = daily.copy(); d["date"] = pd.to_datetime(d["date"])
    ev = events.copy()
    if len(ev):
        ev["start"] = pd.to_datetime(ev["start"]); ev["end"] = pd.to_datetime(ev["end"])
    for b, g in d.groupby("brand"):
        g = g.sort_values("date").reset_index(drop=True)
        med = g["revenue"].rolling(window, center=True, min_periods=7).median()
        spikes = g[g["revenue"] > ratio * med]
        for _, row in spikes.iterrows():
            covered = False
            if len(ev):
                e = ev[ev["brand"] == b]
                covered = ((row["date"] >= e["start"] - pd.Timedelta(days=2))
                           & (row["date"] <= e["end"] + pd.Timedelta(days=2))).any()
            if not covered:
                f.append(dict(severity="ask", brand=b, date=str(row["date"].date()),
                              msg=f"Revenue {row['revenue'] / med[row.name]:.1f}x the local norm and no declared event. What happened?"))
    return f
