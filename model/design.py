"""
Design matrix construction. One function builds everything the model needs
from tidy dataframes, so fitting and forecasting share identical logic.

Inputs:
  daily:  [brand, date, orders, revenue]
  spend:  [brand, date, channel, spend]
  events: [brand, event_id, event_type, start, end, depth]
          optional cols: discounted_share (default 1.0), list_ratio (this
          year's list size / last comparable period, default 1.0),
          anomaly (bool, default False)
  sends:   optional [brand, date, audience]  (email/SMS campaign calendar)
  regimes: optional [brand, start]           (media efficiency regime change,
           e.g. "UGC creative scaled from June 1")

Event component is GENERIC: BFCM, July 4th, a flash sale are all just rows.
Every promo in history MUST be declared or the baseline misreads the spikes;
see ingest/validate.detect_unexplained_spikes.
"""

import numpy as np
import pandas as pd

L_ADSTOCK = 14
FOURIER_K = 3
DEPTH_CENTER = 0.30
PF_PRE, PF_POST = 7, 10  # pull-forward dip window around events

# adstock decay Beta(a, b) priors by channel kind
CHANNEL_KIND_PRIORS = {
    "prospecting": (4.0, 3.0),   # Meta/TikTok: 1-3 day half-life typical
    "search": (1.5, 6.0),        # branded search: near-immediate
    "owned": (1.2, 8.0),         # email/SMS clicks: immediate
}


def make_vocab(daily, spend, events, channel_kinds=None):
    channels = sorted(spend["channel"].unique().tolist())
    kinds = {c: (channel_kinds or {}).get(c, "prospecting") for c in channels}
    return {
        "brands": sorted(daily["brand"].unique().tolist()),
        "channels": channels,
        "channel_kinds": kinds,
        "event_types": sorted(events["event_type"].unique().tolist()),
        "epoch": pd.to_datetime(daily["date"]).min(),
    }


def _lag_matrix(x, L):
    n = len(x)
    M = np.zeros((n, L + 1))
    for l in range(L + 1):
        M[l:, l] = x[: n - l]
    return M


def _pf_weights():
    pre = np.linspace(0.3, 1.0, PF_PRE)
    post = np.linspace(1.0, 0.2, PF_POST)
    w = np.concatenate([pre, post])
    return pre / w.mean(), post / w.mean()


def _shape_conc(etype, w_max):
    """Dirichlet concentration for the event daily shape. BFCM gets an
    informative prior encoding the known curve (ramp, BF spike, weekend sag,
    CM second peak); unknown types get a mildly regularizing flat prior."""
    if "bfcm" in etype.lower() or "black" in etype.lower():
        curve = np.array([1, 1, 1.2, 1.5, 2, 3, 6, 2, 1.5, 1.8, 2.5, 5, 2], float)
        c = np.interp(np.linspace(0, len(curve) - 1, w_max), np.arange(len(curve)), curve)
        return c / c.mean() * 8.0
    return np.full(w_max, 2.0)


def build_design(daily, spend, events, vocab, sends=None, regimes=None):
    brands, channels, etypes = vocab["brands"], vocab["channels"], vocab["event_types"]
    bmap = {b: i for i, b in enumerate(brands)}
    tmap = {t: i for i, t in enumerate(etypes)}

    daily = daily.copy()
    daily["date"] = pd.to_datetime(daily["date"])
    daily = daily.sort_values(["brand", "date"]).reset_index(drop=True)
    spend = spend.copy()
    spend["date"] = pd.to_datetime(spend["date"])

    n_obs = len(daily)
    b_idx = daily["brand"].map(bmap).to_numpy()

    # --- baseline features -------------------------------------------------
    t_years = ((daily["date"] - vocab["epoch"]).dt.days / 365.25).to_numpy()
    dow = (
        pd.get_dummies(daily["date"].dt.dayofweek)
        .reindex(columns=range(7), fill_value=0)
        .to_numpy().astype(float)[:, 1:]
    )
    doy = daily["date"].dt.dayofyear.to_numpy()
    four = np.column_stack(
        [f(2 * np.pi * k * doy / 365.25) for k in range(1, FOURIER_K + 1) for f in (np.sin, np.cos)]
    )

    # --- media lag matrices (full spend history => correct adstock context) -
    X_lag = np.zeros((n_obs, len(channels), L_ADSTOCK + 1))
    for bi, b in enumerate(brands):
        sb = spend[spend["brand"] == b]
        if sb.empty:
            continue
        rows = np.where(b_idx == bi)[0]
        full_dates = pd.date_range(sb["date"].min(), daily.loc[rows, "date"].max())
        pos = {d: j for j, d in enumerate(full_dates)}
        for ci, c in enumerate(channels):
            s = (
                sb[sb["channel"] == c].set_index("date")["spend"]
                .reindex(full_dates, fill_value=0.0).to_numpy()
            )
            Ml = _lag_matrix(s, L_ADSTOCK)
            for r in rows:
                j = pos.get(daily.at[r, "date"])
                if j is not None:
                    X_lag[r, ci, :] = Ml[j]
    pos_spend = spend.loc[spend["spend"] > 0].groupby("channel")["spend"].median()
    spend_scale = np.array([2.0 * pos_spend.get(c, 100.0) for c in channels])
    ab = np.array([CHANNEL_KIND_PRIORS[vocab["channel_kinds"][c]] for c in channels])
    adstock_a, adstock_b = ab[:, 0], ab[:, 1]

    # --- optional covariates ----------------------------------------------
    sends_vec = np.zeros(n_obs)
    if sends is not None and len(sends):
        s = sends.copy()
        s["date"] = pd.to_datetime(s["date"])
        mx = s.groupby("brand")["audience"].transform("max").replace(0, 1)
        s["norm"] = s["audience"] / mx
        lk = {(r.brand, r.date): r.norm for r in s.itertuples()}
        for r in range(n_obs):
            sends_vec[r] = lk.get((daily.at[r, "brand"], daily.at[r, "date"]), 0.0)

    regime_vec = np.zeros(n_obs)
    if regimes is not None and len(regimes):
        rg = regimes.copy()
        rg["start"] = pd.to_datetime(rg["start"])
        for row in rg.itertuples():
            regime_vec[(daily["brand"] == row.brand).to_numpy()
                       & (daily["date"] >= row.start).to_numpy()] = 1.0

    # --- events ------------------------------------------------------------
    ev = events.copy()
    ev["start"] = pd.to_datetime(ev["start"])
    ev["end"] = pd.to_datetime(ev["end"])
    for col, default in [("discounted_share", 1.0), ("list_ratio", 1.0), ("anomaly", False)]:
        if col not in ev.columns:
            ev[col] = default
    date_lookup = {(daily.at[r, "brand"], daily.at[r, "date"]): r for r in range(n_obs)}

    inst_brand, inst_type, inst_depth, inst_share, inst_loglist, inst_anom, inst_w = [], [], [], [], [], [], []
    ev_obs, ev_inst, ev_pos = [], [], []
    pf_obs, pf_inst, pf_w = [], [], []
    pre_w, post_w = _pf_weights()
    log_price_factor = np.zeros(n_obs)

    for k, (_, row) in enumerate(ev.iterrows()):
        days = pd.date_range(row["start"], row["end"])
        inst_brand.append(bmap[row["brand"]])
        inst_type.append(tmap[row["event_type"]])
        inst_depth.append(float(row["depth"]))
        inst_share.append(float(row["discounted_share"]))
        inst_loglist.append(float(np.log(max(row["list_ratio"], 1e-3))))
        inst_anom.append(1.0 if bool(row["anomaly"]) else 0.0)
        inst_w.append(len(days))
        for p, d in enumerate(days):
            r = date_lookup.get((row["brand"], d))
            if r is not None:
                ev_obs.append(r); ev_inst.append(k); ev_pos.append(p)
                # deterministic price arithmetic on AOV (sitewide % math):
                log_price_factor[r] += np.log(max(1.0 - row["depth"] * row["discounted_share"], 0.05))
        for j, d in enumerate(pd.date_range(row["start"] - pd.Timedelta(days=PF_PRE), row["start"] - pd.Timedelta(days=1))):
            r = date_lookup.get((row["brand"], d))
            if r is not None:
                pf_obs.append(r); pf_inst.append(k); pf_w.append(pre_w[j])
        for j, d in enumerate(pd.date_range(row["end"] + pd.Timedelta(days=1), row["end"] + pd.Timedelta(days=PF_POST))):
            r = date_lookup.get((row["brand"], d))
            if r is not None:
                pf_obs.append(r); pf_inst.append(k); pf_w.append(post_w[j])

    n_inst = len(inst_brand)
    w_max = max(inst_w) if n_inst else 1
    wlen = np.ones(len(etypes))
    for k in range(n_inst):
        wlen[inst_type[k]] = max(wlen[inst_type[k]], inst_w[k])
    shape_conc = np.vstack([_shape_conc(t, w_max) for t in etypes]) if etypes else np.ones((1, 1))

    # --- targets -----------------------------------------------------------
    orders = daily["orders"].to_numpy(float)
    with np.errstate(divide="ignore", invalid="ignore"):
        log_aov = np.where(orders > 0,
                           np.log(daily["revenue"].to_numpy(float) / np.maximum(orders, 1)), np.nan)
    aov_rows = np.where(orders > 0)[0]
    lom = np.array([np.log(np.maximum(orders[b_idx == i].mean(), 1.0)) for i in range(len(brands))])
    lam_ = np.array([np.nanmean(log_aov[b_idx == i]) for i in range(len(brands))])

    return dict(
        daily=daily, n_obs=n_obs, b_idx=b_idx, t=t_years, dow=dow, four=four,
        X_lag=X_lag, spend_scale=spend_scale, adstock_a=adstock_a, adstock_b=adstock_b,
        sends=sends_vec, regime=regime_vec,
        n_inst=n_inst, w_max=w_max, wlen=wlen, shape_conc=shape_conc,
        inst_brand=np.array(inst_brand, int), inst_type=np.array(inst_type, int),
        inst_depth=np.array(inst_depth, float), inst_share=np.array(inst_share, float),
        inst_loglist=np.array(inst_loglist, float), inst_anom=np.array(inst_anom, float),
        ev_obs=np.array(ev_obs, int), ev_inst=np.array(ev_inst, int), ev_pos=np.array(ev_pos, int),
        pf_obs=np.array(pf_obs, int), pf_inst=np.array(pf_inst, int), pf_w=np.array(pf_w, float),
        log_price_factor=log_price_factor,
        orders=orders, log_aov=log_aov, aov_rows=aov_rows,
        log_orders_mean_b=lom, log_aov_mean_b=lam_,
        n_brands=len(brands), n_channels=len(channels), n_types=len(etypes),
    )
