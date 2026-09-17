"""
Forecasting: posterior draws pushed through the same generative math in numpy.
Future event instances draw magnitudes from the fitted hierarchy (covariates
move the center, fresh instance noise keeps uncertainty honest).

Also here: inventory ceilings (truncate demand, report P(sellout)) and
launch components (client assumptions propagated as labeled lognormals).
"""

import numpy as np
import arviz as az

from .design import L_ADSTOCK, DEPTH_CENTER

PARAMS = [
    "brand_intercept", "trend_slope", "dow", "fourier",
    "event_base", "event_brand", "event_depth_coef", "event_type_effect",
    "event_list_coef", "event_eps_sigma", "event_pf_sigma", "event_shape",
    "adstock_decay", "sat_halfpoint", "media_beta", "media_regime", "send_coef",
    "nb_alpha", "aov_brand", "aov_type_effect", "aov_depth_coef", "aov_sigma",
]


def extract_posterior(idata, names=PARAMS):
    ds = az.extract(idata)
    return {n: np.moveaxis(ds[n].values, -1, 0) for n in names}


def _media(P, s, D):
    powers = P["adstock_decay"][s][:, None] ** np.arange(L_ADSTOCK + 1)[None, :]
    ad = np.sum(D["X_lag"] * powers[None, :, :], axis=2)
    media = np.sum(P["media_beta"][s] * ad / (ad + P["sat_halfpoint"][s] + 1e-6), axis=1)
    return media * np.exp(P["media_regime"][s] * D["regime"])


def forecast(P, Df, D_train, rng=None):
    """D_train: the design the posterior was fit on. The model centers media on
    each brand's training-window mean per draw, so the forecast must subtract
    that same training mean, never one recomputed on the forecast window."""
    rng = rng or np.random.default_rng(7)
    S = P["brand_intercept"].shape[0]
    n = Df["n_obs"]
    n_brands = D_train["n_brands"]
    n_train_per_brand = np.maximum(np.bincount(D_train["b_idx"], minlength=n_brands), 1)
    orders_d = np.zeros((S, n))
    aov_d = np.zeros((S, n))
    typ_of_ev = Df["inst_type"][Df["ev_inst"]] if Df["n_inst"] else np.array([], int)

    for s in range(S):
        lift = np.zeros(n)
        basket = np.zeros(n)
        if Df["n_inst"]:
            eps = rng.normal(0.0, P["event_eps_sigma"][s], Df["n_inst"])
            anom = rng.normal(0.0, 0.5, Df["n_inst"]) * Df["inst_anom"]
            M = (
                P["event_base"][s]
                + P["event_brand"][s][Df["inst_brand"]]
                + P["event_depth_coef"][s] * (Df["inst_depth"] - DEPTH_CENTER)
                + P["event_type_effect"][s][Df["inst_type"]]
                + P["event_list_coef"][s] * Df["inst_loglist"]
                + anom + eps
            )
            contrib = M[Df["ev_inst"]] * P["event_shape"][s][typ_of_ev, Df["ev_pos"]] * Df["wlen"][typ_of_ev]
            np.add.at(lift, Df["ev_obs"], contrib)
            pf = np.abs(rng.normal(0.0, P["event_pf_sigma"][s], Df["n_inst"]))
            np.add.at(lift, Df["pf_obs"], -pf[Df["pf_inst"]] * Df["pf_w"])
            np.add.at(
                basket, Df["ev_obs"],
                P["aov_type_effect"][s][typ_of_ev]
                + P["aov_depth_coef"][s] * (Df["inst_depth"][Df["ev_inst"]] - DEPTH_CENTER),
            )

        train_mean_b = (
            np.bincount(D_train["b_idx"], weights=_media(P, s, D_train), minlength=n_brands) / n_train_per_brand
        )
        media = _media(P, s, Df) - train_mean_b[Df["b_idx"]]

        log_mu = (
            P["brand_intercept"][s][Df["b_idx"]]
            + P["trend_slope"][s][Df["b_idx"]] * Df["t"]
            + Df["dow"] @ P["dow"][s]
            + Df["four"] @ P["fourier"][s]
            + lift + media
            + P["send_coef"][s] * Df["sends"]
        )
        mu = np.exp(log_mu)
        alpha = P["nb_alpha"][s][Df["b_idx"]]
        orders_d[s] = rng.poisson(rng.gamma(alpha, mu / alpha))
        aov_d[s] = np.exp(rng.normal(
            P["aov_brand"][s][Df["b_idx"]] + basket + Df["log_price_factor"],
            P["aov_sigma"][s][Df["b_idx"]],
        ))

    return dict(orders=orders_d, aov=aov_d, revenue=orders_d * aov_d)


def add_launch(draws, Df, brand, low, high, start, end, rng=None):
    """Client-assumed launch revenue (10th..90th pct = low..high), spread over
    its window and added to revenue draws. LABEL AS ASSUMPTION in all outputs."""
    rng = rng or np.random.default_rng(3)
    import pandas as pd
    S = draws["revenue"].shape[0]
    mu = (np.log(low) + np.log(high)) / 2
    sig = (np.log(high) - np.log(low)) / (2 * 1.2816)
    total = rng.lognormal(mu, sig, S)
    dts = Df["daily"]["date"]
    mask = ((Df["daily"]["brand"] == brand).to_numpy()
            & (dts >= pd.Timestamp(start)).to_numpy() & (dts <= pd.Timestamp(end)).to_numpy())
    cols = np.where(mask)[0]
    if len(cols):
        draws["revenue"][:, cols] += total[:, None] / len(cols)
    return draws


def apply_inventory(draws, Df, ceilings):
    """ceilings: {brand_index: max units sellable}. Truncates order draws at the
    ceiling and reports sellout risk. Returns capped draws + per-brand stats."""
    out = {}
    orders = draws["orders"].copy()
    for bi, ceil in ceilings.items():
        cols = np.where(Df["b_idx"] == bi)[0]
        cum = np.cumsum(orders[:, cols], axis=1)
        capped = np.diff(np.minimum(cum, ceil), axis=1, prepend=0)
        orders[:, cols] = capped
        sold_out = cum[:, -1] >= ceil
        first = np.argmax(cum >= ceil, axis=1)
        out[bi] = dict(
            p_sellout=float(sold_out.mean()),
            median_sellout_day=(str(Df["daily"].loc[cols[int(np.median(first[sold_out]))], "date"].date())
                                if sold_out.any() else None),
        )
    return dict(orders=orders, aov=draws["aov"], revenue=orders * draws["aov"]), out


def summarize(draws, Df, probs=(0.1, 0.25, 0.5, 0.75, 0.9)):
    out = {}
    rev = draws["revenue"]
    for bi in range(Df["n_brands"]):
        cols = np.where(Df["b_idx"] == bi)[0]
        totals = rev[:, cols].sum(axis=1)
        out[bi] = dict(
            dates=Df["daily"].loc[cols, "date"].to_numpy(),
            daily_q=np.quantile(rev[:, cols], probs, axis=0), probs=probs,
            total_q=np.quantile(totals, probs), total_draws=totals,
        )
    return out
