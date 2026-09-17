"""
Core Bayesian model, v2. Log-scale (multiplicative) throughout.

ORDERS (NegBin):
  log mu = brand intercept + brand trend + day-of-week + annual Fourier
         + event lift (magnitude x shared shape) - pull-forward dips
         + media (adstock -> saturation, per channel, regime-scaled)
         + email/SMS send effect

Event magnitude hierarchy (the moat component):
  M_i = base + brand offset + depth coef + offer-type effect
      + list-growth coef * log(list ratio) + anomaly dummy + instance noise

AOV split into MECHANICS vs BEHAVIOR:
  observed log AOV = pre-discount basket level (statistical)
                   + basket growth during events (statistical)
                   + log(1 - depth * discounted_share)  (deterministic arithmetic)
  The price effect of a discount is known math, not a parameter to estimate.

Prior notes vs v1 (calibrated against DTC experience):
  event_depth_coef recentered 1.0 -> 1.8 (15-30% lift per 10 pts of depth)
  event_shape now informative for BFCM types (known holiday curve)
  adstock priors per channel KIND (prospecting vs search vs owned)
  aov_depth_coef now models BASKET growth (positive) since price math moved out
Run pm.sample_prior_predictive and eyeball implied dollars before every real fit.
"""

import numpy as np
import pymc as pm
import pytensor.tensor as pt

from .design import L_ADSTOCK, DEPTH_CENTER


def build_model(D):
    nb, nc, nt = D["n_brands"], D["n_channels"], D["n_types"]
    ni = max(D["n_inst"], 1)

    with pm.Model() as m:
        # ---- baseline -----------------------------------------------------
        a = pm.Normal("brand_intercept", mu=D["log_orders_mean_b"], sigma=1.0, shape=nb)
        slope = pm.Normal("trend_slope", 0.0, 0.3, shape=nb)  # widen to 0.5 for young brands
        dow_c = pm.Normal("dow", 0.0, 0.3, shape=6)
        four_c = pm.Normal("fourier", 0.0, 0.3, shape=D["four"].shape[1])

        # ---- event magnitude hierarchy -------------------------------------
        # Hierarchical offsets are non-centered (z * sigma) to avoid funnel
        # divergences under NUTS; Deterministics keep the posterior names stable.
        base = pm.Normal("event_base", 0.8, 0.5)
        if nb > 1:
            sb = pm.HalfNormal("event_brand_sigma", 0.4)
            eb_z = pm.Normal("event_brand_z", 0.0, 1.0, shape=nb)
            eb = sb * eb_z
        else:
            # event_base already carries the one brand's offset; sampling both is redundant.
            eb = pt.zeros(nb)
        eb = pm.Deterministic("event_brand", eb)
        gamma = pm.Normal("event_depth_coef", 1.8, 0.6)
        if nt == 1:
            # One event type: its effect is indistinguishable from event_base.
            dtyp = pm.Deterministic("event_type_effect", pt.zeros(nt))
        else:
            dtyp = pm.Normal("event_type_effect", 0.0, 0.3, shape=nt)
        eta = pm.Normal("event_list_coef", 0.6, 0.3)
        eps_s = pm.HalfNormal("event_eps_sigma", 0.25)
        eps_z = pm.Normal("event_eps_z", 0.0, 1.0, shape=ni)
        eps = pm.Deterministic("event_eps", eps_s * eps_z)
        anom = pm.Normal("event_anomaly", 0.0, 0.5, shape=ni)

        M = (
            base
            + eb[D["inst_brand"]]
            + gamma * (D["inst_depth"] - DEPTH_CENTER)
            + dtyp[D["inst_type"]]
            + eta * D["inst_loglist"]
            + anom[: D["n_inst"]] * D["inst_anom"]
            + eps[: D["n_inst"]]
        )
        shape = pm.Dirichlet("event_shape", a=D["shape_conc"], shape=(nt, D["w_max"]))
        typ_of_ev = D["inst_type"][D["ev_inst"]]
        contrib = M[D["ev_inst"]] * shape[typ_of_ev, D["ev_pos"]] * D["wlen"][typ_of_ev]
        lift = pt.zeros(D["n_obs"])
        lift = pt.inc_subtensor(lift[D["ev_obs"]], contrib)

        # pull-forward: demand borrowed from before/after the event
        pf_s = pm.HalfNormal("event_pf_sigma", 0.10)
        pf_z = pm.HalfNormal("event_pullforward_z", 1.0, shape=ni)
        pf = pm.Deterministic("event_pullforward", pf_s * pf_z)
        lift = pt.inc_subtensor(lift[D["pf_obs"]], -pf[D["pf_inst"]] * D["pf_w"])

        # ---- media ---------------------------------------------------------
        lam = pm.Beta("adstock_decay", D["adstock_a"], D["adstock_b"], shape=nc)
        powers = lam[:, None] ** np.arange(L_ADSTOCK + 1)[None, :]
        ad = pt.sum(D["X_lag"] * powers[None, :, :], axis=2)
        K = pm.HalfNormal("sat_halfpoint", sigma=D["spend_scale"])
        beta = pm.HalfNormal("media_beta", 0.3, shape=nc)  # loosen to 0.5 if heavily paid-driven
        xi = pm.Normal("media_regime", 0.0, 0.15)  # skeptical creative/account-change multiplier
        media = pt.sum(beta * ad / (ad + K + 1e-6), axis=1) * pt.exp(xi * D["regime"])
        # Center per brand on this draw's training-window mean: adstocked spend
        # barely moves day to day, so uncentered media acts as a second intercept.
        # forecast.py subtracts the same training mean; never a future-window one.
        n_per_brand = np.maximum(np.bincount(D["b_idx"], minlength=nb), 1)
        media_mean_b = pt.inc_subtensor(pt.zeros(nb)[D["b_idx"]], media) / n_per_brand
        media = media - media_mean_b[D["b_idx"]]

        send_c = pm.HalfNormal("send_coef", 0.15)

        # ---- orders likelihood ---------------------------------------------
        log_mu = (
            a[D["b_idx"]]
            + slope[D["b_idx"]] * D["t"]
            + D["dow"] @ dow_c
            + D["four"] @ four_c
            + lift
            + media
            + send_c * D["sends"]
        )
        alpha = pm.Gamma("nb_alpha", 2.0, 0.1, shape=nb)
        pm.NegativeBinomial("orders", mu=pt.exp(log_mu), alpha=alpha[D["b_idx"]], observed=D["orders"])

        # ---- AOV: behavior estimated, price arithmetic imposed --------------
        av = pm.Normal("aov_brand", mu=D["log_aov_mean_b"] - 0.0, sigma=0.5, shape=nb)
        kappa = pm.Normal("aov_type_effect", 0.0, 0.15, shape=nt)   # BOGO lifts units/basket
        rho = pm.Normal("aov_depth_coef", 0.3, 0.3)                  # deeper deal -> bigger basket
        basket = pt.zeros(D["n_obs"])
        basket = pt.inc_subtensor(
            basket[D["ev_obs"]],
            kappa[typ_of_ev] + rho * (D["inst_depth"][D["ev_inst"]] - DEPTH_CENTER),
        )
        sig_aov = pm.HalfNormal("aov_sigma", 0.15, shape=nb)
        r = D["aov_rows"]
        pm.Normal(
            "log_aov",
            mu=(av[D["b_idx"]] + basket + D["log_price_factor"])[r],
            sigma=sig_aov[D["b_idx"][r]],
            observed=D["log_aov"][r],
        )

    return m
