"""Model-run pipeline: Postgres tenant data -> design matrix -> fit ->
forecast -> forecast_results. Same protocol as scripts/smoke_backtest.py
(train up to a cutoff, score the held-out window against actuals),
rearranged to read/write Postgres instead of model.synthetic.generate().

Scope note: this fits ONE brand per tenant -- the schema has no brand
column, and cross-client pooling is upgrade #3 in the README, not
implemented yet. The held-out ("forecast") window must fall inside dates
the tenant already has in daily_metrics: this mirrors the backtest
protocol (score against known actuals), not blind future forecasting from
an empty calendar, which needs synthesized future rows and is separate
product work.
"""
import os
import subprocess
import traceback
from datetime import datetime, timezone

import arviz as az
import numpy as np
import pandas as pd
import psycopg2.extras
import pymc as pm

from ingest.validate import detect_unexplained_spikes, validate_daily
from model.design import build_design, make_vocab
from model.forecast import PARAMS, extract_posterior, forecast, summarize
from model.model import build_model

DEFAULT_BACKTEST_DAYS = 21
PROBS = (0.1, 0.25, 0.5, 0.75, 0.9)  # matches forecast_results.q10..q90

# ADVI is smoke/dev speed. NUTS (config_json fit_method: "nuts") is the real
# client protocol from the README; its defaults match that protocol, and a
# run can override any of them via config_json nuts: {draws, tune, ...}.
FIT_METHOD = os.environ.get("WORKER_FIT_METHOD", "advi")
ADVI_ITERS = int(os.environ.get("WORKER_ADVI_ITERS", "20000"))
ADVI_DRAWS = int(os.environ.get("WORKER_ADVI_DRAWS", "500"))
NUTS_DEFAULTS = {
    "chains": int(os.environ.get("WORKER_NUTS_CHAINS", "4")),
    "draws": int(os.environ.get("WORKER_NUTS_DRAWS", "1000")),
    "tune": int(os.environ.get("WORKER_NUTS_TUNE", "1000")),
    "target_accept": float(os.environ.get("WORKER_NUTS_TARGET_ACCEPT", "0.9")),
}
DRAWS_DIR = os.environ.get("WORKER_DRAWS_DIR", "/data/draws")

# Coefficients the page may need to label as prior-driven rather than learned.
SHRINKAGE_COEFS = ("event_depth_coef", "event_list_coef", "event_type_effect")
PRIOR_DRIVEN_SD_RATIO = 0.9
PRIOR_DRIVEN_MEAN_SHIFT = 0.2


GIT_SHA_FILE = os.path.join(os.path.dirname(os.path.dirname(os.path.abspath(__file__))), "GIT_SHA")


def _git_sha():
    # Railway sets the env var; worker/Dockerfile bakes GIT_SHA at build time
    # (.git is kept out of the image); bare-metal runs ask git directly.
    for env in ("RAILWAY_GIT_COMMIT_SHA", "GIT_SHA", "GITHUB_SHA"):
        if os.environ.get(env):
            return os.environ[env]
    try:
        with open(GIT_SHA_FILE) as fh:
            sha = fh.read().strip()
        if sha:
            return sha
    except OSError:
        pass
    try:
        return (
            subprocess.check_output(["git", "rev-parse", "HEAD"], stderr=subprocess.DEVNULL)
            .decode()
            .strip()
        )
    except Exception:
        return None


def _read_df(conn, sql, params=()):
    with conn.cursor() as cur:
        cur.execute(sql, params)
        cols = [c[0] for c in cur.description]
        rows = cur.fetchall()
    return pd.DataFrame(rows, columns=cols)


def fetch_tenant_frames(conn, tenant_id):
    """Single-tenant daily/spend/events frames, labeled as one 'brand' so
    they slot straight into model/design.py (built for multi-brand pooling).
    Numeric columns are cast to float8 so psycopg2 hands back plain floats,
    not decimal.Decimal (which numpy/pymc don't handle)."""
    tenant = _read_df(conn, "select name from tenants where tenant_id = %s", (tenant_id,))
    if tenant.empty:
        raise ValueError(f"unknown tenant_id {tenant_id}")
    brand = tenant.at[0, "name"]

    daily = _read_df(
        conn,
        "select date, orders, revenue::float8 as revenue "
        "from daily_metrics where tenant_id = %s order by date",
        (tenant_id,),
    )
    daily.insert(0, "brand", brand)

    spend = _read_df(
        conn,
        "select date, channel, channel_kind, spend::float8 as spend "
        "from ad_spend where tenant_id = %s order by date",
        (tenant_id,),
    )
    spend.insert(0, "brand", brand)

    events = _read_df(
        conn,
        """
        select event_id, event_type, start_date as start, end_date as "end",
               depth::float8 as depth, discounted_share::float8 as discounted_share,
               list_ratio::float8 as list_ratio, anomaly
        from events where tenant_id = %s order by start_date
        """,
        (tenant_id,),
    )
    events.insert(0, "brand", brand)

    return brand, daily, spend, events


def _nuts_diagnostics(idata, free_var_names):
    """Worst rhat per free (sampled) variable plus divergence count.
    Deterministics are skipped: they're functions of the free variables, and a
    constant one (single-brand event_brand) has an undefined, NaN rhat. A NaN
    in a free variable (e.g. a stuck chain) is stored as null so JSON stays
    valid and the API flags it. Errors propagate: no diagnostics, no run."""
    rhat = az.rhat(idata)
    per_var = {}
    for name in free_var_names:
        values = np.asarray(rhat[name].values, dtype=float)
        per_var[name] = None if np.isnan(values).any() else float(values.max())
    finite = [v for v in per_var.values() if v is not None]
    return {
        "rhat": per_var,
        "rhat_max": max(finite) if finite else None,
        "divergences": int(idata.sample_stats["diverging"].sum()),
    }


def _prior_shrinkage(model, idata, event_types):
    """Posterior sd / prior sd and |posterior mean - prior mean| in prior sds,
    per coefficient. Prior parameters are read from the model graph, so this
    stays in sync with model/model.py. prior_driven means the data barely
    moved the prior. Coefficients fixed at zero (not sampled) are reported so."""
    free = {rv.name: rv for rv in model.free_RVs}
    out = {}
    for name in SHRINKAGE_COEFS:
        rv = free.get(name)
        if rv is None:
            out[name] = {"sampled": False}
            continue
        if type(rv.owner.op).__name__ != "NormalRV":
            raise ValueError(f"prior shrinkage assumes a Normal prior on {name}")
        values = idata.posterior[name].values  # (chain, draw, *shape)
        prior_mu, prior_sd = (
            np.ravel(np.broadcast_to(p.eval(), values.shape[2:])) for p in rv.owner.op.dist_params(rv.owner)
        )
        values = values.reshape(values.shape[0] * values.shape[1], -1)
        elements = []
        for j in range(values.shape[1]):
            sd_ratio = float(values[:, j].std() / prior_sd[j])
            shift = float(abs(values[:, j].mean() - prior_mu[j]) / prior_sd[j])
            elements.append({
                "posterior_sd_ratio": sd_ratio,
                "mean_shift_prior_sd": shift,
                "prior_driven": sd_ratio > PRIOR_DRIVEN_SD_RATIO and shift < PRIOR_DRIVEN_MEAN_SHIFT,
            })
        if rv.type.ndim == 0:
            out[name] = {"sampled": True, **elements[0]}
        else:
            out[name] = {"sampled": True, "by_type": dict(zip(event_types, elements))}
    return out


def run_model_run(conn, run_id):
    """Fetch the model_runs row, run the fit/forecast pipeline for its
    tenant, and write forecast_results + the final model_runs status.
    Leaves model_runs in status='error' (with the exception in
    diagnostics_json) rather than raising silently."""
    with conn.cursor(cursor_factory=psycopg2.extras.RealDictCursor) as cur:
        cur.execute("select run_id, tenant_id, config_json from model_runs where run_id = %s", (run_id,))
        run = cur.fetchone()
    if run is None:
        raise ValueError(f"unknown run_id {run_id}")

    config = dict(run["config_json"] or {})
    config["model_version"] = config.get("model_version", "v2")
    config["git_sha"] = _git_sha()
    tenant_id = run["tenant_id"]

    try:
        brand, daily, spend, events = fetch_tenant_frames(conn, tenant_id)

        blocks = [f for f in validate_daily(daily) if f["severity"] == "block"]
        if blocks:
            raise ValueError("validation blocked: " + "; ".join(f["msg"] for f in blocks))
        asks = detect_unexplained_spikes(daily, events)

        dates = pd.to_datetime(daily["date"])
        if dates.empty:
            raise ValueError("no daily_metrics rows for this tenant")
        max_date = dates.max()
        backtest_days = int(config.get("backtest_days", DEFAULT_BACKTEST_DAYS))
        cutoff = pd.Timestamp(config["cutoff"]) if config.get("cutoff") else max_date - pd.Timedelta(days=backtest_days)
        forecast_end = pd.Timestamp(config["forecast_end"]) if config.get("forecast_end") else max_date
        if cutoff >= forecast_end:
            raise ValueError(f"cutoff {cutoff.date()} must be before forecast_end {forecast_end.date()}")

        vocab = make_vocab(daily, spend, events, channel_kinds=config.get("channel_kinds"))

        train_daily = daily[dates <= cutoff]
        train_events = events[pd.to_datetime(events["start"]) <= cutoff]
        fut_daily = daily[(dates > cutoff) & (dates <= forecast_end)]
        fut_events = events[pd.to_datetime(events["start"]) > cutoff]
        if train_daily.empty or fut_daily.empty:
            raise ValueError("not enough tenant history to fill both the train and forecast windows")

        D = build_design(train_daily, spend, train_events, vocab)

        fit_method = config.get("fit_method", FIT_METHOD)
        if fit_method not in ("advi", "nuts"):
            raise ValueError(f"fit_method must be 'advi' or 'nuts', got {fit_method!r}")
        seed = int(config.get("random_seed", 7))
        in_window = fut_events[pd.to_datetime(fut_events["start"]) <= forecast_end]
        diagnostics = {
            "fit_method": fit_method,
            "random_seed": seed,
            "n_train_obs": int(D["n_obs"]),
            "undeclared_spike_findings": len(asks),
            "event_depths": {
                "train": sorted({round(float(x), 3) for x in train_events["depth"]}),
                "forecast": sorted({round(float(x), 3) for x in in_window["depth"]}),
            },
        }
        with build_model(D) as model:
            if fit_method == "nuts":
                sampler = {**NUTS_DEFAULTS, **config.get("nuts", {})}
                diagnostics["sampler"] = sampler
                idata = pm.sample(
                    draws=int(sampler["draws"]), tune=int(sampler["tune"]), chains=int(sampler["chains"]),
                    target_accept=float(sampler["target_accept"]), progressbar=False, random_seed=seed,
                )
                diagnostics.update(_nuts_diagnostics(idata, [rv.name for rv in model.free_RVs]))
            else:
                approx = pm.fit(ADVI_ITERS, method="advi", random_seed=seed, progressbar=False)
                idata = approx.sample(ADVI_DRAWS)
            diagnostics["prior_shrinkage"] = _prior_shrinkage(model, idata, vocab["event_types"])

        if config.get("persist_draws"):
            os.makedirs(DRAWS_DIR, exist_ok=True)
            path = os.path.join(DRAWS_DIR, f"{run_id}.nc")
            idata.to_netcdf(path, engine="h5netcdf")
            diagnostics["draws_path"] = path

        P = extract_posterior(idata, PARAMS)
        Df = build_design(fut_daily, spend, fut_events, vocab)
        draws = forecast(P, Df, D)
        summ = summarize(draws, Df, probs=PROBS)[0]  # single-brand vocab -> index 0

        actual = Df["daily"]["revenue"].to_numpy()
        med, lo, hi = summ["daily_q"][2], summ["daily_q"][0], summ["daily_q"][4]  # q50, q10, q90
        diagnostics["backtest_mape"] = float(np.mean(np.abs(med - actual) / np.maximum(actual, 1e-9)))
        diagnostics["backtest_coverage"] = float(np.mean((actual >= lo) & (actual <= hi)))
        diagnostics["n_forecast_obs"] = int(Df["n_obs"])

        dates_out = pd.to_datetime(Df["daily"]["date"]).dt.date.tolist()
        rows = []
        for metric, arr in (("revenue", draws["revenue"]), ("orders", draws["orders"]), ("aov", draws["aov"])):
            q = np.quantile(arr, PROBS, axis=0)
            for i, d in enumerate(dates_out):
                rows.append(("base", d, metric, *(float(q[p, i]) for p in range(len(PROBS)))))
        total_pct = [float(v) for v in np.percentile(summ["total_draws"], np.arange(101))]

        with conn.cursor() as cur:
            cur.execute(
                """
                insert into forecast_totals (run_id, tenant_id, scenario, metric, start_date, end_date, percentiles)
                values (%s, %s, 'base', 'revenue', %s, %s, %s)
                on conflict (run_id, scenario, metric) do update
                set start_date = excluded.start_date, end_date = excluded.end_date,
                    percentiles = excluded.percentiles
                """,
                (run_id, tenant_id, dates_out[0], dates_out[-1], psycopg2.extras.Json(total_pct)),
            )
            psycopg2.extras.execute_values(
                cur,
                """
                insert into forecast_results (run_id, tenant_id, scenario, date, metric, q10, q25, q50, q75, q90)
                values %s
                on conflict (run_id, scenario, date, metric) do update
                set q10 = excluded.q10, q25 = excluded.q25, q50 = excluded.q50,
                    q75 = excluded.q75, q90 = excluded.q90
                """,
                [(run_id, tenant_id, *row) for row in rows],
            )
            cur.execute(
                """
                update model_runs
                set status = 'done', finished_at = %s, config_json = %s, diagnostics_json = %s
                where run_id = %s
                """,
                (datetime.now(timezone.utc), psycopg2.extras.Json(config), psycopg2.extras.Json(diagnostics), run_id),
            )
        conn.commit()

    except Exception as e:
        conn.rollback()
        with conn.cursor() as cur:
            cur.execute(
                """
                update model_runs
                set status = 'error', finished_at = %s, config_json = %s, diagnostics_json = %s
                where run_id = %s
                """,
                (
                    datetime.now(timezone.utc),
                    psycopg2.extras.Json(config),
                    psycopg2.extras.Json({"error": str(e), "traceback": traceback.format_exc()[-4000:]}),
                    run_id,
                ),
            )
        conn.commit()
        raise
