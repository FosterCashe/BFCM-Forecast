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

# ADVI is smoke/dev speed; NUTS is the non-negotiable for real client runs
# (see README "Non-negotiable workflow per client"). Draw/tune counts here
# are dev-sized so `docker compose up` finishes in seconds, not minutes --
# raise them (or set fit_method: "nuts" with bigger counts) for real runs.
FIT_METHOD = os.environ.get("WORKER_FIT_METHOD", "advi")
ADVI_ITERS = int(os.environ.get("WORKER_ADVI_ITERS", "20000"))
ADVI_DRAWS = int(os.environ.get("WORKER_ADVI_DRAWS", "500"))
NUTS_DRAWS = int(os.environ.get("WORKER_NUTS_DRAWS", "300"))
NUTS_TUNE = int(os.environ.get("WORKER_NUTS_TUNE", "300"))
NUTS_CHAINS = int(os.environ.get("WORKER_NUTS_CHAINS", "2"))


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


def _nuts_diagnostics(idata):
    try:
        s = az.summary(idata, var_names=PARAMS)
        rhat_max = float(s["r_hat"].max())
    except Exception:
        rhat_max = None
    try:
        divergences = int(idata.sample_stats["diverging"].sum())
    except Exception:
        divergences = None
    return {"rhat_max": rhat_max, "divergences": divergences}


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
        diagnostics = {
            "fit_method": fit_method,
            "n_train_obs": int(D["n_obs"]),
            "undeclared_spike_findings": len(asks),
        }
        with build_model(D):
            if fit_method == "nuts":
                idata = pm.sample(
                    draws=NUTS_DRAWS, tune=NUTS_TUNE, chains=NUTS_CHAINS,
                    target_accept=0.9, progressbar=False, random_seed=7,
                )
                diagnostics.update(_nuts_diagnostics(idata))
            else:
                approx = pm.fit(ADVI_ITERS, method="advi", random_seed=7, progressbar=False)
                idata = approx.sample(ADVI_DRAWS)

        P = extract_posterior(idata, PARAMS)
        Df = build_design(fut_daily, spend, fut_events, vocab)
        draws = forecast(P, Df)
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

        with conn.cursor() as cur:
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
