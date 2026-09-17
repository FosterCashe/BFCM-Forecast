"""Per-component decomposition of log orders around BFCM 2024: saved worker NUTS
draws vs the generator's true components.

For each brand, takes the latest N finished runs with persisted draws, rebuilds
their training design from Postgres (assumes tenant data unchanged since those
runs), and reports each component's mean over event days and pull-forward days
minus its mean over adjacent non-event weeks. The key row is "seasonal": how
much of the event's lift the model's Fourier term absorbed.

Usage (worker container): python scripts/diag_components.py [--runs-per-tenant 3]
"""
import argparse
import os
import sys

sys.path.insert(0, os.path.dirname(os.path.dirname(os.path.abspath(__file__))))

import numpy as np
import pandas as pd
import psycopg2.extras
import xarray as xr

from model.design import DEPTH_CENTER, L_ADSTOCK, build_design, make_vocab
from scripts.diag_truth import check_replica, truth
from worker.db import get_conn
from worker.pipeline import DEFAULT_BACKTEST_DAYS, fetch_tenant_frames

EVENT = ("2024-11-22", "2024-12-04")
PULL_FORWARD = [("2024-11-15", "2024-11-21"), ("2024-12-05", "2024-12-14")]
BASELINE = [("2024-11-01", "2024-11-14"), ("2024-12-15", "2024-12-28")]
COMPONENTS = ["event", "pull_forward", "media", "seasonal", "dow", "trend_intercept"]


def day_mask(dates, ranges):
    dates = pd.to_datetime(pd.Series(dates)).reset_index(drop=True)
    mask = np.zeros(len(dates), bool)
    for s, e in ranges:
        mask |= ((dates >= s) & (dates <= e)).to_numpy()
    return mask


def latest_runs(cur, n):
    cur.execute(
        """
        select r.run_id, t.name, r.tenant_id, r.config_json, r.diagnostics_json
        from model_runs r join tenants t using (tenant_id)
        where r.status = 'done' and r.diagnostics_json ? 'draws_path'
        order by r.finished_at desc
        """
    )
    picked = {}
    for row in cur.fetchall():
        picked.setdefault(row["name"], [])
        if len(picked[row["name"]]) < n:
            picked[row["name"]].append(row)
    return picked


def model_components(D, post):
    """Per-draw log-orders components on the training rows, shape (draws, n_obs)."""
    flat = lambda n: post[n].values.reshape(post[n].shape[0] * post[n].shape[1], *post[n].shape[2:])
    S, n = flat("event_base").shape[0], D["n_obs"]
    k = D["n_inst"]
    M = (flat("event_base")[:, None]
         + flat("event_brand")[:, D["inst_brand"]]
         + flat("event_depth_coef")[:, None] * (D["inst_depth"] - DEPTH_CENTER)
         + flat("event_type_effect")[:, D["inst_type"]]
         + flat("event_list_coef")[:, None] * D["inst_loglist"]
         + flat("event_anomaly")[:, :k] * D["inst_anom"]
         + flat("event_eps")[:, :k])
    shape, typ = flat("event_shape"), D["inst_type"][D["ev_inst"]]
    event = np.zeros((S, n))
    for j in range(len(D["ev_obs"])):
        event[:, D["ev_obs"][j]] += M[:, D["ev_inst"][j]] * shape[:, typ[j], D["ev_pos"][j]] * D["wlen"][typ[j]]
    pull = np.zeros((S, n))
    pf = flat("event_pullforward")
    for j in range(len(D["pf_obs"])):
        pull[:, D["pf_obs"][j]] -= pf[:, D["pf_inst"][j]] * D["pf_w"][j]
    lam, K, beta, xi = flat("adstock_decay"), flat("sat_halfpoint"), flat("media_beta"), flat("media_regime")
    media = np.zeros((S, n))
    for c in range(D["n_channels"]):
        ad = (D["X_lag"][:, c, :] @ (lam[:, c, None] ** np.arange(L_ADSTOCK + 1)[None, :]).T).T
        media += beta[:, c, None] * ad / (ad + K[:, c, None] + 1e-6)
    media *= np.exp(xi[:, None] * D["regime"][None, :])  # centering constant cancels in the deltas
    return {
        "event": event,
        "pull_forward": pull,
        "media": media,
        "seasonal": flat("fourier") @ D["four"].T,
        "dow": flat("dow") @ D["dow"].T,
        "trend_intercept": flat("brand_intercept")[:, D["b_idx"]] + flat("trend_slope")[:, D["b_idx"]] * D["t"][None, :],
    }


def main():
    ap = argparse.ArgumentParser()
    ap.add_argument("--runs-per-tenant", type=int, default=3)
    args = ap.parse_args()

    daily_true, _, _, _, comps_true = truth()
    check_replica(daily_true, comps_true)

    with get_conn() as conn, conn.cursor(cursor_factory=psycopg2.extras.RealDictCursor) as cur:
        runs = latest_runs(cur, args.runs_per_tenant)
        for brand, rows in sorted(runs.items()):
            cfg = rows[0]["config_json"]
            _, daily, spend, events = fetch_tenant_frames(conn, rows[0]["tenant_id"])
            dates = pd.to_datetime(daily["date"])
            cutoff = dates.max() - pd.Timedelta(days=int(cfg.get("backtest_days", DEFAULT_BACKTEST_DAYS)))
            vocab = make_vocab(daily, spend, events, channel_kinds=cfg.get("channel_kinds"))
            D = build_design(daily[dates <= cutoff], spend, events[pd.to_datetime(events["start"]) <= cutoff], vocab)
            for r in rows:
                assert D["n_obs"] == r["diagnostics_json"]["n_train_obs"], "tenant data changed since this run"

            tdates = D["daily"]["date"]
            ev, pfm, base = day_mask(tdates, [EVENT]), day_mask(tdates, PULL_FORWARD), day_mask(tdates, BASELINE)
            deltas = {c: {"ev": [], "pf": []} for c in COMPONENTS}
            n_draws = 0
            for r in rows:
                post = xr.open_datatree(r["diagnostics_json"]["draws_path"], engine="h5netcdf")["posterior"]
                comps = model_components(D, post)
                for c in COMPONENTS:
                    baseline = comps[c][:, base].mean(axis=1)
                    deltas[c]["ev"].append(comps[c][:, ev].mean(axis=1) - baseline)
                    deltas[c]["pf"].append(comps[c][:, pfm].mean(axis=1) - baseline)
                n_draws += comps["event"].shape[0]

            truth_c = comps_true[brand].set_index("date").loc[pd.to_datetime(tdates)].reset_index()
            obs_log = np.log(D["daily"]["orders"].to_numpy())

            seeds = [r["diagnostics_json"].get("random_seed") for r in rows]
            print(f"\n=== {brand}: {len(rows)} runs (seeds {seeds}), {n_draws} draws; "
                  f"deltas vs adjacent non-event weeks {BASELINE}")
            print(f"{'component':16s} | {'event days: model (sd)':>24s} {'truth':>7s} {'model-truth':>11s} | "
                  f"{'pull-fwd days: model (sd)':>26s} {'truth':>7s} {'model-truth':>11s}")
            totals = {"model_ev": 0.0, "truth_ev": 0.0, "model_pf": 0.0, "truth_pf": 0.0}
            for c in COMPONENTS:
                m_ev, m_pf = np.concatenate(deltas[c]["ev"]), np.concatenate(deltas[c]["pf"])
                t_ev = truth_c[c].to_numpy()[ev].mean() - truth_c[c].to_numpy()[base].mean()
                t_pf = truth_c[c].to_numpy()[pfm].mean() - truth_c[c].to_numpy()[base].mean()
                totals["model_ev"] += m_ev.mean(); totals["truth_ev"] += t_ev
                totals["model_pf"] += m_pf.mean(); totals["truth_pf"] += t_pf
                print(f"{c:16s} | {m_ev.mean():+18.3f} ({m_ev.std():.3f}) {t_ev:+7.3f} {m_ev.mean() - t_ev:+11.3f} | "
                      f"{m_pf.mean():+20.3f} ({m_pf.std():.3f}) {t_pf:+7.3f} {m_pf.mean() - t_pf:+11.3f}")
            o_ev = obs_log[ev].mean() - obs_log[base].mean()
            o_pf = obs_log[pfm].mean() - obs_log[base].mean()
            print(f"{'sum':16s} | {totals['model_ev']:+18.3f}         {totals['truth_ev']:+7.3f} {totals['model_ev'] - totals['truth_ev']:+11.3f} | "
                  f"{totals['model_pf']:+20.3f}         {totals['truth_pf']:+7.3f} {totals['model_pf'] - totals['truth_pf']:+11.3f}")
            print(f"{'observed log orders':16s} | {o_ev:+18.3f} (data) | {o_pf:+20.3f} (data)")


if __name__ == "__main__":
    main()
