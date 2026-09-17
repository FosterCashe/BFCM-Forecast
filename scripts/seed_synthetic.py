"""Seed a local Postgres (docker-compose) with model/synthetic.py brands as
fake tenants, so worker/loop.py has something real to pick up end to end.

Idempotent: re-running upserts the same two tenants (deterministic UUIDs
derived from brand name) instead of duplicating rows, and only enqueues a
model_run the first time so repeated `docker compose up` runs don't spam
the queue.

Usage: DATABASE_URL=postgresql://... python -m scripts.seed_synthetic
"""
import datetime as dt
import os
import sys
import uuid

sys.path.insert(0, os.path.dirname(os.path.dirname(os.path.abspath(__file__))))

import psycopg2
import psycopg2.extras

from model.synthetic import generate
from worker.db import DATABASE_URL

NAMESPACE = uuid.UUID("6f3f7d3a-2b1e-4a9a-9c1a-000000000000")


def tenant_id_for(brand):
    return uuid.uuid5(NAMESPACE, brand)


def intake_answers(d, e):
    """Tiered intake answers for a fake tenant, derived from what was generated
    so every DATA-BACKED line is actually true of the seeded rows."""
    first, last = d["date"].min(), d["date"].max()
    months = round((last - first).days / 30.44)
    bfcm = e.sort_values("start")
    past, planned = bfcm.iloc[:-1], bfcm.iloc[-1]

    def fmt(iso):
        day = dt.date.fromisoformat(iso)
        return f"{day:%b} {day.day}, {day.year}"

    return [
        ("shopify_csv", "data", {
            "summary": f"{months} months of daily orders and revenue ({first:%b %Y} to {last:%b %Y})",
        }),
        ("prior_promos", "data", {
            "summary": "; ".join(f"BFCM {r.start[:4]} at {r.depth:.0%} sitewide" for r in past.itertuples())
                       + " in history, declared with dates and depth",
        }),
        ("events_planned", "data", {
            "summary": f"BFCM {planned.start[:4]} planned at {planned.depth:.0%} sitewide, "
                       f"{fmt(planned.start)} to {fmt(planned.end)}",
        }),
        ("ad_csvs", "data", {"summary": "Daily Meta spend for the full history"}),
        ("launch_calendar", "assumption", {"summary": "No product launch during the window (client-stated)"}),
    ]


def main():
    daily, spend, events = generate()
    conn = psycopg2.connect(DATABASE_URL)
    try:
        with conn.cursor() as cur:
            for brand in sorted(daily["brand"].unique()):
                tid = tenant_id_for(brand)
                cur.execute(
                    """
                    insert into tenants (tenant_id, name, pooling_consent)
                    values (%s, %s, true)
                    on conflict (tenant_id) do update set name = excluded.name
                    """,
                    (str(tid), brand),
                )

                d = daily[daily["brand"] == brand]
                psycopg2.extras.execute_values(
                    cur,
                    """
                    insert into daily_metrics (tenant_id, date, orders, revenue, revenue_definition)
                    values %s
                    on conflict (tenant_id, date) do update
                    set orders = excluded.orders, revenue = excluded.revenue,
                        revenue_definition = excluded.revenue_definition
                    """,
                    # synthetic AOV already has the discount price factor applied
                    [(str(tid), r.date.date(), int(r.orders), float(r.revenue), "net_discounts") for r in d.itertuples()],
                )

                s = spend[spend["brand"] == brand]
                psycopg2.extras.execute_values(
                    cur,
                    """
                    insert into ad_spend (tenant_id, date, channel, spend)
                    values %s
                    on conflict (tenant_id, date, channel) do update
                    set spend = excluded.spend
                    """,
                    [(str(tid), r.date.date(), r.channel, float(r.spend)) for r in s.itertuples()],
                )

                e = events[events["brand"] == brand]
                psycopg2.extras.execute_values(
                    cur,
                    """
                    insert into events (tenant_id, event_id, event_type, start_date, end_date, depth)
                    values %s
                    on conflict (tenant_id, event_id) do update
                    set start_date = excluded.start_date, end_date = excluded.end_date, depth = excluded.depth
                    """,
                    [
                        (str(tid), r.event_id, r.event_type, dt.date.fromisoformat(r.start), dt.date.fromisoformat(r.end), float(r.depth))
                        for r in e.itertuples()
                    ],
                )

                psycopg2.extras.execute_values(
                    cur,
                    """
                    insert into intake_answers (tenant_id, question_key, answer_json, evidence_tier)
                    values %s
                    on conflict (tenant_id, question_key) do update
                    set answer_json = excluded.answer_json, evidence_tier = excluded.evidence_tier
                    """,
                    [(str(tid), key, psycopg2.extras.Json(answer), tier) for key, tier, answer in intake_answers(d, e)],
                )

                cur.execute("select count(*) from model_runs where tenant_id = %s", (str(tid),))
                (n,) = cur.fetchone()
                queued_note = ""
                if n == 0:
                    cur.execute(
                        """
                        insert into model_runs (tenant_id, status, config_json)
                        values (%s, 'queued', %s)
                        """,
                        (str(tid), psycopg2.extras.Json({"fit_method": "advi", "backtest_days": 21})),
                    )
                    queued_note = ", queued a model_run"

                print(f"seeded fake tenant '{brand}': {tid}  ({len(d)} days, {len(e)} events{queued_note})")
        conn.commit()
    finally:
        conn.close()


if __name__ == "__main__":
    main()
