"""FastAPI surface for the worker service. See worker/README.md.

/upload and /connect/shopify are stubs: real ingestion (Supabase Storage,
ShopifyTransport.iter_orders) isn't wired up yet -- they validate input and
echo back a canned response so the frontend/CLI has something to integrate
against now.

/runs, /runs/{id}, and /results/{run_id} are real: they read and write
Postgres. Writing a 'queued' row here is what worker/loop.py picks up.
"""
import os
import uuid

import psycopg2.extras
from fastapi import FastAPI, HTTPException, UploadFile
from fastapi.middleware.cors import CORSMiddleware

from worker.db import get_conn
from worker.schemas import (
    ResultRow,
    RunCreate,
    RunOut,
    ShopifyConnectRequest,
    ShopifyConnectResponse,
    TenantOut,
    TotalOut,
    UploadResponse,
)

app = FastAPI(title="bfcm-forecast worker")

app.add_middleware(
    CORSMiddleware,
    allow_origins=os.environ.get(
        "CORS_ORIGINS", "http://localhost:3000,http://127.0.0.1:3000"
    ).split(","),
    allow_methods=["GET", "POST"],
    allow_headers=["content-type"],
)


@app.get("/health")
def health():
    return {"status": "ok"}


@app.post("/upload", response_model=UploadResponse)
async def upload(file: UploadFile):
    body = await file.read()
    return UploadResponse(
        upload_id=uuid.uuid4(),
        filename=file.filename or "unnamed",
        size_bytes=len(body),
        status="received",
        note=(
            "stub: not yet parsed, validated (ingest/validate.py), or stored "
            "in Supabase Storage. See worker/README.md."
        ),
    )


@app.post("/connect/shopify", response_model=ShopifyConnectResponse)
def connect_shopify(req: ShopifyConnectRequest):
    return ShopifyConnectResponse(
        tenant_id=req.tenant_id,
        shop_domain=req.shop_domain,
        status="stubbed",
        note=(
            "stub: token accepted but not persisted or verified. "
            "ShopifyTransport.iter_orders (worker/shopify_fetch.py) is not implemented yet."
        ),
    )


@app.post("/runs", response_model=RunOut, status_code=201)
def create_run(req: RunCreate):
    with get_conn() as conn, conn.cursor(cursor_factory=psycopg2.extras.RealDictCursor) as cur:
        cur.execute(
            """
            insert into model_runs (tenant_id, status, config_json)
            values (%s, 'queued', %s)
            returning run_id, tenant_id, status, started_at, finished_at, config_json, diagnostics_json
            """,
            (str(req.tenant_id), psycopg2.extras.Json(req.config_json or {})),
        )
        row = cur.fetchone()
        conn.commit()
    return RunOut(**row)


@app.get("/runs", response_model=list[RunOut])
def list_runs(limit: int = 50):
    with get_conn() as conn, conn.cursor(cursor_factory=psycopg2.extras.RealDictCursor) as cur:
        cur.execute(
            """
            select r.run_id, r.tenant_id, t.name as tenant_name, r.status, r.started_at,
                   r.finished_at, r.config_json, r.diagnostics_json
            from model_runs r left join tenants t using (tenant_id)
            order by r.started_at desc
            limit %s
            """,
            (min(max(limit, 1), 500),),
        )
        rows = cur.fetchall()
    return [RunOut(**r) for r in rows]


@app.get("/runs/{run_id}", response_model=RunOut)
def get_run(run_id: uuid.UUID):
    with get_conn() as conn, conn.cursor(cursor_factory=psycopg2.extras.RealDictCursor) as cur:
        cur.execute(
            """
            select r.run_id, r.tenant_id, t.name as tenant_name, r.status, r.started_at,
                   r.finished_at, r.config_json, r.diagnostics_json
            from model_runs r left join tenants t using (tenant_id)
            where r.run_id = %s
            """,
            (str(run_id),),
        )
        row = cur.fetchone()
    if row is None:
        raise HTTPException(status_code=404, detail="run not found")
    return RunOut(**row)


@app.get("/results/{run_id}", response_model=list[ResultRow])
def get_results(run_id: uuid.UUID):
    with get_conn() as conn, conn.cursor(cursor_factory=psycopg2.extras.RealDictCursor) as cur:
        cur.execute(
            """
            select scenario, date, metric, q10, q25, q50, q75, q90
            from forecast_results where run_id = %s
            order by scenario, metric, date
            """,
            (str(run_id),),
        )
        rows = cur.fetchall()
    return [ResultRow(**r) for r in rows]


@app.get("/results/{run_id}/totals", response_model=list[TotalOut])
def get_totals(run_id: uuid.UUID):
    with get_conn() as conn, conn.cursor(cursor_factory=psycopg2.extras.RealDictCursor) as cur:
        cur.execute(
            """
            select scenario, metric, start_date, end_date, percentiles
            from forecast_totals where run_id = %s
            order by scenario, metric
            """,
            (str(run_id),),
        )
        rows = cur.fetchall()
    return [TotalOut(**r) for r in rows]


@app.get("/tenants/{tenant_id}", response_model=TenantOut)
def get_tenant(tenant_id: uuid.UUID):
    tid = str(tenant_id)
    with get_conn() as conn, conn.cursor(cursor_factory=psycopg2.extras.RealDictCursor) as cur:
        cur.execute("select tenant_id, name, currency from tenants where tenant_id = %s", (tid,))
        tenant = cur.fetchone()
        if tenant is None:
            raise HTTPException(status_code=404, detail="tenant not found")
        cur.execute(
            "select distinct revenue_definition from daily_metrics where tenant_id = %s", (tid,)
        )
        definitions = [r["revenue_definition"] for r in cur.fetchall()]
        cur.execute(
            """
            select question_key, evidence_tier,
                   coalesce(answer_json->>'summary', answer_json::text) as summary
            from intake_answers where tenant_id = %s
            order by case evidence_tier when 'data' then 0 when 'evidenced' then 1 else 2 end,
                     question_key
            """,
            (tid,),
        )
        assumptions = cur.fetchall()
    return TenantOut(
        **tenant,
        revenue_definition=definitions[0] if len(definitions) == 1 else ("mixed" if definitions else None),
        assumptions=assumptions,
    )
