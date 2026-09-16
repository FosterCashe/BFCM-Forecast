# Worker service (Railway deployment target)

One Python service, two roles:
  API (FastAPI): POST /upload, POST /connect/shopify, POST /runs,
                 GET /runs/{id}, GET /results/{run_id}
  Worker loop:   poll model_runs where status='queued' every 30s, then:
                 fetch/refresh tenant data -> build_design -> NUTS fit ->
                 write quantiles to forecast_results -> status='done'
                 (this is scripts/smoke_backtest.py rearranged around Postgres)

Shopify ingestion: worker/shopify_fetch.py. Transport is injected:
  FixtureTransport  recorded JSONL (tests, demos; see tests/)
  ShopifyTransport  live Admin GraphQL, custom-app token from intake
  Next step: finish ShopifyTransport.iter_orders (bulk operation poll +
  JSONL stream) and verify against a FREE Shopify Partner development store
  with fake orders. No client access needed to test the live path.

Ad platforms stay CSV this season (their API access requires multi-week
review processes). Start Meta + Google API applications in January; the
clock on those runs in calendar time, not effort.

Deployment notes:
  - Pin dependencies hard; PyMC on a fresh container will bite otherwise.
  - 2GB+ memory: NUTS with 4 chains is hungry.
  - Store raw uploads in Supabase Storage before parsing (replayable ingestion).
  - Write model version + git SHA into model_runs.config_json on every run.
