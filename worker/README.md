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

Local dev (docker-compose): `docker compose up --build` from the repo root
starts Postgres (loaded with db/schema.sql), seeds model/synthetic.py's two
brands as fake tenants (scripts/seed_synthetic.py) with a model_run already
queued for each, then starts the API on :8000 and the worker loop. Code:
  worker/app.py       FastAPI: /upload and /connect/shopify are stubs;
                       /runs, /runs/{id}, /results/{run_id} hit Postgres for real.
  worker/pipeline.py  scripts/smoke_backtest.py's fit/forecast/score protocol,
                       rearranged to read tenant data from Postgres and write
                       forecast_results (one brand per tenant; see its docstring).
  worker/loop.py       polls model_runs, claims with FOR UPDATE SKIP LOCKED.
Fit is ADVI by default for a fast local loop (WORKER_FIT_METHOD=advi); set
model_runs.config_json.fit_method or WORKER_FIT_METHOD to "nuts" for the real
client protocol (4 chains, 1000 tune + 1000 draws, target_accept 0.9; override
per run with config_json.nuts). NUTS runs store per-variable worst rhat and
divergences in diagnostics_json; run responses carry rhat_flag and
rhat_flagged_params when any rhat exceeds 1.01.
Other run options (config_json): random_seed (default 7); persist_draws: true
writes the full posterior + sample stats as NetCDF to WORKER_DRAWS_DIR/<run_id>.nc
(the `draws` volume in docker-compose; path recorded in diagnostics_json.draws_path),
reopen with xarray.open_datatree(path, engine="h5netcdf"). Every fit also records
diagnostics_json.prior_shrinkage (posterior sd / prior sd, mean shift in prior sds;
prior_driven when > 0.9 and < 0.2) for event_depth_coef, event_list_coef and
event_type_effect, plus event_depths for training vs forecast-window events.
