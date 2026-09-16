-- Core schema v2. tenant_id everywhere; on Supabase add RLS policies per tenant.

create table tenants (
    tenant_id   uuid primary key default gen_random_uuid(),
    name        text not null,
    timezone    text not null default 'America/Denver',  -- normalize ALL dates to this
    currency    text not null default 'USD',
    created_at  timestamptz default now(),
    pooling_consent boolean not null default false  -- ToS flag: cross-brand pooling allowed?
);

create table daily_metrics (
    tenant_id   uuid references tenants,
    date        date not null,
    orders      integer not null,
    revenue     numeric(14,2) not null,
    revenue_definition text default 'gross',  -- gross | net_discounts | net_returns
    new_orders        integer,               -- enables new vs returning split (upgrade #1)
    returning_orders  integer,
    primary key (tenant_id, date)
);

create table ad_spend (
    tenant_id   uuid references tenants,
    date        date not null,
    channel     text not null,
    channel_kind text default 'prospecting', -- prospecting | search | owned (sets adstock prior)
    spend       numeric(12,2) not null,
    impressions bigint,
    clicks      bigint,
    primary key (tenant_id, date, channel)
);

-- Generic events. BFCM, July 4th, flash sales: all rows here. Every promo in
-- history must be declared (see spike detection) or the baseline misreads it.
create table events (
    tenant_id   uuid references tenants,
    event_id    text not null,
    event_type  text not null,
    start_date  date not null,
    end_date    date not null,
    offer_type  text,                    -- sitewide_pct, bogo, tiered, gwp, none
    depth       numeric(4,3),            -- effective discount fraction
    discounted_share numeric(4,3) default 1.0,  -- share of revenue on discounted items
    list_ratio  numeric(6,3) default 1.0,       -- list size vs last comparable period
    anomaly     boolean default false,          -- stockout/viral/etc: down-weight this instance
    exclusions  boolean default false,
    notes       text,
    primary key (tenant_id, event_id)
);

create table email_sends (
    tenant_id   uuid references tenants,
    date        date not null,
    audience    bigint not null,         -- recipients of major campaigns that day
    primary key (tenant_id, date)
);

create table inventory_ceilings (
    tenant_id   uuid references tenants,
    label       text not null,           -- 'total' or a hero SKU
    units       integer not null,
    as_of       date not null,
    primary key (tenant_id, label)
);

create table intake_answers (
    tenant_id   uuid references tenants,
    question_key text not null,
    answer_json  jsonb not null,
    evidence_tier text not null default 'assumption',  -- data | evidenced | assumption
    answered_at  timestamptz default now(),
    primary key (tenant_id, question_key)
);

create table model_runs (
    run_id      uuid primary key default gen_random_uuid(),
    tenant_id   uuid references tenants,
    started_at  timestamptz default now(),
    finished_at timestamptz,
    status      text default 'queued',
    config_json jsonb,
    diagnostics_json jsonb   -- rhat, divergences, AND backtest mape/coverage
);

create table forecast_results (
    run_id      uuid references model_runs,
    tenant_id   uuid references tenants,
    scenario    text not null,
    date        date not null,
    metric      text not null,           -- revenue, orders, aov, margin
    q10 numeric, q25 numeric, q50 numeric, q75 numeric, q90 numeric,
    primary key (run_id, scenario, date, metric)
);
