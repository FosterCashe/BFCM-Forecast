export const API_URL = process.env.NEXT_PUBLIC_API_URL ?? "http://localhost:8000";

export type Diagnostics = {
  backtest_mape?: number;
  backtest_coverage?: number;
  n_forecast_obs?: number;
  fit_method?: string;
  error?: string;
  event_depths?: { train: number[]; forecast: number[] };
  prior_shrinkage?: Record<string, { sampled: boolean; prior_driven?: boolean }>;
  [key: string]: unknown;
};

export type Run = {
  run_id: string;
  tenant_id: string;
  tenant_name: string | null;
  status: string;
  started_at: string | null;
  finished_at: string | null;
  config_json: Record<string, unknown> | null;
  diagnostics_json: Diagnostics | null;
};

export type ResultRow = {
  scenario: string;
  date: string;
  metric: string;
  q10: number;
  q25: number;
  q50: number;
  q75: number;
  q90: number;
};

export type Total = {
  scenario: string;
  metric: string;
  start_date: string;
  end_date: string;
  percentiles: number[];
};

export type Assumption = {
  question_key: string;
  evidence_tier: string;
  summary: string;
};

export type Tenant = {
  tenant_id: string;
  name: string;
  currency: string;
  revenue_definition: string | null;
  assumptions: Assumption[];
};

async function getJson<T>(path: string): Promise<T> {
  const res = await fetch(`${API_URL}${path}`, { cache: "no-store" });
  if (!res.ok) throw new Error(`GET ${path} returned HTTP ${res.status}`);
  return res.json();
}

export const listRuns = () => getJson<Run[]>("/runs");
export const getResults = (runId: string) => getJson<ResultRow[]>(`/results/${runId}`);
export const getTotals = (runId: string) => getJson<Total[]>(`/results/${runId}/totals`);
export const getTenant = (tenantId: string) => getJson<Tenant>(`/tenants/${tenantId}`);
