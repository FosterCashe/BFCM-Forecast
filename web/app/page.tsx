"use client";

import { useEffect, useMemo, useState } from "react";

import {
  API_URL,
  getResults,
  getTenant,
  getTotals,
  listRuns,
  type Assumption,
  type Diagnostics,
  type ResultRow,
  type Run,
  type Tenant,
  type Total,
} from "@/lib/api";
import { dayLong, moneyWhole, pct, tierLabel, timestamp } from "@/lib/format";
import styles from "./page.module.css";
import RevenueChart, { BandKey, type DailyBand } from "./revenue-chart";
import TotalSummary from "./total-summary";

type Results =
  | { runId: string; rows: ResultRow[]; totals: Total[] }
  | { runId: string; error: string };

type TenantState = { tenantId: string; tenant: Tenant } | { tenantId: string; error: string };

function runLabel(r: Run) {
  const who = r.tenant_name ?? r.tenant_id.slice(0, 8);
  const when = r.started_at ? timestamp(r.started_at) : "not started";
  return `${who} · ${when} · ${r.status}`;
}

const TIER_ORDER: Record<string, number> = { data: 0, evidenced: 1, assumption: 2 };

/** The base forecast leans on the depth prior when the fit didn't learn the
 * coefficient and the window runs a depth the history never tried. */
function depthResponseFromPriors(diag: Diagnostics | null): Assumption | null {
  const depths = diag?.event_depths;
  if (!diag?.prior_shrinkage?.event_depth_coef?.prior_driven || !depths) return null;
  const novel = depths.forecast.some((d) => depths.train.every((t) => Math.abs(t - d) > 5e-4));
  if (!novel) return null;
  return {
    question_key: "offer_depth_response",
    evidence_tier: "evidenced",
    summary:
      "Offer-depth response: from cross-brand and expert priors, not yet learned from your own history (your past promos lack depth variation).",
  };
}

function coverageGloss(coverage: number) {
  if (coverage > 0.85) return "Above 80% means our ranges run conservative: wider than needed, not overconfident.";
  if (coverage < 0.75) return "Below 80% means our ranges run overconfident: narrower than the outcomes warrant.";
  return "Close to 80% means our ranges are calibrated.";
}

export default function ResultsPage() {
  const [runs, setRuns] = useState<Run[] | null>(null);
  const [runsError, setRunsError] = useState<string | null>(null);
  const [runId, setRunId] = useState<string | null>(null);
  const [results, setResults] = useState<Results | null>(null);
  const [tenantState, setTenantState] = useState<TenantState | null>(null);

  useEffect(() => {
    listRuns()
      .then((rs) => {
        setRuns(rs);
        setRunId(rs.find((r) => r.status === "done")?.run_id ?? rs[0]?.run_id ?? null);
      })
      .catch((e: Error) => setRunsError(e.message));
  }, []);

  useEffect(() => {
    if (!runId) return;
    let cancelled = false;
    Promise.all([getResults(runId), getTotals(runId)])
      .then(([rows, totals]) => !cancelled && setResults({ runId, rows, totals }))
      .catch((e: Error) => !cancelled && setResults({ runId, error: e.message }));
    return () => {
      cancelled = true;
    };
  }, [runId]);

  const run = runs?.find((r) => r.run_id === runId) ?? null;
  const tenantId = run?.tenant_id ?? null;

  useEffect(() => {
    if (!tenantId) return;
    let cancelled = false;
    getTenant(tenantId)
      .then((tenant) => !cancelled && setTenantState({ tenantId, tenant }))
      .catch((e: Error) => !cancelled && setTenantState({ tenantId, error: e.message }));
    return () => {
      cancelled = true;
    };
  }, [tenantId]);

  const current = results?.runId === runId ? results : null;
  const tenantCurrent = tenantState?.tenantId === tenantId ? tenantState : null;
  const tenant = tenantCurrent && "tenant" in tenantCurrent ? tenantCurrent.tenant : null;

  const daily = useMemo<DailyBand[]>(() => {
    if (!current || "error" in current) return [];
    return current.rows
      .filter((r) => r.metric === "revenue" && r.scenario === "base")
      .sort((a, b) => a.date.localeCompare(b.date))
      .map((r) => ({ date: r.date, q50: r.q50, band50: [r.q25, r.q75], band80: [r.q10, r.q90] }));
  }, [current]);

  const total =
    current && "totals" in current
      ? current.totals.find((t) => t.metric === "revenue" && t.scenario === "base")
      : undefined;

  if (runsError) {
    return (
      <main className={styles.main}>
        <p className={styles.empty}>
          Couldn&apos;t reach the API at {API_URL} ({runsError}). Is <code>docker compose up</code> running?
        </p>
      </main>
    );
  }

  if (!runs) {
    return <main className={styles.main} aria-busy="true" />;
  }

  if (!run) {
    return (
      <main className={styles.main}>
        <p className={styles.empty}>No runs yet. Queue one with POST /runs.</p>
      </main>
    );
  }

  const diag = run.diagnostics_json;
  const depthPrior = depthResponseFromPriors(diag);
  const assumptions = [...(tenant?.assumptions ?? []), ...(depthPrior ? [depthPrior] : [])].sort(
    (a, b) => (TIER_ORDER[a.evidence_tier] ?? 3) - (TIER_ORDER[b.evidence_tier] ?? 3),
  );

  return (
    <main className={styles.main}>
      <header className={styles.header}>
        <div>
          <p className={styles.eyebrow}>Revenue forecast</p>
          <h1 className={styles.title}>{run.tenant_name ?? run.tenant_id}</h1>
          {daily.length > 0 && (
            <p className={styles.meta}>
              {dayLong(daily[0].date)} – {dayLong(daily[daily.length - 1].date)} · {daily.length} days · base
              scenario
            </p>
          )}
        </div>
        <label className={styles.picker}>
          <span>Run</span>
          <select value={run.run_id} onChange={(e) => setRunId(e.target.value)}>
            {runs.map((r) => (
              <option key={r.run_id} value={r.run_id}>
                {runLabel(r)}
              </option>
            ))}
          </select>
        </label>
      </header>

      {run.status !== "done" ? (
        <p className={styles.empty}>
          {run.status === "error"
            ? `This run failed: ${diag?.error ?? "no error recorded"}`
            : `This run is ${run.status}. Results appear when it finishes.`}
        </p>
      ) : current && "error" in current ? (
        <p className={styles.empty}>Couldn&apos;t load results: {current.error}</p>
      ) : !current ? (
        <div className={styles.loading} aria-busy="true" />
      ) : daily.length === 0 ? (
        <p className={styles.empty}>This run has no revenue rows.</p>
      ) : (
        <>
          {total ? (
            <TotalSummary
              key={run.run_id}
              percentiles={total.percentiles}
              revenueDefinition={tenant?.revenue_definition ?? null}
            />
          ) : (
            <p className={styles.empty}>
              This run was saved before window totals were stored. Queue a new run to see the total and
              P(total &gt; target).
            </p>
          )}

          <section className={styles.chartSection}>
            <div className={styles.sectionHead}>
              <h2 className={styles.h2}>Daily revenue</h2>
              <BandKey />
            </div>
            <RevenueChart data={daily} />
            <p className={styles.explainer}>On each day, 80% of simulated outcomes fall inside the light band.</p>
            <details className={styles.tableView}>
              <summary>View as table</summary>
              <table>
                <thead>
                  <tr>
                    <th scope="col">Date</th>
                    <th scope="col">10th</th>
                    <th scope="col">25th</th>
                    <th scope="col">Median</th>
                    <th scope="col">75th</th>
                    <th scope="col">90th</th>
                  </tr>
                </thead>
                <tbody>
                  {daily.map((d) => (
                    <tr key={d.date}>
                      <th scope="row">{dayLong(d.date)}</th>
                      <td>{moneyWhole(d.band80[0])}</td>
                      <td>{moneyWhole(d.band50[0])}</td>
                      <td>{moneyWhole(d.q50)}</td>
                      <td>{moneyWhole(d.band50[1])}</td>
                      <td>{moneyWhole(d.band80[1])}</td>
                    </tr>
                  ))}
                </tbody>
              </table>
            </details>
          </section>

          <section className={styles.section}>
            <h2 className={styles.h2}>Expected error</h2>
            <p className={styles.sub}>From backtests on your own held-out history, not model self-grading.</p>
            {diag?.backtest_mape == null || diag?.backtest_coverage == null ? (
              <p className={styles.empty}>No backtest recorded for this run.</p>
            ) : (
              <div className={styles.metrics}>
                <div>
                  <p className={styles.metricValue}>{pct(diag.backtest_mape)}</p>
                  <p className={styles.label}>Daily median-path MAPE</p>
                </div>
                <div>
                  <p className={styles.metricValue}>{pct(diag.backtest_coverage)}</p>
                  <p className={styles.label}>80% interval coverage</p>
                  <p className={styles.note}>{coverageGloss(diag.backtest_coverage)}</p>
                </div>
              </div>
            )}
          </section>

          <section className={styles.section}>
            <h2 className={styles.h2}>Inputs and assumptions</h2>
            {tenantCurrent && "error" in tenantCurrent ? (
              <p className={styles.sub}>Couldn&apos;t load inputs: {tenantCurrent.error}</p>
            ) : !tenant ? null : assumptions.length === 0 ? (
              <p className={styles.sub}>No intake answers recorded for this tenant.</p>
            ) : (
              <>
                <dl className={styles.assumptions}>
                  {assumptions.map((a) => (
                    <div key={a.question_key} className={styles.assumption}>
                      <dt className={a.evidence_tier === "data" ? styles.tier : styles.tierUnbacked}>
                        {tierLabel(a.evidence_tier)}
                      </dt>
                      <dd>{a.summary}</dd>
                    </div>
                  ))}
                </dl>
                <p className={styles.note}>
                  {depthPrior
                    ? "The base forecast uses data-backed inputs plus the evidenced prior above. Unverified assumptions are not in these numbers."
                    : "The base forecast uses data-backed inputs only. Assumptions are not in these numbers."}
                </p>
              </>
            )}
          </section>
        </>
      )}
    </main>
  );
}
