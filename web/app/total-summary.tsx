"use client";

import { useState } from "react";

import { editableMoney, money, parseMoney, revenueDefinition } from "@/lib/format";
import styles from "./total-summary.module.css";

/** P(total > x) from the 0th..100th percentiles, linear between grid points. */
function probabilityAbove(percentiles: number[], x: number): number {
  const last = percentiles.length - 1;
  if (x < percentiles[0]) return 1;
  if (x >= percentiles[last]) return 0;
  let i = 0;
  while (percentiles[i + 1] <= x) i++;
  const span = percentiles[i + 1] - percentiles[i];
  const within = span > 0 ? (x - percentiles[i]) / span : 0;
  return 1 - (i + within) / last;
}

// Percentile grid is 1 point wide, so the tails can't be resolved past 1%.
function formatProbability(p: number) {
  if (p < 0.01) return "<1%";
  if (p > 0.99) return ">99%";
  return `${Math.round(p * 100)}%`;
}

function defaultTarget(median: number) {
  const step = 10 ** Math.floor(Math.log10(median)) / 2;
  return Math.ceil(median / step) * step;
}

export default function TotalSummary({
  percentiles,
  revenueDefinition: definition,
}: {
  percentiles: number[];
  revenueDefinition: string | null;
}) {
  const median = percentiles[50];
  const [targetText, setTargetText] = useState(() => editableMoney(defaultTarget(median)));
  const target = parseMoney(targetText);

  return (
    <section className={styles.summary} aria-label="Window total">
      <p className={styles.label}>Window total, median</p>
      <p className={styles.label}>80% range</p>
      <label className={styles.label} htmlFor="target">
        P(total &gt; $<input
          id="target"
          className={styles.target}
          value={targetText}
          onChange={(e) => setTargetText(e.target.value)}
          inputMode="decimal"
          spellCheck={false}
          aria-invalid={target === null}
          aria-label="Target window total in dollars"
        />)
      </label>

      <p className={styles.hero}>{money(median)}</p>
      <p className={styles.secondary}>
        {money(percentiles[10])} – {money(percentiles[90])}
      </p>
      <p className={styles.secondary} aria-live="polite">
        {target === null ? "—" : formatProbability(probabilityAbove(percentiles, target))}
      </p>

      {definition && <p className={styles.definition}>{revenueDefinition(definition)}</p>}
    </section>
  );
}
