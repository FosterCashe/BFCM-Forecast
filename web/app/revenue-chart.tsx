"use client";

import {
  Area,
  CartesianGrid,
  ComposedChart,
  Line,
  ResponsiveContainer,
  Tooltip,
  XAxis,
  YAxis,
  type TooltipContentProps,
  type TooltipValueType,
} from "recharts";

import { dayLong, dayShort, money } from "@/lib/format";
import styles from "./revenue-chart.module.css";

export type DailyBand = {
  date: string;
  q50: number;
  band50: [number, number];
  band80: [number, number];
};

function BandTooltip({ active, payload }: TooltipContentProps<TooltipValueType, string | number>) {
  if (!active || !payload?.length) return null;
  const d = payload[0].payload as DailyBand;
  return (
    <div className={styles.tooltip}>
      <p className={styles.tooltipDate}>{dayLong(d.date)}</p>
      <p className={styles.tooltipValue}>{money(d.q50)}</p>
      <dl className={styles.tooltipRanges}>
        <dt>50% range</dt>
        <dd>
          {money(d.band50[0])} – {money(d.band50[1])}
        </dd>
        <dt>80% range</dt>
        <dd>
          {money(d.band80[0])} – {money(d.band80[1])}
        </dd>
      </dl>
    </div>
  );
}

export default function RevenueChart({ data }: { data: DailyBand[] }) {
  return (
    <div className={styles.frame}>
      <ResponsiveContainer width="100%" height="100%">
        <ComposedChart data={data} margin={{ top: 8, right: 8, bottom: 0, left: 0 }}>
          <CartesianGrid vertical={false} stroke="var(--grid)" />
          <XAxis
            dataKey="date"
            tickFormatter={dayShort}
            tickLine={false}
            axisLine={{ stroke: "var(--baseline)" }}
            tick={{ fill: "var(--muted)", fontSize: 12 }}
            tickMargin={12}
            minTickGap={32}
          />
          <YAxis
            tickFormatter={money}
            tickLine={false}
            axisLine={false}
            tick={{ fill: "var(--muted)", fontSize: 12 }}
            width={56}
          />
          <Tooltip
            content={(props) => <BandTooltip {...props} />}
            cursor={{ stroke: "var(--baseline)", strokeWidth: 1 }}
            isAnimationActive={false}
          />
          <Area
            dataKey="band80"
            stroke="none"
            fill="var(--accent)"
            fillOpacity="var(--band-outer)"
            activeDot={false}
            isAnimationActive={false}
          />
          <Area
            dataKey="band50"
            stroke="none"
            fill="var(--accent)"
            fillOpacity="var(--band-inner)"
            activeDot={false}
            isAnimationActive={false}
          />
          <Line
            dataKey="q50"
            stroke="var(--accent)"
            strokeWidth={2}
            strokeLinecap="round"
            strokeLinejoin="round"
            dot={false}
            activeDot={{ r: 4, fill: "var(--accent)", stroke: "var(--surface)", strokeWidth: 2 }}
            isAnimationActive={false}
          />
        </ComposedChart>
      </ResponsiveContainer>
    </div>
  );
}

export function BandKey() {
  return (
    <ul className={styles.key}>
      <li>
        <span className={styles.keyLine} /> Median
      </li>
      <li>
        <span className={styles.keySwatch} style={{ opacity: "var(--band-inner-key)" }} /> 50% range
      </li>
      <li>
        <span className={styles.keySwatch} style={{ opacity: "var(--band-outer)" }} /> 80% range
      </li>
    </ul>
  );
}
