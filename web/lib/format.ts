const compactMoney = new Intl.NumberFormat("en-US", {
  style: "currency",
  currency: "USD",
  notation: "compact",
  maximumSignificantDigits: 3,
});

const wholeMoney = new Intl.NumberFormat("en-US", {
  style: "currency",
  currency: "USD",
  maximumFractionDigits: 0,
});

// Result dates are calendar days; format in UTC so they never shift a day.
const shortDate = new Intl.DateTimeFormat("en-US", { month: "short", day: "numeric", timeZone: "UTC" });
const longDate = new Intl.DateTimeFormat("en-US", {
  month: "short",
  day: "numeric",
  year: "numeric",
  timeZone: "UTC",
});

const asUtcDay = (d: string) => new Date(`${d.slice(0, 10)}T00:00:00Z`);

export const money = (v: number) => compactMoney.format(v);
export const moneyWhole = (v: number) => wholeMoney.format(v);
export const pct = (v: number) => `${Math.round(v * 100)}%`;
export const dayShort = (d: string) => shortDate.format(asUtcDay(d));
export const dayLong = (d: string) => longDate.format(asUtcDay(d));
export const timestamp = (iso: string) => longDate.format(new Date(iso));

const REVENUE_DEFINITIONS: Record<string, string> = {
  gross: "Gross revenue, before discounts and returns",
  net_discounts: "Revenue net of discounts, before returns",
  net_returns: "Revenue net of returns",
  gross_post_discount: "Revenue after discounts, before refunds",
  mixed: "Mixed revenue definitions in the upload",
};

export const revenueDefinition = (d: string) => REVENUE_DEFINITIONS[d] ?? d;

// Same labels as model/output.py TIERS.
const TIERS: Record<string, string> = {
  data: "Data-backed",
  evidenced: "Evidenced assumption",
  assumption: "Unverified assumption",
};

export const tierLabel = (t: string) => TIERS[t] ?? t;

/** Accepts 750000, 750,000, $750k, 1.2m. Returns null if unparseable. */
export function parseMoney(input: string): number | null {
  const m = input.trim().toLowerCase().replace(/[$,\s]/g, "").match(/^(\d*\.?\d+)([km])?$/);
  if (!m) return null;
  return Number(m[1]) * (m[2] === "m" ? 1e6 : m[2] === "k" ? 1e3 : 1);
}

export function editableMoney(v: number): string {
  if (v >= 1e6) return `${+(v / 1e6).toFixed(2)}M`;
  if (v >= 1e3) return `${+(v / 1e3).toFixed(1)}K`;
  return String(Math.round(v));
}
