"""
Shopify ingestion via a client-created CUSTOM APP token (no Shopify review).
Client setup is 5 minutes with docs/shopify_setup.md; they grant read_orders
and paste the token at intake.

Privacy stance, enforced in code: line-level order data is aggregated to
daily totals IN MEMORY and discarded. We never persist customer PII.
Daily totals are all the model needs and all we want the liability for.

Testability: the transport is injected. Production uses ShopifyTransport
(live Admin GraphQL API); tests use FixtureTransport (recorded JSONL).
A free Shopify Partner development store tests the live path without a client.

Definitions pinned here, so every tenant is consistent (the thing CSVs never give you):
  revenue  = current_total_price (post-discount, pre-refund) in shop currency
  orders   = non-cancelled, non-test orders, bucketed by created_at in the
             SHOP'S OWN timezone (we ask Shopify for it, never assume)
  refunds  = ignored in v1, revenue_definition recorded as 'gross_post_discount';
             a net-of-refunds pull is a v2 flag, not a silent change
"""

import json
from collections import defaultdict
from dataclasses import dataclass
from datetime import datetime
from zoneinfo import ZoneInfo

API_VERSION = "2026-07"

ORDERS_BULK_QUERY = """
mutation {
  bulkOperationRunQuery(
    query: \"\"\"
    {
      orders(query: "created_at:>=%(since)s AND -status:cancelled AND test:false") {
        edges { node { id createdAt currentTotalPriceSet { shopMoney { amount } } } }
      }
    }
    \"\"\"
  ) { bulkOperation { id status } userErrors { field message } }
}
"""

SHOP_QUERY = '{ shop { ianaTimezone currencyCode } }'


@dataclass
class DailyAgg:
    date: str
    orders: int
    revenue: float


class FixtureTransport:
    """Replays a recorded bulk-export JSONL file. Used by tests and demos."""

    def __init__(self, shop_json_path, orders_jsonl_path):
        self.shop = json.load(open(shop_json_path))
        self.orders_path = orders_jsonl_path

    def get_shop(self):
        return self.shop

    def iter_orders(self, since):
        with open(self.orders_path) as fh:
            for line in fh:
                if line.strip():
                    yield json.loads(line)


class ShopifyTransport:
    """Live Admin API. Same interface as FixtureTransport. Flow:
    1. POST bulkOperationRunQuery (ORDERS_BULK_QUERY % {'since': ...})
    2. Poll currentBulkOperation until COMPLETED, download the JSONL url
    3. Stream lines through iter_orders
    Requires: requests, token with read_orders scope.
    """

    def __init__(self, shop_domain, access_token):
        self.base = f"https://{shop_domain}/admin/api/{API_VERSION}/graphql.json"
        self.headers = {"X-Shopify-Access-Token": access_token,
                        "Content-Type": "application/json"}

    def get_shop(self):
        import requests
        r = requests.post(self.base, headers=self.headers, json={"query": SHOP_QUERY}, timeout=30)
        r.raise_for_status()
        return r.json()["data"]["shop"]

    def iter_orders(self, since):
        raise NotImplementedError(
            "Live bulk flow: run ORDERS_BULK_QUERY, poll, stream result JSONL. "
            "Implement and verify against a free Partner development store."
        )


def aggregate_daily(transport, since="2024-01-01"):
    """Orders stream -> daily totals in the shop's own timezone.
    Line-level data never leaves this function."""
    shop = transport.get_shop()
    tz = ZoneInfo(shop["ianaTimezone"])
    buckets = defaultdict(lambda: [0, 0.0])
    for node in transport.iter_orders(since):
        created = datetime.fromisoformat(node["createdAt"].replace("Z", "+00:00"))
        day = created.astimezone(tz).date().isoformat()
        buckets[day][0] += 1
        buckets[day][1] += float(node["currentTotalPriceSet"]["shopMoney"]["amount"])
    daily = [DailyAgg(d, c, round(r, 2)) for d, (c, r) in sorted(buckets.items())]
    return daily, dict(timezone=shop["ianaTimezone"], currency=shop["currencyCode"],
                       revenue_definition="gross_post_discount")
