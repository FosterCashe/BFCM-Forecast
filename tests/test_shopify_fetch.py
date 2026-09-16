"""The test that matters: UTC timestamps bucket into the SHOP's timezone.
Order 3 is Nov 29 in UTC but Nov 28 in Denver; a naive UTC bucketing (what
every raw CSV export does to you) puts $200 on the wrong day of BFCM week."""
import sys, os
sys.path.insert(0, os.path.dirname(os.path.dirname(os.path.abspath(__file__))))
from worker.shopify_fetch import FixtureTransport, aggregate_daily

def test_timezone_bucketing():
    t = FixtureTransport("tests/fixtures/shop.json", "tests/fixtures/orders.jsonl")
    daily, meta = aggregate_daily(t)
    by_date = {d.date: d for d in daily}
    assert meta["timezone"] == "America/Denver"
    assert meta["revenue_definition"] == "gross_post_discount"
    assert by_date["2025-11-28"].orders == 3
    assert abs(by_date["2025-11-28"].revenue - 400.00) < 0.01
    assert by_date["2025-11-29"].orders == 1
    assert abs(by_date["2025-11-29"].revenue - 60.00) < 0.01
    print("PASS: tz bucketing", {d.date: (d.orders, d.revenue) for d in daily})

if __name__ == "__main__":
    test_timezone_bucketing()
