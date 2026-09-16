# Connect your Shopify store (5 minutes, read-only)

We pull daily order totals directly so your forecast uses clean, consistent
data. This access is read-only, and we only ever store daily totals: order
count and revenue per day. We never store customer names, emails, addresses,
or individual orders.

## Steps

1. In Shopify admin go to Settings, then Apps and sales channels.
2. Click Develop apps (top right). If prompted, click Allow custom app
   development. (Requires store owner or an authorized staff account.)
3. Click Create an app. Name it "Forecasting" and create.
4. Open the Configuration tab, click Configure under Admin API integration.
5. Under Admin API access scopes, check ONLY:
   - read_orders
6. Save, then open the API credentials tab and click Install app.
7. Reveal the Admin API access token (starts with shpat_). Copy it.
8. Paste the token and your store domain (yourstore.myshopify.com) into the
   connection step of our intake form.

The token is shown once by Shopify, so paste it right away. You can revoke
this access at any moment by uninstalling the app from the same screen, and
we'd ask you to do exactly that if we ever stop working together.

## What we pull

Daily order counts and daily revenue (post-discount), in your store's own
timezone and currency, going back two years. That's the whole list.
