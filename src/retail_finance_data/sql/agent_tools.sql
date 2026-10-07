-- Read-only tools of the month-end agents (story 7.1, ADR-006), exposed over MCP by the
-- Databricks managed MCP server for Unity Catalog functions (finance.agent).
-- Rules: store level or above only (never cashier IDs, R-12); every figure the agent may
-- quote is computed here (variances, percentages), so the model never does arithmetic;
-- no free text from the ledger (journal descriptions excluded against prompt injection).
-- {catalog} is replaced by the job. Statements are separated by a line holding only ";;".

CREATE OR REPLACE FUNCTION {catalog}.agent.close_overview(m STRING COMMENT 'Close month, YYYY-MM')
RETURNS TABLE (
  month STRING, net_sales_eur DECIMAL(18,2), budget_to_date_eur DECIMAL(18,2), variance_eur DECIMAL(18,2),
  variance_pct DECIMAL(9,2), prior_year_net_sales_eur DECIMAL(18,2), yoy_growth_pct DECIMAL(9,2),
  gross_margin_pct DECIMAL(9,2), discount_rate_pct DECIMAL(9,2), complete_month BOOLEAN, days_with_sales INT)
COMMENT 'Chain totals for a close month: net sales (EUR, excl. VAT) vs budget to date (stores with a budget), growth vs the same days of the same month last year, gross margin and discount rate. Percentages are already computed. complete_month = false means the month is still in progress.'
RETURN
  WITH bv AS (
    SELECT sum(actual_net_sales_eur) AS total, sum(actual_net_sales_eur) FILTER (WHERE has_budget) AS actual,
      sum(budget_to_date_eur) FILTER (WHERE has_budget) AS budget, bool_and(complete_month) AS complete
    FROM {catalog}.gold.budget_variance WHERE month = m),
  cur AS (
    SELECT max(day(business_date)) AS last_day, count(DISTINCT business_date) AS days FROM {catalog}.gold.daily_revenue
    WHERE date_format(business_date, 'yyyy-MM') = m),
  py AS (
    SELECT sum(net_sales_eur) AS prior FROM {catalog}.gold.daily_revenue
    WHERE date_format(business_date, 'yyyy-MM') = date_format(add_months(to_date(concat(m, '-01')), -12), 'yyyy-MM')
      AND day(business_date) <= (SELECT last_day FROM cur)),
  mg AS (
    SELECT sum(net_sales_eur) AS net, sum(cogs_eur) AS cogs, sum(discount_eur) AS disc, sum(paid_incl_vat_eur) AS paid
    FROM {catalog}.gold.margin WHERE month = m)
  SELECT m,
    CAST(bv.total AS DECIMAL(18,2)), CAST(bv.budget AS DECIMAL(18,2)), CAST(bv.actual - bv.budget AS DECIMAL(18,2)),
    CAST(round(100 * try_divide(bv.actual - bv.budget, bv.budget), 2) AS DECIMAL(9,2)),
    CAST(py.prior AS DECIMAL(18,2)),
    CAST(round(100 * try_divide(bv.total - py.prior, py.prior), 2) AS DECIMAL(9,2)),
    CAST(round(100 * (1 - try_divide(mg.cogs, mg.net)), 2) AS DECIMAL(9,2)),
    CAST(round(100 * try_divide(mg.disc, mg.paid + mg.disc), 2) AS DECIMAL(9,2)),
    bv.complete, CAST(cur.days AS INT)
  FROM bv CROSS JOIN cur CROSS JOIN py CROSS JOIN mg
;;
CREATE OR REPLACE FUNCTION {catalog}.agent.store_variances(m STRING COMMENT 'Close month, YYYY-MM')
RETURNS TABLE (
  store_id STRING, net_sales_eur DECIMAL(18,2), budget_to_date_eur DECIMAL(18,2), variance_eur DECIMAL(18,2),
  variance_pct DECIMAL(9,2), complete_month BOOLEAN)
COMMENT 'Net sales vs budget to date per store for a close month, worst variance first.'
RETURN
  SELECT store_id, actual_net_sales_eur, budget_to_date_eur, variance_eur,
    CAST(round(100 * try_divide(variance_eur, budget_to_date_eur), 2) AS DECIMAL(9,2)), complete_month
  FROM {catalog}.gold.budget_variance
  WHERE month = m AND has_budget
  ORDER BY try_divide(variance_eur, budget_to_date_eur)
;;
CREATE OR REPLACE FUNCTION {catalog}.agent.store_margin_alerts(m STRING COMMENT 'Close month, YYYY-MM')
RETURNS TABLE (
  store_id STRING, rank_in_month BIGINT, flagged BOOLEAN, margin_pct DECIMAL(9,2), margin_pct_prior_6m DECIMAL(9,2),
  margin_change_pts DECIMAL(9,2), discount_rate_pct DECIMAL(9,2), discount_rate_pct_prior_6m DECIMAL(9,2))
COMMENT 'Stores the margin detector ranks highest for a month (flagged or top 3), with their margin and discount rate against their own previous six months. Store level only.'
RETURN
  WITH sm AS (
    SELECT store_id, month,
      100 * (1 - try_divide(sum(cogs_eur), sum(net_sales_eur))) AS margin,
      100 * try_divide(sum(discount_eur), sum(paid_incl_vat_eur) + sum(discount_eur)) AS discount
    FROM {catalog}.gold.margin GROUP BY store_id, month),
  base AS (
    SELECT store_id, avg(margin) AS margin6, avg(discount) AS discount6 FROM sm
    WHERE month BETWEEN date_format(add_months(to_date(concat(m, '-01')), -6), 'yyyy-MM')
                    AND date_format(add_months(to_date(concat(m, '-01')), -1), 'yyyy-MM')
    GROUP BY store_id)
  SELECT a.store_id, a.rank_in_month, a.flagged,
    CAST(round(cur.margin, 2) AS DECIMAL(9,2)), CAST(round(base.margin6, 2) AS DECIMAL(9,2)),
    CAST(round(cur.margin - base.margin6, 2) AS DECIMAL(9,2)),
    CAST(round(cur.discount, 2) AS DECIMAL(9,2)), CAST(round(base.discount6, 2) AS DECIMAL(9,2))
  FROM {catalog}.gold.margin_alerts a
  JOIN sm cur ON cur.store_id = a.store_id AND cur.month = a.month
  LEFT JOIN base ON base.store_id = a.store_id
  WHERE a.month = m AND (a.flagged OR a.rank_in_month <= 3)
  ORDER BY a.rank_in_month
;;
CREATE OR REPLACE FUNCTION {catalog}.agent.reconciliation_exceptions(m STRING COMMENT 'Close month, YYYY-MM')
RETURNS TABLE (
  store_id STRING, business_date DATE, journal_id STRING, amount_eur DECIMAL(18,2), reason STRING,
  day_pos_net_eur DECIMAL(18,2), day_gl_revenue_eur DECIMAL(18,2))
COMMENT 'Revenue journals the GL vs POS reconciliation could not match to till sales in a month. Journal descriptions are deliberately not returned.'
RETURN
  SELECT store_id, business_date, journal_id, CAST(amount_eur AS DECIMAL(18,2)), reason,
    CAST(day_pos_net_eur AS DECIMAL(18,2)), CAST(day_gl_revenue_eur AS DECIMAL(18,2))
  FROM {catalog}.gold.recon_exceptions
  WHERE date_format(business_date, 'yyyy-MM') = m
  ORDER BY abs(amount_eur) DESC
;;
CREATE OR REPLACE FUNCTION {catalog}.agent.revenue_outlook(m STRING COMMENT 'Close month, YYYY-MM; the outlook covers the months after it')
RETURNS TABLE (
  month STRING, forecast_eur DECIMAL(18,2), low_80_eur DECIMAL(18,2), high_80_eur DECIMAL(18,2), range_pct INT, method STRING)
COMMENT 'Chain revenue forecast (latest model run) for the months after the close month, with its range (range_pct = 80: an 80% range), and the method that won the backtest.'
RETURN
  SELECT month, CAST(sum(forecast_eur) AS DECIMAL(18,2)), CAST(sum(low_80_eur) AS DECIMAL(18,2)),
    CAST(sum(high_80_eur) AS DECIMAL(18,2)), 80, max(method)
  FROM {catalog}.gold.revenue_forecast
  WHERE month > m
  GROUP BY month ORDER BY month
;;
CREATE OR REPLACE FUNCTION {catalog}.agent.cashier_case_count(m STRING COMMENT 'Close month, YYYY-MM')
RETURNS TABLE (month STRING, cashier_cases_referred BIGINT, cashier_months_scored BIGINT)
COMMENT 'How many cashier-months the refund detector flagged for internal audit. A count only: no cashier or store is identified (R-12).'
RETURN
  SELECT any_value(m), count_if(flagged), count(*) FROM {catalog}.gold.fraud_scores WHERE month = m

;;
CREATE OR REPLACE FUNCTION {catalog}.agent.gold_certification_status()
RETURNS TABLE (certified BOOLEAN, gold_run STRING, certified_at TIMESTAMP)
COMMENT 'Latest Gold build certification. Read by the agent code (not the model) before every run: no certified Gold, no run.'
RETURN
  SELECT certified, run_id, certified_at FROM {catalog}.ops.gold_certification ORDER BY certified_at DESC LIMIT 1
