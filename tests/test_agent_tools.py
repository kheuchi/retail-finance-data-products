"""Story 7.1: the agent's tools stay read-only, store-level and free of ledger text."""

import re

from retail_finance_data.jobs import agent_tools

SQL = agent_tools.statements("finance")
ALLOWED_TABLES = {
    "budget_variance",
    "daily_revenue",
    "margin",
    "margin_alerts",
    "recon_exceptions",
    "revenue_forecast",
    "fraud_scores",
}


def body(statement):
    return re.split(r"\nRETURN\n", statement, maxsplit=1)[1]


def test_six_functions_in_the_agent_schema():
    names = [re.search(r"FUNCTION\s+(\S+)\(", s).group(1) for s in SQL]
    expected = (
        "close_overview",
        "store_variances",
        "store_margin_alerts",
        "reconciliation_exceptions",
        "revenue_outlook",
        "cashier_case_count",
    )
    assert names == [f"finance.agent.{n}" for n in expected]


def test_read_only_and_only_gold():
    for s in SQL:
        assert not re.search(r"\b(INSERT|UPDATE|DELETE|MERGE|DROP|ALTER|GRANT|CREATE)\b", body(s).upper())
        for table in re.findall(r"finance\.(\w+)\.(\w+)", body(s)):
            assert table[0] == "gold" and table[1] in ALLOWED_TABLES, table


def test_no_cashier_identity_and_no_journal_text_reach_the_agent():
    for s in SQL:
        returns = s.split("RETURNS TABLE", 1)[1].split("COMMENT", 1)[0]
        assert "cashier_id" not in returns and "description" not in returns
    assert "cashier_id" not in body(SQL[-1])
