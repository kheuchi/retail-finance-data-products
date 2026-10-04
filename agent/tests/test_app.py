"""Story 7.1: wiring of the month-end agents (no network, no model)."""

from datetime import date

from app import main


def test_tool_names_lose_their_server_prefix():
    assert main.short("finance__agent__close_overview") == "close_overview"
    assert main.short("channels___submit_draft") == "submit_draft"


def test_default_month_is_the_last_complete_one():
    assert main.last_complete_month(date(2026, 10, 5)) == "2026-09"
    assert main.last_complete_month(date(2027, 1, 3)) == "2026-12"


def test_rules_reach_every_sub_agent_and_no_one_can_approve():
    assert "never compute a new figure" in main.RULES
    assert "cannot approve" in main.RULES
    for _desc, _prompt, tools in main.SUBAGENTS.values():
        assert "approvals" not in " ".join(tools)
    # only the distributor may send; drafting agents cannot
    senders = [name for name, (_d, _p, tools) in main.SUBAGENTS.items() if "send_approved" in tools]
    assert senders == ["distributor"]


def test_no_tool_exposes_cashiers():
    wanted = {t for _d, _p, tools in main.SUBAGENTS.values() for t in tools}
    assert not any("cashier" in t and t != "cashier_case_count" for t in wanted)
