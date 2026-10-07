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


def test_capped_tool_refuses_after_its_limit():
    import asyncio

    from langchain_core.tools import StructuredTool

    async def echo(text: str) -> str:
        return text

    tool = main.capped(StructuredTool.from_function(coroutine=echo, name="submit_draft", description="d"), 2)
    results = asyncio.run(_call_three(tool))
    assert results[:2] == ["a", "a"] and results[2].startswith("refused")


async def _call_three(tool):
    return [await tool.ainvoke({"text": "a"}) for _ in range(3)]


def test_final_text_from_claude_or_gemini_messages():
    class Msg:
        def __init__(self, content):
            self.content = content

    assert main.final_text(Msg("done")) == "done"
    assert main.final_text(Msg([{"type": "text", "text": "a"}, {"type": "text", "text": "b", "extras": {}}])) == "ab"


def test_agent_engine_adapter_pickles_without_building_anything():
    import pickle

    from app.agent_engine import MonthEndAgent

    agent = pickle.loads(pickle.dumps(MonthEndAgent()))
    assert hasattr(agent, "query") and hasattr(agent, "async_query")
