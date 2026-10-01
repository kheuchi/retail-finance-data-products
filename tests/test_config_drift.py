"""Story 6.2: the nightly check fails on any pending change, and on a plan it cannot read."""

import pytest

from retail_finance_data import config_drift


def test_unchanged_plan_has_no_drift():
    plan = {"plan_version": 2, "plan": {"resources.jobs.a": {"action": "skip"}, "resources.jobs.b": {"action": "skip"}}}
    assert config_drift.pending(plan) == []


def test_changed_resources_are_listed():
    plan = {"plan": {"resources.jobs.a": {"action": "update"}, "resources.jobs.b": {"action": "Skip"}}}
    assert config_drift.pending(plan) == [("resources.jobs.a", "update")]


def test_unknown_format_fails_loudly():
    with pytest.raises(ValueError):
        config_drift.pending({"something": "else"})
