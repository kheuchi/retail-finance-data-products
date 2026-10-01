"""Config drift (story 6.2): does the workspace still match main?

The bundle is pushed by the deploy workflow; nothing reverts a change made by hand in the
workspace until the next deploy. A nightly workflow runs ``databricks bundle plan -o json`` as
the deployer and passes the plan here: any create, update or delete means the workspace and
main disagree. Run: ``python -m retail_finance_data.config_drift plan.json``.
"""

from __future__ import annotations

import json
import sys

UNCHANGED = {"skip", ""}


def pending(plan: dict) -> list[tuple[str, str]]:
    """(resource, action) for every resource the next deploy would change."""
    if not isinstance(plan.get("plan"), dict):
        raise ValueError(f"unexpected plan format: top-level keys {sorted(plan)}")
    return sorted(
        (key, item.get("action", "")) for key, item in plan["plan"].items() if item.get("action", "") not in UNCHANGED
    )


def main() -> int:
    with open(sys.argv[1], encoding="utf-8") as f:
        changes = pending(json.load(f))
    if not changes:
        print("No drift: the workspace matches main.")
        return 0
    print(f"Drift: {len(changes)} resource(s) differ from main (changed by hand, or main not deployed yet):")
    for key, action in changes:
        print(f"  {action:<8} {key}")
    return 1


if __name__ == "__main__":
    sys.exit(main())
