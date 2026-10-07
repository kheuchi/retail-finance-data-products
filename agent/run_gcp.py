"""Run the month-end agents on Google Agent Runtime for a close month (D-032)."""

from __future__ import annotations

import json
import os
import sys

import vertexai
from vertexai import agent_engines


def main() -> None:
    month = sys.argv[1]
    vertexai.init(project=os.environ["GCP_PROJECT"], location=os.environ.get("GCP_REGION", "europe-west1"))
    engine = list(agent_engines.list(filter='display_name="finance-month-end-agents"'))[0]
    result = engine.query(month=month)
    print(json.dumps({k: v for k, v in result.items() if k != "final"}, indent=1))
    print(result.get("final", ""))


if __name__ == "__main__":
    main()
