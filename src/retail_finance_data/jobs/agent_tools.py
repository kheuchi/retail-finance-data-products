"""Job: create the month-end agents' read-only tools in finance.agent (story 7.1).

The tools are Unity Catalog SQL functions (sql/agent_tools.sql). Databricks' managed MCP
server exposes them to the agent; the channel tools' figure check calls the same functions,
so both see exactly the same numbers. Runs as the runner, which may create functions in
finance.agent and nothing else there; the agent identity may only execute them.
"""

from __future__ import annotations

import argparse
from importlib.resources import files

SEPARATOR = "\n;;\n"


def statements(catalog: str) -> list[str]:
    text = files("retail_finance_data").joinpath("sql/agent_tools.sql").read_text(encoding="utf-8")
    body = "\n".join(line for line in text.splitlines() if not line.startswith("--"))
    out = [s.strip().replace("{catalog}", catalog) for s in body.split(SEPARATOR) if s.strip()]
    if not all(s.count("CREATE OR REPLACE FUNCTION") == 1 for s in out):
        raise ValueError("each statement must hold exactly one CREATE OR REPLACE FUNCTION (check the ';;' lines)")
    return out


def main() -> None:
    from pyspark.sql import SparkSession

    p = argparse.ArgumentParser()
    p.add_argument("--catalog", default="finance")
    a = p.parse_args()
    spark = SparkSession.builder.getOrCreate()
    for sql in statements(a.catalog):
        name = sql.split("FUNCTION", 1)[1].split("(", 1)[0].strip()
        spark.sql(sql)
        print(f"created {name}", flush=True)


if __name__ == "__main__":
    main()
