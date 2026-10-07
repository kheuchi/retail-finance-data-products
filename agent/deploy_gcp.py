"""Deploy (create or update) the month-end agents on Google Agent Runtime (D-032).

Run from agent/ by the data repo's main branch through Workload Identity Federation (no key).
Configuration comes from the environment, set from GitHub variables (identifiers, no secrets);
the Databricks credentials are a Secret Manager reference resolved by Agent Runtime itself.
"""

from __future__ import annotations

import os

import vertexai
from app.agent_engine import MonthEndAgent
from google.cloud.aiplatform_v1.types import SecretRef
from vertexai import agent_engines

DISPLAY_NAME = "finance-month-end-agents"


def main() -> None:
    e = os.environ
    vertexai.init(
        project=e["GCP_PROJECT"], location=e.get("GCP_REGION", "europe-west1"), staging_bucket=e["GCP_STAGING_BUCKET"]
    )
    env_vars = {
        "MODEL_PROVIDER": "vertex",
        "MODEL_ID": e.get("AGENT_MODEL_ID", "gemini-3.8-flash"),
        "MODEL_LOCATION": e.get("AGENT_MODEL_LOCATION", "eu"),
        "AGENT_GCP_PROJECT": e["GCP_PROJECT"],
        "DATABRICKS_HOST": e["DATABRICKS_HOST"],
        "AGENT_TOOLS_MCP_URL": f"{e['DATABRICKS_HOST']}/api/2.0/mcp/functions/finance/agent",
        "GATEWAY_URL": e["GATEWAY_URL"],
        "AWS_ROLE_ARN": e["AGENT_GCP_ROLE_ARN"],
        "AWS_REGION": "eu-central-1",
        "DATABRICKS_CREDENTIALS": SecretRef(secret="finance-agent-databricks", version="latest"),
    }
    kwargs = dict(
        requirements="requirements-gcp.txt",
        extra_packages=["app"],
        env_vars=env_vars,
        service_account=e["GCP_RUNTIME_SA"],
        display_name=DISPLAY_NAME,
        description="Month-end close agents (story 7.1): Deep Agents supervisor + sub-agents, Gemini on Vertex AI EU.",
    )
    existing = list(agent_engines.list(filter=f'display_name="{DISPLAY_NAME}"'))
    if existing:
        engine = existing[0].update(agent_engine=MonthEndAgent(), **kwargs)
        print("updated", engine.resource_name)
    else:
        engine = agent_engines.create(MonthEndAgent(), **kwargs)
        print("created", engine.resource_name)


if __name__ == "__main__":
    main()
