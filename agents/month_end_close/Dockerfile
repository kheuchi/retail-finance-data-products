# Month-end close agents (story 7.1). AgentCore Runtime runs ARM64 containers.
FROM --platform=linux/arm64 python:3.12-slim

# DOCKER_CONTAINER=1 makes BedrockAgentCoreApp listen on 0.0.0.0: AgentCore runs the image in a
# microVM without /.dockerenv, and a server bound to 127.0.0.1 fails the health check.
ENV PYTHONUNBUFFERED=1 PIP_NO_CACHE_DIR=1 PIP_DISABLE_PIP_VERSION_CHECK=1 DOCKER_CONTAINER=1
WORKDIR /srv

COPY requirements-agentcore.txt .
RUN pip install --no-deps -r requirements-agentcore.txt && pip check

COPY month_end_agent/ month_end_agent/
RUN useradd --uid 10001 --create-home agent
USER agent

EXPOSE 8080
CMD ["python", "-m", "month_end_agent.main"]
