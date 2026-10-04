# Month-end close agents (story 7.1). AgentCore Runtime runs ARM64 containers.
FROM --platform=linux/arm64 python:3.12-slim

ENV PYTHONUNBUFFERED=1 PIP_NO_CACHE_DIR=1 PIP_DISABLE_PIP_VERSION_CHECK=1
WORKDIR /srv

COPY requirements.txt .
RUN pip install --require-hashes --no-deps -r requirements.txt 2>/dev/null || pip install --no-deps -r requirements.txt

COPY app/ app/
RUN useradd --uid 10001 --create-home agent
USER agent

EXPOSE 8080
CMD ["python", "-m", "app.main"]
