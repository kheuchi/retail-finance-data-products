# AGENTS.md

Instructions for any AI assistant working in this repo. Tool-neutral; `CLAUDE.md` imports this file.

| Question | Answer |
|---|---|
| Project rules, way of working, how to report | **Read first:** [control-plane AGENTS.md](https://github.com/kheuchi/retail-finance-platform-control-plane/blob/main/AGENTS.md) |
| This repo | Synthetic data generator, Bronze/Silver/Gold jobs, tests, Databricks bundle ([README](README.md)) |
| Changes | Pull request only; `Lint and test` and `Validate bundle` must pass; deploy via **Actions → Deploy to Databricks** (type `deploy`), never from a laptop |
| Tests | `pip install -e ".[dev]" && pytest`; Spark tests need `export JAVA_HOME=$HOME/.local/jdk17` |
| Data rules | Synthetic data only; nothing downloaded at runtime (no internet in the workspace); quarantine bad rows, never drop them |
| Facts and history | [`cmdb.yml`](cmdb.yml); explanations in the [stories](https://github.com/kheuchi/retail-finance-platform-control-plane/blob/main/docs/stories/README.md) |
