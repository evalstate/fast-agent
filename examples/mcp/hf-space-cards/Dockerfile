FROM ghcr.io/astral-sh/uv:python3.13-bookworm

WORKDIR /app
COPY . /app
RUN uv pip install --system fast-agent-mcp

ENV FAST_AGENT_SERVE_OAUTH=huggingface
ENV PYTHONUNBUFFERED=1

EXPOSE 7860
CMD ["python", "app.py"]
