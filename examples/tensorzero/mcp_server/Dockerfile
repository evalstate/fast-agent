FROM python:3.12-slim

WORKDIR /app

RUN apt-get update && apt-get install -y curl wget && \
    wget https://dl.min.io/client/mc/release/linux-amd64/mc -O /usr/local/bin/mc && \
    chmod +x /usr/local/bin/mc && \
    apt-get clean && rm -rf /var/lib/apt/lists/*

RUN pip install uv

COPY pyproject.toml /app/
COPY uv.lock /app/
COPY LICENSE /app/
COPY README.md /app/

RUN uv pip install --system .

COPY examples/tensorzero/mcp_server/mcp_server.py /app/
COPY examples/tensorzero/mcp_server/entrypoint.sh /app/entrypoint.sh

RUN chmod +x /app/entrypoint.sh

EXPOSE 8000

ENTRYPOINT ["/app/entrypoint.sh"]

CMD ["uvicorn", "mcp_server:app", "--host", "0.0.0.0", "--port", "8000"]
