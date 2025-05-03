FROM python:3.12-slim

WORKDIR /app

RUN pip install uv

COPY pyproject.toml /app/
COPY uv.lock /app/
COPY LICENSE /app/
COPY README.md /app/

RUN uv pip install --system .

COPY examples/tensorzero/mcp_server/mcp_server.py /app/

EXPOSE 8000

CMD ["uvicorn", "mcp_server:app", "--host", "0.0.0.0", "--port", "8000"]
