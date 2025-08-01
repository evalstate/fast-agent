FROM python:3.12-slim
WORKDIR /app

RUN apt-get update && apt-get install -y curl wget && \
    wget https://dl.min.io/client/mc/release/linux-amd64/mc -O /usr/local/bin/mc && \
    chmod +x /usr/local/bin/mc && \
    apt-get clean && rm -rf /var/lib/apt/lists/*

RUN pip install uv

# Copy dependency files only
COPY pyproject.toml /tmp/
#COPY uv.lock /tmp/

# Install dependencies
WORKDIR /tmp
RUN uv pip install --system .

# Switch back to /app (this will be overridden by volume mount)
WORKDIR /app

# These files will come from the volume mount at runtime
EXPOSE 8000
ENTRYPOINT ["/app/entrypoint.sh"]
CMD ["uvicorn", "mcp_server:app", "--host", "0.0.0.0", "--port", "8000"]
